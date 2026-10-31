// fLLM CUDA backend: tensor-core GEMM for Q4_K weights with fragment-level dequantization.
//
//   Y[token][row] = sum_k X[token][k] * W[row][k]  (+ bias[row])
//
// Design (same idea as Marlin, adapted to the GGUF Q4_K layout, no weight repacking):
//  - Raw quantized bytes (a 16-byte header and the 32 nibble bytes of one 64-element k-tile) and
//    the FP16 activations are copied to shared memory with multi-stage `cp.async` pipelines.
//  - Each thread reads the 32-bit word it needs for its slice of the mma A fragment, turns the
//    nibbles into FP16 with a LOP3/PRMT magic-number trick and applies the sub-block scale and
//    minimum with two packed half operations. The decoded weights never touch shared memory.
//  - Activations are converted to FP16 once per call by `convert_x_kernel`, which also applies a
//    fixed permutation inside every group of 16 values so a thread's four bytes map onto exactly
//    the k positions its mma fragment needs (any common k permutation leaves the sum unchanged).
// FP16 weights/activations with FP32 accumulation: ~1e-3 relative error, below the Q4_K noise.
//
// QG_VARIANT selects one optimisation step at a time. Every step keeps the arithmetic of the
// previous one bit for bit, so all variants must reproduce variant 0 exactly:
//   0  the kernel as it was
//   1  one LOP3 instead of shift + mask + OR per two nibbles
//   2  half2 scale arithmetic, with the negation of the minimum hoisted out of the dequant
//   3  1 + 2 ("fewer instructions per mma")
//   4  3 + warp tiles that split the activation tile instead of replicating it in every warp
// The tile shape itself is picked at run time by fllm_autotune; QG_BIG_* / QG_S_* override it.
// See docs/cuda-qgemm4k.md for the measurements each step is expected to move.
#ifndef QG_VARIANT
#define QG_VARIANT 0
#endif

#include "common.cuh"
#include "fllm_cuda_internal.h"
#include "mma_util.cuh"

#if QG_VARIANT == 1 || QG_VARIANT >= 3
#define QG_LOP3 1
#else
#define QG_LOP3 0
#endif
#if QG_VARIANT == 2 || QG_VARIANT >= 3
#define QG_SCFAST 1
#else
#define QG_SCFAST 0
#endif
#define QG_MIN o2  // dmin * mn: the term every dequant subtracts
#if QG_VARIANT >= 4
#define QG_TILES 1
#else
#define QG_TILES 0
#endif
// Two nibbles of `raw` (one per byte, low half for grp 0 and high half for grp 1) spread into two
// 16-bit lanes, turned into exact FP16 values and scaled: n * s2 - sub. The LOP3 variant merges the nibble mask and the
// 1024 + n magic into one instruction (the shift of the high nibbles stays), the result is bit-identical.
__device__ __forceinline__ uint32_t q4k_pair(uint32_t raw, uint32_t sel, int grp, __half2 s2, __half2 sub) {
#if QG_LOP3
    return dequant_pair_lop3(grp == 0 ? raw : (raw >> 4), sel, 0x000F000Fu, s2, sub);
#else
    return dequant_pair((grp == 0 ? raw : (raw >> 4)) & 0x0F0F0F0Fu, sel, s2, sub);
#endif
}

// X16[n][16*blk + m] = X[n][16*blk + perm(m)],  perm(m) = 4 * ((m & 7) >> 1) + 2 * (m >> 3) + (m & 1)
template <bool GLU>
__global__ void convert_x_kernel(const float *__restrict__ x, const float *__restrict__ up, int mode,
                               __half *__restrict__ out, size_t groups, int cols, int cols_pad) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < groups; i += (size_t)gridDim.x * blockDim.x) {
        // output row `tok` has cols_pad values, the activations only cols: the rest is zero (padded weight blocks)
        const size_t gpr = (size_t)cols_pad / 16;  // groups of 16 per output row
        const size_t tok = i / gpr, kk = i % gpr;
        const size_t src = tok * cols + kk * 16;
        if (kk * 16 >= (size_t)cols) {
            uint4 *o = reinterpret_cast<uint4 *>(out + i * 16);
            o[0] = make_uint4(0, 0, 0, 0);
            o[1] = make_uint4(0, 0, 0, 0);
            continue;
        }
        const float4 *p = reinterpret_cast<const float4 *>(x + src);
        float4 v[4];
#pragma unroll
        for (int j = 0; j < 4; j++) v[j] = __ldg(p + j);
        float f[16] = {v[0].x, v[0].y, v[0].z, v[0].w, v[1].x, v[1].y, v[1].z, v[1].w,
                             v[2].x, v[2].y, v[2].z, v[2].w, v[3].x, v[3].y, v[3].z, v[3].w};
        if (GLU) {
            const float4 *pu = reinterpret_cast<const float4 *>(up + src);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const float4 t = __ldg(pu + j);
                f[4 * j] = fllm_glu(f[4 * j], t.x, mode);
                f[4 * j + 1] = fllm_glu(f[4 * j + 1], t.y, mode);
                f[4 * j + 2] = fllm_glu(f[4 * j + 2], t.z, mode);
                f[4 * j + 3] = fllm_glu(f[4 * j + 3], t.w, mode);
            }
        }
        __half r[16];
#pragma unroll
        for (int m = 0; m < 16; m++) r[m] = __float2half_rn(f[4 * ((m & 7) >> 1) + 2 * (m >> 3) + (m & 1)]);
        uint4 *o = reinterpret_cast<uint4 *>(out + i * 16);
        o[0] = *reinterpret_cast<const uint4 *>(&r[0]);
        o[1] = *reinterpret_cast<const uint4 *>(&r[8]);
    }
}

constexpr int Q4K_STAGE_A = 48;   // bytes per row and k-tile: 16 header + 32 nibble bytes
constexpr int Q4K_STAGE_B = 128;  // bytes per token and k-tile: 64 halves

template <int BM, int BN, int WM, int WN, int STAGES>
__global__ void __launch_bounds__(WM * WN * 32)
qgemm4k_kernel(const uint8_t *__restrict__ W, const __half *__restrict__ X16, float *__restrict__ Y,
               const float *__restrict__ bias, int rows, int cols, int batch) {
#if FLLM_TC
    constexpr int NT = WM * WN * 32;
    constexpr int MI = BM / WM / 16;
    constexpr int NI = BN / WN / 8;
    static_assert(NI % 2 == 0 && MI >= 1, "tile shape");
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char *As = smem;                                   // [STAGES][BM][48]
    unsigned char *Bs = smem + (size_t)STAGES * BM * Q4K_STAGE_A;  // [STAGES][BN][128], 16-byte chunks swizzled
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp % WM, wn = warp / WM;
    const int g = lane >> 2, t4 = lane & 3;
    int m0, n0;
    qg_tile_origin(BM, BN, batch, m0, n0);
    const int nsb = cols >> 8;
    const size_t row_bytes = (size_t)nsb * 144;
    const int ktiles_all = cols >> 6;
    const int splits = gridDim.z;
    const int per = (ktiles_all + splits - 1) / splits;
    const int kt0 = blockIdx.z * per;
    const int ktiles = max(0, min(ktiles_all, kt0 + per) - kt0);  // k tiles of this block

    auto issue = [&](int kt, int stage) {
        const int sb = kt >> 2, sub = kt & 3;
        for (int c = tid; c < BM * 3; c += NT) {  // 3 chunks of 16 bytes per row
            const int r = c / 3, part = c % 3;
            const int row = m0 + r;
            const bool ok = row < rows;
            const uint8_t *src = W + (size_t)(ok ? row : 0) * row_bytes + (size_t)sb * 144 + (part == 0 ? 0 : 16 + sub * 32 + (part - 1) * 16);
            cp_async16(As + ((size_t)stage * BM + r) * Q4K_STAGE_A + part * 16, src, ok);
        }
        for (int c = tid; c < BN * 8; c += NT) {  // 8 chunks of 16 bytes per token
            const int n = c >> 3, chunk = c & 7;
            const int tok = n0 + n;
            const bool ok = tok < batch;
            const __half *src = X16 + (size_t)(ok ? tok : 0) * cols + (size_t)kt * 64 + chunk * 8;
            cp_async16(Bs + ((size_t)stage * BN + n) * Q4K_STAGE_B + ((chunk ^ (n & 7)) * 16), src, ok);
        }
    };

    float acc[MI][NI][4];
#pragma unroll
    for (int a = 0; a < MI; a++)
#pragma unroll
        for (int b = 0; b < NI; b++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[a][b][c] = 0.f;

#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < ktiles) issue(kt0 + s, s);
        cp_async_commit();
    }

    for (int kt = 0; kt < ktiles; kt++) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        if (kt + STAGES - 1 < ktiles) issue(kt0 + kt + STAGES - 1, (kt + STAGES - 1) % STAGES);
        cp_async_commit();

        const int stage = kt % STAGES;
        const unsigned char *a_stage = As + (size_t)stage * BM * Q4K_STAGE_A;
        const unsigned char *b_stage = Bs + (size_t)stage * BN * Q4K_STAGE_B;
        const int g0 = 2 * ((kt0 + kt) & 3);

        // scale and minimum of this k-tile's two sub-blocks for the rows of this thread
#if QG_SCFAST
        __half2 s2[MI][2][2], o2[MI][2][2];  // [mi][row half][sub-block]: d*sc and dmin*mn as fp16 products
#pragma unroll
        for (int mi = 0; mi < MI; mi++)
#pragma unroll
            for (int rh = 0; rh < 2; rh++) {
                const int r = wm * (BM / WM) + mi * 16 + g + rh * 8;
                const uint4 h = *reinterpret_cast<const uint4 *>(a_stage + (size_t)r * Q4K_STAGE_A);
                // an fp16 product rounds like the float product of variant 0 rounded once
                const __half2 d2 = __half2half2(__ushort_as_half((uint16_t)(h.x & 0xffffu)));
                const __half2 dm2 = __half2half2(__ushort_as_half((uint16_t)(h.x >> 16)));
#pragma unroll
                for (int sbk = 0; sbk < 2; sbk++) {
                    const int gi = g0 + sbk;
                    auto byte_of = [&](int i) -> uint32_t {
                        const uint32_t w = i < 4 ? h.y : (i < 8 ? h.z : h.w);
                        return (w >> ((i & 3) * 8)) & 0xffu;
                    };
                    uint32_t raw_sc, raw_mn;
                    if (gi < 4) {
                        raw_sc = byte_of(gi) & 63u;
                        raw_mn = byte_of(gi + 4) & 63u;
                    } else {
                        raw_sc = (byte_of(gi + 4) & 15u) | ((byte_of(gi - 4) >> 6) << 4);
                        raw_mn = (byte_of(gi + 4) >> 4) | ((byte_of(gi) >> 6) << 4);
                    }
                    s2[mi][rh][sbk] = __hmul2(d2, __half2half2(__ushort2half_rn((unsigned short)raw_sc)));
                    o2[mi][rh][sbk] = __hmul2(dm2, __half2half2(__ushort2half_rn((unsigned short)raw_mn)));
                }
            }
#else
        __half2 s2[MI][2][2], o2[MI][2][2];  // [mi][row half][sub-block]
#pragma unroll
        for (int mi = 0; mi < MI; mi++)
#pragma unroll
            for (int rh = 0; rh < 2; rh++) {
                const int r = wm * (BM / WM) + mi * 16 + g + rh * 8;
                const uint4 h = *reinterpret_cast<const uint4 *>(a_stage + (size_t)r * Q4K_STAGE_A);
                const float d = h2f_bits((uint16_t)(h.x & 0xffffu)), dmin = h2f_bits((uint16_t)(h.x >> 16));
#pragma unroll
                for (int sbk = 0; sbk < 2; sbk++) {
                    const int gi = g0 + sbk;
                    float sc, mn;
                    {
                        auto byte_of = [&](int i) -> uint32_t {
                            const uint32_t w = i < 4 ? h.y : (i < 8 ? h.z : h.w);
                            return (w >> ((i & 3) * 8)) & 0xffu;
                        };
                        if (gi < 4) {
                            sc = (float)(byte_of(gi) & 63u);
                            mn = (float)(byte_of(gi + 4) & 63u);
                        } else {
                            sc = (float)((byte_of(gi + 4) & 15u) | ((byte_of(gi - 4) >> 6) << 4));
                            mn = (float)((byte_of(gi + 4) >> 4) | ((byte_of(gi) >> 6) << 4));
                        }
                    }
#ifdef QG_EXP_NOSCALE
                    s2[mi][rh][sbk] = __half2half2(__float2half_rn(d + (float)h.y));
                    o2[mi][rh][sbk] = __half2half2(__float2half_rn(dmin + (float)h.z));
#else
                    s2[mi][rh][sbk] = __half2half2(__float2half_rn(d * sc));
                    o2[mi][rh][sbk] = __half2half2(__float2half_rn(dmin * mn));
#endif
                }
            }
#endif

#pragma unroll
        for (int step = 0; step < 4; step++) {
            const int grp = step >> 1, b0 = (step & 1) * 16;
            uint32_t af[MI][4];
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
                for (int rh = 0; rh < 2; rh++) {
                    const int r = wm * (BM / WM) + mi * 16 + g + rh * 8;
                    const uint32_t w = *reinterpret_cast<const uint32_t *>(a_stage + (size_t)r * Q4K_STAGE_A + 16 + b0 + 4 * t4);
#ifdef QG_EXP_NODEQ
                    af[mi][rh] = w ^ half2_as_u32(s2[mi][rh][grp]);
                    af[mi][2 + rh] = (w >> 8) ^ half2_as_u32(QG_MIN[mi][rh][grp]);
#else
                    af[mi][rh] = q4k_pair(w, 0x4140u, grp, s2[mi][rh][grp], QG_MIN[mi][rh][grp]);       // k 2t4, 2t4+1
                    af[mi][2 + rh] = q4k_pair(w, 0x4342u, grp, s2[mi][rh][grp], QG_MIN[mi][rh][grp]);   // k 2t4+8, 2t4+9
#endif
                }
            uint32_t bf[NI][2];
#pragma unroll
            for (int np = 0; np < NI / 2; np++) {
                const int n = wn * (BN / WN) + np * 16 + (lane & 7) + ((lane >> 4) << 3);
                const int chunk = step * 2 + ((lane >> 3) & 1);
                ldmatrix4(bf[np * 2][0], bf[np * 2][1], bf[np * 2 + 1][0], bf[np * 2 + 1][1],
                          b_stage + (size_t)n * Q4K_STAGE_B + ((chunk ^ (n & 7)) * 16));
            }
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
                for (int ni = 0; ni < NI; ni++) mma16816(acc[mi][ni], af[mi], bf[ni][0], bf[ni][1]);
        }
    }
    cp_async_wait<0>();

#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int ni = 0; ni < NI; ni++)
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int m = m0 + wm * (BM / WM) + mi * 16 + g + (c >> 1) * 8;
                const int n = n0 + wn * (BN / WN) + ni * 8 + 2 * t4 + (c & 1);
                if (m < rows && n < batch) {
                    if (splits > 1) Y[(size_t)blockIdx.z * batch * rows + (size_t)n * rows + m] = acc[mi][ni][c];
                    else Y[(size_t)n * rows + m] = acc[mi][ni][c] + (bias ? bias[m] : 0.f);
                }
            }
#endif
}


// Sums the split-K partial results: Y[i] = sum_z part[z][i] + bias[row]
__global__ void splitk_reduce_kernel(const float *__restrict__ part, float *__restrict__ Y,
                                     const float *__restrict__ bias, int rows, size_t count, int splits) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x) {
        float s = bias ? bias[i % rows] : 0.f;
        for (int z = 0; z < splits; z++) s += part[(size_t)z * count + i];
        Y[i] = s;
    }
}

// How many k slices a narrow batch should be split into so that every SM gets work (1 = none).
static int splitk_choice(int row_blocks,  /* all blocks of the grid */  int ktiles, int batch, int rows, size_t spare_bytes) {
    if (row_blocks >= 92) return 1;
    int splits = (92 + row_blocks - 1) / row_blocks;
    if (splits > 8) splits = 8;
    while (splits > 1 && (ktiles / splits < 8 || (size_t)splits * batch * rows * 4 > spare_bytes)) splits--;
    return splits;
}

template <int BM, int BN, int WM, int WN, int STAGES>
static int launch_cfg(cudaStream_t s, const uint8_t *W, const __half *X16, float *Y, const float *bias, int rows,
                      int cols, int batch, float *part, size_t part_bytes) {
    constexpr size_t smem = (size_t)STAGES * (BM * Q4K_STAGE_A + BN * Q4K_STAGE_B);
    if (smem > (size_t)fllm_smem_limit()) {
        if constexpr (STAGES > 2) return launch_cfg<BM, BN, WM, WN, 2>(s, W, X16, Y, bias, rows, cols, batch, part, part_bytes);
        else return 0;
    }
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(qgemm4k_kernel<BM, BN, WM, WN, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        configured = true;
    }
    const int row_blocks = (rows + BM - 1) / BM;
    const int splits = splitk_choice(row_blocks * ((batch + BN - 1) / BN), cols >> 6, batch, rows, part_bytes);
    const dim3 grid(row_blocks, (batch + BN - 1) / BN, splits), block(WM * WN * 32);
    qgemm4k_kernel<BM, BN, WM, WN, STAGES><<<grid, block, smem, s>>>(W, X16, splits > 1 ? part : Y, bias, rows, cols, batch);
    if (splits > 1) {
        const size_t count = (size_t)batch * rows;
        splitk_reduce_kernel<<<(unsigned)((count + 255) / 256), 256, 0, s>>>(part, Y, bias, rows, count, splits);
    }
    return 1;
}


// ---------------------------------------------------------------------------------------------
// Narrow batches: one pipeline stage is a whole super-block (256 values, 144 contiguous bytes per row), so
// every weight request is one long, aligned burst and the header is fetched once instead of four times.
// ---------------------------------------------------------------------------------------------
constexpr int Q4K_SB_A = 144;  // bytes per row and stage
constexpr int Q4K_SB_B = 512;  // bytes per token and stage: 256 halves

template <int BM, int BN, int WM, int WN, int STAGES>
__global__ void __launch_bounds__(WM * WN * 32)
qgemm4k_sb_kernel(const uint8_t *__restrict__ W, const __half *__restrict__ X16, float *__restrict__ Y,
                  const float *__restrict__ bias, int rows, int cols, int batch) {
#if FLLM_TC
    constexpr int NT = WM * WN * 32;
    constexpr int MI = BM / WM / 16;
    constexpr int NI = BN / WN / 8;
    static_assert(NI % 2 == 0 && MI >= 1, "tile shape");
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char *As = smem;                                     // [STAGES][BM][144]
    unsigned char *Bs = smem + (size_t)STAGES * BM * Q4K_SB_A;    // [STAGES][BN][512], chunks swizzled
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp % WM, wn = warp / WM;
    const int g = lane >> 2, t4 = lane & 3;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    const int nsb_all = cols >> 8;
    const size_t row_bytes = (size_t)nsb_all * 144;
    const int splits = gridDim.z;
    const int per = (nsb_all + splits - 1) / splits;
    const int sb0 = blockIdx.z * per;
    const int nsb = max(0, min(nsb_all, sb0 + per) - sb0);

    auto issue = [&](int sb, int stage) {
        for (int c = tid; c < BM * 9; c += NT) {  // 9 chunks of 16 bytes per row
            const int r = c / 9, part = c % 9;
            const int row = m0 + r;
            const bool ok = row < rows;
            const uint8_t *src = W + (size_t)(ok ? row : 0) * row_bytes + (size_t)sb * 144 + part * 16;
            cp_async16(As + ((size_t)stage * BM + r) * Q4K_SB_A + part * 16, src, ok);
        }
        for (int c = tid; c < BN * 32; c += NT) {  // 32 chunks of 16 bytes per token
            const int n = c >> 5, chunk = c & 31;
            const int tok = n0 + n;
            const bool ok = tok < batch;
            const __half *src = X16 + (size_t)(ok ? tok : 0) * cols + (size_t)sb * 256 + chunk * 8;
            cp_async16(Bs + ((size_t)stage * BN + n) * Q4K_SB_B + ((chunk ^ (n & 7)) * 16), src, ok);
        }
    };

    float acc[MI][NI][4];
#pragma unroll
    for (int a = 0; a < MI; a++)
#pragma unroll
        for (int b = 0; b < NI; b++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[a][b][c] = 0.f;

#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < nsb) issue(sb0 + s, s);
        cp_async_commit();
    }

    for (int sbi = 0; sbi < nsb; sbi++) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        if (sbi + STAGES - 1 < nsb) issue(sb0 + sbi + STAGES - 1, (sbi + STAGES - 1) % STAGES);
        cp_async_commit();

        const int stage = sbi % STAGES;
        const unsigned char *a_stage = As + (size_t)stage * BM * Q4K_SB_A;
        const unsigned char *b_stage = Bs + (size_t)stage * BN * Q4K_SB_B;

        uint4 hdr[MI][2];
#pragma unroll
        for (int mi = 0; mi < MI; mi++)
#pragma unroll
            for (int rh = 0; rh < 2; rh++) {
                const int r = wm * (BM / WM) + mi * 16 + g + rh * 8;
                hdr[mi][rh] = *reinterpret_cast<const uint4 *>(a_stage + (size_t)r * Q4K_SB_A);
            }

#pragma unroll
        for (int sub = 0; sub < 4; sub++) {  // 64 values (two sub-blocks) each
#if QG_SCFAST
            __half2 s2[MI][2][2], o2[MI][2][2];
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
                for (int rh = 0; rh < 2; rh++) {
                    const uint4 h = hdr[mi][rh];
                    const __half2 d2 = __half2half2(__ushort_as_half((uint16_t)(h.x & 0xffffu)));
                    const __half2 dm2 = __half2half2(__ushort_as_half((uint16_t)(h.x >> 16)));
#pragma unroll
                    for (int sbk = 0; sbk < 2; sbk++) {
                        const int gi = 2 * sub + sbk;
                        auto byte_of = [&](int i) -> uint32_t {
                            const uint32_t w = i < 4 ? h.y : (i < 8 ? h.z : h.w);
                            return (w >> ((i & 3) * 8)) & 0xffu;
                        };
                        uint32_t raw_sc, raw_mn;
                        if (gi < 4) {
                            raw_sc = byte_of(gi) & 63u;
                            raw_mn = byte_of(gi + 4) & 63u;
                        } else {
                            raw_sc = (byte_of(gi + 4) & 15u) | ((byte_of(gi - 4) >> 6) << 4);
                            raw_mn = (byte_of(gi + 4) >> 4) | ((byte_of(gi) >> 6) << 4);
                        }
                        s2[mi][rh][sbk] = __hmul2(d2, __half2half2(__ushort2half_rn((unsigned short)raw_sc)));
                        o2[mi][rh][sbk] = __hmul2(dm2, __half2half2(__ushort2half_rn((unsigned short)raw_mn)));
                    }
                }
#else
            __half2 s2[MI][2][2], o2[MI][2][2];
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
                for (int rh = 0; rh < 2; rh++) {
                    const uint4 h = hdr[mi][rh];
                    const float d = h2f_bits((uint16_t)(h.x & 0xffffu)), dmin = h2f_bits((uint16_t)(h.x >> 16));
#pragma unroll
                    for (int sbk = 0; sbk < 2; sbk++) {
                        const int gi = 2 * sub + sbk;
                        auto byte_of = [&](int i) -> uint32_t {
                            const uint32_t w = i < 4 ? h.y : (i < 8 ? h.z : h.w);
                            return (w >> ((i & 3) * 8)) & 0xffu;
                        };
                        float sc, mn;
                        if (gi < 4) {
                            sc = (float)(byte_of(gi) & 63u);
                            mn = (float)(byte_of(gi + 4) & 63u);
                        } else {
                            sc = (float)((byte_of(gi + 4) & 15u) | ((byte_of(gi - 4) >> 6) << 4));
                            mn = (float)((byte_of(gi + 4) >> 4) | ((byte_of(gi) >> 6) << 4));
                        }
                        s2[mi][rh][sbk] = __half2half2(__float2half_rn(d * sc));
                        o2[mi][rh][sbk] = __half2half2(__float2half_rn(dmin * mn));
                    }
                }
#endif
#pragma unroll
            for (int step = 0; step < 4; step++) {
                const int grp = step >> 1, b0 = (step & 1) * 16;
                uint32_t af[MI][4];
#pragma unroll
                for (int mi = 0; mi < MI; mi++)
#pragma unroll
                    for (int rh = 0; rh < 2; rh++) {
                        const int r = wm * (BM / WM) + mi * 16 + g + rh * 8;
                        const uint32_t w = *reinterpret_cast<const uint32_t *>(a_stage + (size_t)r * Q4K_SB_A + 16 + sub * 32 + b0 + 4 * t4);
                        af[mi][rh] = q4k_pair(w, 0x4140u, grp, s2[mi][rh][grp], QG_MIN[mi][rh][grp]);
                        af[mi][2 + rh] = q4k_pair(w, 0x4342u, grp, s2[mi][rh][grp], QG_MIN[mi][rh][grp]);
                    }
                uint32_t bf[NI][2];
#pragma unroll
                for (int np = 0; np < NI / 2; np++) {
                    const int n = wn * (BN / WN) + np * 16 + (lane & 7) + ((lane >> 4) << 3);
                    const int chunk = (sub * 4 + step) * 2 + ((lane >> 3) & 1);
                    ldmatrix4(bf[np * 2][0], bf[np * 2][1], bf[np * 2 + 1][0], bf[np * 2 + 1][1],
                              b_stage + (size_t)n * Q4K_SB_B + ((chunk ^ (n & 7)) * 16));
                }
#pragma unroll
                for (int mi = 0; mi < MI; mi++)
#pragma unroll
                    for (int ni = 0; ni < NI; ni++) mma16816(acc[mi][ni], af[mi], bf[ni][0], bf[ni][1]);
            }
        }
    }
    cp_async_wait<0>();

#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int ni = 0; ni < NI; ni++)
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int m = m0 + wm * (BM / WM) + mi * 16 + g + (c >> 1) * 8;
                const int n = n0 + wn * (BN / WN) + ni * 8 + 2 * t4 + (c & 1);
                if (m < rows && n < batch) {
                    if (splits > 1) Y[(size_t)blockIdx.z * batch * rows + (size_t)n * rows + m] = acc[mi][ni][c];
                    else Y[(size_t)n * rows + m] = acc[mi][ni][c] + (bias ? bias[m] : 0.f);
                }
            }
#endif
}

template <int BM, int BN, int WM, int WN, int STAGES>
static int launch_sb(cudaStream_t s, const uint8_t *W, const __half *X16, float *Y, const float *bias, int rows,
                     int cols, int batch, float *part, size_t part_bytes) {
    constexpr size_t smem = (size_t)STAGES * (BM * Q4K_SB_A + BN * Q4K_SB_B);
    if (smem > (size_t)fllm_smem_limit()) {
        if constexpr (STAGES > 2) return launch_sb<BM, BN, WM, WN, 2>(s, W, X16, Y, bias, rows, cols, batch, part, part_bytes);
        else return 0;
    }
    static_assert(smem <= 99 * 1024, "shared memory budget");
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(qgemm4k_sb_kernel<BM, BN, WM, WN, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        configured = true;
    }
    const int row_blocks = (rows + BM - 1) / BM;
    int splits = splitk_choice(row_blocks * ((batch + BN - 1) / BN), cols >> 6, batch, rows, part_bytes);
    while (splits > 1 && (cols >> 8) / splits < 2) splits--;
    const dim3 grid(row_blocks, (batch + BN - 1) / BN, splits), block(WM * WN * 32);
    qgemm4k_sb_kernel<BM, BN, WM, WN, STAGES><<<grid, block, smem, s>>>(W, X16, splits > 1 ? part : Y, bias, rows, cols, batch);
    if (splits > 1) {
        const size_t count = (size_t)batch * rows;
        splitk_reduce_kernel<<<(unsigned)((count + 255) / 256), 256, 0, s>>>(part, Y, bias, rows, count, splits);
    }
    return 1;
}

#ifndef QG_SB_BM
#define QG_SB_BM 64
#endif
#ifndef QG_SB_WM
#define QG_SB_WM 4
#endif
#ifndef QG_SB_ST
#define QG_SB_ST 4
#endif

#ifndef QG_S_BM
#define QG_S_BM 128
#endif
#ifndef QG_S_WM
#define QG_S_WM 4
#endif
#ifndef QG_S_ST
#define QG_S_ST 4
#endif
#ifndef QG_BIG_BM
#define QG_BIG_BM 128
#endif
#ifndef QG_BIG_BN
#define QG_BIG_BN 256
#endif
// Variant 4 splits the wide token tile over more warps: with WM = 8 every warp loads the whole
// activation tile, with WM = 4 / WN = 2 two warps share it and each thread holds half as many
// B fragment registers. Both give the same result.
#ifndef QG_BIG_WM
#if QG_TILES
#define QG_BIG_WM 4
#else
#define QG_BIG_WM 8
#endif
#endif
#ifndef QG_BIG_WN
#if QG_TILES
#define QG_BIG_WN 2
#else
#define QG_BIG_WN 1
#endif
#endif
#ifndef QG_BIG_ST
#define QG_BIG_ST 2
#endif

// ---------------------------------------------------------------------------------------------
// The wide-batch shape comes from fllm_autotune, which times the candidates listed in `run`
// above; QG_BIG_BM / QG_BIG_BN / QG_BIG_WM / QG_BIG_WN / QG_BIG_ST pin it instead (variant 4
// defaults them to 128 / 256 / 4 / 2 / 2). Per k tile a warp reads (BN / WN) * 64 activation
// halves, so a block moves WM * BN * 128 bytes of activations and WN * BM * 36 bytes of weights:
// a wider warp tile costs shared-memory bandwidth, BM = 256 halves how often the activations
// are re-read.

// Converts the activations to the permuted FP16 layout the tensor-core GEMM kernels read.
thread_local bool fllm_x16_reuse = false, fllm_x16_filled = false;
thread_local bool fllm_x8_reuse = false, fllm_x8_filled = false;
thread_local const float *fllm_glu_up = nullptr;
thread_local float *fllm_glu_dst = nullptr;
thread_local int fllm_glu_mode = 0;

void fllm_convert_x16_pad(cudaStream_t s, const float *X, void *x16, int batch, int cols, int cols_pad) {
    fllm_x16_filled = true;
    fllm_x8_filled = false;
    if (fllm_x16_reuse) return;
    const size_t groups = (size_t)batch * cols_pad / 16;
    const unsigned grid = (unsigned)((groups + 255) / 256 > 4096 ? 4096 : (groups + 255) / 256);
    if (fllm_glu_up) convert_x_kernel<true><<<grid, 256, 0, s>>>(X, fllm_glu_up, fllm_glu_mode, static_cast<__half *>(x16), groups, cols, cols_pad);
    else convert_x_kernel<false><<<grid, 256, 0, s>>>(X, nullptr, 0, static_cast<__half *>(x16), groups, cols, cols_pad);
}
void fllm_convert_x16(cudaStream_t s, const float *X, void *x16, int batch, int cols) { fllm_convert_x16_pad(s, X, x16, batch, cols, cols); }

// Q4_K GEMM for `batch` >= 2 tokens. `x16` is scratch for batch * cols halves (16-byte aligned).
// Returns 1 when handled, 0 otherwise (the caller falls back to other kernels).
int fllm_qgemm4k_try(cudaStream_t s, const void *W, const float *X, float *Y, const float *bias, int rows, int cols,
                     int batch, void *x16_raw, size_t x16_bytes) {
    if (fllm_cc() < 75) return 0;
    __half *x16 = static_cast<__half *>(x16_raw);
    static const bool disabled = getenv("FLLM_CUDA_NO_QGEMM") != nullptr;
    if (disabled || batch < 2 || (cols & 255) != 0 || (reinterpret_cast<uintptr_t>(X) & 15) != 0 ||
        (reinterpret_cast<uintptr_t>(W) & 15) != 0 || (size_t)batch * cols * 2 > x16_bytes)
        return 0;
    fllm_convert_x16(s, X, x16, batch, cols);
    const uint8_t *w = static_cast<const uint8_t *>(W);
    const size_t x_used = ((size_t)batch * cols * 2 + 255) & ~size_t(255);
    float *part = reinterpret_cast<float *>(static_cast<char *>(x16_raw) + x_used);
    const size_t part_bytes = x16_bytes > x_used ? x16_bytes - x_used : 0;
#ifndef QG_NO_SB
    if (batch > 16 && batch <= 32) return launch_sb<64, 32, 4, 1, 3>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
    if (batch > 32 && batch <= 64) return launch_sb<64, 64, 4, 1, 2>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
    if (batch <= 16) return launch_sb<QG_SB_BM, 16, QG_SB_WM, 1, QG_SB_ST>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
#endif
    if (batch <= 16) return launch_cfg<QG_S_BM, 16, QG_S_WM, 1, QG_S_ST>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
    if (batch <= 32) return launch_cfg<128, 32, 4, 1, 4>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
    if (batch <= 64) return launch_cfg<128, 64, 4, 1, 4>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
    if (batch <= 128) return launch_cfg<128, 128, 4, 2, 4>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
    auto run = [&](int i) -> int {
        switch (i) {
            case 0: {
                // narrower token tile when the last wide tile would be mostly padding
                const double wide = 74.0 * batch / (((batch + 255) / 256) * 256), narrow = 66.0 * batch / (((batch + 127) / 128) * 128);
#if QG_TILES
                if (narrow > wide) return launch_cfg<128, 128, 4, 2, 4>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
#else
                if (narrow > wide) return launch_cfg<128, 128, 8, 1, 3>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
#endif
                return launch_cfg<QG_BIG_BM, QG_BIG_BN, QG_BIG_WM, QG_BIG_WN, QG_BIG_ST>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
            }
            case 1: return launch_cfg<128, 128, 8, 1, 3>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
            case 2: return launch_cfg<128, 128, 4, 2, 4>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
            case 3: return launch_cfg<128, 192, 4, 2, 3>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
            case 4: return launch_cfg<128, 256, 4, 2, 2>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
            case 5: return launch_cfg<256, 128, 8, 2, 2>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
            case 6: return launch_cfg<128, 96, 4, 1, 4>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
            default: return launch_cfg<64, 128, 4, 2, 4>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
        }
    };
    return run(fllm_autotune(s, 0x4B, rows, cols, batch, 8, run));
}
