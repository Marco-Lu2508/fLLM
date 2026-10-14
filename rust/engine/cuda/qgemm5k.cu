// fLLM CUDA backend: tensor-core GEMM for Q5_K weights (see qgemm4k.cu for the design).
//
// A Q5_K super-block is a Q4_K block with 32 more bytes of high bits (qh) after the scales: the fifth bit of
// element l of sub-block j is bit j of qh[l] (j = 0..7). A thread loads the qh word next to its nibble word,
// shifts the bit into position 4 of every byte and dequantizes the 5-bit values exactly like the 4-bit ones.
#include <cstdlib>

#include "common.cuh"
#include "fllm_cuda_internal.h"
#include "mma_util.cuh"

constexpr int Q5K_STAGE_A = 80;   // bytes per row and k-tile: 16 header + 32 low-nibble bytes + 32 high-bit bytes
constexpr int Q5K_STAGE_B = 128;  // bytes per token and k-tile: 64 halves

template <int BM, int BN, int WM, int WN, int STAGES>
__global__ void __launch_bounds__(WM * WN * 32)
qgemm5k_kernel(const uint8_t *__restrict__ W, const __half *__restrict__ X16, float *__restrict__ Y,
               const float *__restrict__ bias, int rows, int cols, int batch) {
#if FLLM_TC
    constexpr int NT = WM * WN * 32;
    constexpr int MI = BM / WM / 16;
    constexpr int NI = BN / WN / 8;
    static_assert(NI % 2 == 0 && MI >= 1, "tile shape");
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char *As = smem;                                   // [STAGES][BM][48]
    unsigned char *Bs = smem + (size_t)STAGES * BM * Q5K_STAGE_A;  // [STAGES][BN][128], 16-byte chunks swizzled
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp % WM, wn = warp / WM;
    const int g = lane >> 2, t4 = lane & 3;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    const int nsb = cols >> 8;
    const size_t row_bytes = (size_t)nsb * 176;
    const int ktiles_all = cols >> 6;
    const int splits = gridDim.z;
    const int per = (ktiles_all + splits - 1) / splits;
    const int kt0 = blockIdx.z * per;
    const int ktiles = max(0, min(ktiles_all, kt0 + per) - kt0);  // k tiles of this block

    auto issue = [&](int kt, int stage) {
        const int sb = kt >> 2, sub = kt & 3;
        for (int c = tid; c < BM * 5; c += NT) {  // header, two chunks of nibbles, two chunks of high bits
            const int r = c / 5, part = c % 5;
            const int row = m0 + r;
            const bool ok = row < rows;
            const uint8_t *src = W + (size_t)(ok ? row : 0) * row_bytes + (size_t)sb * 176 +
                                 (part == 0 ? 0 : (part < 3 ? 48 + sub * 32 + (part - 1) * 16 : 16 + (part - 3) * 16));
            cp_async16(As + ((size_t)stage * BM + r) * Q5K_STAGE_A + part * 16, src, ok);
        }
        for (int c = tid; c < BN * 8; c += NT) {  // 8 chunks of 16 bytes per token
            const int n = c >> 3, chunk = c & 7;
            const int tok = n0 + n;
            const bool ok = tok < batch;
            const __half *src = X16 + (size_t)(ok ? tok : 0) * cols + (size_t)kt * 64 + chunk * 8;
            cp_async16(Bs + ((size_t)stage * BN + n) * Q5K_STAGE_B + ((chunk ^ (n & 7)) * 16), src, ok);
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
        const unsigned char *a_stage = As + (size_t)stage * BM * Q5K_STAGE_A;
        const unsigned char *b_stage = Bs + (size_t)stage * BN * Q5K_STAGE_B;
        const int g0 = 2 * ((kt0 + kt) & 3);

        // scale and minimum of this k-tile's two sub-blocks for the rows of this thread
        __half2 s2[MI][2][2], o2[MI][2][2];  // [mi][row half][sub-block]
#pragma unroll
        for (int mi = 0; mi < MI; mi++)
#pragma unroll
            for (int rh = 0; rh < 2; rh++) {
                const int r = wm * (BM / WM) + mi * 16 + g + rh * 8;
                const uint4 h = *reinterpret_cast<const uint4 *>(a_stage + (size_t)r * Q5K_STAGE_A);
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
#ifdef Q5G_EXP_NOSCALE
                    s2[mi][rh][sbk] = __half2half2(__float2half_rn(d + (float)h.y));
                    o2[mi][rh][sbk] = __half2half2(__float2half_rn(dmin + (float)h.z));
#else
                    s2[mi][rh][sbk] = __half2half2(__float2half_rn(d * sc));
                    o2[mi][rh][sbk] = __half2half2(__float2half_rn(dmin * mn));
#endif
                }
            }

#pragma unroll
        for (int step = 0; step < 4; step++) {
            const int grp = step >> 1, b0 = (step & 1) * 16;
            uint32_t af[MI][4];
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
                for (int rh = 0; rh < 2; rh++) {
                    const int r = wm * (BM / WM) + mi * 16 + g + rh * 8;
                    uint32_t w = *reinterpret_cast<const uint32_t *>(a_stage + (size_t)r * Q5K_STAGE_A + 16 + b0 + 4 * t4);
                    const uint32_t hq = *reinterpret_cast<const uint32_t *>(a_stage + (size_t)r * Q5K_STAGE_A + 48 + b0 + 4 * t4);
                    w = (grp == 0 ? w : (w >> 4)) & 0x0F0F0F0Fu;
                    w |= ((hq >> (2 * ((kt0 + kt) & 3) + grp)) & 0x01010101u) << 4;
#ifdef Q5G_EXP_NODEQ
                    af[mi][rh] = w ^ half2_as_u32(s2[mi][rh][grp]);
                    af[mi][2 + rh] = (w >> 8) ^ half2_as_u32(o2[mi][rh][grp]);
#else
                    af[mi][rh] = dequant_pair(w, 0x4140u, s2[mi][rh][grp], o2[mi][rh][grp]);       // k 2t4, 2t4+1
                    af[mi][2 + rh] = dequant_pair(w, 0x4342u, s2[mi][rh][grp], o2[mi][rh][grp]);   // k 2t4+8, 2t4+9
#endif
                }
            uint32_t bf[NI][2];
#pragma unroll
            for (int np = 0; np < NI / 2; np++) {
                const int n = wn * (BN / WN) + np * 16 + (lane & 7) + ((lane >> 4) << 3);
                const int chunk = step * 2 + ((lane >> 3) & 1);
                ldmatrix4(bf[np * 2][0], bf[np * 2][1], bf[np * 2 + 1][0], bf[np * 2 + 1][1],
                          b_stage + (size_t)n * Q5K_STAGE_B + ((chunk ^ (n & 7)) * 16));
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
__global__ void splitk_reduce5_kernel(const float *__restrict__ part, float *__restrict__ Y,
                                     const float *__restrict__ bias, int rows, size_t count, int splits) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x) {
        float s = bias ? bias[i % rows] : 0.f;
        for (int z = 0; z < splits; z++) s += part[(size_t)z * count + i];
        Y[i] = s;
    }
}

// How many k slices a narrow batch should be split into so that every SM gets work (1 = none).
static int splitk_choice5(int row_blocks,  /* all blocks of the grid */  int ktiles, int batch, int rows, size_t spare_bytes) {
    if (row_blocks >= 92) return 1;
    int splits = (92 + row_blocks - 1) / row_blocks;
    if (splits > 8) splits = 8;
    while (splits > 1 && (ktiles / splits < 8 || (size_t)splits * batch * rows * 4 > spare_bytes)) splits--;
    return splits;
}

template <int BM, int BN, int WM, int WN, int STAGES>
static int launch5_cfg(cudaStream_t s, const uint8_t *W, const __half *X16, float *Y, const float *bias, int rows,
                      int cols, int batch, float *part, size_t part_bytes) {
    constexpr size_t smem = (size_t)STAGES * (BM * Q5K_STAGE_A + BN * Q5K_STAGE_B);
    if (smem > (size_t)fllm_smem_limit()) {
        if constexpr (STAGES > 2) return launch5_cfg<BM, BN, WM, WN, 2>(s, W, X16, Y, bias, rows, cols, batch, part, part_bytes);
        else return 0;
    }
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(qgemm5k_kernel<BM, BN, WM, WN, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        configured = true;
    }
    const int row_blocks = (rows + BM - 1) / BM;
    const int splits = splitk_choice5(row_blocks * ((batch + BN - 1) / BN), cols >> 6, batch, rows, part_bytes);
    const dim3 grid(row_blocks, (batch + BN - 1) / BN, splits), block(WM * WN * 32);
    qgemm5k_kernel<BM, BN, WM, WN, STAGES><<<grid, block, smem, s>>>(W, X16, splits > 1 ? part : Y, bias, rows, cols, batch);
    if (splits > 1) {
        const size_t count = (size_t)batch * rows;
        splitk_reduce5_kernel<<<(unsigned)((count + 255) / 256), 256, 0, s>>>(part, Y, bias, rows, count, splits);
    }
    return 1;
}


// ---------------------------------------------------------------------------------------------
// Narrow batches: one pipeline stage is a whole super-block (256 values, 144 contiguous bytes per row), so
// every weight request is one long, aligned burst and the header is fetched once instead of four times.
// ---------------------------------------------------------------------------------------------
constexpr int Q5K_SB_A = 176;  // bytes per row and stage
constexpr int Q5K_SB_B = 512;  // bytes per token and stage: 256 halves

template <int BM, int BN, int WM, int WN, int STAGES>
__global__ void __launch_bounds__(WM * WN * 32)
qgemm5k_sb_kernel(const uint8_t *__restrict__ W, const __half *__restrict__ X16, float *__restrict__ Y,
                  const float *__restrict__ bias, int rows, int cols, int batch) {
#if FLLM_TC
    constexpr int NT = WM * WN * 32;
    constexpr int MI = BM / WM / 16;
    constexpr int NI = BN / WN / 8;
    static_assert(NI % 2 == 0 && MI >= 1, "tile shape");
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char *As = smem;                                     // [STAGES][BM][144]
    unsigned char *Bs = smem + (size_t)STAGES * BM * Q5K_SB_A;    // [STAGES][BN][512], chunks swizzled
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp % WM, wn = warp / WM;
    const int g = lane >> 2, t4 = lane & 3;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    const int nsb_all = cols >> 8;
    const size_t row_bytes = (size_t)nsb_all * 176;
    const int splits = gridDim.z;
    const int per = (nsb_all + splits - 1) / splits;
    const int sb0 = blockIdx.z * per;
    const int nsb = max(0, min(nsb_all, sb0 + per) - sb0);

    auto issue = [&](int sb, int stage) {
        for (int c = tid; c < BM * 11; c += NT) {  // 11 chunks of 16 bytes per row
            const int r = c / 11, part = c % 11;
            const int row = m0 + r;
            const bool ok = row < rows;
            const uint8_t *src = W + (size_t)(ok ? row : 0) * row_bytes + (size_t)sb * 176 + part * 16;
            cp_async16(As + ((size_t)stage * BM + r) * Q5K_SB_A + part * 16, src, ok);
        }
        for (int c = tid; c < BN * 32; c += NT) {  // 32 chunks of 16 bytes per token
            const int n = c >> 5, chunk = c & 31;
            const int tok = n0 + n;
            const bool ok = tok < batch;
            const __half *src = X16 + (size_t)(ok ? tok : 0) * cols + (size_t)sb * 256 + chunk * 8;
            cp_async16(Bs + ((size_t)stage * BN + n) * Q5K_SB_B + ((chunk ^ (n & 7)) * 16), src, ok);
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
        const unsigned char *a_stage = As + (size_t)stage * BM * Q5K_SB_A;
        const unsigned char *b_stage = Bs + (size_t)stage * BN * Q5K_SB_B;

        uint4 hdr[MI][2];
#pragma unroll
        for (int mi = 0; mi < MI; mi++)
#pragma unroll
            for (int rh = 0; rh < 2; rh++) {
                const int r = wm * (BM / WM) + mi * 16 + g + rh * 8;
                hdr[mi][rh] = *reinterpret_cast<const uint4 *>(a_stage + (size_t)r * Q5K_SB_A);
            }

#pragma unroll
        for (int sub = 0; sub < 4; sub++) {  // 64 values (two sub-blocks) each
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
#pragma unroll
            for (int step = 0; step < 4; step++) {
                const int grp = step >> 1, b0 = (step & 1) * 16;
                uint32_t af[MI][4];
#pragma unroll
                for (int mi = 0; mi < MI; mi++)
#pragma unroll
                    for (int rh = 0; rh < 2; rh++) {
                        const int r = wm * (BM / WM) + mi * 16 + g + rh * 8;
                        uint32_t w = *reinterpret_cast<const uint32_t *>(a_stage + (size_t)r * Q5K_SB_A + 48 + sub * 32 + b0 + 4 * t4);
                        const uint32_t hq = *reinterpret_cast<const uint32_t *>(a_stage + (size_t)r * Q5K_SB_A + 16 + b0 + 4 * t4);
                        w = (grp == 0 ? w : (w >> 4)) & 0x0F0F0F0Fu;
                        w |= ((hq >> (2 * sub + grp)) & 0x01010101u) << 4;
                        af[mi][rh] = dequant_pair(w, 0x4140u, s2[mi][rh][grp], o2[mi][rh][grp]);
                        af[mi][2 + rh] = dequant_pair(w, 0x4342u, s2[mi][rh][grp], o2[mi][rh][grp]);
                    }
                uint32_t bf[NI][2];
#pragma unroll
                for (int np = 0; np < NI / 2; np++) {
                    const int n = wn * (BN / WN) + np * 16 + (lane & 7) + ((lane >> 4) << 3);
                    const int chunk = (sub * 4 + step) * 2 + ((lane >> 3) & 1);
                    ldmatrix4(bf[np * 2][0], bf[np * 2][1], bf[np * 2 + 1][0], bf[np * 2 + 1][1],
                              b_stage + (size_t)n * Q5K_SB_B + ((chunk ^ (n & 7)) * 16));
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
static int launch5_sb(cudaStream_t s, const uint8_t *W, const __half *X16, float *Y, const float *bias, int rows,
                     int cols, int batch, float *part, size_t part_bytes) {
    constexpr size_t smem = (size_t)STAGES * (BM * Q5K_SB_A + BN * Q5K_SB_B);
    if (smem > (size_t)fllm_smem_limit()) {
        if constexpr (STAGES > 2) return launch5_sb<BM, BN, WM, WN, 2>(s, W, X16, Y, bias, rows, cols, batch, part, part_bytes);
        else return 0;
    }
    static_assert(smem <= 99 * 1024, "shared memory budget");
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(qgemm5k_sb_kernel<BM, BN, WM, WN, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        configured = true;
    }
    const int row_blocks = (rows + BM - 1) / BM;
    int splits = splitk_choice5(row_blocks * ((batch + BN - 1) / BN), cols >> 6, batch, rows, part_bytes);
    while (splits > 1 && (cols >> 8) / splits < 2) splits--;
    const dim3 grid(row_blocks, (batch + BN - 1) / BN, splits), block(WM * WN * 32);
    qgemm5k_sb_kernel<BM, BN, WM, WN, STAGES><<<grid, block, smem, s>>>(W, X16, splits > 1 ? part : Y, bias, rows, cols, batch);
    if (splits > 1) {
        const size_t count = (size_t)batch * rows;
        splitk_reduce5_kernel<<<(unsigned)((count + 255) / 256), 256, 0, s>>>(part, Y, bias, rows, count, splits);
    }
    return 1;
}

#ifndef Q5G_SB_BM
#define Q5G_SB_BM 64
#endif
#ifndef Q5G_SB_WM
#define Q5G_SB_WM 4
#endif
#ifndef Q5G_SB_ST
#define Q5G_SB_ST 4
#endif

#ifndef Q5G_S_BM
#define Q5G_S_BM 128
#endif
#ifndef Q5G_S_WM
#define Q5G_S_WM 4
#endif
#ifndef Q5G_S_ST
#define Q5G_S_ST 4
#endif
#ifndef Q5G_BIG_BM
#define Q5G_BIG_BM 128
#endif
#ifndef Q5G_BIG_BN
#define Q5G_BIG_BN 256
#endif
#ifndef Q5G_BIG_WM
#define Q5G_BIG_WM 8
#endif
#ifndef Q5G_BIG_WN
#define Q5G_BIG_WN 1
#endif
#ifndef Q5G_BIG_ST
#define Q5G_BIG_ST 2
#endif

// Q5_K GEMM for `batch` >= 2 tokens. `x16` is scratch for batch * cols halves (16-byte aligned).
// Returns 1 when handled, 0 otherwise (the caller falls back to other kernels).
int fllm_qgemm5k_try(cudaStream_t s, const void *W, const float *X, float *Y, const float *bias, int rows, int cols,
                     int batch, void *x16_raw, size_t x16_bytes) {
    if (fllm_cc() < 75) return 0;
    __half *x16 = static_cast<__half *>(x16_raw);
    static const bool disabled = getenv("FLLM_CUDA_NO_QGEMM") != nullptr;
    if (disabled || batch < 2 || (cols & 255) != 0 || (reinterpret_cast<uintptr_t>(X) & 15) != 0 ||
        (reinterpret_cast<uintptr_t>(W) & 15) != 0 || (size_t)batch * cols * 2 > x16_bytes)
        return 0;
    fllm_convert_x16(s, X, x16_raw, batch, cols);
    const uint8_t *w = static_cast<const uint8_t *>(W);
    const size_t x_used = ((size_t)batch * cols * 2 + 255) & ~size_t(255);
    float *part = reinterpret_cast<float *>(static_cast<char *>(x16_raw) + x_used);
    const size_t part_bytes = x16_bytes > x_used ? x16_bytes - x_used : 0;
#ifndef Q5G_NO_SB
    if (batch > 16 && batch <= 32) return launch5_sb<64, 32, 4, 1, 3>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
    if (batch > 32 && batch <= 64) return launch5_sb<64, 64, 4, 1, 2>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
    if (batch <= 16) return launch5_sb<Q5G_SB_BM, 16, Q5G_SB_WM, 1, Q5G_SB_ST>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
#endif
    if (batch <= 16) return launch5_cfg<Q5G_S_BM, 16, Q5G_S_WM, 1, Q5G_S_ST>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
    if (batch <= 32) return launch5_cfg<128, 32, 4, 1, 4>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
    if (batch <= 64) return launch5_cfg<128, 64, 4, 1, 4>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
    if (batch <= 128) return launch5_cfg<128, 128, 4, 2, 4>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
    auto run = [&](int i) -> int {
        switch (i) {
            case 0: {
                // narrower token tile when the last wide tile would be mostly padding
                const double wide = 74.0 * batch / (((batch + 255) / 256) * 256), narrow = 66.0 * batch / (((batch + 127) / 128) * 128);
                if (narrow > wide) return launch5_cfg<128, 128, 8, 1, 3>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
                return launch5_cfg<Q5G_BIG_BM, Q5G_BIG_BN, Q5G_BIG_WM, Q5G_BIG_WN, Q5G_BIG_ST>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
            }
            case 1: return launch5_cfg<128, 128, 8, 1, 3>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
            case 2: return launch5_cfg<128, 128, 4, 2, 3>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
            case 3: return launch5_cfg<128, 192, 4, 2, 2>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
            case 4: return launch5_cfg<128, 256, 4, 2, 2>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
            case 5: return launch5_cfg<256, 128, 8, 2, 2>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
            default: return launch5_cfg<128, 96, 4, 1, 3>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
        }
    };
    return run(fllm_autotune(s, 0x5B, rows, cols, batch, 7, run));
}
