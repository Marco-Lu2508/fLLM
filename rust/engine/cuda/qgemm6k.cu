// fLLM CUDA backend: tensor-core GEMM for Q6_K weights with fragment-level dequantization.
//
// Same structure as qgemm4k.cu (cp.async pipelines for raw weights and FP16 activations, weights
// turned into FP16 inside the registers that feed `mma.m16n8k16`). Q6_K blocks are stored padded to
// 224 bytes on the device, so every field is 16-byte aligned. One k-tile is half a super-block (128
// values): 64 bytes of low nibbles, 32 bytes of high bit pairs, the 16 scales and d.
//
// Value layout inside a half super-block (ggml): element 32 * quarter + l (l < 32) takes its low
// nibble from ql[32 * (quarter & 1) + l] (high nibble when quarter >= 2) and two more bits from
// qh[l] >> (2 * quarter). A thread needs four consecutive elements per row and 16-element step, which
// are four consecutive bytes of ql and qh: one 32-bit shared-memory load each.
#include <cstdlib>

#include "common.cuh"
#include "fllm_cuda_internal.h"
#include "mma_util.cuh"

constexpr int Q6K_STAGE_A = 128;  // bytes per row and k-tile: 4 chunks ql, 2 chunks qh, scales, d
constexpr int Q6K_STAGE_B = 256;  // bytes per token and k-tile: 128 halves

template <int BM, int BN, int WM, int WN, int STAGES>
__global__ void __launch_bounds__(WM * WN * 32)
qgemm6k_kernel(const uint8_t *__restrict__ W, const __half *__restrict__ X16, float *__restrict__ Y,
               const float *__restrict__ bias, int rows, int cols, int batch) {
#if FLLM_TC
    constexpr int NT = WM * WN * 32;
    constexpr int MI = BM / WM / 16;
    constexpr int NI = BN / WN / 8;
    static_assert(NI % 2 == 0 && MI >= 1, "tile shape");
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char *As = smem;                                      // [STAGES][BM][128], chunks swizzled by row
    unsigned char *Bs = smem + (size_t)STAGES * BM * Q6K_STAGE_A;  // [STAGES][BN][256], chunks swizzled by token
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp % WM, wn = warp / WM;
    const int g = lane >> 2, t4 = lane & 3;
    int m0, n0;
    qg_tile_origin(BM, BN, batch, m0, n0);
    const int nsb = cols >> 8;
    const size_t row_bytes = (size_t)nsb * 224;
    const int ktiles_all = cols >> 7;
    const int splits = gridDim.z;
    const int per = (ktiles_all + splits - 1) / splits;
    const int kt0 = blockIdx.z * per;
    const int ktiles = max(0, min(ktiles_all, kt0 + per) - kt0);  // k tiles of this block

    auto issue = [&](int kt, int stage) {
        const int sb = kt >> 1, half = kt & 1;
        for (int c = tid; c < BM * 8; c += NT) {
            const int r = c >> 3, part = c & 7;
            const int row = m0 + r;
            const bool ok = row < rows;
            const int off = part < 4 ? 64 * half + 16 * part : (part < 6 ? 128 + 32 * half + 16 * (part - 4) : (part == 6 ? 192 : 208));
            const uint8_t *src = W + (size_t)(ok ? row : 0) * row_bytes + (size_t)sb * 224 + off;
            cp_async16(As + ((size_t)stage * BM + r) * Q6K_STAGE_A + ((part ^ (r & 7)) * 16), src, ok);
        }
        for (int c = tid; c < BN * 16; c += NT) {
            const int n = c >> 4, chunk = c & 15;
            const int tok = n0 + n;
            const bool ok = tok < batch;
            const __half *src = X16 + (size_t)(ok ? tok : 0) * cols + (size_t)kt * 128 + chunk * 8;
            cp_async16(Bs + ((size_t)stage * BN + n) * Q6K_STAGE_B + ((chunk ^ (n & 7)) * 16), src, ok);
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
        const unsigned char *a_stage = As + (size_t)stage * BM * Q6K_STAGE_A;
        const unsigned char *b_stage = Bs + (size_t)stage * BN * Q6K_STAGE_B;
        const int half = (kt0 + kt) & 1;
        // (r & 7) == g for every row this thread touches, so one swizzle value serves all of them
        const int sw = g;

        float dd[MI][2];
#pragma unroll
        for (int mi = 0; mi < MI; mi++)
#pragma unroll
            for (int rh = 0; rh < 2; rh++) {
                const int r = wm * (BM / WM) + mi * 16 + g + rh * 8;
                dd[mi][rh] = h2f_bits(*reinterpret_cast<const uint16_t *>(a_stage + (size_t)r * Q6K_STAGE_A + ((7 ^ sw) * 16)));
            }

#pragma unroll
        for (int step = 0; step < 8; step++) {
            uint32_t af[MI][4];
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
                for (int rh = 0; rh < 2; rh++) {
                    const int r = wm * (BM / WM) + mi * 16 + g + rh * 8;
                    const unsigned char *row = a_stage + (size_t)r * Q6K_STAGE_A;
                    const uint32_t qlw = *reinterpret_cast<const uint32_t *>(row + ((((step >> 1) & 1) * 2 + (step & 1)) ^ sw) * 16 + 4 * t4);
                    const uint32_t qhw = *reinterpret_cast<const uint32_t *>(row + ((4 + (step & 1)) ^ sw) * 16 + 4 * t4);
                    const int8_t sc = *reinterpret_cast<const int8_t *>(row + (6 ^ sw) * 16 + 8 * half + step);
                    const uint32_t q = ((qlw >> (4 * (step >> 2))) & 0x0F0F0F0Fu) | (((qhw >> (2 * (step >> 1))) & 0x03030303u) << 4);
                    const float s = dd[mi][rh] * (float)sc;
                    const __half2 s2 = __half2half2(__float2half_rn(s));
                    const __half2 o2 = __half2half2(__float2half_rn(32.f * s));
                    af[mi][rh] = dequant_pair(q, 0x4140u, s2, o2);
                    af[mi][2 + rh] = dequant_pair(q, 0x4342u, s2, o2);
                }
            uint32_t bf[NI][2];
#pragma unroll
            for (int np = 0; np < NI / 2; np++) {
                const int n = wn * (BN / WN) + np * 16 + (lane & 7) + ((lane >> 4) << 3);
                const int chunk = step * 2 + ((lane >> 3) & 1);
                ldmatrix4(bf[np * 2][0], bf[np * 2][1], bf[np * 2 + 1][0], bf[np * 2 + 1][1],
                          b_stage + (size_t)n * Q6K_STAGE_B + ((chunk ^ (n & 7)) * 16));
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
__global__ void splitk_reduce6_kernel(const float *__restrict__ part, float *__restrict__ Y,
                                     const float *__restrict__ bias, int rows, size_t count, int splits) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x) {
        float s = bias ? bias[i % rows] : 0.f;
        for (int z = 0; z < splits; z++) s += part[(size_t)z * count + i];
        Y[i] = s;
    }
}

// How many k slices a narrow batch should be split into so that every SM gets work (1 = none).
static int splitk_choice6(int row_blocks,  /* all blocks of the grid */  int ktiles, int batch, int rows, size_t spare_bytes) {
    if (row_blocks >= 92) return 1;
    int splits = (92 + row_blocks - 1) / row_blocks;
    if (splits > 8) splits = 8;
    while (splits > 1 && (ktiles / splits < 8 || (size_t)splits * batch * rows * 4 > spare_bytes)) splits--;
    return splits;
}

template <int BM, int BN, int WM, int WN, int STAGES>
static int q6_launch(cudaStream_t s, const uint8_t *W, const __half *X16, float *Y, const float *bias, int rows,
                     int cols, int batch, float *part, size_t part_bytes) {
    constexpr size_t smem = (size_t)STAGES * (BM * Q6K_STAGE_A + BN * Q6K_STAGE_B);
    if (smem > (size_t)fllm_smem_limit()) {
        if constexpr (STAGES > 2) return q6_launch<BM, BN, WM, WN, 2>(s, W, X16, Y, bias, rows, cols, batch, part, part_bytes);
        else return 0;
    }
    static_assert(smem <= 99 * 1024, "shared memory budget");
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(qgemm6k_kernel<BM, BN, WM, WN, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        configured = true;
    }
    const int row_blocks = (rows + BM - 1) / BM;
    const int splits = splitk_choice6(row_blocks * ((batch + BN - 1) / BN), cols >> 7, batch, rows, part_bytes);
    const dim3 grid(row_blocks, (batch + BN - 1) / BN, splits), block(WM * WN * 32);
    qgemm6k_kernel<BM, BN, WM, WN, STAGES><<<grid, block, smem, s>>>(W, X16, splits > 1 ? part : Y, bias, rows, cols, batch);
    if (splits > 1) {
        const size_t count = (size_t)batch * rows;
        splitk_reduce6_kernel<<<(unsigned)((count + 255) / 256), 256, 0, s>>>(part, Y, bias, rows, count, splits);
    }
    return 1;
}


// ---------------------------------------------------------------------------------------------
// Narrow batches: a pipeline stage is a whole super-block (224 contiguous bytes per row), see qgemm4k.cu.
// ---------------------------------------------------------------------------------------------
constexpr int Q6K_SB_A = 240;  // 224 data bytes padded so the 32-bit fragment loads are bank-conflict free
constexpr int Q6K_SB_B = 512;

template <int BM, int BN, int WM, int WN, int STAGES>
__global__ void __launch_bounds__(WM * WN * 32)
qgemm6k_sb_kernel(const uint8_t *__restrict__ W, const __half *__restrict__ X16, float *__restrict__ Y,
                  const float *__restrict__ bias, int rows, int cols, int batch) {
#if FLLM_TC
    constexpr int NT = WM * WN * 32;
    constexpr int MI = BM / WM / 16;
    constexpr int NI = BN / WN / 8;
    static_assert(NI % 2 == 0 && MI >= 1, "tile shape");
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char *As = smem;
    unsigned char *Bs = smem + (size_t)STAGES * BM * Q6K_SB_A;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp % WM, wn = warp / WM;
    const int g = lane >> 2, t4 = lane & 3;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    const int nsb_all = cols >> 8;
    const size_t row_bytes = (size_t)nsb_all * 224;
    const int splits = gridDim.z;
    const int per = (nsb_all + splits - 1) / splits;
    const int sb0 = blockIdx.z * per;
    const int nsb = max(0, min(nsb_all, sb0 + per) - sb0);

    auto issue = [&](int sb, int stage) {
        for (int c = tid; c < BM * 14; c += NT) {
            const int r = c / 14, part = c % 14;
            const int row = m0 + r;
            const bool ok = row < rows;
            const uint8_t *src = W + (size_t)(ok ? row : 0) * row_bytes + (size_t)sb * 224 + part * 16;
            cp_async16(As + ((size_t)stage * BM + r) * Q6K_SB_A + part * 16, src, ok);
        }
        for (int c = tid; c < BN * 32; c += NT) {
            const int n = c >> 5, chunk = c & 31;
            const int tok = n0 + n;
            const bool ok = tok < batch;
            const __half *src = X16 + (size_t)(ok ? tok : 0) * cols + (size_t)sb * 256 + chunk * 8;
            cp_async16(Bs + ((size_t)stage * BN + n) * Q6K_SB_B + ((chunk ^ (n & 7)) * 16), src, ok);
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
        const unsigned char *a_stage = As + (size_t)stage * BM * Q6K_SB_A;
        const unsigned char *b_stage = Bs + (size_t)stage * BN * Q6K_SB_B;

        float dd[MI][2];
#pragma unroll
        for (int mi = 0; mi < MI; mi++)
#pragma unroll
            for (int rh = 0; rh < 2; rh++) {
                const int r = wm * (BM / WM) + mi * 16 + g + rh * 8;
                dd[mi][rh] = h2f_bits(*reinterpret_cast<const uint16_t *>(a_stage + (size_t)r * Q6K_SB_A + 208));
            }

#pragma unroll
        for (int half = 0; half < 2; half++) {
#pragma unroll
            for (int step = 0; step < 8; step++) {
                uint32_t af[MI][4];
#pragma unroll
                for (int mi = 0; mi < MI; mi++)
#pragma unroll
                    for (int rh = 0; rh < 2; rh++) {
                        const int r = wm * (BM / WM) + mi * 16 + g + rh * 8;
                        const unsigned char *row = a_stage + (size_t)r * Q6K_SB_A;
                        const uint32_t qlw = *reinterpret_cast<const uint32_t *>(row + 64 * half + ((step >> 1) & 1) * 32 + (step & 1) * 16 + 4 * t4);
                        const uint32_t qhw = *reinterpret_cast<const uint32_t *>(row + 128 + 32 * half + (step & 1) * 16 + 4 * t4);
                        const int8_t sc = *reinterpret_cast<const int8_t *>(row + 192 + 8 * half + step);
                        const uint32_t q = ((qlw >> (4 * (step >> 2))) & 0x0F0F0F0Fu) | (((qhw >> (2 * (step >> 1))) & 0x03030303u) << 4);
                        const float sv = dd[mi][rh] * (float)sc;
                        const __half2 s2 = __half2half2(__float2half_rn(sv));
                        const __half2 o2 = __half2half2(__float2half_rn(32.f * sv));
                        af[mi][rh] = dequant_pair(q, 0x4140u, s2, o2);
                        af[mi][2 + rh] = dequant_pair(q, 0x4342u, s2, o2);
                    }
                uint32_t bf[NI][2];
#pragma unroll
                for (int np = 0; np < NI / 2; np++) {
                    const int n = wn * (BN / WN) + np * 16 + (lane & 7) + ((lane >> 4) << 3);
                    const int chunk = (half * 8 + step) * 2 + ((lane >> 3) & 1);
                    ldmatrix4(bf[np * 2][0], bf[np * 2][1], bf[np * 2 + 1][0], bf[np * 2 + 1][1],
                              b_stage + (size_t)n * Q6K_SB_B + ((chunk ^ (n & 7)) * 16));
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
static int q6_launch_sb(cudaStream_t s, const uint8_t *W, const __half *X16, float *Y, const float *bias, int rows,
                        int cols, int batch, float *part, size_t part_bytes) {
    constexpr size_t smem = (size_t)STAGES * (BM * Q6K_SB_A + BN * Q6K_SB_B);
    if (smem > (size_t)fllm_smem_limit()) {
        if constexpr (STAGES > 2) return q6_launch_sb<BM, BN, WM, WN, 2>(s, W, X16, Y, bias, rows, cols, batch, part, part_bytes);
        else return 0;
    }
    static_assert(smem <= 99 * 1024, "shared memory budget");
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(qgemm6k_sb_kernel<BM, BN, WM, WN, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        configured = true;
    }
    const int row_blocks = (rows + BM - 1) / BM;
    int splits = splitk_choice6(row_blocks * ((batch + BN - 1) / BN), cols >> 7, batch, rows, part_bytes);
    while (splits > 1 && (cols >> 8) / splits < 2) splits--;
    const dim3 grid(row_blocks, (batch + BN - 1) / BN, splits), block(WM * WN * 32);
    qgemm6k_sb_kernel<BM, BN, WM, WN, STAGES><<<grid, block, smem, s>>>(W, X16, splits > 1 ? part : Y, bias, rows, cols, batch);
    if (splits > 1) {
        const size_t count = (size_t)batch * rows;
        splitk_reduce6_kernel<<<(unsigned)((count + 255) / 256), 256, 0, s>>>(part, Y, bias, rows, count, splits);
    }
    return 1;
}

#ifndef Q6_SB_BM
#define Q6_SB_BM 64
#endif
#ifndef Q6_SB_WM
#define Q6_SB_WM 4
#endif
#ifndef Q6_SB_ST
#define Q6_SB_ST 4
#endif

#ifndef Q6_BIG_BM
#define Q6_BIG_BM 128
#endif
#ifndef Q6_BIG_BN
#define Q6_BIG_BN 128
#endif
#ifndef Q6_BIG_WM
#define Q6_BIG_WM 8
#endif
#ifndef Q6_BIG_WN
#define Q6_BIG_WN 1
#endif
#ifndef Q6_BIG_ST
#define Q6_BIG_ST 2
#endif

// Q6_K GEMM for `batch` >= 2 tokens. `x16` is scratch for batch * cols halves (16-byte aligned).
int fllm_qgemm6k_try(cudaStream_t s, const void *W, const float *X, float *Y, const float *bias, int rows, int cols,
                     int batch, void *x16_raw, size_t x16_bytes) {
    if (fllm_cc() < 75) return 0;
    static const bool disabled = getenv("FLLM_CUDA_NO_QGEMM") != nullptr;
    if (disabled || batch < 2 || (cols & 255) != 0 || (reinterpret_cast<uintptr_t>(X) & 15) != 0 ||
        (reinterpret_cast<uintptr_t>(W) & 15) != 0 || (size_t)batch * cols * 2 > x16_bytes)
        return 0;
    fllm_convert_x16(s, X, x16_raw, batch, cols);
    const __half *x16 = static_cast<const __half *>(x16_raw);
    const uint8_t *w = static_cast<const uint8_t *>(W);
    const size_t x_used = ((size_t)batch * cols * 2 + 255) & ~size_t(255);
    float *part = reinterpret_cast<float *>(static_cast<char *>(x16_raw) + x_used);
    const size_t part_bytes = x16_bytes > x_used ? x16_bytes - x_used : 0;
#ifndef Q6_NO_SB
    if (batch > 16 && batch <= 32) return q6_launch_sb<64, 32, 4, 1, 3>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
    if (batch > 32 && batch <= 64) return q6_launch_sb<64, 64, 4, 1, 2>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
    if (batch <= 16) return q6_launch_sb<Q6_SB_BM, 16, Q6_SB_WM, 1, Q6_SB_ST>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
#endif
    if (batch <= 16) return q6_launch<128, 16, 4, 1, 4>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
    if (batch <= 32) return q6_launch<128, 32, 4, 1, 4>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
    if (batch <= 64) return q6_launch<128, 64, 4, 1, 3>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
    auto run = [&](int i) -> int {
        switch (i) {
            case 0: return q6_launch<Q6_BIG_BM, Q6_BIG_BN, Q6_BIG_WM, Q6_BIG_WN, Q6_BIG_ST>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
            case 1: return q6_launch<128, 128, 4, 2, 2>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
            case 2: return q6_launch<128, 64, 4, 1, 3>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
            case 3: return q6_launch<64, 128, 4, 2, 2>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
            default: return q6_launch<128, 96, 4, 1, 2>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
        }
    };
    return run(fllm_autotune(s, 0x6B, rows, cols, batch, 5, run));
}
