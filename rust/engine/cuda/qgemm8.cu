// fLLM CUDA backend: tensor-core GEMM for Q8_0 weights (see qgemm4k.cu for the design).
//
// Device layout of a row (width a multiple of 256, built by backend/cuda.rs): `[f16 scale x blocks][int8 x 32 per
// block]`, so quantized vectors are 16-byte aligned. int8 values are turned into FP16 as (q + 128) - 128 with the
// usual 0x6400 magic number (the byte is xor-ed with 0x80 first), and the block scale is applied by the same
// fused multiply-add, so the dequantized weight is exact up to the final FP16 rounding.
#include <cstdlib>

#include "common.cuh"
#include "fllm_cuda_internal.h"
#include "mma_util.cuh"

__device__ __forceinline__ void cp_async4(void *smem, const void *gmem, bool valid) { cp_async4_ca(smem, gmem, valid); }

// d * (q) for the four int8 values of `w` that go into the two halves of a fragment register pair
__device__ __forceinline__ void q8_pair(uint32_t w, float d, uint32_t &lo, uint32_t &hi) {
    const __half2 s2 = __half2half2(__float2half_rn(d));
    const __half2 o2 = __half2half2(__float2half_rn(128.f * d));
    const uint32_t u = w ^ 0x80808080u;
    lo = dequant_pair(u, 0x4140u, s2, o2);
    hi = dequant_pair(u, 0x4342u, s2, o2);
}

constexpr int Q8_STAGE_A = 80;   // per row and 64-value k-tile: 4 scale bytes (+12 padding) + 64 quant bytes
constexpr int Q8_STAGE_B = 128;  // per token and k-tile: 64 halves

template <int BM, int BN, int WM, int WN, int STAGES>
__global__ void __launch_bounds__(WM * WN * 32)
qgemm8_kernel(const uint8_t *__restrict__ W, const __half *__restrict__ X16, float *__restrict__ Y,
              const float *__restrict__ bias, int rows, int cols, int batch) {
#if FLLM_TC
    constexpr int NT = WM * WN * 32;
    constexpr int MI = BM / WM / 16;
    constexpr int NI = BN / WN / 8;
    static_assert(NI % 2 == 0 && MI >= 1, "tile shape");
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char *As = smem;
    unsigned char *Bs = smem + (size_t)STAGES * BM * Q8_STAGE_A;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp % WM, wn = warp / WM;
    const int g = lane >> 2, t4 = lane & 3;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    const int nb = cols >> 5;
    const size_t row_bytes = (size_t)nb * 34;
    const int ktiles_all = cols >> 6;
    const int splits = gridDim.z;
    const int per = (ktiles_all + splits - 1) / splits;
    const int kt0 = blockIdx.z * per;
    const int ktiles = max(0, min(ktiles_all, kt0 + per) - kt0);

    auto issue = [&](int kt, int stage) {
        for (int c = tid; c < BM * 5; c += NT) {
            const int r = c / 5, part = c % 5;
            const int row = m0 + r;
            const bool ok = row < rows;
            const uint8_t *rowp = W + (size_t)(ok ? row : 0) * row_bytes;
            unsigned char *dst = As + ((size_t)stage * BM + r) * Q8_STAGE_A + part * 16;
            if (part == 0) cp_async4(dst, rowp + (size_t)kt * 4, ok);
            else cp_async16(dst, rowp + (size_t)nb * 2 + (size_t)kt * 64 + (part - 1) * 16, ok);
        }
        for (int c = tid; c < BN * 8; c += NT) {
            const int n = c >> 3, chunk = c & 7;
            const int tok = n0 + n;
            const bool ok = tok < batch;
            const __half *src = X16 + (size_t)(ok ? tok : 0) * cols + (size_t)kt * 64 + chunk * 8;
            cp_async16(Bs + ((size_t)stage * BN + n) * Q8_STAGE_B + ((chunk ^ (n & 7)) * 16), src, ok);
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
        const unsigned char *a_stage = As + (size_t)stage * BM * Q8_STAGE_A;
        const unsigned char *b_stage = Bs + (size_t)stage * BN * Q8_STAGE_B;

#pragma unroll
        for (int step = 0; step < 4; step++) {
            uint32_t af[MI][4];
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
                for (int rh = 0; rh < 2; rh++) {
                    const int r = wm * (BM / WM) + mi * 16 + g + rh * 8;
                    const unsigned char *row = a_stage + (size_t)r * Q8_STAGE_A;
                    const uint32_t w = *reinterpret_cast<const uint32_t *>(row + 16 + step * 16 + 4 * t4);
                    const float d = h2f_bits(*reinterpret_cast<const uint16_t *>(row + (step >> 1) * 2));
                    q8_pair(w, d, af[mi][rh], af[mi][2 + rh]);
                }
            uint32_t bf[NI][2];
#pragma unroll
            for (int np = 0; np < NI / 2; np++) {
                const int n = wn * (BN / WN) + np * 16 + (lane & 7) + ((lane >> 4) << 3);
                const int chunk = step * 2 + ((lane >> 3) & 1);
                ldmatrix4(bf[np * 2][0], bf[np * 2][1], bf[np * 2 + 1][0], bf[np * 2 + 1][1],
                          b_stage + (size_t)n * Q8_STAGE_B + ((chunk ^ (n & 7)) * 16));
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

// Narrow batches: one stage = 256 values (16 scale bytes + 256 quant bytes per row), see qgemm4k.cu.
constexpr int Q8_SB_A = 272;
constexpr int Q8_SB_B = 512;

template <int BM, int BN, int WM, int WN, int STAGES>
__global__ void __launch_bounds__(WM * WN * 32)
qgemm8_sb_kernel(const uint8_t *__restrict__ W, const __half *__restrict__ X16, float *__restrict__ Y,
                 const float *__restrict__ bias, int rows, int cols, int batch) {
#if FLLM_TC
    constexpr int NT = WM * WN * 32;
    constexpr int MI = BM / WM / 16;
    constexpr int NI = BN / WN / 8;
    static_assert(NI % 2 == 0 && MI >= 1, "tile shape");
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char *As = smem;
    unsigned char *Bs = smem + (size_t)STAGES * BM * Q8_SB_A;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp % WM, wn = warp / WM;
    const int g = lane >> 2, t4 = lane & 3;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    const int nb = cols >> 5;
    const size_t row_bytes = (size_t)nb * 34;
    const int nsb_all = cols >> 8;
    const int splits = gridDim.z;
    const int per = (nsb_all + splits - 1) / splits;
    const int sb0 = blockIdx.z * per;
    const int nsb = max(0, min(nsb_all, sb0 + per) - sb0);

    auto issue = [&](int sb, int stage) {
        for (int c = tid; c < BM * 17; c += NT) {
            const int r = c / 17, part = c % 17;
            const int row = m0 + r;
            const bool ok = row < rows;
            const uint8_t *rowp = W + (size_t)(ok ? row : 0) * row_bytes;
            const uint8_t *src = part == 0 ? rowp + (size_t)sb * 16 : rowp + (size_t)nb * 2 + (size_t)sb * 256 + (part - 1) * 16;
            cp_async16(As + ((size_t)stage * BM + r) * Q8_SB_A + part * 16, src, ok);
        }
        for (int c = tid; c < BN * 32; c += NT) {
            const int n = c >> 5, chunk = c & 31;
            const int tok = n0 + n;
            const bool ok = tok < batch;
            const __half *src = X16 + (size_t)(ok ? tok : 0) * cols + (size_t)sb * 256 + chunk * 8;
            cp_async16(Bs + ((size_t)stage * BN + n) * Q8_SB_B + ((chunk ^ (n & 7)) * 16), src, ok);
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
        const unsigned char *a_stage = As + (size_t)stage * BM * Q8_SB_A;
        const unsigned char *b_stage = Bs + (size_t)stage * BN * Q8_SB_B;

#pragma unroll
        for (int sub = 0; sub < 4; sub++) {
#pragma unroll
            for (int step = 0; step < 4; step++) {
                uint32_t af[MI][4];
#pragma unroll
                for (int mi = 0; mi < MI; mi++)
#pragma unroll
                    for (int rh = 0; rh < 2; rh++) {
                        const int r = wm * (BM / WM) + mi * 16 + g + rh * 8;
                        const unsigned char *row = a_stage + (size_t)r * Q8_SB_A;
                        const uint32_t w = *reinterpret_cast<const uint32_t *>(row + 16 + sub * 64 + step * 16 + 4 * t4);
                        const float d = h2f_bits(*reinterpret_cast<const uint16_t *>(row + (sub * 2 + (step >> 1)) * 2));
                        q8_pair(w, d, af[mi][rh], af[mi][2 + rh]);
                    }
                uint32_t bf[NI][2];
#pragma unroll
                for (int np = 0; np < NI / 2; np++) {
                    const int n = wn * (BN / WN) + np * 16 + (lane & 7) + ((lane >> 4) << 3);
                    const int chunk = (sub * 4 + step) * 2 + ((lane >> 3) & 1);
                    ldmatrix4(bf[np * 2][0], bf[np * 2][1], bf[np * 2 + 1][0], bf[np * 2 + 1][1],
                              b_stage + (size_t)n * Q8_SB_B + ((chunk ^ (n & 7)) * 16));
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

__global__ void splitk_reduce8_kernel(const float *__restrict__ part, float *__restrict__ Y, const float *__restrict__ bias,
                                      int rows, size_t count, int splits) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x) {
        float s = bias ? bias[i % rows] : 0.f;
        for (int z = 0; z < splits; z++) s += part[(size_t)z * count + i];
        Y[i] = s;
    }
}

static int splitk_choice8(int grid_blocks, int units, int batch, int rows, size_t spare_bytes) {
    if (grid_blocks >= 92) return 1;
    int splits = (92 + grid_blocks - 1) / grid_blocks;
    if (splits > 8) splits = 8;
    while (splits > 1 && (units / splits < 2 || (size_t)splits * batch * rows * 4 > spare_bytes)) splits--;
    return splits;
}

template <int BM, int BN, int WM, int WN, int STAGES, bool SB>
static int q8_launch(cudaStream_t s, const uint8_t *W, const __half *X16, float *Y, const float *bias, int rows, int cols,
                     int batch, float *part, size_t part_bytes) {
    constexpr size_t smem = SB ? (size_t)STAGES * (BM * Q8_SB_A + BN * Q8_SB_B) : (size_t)STAGES * (BM * Q8_STAGE_A + BN * Q8_STAGE_B);
    if (smem > (size_t)fllm_smem_limit()) {
        if constexpr (STAGES > 2) return q8_launch<BM, BN, WM, WN, 2, SB>(s, W, X16, Y, bias, rows, cols, batch, part, part_bytes);
        else return 0;
    }
    static_assert(smem <= 99 * 1024, "shared memory budget");
    static bool configured = false;
    if (!configured) {
        if (SB) cudaFuncSetAttribute(qgemm8_sb_kernel<BM, BN, WM, WN, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        else cudaFuncSetAttribute(qgemm8_kernel<BM, BN, WM, WN, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        configured = true;
    }
    const int row_blocks = (rows + BM - 1) / BM;
    const int splits = splitk_choice8(row_blocks * ((batch + BN - 1) / BN), SB ? cols >> 8 : cols >> 6, batch, rows, part_bytes);
    const dim3 grid(row_blocks, (batch + BN - 1) / BN, splits), block(WM * WN * 32);
    if (SB) qgemm8_sb_kernel<BM, BN, WM, WN, STAGES><<<grid, block, smem, s>>>(W, X16, splits > 1 ? part : Y, bias, rows, cols, batch);
    else qgemm8_kernel<BM, BN, WM, WN, STAGES><<<grid, block, smem, s>>>(W, X16, splits > 1 ? part : Y, bias, rows, cols, batch);
    if (splits > 1) {
        const size_t count = (size_t)batch * rows;
        splitk_reduce8_kernel<<<(unsigned)((count + 255) / 256), 256, 0, s>>>(part, Y, bias, rows, count, splits);
    }
    return 1;
}

// Q8_0 GEMM for `batch` >= 2 tokens; the weights must be in the device layout (width a multiple of 256).
int fllm_qgemm8_try(cudaStream_t s, const void *W, const float *X, float *Y, const float *bias, int rows, int cols,
                    int batch, void *x16_raw, size_t x16_bytes) {
    if (fllm_cc() < 75) return 0;
    const int cols_real = cols;
    cols = pad256(cols);
    static const bool disabled = getenv("FLLM_CUDA_NO_QGEMM") != nullptr;
    if (disabled || batch < 2 || (cols_real & 31) != 0 || (reinterpret_cast<uintptr_t>(X) & 15) != 0 ||
        (reinterpret_cast<uintptr_t>(W) & 15) != 0 || (size_t)batch * cols * 2 > x16_bytes)
        return 0;
    fllm_convert_x16_pad(s, X, x16_raw, batch, cols_real, cols);
    const __half *x16 = static_cast<const __half *>(x16_raw);
    const uint8_t *w = static_cast<const uint8_t *>(W);
    const size_t x_used = ((size_t)batch * cols * 2 + 255) & ~size_t(255);
    float *part = reinterpret_cast<float *>(static_cast<char *>(x16_raw) + x_used);
    const size_t part_bytes = x16_bytes > x_used ? x16_bytes - x_used : 0;
    if (batch <= 16) return q8_launch<64, 16, 4, 1, 3, true>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
    if (batch <= 32) return q8_launch<64, 32, 4, 1, 2, true>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
    if (batch <= 64) return q8_launch<128, 64, 4, 1, 3, false>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
    auto run = [&](int i) -> int {
        switch (i) {
            case 0: {
                const double wide = 74.0 * batch / (((batch + 255) / 256) * 256), narrow = 66.0 * batch / (((batch + 127) / 128) * 128);
                if (narrow > wide) return q8_launch<128, 128, 8, 1, 3, false>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
                return q8_launch<128, 256, 8, 1, 2, false>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
            }
            case 1: return q8_launch<128, 128, 8, 1, 3, false>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
            case 2: return q8_launch<128, 128, 4, 2, 3, false>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
            case 3: return q8_launch<128, 192, 4, 2, 2, false>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
            case 4: return q8_launch<128, 256, 4, 2, 2, false>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
            case 5: return q8_launch<256, 128, 8, 2, 2, false>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
            default: return q8_launch<128, 96, 4, 1, 3, false>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
        }
    };
    return run(fllm_autotune(s, 0x8B, rows, cols, batch, 7, run));
}
