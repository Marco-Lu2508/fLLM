// fLLM CUDA backend: tensor-core GEMM for prompt processing and larger decode batches.
//
// Y[token][row] = sum_k X[token][k] * W[row][k]  (+ bias[row])
//
// Weights stay quantized in global memory; every block decodes its weight tile straight into
// shared memory as FP16, converts the activation tile from FP32 to FP16 and runs
// `mma.sync.m16n8k16` (FP16 inputs, FP32 accumulation). The next tile is fetched and decoded while
// the current one is multiplied. Rounding weights and activations to FP16 costs about 1e-3
// relative error, far below the quantization error of the weights themselves; the strict FP32
// kernels in linear.cu remain in use for small batches.
//
// The tile shape is a compile-time choice (TC_* macros) so the same kernel serves the large
// prompt tiles and the narrow tiles of small decode batches; bench/bench_tc.cu sweeps them.
#include <cstdlib>
#include <string>

#include "common.cuh"
#include "fllm_cuda_internal.h"
#include "mma_util.cuh"

#if defined(__HIPCC__)
// AMD: 64x64 output tiles, 256 threads (8 waves of 32 on RDNA, 4 of 64 on CDNA), rocWMMA 16x16x16 fragments
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__GFX11__) || defined(__GFX12__) || defined(__gfx908__) || defined(__gfx90a__) ||     defined(__gfx942__) || defined(__gfx950__)
#include <rocwmma/rocwmma.hpp>
#endif
#define TC_BM 64
#define TC_BN 64
#define TC_BK 64
#define TC_WM 2
#define TC_WN 4
#endif
#ifndef TC_BM
#define TC_BM 128  // weight rows per block
#endif
#ifndef TC_BN
#define TC_BN 128  // tokens per block
#endif
#ifndef TC_BK
#define TC_BK 64  // k per step (multiple of 16)
#endif
#ifndef TC_WM
#define TC_WM 2  // warps along the weight rows
#endif
#ifndef TC_WN
#define TC_WN 4  // warps along the tokens
#endif
#ifndef TC_MINB
#define TC_MINB 1  // blocks per SM the register budget is sized for
#endif

constexpr int TC_LD = TC_BK + 8;
constexpr int TC_NT = TC_WM * TC_WN * 32;
constexpr int TC_MI = TC_BM / TC_WM / 16;  // 16-row fragments per warp
constexpr int TC_NI = TC_BN / TC_WN / 8;   // 8-token fragments per warp
constexpr int TC_APT = TC_BM * TC_BK / 16 / TC_NT;  // 16-element weight runs per thread and step
constexpr int TC_BPT = TC_BN * TC_BK / 16 / TC_NT;  // 16-element activation runs per thread and step
constexpr int TC_RPK = TC_BK / 16;                  // runs per tile row
static_assert(TC_APT >= 1 && TC_BPT >= 1, "tile too small for the thread count");
static_assert(TC_NI % 2 == 0 && TC_BK % 16 == 0 && TC_MI >= 1, "tile shape");
constexpr size_t TC_SMEM = (size_t)2 * (TC_BM + TC_BN) * TC_LD * sizeof(__half);

#if !defined(__HIPCC__)
__device__ __forceinline__ void ldmatrix_x4(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, const void *p) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a));
}

__device__ __forceinline__ void mma_16816(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) { mma16816(c, a, b0, b1); }
#endif

__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t *>(&h);
}

__device__ __forceinline__ void load16_x(const float *p, int e0, int cols, float *x) {
    if (e0 + 16 <= cols) {
        const float4 *q = reinterpret_cast<const float4 *>(p + e0);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float4 v = __ldg(q + i);
            x[4 * i] = v.x; x[4 * i + 1] = v.y; x[4 * i + 2] = v.z; x[4 * i + 3] = v.w;
        }
    } else {
#pragma unroll
        for (int i = 0; i < 16; i++) x[i] = e0 + i < cols ? __ldg(p + e0 + i) : 0.f;
    }
}

__device__ __forceinline__ void store16(__half *dst, const float *v) {
    uint4 a, b;
    a.x = pack_half2(v[0], v[1]);   a.y = pack_half2(v[2], v[3]);
    a.z = pack_half2(v[4], v[5]);   a.w = pack_half2(v[6], v[7]);
    b.x = pack_half2(v[8], v[9]);   b.y = pack_half2(v[10], v[11]);
    b.z = pack_half2(v[12], v[13]); b.w = pack_half2(v[14], v[15]);
    uint4 *d = reinterpret_cast<uint4 *>(dst);
    d[0] = a;
    d[1] = b;
}

template <uint32_t KIND>
__global__ void __launch_bounds__(TC_NT, TC_MINB)
tc_gemm_kernel(const uint8_t *__restrict__ W, const float *__restrict__ X, float *__restrict__ Y,
               const float *__restrict__ bias, int rows, int cols, int batch) {
#if FLLM_TC
    extern __shared__ __align__(16) unsigned char tc_smem[];
    __half *As = reinterpret_cast<__half *>(tc_smem);                       // [2][TC_BM][TC_LD]
    __half *Bs = As + (size_t)2 * TC_BM * TC_LD;                            // [2][TC_BN][TC_LD]
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp % TC_WM, wn = warp / TC_WM;
    const int m0 = blockIdx.x * TC_BM, n0 = blockIdx.y * TC_BN;
    const size_t row_bytes = kind_row_bytes(KIND, cols);
    const int ktiles = (cols + TC_BK - 1) / TC_BK;

    float acc[TC_MI][TC_NI][4];
#pragma unroll
    for (int a = 0; a < TC_MI; a++)
#pragma unroll
        for (int b = 0; b < TC_NI; b++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[a][b][c] = 0.f;

    float wv[TC_APT][16], xv[TC_BPT][16];
    auto fetch = [&](int kt) {
#pragma unroll
        for (int i = 0; i < TC_APT; i++) {
            const int run = tid + i * TC_NT;
            const int m = m0 + run / TC_RPK;
            const int k = kt * TC_BK + (run % TC_RPK) * 16;
            if (m < rows) {
                decode16<(uint32_t)KIND>(W + (size_t)m * row_bytes, k >> 4, cols, wv[i]);
            } else {
#pragma unroll
                for (int j = 0; j < 16; j++) wv[i][j] = 0.f;
            }
        }
#pragma unroll
        for (int i = 0; i < TC_BPT; i++) {
            const int run = tid + i * TC_NT;
            const int n = n0 + run / TC_RPK;
            const int k = kt * TC_BK + (run % TC_RPK) * 16;
            if (n < batch) {
                load16_x(X + (size_t)n * cols, k, cols, xv[i]);
            } else {
#pragma unroll
                for (int j = 0; j < 16; j++) xv[i][j] = 0.f;
            }
        }
    };
    auto stash = [&](int buf) {
#pragma unroll
        for (int i = 0; i < TC_APT; i++) {
            const int run = tid + i * TC_NT;
            store16(As + ((size_t)buf * TC_BM + run / TC_RPK) * TC_LD + (run % TC_RPK) * 16, wv[i]);
        }
#pragma unroll
        for (int i = 0; i < TC_BPT; i++) {
            const int run = tid + i * TC_NT;
            store16(Bs + ((size_t)buf * TC_BN + run / TC_RPK) * TC_LD + (run % TC_RPK) * 16, xv[i]);
        }
    };

    fetch(0);
    stash(0);
    __syncthreads();
    for (int kt = 0; kt < ktiles; kt++) {
        const int buf = kt & 1;
        if (kt + 1 < ktiles) fetch(kt + 1);
        const __half *a_base = As + (size_t)buf * TC_BM * TC_LD;
        const __half *b_base = Bs + (size_t)buf * TC_BN * TC_LD;
#pragma unroll
        for (int kk = 0; kk < TC_BK; kk += 16) {
            uint32_t af[TC_MI][4], bf[TC_NI][2];
#pragma unroll
            for (int mi = 0; mi < TC_MI; mi++)
                ldmatrix_x4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                            a_base + (size_t)(wm * (TC_BM / TC_WM) + mi * 16 + (lane & 15)) * TC_LD + kk + (lane >> 4) * 8);
#pragma unroll
            for (int np = 0; np < TC_NI / 2; np++)
                ldmatrix_x4(bf[np * 2][0], bf[np * 2][1], bf[np * 2 + 1][0], bf[np * 2 + 1][1],
                            b_base + (size_t)(wn * (TC_BN / TC_WN) + np * 16 + (lane & 7) + ((lane >> 4) << 3)) * TC_LD +
                                kk + ((lane >> 3) & 1) * 8);
#pragma unroll
            for (int mi = 0; mi < TC_MI; mi++)
#pragma unroll
                for (int ni = 0; ni < TC_NI; ni++) mma_16816(acc[mi][ni], af[mi], bf[ni][0], bf[ni][1]);
        }
        if (kt + 1 < ktiles) stash(buf ^ 1);
        __syncthreads();
    }

    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int mi = 0; mi < TC_MI; mi++)
#pragma unroll
        for (int ni = 0; ni < TC_NI; ni++)
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int m = m0 + wm * (TC_BM / TC_WM) + mi * 16 + g + (c >> 1) * 8;
                const int n = n0 + wn * (TC_BN / TC_WN) + ni * 8 + 2 * t + (c & 1);
                if (m < rows && n < batch) Y[(size_t)n * rows + m] = acc[mi][ni][c] + (bias ? bias[m] : 0.f);
            }
#endif
}

#if defined(__HIPCC__)
// AMD matrix cores through rocWMMA (MIT, https://github.com/ROCm/rocWMMA): the same decode-into-shared-memory scheme as above;
// the 16x16 output tiles of the block are dealt to the waves, A = activations (token x k), B = decoded weights (k x row).
constexpr int TC_EPI_BYTES = 8 * 256 * sizeof(float);  // epilogue staging for up to 8 waves
template <uint32_t KIND>
__global__ void __launch_bounds__(TC_NT)
tc_gemm_wmma_kernel(const uint8_t *__restrict__ W, const float *__restrict__ X, float *__restrict__ Y,
                    const float *__restrict__ bias, int rows, int cols, int batch) {
#if !defined(__HIP_DEVICE_COMPILE__) || defined(__GFX11__) || defined(__GFX12__) || defined(__gfx908__) || defined(__gfx90a__) ||     defined(__gfx942__) || defined(__gfx950__)
    using namespace rocwmma;
    constexpr int WAVE = Constants::AMDGCN_WAVE_SIZE;
    constexpr int NWAVES = TC_NT / WAVE;
    constexpr int TM = TC_BM / 16, TILES = TM * (TC_BN / 16);
    constexpr int PER_WAVE = (TILES + NWAVES - 1) / NWAVES;
    extern __shared__ __align__(16) unsigned char tc_smem[];
    __half *As = reinterpret_cast<__half *>(tc_smem);                      // [2][TC_BM][TC_LD]
    __half *Bs = As + (size_t)2 * TC_BM * TC_LD;                           // [2][TC_BN][TC_LD]
    float *epi = reinterpret_cast<float *>(Bs + (size_t)2 * TC_BN * TC_LD);  // [NWAVES][16][16]
    const int tid = threadIdx.x, wave = tid / WAVE, lane = tid % WAVE;
    const int m0 = blockIdx.x * TC_BM, n0 = blockIdx.y * TC_BN;
    const size_t row_bytes = kind_row_bytes(KIND, cols);
    const int ktiles = (cols + TC_BK - 1) / TC_BK;

    fragment<accumulator, 16, 16, 16, float> acc[PER_WAVE];
#pragma unroll
    for (int i = 0; i < PER_WAVE; i++) fill_fragment(acc[i], 0.f);

    float wv[TC_APT][16], xv[TC_BPT][16];
    auto fetch = [&](int kt) {
#pragma unroll
        for (int i = 0; i < TC_APT; i++) {
            const int run = tid + i * TC_NT;
            const int m = m0 + run / TC_RPK;
            const int k = kt * TC_BK + (run % TC_RPK) * 16;
            if (m < rows) {
                decode16<(uint32_t)KIND>(W + (size_t)m * row_bytes, k >> 4, cols, wv[i]);
            } else {
#pragma unroll
                for (int j = 0; j < 16; j++) wv[i][j] = 0.f;
            }
        }
#pragma unroll
        for (int i = 0; i < TC_BPT; i++) {
            const int run = tid + i * TC_NT;
            const int n = n0 + run / TC_RPK;
            const int k = kt * TC_BK + (run % TC_RPK) * 16;
            if (n < batch) {
                load16_x(X + (size_t)n * cols, k, cols, xv[i]);
            } else {
#pragma unroll
                for (int j = 0; j < 16; j++) xv[i][j] = 0.f;
            }
        }
    };
    auto stash = [&](int buf) {
#pragma unroll
        for (int i = 0; i < TC_APT; i++) {
            const int run = tid + i * TC_NT;
            store16(As + ((size_t)buf * TC_BM + run / TC_RPK) * TC_LD + (run % TC_RPK) * 16, wv[i]);
        }
#pragma unroll
        for (int i = 0; i < TC_BPT; i++) {
            const int run = tid + i * TC_NT;
            store16(Bs + ((size_t)buf * TC_BN + run / TC_RPK) * TC_LD + (run % TC_RPK) * 16, xv[i]);
        }
    };

    fetch(0);
    stash(0);
    __syncthreads();
    for (int kt = 0; kt < ktiles; kt++) {
        const int buf = kt & 1;
        if (kt + 1 < ktiles) fetch(kt + 1);
        const __half *a_base = As + (size_t)buf * TC_BM * TC_LD;
        const __half *b_base = Bs + (size_t)buf * TC_BN * TC_LD;
#pragma unroll
        for (int kk = 0; kk < TC_BK; kk += 16) {
#pragma unroll
            for (int i = 0; i < PER_WAVE; i++) {
                const int t = wave + i * NWAVES;
                if (t < TILES) {
                    const int tm = t % TM, tn = t / TM;
                    fragment<matrix_a, 16, 16, 16, __half, row_major> fa;
                    fragment<matrix_b, 16, 16, 16, __half, col_major> fb;
                    load_matrix_sync(fa, b_base + (size_t)tn * 16 * TC_LD + kk, TC_LD);
                    load_matrix_sync(fb, a_base + (size_t)tm * 16 * TC_LD + kk, TC_LD);
                    mma_sync(acc[i], fa, fb, acc[i]);
                }
            }
        }
        if (kt + 1 < ktiles) stash(buf ^ 1);
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < PER_WAVE; i++) {
        const int t = wave + i * NWAVES;
        if (t < TILES) store_matrix_sync(epi + wave * 256, acc[i], 16, mem_row_major);
        __syncthreads();
        if (t < TILES) {
            const int tm = t % TM, tn = t / TM;
            for (int e = lane; e < 256; e += WAVE) {
                const int n = n0 + tn * 16 + e / 16, m = m0 + tm * 16 + e % 16;
                if (m < rows && n < batch) Y[(size_t)n * rows + m] = epi[wave * 256 + e] + (bias ? bias[m] : 0.f);
            }
        }
        __syncthreads();
    }
#endif
}
#endif

template <uint32_t KIND>
static int run(cudaStream_t s, const uint8_t *W, const float *X, float *Y, const float *bias, int rows, int cols,
               int batch) {
    if (TC_SMEM > (size_t)fllm_smem_limit()) return 0;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(tc_gemm_kernel<KIND>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)TC_SMEM);
        configured = true;
    }
    const dim3 grid((rows + TC_BM - 1) / TC_BM, (batch + TC_BN - 1) / TC_BN), block(TC_NT);
#if defined(__HIPCC__)
    tc_gemm_wmma_kernel<KIND><<<grid, block, TC_SMEM + TC_EPI_BYTES, s>>>(W, X, Y, bias, rows, cols, batch);
#else
    tc_gemm_kernel<KIND><<<grid, block, TC_SMEM, s>>>(W, X, Y, bias, rows, cols, batch);
#endif
    return 1;
}

// Returns 1 when the tensor-core path handled the call, 0 when the caller should use the exact
// FP32 kernels (small batches, unaligned activations, FLLM_CUDA_NO_TC set).
int fllm_tc_gemm_try(cudaStream_t s, const void *W, int kind, const float *X, float *Y, const float *bias, int rows,
                     int cols, int batch) {
#if defined(__HIPCC__)
    {
        static const bool matrix_cores = [] {
            int dev = 0;
            hipDeviceProp_t prop;
            if (hipGetDevice(&dev) != hipSuccess || hipGetDeviceProperties(&prop, dev) != hipSuccess) return false;
            const std::string arch = prop.gcnArchName;
            return arch.rfind("gfx11", 0) == 0 || arch.rfind("gfx12", 0) == 0 || arch.rfind("gfx908", 0) == 0 ||
                   arch.rfind("gfx90a", 0) == 0 || arch.rfind("gfx94", 0) == 0 || arch.rfind("gfx95", 0) == 0;
        }();
        if (!matrix_cores) return 0;
    }
#else
    if (fllm_cc() < 75) return 0;
#endif
    static const bool disabled = getenv("FLLM_CUDA_NO_TC") != nullptr;
    if (disabled || batch < 16 || (cols & 15) != 0 || (reinterpret_cast<uintptr_t>(X) & 15) != 0) return 0;
#if defined(__HIPCC__)
    // matrix-core GEMM written for AMD (hip/hip_gemm.cu: MFMA on CDNA, WMMA on RDNA3 / RDNA4); the rocWMMA kernel below is the fallback
    if (fllm_hip_gemm_try(s, W, kind, X, Y, bias, rows, cols, batch)) return 1;
#endif
    const uint8_t *w = static_cast<const uint8_t *>(W);
    switch ((uint32_t)kind) {
        case KIND_F32: return run<KIND_F32>(s, w, X, Y, bias, rows, cols, batch);
        case KIND_F16: return run<KIND_F16>(s, w, X, Y, bias, rows, cols, batch);
        case KIND_BF16: return run<KIND_BF16>(s, w, X, Y, bias, rows, cols, batch);
        case KIND_Q4_0: return run<KIND_Q4_0>(s, w, X, Y, bias, rows, cols, batch);
        case KIND_IQ4_NL: return run<KIND_IQ4_NL>(s, w, X, Y, bias, rows, cols, batch);
        case KIND_MXFP4: return run<KIND_MXFP4>(s, w, X, Y, bias, rows, cols, batch);
        case KIND_Q4_1: return run<KIND_Q4_1>(s, w, X, Y, bias, rows, cols, batch);
        case KIND_Q5_0: return run<KIND_Q5_0>(s, w, X, Y, bias, rows, cols, batch);
        case KIND_Q5_1: return run<KIND_Q5_1>(s, w, X, Y, bias, rows, cols, batch);
        case KIND_Q8_0: return run<KIND_Q8_0>(s, w, X, Y, bias, rows, cols, batch);
        case KIND_Q4_K: return run<KIND_Q4_K>(s, w, X, Y, bias, rows, cols, batch);
        case KIND_Q5_K: return run<KIND_Q5_K>(s, w, X, Y, bias, rows, cols, batch);
        case KIND_Q6_K: return run<KIND_Q6_K>(s, w, X, Y, bias, rows, cols, batch);
        case KIND_Q2_K: return run<KIND_Q2_K>(s, w, X, Y, bias, rows, cols, batch);
        case KIND_Q3_K: return run<KIND_Q3_K>(s, w, X, Y, bias, rows, cols, batch);
        case KIND_IQ2_XXS: return run<KIND_IQ2_XXS>(s, w, X, Y, bias, rows, cols, batch);
        case KIND_IQ2_XS: return run<KIND_IQ2_XS>(s, w, X, Y, bias, rows, cols, batch);
        case KIND_IQ3_XXS: return run<KIND_IQ3_XXS>(s, w, X, Y, bias, rows, cols, batch);
        case KIND_IQ1_S: return run<KIND_IQ1_S>(s, w, X, Y, bias, rows, cols, batch);
        case KIND_IQ3_S: return run<KIND_IQ3_S>(s, w, X, Y, bias, rows, cols, batch);
        case KIND_IQ2_S: return run<KIND_IQ2_S>(s, w, X, Y, bias, rows, cols, batch);
        case KIND_IQ1_M: return run<KIND_IQ1_M>(s, w, X, Y, bias, rows, cols, batch);
        case KIND_NF4B64: return run<KIND_NF4B64>(s, w, X, Y, bias, rows, cols, batch);
        case KIND_NF4B128: return run<KIND_NF4B128>(s, w, X, Y, bias, rows, cols, batch);
        case KIND_FP4B64: return run<KIND_FP4B64>(s, w, X, Y, bias, rows, cols, batch);
        case KIND_FP4B128: return run<KIND_FP4B128>(s, w, X, Y, bias, rows, cols, batch);
        case KIND_INT8R: return run<KIND_INT8R>(s, w, X, Y, bias, rows, cols, batch);
        default: return 0;
    }
}

// Grouped execution of up to seven projections in a layer
extern "C" int fllm_cuda_tc_grouped_gemm(cudaStream_t s, const void *const *W, const int *kinds, const float *const *X,
                                         float *const *Y, const float *const *bias, const int *rows, const int *cols,
                                         int count, int batch) {
    if (count <= 0 || !W || !X || !Y || !kinds || !rows || !cols) return -1;
    for (int i = 0; i < count; i++) {
        if (!W[i]) continue;
        const float *b = bias ? bias[i] : nullptr;
        if (!fllm_tc_gemm_try(s, W[i], kinds[i], X[i], Y[i], b, rows[i], cols[i], batch)) {
            return -1;
        }
    }
    return 0;
}
