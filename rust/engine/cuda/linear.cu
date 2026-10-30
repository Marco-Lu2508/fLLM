// fLLM CUDA backend: quantized linear layers (y = W x [+ bias]) and weight conversion.
#include <cstdio>
#include <cstdlib>
#include <map>
#include <utility>

#include "common.cuh"
#include "fllm_cuda_internal.h"
#if !defined(__HIPCC__)
#include "fp8_gemm.h"
int fllm_cublaslt_gemm_try(cudaStream_t s, const void *W, int kind, const float *X, float *Y, const float *bias, int rows, int cols, int batch);
#endif

// ---------------------------------------------------------------------------------------------
// GEMV for small batches: one warp per output row, every lane decodes runs of 16 weights and
// multiplies them with up to NB activation rows. Exact FP32 arithmetic (activations stay float).
// ---------------------------------------------------------------------------------------------
template <int NB>
__device__ __forceinline__ void load16(const float *p, int e0, int cols, float *x, bool vec) {
    if (vec && e0 + 16 <= cols) {
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

template <uint32_t KIND, int NB>
__global__ void __launch_bounds__(128)
gemv_kernel(const uint8_t *__restrict__ W, const float *__restrict__ X, float *__restrict__ Y,
            const float *__restrict__ bias, int rows, int cols, int batch, int b0) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= rows) return;
    const uint8_t *rp = W + (size_t)row * kind_row_bytes(KIND, cols);
    bool vec[NB];
#pragma unroll
    for (int b = 0; b < NB; b++) vec[b] = ((reinterpret_cast<uintptr_t>(X + (size_t)(b0 + b) * cols)) & 15) == 0;
    float acc[NB];
#pragma unroll
    for (int b = 0; b < NB; b++) acc[b] = 0.f;
    const int runs = (cols + 15) / 16;
    for (int run = lane; run < runs; run += FLLM_WARP) {
        float w[16];
        decode16<(uint32_t)KIND>(rp, run, cols, w);
#pragma unroll
        for (int b = 0; b < NB; b++) {
            if (b0 + b < batch) {
                float x[16];
                load16<NB>(X + (size_t)(b0 + b) * cols, run * 16, cols, x, vec[b]);
#pragma unroll
                for (int i = 0; i < 16; i++) acc[b] = fmaf(w[i], x[i], acc[b]);
            }
        }
    }
#pragma unroll
    for (int b = 0; b < NB; b++) acc[b] = warp_sum(acc[b]);
    if (lane == 0) {
        const float bv = bias ? bias[row] : 0.f;
#pragma unroll
        for (int b = 0; b < NB; b++)
            if (b0 + b < batch) Y[(size_t)(b0 + b) * rows + row] = acc[b] + bv;
    }
}

// ---------------------------------------------------------------------------------------------
// GEMM for larger batches (prefill): 64x64 output tile per block, K step 64, FP32 FMA.
// Weights are decoded straight into shared memory, one run of 16 per thread.
// ---------------------------------------------------------------------------------------------
template <uint32_t KIND>
__global__ void __launch_bounds__(256)
gemm_kernel(const uint8_t *__restrict__ W, const float *__restrict__ X, float *__restrict__ Y,
            const float *__restrict__ bias, int rows, int cols, int batch) {
    constexpr int BM = 64, BN = 64, BK = 64;
    __shared__ float Ws[BK][BM + 1];
    __shared__ float Xs[BK][BN + 1];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
    const int ld_row = tid >> 2, ld_q = tid & 3;  // 64 rows x 4 runs of 16
    for (int k0 = 0; k0 < cols; k0 += BK) {
        {   // weights: row m0 + ld_row, run (k0 / 16 + ld_q)
            float w[16];
            const int m = m0 + ld_row;
            if (m < rows) {
                const uint8_t *rp = W + (size_t)m * kind_row_bytes(KIND, cols);
                decode16<(uint32_t)KIND>(rp, k0 / 16 + ld_q, cols, w);
            } else {
#pragma unroll
                for (int i = 0; i < 16; i++) w[i] = 0.f;
            }
#pragma unroll
            for (int i = 0; i < 16; i++) Ws[ld_q * 16 + i][ld_row] = w[i];
        }
        {   // activations: token n0 + ld_row, elements k0 + ld_q * 16 ..
            const int n = n0 + ld_row;
            const int e0 = k0 + ld_q * 16;
            float x[16];
            if (n < batch) {
                const float *p = X + (size_t)n * cols;
                load16<1>(p, e0, cols, x, ((reinterpret_cast<uintptr_t>(p)) & 15) == 0);
            } else {
#pragma unroll
                for (int i = 0; i < 16; i++) x[i] = 0.f;
            }
#pragma unroll
            for (int i = 0; i < 16; i++) Xs[ld_q * 16 + i][ld_row] = x[i];
        }
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < BK; kk++) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; i++) a[i] = Ws[kk][tx + 16 * i];
#pragma unroll
            for (int j = 0; j < 4; j++) b[j] = Xs[kk][ty + 16 * j];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int n = n0 + ty + 16 * j;
        if (n >= batch) continue;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int m = m0 + tx + 16 * i;
            if (m < rows) Y[(size_t)n * rows + m] = acc[i][j] + (bias ? bias[m] : 0.f);
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------------------------
#include "bnb.cuh"

template <int KIND>
static int launch_kind(cudaStream_t s, const uint8_t *W, const float *X, float *Y, const float *bias,
                       int rows, int cols, int batch, void *x16, size_t x16_bytes) {
    if (KIND == KIND_Q4_K_MARLIN && !fllm_exact_math() &&
        fllm_marlin_q4k_try(s, W, X, Y, bias, rows, cols, batch, x16, x16_bytes))
        return fllm_check_launch("linear");
    if (KIND == KIND_Q4_K_MARLIN) {
        return fllm_set_error("marlin failed to launch");
    }
    static const bool old_gemv = getenv("FLLM_CUDA_OLD_GEMV") != nullptr;
#if defined(__HIPCC__)
    // AMD: F16 / BF16 weights and 1..4 tokens go to the skinny GEMM after vLLM's wvSplitK (hip/hip_skinny.cu)
    if ((KIND == KIND_F16 || KIND == KIND_BF16) && batch <= 4 && !fllm_exact_math() &&
        fllm_hip_skinny_try(s, W, KIND, X, Y, bias, rows, cols, batch, x16, x16_bytes))
        return fllm_check_launch("linear");
#endif
    // single tokens and a few concurrent users on the 32-value formats: integer dot products with int8 activations
    static const bool no_mmvq_leg = getenv("FLLM_CUDA_NO_MMVQ") != nullptr || getenv("FLLM_CUDA_NO_MMVQ_LEG") != nullptr;
    if (!no_mmvq_leg && (KIND == KIND_Q4_0 || KIND == KIND_Q4_1 || KIND == KIND_Q5_0 || KIND == KIND_Q5_1 || FLLM_LUT_KIND(KIND) ||
                         KIND == KIND_Q8_0) &&
        batch <= 4 && !fllm_exact_math() && fllm_mmvq_leg_try(s, W, KIND, X, Y, bias, rows, cols, batch, x16, x16_bytes))
        return fllm_check_launch("linear");
    if (kind_is_iq_or_kextra(KIND) && KIND != KIND_IQ1_M && batch <= 4 && !fllm_exact_math() &&
        fllm_mmvq_x_try(s, W, KIND, X, Y, bias, rows, cols, batch, x16, x16_bytes))
        return fllm_check_launch("linear");
    static const bool mmvq1 = getenv("FLLM_CUDA_NO_MMVQ1") == nullptr && getenv("FLLM_CUDA_NO_MMVQ") == nullptr;
    if (mmvq1 && (KIND == KIND_Q4_K || KIND == KIND_Q5_K || KIND == KIND_Q6_K) && batch == 1 && !fllm_exact_math() &&
        fllm_mmvq_try(s, W, KIND, X, Y, bias, rows, cols, batch, x16, x16_bytes))
        return fllm_check_launch("linear");
    // narrow batches (decode, a few concurrent users) are bandwidth-bound: decode the weights once per pass
    static const int gemv_max_batch = getenv("FLLM_CUDA_GEMV_MAXB") ? atoi(getenv("FLLM_CUDA_GEMV_MAXB")) : 1;
    if (!old_gemv && (KIND == KIND_Q4_K || KIND == KIND_Q5_K || KIND == KIND_Q6_K || KIND == KIND_Q8_0 || KIND == KIND_Q4_0 || FLLM_LUT_KIND(KIND) || KIND == KIND_Q4_1 ||
         KIND == KIND_Q5_0 || KIND == KIND_Q5_1) &&
        batch <= gemv_max_batch &&
        fllm_gemv2_try(s, W, KIND, X, Y, bias, rows, cols, batch))
        return fllm_check_launch("linear");
    // 2..4 tokens: integer dot products (int8 activations) are the fastest way to stream the weights
    static const bool no_mmvq = getenv("FLLM_CUDA_NO_MMVQ") != nullptr;
    if (!no_mmvq && (KIND == KIND_Q4_K || KIND == KIND_Q5_K || KIND == KIND_Q6_K) && batch >= 2 && batch <= fllm_mmvq_maxb() && !fllm_exact_math() &&
        fllm_mmvq_try(s, W, KIND, X, Y, bias, rows, cols, batch, x16, x16_bytes))
        return fllm_check_launch("linear");
#if !defined(__HIPCC__)
    // dense 16-bit weights, prompt batches: cuBLASLt loaded at run time (cublaslt_gemm.cu; Hopper / Blackwell kernels), else the built-in GEMM below
    if ((KIND == KIND_F16 || KIND == KIND_BF16) && batch >= 16) {
        const int rc = fllm_cublaslt_gemm_try(s, W, KIND, X, Y, bias, rows, cols, batch);
        if (rc >= 0) return rc;
    }
    // Q8_0 prompts on the int8 tensor cores (int8_gemm.cu, FLLM_INT8_GEMM=1)
    if (KIND == KIND_Q8_0 && batch >= 16) {
        const int rc = fllm_int8_q8_try(s, W, X, Y, bias, rows, cols, batch);
        if (rc >= 0) return rc;
    }
#endif
    if (KIND == KIND_Q4_K && batch >= 2 && !fllm_exact_math() &&
        fllm_qgemm4k_try(s, W, X, Y, bias, rows, cols, batch, x16, x16_bytes))
        return fllm_check_launch("linear");
    if ((KIND == KIND_Q4_0 || FLLM_LUT_KIND(KIND) || KIND == KIND_Q4_1 || KIND == KIND_Q5_0 || KIND == KIND_Q5_1) && batch >= 2 && !fllm_exact_math() &&
        (cols & 31) == 0 && fllm_qgemm_leg_try(s, W, KIND, X, Y, bias, rows, cols, batch, x16, x16_bytes))
        return fllm_check_launch("linear");
    if (KIND == KIND_Q8_0 && batch >= 2 && !fllm_exact_math() && (cols & 31) == 0 &&
        fllm_qgemm8_try(s, W, X, Y, bias, rows, cols, batch, x16, x16_bytes))
        return fllm_check_launch("linear");
    if (KIND == KIND_Q5_K && batch >= 2 && !fllm_exact_math() &&
        fllm_qgemm5k_try(s, W, X, Y, bias, rows, cols, batch, x16, x16_bytes))
        return fllm_check_launch("linear");
    if (KIND == KIND_Q6_K && batch >= 2 && !fllm_exact_math() &&
        fllm_qgemm6k_try(s, W, X, Y, bias, rows, cols, batch, x16, x16_bytes))
        return fllm_check_launch("linear");
    if (!old_gemv && (KIND == KIND_Q4_K || KIND == KIND_Q6_K) && fllm_gemv2_try(s, W, KIND, X, Y, bias, rows, cols, batch))
        return fllm_check_launch("linear");
#if defined(__HIPCC__)
    // AMD matrix-core GEMM with the activations converted to FP16 once (hip/hip_gemm.cu)
    if (batch >= 16 && !fllm_exact_math() && fllm_hip_gemm_x16_try(s, W, KIND, X, Y, bias, rows, cols, batch, x16, x16_bytes)) return fllm_check_launch("linear");
#endif
    if (batch >= 16 && !fllm_exact_math() && fllm_tc_gemm_try(s, W, KIND, X, Y, bias, rows, cols, batch)) return fllm_check_launch("linear");
    if (batch <= 8) {
        const dim3 block(128), grid((rows + 3) / 4);
        for (int b0 = 0; b0 < batch; b0 += 4) {
            const int nb = batch - b0 < 4 ? batch - b0 : 4;
            switch (nb) {
                case 1: gemv_kernel<KIND, 1><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, b0); break;
                case 2: gemv_kernel<KIND, 2><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, b0); break;
                case 3: gemv_kernel<KIND, 3><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, b0); break;
                default: gemv_kernel<KIND, 4><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, b0); break;
            }
        }
    } else {
        const dim3 grid((rows + 63) / 64, (batch + 63) / 64), block(256);
        gemm_kernel<KIND><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch);
    }
    return fllm_check_launch("linear");
}

// bitsandbytes NF4 / FP4 / int8 (ids above INT_MAX, so not a launch_kind instantiation): the decode GEMV of bnb.cuh for up to eight tokens, the
// tensor-core GEMM of tc_gemm.cu (blocks decoded into FP16 tiles) from sixteen, else the generic FP32 kernels on decode16
template <uint32_t KIND>
static int launch_bnb(cudaStream_t s, const uint8_t *W, const float *X, float *Y, const float *bias, int rows, int cols, int batch) {
    if (batch <= 8 && fllm_bnb_gemv_try(s, W, KIND, X, Y, bias, rows, cols, batch)) return fllm_check_launch("linear");
    if (batch >= 16 && !fllm_exact_math() && fllm_tc_gemm_try(s, W, (int)KIND, X, Y, bias, rows, cols, batch)) return fllm_check_launch("linear");
    if (batch <= 8) {
        const dim3 block(128), grid((rows + 3) / 4);
        for (int b0 = 0; b0 < batch; b0 += 4) {
            const int nb = batch - b0 < 4 ? batch - b0 : 4;
            switch (nb) {
                case 1: gemv_kernel<KIND, 1><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, b0); break;
                case 2: gemv_kernel<KIND, 2><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, b0); break;
                case 3: gemv_kernel<KIND, 3><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, b0); break;
                default: gemv_kernel<KIND, 4><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, b0); break;
            }
        }
    } else {
        const dim3 grid((rows + 63) / 64, (batch + 63) / 64), block(256);
        gemm_kernel<KIND><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch);
    }
    return fllm_check_launch("linear");
}

static int g_exact = -1;  // -1: not read yet
bool fllm_exact_math() {
    if (g_exact < 0) g_exact = getenv("FLLM_CUDA_EXACT") != nullptr ? 1 : 0;
    return g_exact != 0;
}
// selftest-gpu runs its checks twice: with the exact FP32 kernels and with the fast ones
extern "C" int fllm_cuda_set_exact(int on) {
    g_exact = on ? 1 : 0;
    return 0;
}

static int fllm_linear_launch_impl(cudaStream_t s, const void *W, int kind, const float *X, float *Y, const float *bias,
                                   int rows, int cols, int batch, void *x16, size_t x16_bytes);

// FLLM_CUDA_PROFILE=1: GPU time of every weight projection (events around each launch, so the stream is synchronised after each of them: the
// run is slower than normal), summed per weight type and batch class and printed to stderr every 2000 projections
int fllm_linear_launch(cudaStream_t s, const void *W, int kind, const float *X, float *Y, const float *bias,
                       int rows, int cols, int batch, void *x16, size_t x16_bytes) {
    static const bool profile = getenv("FLLM_CUDA_PROFILE") != nullptr && getenv("FLLM_CUDA_PROFILE")[0] != '0';
    if (!profile) return fllm_linear_launch_impl(s, W, kind, X, Y, bias, rows, cols, batch, x16, x16_bytes);
    struct Acc { double ms = 0; long calls = 0; double macs = 0; };
    static std::map<std::pair<int, int>, Acc> table;
    static long total_calls = 0;
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a, s);
    const int rc = fllm_linear_launch_impl(s, W, kind, X, Y, bias, rows, cols, batch, x16, x16_bytes);
    cudaEventRecord(b, s);
    cudaEventSynchronize(b);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    const int cls = batch == 1 ? 1 : batch <= 4 ? 4 : batch <= 8 ? 8 : batch <= 16 ? 16 : batch <= 32 ? 32 : batch <= 64 ? 64 : batch <= 256 ? 256 : 1024;
    Acc &acc = table[{kind, cls}];
    acc.ms += ms;
    acc.calls++;
    acc.macs += (double)rows * cols * batch;
    if (++total_calls % 2000 == 0) {
        fprintf(stderr, "[fllm-gpu-profile] projections: weight type, batch up to, calls, ms per call, GMAC/s\n");
        for (const auto &e : table)
            fprintf(stderr, "[fllm-gpu-profile] kind %d batch<=%d calls %ld %.3f ms %.1f GMAC/s total %.0f ms\n", e.first.first, e.first.second, e.second.calls,
                    e.second.ms / e.second.calls, e.second.macs / (e.second.ms * 1e6), e.second.ms);
    }
    return rc;
}

static int fllm_linear_launch_impl(cudaStream_t s, const void *W, int kind, const float *X, float *Y, const float *bias,
                                   int rows, int cols, int batch, void *x16, size_t x16_bytes) {
    const uint8_t *w = static_cast<const uint8_t *>(W);
    if (fllm_kind_is_mlx(kind)) return fllm_mlx_launch(s, W, kind, X, Y, bias, rows, cols, batch);
    switch ((uint32_t)kind) {
        case KIND_F32: return launch_kind<KIND_F32>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_F16: return launch_kind<KIND_F16>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_BF16: return launch_kind<KIND_BF16>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_Q4_0: return launch_kind<KIND_Q4_0>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_IQ4_NL: return launch_kind<KIND_IQ4_NL>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_MXFP4: return launch_kind<KIND_MXFP4>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_Q4_1: return launch_kind<KIND_Q4_1>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_Q5_0: return launch_kind<KIND_Q5_0>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_Q5_1: return launch_kind<KIND_Q5_1>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_Q8_0: return launch_kind<KIND_Q8_0>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_Q4_K: return launch_kind<KIND_Q4_K>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_Q4_K_MARLIN: return launch_kind<KIND_Q4_K_MARLIN>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_Q5_K: return launch_kind<KIND_Q5_K>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_Q6_K: return launch_kind<KIND_Q6_K>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_Q2_K: return launch_kind<KIND_Q2_K>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_Q3_K: return launch_kind<KIND_Q3_K>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_IQ2_XXS: return launch_kind<KIND_IQ2_XXS>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_IQ2_XS: return launch_kind<KIND_IQ2_XS>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_IQ3_XXS: return launch_kind<KIND_IQ3_XXS>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_IQ1_S: return launch_kind<KIND_IQ1_S>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_IQ3_S: return launch_kind<KIND_IQ3_S>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_IQ2_S: return launch_kind<KIND_IQ2_S>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_IQ1_M: return launch_kind<KIND_IQ1_M>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_NF4B64: return launch_bnb<KIND_NF4B64>(s, w, X, Y, bias, rows, cols, batch);
        case KIND_NF4B128: return launch_bnb<KIND_NF4B128>(s, w, X, Y, bias, rows, cols, batch);
        case KIND_FP4B64: return launch_bnb<KIND_FP4B64>(s, w, X, Y, bias, rows, cols, batch);
        case KIND_FP4B128: return launch_bnb<KIND_FP4B128>(s, w, X, Y, bias, rows, cols, batch);
        case KIND_INT8R: return launch_bnb<KIND_INT8R>(s, w, X, Y, bias, rows, cols, batch);
        default: return fllm_set_error("linear: unsupported weight type %d", kind);
    }
}

// Embedding lookup: out[t][c] = weight[ids[t]][c]
__global__ void embedding_kernel(const uint8_t *W, int kind, const int64_t *ids, float *out, int cols, int n) {
    const size_t total = (size_t)n * cols;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const size_t t = i / cols, c = i % cols;
        out[i] = weight_at(W, kind, (size_t)ids[t] * cols + c, cols);
    }
}

int fllm_embedding_launch(cudaStream_t s, const void *W, int kind, const int64_t *ids_dev, float *out, int cols, int n) {
    if (!kind_supported(kind)) return fllm_set_error("embedding: unsupported weight type %d", kind);
    const size_t total = (size_t)n * cols;
    const int block = 256;
    const unsigned grid = (unsigned)((total + block - 1) / block < 65535 ? (total + block - 1) / block : 65535);
    embedding_kernel<<<grid, block, 0, s>>>(static_cast<const uint8_t *>(W), kind, ids_dev, out, cols, n);
    return fllm_check_launch("embedding");
}
