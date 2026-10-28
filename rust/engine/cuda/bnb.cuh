// fLLM CUDA backend: decode GEMV for bitsandbytes weights (NF4, FP4, LLM.int8()), included by linear.cu after load16().
//
// The weights stay in the packed blocks of fllm_gguf_probe::bnb ([f32 scale][codes], 64 or 128 values per block, see common.cuh); the
// activations stay FP32, so the arithmetic is the CPU reference's: sum(code_book[nibble] * x) * absmax per block.
//
// 4-bit: one warp per output row, every lane takes runs of 16 weights (8 code bytes = two aligned 32-bit loads). The 16-entry code book
// is looked up through a 256-entry table of float2 in shared memory indexed by the whole code byte (both nibbles at once: one LDS.64 per
// two weights, no nibble extraction); the block scale multiplies the partial sum of its run once. int8: four 32-bit loads per run, the
// bytes are sign-extended and converted. Up to four tokens share one pass over the weights (like gemv_kernel in linear.cu).
// Prompts (batch >= 16) use the tensor-core GEMM of tc_gemm.cu, which decodes the same blocks (decode16 in common.cuh) into FP16 tiles.
#pragma once
#include "common.cuh"

#if !defined(__HIPCC__)

// the table is filled by all threads of the block, so the barrier must be reached by all of them: no thread leaves before it
template <bool FP4>
__device__ __forceinline__ void bnb_fill_pairs(float2 *pairs) {
    for (int e = threadIdx.x; e < 256; e += blockDim.x) pairs[e] = make_float2(bnb_code<FP4>((uint32_t)e >> 4), bnb_code<FP4>((uint32_t)e & 15u));
}

template <uint32_t KIND, int NB>
__global__ void __launch_bounds__(128)
bnb4_gemv_kernel(const uint8_t *__restrict__ W, const float *__restrict__ X, float *__restrict__ Y,
                 const float *__restrict__ bias, int rows, int cols, int batch, int b0) {
    constexpr int be = kind_block_elems(KIND), bb = be / 2 + 4;
    constexpr bool fp4 = KIND == KIND_FP4B64 || KIND == KIND_FP4B128;
    __shared__ float2 pairs[256];
    bnb_fill_pairs<fp4>(pairs);
    __syncthreads();
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= rows) return;  // after the barrier; the warp-wide reductions below are only reached by whole warps
    const uint8_t *rp = W + (size_t)row * (size_t)(cols / be) * bb;
    bool vec[NB];
#pragma unroll
    for (int b = 0; b < NB; b++) vec[b] = ((reinterpret_cast<uintptr_t>(X + (size_t)(b0 + b) * cols)) & 15) == 0;
    float acc[NB];
#pragma unroll
    for (int b = 0; b < NB; b++) acc[b] = 0.f;
    const int runs = cols / 16;
    for (int run = lane; run < runs; run += FLLM_WARP) {
        const int e0 = run * 16;
        const uint8_t *blk = rp + (size_t)(e0 / be) * bb;
        const float scale = __ldg(reinterpret_cast<const float *>(blk));
        const uint32_t *q = reinterpret_cast<const uint32_t *>(blk + 4 + (e0 % be) / 2);
        const uint32_t words[2] = {__ldg(q), __ldg(q + 1)};
        float w[16];
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const float2 p = pairs[(words[j >> 2] >> (8 * (j & 3))) & 0xffu];
            w[2 * j] = p.x;
            w[2 * j + 1] = p.y;
        }
#pragma unroll
        for (int b = 0; b < NB; b++) {
            if (b0 + b < batch) {
                float x[16];
                load16<NB>(X + (size_t)(b0 + b) * cols, e0, cols, x, vec[b]);
                float s = 0.f;
#pragma unroll
                for (int i = 0; i < 16; i++) s = fmaf(w[i], x[i], s);
                acc[b] = fmaf(s, scale, acc[b]);
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

template <int NB>
__global__ void __launch_bounds__(128)
bnb8_gemv_kernel(const uint8_t *__restrict__ W, const float *__restrict__ X, float *__restrict__ Y,
                 const float *__restrict__ bias, int rows, int cols, int batch, int b0) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= rows) return;
    const uint8_t *rp = W + (size_t)row * (size_t)(cols / 64) * 68;
    bool vec[NB];
#pragma unroll
    for (int b = 0; b < NB; b++) vec[b] = ((reinterpret_cast<uintptr_t>(X + (size_t)(b0 + b) * cols)) & 15) == 0;
    float acc[NB];
#pragma unroll
    for (int b = 0; b < NB; b++) acc[b] = 0.f;
    const int runs = cols / 16;
    for (int run = lane; run < runs; run += FLLM_WARP) {
        const int e0 = run * 16;
        const uint8_t *blk = rp + (size_t)(e0 / 64) * 68;
        const float scale = __ldg(reinterpret_cast<const float *>(blk));
        const uint32_t *q = reinterpret_cast<const uint32_t *>(blk + 4 + (e0 % 64));
        float w[16];
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const uint32_t word = __ldg(q + j);
#pragma unroll
            for (int i = 0; i < 4; i++) w[4 * j + i] = (float)(int8_t)(word >> (8 * i));
        }
#pragma unroll
        for (int b = 0; b < NB; b++) {
            if (b0 + b < batch) {
                float x[16];
                load16<NB>(X + (size_t)(b0 + b) * cols, e0, cols, x, vec[b]);
                float s = 0.f;
#pragma unroll
                for (int i = 0; i < 16; i++) s = fmaf(w[i], x[i], s);
                acc[b] = fmaf(s, scale, acc[b]);
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

template <uint32_t KIND, int N>
static void bnb_launch_n(cudaStream_t s, const uint8_t *W, const float *X, float *Y, const float *bias, int rows, int cols, int batch,
                         int b0) {
    const dim3 block(128), grid((rows + 3) / 4);
    if constexpr (KIND == KIND_INT8R) {
        bnb8_gemv_kernel<N><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, b0);
    } else {
        bnb4_gemv_kernel<KIND, N><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, b0);
    }
}

template <uint32_t KIND>
static void bnb_launch_pass(cudaStream_t s, const uint8_t *W, const float *X, float *Y, const float *bias, int rows, int cols, int batch,
                            int b0) {
    switch (batch - b0 < 4 ? batch - b0 : 4) {
        case 1: bnb_launch_n<KIND, 1>(s, W, X, Y, bias, rows, cols, batch, b0); break;
        case 2: bnb_launch_n<KIND, 2>(s, W, X, Y, bias, rows, cols, batch, b0); break;
        case 3: bnb_launch_n<KIND, 3>(s, W, X, Y, bias, rows, cols, batch, b0); break;
        default: bnb_launch_n<KIND, 4>(s, W, X, Y, bias, rows, cols, batch, b0); break;
    }
}

// 1 when the dedicated kernel ran, 0 when the caller should use the generic one (decode16 based)
static int fllm_bnb_gemv_try(cudaStream_t s, const void *W, uint32_t kind, const float *X, float *Y, const float *bias, int rows, int cols,
                             int batch) {
    if (batch < 1 || batch > 8 || rows < 1 || cols < 64 || !kind_is_bnb(kind) || cols % kind_block_elems(kind) != 0)
        return 0;
    static const bool off = getenv("FLLM_CUDA_NO_BNB_GEMV") != nullptr;
    if (off) return 0;
    const uint8_t *w = static_cast<const uint8_t *>(W);
    for (int b0 = 0; b0 < batch; b0 += 4) {
        switch (kind) {
            case KIND_NF4B64: bnb_launch_pass<KIND_NF4B64>(s, w, X, Y, bias, rows, cols, batch, b0); break;
            case KIND_NF4B128: bnb_launch_pass<KIND_NF4B128>(s, w, X, Y, bias, rows, cols, batch, b0); break;
            case KIND_FP4B64: bnb_launch_pass<KIND_FP4B64>(s, w, X, Y, bias, rows, cols, batch, b0); break;
            case KIND_FP4B128: bnb_launch_pass<KIND_FP4B128>(s, w, X, Y, bias, rows, cols, batch, b0); break;
            default: bnb_launch_pass<KIND_INT8R>(s, w, X, Y, bias, rows, cols, batch, b0); break;
        }
    }
    return 1;
}

#else
static int fllm_bnb_gemv_try(cudaStream_t, const void *, uint32_t, const float *, float *, const float *, int, int, int) { return 0; }
#endif
