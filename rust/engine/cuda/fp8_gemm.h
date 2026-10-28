// fLLM CUDA backend: FP8 / int8 W8A8 tensor-core GEMMs (fp8_gemm.cu, int8_gemm.cu). Declarations for api.cu and linear.cu.
#pragma once
#include <cuda_runtime.h>

// ---- FP8 (E4M3) weights x FP8 activations, FP32 accumulation (mma.sync m16n8k32 e4m3, Ada / Hopper / Blackwell) ----
// FLLM_FP8_GEMM = auto | 0 | 1 | cutlass; fllm_fp8_gemm_set_mode overrides it at run time (self test): -1 environment, 0 off, 1 mma.sync, 2 CUTLASS.
void fllm_fp8_gemm_set_mode(int mode);
int fllm_fp8_gemm_mode();
// The mode a caller forced with fllm_fp8_gemm_set_mode (0 off, 1 mma.sync, 2 CUTLASS), -1 when none: lets code that has its own default
// (block-wise FP8 weights, fp8_block.cu) tell "not set" from "off".
int fllm_fp8_gemm_forced_mode();
// Block-wise FP8 weights (GGUF type FP8B; fp8_block.cu): y[batch][rows] = x[batch][cols] * W^T (+ bias) for the device buffer `w` of such a weight
// (rows * cols E4M3 codes, then the FP32 tile scales; tiles of 2^lbr rows x 2^lbc columns). Chooses between the FP8 tensor cores
// (fllm_fp8_w8a8_gemm), the FP16 tensor-core GEMM and the FP32 GEMV; 0 when the product ran, > 0 on error.
int fllm_fp8_block_gemm(cudaStream_t s, const void *w, const float *x, float *y, const float *bias, int rows, int cols, int batch, int lbr, int lbc);
// Should an FP8M weight of rows x cols be uploaded in the W8A8 layout (row-major E4M3 bytes + FP32 scales) on the current device?
bool fllm_fp8_w8a8_enabled(int rows, int cols);
// y[batch][rows] = x[batch][cols] * W^T (+ bias). W: E4M3 [rows][cols]. wblock 0: wscale = FP32 per row; wblock 1: wscale = FP32
// [ceil(rows / 128)][cols / 128] (DeepSeek style 128 x 128 blocks, dequantization factors) and the activations get one scale per
// 1 x 128 group. amode: 0 one dynamic scale per token, 1 one static scale (static_scale, the dequantization factor), 2 one dynamic
// scale per token and 128 values. Returns 0 when the kernel ran, -1 when the shape / device / pointers are not covered, > 0 on error.
int fllm_fp8_w8a8_gemm(cudaStream_t s, const void *w, const float *wscale, int wblock, const float *x, float *y, const float *bias, int rows, int cols,
                       int batch, int amode, float static_scale);

// ---- int8 weights x int8 activations, int32 accumulation (mma.sync m16n8k32 s8, Turing and newer) ----
// FLLM_INT8_GEMM = auto | 0 | 1; fllm_int8_gemm_set_mode overrides it (-1 environment, 0 off, 1 on). auto is off until measured.
void fllm_int8_gemm_set_mode(int mode);
// Q8_0 weights (device layout per row [FP16 scales][int8 values], see backend/cuda.rs): the activations are quantized per token and 32 values in the usual Q8_0 activation format.
// Returns 0 when the kernel ran, -1 when not covered (switch off, batch < min_batch, cc < 75, alignment).
int fllm_int8_q8_try(cudaStream_t s, const void *w, const float *x, float *y, const float *bias, int rows, int cols, int batch);
// Plain int8 weights [rows][cols] with one FP32 scale per row (SmoothQuant / compressed-tensors W8A8 layout), per-token dynamic activations.
int fllm_int8_plain_gemm(cudaStream_t s, const void *w, const float *wscale, const float *x, float *y, const float *bias, int rows, int cols, int batch);

// ---- scratch of the W8A8 kernels: the quantized activations and their scales, one pair per stream (shared by fp8_gemm.cu and int8_gemm.cu) ----
#include <map>
#include <vector>
struct W8A8Scratch {
    void *q = nullptr;
    size_t q_bytes = 0;
    float *s = nullptr;
    size_t s_bytes = 0;
};
// Buffers of at least the given sizes for `stream`; nullptr when they cannot be allocated, or must grow while the stream is being captured
// into a CUDA graph. A buffer that grows is not freed (a graph captured earlier may still point to it): the old one is kept, growth doubles.
static inline W8A8Scratch *w8a8_scratch(cudaStream_t stream, size_t q_bytes, size_t s_bytes) {
    static thread_local std::map<cudaStream_t, W8A8Scratch> table;
    W8A8Scratch &sc = table[stream];
    if (q_bytes <= sc.q_bytes && s_bytes <= sc.s_bytes) return &sc;
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(stream, &cap) != cudaSuccess || cap != cudaStreamCaptureStatusNone) {
        cudaGetLastError();
        return nullptr;
    }
    if (q_bytes > sc.q_bytes) {
        const size_t want = q_bytes > 2 * sc.q_bytes ? q_bytes : 2 * sc.q_bytes;
        void *p = nullptr;
        if (cudaMalloc(&p, want) != cudaSuccess) {
            cudaGetLastError();
            return nullptr;
        }
        sc.q = p;
        sc.q_bytes = want;
    }
    if (s_bytes > sc.s_bytes) {
        const size_t want = s_bytes > 2 * sc.s_bytes ? s_bytes : 2 * sc.s_bytes;
        void *p = nullptr;
        if (cudaMalloc(&p, want) != cudaSuccess) {
            cudaGetLastError();
            return nullptr;
        }
        sc.s = static_cast<float *>(p);
        sc.s_bytes = want;
    }
    return &sc;
}

// CUTLASS Hopper path (fp8_cutlass_sm90.cu): per-token activation scales, per-row weight scales; the activations are already quantized
int fllm_fp8_cutlass_gemm(cudaStream_t s, const void *w, const float *wscale, const void *xq, const float *xs, float *y, const float *bias, int rows,
                          int cols, int batch);
int fllm_fp8_cutlass_blockwise_gemm(cudaStream_t s, const void *w, const float *wscale, const void *xq, const float *xs, float *y, const float *bias, int rows,
                                    int cols, int batch, int wblock);
