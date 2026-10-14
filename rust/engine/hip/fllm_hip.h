// fLLM HIP backend: entry points of the AMD-specific kernels (rust/engine/hip/*.cu).
//
// The CUDA sources in ../cuda are built with hipcc for AMD GPUs (FLLM_HIP=1). The files in this directory add kernels that exist only
// there: matrix-core (MFMA / WMMA) GEMM, skinny GEMV for 16-bit weights, decode attention, 4-bit group-quantized (Q4Z) GEMM. The
// dispatchers in ../cuda (linear.cu, tc_gemm.cu, api.cu) call them under `#if defined(__HIPCC__)`; every `*_try` function returns
// 0 (or -1 for attention) when it does not handle the call so the portable kernels run instead.
// Level of verification: L0 (compiles with the ROCm 7.2 clang for several gfx targets; never run on hardware).
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "../cuda/hip_compat.h"

enum FllmHipArch {
    FLLM_HIP_ARCH_OTHER = 0,   // no matrix cores we use (Vega, RDNA1/2, unknown)
    FLLM_HIP_ARCH_CDNA = 1,    // gfx908 / gfx90a / gfx94x / gfx950: wave64, MFMA
    FLLM_HIP_ARCH_RDNA3 = 2,   // gfx11xx: wave32, WMMA (16-element fragments)
    FLLM_HIP_ARCH_RDNA4 = 3,   // gfx120x: wave32, WMMA (8-element fragments)
};

struct FllmHipDevice {
    int arch;        // FllmHipArch
    int cus;         // compute units
    int lds_bytes;   // LDS usable by one workgroup
    int wave;        // wavefront size the code objects target: 64 on GCN / CDNA, 32 on RDNA
    bool dot2;       // v_dot2 on packed FP16 (gfx90a, gfx94x, gfx950, gfx11, gfx12)
    bool gfx1151;    // Strix Halo: tuned tile table in the skinny GEMM
    char gcn[48];    // gcnArchName, e.g. "gfx1100" or "gfx90a:sramecc+:xnack-"
};

// Properties of the current HIP device (cached per device ordinal; thread-safe).
const FllmHipDevice &fllm_hip_device();

// false when the environment variable FLLM_HIP_NO_<name> is set (e.g. FLLM_HIP_NO_MMA=1): switch a kernel family off at run time.
bool fllm_hip_enabled(const char *name);

// Matrix-core GEMM (hip_gemm.cu): Y[token][row] = sum_k X[token][k] * W[row][k] (+ bias), weights of any decodable kind, FP16 operands
// with FP32 accumulation. Returns 1 when it handled the call.
int fllm_hip_gemm_try(cudaStream_t s, const void *W, int kind, const float *X, float *Y, const float *bias, int rows, int cols, int batch);

// Skinny GEMM for F16 / BF16 weights and 1..4 tokens (hip_skinny.cu, after vLLM's wvSplitK). `scratch` holds the converted activations.
// Returns 1 when it handled the call.
int fllm_hip_skinny_try(cudaStream_t s, const void *W, int kind, const float *X, float *Y, const float *bias, int rows, int cols, int batch,
                        void *scratch, size_t scratch_bytes);

// Decode attention: the GQA kernel ../cuda/attention4.cu sums the key dot products with DPP (hip_dpp.cuh) when built for AMD. vLLM's
// csrc/rocm/attention.cu is not ported: it reads vLLM's cache layout (K [blocks][heads][D/x][block][x], V [blocks][heads][D][block]),
// fLLM's shared cache is [slot][kv_head][D] (see docs in the package report).
