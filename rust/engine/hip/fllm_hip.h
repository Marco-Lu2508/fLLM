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
    FLLM_HIP_ARCH_OTHER = 0,   // plain kernels only: Vega, RDNA1/2, gfx117x, gfx125x, unknown and future chips (hip/arch_table.csv)
    FLLM_HIP_ARCH_CDNA = 1,    // gfx908 / gfx90a / gfx94x / gfx950 and gfx9-4-generic: wave64, MFMA
    FLLM_HIP_ARCH_RDNA3 = 2,   // gfx11xx except gfx117x: wave32, WMMA (16-element fragments)
    FLLM_HIP_ARCH_RDNA4 = 3,   // gfx1200 / gfx1201: wave32, WMMA (8-element fragments)
};

struct FllmHipDevice {
    int arch;        // FllmHipArch: the matrix-core kernel set the host launches (decided from the loaded code object, see hip_device.cu)
    int cus;         // compute units
    int lds_bytes;   // LDS usable by one workgroup
    int wave;        // wavefront size the loaded code object runs with: 64 on GCN / CDNA, 32 on RDNA
    bool dot2;       // the loaded code has the packed FP16 dot product the skinny kernels use (FLLM_HIP_DOT2 of cuda/hip_compat.h)
    int dot4;        // 8-bit integer dot product of the loaded code: 0 byte loop, 1 v_dot4_i32_i8, 2 v_dot4_i32_iu8
    bool big_lds;    // the loaded code is the gfx950 one, which uses 160 KiB of LDS (the generic gfx9-4 code and every other target use 64 KiB)
    bool usable;     // the device runs the code objects of this binary (self-check of hip_device.cu); false: nothing may be launched
    bool gfx1151;    // Strix Halo: tuned tile table in the skinny GEMM
    char gcn[48];    // gcnArchName, e.g. "gfx1100" or "gfx90a:sramecc+:xnack-"
};

// Properties of the current HIP device (cached per device ordinal; thread-safe).
const FllmHipDevice &fllm_hip_device();

// Device check for the Rust side (list_devices): 0 when the device runs this binary's code (a small FP32 product, the integer dot product and,
// where the code object has them, one matrix-core step are compared with the host), else 1 and `msg` says why. `info`: six ints (kernel set of
// the loaded code, wave size, dot kind, dot2, matrix-core step -1 none / 0 wrong / 1 right, kernel set the host launches). Cached per device.
extern "C" int fllm_hip_selfcheck(int dev, char *msg, size_t msg_len, int *info);

// false when the environment variable FLLM_HIP_NO_<name> is set (e.g. FLLM_HIP_NO_MMA=1): switch a kernel family off at run time.
bool fllm_hip_enabled(const char *name);

// Matrix-core GEMM (hip_gemm.cu): Y[token][row] = sum_k X[token][k] * W[row][k] (+ bias), weights of any decodable kind, FP16 operands
// with FP32 accumulation. Returns 1 when it handled the call.
int fllm_hip_gemm_try(cudaStream_t s, const void *W, int kind, const float *X, float *Y, const float *bias, int rows, int cols, int batch);
int fllm_hip_gemm_x16_try(cudaStream_t s, const void *W, int kind, const float *X, float *Y, const float *bias, int rows, int cols, int batch, void *x16,
                          size_t x16_bytes);

// Skinny GEMM for F16 / BF16 weights and 1..4 tokens (hip_skinny.cu, after vLLM's wvSplitK). `scratch` holds the converted activations.
// Returns 1 when it handled the call.
int fllm_hip_skinny_try(cudaStream_t s, const void *W, int kind, const float *X, float *Y, const float *bias, int rows, int cols, int batch,
                        void *scratch, size_t scratch_bytes);

// Decode attention: the GQA kernel ../cuda/attention4.cu sums the key dot products with DPP (hip_dpp.cuh) when built for AMD. vLLM's
// csrc/rocm/attention.cu is not ported: it reads vLLM's cache layout (K [blocks][heads][D/x][block][x], V [blocks][heads][D][block]),
// fLLM's shared cache is [slot][kv_head][D] (see docs in the package report).
