// fLLM HIP backend: matrix-core GEMM for prompt processing and larger decode batches on AMD GPUs.
//
//   CDNA   (gfx908, gfx90a, gfx94x, gfx950)  v_mfma_f32_16x16x16_f16
//   RDNA3  (gfx11xx)                          v_wmma_f32_16x16x16_f16 (wave32)
//   RDNA4  (gfx120x)                          v_wmma_f32_16x16x16_f16 (gfx12 encoding)
//
// It replaces the rocWMMA path of ../cuda/tc_gemm.cu (which stays as the fallback): no library dependency, direct builtins, the wave grid
// chosen per architecture. The kernel is in hip_gemm.cuh. Weights stay quantized in global memory and are decoded into LDS.
// FLLM_HIP_NO_MMA=1 switches it off. The caller (linear.cu) offers it batches of 16 tokens and more, like the CUDA tensor-core path.
// Level of verification: L0 (compiles; never run on hardware).
#include <cstdlib>

#include "../cuda/fllm_cuda_internal.h"
#include "hip_gemm.cuh"

using namespace fllm_hip;

template <int KIND>
static int run(cudaStream_t s, const uint8_t *W, const float *X, float *Y, const float *bias, int rows, int cols, int batch) {
    const KindDecoder<KIND> dec{W, kind_row_bytes(KIND, cols), cols};
    launch_mma_gemm(s, dec, GemmArgs{X, Y, bias, nullptr, rows, cols, batch, 0, 0.f});
    return 1;
}

int fllm_hip_gemm_try(cudaStream_t s, const void *W, int kind, const float *X, float *Y, const float *bias, int rows, int cols, int batch) {
    static const bool enabled = fllm_hip_enabled("MMA");
    if (!enabled || batch < 1) return 0;
    if (fllm_hip_device().arch == FLLM_HIP_ARCH_OTHER) return 0;
    if ((cols & 15) != 0 || (reinterpret_cast<uintptr_t>(X) & 15) != 0 || rows <= 0 || batch <= 0) return 0;
    const uint8_t *w = static_cast<const uint8_t *>(W);
    switch (kind) {
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
        default: return 0;
    }
}
