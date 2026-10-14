// fLLM GPU backend, AMD build: the NVIDIA tensor-core entry points do not exist here. They decline every call (0 / -1), so
// the dispatcher in api.cu / linear.cu falls back to the portable kernels (integer dot product, FP32 GEMM, plain attention).
#include "common.cuh"
#include "fllm_cuda_internal.h"

thread_local bool fllm_x16_reuse = false, fllm_x16_filled = false;
thread_local bool fllm_x8_reuse = false, fllm_x8_filled = false;
thread_local const float *fllm_glu_up = nullptr;
thread_local int fllm_glu_mode = 0;

int fllm_attention3_try(cudaStream_t, const float *, const uint16_t *, const uint16_t *, float *, const int64_t *,
                        const int64_t *, const float *, int, int, int, int, int, int, int, int, float, int, int, int) {
    return -1;
}
int fllm_nvfp4_try(cudaStream_t, const void *, const float *, float *, const float *, float, int, int, int, void *, size_t, int) {
    return 0;
}
int fllm_nvfp4_cutlass_ok(const float *, const float *, const float *, int, int, int, int) { return 0; }
int fllm_qgemm4k_try(cudaStream_t, const void *, const float *, float *, const float *, int, int, int, void *, size_t) { return 0; }
int fllm_qgemm5k_try(cudaStream_t, const void *, const float *, float *, const float *, int, int, int, void *, size_t) { return 0; }
int fllm_qgemm6k_try(cudaStream_t, const void *, const float *, float *, const float *, int, int, int, void *, size_t) { return 0; }
int fllm_qgemm8_try(cudaStream_t, const void *, const float *, float *, const float *, int, int, int, void *, size_t) { return 0; }
int fllm_qgemm_leg_try(cudaStream_t, const void *, int, const float *, float *, const float *, int, int, int, void *, size_t) { return 0; }

// 4-bit AWQ / GPTQ weights: fllm_q4z_supported / _io_bytes / _ctmp_bytes / _workspace_bytes / _launch / _repack are implemented for AMD in
// hip/hip_q4z.cu (matrix-core GEMM and GEMV, no Marlin). Only the GPTQ-ordered repack, which the NVFP4 Marlin path uses, has no AMD version.
int fllm_q4z_gptq_repack(int, const void *, void *, int, int) { return fllm_set_error("Marlin 4-bit GEMM needs an NVIDIA GPU"); }

// FP8 weights through the Marlin GEMM of vLLM need NVIDIA tensor cores (the public entry points are in api.cu)
bool fllm_fp8_marlin_supported(int, int) { return false; }
int fllm_fp8_marlin_launch(cudaStream_t, int, int, const void *, const float *, float *, const float *, int, int, int, void *, void *,
                           void *, const float *, const float *, int, bool) {
    return -1;
}
int fllm_fp8_marlin_repack(int, const void *, void *, int, int) { return fllm_set_error("Marlin FP8 repack needs an NVIDIA GPU"); }

// Q4_K Marlin GEMM (marlin_q4k.cu) is NVIDIA-only: decline, the portable kernels serve the call
int fllm_marlin_q4k_try(cudaStream_t, const void *, const float *, float *, const float *, int, int, int, void *, size_t) { return 0; }

// Gated DeltaNet (deltanet.cu) is not built for AMD: the backend reports it as unsupported and the host path runs
int fllm_delta_supported(int, int, int, int, int) { return 0; }
int fllm_delta_launch(cudaStream_t, const void *, int, int, int, int, int, int, int, int, float, float, size_t, size_t, const float *,
                      const float *, const float *, const float *, const float *, const float *, const float *, const float *, float *,
                      float *) {
    return fllm_set_error("DeltaNet kernels need an NVIDIA GPU");
}

// Architecture-specific element-wise kernels (arch_ops.cu, mla.cu, mamba2.cu, shortconv.cu) are not built for AMD yet: the backend reports
// them through supports_arch_ops (the host path runs those layers)
int fllm_act_launch(cudaStream_t, const float *, float *, size_t, int) { return fllm_set_error("arch ops need an NVIDIA GPU"); }
int fllm_act_glu_launch(cudaStream_t, const float *, const float *, float *, size_t, int) { return fllm_set_error("arch ops need an NVIDIA GPU"); }
int fllm_clamp_launch(cudaStream_t, float *, size_t, float) { return fllm_set_error("arch ops need an NVIDIA GPU"); }
int fllm_mla_latent_launch(cudaStream_t, const float *, float *, int, int, int) { return fllm_set_error("MLA kernels need an NVIDIA GPU"); }
int fllm_mla_assemble_launch(cudaStream_t, const float *, const float *, const float *, float *, float *, float *, int, int, int, int, int,
                             int) {
    return fllm_set_error("MLA kernels need an NVIDIA GPU");
}
int fllm_mamba2_supported(int, int, int, int, int, int) { return 0; }
int fllm_mamba2_launch(cudaStream_t, const void *, int, int, int, int, int, int, int, int, float, float, size_t, size_t, const float *,
                       const float *, const float *, const float *, const float *, const float *, const float *, float *, float *) {
    return fllm_set_error("Mamba-2 kernels need an NVIDIA GPU");
}
int fllm_shortconv_supported(int, int) { return 0; }
int fllm_shortconv_launch(cudaStream_t, const void *, int, int, int, size_t, const float *, const float *, const float *, float *) {
    return fllm_set_error("ShortConv kernels need an NVIDIA GPU");
}
int fllm_softplus_mul_launch(cudaStream_t, float *, const float *, size_t, float) { return fllm_set_error("arch ops need an NVIDIA GPU"); }
int fllm_ssm_conv_launch(cudaStream_t, const void *, int, const float *, int, int, const float *, const float *, float *, int, int, size_t, size_t) {
    return fllm_set_error("state space kernels need an NVIDIA GPU");
}
int fllm_mamba1_supported(int, int, int, int) { return 0; }
int fllm_mamba1_conv_launch(cudaStream_t, const void *, int, int, int, size_t, size_t, const float *, const float *, const float *, float *) {
    return fllm_set_error("Mamba-1 kernels need an NVIDIA GPU");
}
int fllm_mamba1_prep_launch(cudaStream_t, float *, float *, int, int, int, float) { return fllm_set_error("Mamba-1 kernels need an NVIDIA GPU"); }
int fllm_mamba1_scan_launch(cudaStream_t, const void *, int, int, int, int, size_t, const float *, const float *, const float *, const float *,
                            const float *, const float *, float *) {
    return fllm_set_error("Mamba-1 kernels need an NVIDIA GPU");
}
