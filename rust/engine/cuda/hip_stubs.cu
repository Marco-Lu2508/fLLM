// fLLM GPU backend, AMD build: the NVIDIA tensor-core entry points do not exist here. They decline every call (0 / -1), so
// the dispatcher in api.cu / linear.cu falls back to the portable kernels (integer dot product, FP32 GEMM, plain attention).
#include "common.cuh"
#include "fllm_cuda_internal.h"

thread_local bool fllm_x16_reuse = false, fllm_x16_filled = false;
thread_local bool fllm_x8_reuse = false, fllm_x8_filled = false;
thread_local const float *fllm_glu_up = nullptr;
thread_local float *fllm_glu_dst = nullptr;
thread_local int fllm_glu_mode = 0;

int fllm_attention3_try(cudaStream_t, const float *, const uint16_t *, const uint16_t *, float *, const int64_t *,
                        const int64_t *, const float *, int, int, int, int, int, int, int, int, float, int, int, int) {
    return -1;
}
int fllm_fi_attention_try(cudaStream_t, const float *, const uint16_t *, const uint16_t *, float *, const int64_t *, const int64_t *, int, const float *, bool,
                          int, int, int, int, int, int, int, float, float, int, int) {
    return -1;
}
int fllm_nvfp4_try(cudaStream_t, const void *, const float *, float *, const float *, float, int, int, int, void *, size_t, int) {
    return 0;
}
int fllm_nvfp4_cutlass_ok(const float *, const float *, const float *, int, int, int, int) { return 0; }
int fllm_nvfp4_norm_quant_try(cudaStream_t, const float *, const float *, float *, int, int, float, const void *, int, int) { return 0; }
int fllm_flashinfer_expand_half(cudaStream_t, const void *, float *, size_t) { return 0; }
int fllm_qgemm4k_try(cudaStream_t, const void *, const float *, float *, const float *, int, int, int, void *, size_t) { return 0; }
int fllm_qgemm5k_try(cudaStream_t, const void *, const float *, float *, const float *, int, int, int, void *, size_t) { return 0; }
int fllm_qgemm6k_try(cudaStream_t, const void *, const float *, float *, const float *, int, int, int, void *, size_t) { return 0; }
int fllm_qgemm8_try(cudaStream_t, const void *, const float *, float *, const float *, int, int, int, void *, size_t) { return 0; }
int fllm_qgemm_leg_try(cudaStream_t, const void *, int, const float *, float *, const float *, int, int, int, void *, size_t) { return 0; }

// 4-bit AWQ / GPTQ weights: fllm_q4z_supported / _io_bytes / _ctmp_bytes / _workspace_bytes / _launch / _repack are implemented for AMD in
// hip/hip_q4z.cu (matrix-core GEMM and GEMV, no Marlin). Only the GPTQ-ordered repack, which the NVFP4 Marlin path uses, has no AMD version.

// FP8 weights through the Marlin GEMM of vLLM need NVIDIA tensor cores (the public entry points are in api.cu)
bool fllm_fp8_marlin_supported(int, int) { return false; }
int fllm_fp8_marlin_launch(cudaStream_t, int, int, const void *, const float *, float *, const float *, int, int, int, void *, void *,
                           void *, const float *, const float *, int, bool) {
    return -1;
}
int fllm_fp8_marlin_repack(int, const void *, void *, int, int) { return fllm_set_error("Marlin FP8 repack needs an NVIDIA GPU"); }

// Native EXL2 / EXL3 kernels (exl.cu) are not built for AMD: the backend does not list the formats and the decoder expands them on upload
size_t fllm_exl3_scratch_bytes(int, int, int) { return 0; }
int fllm_exl_launch(cudaStream_t, int, const void *, const float *, float *, const float *, int, int, int, void *, size_t) {
    return fllm_set_error("EXL2 / EXL3 kernels need an NVIDIA GPU");
}

// MLX affine weights (mlx_affine.cu) have no AMD kernels yet: the decoder expands them to f16 for these GPUs (the backend does not list them)
int fllm_mlx_launch(cudaStream_t, const void *, int, const float *, float *, const float *, int, int, int) {
    return fllm_set_error("MLX affine kernels need an NVIDIA GPU");
}
extern "C" int fllm_cuda_mlx_force(int) { return 0; }

// Q4_K Marlin GEMM (marlin_q4k.cu) is NVIDIA-only: decline, the portable kernels serve the call
int fllm_marlin_q4k_try(cudaStream_t, const void *, const float *, float *, const float *, int, int, int, void *, size_t) { return 0; }

// Datacenter Blackwell NVFP4 (nvfp4_cutlass_sm100.cu) and the device report (arch_report.cu) are NVIDIA-only
extern "C" int fllm_cuda_nvfp4_sm100_native(int) { return 0; }
extern "C" int fllm_cuda_nvfp4_mma_native(int) { return 0; }
extern "C" int fllm_cuda_fast_paths(int, char *buf, size_t len) {
    if (buf != nullptr && len > 0) buf[0] = 0;
    return 0;
}

// Hopper / FlashInfer attention switches (fi_attn.cu) do not exist in the AMD build
extern "C" int fllm_cuda_attn_get(int) { return 0; }
extern "C" int fllm_cuda_attn_set(int, int) { return -1; }

// Latent-cache MLA (mla_latent.cu, mla_decode.cu) needs NVIDIA tensor cores: the backend keeps the expanded cache on AMD GPUs
#include "mla_latent.h"
int fllm_mla_latent_supported(int, int, int, int, int) { return 0; }
int fllm_mla_kv_launch(cudaStream_t, const float *, const float *, const int32_t *, const float *, const float *, float *, int, int, int, int) {
    return fllm_set_error("latent MLA needs an NVIDIA GPU");
}
int fllm_mla_query_launch(cudaStream_t, const float *, const uint16_t *, const int32_t *, const float *, const float *, uint16_t *, int, int, int, int,
                          int, int, int) {
    return fllm_set_error("latent MLA needs an NVIDIA GPU");
}
int fllm_mla_output_launch(cudaStream_t, const uint16_t *, const uint16_t *, float *, int, int, int, int, int) {
    return fllm_set_error("latent MLA needs an NVIDIA GPU");
}
int fllm_mla_attention_launch(cudaStream_t, const uint16_t *, const uint16_t *, uint16_t *, const int64_t *, const int64_t *, const int64_t *,
                              const int64_t *, int, int, int, int, int, int, const FllmKvDesc *, float, int) {
    return -1;
}
extern "C" int fllm_cuda_nvfp4_sm100_moe_gemm(cudaStream_t, const void *const *, const float *, float *, const int *, int, int, int) { return -1; }
// Device-side mixture-of-experts (moe.cu) is not built for AMD: the MoE decoder paths stay on the existing moe_topk / moe_linear kernels
int fllm_moe_route2_launch(cudaStream_t, const float *, const float *, int32_t *, float *, int, int, int, int, int, float, float, int, int,
                           int) {
    return fllm_set_error("device MoE routing needs an NVIDIA GPU");
}
int fllm_moe_align_launch(cudaStream_t, const int32_t *, int32_t *, int32_t *, int32_t *, int, int, int, int) {
    return fllm_set_error("device MoE sorting needs an NVIDIA GPU");
}
int fllm_moe_grouped_supported(uint32_t, int, int) { return 0; }
int fllm_moe_grouped_launch(cudaStream_t, uint32_t, int, const void *const *, const void *const *, const void *const *, const void *const *,
                            const float *, const float *, float *, const int32_t *, const int32_t *, const int32_t *, int, int, int, int,
                            int, int, int, float, float) {
    return fllm_set_error("grouped MoE GEMM needs an NVIDIA GPU");
}
int fllm_moe_combine2_launch(cudaStream_t, float *, const float *, const float *, const float *, const float *, int, int, int) {
    return fllm_set_error("device MoE combine needs an NVIDIA GPU");
}
int fllm_moe_gate_scale_launch(cudaStream_t, float *, const float *, const float *, int, int) {
    return fllm_set_error("device MoE gate needs an NVIDIA GPU");
}

// Kernels that exist only in the NVIDIA build (sampling, tree attention, speculative verification, encoder attention, Q4Z accumulate).
int fllm_q4z_acc_launch(cudaStream_t s, const void *y16, const float *bias, float *out, int rows, int batch, float scale) {
    return fllm_set_error("this kernel needs an NVIDIA GPU");
}
int fllm_sample_launch(cudaStream_t s, const float *logits, float *work, int rows, int vocab, const void *params, const int32_t *pen_ids,
                       const int32_t *pen_counts, const int32_t *bias_ids, const float *bias_vals, const uint32_t *mask, int mask_words,
                       int32_t *out_ids, float *out_logprob, int32_t *top_ids, float *top_logprob, int top_stride) {
    return fllm_set_error("this kernel needs an NVIDIA GPU");
}
int fllm_tree_attention_launch(cudaStream_t s, const float *q, const uint16_t *kpool_layer,
                               const uint16_t *vpool_layer, float *out, const int64_t *tables,
                               const int64_t *lengths, const uint8_t *tree_mask, int prefix_len,
                               int num_nodes, int rows, int heads, int kv_heads, int dk, int dv,
                               int stride, int block_size, int kind, float scale, float softcap) {
    return fllm_set_error("this kernel needs an NVIDIA GPU");
}
int fllm_spec_verify_launch(cudaStream_t s, const int32_t *draft, const int32_t *target,
                            int num_draft, int32_t *out_accepted, int32_t *out_next,
                            int64_t *seq_length) {
    return fllm_set_error("this kernel needs an NVIDIA GPU");
}
int fllm_attn_encoder_launch(cudaStream_t s, const float *q, const float *k, const float *v, float *out,
                             const int32_t *cu_seqlens, int batch_size, int total_tokens,
                             int heads, int kv_heads, int head_dim, float scale, int window,
                             const int32_t *global_tokens, int num_global) {
    return fllm_set_error("this kernel needs an NVIDIA GPU");
}

// C entry points of the NVIDIA-only paths (narrow activations, Marlin small-m, PDL) that the Rust backend declares; AMD builds answer "unsupported".
extern "C" {
int fllm_cuda_na_check() { return -1; }
int fllm_cuda_na_convert() { return -1; }
int fllm_cuda_na_glu() { return -1; }
int fllm_cuda_na_glu_merged() { return -1; }
int fllm_cuda_na_kv_write() { return -1; }
int fllm_cuda_na_linear_kquant() { return -1; }
int fllm_cuda_na_linear_q4z() { return -1; }
int fllm_cuda_na_residual() { return -1; }
int fllm_cuda_na_rms_norm() { return -1; }
int fllm_cuda_na_rope() { return -1; }
int fllm_cuda_na_rope_kv() { return -1; }
int fllm_cuda_na_rope_kv_qkv() { return -1; }
int fllm_cuda_na_split3() { return -1; }
int fllm_cuda_reserve_workspace() { return -1; }
int fllm_marlin_gemv_bench() { return -1; }
void fllm_marlin_small_m_set() {}
void fllm_pdl_force() {}
}
