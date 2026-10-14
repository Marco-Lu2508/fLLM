// fLLM CUDA backend: declarations shared between the kernel files and the C API.
#pragma once
#if defined(__HIPCC__)
#include "hip_compat.h"
#else
#include <cuda_runtime.h>
#endif
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <map>
#include <mutex>

int fllm_set_error(const char *fmt, ...);
int fllm_check_launch(const char *what);

// Switches of the prefill host-work reductions and kernel fusions (docs/performance-notes.md). The environment gives the initial values
// (FLLM_PREFILL_STEP_CACHE, FLLM_FUSE_PREFILL_ADD_NORM, FLLM_FUSE_PREFILL_NORM_F16, FLLM_FUSE_PREFILL_ATTN_F16, FLLM_MARLIN_PREFILL_CFG);
// selftest-gpu flips them at run time (fllm_cuda_set_opt) to compare the variants on one device.
enum FllmOpt {
    FLLM_OPT_STEP_CACHE = 0,  // host data of one forward pass (block tables, run layout) is staged and checked once, not once per layer
    FLLM_OPT_FUSE_ADD_NORM,   // the residual add of a Marlin GEMM is folded into the RMS norm that follows
    FLLM_OPT_FUSE_NORM_F16,   // the RMS norm also writes the FP16 activations that the next Marlin GEMM needs
    FLLM_OPT_FUSE_ATTN_F16,   // the prompt attention also writes the FP16 activations that the output projection needs
    FLLM_OPT_MARLIN_CFG,      // 0: Marlin launch configuration as vLLM selects it, 1: fLLM variant (the M tail rides in the last tile)
    FLLM_OPT_COUNT
};
int fllm_opt(int id);
// the fusions apply from this many rows on (prompt batches); decode steps keep the unfused paths
constexpr int FLLM_PREFILL_MIN_ROWS = 16;

// RMS norm of prompt batches with the residual add of a pending Marlin result and/or the FP16 copy of the result folded in (elementwise.cu)
bool fllm_rms_norm_prompt_ok(const float *x, const float *w, const float *out, int cols);
int fllm_rms_norm_prompt_launch(cudaStream_t s, const void *y16, const float *bias, float scale, float *x, const float *w, float *out,
                                void *out16, int rows, int cols, float eps);

int fllm_linear_launch(cudaStream_t s, const void *W, int kind, const float *X, float *Y, const float *bias, int rows,
                       int cols, int batch, void *x16, size_t x16_bytes);
// true when FLLM_CUDA_EXACT is set: only the FP32 kernels may run (used by selftest-gpu)
bool fllm_exact_math();
// Layers hand the same activations to several consecutive linears (q, k, v / gate, up). The first one converts them to
// the permuted FP16 layout in the scratch buffer; the stream remembers that, sets `fllm_x16_reuse` for the next call
// and learns from `fllm_x16_filled` whether the scratch still holds the converted activations afterwards.
extern thread_local bool fllm_x16_reuse, fllm_x16_filled;
// The same for the int8 activations of the integer dot-product kernels (mmvq.cu), which live in the same scratch.
extern thread_local bool fllm_x8_reuse, fllm_x8_filled;
int fllm_mmvq_maxb();
// Q2_K, Q3_K and IQ1_S .. IQ3_S with int8 activations (mmvq_x.cuh)
int fllm_mmvq_x_try(cudaStream_t s, const void *W, int kind, const float *X, float *Y, const float *bias, int rows, int cols,
                    int batch, void *scratch, size_t scratch_bytes);
int fllm_mmvq_leg_try(cudaStream_t s, const void *W, int kind, const float *X, float *Y, const float *bias, int rows,
                      int cols, int batch, void *scratch, size_t scratch_bytes);
// When set, the activation conversion of the next GEMM reads `gate` (its X argument) and `up` and computes the GLU
// itself instead of reading the output of a separate glu kernel.
extern thread_local const float *fllm_glu_up;
extern thread_local int fllm_glu_mode;
// the pointers of that GLU (X and fllm_glu_up) are FP16 arrays (the NVFP4 CUTLASS path only)
extern thread_local bool fllm_glu_half;
// compute capability of the current device times ten (80 = sm_80); tensor-core kernels need >= 80
int fllm_cc();
// dynamic shared memory per block the device allows
int fllm_smem_limit();
int fllm_gemv2_try(cudaStream_t s, const void *W, int kind, const float *X, float *Y, const float *bias, int rows,
                   int cols, int batch);
int fllm_tc_gemm_try(cudaStream_t s, const void *W, int kind, const float *X, float *Y, const float *bias, int rows,
                     int cols, int batch);
int fllm_qgemm4k_try(cudaStream_t s, const void *W, const float *X, float *Y, const float *bias, int rows, int cols,
                     int batch, void *x16, size_t x16_bytes);
int fllm_embedding_launch(cudaStream_t s, const void *W, int kind, const int64_t *ids_dev, float *out, int cols, int n);
int fllm_rms_norm_launch(cudaStream_t s, const float *x, const float *w, float *out, int rows, int cols, float eps);
int fllm_layer_norm_launch(cudaStream_t s, const float *x, const float *w, const float *b, float *out, int rows,
                           int cols, float eps);
int fllm_gather_rows_launch(cudaStream_t s, float *dst, const float *src, const int32_t *idx, const float *w, int n, int row_len);
int fllm_scatter_add_rows_launch(cudaStream_t s, float *dst, const float *src, const int32_t *idx, const float *w, int n, int row_len);
int fllm_moe_linear_launch(cudaStream_t s, const void *const *tab, int kind, const int32_t *ids, const float *X, float *Y,
                           int rows, int cols, int slots, int xdiv, const void *const *btab, void *scratch, size_t scratch_bytes);
int fllm_moe_route_launch(cudaStream_t s, const float *logits, int32_t *ids, float *w, int rows, int experts, int k,
                          int normalize, float scale);
int fllm_moe_combine_launch(cudaStream_t s, float *out, const float *y, const float *w, int tokens, int k, int hidden);
int fllm_swiglu_oai_launch(cudaStream_t s, const float *gate, const float *up, float *out, size_t count, float alpha, float limit);
void fllm_convert_x16_pad(cudaStream_t s, const float *X, void *x16, int batch, int cols, int cols_pad);
int fllm_repack_launch(cudaStream_t s, void *dst, const void *src, int kind, size_t rows, size_t cols);
int fllm_glu_launch(cudaStream_t s, const float *gate, const float *up, float *out, size_t count, int mode);
// the same for FP16 gate and up (count even)
int fllm_glu_launch_f16(cudaStream_t s, const void *gate16, const void *up16, float *out, size_t count, int mode);
int fllm_argmax_launch(cudaStream_t s, const float *logits, int rows, int vocab, int *out);
int fllm_rope_kv_f16_launch(cudaStream_t s, const uint16_t *q16, const uint16_t *k16, const uint16_t *v16, const float *qb,
                            const float *kb, const float *vb, float *q_out, uint16_t *pool_k, uint16_t *pool_v, const int32_t *pos,
                            const int32_t *slots, const float *cs, const float *sn, int rows, int heads, int kv_heads, int dk, int dv,
                            int half_, int neox);
extern thread_local bool fllm_q4z_f16out;  // the next Marlin 4-bit GEMM leaves its FP16 result in y (no conversion, no bias)
// the next Marlin 4-bit GEMM stops after the product: the FP16 result stays in the io buffer for a fused consumer (fllm_cuda_linear_q4z_add_norm)
extern thread_local bool fllm_q4z_skip_result;

// ---- fused decode kernels (fused.cu). Each fusion has a switch (FLLM_FUSE, FLLM_FUSE_<NAME>=0|1) and can be forced by the self test.
enum { FLLM_FUSE_ADD_NORM = 0, FLLM_FUSE_NORM_QUANT = 1, FLLM_FUSE_ROPE_KV = 2, FLLM_FUSE_GLU_Q8 = 3, FLLM_FUSE_COUNT = 4 };
bool fllm_fuse_enabled(int which);
void fllm_fuse_force(int which, int on);
#if !defined(__HIPCC__)
// x += scale * branch (res_kind 1: FP32; 2: FP16 plus optional bias), out = rms_norm(x), and the activations of the next GEMM quantized
// (quant 0 none, 1 Marlin FP16 in q0, 2 NVFP4 in q0 / q1, 3 int8 in q0 / q1). Returns 1 without launching when the shape is unsupported.
int fllm_add_rms_norm_launch(cudaStream_t s, float *x, const void *branch, int res_kind, const float *rbias, float scale, const float *w,
                             float *out, int rows, int cols, float eps, int quant, void *q0, void *q1);
int fllm_rope_kv_f32_launch(cudaStream_t s, float *q, const float *k, const float *v, uint16_t *pool_k, uint16_t *pool_v,
                            const int32_t *pos, const int32_t *slots, const float *cs, const float *sn, int rows, int heads,
                            int kv_heads, int dk, int dv, int half_, int neox);
void fllm_quantize_x8_glu_launch(cudaStream_t s, const float *gate, const float *up, int mode, int8_t *xq, float2 *xs, size_t groups);
// may the NVFP4 GEMM of this batch read activations quantized elsewhere into its scratch? / they now are there (nvfp4.cu)
bool fllm_nvfp4_prep_ok(int cols, int batch);
void fllm_nvfp4_scratch_filled();
#endif
// would fllm_mmvq_try run this K-quant product (int8 activations) for this batch? (mmvq.cu; always false for HIP)
bool fllm_mmvq_prep_ok(int kind, int cols, int batch);
int fllm_residual_launch(cudaStream_t s, float *x, const float *res, size_t count, float scale);
int fllm_split_q_gate_launch(cudaStream_t s, const float *qg, float *q, float *gate, int rows, int heads, int dk, int per_head);
int fllm_sigmoid_mul_launch(cudaStream_t s, float *x, const float *g, size_t n);
// Gated DeltaNet layer (deltanet.cu)
int fllm_delta_supported(int kh, int vh, int hk, int hv, int ksz);
int fllm_delta_launch(cudaStream_t s, const void *runs, int nruns, int max_n, int kh, int vh, int hk, int hv, int ksz, int tiled,
                      float beta_scale, float eps, size_t state_off, size_t rec_bytes, const float *qkv, const float *z, const float *beta,
                      const float *alpha, const float *convw, const float *dt_bias, const float *decay, const float *norm, float *mixed,
                      float *out);
int fllm_rope_launch(cudaStream_t s, float *q, float *k, const int32_t *pos, const float *cs, const float *sn, int rows,
                     int heads, int kv_heads, int dim, int half_, int neox);
int fllm_kv_write_launch(cudaStream_t s, const float *src, uint16_t *pool, const int32_t *slots, int rows, int kv,
                         int kind, int group, int scale_base_words);
int fllm_attention_launch(cudaStream_t s, const float *q, const uint16_t *kpool, const uint16_t *vpool, float *out,
                          const int64_t *tables, const int64_t *lengths, const float *sinks, const float *alibi,
                          int rows, int heads, int kv_heads, int dk, int dv, int stride, int block_size, int kind,
                          int kgroup, int vgroup, int kscale_base, int vscale_base, float scale, float softcap,
                          int window, int chunk);
int fllm_attention2_try(cudaStream_t s, const float *q, const uint16_t *kpool, const uint16_t *vpool, float *out,
                        const int64_t *tables, const int64_t *lengths, int max_len, const float *sinks,
                        const float *alibi, int rows, int heads, int kv_heads, int dk, int dv, int stride,
                        int block_size, int kind, float scale, float softcap, int window, int chunk, float *ws,
                        size_t ws_bytes, int kscale_base, int vscale_base);
int fllm_attention3_try(cudaStream_t s, const float *q, const uint16_t *kpool, const uint16_t *vpool, float *out,
                        const int64_t *tables, const int64_t *lengths, const float *sinks, int rows, int heads,
                        int kv_heads, int dk, int dv, int stride, int block_size, int kind, float scale,
                        int kscale_base, int vscale_base, int kv_len);
int fllm_flashinfer_prefill_try(cudaStream_t s, const float *q, const uint16_t *kpool, const uint16_t *vpool, float *out,
                                const int64_t *table, int rows, int kv_len, int heads, int kv_heads, int dk, int dv,
                                int block_size, float scale);
int fllm_qgemm6k_try(cudaStream_t s, const void *W, const float *X, float *Y, const float *bias, int rows, int cols,
                     int batch, void *x16, size_t x16_bytes);
void fllm_convert_x16(cudaStream_t s, const float *X, void *x16, int batch, int cols);
int fllm_mmvq_try(cudaStream_t s, const void *W, int kind, const float *X, float *Y, const float *bias, int rows,
                  int cols, int batch, void *scratch, size_t scratch_bytes);
int fllm_nvfp4_try(cudaStream_t s, const void *W, const float *X, float *Y, const float *bias, float gscale, int rows,
                   int cols, int batch, void *scratch, size_t scratch_bytes, int quantize);
// 1 when fllm_nvfp4_try computes this product with the CUTLASS GEMM (acc != 0: and can add it to y), else 0
int fllm_nvfp4_cutlass_ok(const float *X, const float *Y, const float *bias, int rows, int cols, int batch, int acc);
int fllm_qgemm5k_try(cudaStream_t s, const void *W, const float *X, float *Y, const float *bias, int rows, int cols,
                     int batch, void *x16, size_t x16_bytes);
int fllm_qgemm8_try(cudaStream_t s, const void *W, const float *X, float *Y, const float *bias, int rows, int cols,
                    int batch, void *x16, size_t x16_bytes);
int fllm_qgemm_leg_try(cudaStream_t s, const void *W, int kind, const float *X, float *Y, const float *bias, int rows,
                       int cols, int batch, void *x16, size_t x16_bytes);
int fllm_attention4_try(cudaStream_t s, const float *q, const uint16_t *kpool, const uint16_t *vpool, float *out,
                        const int64_t *tables, const int64_t *lengths, int max_len, const float *sinks,
                        const float *alibi, int rows, int heads, int kv_heads, int dk, int dv, int stride,
                        int block_size, int kind, float scale, float softcap, int window, int chunk, float *ws,
                        size_t ws_bytes, int kscale_base, int vscale_base);

// Tile-shape autotuning. The best tile configuration of a tensor-core GEMM depends on how many tiles there are
// compared with the number of SMs (wave quantization), so the first call with a new (kernel, shape, batch bucket)
// times every candidate on the real data and remembers the winner. `run(i)` launches candidate i on `s` and returns
// non-zero when it handled the call. FLLM_CUDA_NO_AUTOTUNE=1 keeps candidate 0, the static heuristic.
template <typename F>
static inline int fllm_autotune(cudaStream_t s, int tag, int rows, int cols, int batch, int candidates, F &&run) {
    static const bool off = getenv("FLLM_CUDA_NO_AUTOTUNE") != nullptr, log = getenv("FLLM_CUDA_TUNE_LOG") != nullptr;
    if (off || candidates <= 1) return 0;
    static std::map<uint64_t, int> best;
    static std::mutex mu;
    const uint64_t key = ((uint64_t)tag << 56) ^ ((uint64_t)rows << 40) ^ ((uint64_t)cols << 24) ^ (uint64_t)((batch + 63) / 64);
    {
        std::lock_guard<std::mutex> lock(mu);
        auto it = best.find(key);
        if (it != best.end()) return it->second;
    }
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    int pick = 0;
    float best_ms = 1e30f;
    for (int c = 0; c < candidates; c++) {
        cudaGetLastError();
        if (!run(c)) continue;  // warm-up (also configures the kernel's shared memory)
        if (cudaGetLastError() != cudaSuccess) continue;  // e.g. more shared memory than the device offers
        cudaEventRecord(a, s);
        run(c);
        run(c);
        cudaEventRecord(b, s);
        cudaEventSynchronize(b);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        if (log) fprintf(stderr, "[autotune %02x %dx%d b%d] candidate %d: %.1f us\n", tag, rows, cols, batch, c, ms * 500.f);
        if (ms < best_ms) {
            best_ms = ms;
            pick = c;
        }
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    cudaGetLastError();
    std::lock_guard<std::mutex> lock(mu);
    best[key] = pick;
    return pick;
}

// Q4_K prompt GEMM through the Marlin kernel (marlin_q4k.cu)
int fllm_marlin_q4k_try(cudaStream_t s, const void *W, const float *X, float *Y, const float *bias,
                        int rows, int cols, int batch, void *x16, size_t x16_bytes);

// 4-bit group-quantized weights through the Marlin GEMM of vLLM (marlin_vllm_glue.cu)
// set by fllm_cuda_q4z_accumulate: the next Marlin 4-bit GEMM adds scale * (result + bias) to y instead of overwriting it
extern thread_local bool fllm_q4z_acc;
extern thread_local float fllm_q4z_acc_scale;
// prefill: with fllm_q4z_defer_acc set the accumulate mode does not launch its add kernel; fllm_q4z_deferred reports that and fllm_q4z_y16 where the FP16 result sits
// (the caller adds it later, possibly inside the next RMS norm, see fllm_add_rms_norm_launch)
extern thread_local bool fllm_q4z_defer_acc, fllm_q4z_deferred;
extern thread_local const void *fllm_q4z_y16;
int fllm_q4z_acc_launch(cudaStream_t s, const void *y16, const float *bias, float *out, int rows, int batch, float scale);
// prompt attention (flashinfer_prefill.cu): see fllm_attn_x16 in api.cu
extern thread_local void *fllm_attn_x16;
extern thread_local bool fllm_attn_x16_written;
// set by fllm_cuda_nvfp4_accumulate: the next NVFP4 GEMM (CUTLASS path only, see fllm_nvfp4_cutlass_ok) adds fllm_fp4_acc_scale * result to y
extern thread_local bool fllm_fp4_acc;
extern thread_local float fllm_fp4_acc_scale;
// set by fllm_cuda_nvfp4_f16out: the next NVFP4 GEMM (CUTLASS only) leaves FP16 results in y (batch x rows halves)
extern thread_local bool fllm_fp4_f16out;
bool fllm_q4z_supported(int rows, int cols, int group);
size_t fllm_q4z_io_bytes(int rows, int cols, int batch);
size_t fllm_q4z_ctmp_bytes(int batch, int sms);
size_t fllm_q4z_workspace_bytes(int rows);
int fllm_q4z_launch(cudaStream_t s, int dev, int sms, const void *w, int group, const float *x, float *y, const float *bias, int rows,
                    int cols, int batch, void *io, void *ctmp, void *workspace, const float *glu_gate, const float *glu_up, int glu_mode,
                    bool reuse_x16, const int *perm);
int fllm_q4z_repack(int dev, const void *awq_words, void *out, int k, int n);
int fllm_q4z_gptq_repack(int dev, const void *words, void *out, int k, int n);

// FP8 weights through the Marlin GEMM of vLLM
bool fllm_fp8_marlin_supported(int rows, int cols);
int fllm_fp8_marlin_launch(cudaStream_t s, int dev, int sms, const void *w, const float *x, float *y, const float *bias, int rows,
                           int cols, int batch, void *io, void *ctmp, void *workspace, const float *glu_gate, const float *glu_up,
                           int glu_mode, bool reuse_x16);
int fllm_fp8_marlin_repack(int dev, const void *words, void *out, int k, int n);

// KV cache formats beyond f16 / q8_0 (kv_quant.cu)
#include "fllm_cuda.h"
int fllm_kvq_write_launch(cudaStream_t s, const float *src, uint16_t *pool, const int32_t *slots, int rows,
                          const FllmKvDesc &d);
int fllm_kvq_attention(cudaStream_t s, void *stream_key, const float *q, const uint16_t *kpool, const uint16_t *vpool, float *out,
                       const int64_t *dtables, const int64_t *dlengths, const int64_t *tables_host,
                       const int64_t *lengths_host, const float *sinks, const float *alibi, int rows, int heads, int kv_heads,
                       int dk, int dv, int stride, int block_size, const FllmKvDesc &kd, const FllmKvDesc &vd, float scale,
                       float softcap, int window, int chunk, float *ws, size_t ws_bytes, bool capturing, int capture_max_len);
// Activations of older architectures (arch_ops.cu): 0 SiLU, 1 GELU tanh, 2 GELU erf, 3 ReLU, 4 ReLU^2
int fllm_act_launch(cudaStream_t s, const float *x, float *out, size_t n, int mode);
int fllm_act_glu_launch(cudaStream_t s, const float *gate, const float *up, float *out, size_t n, int mode);
int fllm_clamp_launch(cudaStream_t s, float *x, size_t n, float limit);
// Multi-head latent attention data movement (mla.cu)
int fllm_mla_latent_launch(cudaStream_t s, const float *kv_a, float *latent, int rows, int kv_lora, int width);
int fllm_mla_assemble_launch(cudaStream_t s, const float *q_full, const float *kv_a, const float *kv, float *q, float *k, float *v, int rows,
                             int heads, int nope, int rope, int value, int kv_lora);
// Mamba-2 selective state space layer (mamba2.cu); `runs` points to device memory
int fllm_mamba2_supported(int inner, int state, int heads, int groups, int ksz, int norm_groups);
int fllm_mamba2_launch(cudaStream_t s, const void *runs, int nruns, int rows, int inner, int state, int heads, int groups, int ksz,
                       int norm_groups, float step_min, float eps, size_t state_off, size_t rec_bytes, const float *proj,
                       const float *convw, const float *convb, const float *dt_bias, const float *decay, const float *skip,
                       const float *norm, float *mixed, float *out);
// LFM2 gated short convolution (shortconv.cu); `runs` points to device memory
int fllm_shortconv_supported(int hidden, int ksz);
int fllm_shortconv_launch(cudaStream_t s, const void *runs, int nruns, int hidden, int ksz, size_t state_off, const float *proj,
                          const float *convw, const float *convb, float *out);
int fllm_softplus_mul_launch(cudaStream_t s, float *x, const float *g, size_t n, float beta);
// Mamba-1 selective state space layer (mamba1.cu); `runs` points to device memory
int fllm_ssm_conv_launch(cudaStream_t s, const void *runs, int nruns, const float *proj, int width, int off, const float *convw, const float *convb,
                         float *mixed, int channels, int ksz, size_t state_off, size_t rec_bytes);
int fllm_mamba1_supported(int inner, int state, int rank, int ksz);
int fllm_mamba1_conv_launch(cudaStream_t s, const void *runs, int nruns, int inner, int ksz, size_t state_off, size_t rec_bytes, const float *proj,
                            const float *convw, const float *convb, float *mixed);
int fllm_mamba1_prep_launch(cudaStream_t s, float *split, float *dt_in, int rows, int rank, int state, float eps);
int fllm_mamba1_scan_launch(cudaStream_t s, const void *runs, int nruns, int inner, int state, int rank, size_t state_off, const float *split,
                            const float *dt, const float *mixed, const float *proj, const float *decay, const float *skip, float *out);
#if defined(__HIPCC__)
// AMD-only kernels (matrix-core GEMM, skinny GEMM, decode attention, Q4Z): rust/engine/hip/
#include "../hip/fllm_hip.h"
#endif
