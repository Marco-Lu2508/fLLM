// fLLM CUDA backend: declarations shared between the kernel files and the C API.
#pragma once
int fllm_attn_encoder_launch(cudaStream_t s, const float *q, const float *k, const float *v, float *out,
                             const int32_t *cu_seqlens, int batch_size, int total_tokens,
                             int heads, int kv_heads, int head_dim, float scale, int window,
                             const int32_t *global_tokens, int num_global);

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
// MLX affine group-quantized weights (mlx_affine.cu): `kind` is the MLXA type id (0x80010000 | bf16 << 12 | (group / 32) << 4 | bits), `W` the
// device image of backend/mlx.rs
int fllm_mlx_launch(cudaStream_t s, const void *W, int kind, const float *X, float *Y, const float *bias, int rows, int cols, int batch);
inline bool fllm_kind_is_mlx(int kind) { return ((uint32_t)kind & ~0x1fffu) == 0x80010000u; }
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
// where the activation of that GLU belongs (the buffer the caller passed as X): a GEMM that cannot fold the GLU into its conversion computes it there first
extern thread_local float *fllm_glu_dst;
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
int fllm_rms_norm_grouped_launch(cudaStream_t s, const float *x, const float *w, float *out, int rows, int cols, int groups, float eps);
int fllm_layer_norm_launch(cudaStream_t s, const float *x, const float *w, const float *b, float *out, int rows,
                           int cols, float eps);
int fllm_gather_rows_launch(cudaStream_t s, float *dst, const float *src, const int32_t *idx, const float *w, int n, int row_len);
int fllm_scatter_add_rows_launch(cudaStream_t s, float *dst, const float *src, const int32_t *idx, const float *w, int n, int row_len);
int fllm_moe_linear_launch(cudaStream_t s, const void *const *tab, int kind, const int32_t *ids, const float *X, float *Y,
                           int rows, int cols, int slots, int xdiv, const void *const *btab, void *scratch, size_t scratch_bytes);
int fllm_moe_route_launch(cudaStream_t s, const float *logits, int32_t *ids, float *w, int rows, int experts, int k,
                          int normalize, float scale);
int fllm_moe_combine_launch(cudaStream_t s, float *out, const float *y, const float *w, int tokens, int k, int hidden);
// Device-side mixture-of-experts (moe.cu): general routing, expert-major sort, grouped tensor-core GEMM, combination with the shared expert
int fllm_moe_route2_launch(cudaStream_t s, const float *logits, const float *bias, int32_t *ids, float *w, int rows, int experts, int k,
                           int sigmoid, int normalize, float scale, float eps, int groups, int groups_used, int top2);
int fllm_moe_align_launch(cudaStream_t s, const int32_t *ids, int32_t *sorted, int32_t *block_expert, int32_t *num_post, int pairs,
                          int expert_offset, int num_local, int block);
int fllm_moe_grouped_supported(uint32_t kind, int cols, int block);
int fllm_moe_grouped_launch(cudaStream_t s, uint32_t kind, int glu, const void *const *wtab, const void *const *wtab2, const void *const *btab,
                            const void *const *btab2, const float *X, const float *pw, float *Y, const int32_t *sorted,
                            const int32_t *block_expert, const int32_t *num_post, int rows, int cols, int pairs, int xdiv, int block,
                            int max_blocks, int glu_mode, float alpha, float limit);
int fllm_moe_combine2_launch(cudaStream_t s, float *out, const float *y, const float *w, const float *shared, const float *gate, int tokens,
                             int k, int hidden);
int fllm_moe_gate_scale_launch(cudaStream_t s, float *gate, const float *x, const float *g, int tokens, int hidden);
int fllm_moe_zero_fold_launch(cudaStream_t s, float *zs, int32_t *ids, float *w, int rows, int k, int routed, int rewrite);
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
                             float *out, int rows, int cols, float eps, int quant, void *q0, void *q1, const float *gsp = nullptr);
int fllm_rope_kv_f32_launch(cudaStream_t s, float *q, const float *k, const float *v, uint16_t *pool_k, uint16_t *pool_v,
                            const int32_t *pos, const int32_t *slots, const float *cs, const float *sn, int rows, int heads,
                            int kv_heads, int dk, int dv, int half_, int neox);
void fllm_quantize_x8_glu_launch(cudaStream_t s, const void *gate, const void *up, int mode, int8_t *xq, float2 *xs, size_t groups, bool half_in = false);
// may the NVFP4 GEMM of this batch read activations quantized elsewhere into its scratch? / they now are there (nvfp4.cu)
bool fllm_nvfp4_prep_ok(int cols, int batch);
// May the fused norm quantize for the NVFP4 GEMM of weight `w` (rows x cols)? *gs: device pointer of the weight's static activation factor
// (nullptr = 1). False for FLLM_NVFP4_ACT_SCALE=dynamic (the factor needs the amax of the whole tensor) or without the weight.
bool fllm_nvfp4_prep_scale(const void *w, int rows, int cols, const float **gs);
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
// Kimi Delta Attention (KDA) layer (kda.cu)
int fllm_kda_supported(int heads, int head_dim, int ksz);
int fllm_copy_cols_launch(cudaStream_t s, float *dst, const float *src, int rows, int width, int dst_stride, int dst_col);
int fllm_kda_launch(cudaStream_t s, const void *runs, int nruns, int max_n, int heads, int head_dim, int ksz,
                    int bounded, float lower_bound, float eps, size_t state_off, size_t rec_bytes,
                    const float *qkv, const float *forget, const float *beta, const float *gate,
                    const float *convw, const float *dt_bias, const float *rate, const float *norm,
                    float *mixed, float *out);
// RWKV-6 / RWKV-7 (rwkv.cu)
int fllm_rwkv_supported(int hidden, int head_size);
int fllm_rwkv_shift_launch(cudaStream_t s, const void *runs, int nruns, int total_rows, int hidden, size_t shift_off, const float *x, float *shift);
int fllm_rwkv_lerp_launch(cudaStream_t s, const float *x, const float *shift, const float *mix, const float *delta, float *out, size_t count, int hidden);
int fllm_rwkv_act_launch(cudaStream_t s, float *x, const float *gate, size_t count, int mode);
int fllm_rwkv7_launch(cudaStream_t s, const void *runs, int nruns, int heads, int head_size, int hidden, float eps, size_t state_off,
                      const float *r, const float *k, const float *v, const float *w, const float *a, const float *vres, const float *gate,
                      const float *norm_w, const float *norm_b, const float *k_k, const float *k_a, const float *r_k, float *v_first,
                      float *out);
int fllm_rwkv6_launch(cudaStream_t s, const void *runs, int nruns, int heads, int head_size, int hidden, float eps, size_t state_off,
                      const float *r, const float *k, const float *v, const float *w, const float *gate, const float *norm_w,
                      const float *norm_b, const float *first, float *out);
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
// Hopper / FlashInfer attention variants (fa3_prefill.cu, fi_attn.cu): -1 = not covered, the caller continues with the next path
bool fllm_fa3_enabled();
bool fllm_fa3_built();
int fllm_attn_mode(int which);
extern thread_local int fllm_attn_last_path;
int fllm_fa3_prefill_try(cudaStream_t s, const float *q, const uint16_t *kpool, const uint16_t *vpool, float *out, const int64_t *table, int rows, int kv_len,
                         int heads, int kv_heads, int dk, int dv, int block_size, float scale, int window,
                         float softcap = 0.f, const float *sinks = nullptr, int kind = 1);
int fllm_fa3_prefill_runs(cudaStream_t s, const float *q, const uint16_t *kpool, const uint16_t *vpool, float *out, int nreq, const int32_t *qo,
                          const int32_t *kvp, const int32_t *kv_len, const int32_t *pages, int heads, int kv_heads, int dk, int dv, int block_size,
                          float scale, int window);
int fllm_fi_attention_try(cudaStream_t s, const float *q, const uint16_t *kpool, const uint16_t *vpool, float *out, const int64_t *tables_host,
                          const int64_t *lengths_host, int rows, const float *sinks, bool has_alibi, int heads, int kv_heads, int dk, int dv, int stride,
                          int block_size, int kind, float scale, float softcap, int window, int chunk);
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
// FLLM_ACT_DTYPE=f16: the q and out pointers of fllm_attention3_try / fllm_flashinfer_prefill_try are FP16 arrays (flashinfer_prefill.cu)
extern thread_local bool fllm_attn_half_io;
// FLLM_ACT_DTYPE: activation type (0 FP32, 1 FP16, 2 BF16) of the q and out arrays of the next fllm_attention4_try (attention4.cu)
extern thread_local int fllm_attn_act_dt;
// set by fllm_cuda_nvfp4_accumulate: the next NVFP4 GEMM (CUTLASS path only, see fllm_nvfp4_cutlass_ok) adds fllm_fp4_acc_scale * result to y
extern thread_local bool fllm_fp4_acc;
extern thread_local float fllm_fp4_acc_scale;
// set by fllm_cuda_nvfp4_f16out: the next NVFP4 GEMM (CUTLASS only) leaves FP16 results in y (batch x rows halves)
extern thread_local bool fllm_fp4_f16out;
// FLLM_FUSE_NVFP4_ATTN_OUT: the FP16 result of a FlashInfer prompt attention that has not been expanded to the FP32 buffer `out` yet. The next NVFP4
// CUTLASS GEMM that reads `out` quantizes the halves directly (fllm_x_half); any other operation expands them first (flush_glu in api.cu).
struct FllmPendingAttn {
    const void *o16 = nullptr;
    float *out = nullptr;
    size_t count = 0;
};
extern thread_local FllmPendingAttn fllm_pending_attn;
// set by fllm_cuda_linear_nvfp4 around the CUTLASS call: the activations are these FP16 values instead of the FP32 ones in x
extern thread_local const void *fllm_x_half;
bool fllm_fuse_nvfp4_attn_out();
bool fllm_fuse_nvfp4_norm();
int fllm_flashinfer_expand_half(cudaStream_t s, const void *o16, float *out, size_t count);
// RMSNorm + activation quantization for the CUTLASS NVFP4 GEMM (nvfp4_cutlass.cu); mode as in fllm_nvfp4_cutlass_check
int fllm_nvfp4_norm_quant_try(cudaStream_t s, const float *x, const float *w, float *out, int rows, int cols, float eps, const void *weight,
                          int weight_rows, int mode);
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
                       int norm_groups, int norm_mode, float step_min, float eps, size_t state_off, size_t rec_bytes, const float *proj,
                       const float *convw, const float *convb, const float *dt_bias, const float *decay, const float *skip,
                       const float *norm, float *mixed, float *out);
// LFM2 gated short convolution (shortconv.cu); `runs` points to device memory
int fllm_shortconv_supported(int hidden, int ksz);
int fllm_shortconv_launch(cudaStream_t s, const void *runs, int nruns, int hidden, int ksz, size_t state_off, const float *proj,
                          const float *convw, const float *convb, float *out);
int fllm_softplus_mul_launch(cudaStream_t s, float *x, const float *g, size_t n, float beta);
int fllm_head_gate_launch(cudaStream_t s, float *x, const float *g, size_t n, size_t dv, float beta);
int fllm_scale_launch(cudaStream_t s, float *x, size_t n, float factor);
int fllm_glu_ext_launch(cudaStream_t s, const float *gate, const float *up, float *out, size_t n, int kind, float p0, float p1);
// Hyper-connections (hc.cu)
int fllm_hc_supported(int hc);
int fllm_hc_mix_launch(cudaStream_t s, const float *streams, const float *mixes, const float *base, const float *scale, float *post, float *comb,
                       float *collapsed, int rows, int hc, int hidden, int mode, int iters, float eps, float magnitude);
int fllm_hc_combine_launch(cudaStream_t s, float *streams, const float *post, const float *comb, const float *block, int rows, int hc, int hidden);
// DeepSeek-V4 attention (v4.cu)
int fllm_v4_supported(int head_dim, int rope, int window, int heads, int o_groups);
int fllm_v4_rope_launch(cudaStream_t s, float *x, const float *cs, int rows, int groups, int dim, int rope, float sign);
int fllm_v4_compress_launch(cudaStream_t s, float *state, const float *kv, const float *gate, const float *pos, const float *norm, const float *cs, int row0, int n,
                            int position0, int dim, int coff, int ratio, int rope, float eps, size_t off_buf_kv, size_t off_buf_gate, size_t off_ov_kv,
                            size_t off_ov_gate, size_t off_entries);
int fllm_v4_attention_launch(cudaStream_t s, float *out, const float *q, const float *kv, const float *state, const float *sinks, const int32_t *sel,
                             const float *cs, int row0, int n, int position0, int heads, int hd, int rope, int window, int ratio, int topk, int o_groups,
                             int rows_total, size_t off_ring, size_t off_entries, float scale);
int fllm_v4_ring_launch(cudaStream_t s, float *state, const float *kv, int row0, int n, int position0, int hd, int window, size_t off_ring);
// DeepSeek-V3.2 / GLM-5 lightning indexer and sparse attention (dsa.cu)
int fllm_rotate_heads_launch(cudaStream_t s, const float *x, float *out, size_t count, int dim, int shift);
int fllm_dsa_keys_launch(cudaStream_t s, float *store, const float *k, const float *gate, const int32_t *slots, int rows, int dim, int width);
int fllm_dsa_pool_supported(int dim, int kpool);
int fllm_dsa_scores_supported(int heads, int dim);
int fllm_dsa_select_launch(cudaStream_t s, const float *q, const float *w, const float *store, const float *ape, const int64_t *tables, const int64_t *lengths,
                           float *scores, size_t score_stride, int32_t *sel, int32_t *psel, int rows, int max_len, int heads, int dim, int width, int topk,
                           int kpool, int always_tail, int stride, int block_size, float scale, float head_scale);
int fllm_dsa_attention_supported(int topk, int dk);
int fllm_dsa_attention_launch(cudaStream_t s, const float *q, const uint16_t *kpool, const uint16_t *vpool, float *out, const int32_t *sel,
                              const int64_t *tables, const float *sinks, int rows, int topk, int heads, int kv_heads, int dk, int dv, int stride, int block_size,
                              float scale);
int fllm_dsa_topk_launch(cudaStream_t s, const float *scores, size_t score_stride, const int64_t *lengths, int32_t *sel, int rows, int topk, int len_div);
// MiniMax-M3 block-sparse attention (block_sparse.cu)
int fllm_block_sparse_supported(int heads, int dim, int topk, int block, int dk);
int fllm_block_scores_launch(cudaStream_t s, const float *q, const float *store, const int64_t *tables, const int64_t *lengths, float *scores, size_t score_stride,
                             int row0, int rows, int blocks, int heads, int dim, int block, int local, int stride, int block_size);
int fllm_block_attention_launch(cudaStream_t s, const float *q, const uint16_t *kpool, const uint16_t *vpool, float *out, const int32_t *sel, const int64_t *tables,
                                const int64_t *lengths, int rows, int topk, int groups, int block, int heads, int kv_heads, int dk, int dv, int stride,
                                int block_size, float scale);
// Inkling convolutions and relative-position attention (inkling.cu)
int fllm_conv_resid_supported(int channels, int taps);
int fllm_conv_resid_launch(cudaStream_t s, const void *runs, int nruns, int channels, int taps, size_t state_off, const float *x, const float *convw, float *out);
int fllm_rel_attention_supported(int hd, int d_rel);
int fllm_rel_attention_launch(cudaStream_t s, const float *q, const float *rel, const float *profile, const uint16_t *kpool, const uint16_t *vpool, float *out,
                              const int64_t *tables, const int64_t *lengths, int rows, int heads, int kv_heads, int hd, int d_rel, int extent, int window,
                              int log_floor, float log_alpha, float scale, int stride, int block_size);
// Qwen3.8-Flash-Next (qwen4.cu)
int fllm_q4_supported(int hc, int kernel, int dilation);
int fllm_q4_pool_supported(int dim, int rope_half);
int fllm_q4_act_launch(cudaStream_t s, float *x, size_t n, int mode, float scale, float factor);
int fllm_q4_gate_mix_launch(cudaStream_t s, float *out, const float *normed, const float *gates, int rows, int hc, int hidden);
int fllm_q4_ple_gate_launch(cudaStream_t s, float *out, const float *key, const float *query, const float *value, int rows, int hc, int hidden);
int fllm_q4_ple_conv_launch(cudaStream_t s, const void *runs, int nruns, int width, int kernel, int dilation, size_t state_off, const float *gated,
                            const float *normed, const float *convw, float *out);
int fllm_q4_pool_keys_launch(cudaStream_t s, float *out, const float *k, const float *store, const float *norm, const float *cs, const float *sn,
                             const int32_t *meta, int rows, int dim, int width, int ratio, int rope_half, int neox, float eps);
// Native EXL2 / EXL3 projections (exl.cu); `kind` is the GGUF type id (0x80000030 / 0x80000031), `w` the blob of rust/gguf-probe/src/exl.rs. EXL3 needs
// fllm_exl3_scratch_bytes(rows, cols, batch) of device scratch (EXL2 none).
size_t fllm_exl3_scratch_bytes(int rows, int cols, int batch);
int fllm_exl_launch(cudaStream_t s, int kind, const void *w, const float *x, float *y, const float *bias, int rows, int cols, int batch,
                    void *scratch, size_t scratch_bytes);
// Mamba-1 selective state space layer (mamba1.cu); `runs` points to device memory
int fllm_ssm_conv_launch(cudaStream_t s, const void *runs, int nruns, const float *proj, int width, int off, const float *convw, const float *convb,
                         float *mixed, int channels, int ksz, size_t state_off, size_t rec_bytes);
int fllm_mamba1_supported(int inner, int state, int rank, int ksz);
int fllm_mamba1_conv_launch(cudaStream_t s, const void *runs, int nruns, int inner, int ksz, size_t state_off, size_t rec_bytes, const float *proj,
                            const float *convw, const float *convb, float *mixed);
int fllm_mamba1_prep_launch(cudaStream_t s, float *split, float *dt_in, int rows, int rank, int state, float eps, const float *w_dt, const float *w_b,
                            const float *w_c);
int fllm_mamba1_scan_launch(cudaStream_t s, const void *runs, int nruns, int inner, int state, int rank, size_t state_off, const float *split,
                            const float *dt, const float *mixed, const float *proj, const float *decay, const float *skip, float *out);
#if !defined(__HIPCC__)
// Activations in FP16 / BF16 (act_narrow.cu, FLLM_ACT_DTYPE). dt: 0 FP32, 1 FP16, 2 BF16. Buffers are arrays of that type; weights, norm weights, biases
// and rotary tables stay FP32.
int fllm_na_convert_launch(cudaStream_t s, const void *src, int src_dt, void *dst, int dst_dt, size_t n);
int fllm_na_add_rms_norm_launch(cudaStream_t s, int dt, void *x, const void *branch, float scale, const float *w, void *out, int rows, int cols, float eps);
int fllm_na_residual_launch(cudaStream_t s, int dt, void *x, const void *res, size_t count, float scale);
int fllm_na_glu_launch(cudaStream_t s, int dt, const void *gate, const void *up, void *out, size_t count, int mode);
int fllm_na_bias_launch(cudaStream_t s, int dt, void *y, const float *bias, int rows, int batch);
int fllm_na_rope_launch(cudaStream_t s, int dt, void *q, void *k, const int32_t *pos, const float *cs, const float *sn, int rows, int heads, int kv_heads,
                         int dim, int half_, int neox);
int fllm_na_kv_write_launch(cudaStream_t s, int dt, const void *src, uint16_t *pool, const int32_t *slots, int rows, int kv);
int fllm_na_rope_kv_launch(cudaStream_t s, int dt, void *q, const void *k, const void *v, uint16_t *pool_k, uint16_t *pool_v, const int32_t *pos,
                            const int32_t *slots, const float *cs, const float *sn, const float *qw, const float *kw, float eps, int rows, int heads,
                            int kv_heads, int dk, int dv, int half_, int neox);
int fllm_na_rope_kv_strided_launch(cudaStream_t s, int dt, void *q_dst, const void *q_src, int q_stride, const void *k, int k_stride, const void *v, int v_stride,
                                   uint16_t *pool_k, uint16_t *pool_v, const int32_t *pos, const int32_t *slots, const float *cs, const float *sn, const float *qw,
                                   const float *kw, const float *qb, const float *kb, const float *vb, float eps, int rows, int heads, int kv_heads, int dk,
                                   int dv, int half_, int neox);
int fllm_na_split3_launch(cudaStream_t s, int dt, const void *src, void *d0, void *d1, void *d2, int rows, int n0, int n1, int n2);
int fllm_na_glu_merged_launch(cudaStream_t s, int dt, const void *gu, void *out, int rows, int ff, int mode);
int fllm_na_quantize_x8_launch(cudaStream_t s, int dt, const void *x, int8_t *xq, float2 *xs, size_t groups);
int fllm_na_check_launch(cudaStream_t s, int dt, const void *x, size_t count, int *out);
// Marlin GEMM with FP16 (bf16 = 0) or BF16 activations, scales and result (marlin_vllm/marlin_vllm.cu); -1 when no kernel of that type was built
extern "C" int fllm_vllm_marlin_gemm_t(const void *A, const void *B, void *C, void *C_tmp, const void *scales, const void *zeros, const void *bias,
                                       const void *global_scale, int m, int n, int k, int lda, void *workspace, int kind, int group_size, int dev, int sms,
                                       cudaStream_t stream, int bf16);
#endif
#if defined(__HIPCC__)
// AMD-only kernels (matrix-core GEMM, skinny GEMM, decode attention, Q4Z): rust/engine/hip/
#include "../hip/fllm_hip.h"
#endif

// Sampling on the device (sampling.cu): `params` is an array of `rows` FllmSampleRow (80 bytes each) in device memory; the lists they
// index (penalty tokens and counts, bias, grammar bitmask) and the outputs are device arrays. `work` receives the processed logits
// (it may equal `logits` to work in place).
int fllm_sample_launch(cudaStream_t s, const float *logits, float *work, int rows, int vocab, const void *params, const int32_t *pen_ids,
                       const int32_t *pen_counts, const int32_t *bias_ids, const float *bias_vals, const uint32_t *mask, int mask_words,
                       int32_t *out_ids, float *out_logprob, int32_t *top_ids, float *top_logprob, int top_stride);

int fllm_tree_attention_launch(cudaStream_t s, const float *q, const uint16_t *kpool_layer,
                               const uint16_t *vpool_layer, float *out, const int64_t *tables,
                               const int64_t *lengths, const uint8_t *tree_mask, int prefix_len,
                               int num_nodes, int rows, int heads, int kv_heads, int dk, int dv,
                               int stride, int block_size, int kind, float scale, float softcap);
int fllm_spec_verify_launch(cudaStream_t s, const int32_t *draft, const int32_t *target,
                            int num_draft, int32_t *out_accepted, int32_t *out_next,
                            int64_t *seq_length);
