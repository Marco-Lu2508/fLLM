// fLLM CUDA backend: C API (see api.cu). Mirrored by the extern block in backend/cuda.rs.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FllmStream FllmStream;

const char *fllm_cuda_last_error(void);
int fllm_cuda_device_count(void);
int fllm_cuda_device_info(int dev, char *name, size_t name_len, int *cc_major, int *cc_minor, size_t *free_bytes,
                          size_t *total_bytes, int *sm_count);
int fllm_cuda_malloc(int dev, void **ptr, size_t bytes);
int fllm_cuda_free(void *ptr);
int fllm_cuda_h2d_sync(int dev, void *dst, const void *src, size_t bytes);
int fllm_cuda_memset(int dev, void *ptr, int value, size_t bytes);

FllmStream *fllm_cuda_stream_create(int dev);
void fllm_cuda_stream_destroy(FllmStream *st);
int fllm_cuda_stream_sync(FllmStream *st);
int fllm_cuda_stream_wait_stream(FllmStream *waiter, FllmStream *target);
int fllm_cuda_upload(FllmStream *st, void *dst, const void *src, size_t bytes);
int fllm_cuda_download(FllmStream *st, void *dst, const void *src, size_t bytes);
int fllm_cuda_copy(FllmStream *st, void *dst, const void *src, size_t bytes);

int fllm_cuda_linear(FllmStream *st, const void *w, int kind, const float *x, float *y, const float *bias, int rows,
                     int cols, int batch);
int fllm_cuda_linear_nvfp4(FllmStream *st, const void *w, const float *x, float *y, const float *bias, float gscale,
                           int rows, int cols, int batch);
int fllm_cuda_embedding(FllmStream *st, const void *w, int kind, const int64_t *ids, int n, float *out, int cols);
int fllm_cuda_rms_norm(FllmStream *st, const float *x, const float *w, float *out, int rows, int cols, float eps);
int fllm_cuda_layer_norm(FllmStream *st, const float *x, const float *w, const float *b, float *out, int rows, int cols,
                         float eps);
int fllm_cuda_repack(FllmStream *st, void *dst, const void *src, int kind, size_t rows, size_t cols);
int fllm_cuda_gather_rows(FllmStream *st, float *dst, const float *src, const int32_t *idx, const float *w, int n, int row_len);
int fllm_cuda_scatter_add_rows(FllmStream *st, float *dst, const float *src, const int32_t *idx, const float *w, int n, int row_len);
int fllm_cuda_zero(FllmStream *st, void *ptr, size_t bytes);
int fllm_cuda_host_register(int dev, void *ptr, size_t bytes);
int fllm_cuda_host_unregister(void *ptr);
int fllm_cuda_moe_route(FllmStream *st, const float *logits, int32_t *ids, float *w, int rows, int experts, int k,
                        int normalize, float scale);
int fllm_cuda_moe_combine(FllmStream *st, float *out, const float *y, const float *w, int tokens, int k, int hidden);
int fllm_cuda_moe_linear(FllmStream *st, const void *const *tab, const void *const *btab, int kind, const int32_t *ids,
                         const float *x, float *y, int rows, int cols, int slots, int xdiv);
int fllm_cuda_moe_route2(FllmStream *st, const float *logits, const float *bias, int32_t *ids, float *w, int rows, int experts, int k,
                         int sigmoid, int normalize, float scale, float eps, int groups, int groups_used, int top2);
int fllm_cuda_moe_align(FllmStream *st, const int32_t *ids, int32_t *sorted, int32_t *block_expert, int32_t *num_post, int pairs,
                        int expert_offset, int num_local, int block);
int fllm_cuda_moe_grouped_supported(int kind, int cols, int block);
int fllm_cuda_moe_grouped(FllmStream *st, int kind, int glu, const void *const *wtab, const void *const *wtab2, const void *const *btab,
                          const void *const *btab2, const float *x, const float *pw, float *y, const int32_t *sorted,
                          const int32_t *block_expert, const int32_t *num_post, int rows, int cols, int pairs, int xdiv, int block,
                          int max_blocks, int glu_mode, float alpha, float limit);
int fllm_cuda_moe_combine2(FllmStream *st, float *out, const float *y, const float *w, const float *shared, const float *gate, int tokens,
                           int k, int hidden);
int fllm_cuda_moe_gate_scale(FllmStream *st, float *gate, const float *x, const float *g, int tokens, int hidden);
int fllm_cuda_swiglu_oai(FllmStream *st, const float *gate, const float *up, float *out, size_t count, float alpha, float limit);
int fllm_cuda_glu(FllmStream *st, const float *gate, const float *up, float *out, size_t count, int mode);
int fllm_cuda_residual(FllmStream *st, float *x, const float *res, size_t count, float scale);
int fllm_cuda_rope(FllmStream *st, float *q, float *k, const int32_t *positions, int rows, const float *cs,
                   const float *sn, int heads, int kv_heads, int dim, int half_, int neox);
int fllm_cuda_kv_write(FllmStream *st, const float *src, uint16_t *pool_layer, const int32_t *slots, int rows, int kv,
                       int kind, int group, int scale_base_words);
// Fused decode kernels (fused.cu). They return 3 without launching anything when the fusion is off or does not apply to the call, 0 after
// the launch, anything else is an error. quant: what the GEMM that follows reads (0 nothing, 1 Marlin FP16, 2 NVFP4, 3 int8 for a
// Q4_K / Q5_K / Q6_K weight of type `kind`).
int fllm_cuda_add_rms_norm(FllmStream *st, float *x, const float *branch, float scale, const float *w, float *out, int rows, int cols,
                           float eps, int quant, int kind, const void *next_w, int next_rows);
int fllm_cuda_linear_q4z_add_norm(FllmStream *st, const void *w, int group, const float *act, float *res, const float *bias, int rows,
                                  int cols, int batch, const int *perm, float scale, const float *nw, float *out, float eps, int quant,
                                  int kind);
int fllm_cuda_rope_kv_f32(FllmStream *st, float *q, const float *k, const float *v, uint16_t *pool_k_layer, uint16_t *pool_v_layer,
                          const int32_t *positions, const int32_t *slots, int rows, const float *cs, const float *sn, int heads,
                          int kv_heads, int dk, int dv, int half_, int neox);
// which: 0 add + norm, 1 norm + quantization, 2 rotary + cache write, 3 gated activation + int8 quantization; on 1 / 0 forces, -1 = environment
int fllm_cuda_set_fuse(int which, int on);
int fllm_cuda_attention(FllmStream *st, const float *q, const uint16_t *kpool_layer, const uint16_t *vpool_layer,
                        float *out, const int64_t *tables_host, size_t tables_len, const int64_t *lengths_host,
                        int rows, const float *sinks, const float *alibi, int heads, int kv_heads, int dk, int dv,
                        int stride, int block_size, int kind, int kgroup, int vgroup, int kscale_base,
                        int vscale_base, float scale, float softcap, int window, int chunk);

// KV cache layout of one pool (src/kv_format.rs): format code, values per slot, values per scale group, head width,
// log2(group) (-1: one group per head) and the 16-bit-word offsets of the planes relative to the layer start.
typedef struct FllmKvDesc {
    int fmt, kv, group, dim, gshift, gps;  // gps: scale groups per slot (kv / group)
    int64_t hi_off, scale_off, min_off;
} FllmKvDesc;
int fllm_cuda_kvq_write(FllmStream *st, const float *src, uint16_t *pool_layer, const int32_t *slots, int rows,
                        const FllmKvDesc *desc);
int fllm_cuda_kvq_attention(FllmStream *st, const float *q, const uint16_t *kpool_layer, const uint16_t *vpool_layer,
                            float *out, const int64_t *tables_host, size_t tables_len, const int64_t *lengths_host,
                            int rows, const float *sinks, const float *alibi, int heads, int kv_heads, int dk, int dv,
                            int stride, int block_size, const FllmKvDesc *kd, const FllmKvDesc *vd, float scale,
                            float softcap, int window, int chunk);

int fllm_cuda_attn_encoder(FllmStream *st, const float *q, const float *k, const float *v, float *out,
                           const int32_t *cu_seqlens_host, int batch_size, int total_tokens,
                           int heads, int kv_heads, int head_dim, float scale, int window,
                           const int32_t *global_tokens_host, int num_global);
int fllm_cuda_tree_attention(FllmStream *st, const float *q, const uint16_t *kpool_layer,
                             const uint16_t *vpool_layer, float *out, const int64_t *tables,
                             const int64_t *lengths, const uint8_t *tree_mask, int prefix_len,
                             int num_nodes, int rows, int heads, int kv_heads, int dk, int dv,
                             int stride, int block_size, int kind, float scale, float softcap);
int fllm_cuda_spec_verify(FllmStream *st, const int32_t *draft, const int32_t *target,
                          int num_draft, int32_t *out_accepted, int32_t *out_next,
                          int64_t *seq_length);

#ifdef __cplusplus
}
#endif
