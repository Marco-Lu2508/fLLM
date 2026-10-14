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
int fllm_cuda_swiglu_oai(FllmStream *st, const float *gate, const float *up, float *out, size_t count, float alpha, float limit);
int fllm_cuda_glu(FllmStream *st, const float *gate, const float *up, float *out, size_t count, int mode);
int fllm_cuda_residual(FllmStream *st, float *x, const float *res, size_t count, float scale);
int fllm_cuda_rope(FllmStream *st, float *q, float *k, const int32_t *positions, int rows, const float *cs,
                   const float *sn, int heads, int kv_heads, int dim, int half_, int neox);
int fllm_cuda_kv_write(FllmStream *st, const float *src, uint16_t *pool_layer, const int32_t *slots, int rows, int kv,
                       int kind, int group, int scale_base_words);
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

#ifdef __cplusplus
}
#endif
