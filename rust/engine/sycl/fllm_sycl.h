// SPDX-License-Identifier: Apache-2.0
// fLLM Intel GPU backend (SYCL / oneAPI): C interface of libfllmsycl, called from rust/engine/src/backend/sycl/ffi.rs.
//
// Conventions (same as the CUDA backend, rust/engine/cuda/fllm_cuda.h):
//  - every function returns 0 on success and a nonzero code on failure; `fllm_sycl_last_error()` then holds the message
//    (thread local); functions returning a pointer return NULL on failure;
//  - pointers named `dev`/not marked `host` are SYCL USM device pointers (sycl::malloc_device) of the stream's device;
//  - activations are f32, weights keep a packed GGUF block layout (see weights.cpp), the KV cache uses the shared 16-bit
//    word layout described in rust/engine/src/backend/mod.rs (DeviceKvCache);
//  - small per-call arrays (positions, cache slots, block tables, lengths, gather indices) are passed as HOST pointers and
//    staged by the library; the host memory only has to stay valid until the call returns;
//  - all operations are enqueued on the stream's in-order queue; only `*_sync`, `download`, `argmax` wait.
// Verification level: L0 (compiles with the DPC++ front end; never run on an Intel GPU).
#ifndef FLLM_SYCL_H
#define FLLM_SYCL_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct FllmSyclStream FllmSyclStream;

// Mirror of the Rust struct `DeviceProps` (ffi.rs); only append fields at the end.
typedef struct FllmSyclDeviceInfo {
    char name[256];
    char driver[128];
    char arch[64];            // sycl::ext::oneapi::experimental::architecture as text, "" when unknown
    uint64_t total_memory;    // bytes of global memory
    uint64_t free_memory;     // bytes currently free, 0 when the driver does not report it
    int32_t compute_units;    // sycl::info::device::max_compute_units
    int32_t eu_count;         // ext_intel_gpu_eu_count, 0 when unknown
    int32_t max_wg_size;
    int32_t local_mem_bytes;
    int32_t is_gpu;
    int32_t is_integrated;    // host_unified_memory (iGPU sharing system RAM)
    int32_t has_fp16;
    int32_t has_sg16;         // sub-group size 16 supported (required by every kernel)
    int32_t xmx_n16;          // joint_matrix 8x16x16 half->float with sub-group 16 (PVC, Xe2: Battlemage, Lunar Lake)
    int32_t xmx_n8;           // joint_matrix 8x8x16 half->float with sub-group 8 (Arc Alchemist/DG2)
    int32_t level_zero;       // device is driven through the Level Zero backend
    uint32_t vendor_id;
    uint64_t max_alloc;       // largest single allocation (Level Zero defaults to 4 GiB, see UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS)
} FllmSyclDeviceInfo;

// ---- runtime, memory, streams -------------------------------------------------------------------------------------
const char* fllm_sycl_last_error(void);
// Number of usable GPU devices; -1 when the SYCL runtime reports none or fails (the error text says why).
int fllm_sycl_device_count(void);
int fllm_sycl_device_info(int dev, FllmSyclDeviceInfo* out);
int fllm_sycl_malloc(int dev, void** ptr, size_t bytes);
int fllm_sycl_free(int dev, void* ptr);
int fllm_sycl_h2d_sync(int dev, void* dst, const void* src, size_t bytes);
int fllm_sycl_memset(int dev, void* dst, int value, size_t bytes);
// Allow the fp16 matrix-engine (XMX) kernels for large batches (default on); off = exact FP32 arithmetic everywhere.
int fllm_sycl_set_exact(int on);
int fllm_sycl_is_exact(void);
FllmSyclStream* fllm_sycl_stream_create(int dev);
void fllm_sycl_stream_destroy(FllmSyclStream* st);
int fllm_sycl_stream_sync(FllmSyclStream* st);
int fllm_sycl_upload(FllmSyclStream* st, void* dst, const void* src_host, size_t bytes);
int fllm_sycl_download(FllmSyclStream* st, void* dst_host, const void* src, size_t bytes);
int fllm_sycl_copy(FllmSyclStream* st, void* dst, const void* src, size_t bytes);
int fllm_sycl_zero(FllmSyclStream* st, void* ptr, size_t bytes);

// ---- weight layouts (host side, no device access) ------------------------------------------------------------------
// GGUF type ids: 0 F32, 1 F16, 30 BF16, 2 Q4_0, 3 Q4_1, 6 Q5_0, 7 Q5_1, 8 Q8_0, 10 Q2_K, 11 Q3_K, 12 Q4_K, 13 Q5_K, 14 Q6_K,
// 20 IQ4_NL, 23 IQ4_XS (stored as IQ4_NL), 39 MXFP4. Bytes of a rows x cols weight in the device layout; 0 = unsupported
// kind or a width that is not a multiple of the block size.
size_t fllm_sycl_weight_bytes(int kind, size_t rows, size_t cols);
// Bytes of one row in the device layout (0 = unsupported); the Rust side needs it for expert tables.
size_t fllm_sycl_row_bytes(int kind, size_t cols);
// Bytes of the GGUF encoding of rows x cols values (0 = unsupported).
size_t fllm_sycl_raw_bytes(int kind, size_t rows, size_t cols);
// Convert the GGUF bytes `raw` into the device layout `out` (fllm_sycl_weight_bytes bytes, host memory).
int fllm_sycl_pack_weight(int kind, size_t rows, size_t cols, const void* raw, size_t raw_bytes, void* out);

// ---- operations ----------------------------------------------------------------------------------------------------
// y[b][r] = dot(x[b], w[r]) + bias[r]; `kind` as above (IQ4_XS weights use 20), bias may be NULL.
int fllm_sycl_linear(FllmSyclStream* st, const void* w, int kind, int rows, int cols, const float* x, float* y,
                     const float* bias, int batch);
// out[slot] = W[ids[slot]] x[slot / x_div]; `tab` = device array of device addresses of the experts' weights, `btab` the
// same for their f32 biases (NULL = none).
int fllm_sycl_moe_linear(FllmSyclStream* st, const void* const* tab, const float* const* btab, int kind, const int32_t* ids,
                         const float* x, float* y, int rows, int cols, int slots, int x_div);
// out[i] = row ids[i] (host int64) of the weight (rows of `cols` values), dequantized to f32.
int fllm_sycl_embedding(FllmSyclStream* st, const void* w, int kind, const int64_t* ids_host, int n, float* out, int cols);
int fllm_sycl_rms_norm(FllmSyclStream* st, const float* x, const float* w, float* out, int rows, int cols, float eps);
int fllm_sycl_layer_norm(FllmSyclStream* st, const float* x, const float* w, const float* b, float* out, int rows, int cols,
                         float eps);
// mode 0 = SwiGLU, 1 = GeGLU (tanh form)
int fllm_sycl_glu(FllmSyclStream* st, const float* gate, const float* up, float* out, size_t count, int mode);
int fllm_sycl_swiglu_oai(FllmSyclStream* st, const float* gate, const float* up, float* out, size_t count, float alpha,
                         float limit);
int fllm_sycl_residual(FllmSyclStream* st, float* x, const float* res, size_t count, float scale);
int fllm_sycl_rope(FllmSyclStream* st, float* q, float* k, const int32_t* pos_host, int rows, const float* cos_dev,
                   const float* sin_dev, int heads, int kv_heads, int dim, int half_rot, int neox);
// kind 1 = f16 pool, 2 = q8 pool (bytes then f16 scales at 16-bit word `scale_base_words`)
int fllm_sycl_kv_write(FllmSyclStream* st, const float* src, uint16_t* pool, const int32_t* slots_host, int rows, int kv,
                       int kind, int group, int scale_base_words);
int fllm_sycl_attention(FllmSyclStream* st, const float* q, const uint16_t* kpool, const uint16_t* vpool, float* out,
                        const int64_t* tables_host, size_t table_entries, const int64_t* lengths_host, int rows,
                        const float* sinks_dev, const float* alibi_dev, int heads, int kv_heads, int dk, int dv, int stride,
                        int block_size, int kind, int kgroup, int vgroup, int kscale_base, int vscale_base, float scale,
                        float softcap, int window, int chunk);
int fllm_sycl_argmax(FllmSyclStream* st, const float* logits, int rows, int vocab, int32_t* host_out);
int fllm_sycl_gather_rows(FllmSyclStream* st, float* dst, const float* src, const int32_t* idx_host, const float* w_host,
                          int n, int row_len);
int fllm_sycl_scatter_add_rows(FllmSyclStream* st, float* dst, const float* src, const int32_t* idx_host,
                               const float* w_host, int n, int row_len);
int fllm_sycl_moe_route(FllmSyclStream* st, const float* logits, int32_t* ids, float* w, int rows, int experts, int k,
                        int normalize, float scale);
int fllm_sycl_moe_combine(FllmSyclStream* st, float* out, const float* y, const float* w, int tokens, int k, int hidden);
int fllm_sycl_split_q_gate(FllmSyclStream* st, const float* qg, float* q, float* gate, int rows, int heads, int dk,
                           int per_head);
int fllm_sycl_sigmoid_mul(FllmSyclStream* st, float* x, const float* g, size_t n);

#ifdef __cplusplus
}
#endif
#endif
