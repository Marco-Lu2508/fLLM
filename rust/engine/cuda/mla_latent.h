// fLLM CUDA backend: latent-cache multi-head latent attention (mla_latent.cu, mla_decode.cu). Declarations for api.cu and hip_stubs.cu.
//
// The cache holds, per token and layer, one row of kv_lora + rope values: [normalized latent | rotated shared rope key]. The attention runs in the
// latent space (the up projection of the keys is absorbed into the queries, the up projection of the values is applied to the attention output).
#pragma once
#include <stddef.h>
#include <stdint.h>
#if defined(__HIPCC__)
#include "hip_compat.h"
#else
#include <cuda_runtime.h>
#endif
#include "fllm_cuda.h"

// 1 when the kernels cover these dimensions and cache format (KvFormat code) on the current device.
int fllm_mla_latent_supported(int kv_lora, int rope, int nope, int value, int kv_fmt);
// k_row[r] = [latent[r] | rope(kv_a[r][kv_lora..])] (rotary embedding at pos[r], tables cs / sn as in rope_kernel); the row the cache stores.
int fllm_mla_kv_launch(cudaStream_t s, const float *kv_a, const float *latent, const int32_t *pos, const float *cs, const float *sn,
                       float *k_row, int rows, int kv_lora, int rope, int neox);
// q_lat[r][h] = [q_nope[r][h] * W_UK[h] | rope(q_pe[r][h])] in FP16; w_kvb: the dense FP16 kv_b matrix [heads * (nope + value)][kv_lora].
int fllm_mla_query_launch(cudaStream_t s, const float *q_full, const uint16_t *w_kvb, const int32_t *pos, const float *cs, const float *sn,
                          uint16_t *q_lat, int rows, int heads, int nope, int rope, int value, int kv_lora, int neox);
// out[r][h * value + v] = sum_c o_lat[r][h][c] * W_UV[h][v][c] (FP32 result).
int fllm_mla_output_launch(cudaStream_t s, const uint16_t *o_lat, const uint16_t *w_kvb, float *out, int rows, int heads, int nope, int value,
                           int kv_lora);
// Attention of `rows` query rows over the latent cache of one layer (`pool`: layer start, `kd`: its layout): o_lat[r][h] (FP16, kv_lora wide).
// The tables / lengths are device copies; the host copies decide how rows are grouped (runs of one prompt vs. independent sequences).
// mode 0: tensor-core kernels, 1: CUDA-core reference kernel. Returns 0 on success, -1 when the shape is not covered, > 0 on an error.
int fllm_mla_attention_launch(cudaStream_t s, const uint16_t *q_lat, const uint16_t *pool, uint16_t *o_lat, const int64_t *dtables,
                              const int64_t *dlengths, const int64_t *htables, const int64_t *hlengths, int rows, int heads, int kv_lora,
                              int rope, int stride, int block_size, const FllmKvDesc *kd, float scale, int mode);
