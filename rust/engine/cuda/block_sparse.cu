// fLLM CUDA backend: block-sparse attention of MiniMax-M3 (the reference is `minimax3_layers` of rust_decoder/minimax3.rs).
//
// A small indexer scores the keys of a sequence with `heads` query heads and a single key per token; the scores are pooled per block of `block` keys
// (maximum), the `local` newest blocks are forced in, and the best `topk` blocks (ties: the lower block) are kept for every indexer head. Attention
// head h reads the blocks of indexer head h / (heads_attention / heads_indexer).
//
//   block_scores  scores[(r * heads + h) * stride + b] = max over the keys j of block b of dot(q[r][h], key_j); +inf for the local blocks
//   (the selection itself is `dsa_topk_kernel` of dsa.cu over rows * heads rows of block scores)
//   block_attn    attention of (row, head) over the keys of the listed blocks of an f16 KV cache
//
// FP32 CUDA-core code with block-level primitives only, so the same file builds for AMD GPUs.
#include <math.h>
#include "common.cuh"
#include "fllm_cuda_internal.h"

#define BS_THREADS 128

// grid (blocks of the longest row, heads, rows of this launch); dynamic shared memory: dim query values
__global__ void __launch_bounds__(BS_THREADS)
block_scores_kernel(const float *__restrict__ q, const float *__restrict__ store, const int64_t *__restrict__ tables, const int64_t *__restrict__ lengths,
                    float *__restrict__ scores, size_t score_stride, int row0, int heads, int dim, int block, int local, int stride, int block_size) {
    extern __shared__ float bs_q[];
    __shared__ float red[BS_THREADS];
    const int b = blockIdx.x, h = blockIdx.y, rl = blockIdx.z, r = row0 + rl;
    const int n = (int)lengths[r];
    const int nb = (n + block - 1) / block;
    if (b >= nb) return;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) bs_q[i] = q[((size_t)r * heads + h) * dim + i];
    __syncthreads();
    float best = -INFINITY;
    const int first = b * block, last = min(first + block, n);
    for (int p = first + threadIdx.x; p < last; p += blockDim.x) {
        const int64_t page = tables[(size_t)r * stride + p / block_size];
        const float *key = store + ((size_t)page * block_size + p % block_size) * dim;
        float dot = 0.f;
        if ((dim & 3) == 0) {
            float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
            const float4 *q4 = reinterpret_cast<const float4 *>(bs_q);
            const float4 *k4 = reinterpret_cast<const float4 *>(key);
            for (int d = 0; d < dim / 4; d++) {
                const float4 qv = q4[d], kv = k4[d];
                a0 += qv.x * kv.x;
                a1 += qv.y * kv.y;
                a2 += qv.z * kv.z;
                a3 += qv.w * kv.w;
            }
            dot = (a0 + a1) + (a2 + a3);
        } else {
            for (int d = 0; d < dim; d++) dot += bs_q[d] * key[d];
        }
        best = fmaxf(best, dot);
    }
    red[threadIdx.x] = best;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if ((int)threadIdx.x < off) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + off]);
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        const int qb = (n - 1) / block;
        // the newest `local` blocks are always kept
        const float v = (b <= qb && qb - b < local) ? INFINITY : red[0];
        scores[((size_t)rl * heads + h) * score_stride + b] = v;
    }
}

__device__ __forceinline__ float bs_block_max(float v, float *red) {
    const int t = threadIdx.x;
    red[t] = v;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (t < off) red[t] = fmaxf(red[t], red[t + off]);
        __syncthreads();
    }
    const float out = red[0];
    __syncthreads();
    return out;
}

__device__ __forceinline__ float bs_block_sum(float v, float *red) {
    const int t = threadIdx.x;
    red[t] = v;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (t < off) red[t] += red[t + off];
        __syncthreads();
    }
    const float out = red[0];
    __syncthreads();
    return out;
}

// grid (heads, rows); dynamic shared memory: dk query values, npos scores, npos slots, BS_THREADS reduction values (npos = topk * block)
__global__ void __launch_bounds__(BS_THREADS)
block_attn_kernel(const float *__restrict__ q, const uint16_t *__restrict__ kpool, const uint16_t *__restrict__ vpool, float *__restrict__ out,
                  const int32_t *__restrict__ sel, const int64_t *__restrict__ tables, const int64_t *__restrict__ lengths, int topk, int groups, int block,
                  int heads, int kv_heads, int dk, int dv, int stride, int block_size, float scale) {
    extern __shared__ float bs_sm[];
    const int npos = topk * block;
    float *qs = bs_sm;
    float *sc = qs + dk;
    int *slot_s = reinterpret_cast<int *>(sc + npos);
    float *red = reinterpret_cast<float *>(slot_s + npos);
    const int h = blockIdx.x, r = blockIdx.y, t = threadIdx.x;
    const int kh = h / (heads / kv_heads);
    const int g = h / (heads / groups);
    const int32_t *list = sel + ((size_t)r * groups + g) * topk;
    const int n = (int)lengths[r];
    for (int i = t; i < dk; i += blockDim.x) qs[i] = q[((size_t)r * heads + h) * dk + i];
    // blocks listed (ascending, -1 padded)
    int cnt = 0;
    while (cnt < topk && list[cnt] >= 0) cnt++;
    const int span = cnt * block;
    for (int j = t; j < span; j += blockDim.x) {
        const int p = list[j / block] * block + j % block;
        slot_s[j] = p < n ? (int)(tables[(size_t)r * stride + p / block_size] * block_size + p % block_size) : -1;
    }
    __syncthreads();
    float m = -3.0e38f;
    for (int j = t; j < span; j += blockDim.x) {
        if (slot_s[j] < 0) {
            sc[j] = -3.0e38f;
            continue;
        }
        const uint16_t *key = kpool + (size_t)slot_s[j] * kv_heads * dk + (size_t)kh * dk;
        float dot = 0.f;
        for (int d = 0; d < dk; d++) dot += qs[d] * h2f_bits(key[d]);
        sc[j] = dot * scale;
        m = fmaxf(m, sc[j]);
    }
    m = bs_block_max(m, red);
    float l = 0.f;
    for (int j = t; j < span; j += blockDim.x) {
        sc[j] = slot_s[j] < 0 ? 0.f : expf(sc[j] - m);
        l += sc[j];
    }
    l = bs_block_sum(l, red);
    for (int d = t; d < dv; d += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < span; j++) {
            if (slot_s[j] >= 0) acc += sc[j] * h2f_bits(vpool[(size_t)slot_s[j] * kv_heads * dv + (size_t)kh * dv + d]);
        }
        out[((size_t)r * heads + h) * dv + d] = l > 0.f ? acc / l : 0.f;
    }
}

int fllm_block_sparse_supported(int heads, int dim, int topk, int block, int dk) {
    return heads > 0 && dim > 0 && topk > 0 && block > 0 && (size_t)dim * sizeof(float) <= 48000 && (size_t)(dk + 2 * (size_t)topk * block + BS_THREADS) * sizeof(float) <= 48000;
}

int fllm_block_scores_launch(cudaStream_t s, const float *q, const float *store, const int64_t *tables, const int64_t *lengths, float *scores, size_t score_stride,
                             int row0, int rows, int blocks, int heads, int dim, int block, int local, int stride, int block_size) {
    if (rows <= 0) return 0;
    dim3 grid(blocks, heads, rows);
    block_scores_kernel<<<grid, BS_THREADS, (size_t)dim * sizeof(float), s>>>(q, store, tables, lengths, scores, score_stride, row0, heads, dim, block, local, stride,
                                                                           block_size);
    return fllm_check_launch("block_scores");
}

int fllm_block_attention_launch(cudaStream_t s, const float *q, const uint16_t *kpool, const uint16_t *vpool, float *out, const int32_t *sel, const int64_t *tables,
                                const int64_t *lengths, int rows, int topk, int groups, int block, int heads, int kv_heads, int dk, int dv, int stride,
                                int block_size, float scale) {
    if (rows <= 0) return 0;
    if (!fllm_block_sparse_supported(groups, 1, topk, block, dk)) return fllm_set_error("block_attention: top-k %d blocks of %d keys with key width %d do not fit shared memory", topk, block, dk);
    const size_t smem = (size_t)(dk + 2 * (size_t)topk * block + BS_THREADS) * sizeof(float);
    dim3 grid(heads, rows);
    block_attn_kernel<<<grid, BS_THREADS, smem, s>>>(q, kpool, vpool, out, sel, tables, lengths, topk, groups, block, heads, kv_heads, dk, dv, stride, block_size, scale);
    return fllm_check_launch("block_attention");
}
