// fLLM CUDA backend: the lightning indexer and sparse attention of DeepSeek-V3.2 / GLM-5 (the reference is `dsa_select` and
// `attention_selected` of rust_decoder.rs).
//
//   dsa_keys    store[slot[r]] = k[r]                  the indexer keys of the new tokens go to a per-slot f32 store
//   dsa_scores  score[r][p] = sum_h relu(dot(q[r][h], store[slot(r, p)]) * scale) * w[r][h] * head_scale   for p < lengths[r]
//   dsa_topk    sel[r] = the `topk` best positions of row r in ascending order (ties: lower position first), -1 padded; a row with
//               at most `topk` tokens lists all of them
//   dsa_attn    attention of (row, head) over the listed positions of an f16 KV cache
//
// FP32 CUDA-core code with block-level primitives only (no warp intrinsics), so the same file builds for AMD GPUs.
#include "common.cuh"
#include "fllm_cuda_internal.h"

#define DSA_TOPK_THREADS 512
#define DSA_ATTN_THREADS 128

__global__ void dsa_keys_kernel(float *__restrict__ store, const float *__restrict__ k, const int32_t *__restrict__ slots, int rows, int dim) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)rows * dim; i += (size_t)gridDim.x * blockDim.x) {
        const size_t r = i / dim, d = i % dim;
        store[(size_t)slots[r] * dim + d] = k[i];
    }
}

// grid (ceil(max_len / blockDim), rows); dynamic shared memory: heads * dim query values and heads weights of the row
__global__ void dsa_scores_kernel(const float *__restrict__ q, const float *__restrict__ w, const float *__restrict__ store, const int64_t *__restrict__ tables,
                                  const int64_t *__restrict__ lengths, float *__restrict__ scores, size_t score_stride, int heads, int dim, int stride,
                                  int block_size, float scale, float head_scale) {
    extern __shared__ float dsa_sm[];
    float *qs = dsa_sm;
    float *ws = dsa_sm + (size_t)heads * dim;
    const int r = blockIdx.y;
    for (int i = threadIdx.x; i < heads * dim; i += blockDim.x) qs[i] = q[(size_t)r * heads * dim + i];
    for (int i = threadIdx.x; i < heads; i += blockDim.x) ws[i] = w[(size_t)r * heads + i];
    __syncthreads();
    const int n = (int)lengths[r];
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= n) return;
    const int64_t block = tables[(size_t)r * stride + p / block_size];
    const size_t slot = (size_t)block * block_size + p % block_size;
    const float *key = store + slot * dim;
    float score = 0.f;
    for (int h = 0; h < heads; h++) {
        float dot = 0.f;
        const float *qh = qs + (size_t)h * dim;
        for (int d = 0; d < dim; d++) dot += qh[d] * key[d];
        dot *= scale;
        if (dot > 0.f) score += dot * ws[h] * head_scale;
    }
    scores[(size_t)r * score_stride + p] = score;
}

// float ordered like its value as an unsigned integer (a larger float is a larger key)
__device__ __forceinline__ uint32_t dsa_key(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// inclusive prefix sum over the threads of the block; `buf` holds blockDim.x ints
__device__ __forceinline__ int dsa_scan(int v, int *buf) {
    const int t = threadIdx.x;
    buf[t] = v;
    __syncthreads();
    for (int off = 1; off < (int)blockDim.x; off <<= 1) {
        const int add = t >= off ? buf[t - off] : 0;
        __syncthreads();
        buf[t] += add;
        __syncthreads();
    }
    const int out = buf[t];
    __syncthreads();
    return out;
}

// one block per row
__global__ void dsa_topk_kernel(const float *__restrict__ scores, size_t score_stride, const int64_t *__restrict__ lengths, int32_t *__restrict__ sel, int topk) {
    __shared__ int hist[256];
    __shared__ int scan_buf[DSA_TOPK_THREADS];
    __shared__ uint32_t s_prefix, s_mask;
    __shared__ int s_remaining, s_base_eq, s_base_acc;
    const int r = blockIdx.x, t = threadIdx.x;
    const int n = (int)lengths[r];
    int32_t *out = sel + (size_t)r * topk;
    if (n <= topk) {
        for (int j = t; j < topk; j += blockDim.x) out[j] = j < n ? j : -1;
        return;
    }
    const float *sc = scores + (size_t)r * score_stride;
    if (t == 0) {
        s_prefix = 0;
        s_mask = 0;
        s_remaining = topk;
    }
    __syncthreads();
    // radix select of the topk-th largest key, eight bits at a time
    for (int pass = 0; pass < 4; pass++) {
        const int shift = 24 - 8 * pass;
        for (int i = t; i < 256; i += blockDim.x) hist[i] = 0;
        __syncthreads();
        const uint32_t prefix = s_prefix, mask = s_mask;
        for (int p = t; p < n; p += blockDim.x) {
            const uint32_t key = dsa_key(sc[p]);
            if ((key & mask) == prefix) atomicAdd(&hist[(key >> shift) & 255u], 1);
        }
        __syncthreads();
        if (t == 0) {
            int cum = 0, bin = 255;
            for (; bin > 0; bin--) {
                if (cum + hist[bin] >= s_remaining) break;
                cum += hist[bin];
            }
            s_remaining -= cum;
            s_prefix = prefix | ((uint32_t)bin << shift);
            s_mask = mask | (255u << shift);
        }
        __syncthreads();
    }
    const uint32_t threshold = s_prefix;
    const int need_eq = s_remaining;  // keys equal to the threshold that are taken (the lowest positions)
    if (t == 0) {
        s_base_eq = 0;
        s_base_acc = 0;
    }
    __syncthreads();
    for (int p0 = 0; p0 < n; p0 += blockDim.x) {
        const int p = p0 + t;
        const uint32_t key = p < n ? dsa_key(sc[p]) : 0u;
        const int gt = p < n && key > threshold;
        const int eq = p < n && key == threshold;
        const int eq_incl = dsa_scan(eq, scan_buf);
        const int rank = s_base_eq + eq_incl - eq;
        const int acc = gt || (eq && rank < need_eq);
        const int acc_incl = dsa_scan(acc, scan_buf);
        if (acc) out[s_base_acc + acc_incl - 1] = p;
        __syncthreads();
        if (t == (int)blockDim.x - 1) {
            s_base_eq += eq_incl;
            s_base_acc += acc_incl;
        }
        __syncthreads();
    }
}

__device__ __forceinline__ float dsa_block_max(float v, float *red) {
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
__device__ __forceinline__ float dsa_block_sum(float v, float *red) {
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

// grid (heads, rows); dynamic shared memory: dk query values, topk scores, topk slots, DSA_ATTN_THREADS reduction values
__global__ void dsa_attn_kernel(const float *__restrict__ q, const uint16_t *__restrict__ kpool, const uint16_t *__restrict__ vpool, float *__restrict__ out,
                                const int32_t *__restrict__ sel, const int64_t *__restrict__ tables, const float *__restrict__ sinks, int topk, int heads,
                                int kv_heads, int dk, int dv, int stride, int block_size, float scale) {
    extern __shared__ float dsa_sm2[];
    float *qs = dsa_sm2;
    float *sc = qs + dk;
    int *slot_s = reinterpret_cast<int *>(sc + topk);
    float *red = reinterpret_cast<float *>(slot_s + topk);
    const int h = blockIdx.x, r = blockIdx.y, t = threadIdx.x;
    const int kh = h / (heads / kv_heads);
    const int32_t *list = sel + (size_t)r * topk;
    for (int i = t; i < dk; i += blockDim.x) qs[i] = q[((size_t)r * heads + h) * dk + i];
    int local = 0;
    for (int j = t; j < topk; j += blockDim.x) local += list[j] >= 0;
    const int count = (int)dsa_block_sum((float)local, red);
    for (int j = t; j < count; j += blockDim.x) {
        const int p = list[j];
        slot_s[j] = (int)(tables[(size_t)r * stride + p / block_size] * block_size + p % block_size);
    }
    __syncthreads();
    float m = -3.0e38f;
    for (int j = t; j < count; j += blockDim.x) {
        const uint16_t *key = kpool + (size_t)slot_s[j] * kv_heads * dk + (size_t)kh * dk;
        float dot = 0.f;
        for (int d = 0; d < dk; d++) dot += qs[d] * h2f_bits(key[d]);
        sc[j] = dot * scale;
        m = fmaxf(m, sc[j]);
    }
    // a sink adds exp(sink) to the denominator without a value
    const float sink = sinks ? sinks[h] : -3.0e38f;
    m = dsa_block_max(fmaxf(m, t == 0 ? sink : -3.0e38f), red);
    float l = (sinks && t == 0) ? expf(sink - m) : 0.f;
    for (int j = t; j < count; j += blockDim.x) {
        sc[j] = expf(sc[j] - m);
        l += sc[j];
    }
    l = dsa_block_sum(l, red);
    for (int d = t; d < dv; d += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < count; j++) acc += sc[j] * h2f_bits(vpool[(size_t)slot_s[j] * kv_heads * dv + (size_t)kh * dv + d]);
        out[((size_t)r * heads + h) * dv + d] = acc / l;
    }
}

// out[g][(d + shift) % dim] = x[g][d]: the last `shift` values of every group move to the front (the tail rotation of the Hy4 indexer heads)
__global__ void rotate_heads_kernel(const float *__restrict__ x, float *__restrict__ out, size_t count, int dim, int shift) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x) {
        const size_t g = i / dim;
        const int d = (int)(i % dim);
        out[g * dim + (d + shift) % dim] = x[i];
    }
}

static unsigned dsa_grid(size_t n) {
    size_t g = (n + 255) / 256;
    return (unsigned)(g > 65535 ? 65535 : (g == 0 ? 1 : g));
}

int fllm_rotate_heads_launch(cudaStream_t s, const float *x, float *out, size_t count, int dim, int shift) {
    if (count == 0) return 0;
    rotate_heads_kernel<<<dsa_grid(count), 256, 0, s>>>(x, out, count, dim, shift);
    return fllm_check_launch("rotate_heads");
}

int fllm_dsa_keys_launch(cudaStream_t s, float *store, const float *k, const int32_t *slots, int rows, int dim) {
    if (rows <= 0) return 0;
    dsa_keys_kernel<<<dsa_grid((size_t)rows * dim), 256, 0, s>>>(store, k, slots, rows, dim);
    return fllm_check_launch("dsa_keys");
}

// the largest query block (heads * dim floats) a scoring block keeps in shared memory
#define DSA_MAX_SCORE_SMEM 48000

int fllm_dsa_scores_supported(int heads, int dim) { return ((size_t)heads * dim + heads) * sizeof(float) <= DSA_MAX_SCORE_SMEM; }

int fllm_dsa_select_launch(cudaStream_t s, const float *q, const float *w, const float *store, const int64_t *tables, const int64_t *lengths, float *scores,
                           size_t score_stride, int32_t *sel, int rows, int max_len, int heads, int dim, int topk, int stride, int block_size, float scale,
                           float head_scale) {
    if (rows <= 0) return 0;
    if (!fllm_dsa_scores_supported(heads, dim)) return fllm_set_error("dsa_select: %d heads x %d values do not fit shared memory", heads, dim);
    if (max_len > topk) {
        const size_t smem = ((size_t)heads * dim + heads) * sizeof(float);
        dim3 grid((max_len + 127) / 128, rows);
        dsa_scores_kernel<<<grid, 128, smem, s>>>(q, w, store, tables, lengths, scores, score_stride, heads, dim, stride, block_size, scale, head_scale);
        if (int e = fllm_check_launch("dsa_scores")) return e;
    }
    dsa_topk_kernel<<<rows, DSA_TOPK_THREADS, 0, s>>>(scores, score_stride, lengths, sel, topk);
    return fllm_check_launch("dsa_topk");
}

int fllm_dsa_attention_supported(int topk, int dk) { return (size_t)(dk + 2 * topk + DSA_ATTN_THREADS) * sizeof(float) <= 48000; }

int fllm_dsa_attention_launch(cudaStream_t s, const float *q, const uint16_t *kpool, const uint16_t *vpool, float *out, const int32_t *sel,
                              const int64_t *tables, const float *sinks, int rows, int topk, int heads, int kv_heads, int dk, int dv, int stride, int block_size,
                              float scale) {
    if (rows <= 0) return 0;
    if (!fllm_dsa_attention_supported(topk, dk)) return fllm_set_error("dsa_attention: top-k %d with key width %d does not fit shared memory", topk, dk);
    const size_t smem = (size_t)(dk + 2 * topk + DSA_ATTN_THREADS) * sizeof(float);
    dim3 grid(heads, rows);
    dsa_attn_kernel<<<grid, DSA_ATTN_THREADS, smem, s>>>(q, kpool, vpool, out, sel, tables, sinks, topk, heads, kv_heads, dk, dv, stride, block_size, scale);
    return fllm_check_launch("dsa_attention");
}
