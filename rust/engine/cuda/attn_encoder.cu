// fLLM CUDA backend: bidirectional variable-length encoder attention with sequence packing.
//
// Handles:
// - Packed sequence batches (cu_seqlens: start offsets for each sequence in the batch)
// - Full bidirectional attention or local sliding-window attention (window parameter)
// - Global attention tokens (e.g. ModernBERT CLS or unpadded tokens)
// - GQA/MHA with arbitrary heads / kv_heads
// - Numerically stable online softmax without allocating full S x S attention matrix

#include "common.cuh"
#include "fllm_cuda_internal.h"

#define ENC_WARPS 4
#define ENC_MAX_D 512

struct EncoderAttnArgs {
    const float *q;              // [total_tokens, heads * head_dim]
    const float *k;              // [total_tokens, kv_heads * head_dim]
    const float *v;              // [total_tokens, kv_heads * head_dim]
    float *out;                  // [total_tokens, heads * head_dim]
    const int32_t *cu_seqlens;   // [batch_size + 1] on device
    int batch_size;
    int total_tokens;
    int heads;
    int kv_heads;
    int head_dim;
    float scale;
    int window;                  // 0 or negative = full attention; > 0 = local window size
    const int32_t *global_tokens;// optional array of token indices that attend globally
    int num_global;
};

__device__ __forceinline__ bool is_token_global(int token_idx, const int32_t *globals, int n_globals) {
    if (n_globals <= 0 || globals == nullptr) return false;
    for (int g = 0; g < n_globals; ++g) {
        if (globals[g] == token_idx) return true;
    }
    return false;
}

__global__ void __launch_bounds__(ENC_WARPS * 32) encoder_attention_kernel(EncoderAttnArgs a) {
    __shared__ float s_q[ENC_MAX_D];
    __shared__ float wm[ENC_WARPS], wl[ENC_WARPS];
    __shared__ float wacc[ENC_WARPS][ENC_MAX_D];

    const int item = blockIdx.x; // total_tokens * heads
    if (item >= a.total_tokens * a.heads) return;

    const int token_idx = item / a.heads;
    const int h = item % a.heads;
    const int kh = h / (a.heads / a.kv_heads);
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;

    // Load query row into shared memory
    const size_t q_base = (size_t)token_idx * (a.heads * a.head_dim) + (size_t)h * a.head_dim;
    for (int d = threadIdx.x; d < a.head_dim; d += blockDim.x) {
        s_q[d] = a.q[q_base + d];
    }
    __syncthreads();

    // Find sequence b for token_idx using cu_seqlens
    int b = 0;
    while (b < a.batch_size && a.cu_seqlens[b + 1] <= token_idx) {
        b++;
    }
    const int seq_start = a.cu_seqlens[b];
    const int seq_end = a.cu_seqlens[b + 1];
    const int seq_len = seq_end - seq_start;
    const int local_i = token_idx - seq_start;

    const bool i_global = is_token_global(token_idx, a.global_tokens, a.num_global);

    float m = -1e38f;
    float l = 0.0f;
    float acc[ENC_MAX_D / 32];
    #pragma unroll
    for (int i = 0; i < ENC_MAX_D / 32; i++) acc[i] = 0.0f;

    // Iterate through all key tokens in the sequence
    for (int j = seq_start + warp; j < seq_end; j += ENC_WARPS) {
        const int local_j = j - seq_start;
        // Check window visibility:
        bool visible = true;
        if (a.window > 0 && a.window < seq_len) {
            const bool j_global = is_token_global(j, a.global_tokens, a.num_global);
            if (!i_global && !j_global) {
                int dist = local_i - local_j;
                if (dist < 0) dist = -dist;
                if (dist > (a.window / 2)) {
                    visible = false;
                }
            }
        }
        if (!visible) continue;

        // Compute dot product s_q . k[j]
        const size_t k_base = (size_t)j * (a.kv_heads * a.head_dim) + (size_t)kh * a.head_dim;
        float score = 0.0f;
        for (int d = lane; d < a.head_dim; d += 32) {
            score += s_q[d] * a.k[k_base + d];
        }
        score = warp_sum(score) * a.scale;

        // Online softmax
        const float m_prev = m;
        m = fmaxf(m, score);
        const float alpha = expf(m_prev - m);
        const float beta = expf(score - m);
        l = l * alpha + beta;

        const size_t v_base = (size_t)j * (a.kv_heads * a.head_dim) + (size_t)kh * a.head_dim;
        for (int i = 0; i < (a.head_dim + 31) / 32; i++) {
            const int d = i * 32 + lane;
            float v_val = (d < a.head_dim) ? a.v[v_base + d] : 0.0f;
            acc[i] = acc[i] * alpha + beta * v_val;
        }
    }

    // Lane 0 holds warp's m and l; accumulators are in lanes
    if (lane == 0) {
        wm[warp] = m;
        wl[warp] = l;
    }
    for (int i = 0; i < (a.head_dim + 31) / 32; i++) {
        const int d = i * 32 + lane;
        if (d < a.head_dim) {
            wacc[warp][d] = acc[i];
        }
    }
    __syncthreads();

    // Warp 0 merges results from all 4 warps
    if (warp == 0) {
        float global_m = -1e38f;
        for (int w = 0; w < ENC_WARPS; w++) {
            global_m = fmaxf(global_m, wm[w]);
        }
        float global_l = 0.0f;
        for (int w = 0; w < ENC_WARPS; w++) {
            if (wm[w] > -1e37f) {
                global_l += wl[w] * expf(wm[w] - global_m);
            }
        }
        const float inv_l = (global_l > 0.0f) ? (1.0f / global_l) : 0.0f;

        const size_t out_base = (size_t)token_idx * (a.heads * a.head_dim) + (size_t)h * a.head_dim;
        for (int d = lane; d < a.head_dim; d += 32) {
            float sum = 0.0f;
            for (int w = 0; w < ENC_WARPS; w++) {
                if (wm[w] > -1e37f) {
                    sum += wacc[w][d] * expf(wm[w] - global_m);
                }
            }
            a.out[out_base + d] = sum * inv_l;
        }
    }
}

int fllm_attn_encoder_launch(cudaStream_t s, const float *q, const float *k, const float *v, float *out,
                             const int32_t *cu_seqlens, int batch_size, int total_tokens,
                             int heads, int kv_heads, int head_dim, float scale, int window,
                             const int32_t *global_tokens, int num_global) {
    if (total_tokens <= 0 || heads <= 0 || head_dim <= 0 || head_dim > ENC_MAX_D) {
        return fllm_set_error("invalid dimensions for encoder attention");
    }
    EncoderAttnArgs a;
    a.q = q;
    a.k = k;
    a.v = v;
    a.out = out;
    a.cu_seqlens = cu_seqlens;
    a.batch_size = batch_size;
    a.total_tokens = total_tokens;
    a.heads = heads;
    a.kv_heads = kv_heads;
    a.head_dim = head_dim;
    a.scale = scale;
    a.window = window;
    a.global_tokens = global_tokens;
    a.num_global = num_global;

    int blocks = total_tokens * heads;
    encoder_attention_kernel<<<blocks, ENC_WARPS * 32, 0, s>>>(a);
    cudaError_t status = cudaGetLastError();
    if (status != cudaSuccess) {
        return fllm_set_error("encoder_attention_kernel: %s", cudaGetErrorString(status));
    }
    return 0;
}
