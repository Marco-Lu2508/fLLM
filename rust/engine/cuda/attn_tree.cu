// fLLM CUDA backend: tree attention over the device KV cache for speculative draft trees.
#include "common.cuh"
#include "fllm_cuda.h"
#include "fllm_cuda_internal.h"
#include <cuda_fp16.h>

#define ATT_WARPS 4
#define ATT_MAX_D 512

struct TreeAttnArgs {
    const float *q;
    const uint16_t *kpool, *vpool;
    float *out;
    const int64_t *tables, *lengths;
    const uint8_t *tree_mask; // M x M boolean matrix
    int prefix_len, num_nodes;
    int rows, heads, kv_heads, dk, dv, stride, block_size, kind;
    float scale, softcap;
};

__device__ __forceinline__ float tree_key_at(const TreeAttnArgs &a, size_t slot, int kh, int d) {
    const size_t kv = (size_t)a.kv_heads * a.dk;
    const size_t c = (size_t)kh * a.dk + d;
    return h2f_bits(a.kpool[slot * kv + c]);
}

__device__ __forceinline__ float tree_value_at(const TreeAttnArgs &a, size_t slot, int kh, int d) {
    const size_t kv = (size_t)a.kv_heads * a.dv;
    const size_t c = (size_t)kh * a.dv + d;
    return h2f_bits(a.vpool[slot * kv + c]);
}

__global__ void __launch_bounds__(ATT_WARPS * 32) tree_attention_kernel(TreeAttnArgs a) {
    __shared__ float qs[ATT_MAX_D];
    __shared__ float wm[ATT_WARPS], wl[ATT_WARPS];
    __shared__ float wacc[ATT_WARPS][ATT_MAX_D];

    const int item = blockIdx.x;
    const int r = item / a.heads, h = item % a.heads;
    const int kh = h / (a.heads / a.kv_heads);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;

    for (int d = threadIdx.x; d < a.dk; d += blockDim.x) {
        qs[d] = a.q[(size_t)item * a.dk + d];
    }
    __syncthreads();

    const int n = (int)a.lengths[r];
    const int64_t *table = a.tables + (size_t)r * a.stride;

    float max_score = -1e38f;
    float l = 0.0f;
    float acc[ATT_MAX_D / 32];
    #pragma unroll
    for (int i = 0; i < ATT_MAX_D / 32; ++i) acc[i] = 0.0f;

    for (int p = warp; p < n; p += ATT_WARPS) {
        // Ancestral tree mask check
        if (a.tree_mask != nullptr && p >= a.prefix_len && r >= a.prefix_len) {
            int qi = r - a.prefix_len;
            int ki = p - a.prefix_len;
            if (qi < a.num_nodes && ki < a.num_nodes) {
                if (!a.tree_mask[qi * a.num_nodes + ki]) {
                    continue; // masked out: not an ancestor
                }
            }
        }

        const size_t block = (size_t)table[p / a.block_size];
        const size_t slot = block * a.block_size + (p % a.block_size);

        float dot = 0.0f;
        for (int d = lane; d < a.dk; d += 32) {
            dot += qs[d] * tree_key_at(a, slot, kh, d);
        }
        #pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            dot += __shfl_xor_sync(0xffffffff, dot, mask);
        }

        float s = dot * a.scale;
        if (a.softcap > 0.0f) {
            s = a.softcap * tanhf(s / a.softcap);
        }

        float old_max = max_score;
        max_score = fmaxf(max_score, s);
        float alpha = expf(old_max - max_score);
        float w = expf(s - max_score);
        l = l * alpha + w;

        #pragma unroll
        for (int i = 0; i < a.dv / 32; ++i) {
            int d = i * 32 + lane;
            acc[i] = acc[i] * alpha + w * tree_value_at(a, slot, kh, d);
        }
    }

    if (lane == 0) {
        wm[warp] = max_score;
        wl[warp] = l;
    }
    for (int i = 0; i < a.dv / 32; ++i) {
        wacc[warp][i * 32 + lane] = acc[i];
    }
    __syncthreads();

    if (warp == 0) {
        float global_max = -1e38f;
        for (int w = 0; w < ATT_WARPS; ++w) {
            global_max = fmaxf(global_max, wm[w]);
        }
        float global_l = 0.0f;
        for (int w = 0; w < ATT_WARPS; ++w) {
            global_l += wl[w] * expf(wm[w] - global_max);
        }
        float inv_l = (global_l > 0.0f) ? (1.0f / global_l) : 0.0f;

        for (int d = lane; d < a.dv; d += 32) {
            float out_val = 0.0f;
            for (int w = 0; w < ATT_WARPS; ++w) {
                out_val += wacc[w][d] * expf(wm[w] - global_max);
            }
            a.out[(size_t)item * a.dv + d] = out_val * inv_l;
        }
    }
}

int fllm_tree_attention_launch(cudaStream_t s, const float *q, const uint16_t *kpool_layer,
                               const uint16_t *vpool_layer, float *out, const int64_t *tables,
                               const int64_t *lengths, const uint8_t *tree_mask, int prefix_len,
                               int num_nodes, int rows, int heads, int kv_heads, int dk, int dv,
                               int stride, int block_size, int kind, float scale, float softcap) {
    if (rows <= 0 || heads <= 0) return 0;
    TreeAttnArgs a;
    a.q = q;
    a.kpool = kpool_layer;
    a.vpool = vpool_layer;
    a.out = out;
    a.tables = tables;
    a.lengths = lengths;
    a.tree_mask = tree_mask;
    a.prefix_len = prefix_len;
    a.num_nodes = num_nodes;
    a.rows = rows;
    a.heads = heads;
    a.kv_heads = kv_heads;
    a.dk = dk;
    a.dv = dv;
    a.stride = stride;
    a.block_size = block_size;
    a.kind = kind;
    a.scale = scale;
    a.softcap = softcap;

    int total_items = rows * heads;
    tree_attention_kernel<<<total_items, ATT_WARPS * 32, 0, s>>>(a);
    return fllm_check_launch("tree_attention");
}
