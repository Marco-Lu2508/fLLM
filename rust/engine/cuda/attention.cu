// fLLM CUDA backend: paged attention over the device KV cache (decode and prefill rows).
//
// One block per (query row, head). Four warps split the keys; each keeps an online-softmax state
// (max, denominator, value accumulators spread over its lanes) and the states are merged at the
// end. Handles f16 and q8_0 caches, GQA, softcap, sliding window, chunked attention, sinks, ALiBi.
#include "common.cuh"
#include "fllm_cuda_internal.h"

#define ATT_WARPS 4
#define ATT_MAX_D 512
#define ATT_FLOOR (-3.0e38f)

struct AttnArgs {
    const float *q;
    const uint16_t *kpool, *vpool;  // start of this layer, 16-bit words
    float *out;
    const int64_t *tables, *lengths;
    const float *sinks, *alibi;
    int rows, heads, kv_heads, dk, dv, stride, block_size, kind;
    int kgroup, vgroup, kscale_base, vscale_base;  // q8 only (scale bases in 16-bit words)
    float scale, softcap;
    int window, chunk;
};

__device__ __forceinline__ float key_at(const AttnArgs &a, size_t slot, int kh, int d) {
    const size_t kv = (size_t)a.kv_heads * a.dk;
    const size_t c = (size_t)kh * a.dk + d;
    if (a.kind == 1) return h2f_bits(a.kpool[slot * kv + c]);
    const int8_t q = reinterpret_cast<const int8_t *>(a.kpool)[slot * kv + c];
    return (float)q * h2f_bits(a.kpool[a.kscale_base + slot * (kv / a.kgroup) + c / a.kgroup]);
}

__device__ __forceinline__ float value_at(const AttnArgs &a, size_t slot, int kh, int d) {
    const size_t kv = (size_t)a.kv_heads * a.dv;
    const size_t c = (size_t)kh * a.dv + d;
    if (a.kind == 1) return h2f_bits(a.vpool[slot * kv + c]);
    const int8_t q = reinterpret_cast<const int8_t *>(a.vpool)[slot * kv + c];
    return (float)q * h2f_bits(a.vpool[a.vscale_base + slot * (kv / a.vgroup) + c / a.vgroup]);
}

__global__ void __launch_bounds__(ATT_WARPS * 32) attention_kernel(AttnArgs a) {
    __shared__ float qs[ATT_MAX_D];
    __shared__ float wm[ATT_WARPS], wl[ATT_WARPS];
    __shared__ float wacc[ATT_WARPS][ATT_MAX_D];
    const int item = blockIdx.x;
    const int r = item / a.heads, h = item % a.heads;
    const int kh = h / (a.heads / a.kv_heads);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int d = threadIdx.x; d < a.dk; d += blockDim.x) qs[d] = a.q[(size_t)item * a.dk + d];
    __syncthreads();

    const int n = (int)a.lengths[r];
    int start = 0;
    if (a.chunk != 0) start = (n - 1) / a.chunk * a.chunk;
    else if (a.window != 0 && n > a.window) start = n - a.window;
    const int64_t *table = a.tables + (size_t)r * a.stride;

    constexpr int NACC = ATT_MAX_D / 32;
    float acc[NACC];
#pragma unroll
    for (int i = 0; i < NACC; i++) acc[i] = 0.f;
    float m = ATT_FLOOR, l = 0.f;

    for (int p = start + warp; p < n; p += ATT_WARPS) {
        const size_t slot = (size_t)table[p / a.block_size] * a.block_size + p % a.block_size;
        float part = 0.f;
        for (int d = lane; d < a.dk; d += 32) part = fmaf(qs[d], key_at(a, slot, kh, d), part);
        float s = warp_sum(part) * a.scale;
        if (a.alibi) s += a.alibi[h] * ((float)p - (float)(n - 1));
        if (a.softcap != 0.f) s = a.softcap * tanhf(fminf(fmaxf(s / a.softcap, -15.f), 15.f));
        const float m_new = fmaxf(m, s);
        const float corr = __expf(m - m_new), e = __expf(s - m_new);
        l = l * corr + e;
#pragma unroll
        for (int i = 0; i < NACC; i++) {
            const int d = lane + 32 * i;
            if (d < a.dv) acc[i] = acc[i] * corr + e * value_at(a, slot, kh, d);
        }
        m = m_new;
    }

#pragma unroll
    for (int i = 0; i < NACC; i++) {
        const int d = lane + 32 * i;
        if (d < a.dv) wacc[warp][d] = acc[i];
    }
    if (lane == 0) { wm[warp] = m; wl[warp] = l; }
    __syncthreads();

    float M = a.sinks ? a.sinks[h] : ATT_FLOOR;
    for (int w = 0; w < ATT_WARPS; w++) M = fmaxf(M, wm[w]);
    float L = a.sinks ? __expf(a.sinks[h] - M) : 0.f;
    float f[ATT_WARPS];
    for (int w = 0; w < ATT_WARPS; w++) {
        f[w] = __expf(wm[w] - M);
        L += wl[w] * f[w];
    }
    const float inv = 1.0f / L;
    for (int d = threadIdx.x; d < a.dv; d += blockDim.x) {
        float v = 0.f;
        for (int w = 0; w < ATT_WARPS; w++) v += wacc[w][d] * f[w];
        a.out[(size_t)item * a.dv + d] = v * inv;
    }
}

int fllm_attention_launch(cudaStream_t s, const float *q, const uint16_t *kpool, const uint16_t *vpool, float *out,
                          const int64_t *tables, const int64_t *lengths, const float *sinks, const float *alibi,
                          int rows, int heads, int kv_heads, int dk, int dv, int stride, int block_size, int kind,
                          int kgroup, int vgroup, int kscale_base, int vscale_base, float scale, float softcap,
                          int window, int chunk) {
    if (rows <= 0) return 0;
    if (dk > ATT_MAX_D || dv > ATT_MAX_D) return fllm_set_error("attention: head width %d/%d exceeds %d", dk, dv, ATT_MAX_D);
    AttnArgs a{q, kpool, vpool, out, tables, lengths, sinks, alibi, rows, heads, kv_heads, dk, dv, stride, block_size,
               kind, kgroup, vgroup, kscale_base, vscale_base, scale, softcap, window, chunk};
    attention_kernel<<<rows * heads, ATT_WARPS * 32, 0, s>>>(a);
    return fllm_check_launch("attention");
}
