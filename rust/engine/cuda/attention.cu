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

// ---- Doge: dynamic mask attention (see Stream::doge_attention) -------------------------------------------------------------------------------------
// One block (4 warps) per (row, head). Every key of the row gets the decay dt = exp(A[kh] * softplus(w[kh] . value row of all key heads)) (kept in dynamic shared
// memory), it is added to the score, and only the `keep` keys with the largest dt (ties: the lower position) are attended to: the threshold is found by bisection on
// the float bits (dt > 0), the tie cut by bisection on the position.
#define DOGE_BLOCK 128
#define DOGE_MAX_CONTEXT 8192

__device__ unsigned doge_block_sum(unsigned v, unsigned *red) {
    const int tid = threadIdx.x;
    red[tid] = v;
    __syncthreads();
    for (int s2 = DOGE_BLOCK / 2; s2 > 0; s2 >>= 1) {
        if (tid < s2) red[tid] += red[tid + s2];
        __syncthreads();
    }
    const unsigned r = red[0];
    __syncthreads();
    return r;
}

__global__ void __launch_bounds__(DOGE_BLOCK) doge_attention_kernel(AttnArgs a, const float *__restrict__ dtw, int keep) {
    extern __shared__ float dt_s[];
    __shared__ float qs[ATT_MAX_D];
    __shared__ float wm[ATT_WARPS], wl[ATT_WARPS];
    __shared__ float wacc[ATT_WARPS][ATT_MAX_D];
    __shared__ unsigned red[DOGE_BLOCK];
    const int item = blockIdx.x;
    const int r = item / a.heads, h = item % a.heads;
    const int kh = h / (a.heads / a.kv_heads);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int d = threadIdx.x; d < a.dk; d += blockDim.x) qs[d] = a.q[(size_t)item * a.dk + d];
    const int n = (int)a.lengths[r];
    const int64_t *table = a.tables + (size_t)r * a.stride;
    const int kvdv = a.kv_heads * a.dv;
    const float *w = dtw + a.kv_heads + (size_t)kh * kvdv;
    for (int t = threadIdx.x; t < n; t += blockDim.x) {
        const size_t slot = (size_t)table[t / a.block_size] * a.block_size + t % a.block_size;
        float logit = 0.f;
        for (int g = 0; g < a.kv_heads; g++)
            for (int d = 0; d < a.dv; d++) logit = fmaf(w[g * a.dv + d], value_at(a, slot, g, d), logit);
        const float sp = logit > 20.f ? logit : log1pf(expf(logit));
        dt_s[t] = expf(dtw[kh] * sp);
    }
    __syncthreads();
    const bool select_keys = n > keep;
    unsigned thr = 0u;
    int cut = 0x7FFFFFFF;
    if (select_keys) {
        unsigned lo = 0u, hi = 0x7F800001u;
        while (hi - lo > 1u) {
            const unsigned mid = lo + (hi - lo) / 2u;
            unsigned c = 0;
            for (int t = threadIdx.x; t < n; t += blockDim.x) c += __float_as_uint(dt_s[t]) >= mid ? 1u : 0u;
            if (doge_block_sum(c, red) >= (unsigned)keep) lo = mid; else hi = mid;
        }
        thr = lo;
        unsigned cg = 0;
        for (int t = threadIdx.x; t < n; t += blockDim.x) cg += __float_as_uint(dt_s[t]) > thr ? 1u : 0u;
        const unsigned need = (unsigned)keep - doge_block_sum(cg, red);
        int lo_c = -1, hi_c = n - 1;
        while (hi_c - lo_c > 1) {
            const int mid = lo_c + (hi_c - lo_c) / 2;
            unsigned c = 0;
            for (int t = threadIdx.x; t < n; t += blockDim.x) c += (__float_as_uint(dt_s[t]) == thr && t <= mid) ? 1u : 0u;
            if (doge_block_sum(c, red) >= need) hi_c = mid; else lo_c = mid;
        }
        cut = hi_c;
    }

    constexpr int NACC = ATT_MAX_D / 32;
    float acc[NACC];
#pragma unroll
    for (int i = 0; i < NACC; i++) acc[i] = 0.f;
    float m = ATT_FLOOR, l = 0.f;
    for (int p = warp; p < n; p += ATT_WARPS) {
        const unsigned bits = __float_as_uint(dt_s[p]);
        if (select_keys && !(bits > thr || (bits == thr && p <= cut))) continue;
        const size_t slot = (size_t)table[p / a.block_size] * a.block_size + p % a.block_size;
        float part = 0.f;
        for (int d = lane; d < a.dk; d += 32) part = fmaf(qs[d], key_at(a, slot, kh, d), part);
        const float s = warp_sum(part) * a.scale + dt_s[p];
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
    float M = ATT_FLOOR;
    for (int wi = 0; wi < ATT_WARPS; wi++) M = fmaxf(M, wm[wi]);
    float L = 0.f;
    float f[ATT_WARPS];
    for (int wi = 0; wi < ATT_WARPS; wi++) {
        f[wi] = __expf(wm[wi] - M);
        L += wl[wi] * f[wi];
    }
    const float inv = 1.0f / L;
    for (int d = threadIdx.x; d < a.dv; d += blockDim.x) {
        float v = 0.f;
        for (int wi = 0; wi < ATT_WARPS; wi++) v += wacc[wi][d] * f[wi];
        a.out[(size_t)item * a.dv + d] = v * inv;
    }
}

int fllm_doge_attention_launch(cudaStream_t s, const float *q, const uint16_t *kpool, const uint16_t *vpool, float *out, const int64_t *tables,
                               const int64_t *lengths, int max_len, int rows, int heads, int kv_heads, int dk, int dv, int stride, int block_size, int kind,
                               int kgroup, int vgroup, int kscale_base, int vscale_base, float scale, const float *dtw, int keep) {
    if (rows <= 0) return 0;
    if (dk > ATT_MAX_D || dv > ATT_MAX_D) return fllm_set_error("doge attention: head width %d/%d exceeds %d", dk, dv, ATT_MAX_D);
    if (max_len > DOGE_MAX_CONTEXT) return fllm_set_error("doge attention: %d keys exceed %d", max_len, DOGE_MAX_CONTEXT);
    AttnArgs a{q, kpool, vpool, out, tables, lengths, nullptr, nullptr, rows, heads, kv_heads, dk, dv, stride, block_size,
               kind, kgroup, vgroup, kscale_base, vscale_base, scale, 0.f, 0, 0};
    doge_attention_kernel<<<rows * heads, DOGE_BLOCK, (size_t)max_len * sizeof(float), s>>>(a, dtw, keep);
    return fllm_check_launch("doge_attention");
}
