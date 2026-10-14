// fLLM CUDA backend: faster paged attention for the common case (f16 cache, key and value width
// 64, 128 or 256, one head per block).
//
//  - A lane owns D / 32 consecutive dimensions, so a warp reads a whole key or value row with one
//    vector load per lane.
//  - A warp works on four keys at a time: all loads are issued first, the four dot products are
//    reduced together (independent shuffle chains), and the online softmax is updated once.
//  - Long contexts are split over several blocks (flash-decoding); a small kernel merges the
//    partial results. Short contexts and prefill rows use one block and write the final result.
// Other shapes (q8 cache, odd head widths) use the generic kernel in attention.cu.
#include "common.cuh"
#include "fllm_cuda_internal.h"

#define A2_WARPS 4
#define A2_KPW 4
#define A2_FLOOR (-3.0e38f)

struct Attn2Args {
    const float *q;
    const uint16_t *kpool, *vpool;
    float *out;
    float *ws;  // partial results when splits > 1
    const int64_t *tables, *lengths;
    const float *sinks, *alibi;
    int heads, kv_heads, stride, block_size, per_split, splits;
    float scale, softcap;
    int window, chunk;
    int kscale_base, vscale_base;  // q8_0 cache only: first scale (in 16-bit words) of the layer's key / value pool
};

template <int V>
struct HalfVec;  // V halves loaded with one instruction
template <>
struct HalfVec<2> {
    uint32_t w[1];
    __device__ static HalfVec load(const uint16_t *p) { HalfVec v; v.w[0] = __ldg(reinterpret_cast<const uint32_t *>(p)); return v; }
};
template <>
struct HalfVec<4> {
    uint32_t w[2];
    __device__ static HalfVec load(const uint16_t *p) {
        HalfVec v;
        const uint2 t = __ldg(reinterpret_cast<const uint2 *>(p));
        v.w[0] = t.x; v.w[1] = t.y;
        return v;
    }
};
template <>
struct HalfVec<8> {
    uint32_t w[4];
    __device__ static HalfVec load(const uint16_t *p) {
        HalfVec v;
        const uint4 t = __ldg(reinterpret_cast<const uint4 *>(p));
        v.w[0] = t.x; v.w[1] = t.y; v.w[2] = t.z; v.w[3] = t.w;
        return v;
    }
};

template <int V>
__device__ __forceinline__ float half_at(const HalfVec<V> &v, int i) {
    const uint32_t w = v.w[i >> 1];
    return h2f_bits((uint16_t)(i & 1 ? (w >> 16) : (w & 0xffffu)));
}

// V int8 values of a q8_0 cache row (V = 2, 4 or 8) and the scale of their 32-value group.
template <int V>
struct Q8Vec {
    int8_t q[V];
    float scale;
    __device__ static Q8Vec load(const uint16_t *pool, const uint16_t *scales, size_t slot, size_t kv, int head_col, int lane) {
        Q8Vec r;
        const int8_t *p = reinterpret_cast<const int8_t *>(pool) + slot * kv + head_col + lane * V;
        if (V == 2) { const uint16_t t = *reinterpret_cast<const uint16_t *>(p); memcpy(r.q, &t, 2); }
        else if (V == 4) { const uint32_t t = __ldg(reinterpret_cast<const uint32_t *>(p)); memcpy(r.q, &t, 4); }
        else { const uint2 t = __ldg(reinterpret_cast<const uint2 *>(p)); memcpy(r.q, &t, 8); }
        r.scale = h2f_bits(scales[slot * (kv / 32) + (head_col + lane * V) / 32]);
        return r;
    }
};

template <int D, bool Q8>
__global__ void __launch_bounds__(A2_WARPS * 32) attn2_kernel(Attn2Args a) {
    constexpr int V = D / 32;
    __shared__ float wm[A2_WARPS], wl[A2_WARPS];
    __shared__ float wacc[A2_WARPS][D];
    const int h = blockIdx.x, r = blockIdx.y, sp = blockIdx.z;
    const int kh = h / (a.heads / a.kv_heads);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = (int)a.lengths[r];
    int start = 0;
    if (a.chunk != 0) start = (n - 1) / a.chunk * a.chunk;
    else if (a.window != 0 && n > a.window) start = n - a.window;
    const int begin = start + sp * a.per_split;
    const int end = min(n, begin + a.per_split);

    float qv[V];
    const float *qp = a.q + ((size_t)r * a.heads + h) * D + lane * V;
#pragma unroll
    for (int i = 0; i < V; i++) qv[i] = qp[i] * a.scale;

    float acc[V];
#pragma unroll
    for (int i = 0; i < V; i++) acc[i] = 0.f;
    float m = A2_FLOOR, l = 0.f;
    const int64_t *table = a.tables + (size_t)r * a.stride;
    const size_t kvw = (size_t)a.kv_heads * D;
    const uint16_t *kbase = a.kpool + (size_t)kh * D + lane * V;
    const uint16_t *vbase = a.vpool + (size_t)kh * D + lane * V;

    for (int p0 = begin + warp * A2_KPW; p0 < end; p0 += A2_WARPS * A2_KPW) {
        HalfVec<V> kvec[A2_KPW], vvec[A2_KPW];
        Q8Vec<V> kq[Q8 ? A2_KPW : 1], vq[Q8 ? A2_KPW : 1];
        bool ok[A2_KPW];
#pragma unroll
        for (int j = 0; j < A2_KPW; j++) {
            const int p = p0 + j;
            ok[j] = p < end;
            const size_t slot = ok[j] ? (size_t)table[p / a.block_size] * a.block_size + p % a.block_size : 0;
            if (Q8) {
                kq[j] = Q8Vec<V>::load(a.kpool, a.kpool + a.kscale_base, slot, kvw, kh * D, lane);
                vq[j] = Q8Vec<V>::load(a.vpool, a.vpool + a.vscale_base, slot, kvw, kh * D, lane);
            } else {
                kvec[j] = HalfVec<V>::load(kbase + slot * kvw);
                vvec[j] = HalfVec<V>::load(vbase + slot * kvw);
            }
        }
        float s[A2_KPW];
#pragma unroll
        for (int j = 0; j < A2_KPW; j++) {
            float d = 0.f;
            if (Q8) {
#pragma unroll
                for (int i = 0; i < V; i++) d = fmaf(qv[i], (float)kq[j].q[i], d);
                d *= kq[j].scale;
            } else {
#pragma unroll
                for (int i = 0; i < V; i++) d = fmaf(qv[i], half_at<V>(kvec[j], i), d);
            }
            s[j] = d;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
#pragma unroll
            for (int j = 0; j < A2_KPW; j++) s[j] += __shfl_xor_sync(0xffffffffu, s[j], o);
        float m_new = m;
#pragma unroll
        for (int j = 0; j < A2_KPW; j++) {
            if (!ok[j]) { s[j] = A2_FLOOR; continue; }
            if (a.alibi) s[j] += a.alibi[h] * ((float)(p0 + j) - (float)(n - 1));
            if (a.softcap != 0.f) s[j] = a.softcap * tanhf(fminf(fmaxf(s[j] / a.softcap, -15.f), 15.f));
            m_new = fmaxf(m_new, s[j]);
        }
        const float corr = __expf(m - m_new);
        float e[A2_KPW];
        float esum = 0.f;
#pragma unroll
        for (int j = 0; j < A2_KPW; j++) {
            e[j] = ok[j] ? __expf(s[j] - m_new) : 0.f;
            esum += e[j];
        }
        l = l * corr + esum;
#pragma unroll
        for (int i = 0; i < V; i++) {
            float x = acc[i] * corr;
#pragma unroll
            for (int j = 0; j < A2_KPW; j++)
                x = fmaf(Q8 ? e[j] * vq[Q8 ? j : 0].scale : e[j], Q8 ? (float)vq[Q8 ? j : 0].q[i] : half_at<V>(vvec[j], i), x);
            acc[i] = x;
        }
        m = m_new;
    }

#pragma unroll
    for (int i = 0; i < V; i++) wacc[warp][lane * V + i] = acc[i];
    if (lane == 0) { wm[warp] = m; wl[warp] = l; }
    __syncthreads();

    float M = A2_FLOOR;
    for (int w = 0; w < A2_WARPS; w++) M = fmaxf(M, wm[w]);
    float f[A2_WARPS], L = 0.f;
    for (int w = 0; w < A2_WARPS; w++) {
        f[w] = __expf(wm[w] - M);
        L += wl[w] * f[w];
    }
    if (a.splits == 1) {
        float Mf = M;
        if (a.sinks) {
            Mf = fmaxf(M, a.sinks[h]);
            const float rescale = __expf(M - Mf);
            for (int w = 0; w < A2_WARPS; w++) f[w] *= rescale;
            L = L * rescale + __expf(a.sinks[h] - Mf);
        }
        const float inv = 1.0f / L;
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            float v = 0.f;
            for (int w = 0; w < A2_WARPS; w++) v += wacc[w][d] * f[w];
            a.out[((size_t)r * a.heads + h) * D + d] = v * inv;
        }
    } else {
        float *dst = a.ws + (((size_t)r * a.heads + h) * a.splits + sp) * (D + 2);
        for (int d = threadIdx.x; d < D; d += blockDim.x) {
            float v = 0.f;
            for (int w = 0; w < A2_WARPS; w++) v += wacc[w][d] * f[w];
            dst[d] = v;
        }
        if (threadIdx.x == 0) { dst[D] = M; dst[D + 1] = L; }
    }
}

template <int D>
__global__ void __launch_bounds__(D) attn2_merge_kernel(Attn2Args a) {
    const int h = blockIdx.x, r = blockIdx.y;
    const float *src = a.ws + ((size_t)r * a.heads + h) * a.splits * (D + 2);
    float M = a.sinks ? a.sinks[h] : A2_FLOOR;
    for (int s = 0; s < a.splits; s++) M = fmaxf(M, src[(size_t)s * (D + 2) + D]);
    float L = a.sinks ? __expf(a.sinks[h] - M) : 0.f, v = 0.f;
    for (int s = 0; s < a.splits; s++) {
        const float *p = src + (size_t)s * (D + 2);
        const float f = __expf(p[D] - M);
        L += p[D + 1] * f;
        v += p[threadIdx.x] * f;
    }
    a.out[((size_t)r * a.heads + h) * D + threadIdx.x] = v / L;
}

template <int D, bool Q8>
static int launch(cudaStream_t st, Attn2Args a, int rows, int max_len, float *ws, size_t ws_bytes) {
    // split long contexts when there are too few (row, head) blocks to fill the GPU
    int splits = 1;
    const int blocks = rows * a.heads;
    if (blocks < 256 && max_len > 512) {
        splits = (max_len + 255) / 256;
        if (splits > 32) splits = 32;
        while (splits > 1 && (size_t)rows * a.heads * splits * (D + 2) * 4 > ws_bytes) splits--;
    }
    a.splits = splits;
    a.per_split = (max_len + splits - 1) / splits;
    a.ws = ws;
    attn2_kernel<D, Q8><<<dim3(a.heads, rows, splits), A2_WARPS * 32, 0, st>>>(a);
    if (splits > 1) attn2_merge_kernel<D><<<dim3(a.heads, rows), D, 0, st>>>(a);
    return fllm_check_launch("attention2");
}

// Returns -1 when the shape is not covered (the caller uses the generic kernel).
int fllm_attention2_try(cudaStream_t s, const float *q, const uint16_t *kpool, const uint16_t *vpool, float *out,
                        const int64_t *tables, const int64_t *lengths, int max_len, const float *sinks,
                        const float *alibi, int rows, int heads, int kv_heads, int dk, int dv, int stride,
                        int block_size, int kind, float scale, float softcap, int window, int chunk, float *ws,
                        size_t ws_bytes, int kscale_base, int vscale_base) {
    if ((kind != 1 && kind != 2) || dk != dv || (dk != 64 && dk != 128 && dk != 256) || rows <= 0) return -1;
    if (kind == 1 && ((reinterpret_cast<uintptr_t>(kpool) & 15) != 0 || (reinterpret_cast<uintptr_t>(vpool) & 15) != 0)) return -1;
    if (kind == 2 && (dk == 256 || (reinterpret_cast<uintptr_t>(kpool) & 3) != 0 || (reinterpret_cast<uintptr_t>(vpool) & 3) != 0)) return -1;
    Attn2Args a{q, kpool, vpool, out, ws, tables, lengths, sinks, alibi, heads, kv_heads, stride, block_size, 1, 1,
                scale, softcap, window, chunk, kscale_base, vscale_base};
    if (kind == 2) return dk == 64 ? launch<64, true>(s, a, rows, max_len, ws, ws_bytes) : launch<128, true>(s, a, rows, max_len, ws, ws_bytes);
    switch (dk) {
        case 64: return launch<64, false>(s, a, rows, max_len, ws, ws_bytes);
        case 128: return launch<128, false>(s, a, rows, max_len, ws, ws_bytes);
        default: return launch<256, false>(s, a, rows, max_len, ws, ws_bytes);
    }
}
