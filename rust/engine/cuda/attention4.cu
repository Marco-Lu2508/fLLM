// fLLM CUDA backend: decode attention that shares every K/V row between the query heads of a KV head (GQA).
//
// attention2.cu processes one query head per block, so a model with 4 query heads per KV head reads the same keys and
// values four times. Here a block owns one KV head: sixteen lanes read one key row (8 halves = one 16-byte load each),
// two such groups per warp, four warps per block, every group walks its own run of keys with an online softmax
// that is kept for all G query heads of the KV head at once. The partial results of the eight groups are merged
// through shared memory; long contexts are split over several blocks (flash-decoding) and merged by a second kernel.
//
// Covers: f16 cache, key = value width 128, G in {1, 2, 4, 8}, sliding window / chunk / soft-capping / sinks (no ALiBi).
#include <type_traits>

#include "common.cuh"
#include "fllm_cuda_internal.h"
#if !defined(__HIPCC__)
#include <cuda_bf16.h>
#endif
#if defined(__HIPCC__)
#include "../hip/hip_dpp.cuh"
#endif

#define A4_WARPS 4
#define A4_UK 4
#define A4_FLOOR (-3.0e38f)

struct Attn4Args {
    const void *q;  // FP32, or FP16 / BF16 when adt != 0 (FLLM_ACT_DTYPE)
    const uint16_t *kpool, *vpool;
    void *out;
    float *ws;
    const int64_t *tables, *lengths;
    const float *sinks;
    int heads, kv_heads, stride, block_size, splits;
    int min_splits;  // splits every row gets when it has more than 128 keys (enough blocks to fill the GPU)
    float scale, softcap;
    int window, chunk;
    int kscale_base, vscale_base;  // q8_0 cache: first scale (16-bit words) of the layer's key / value pool
    int adt = 0;                   // activation type of q and out: 0 FP32, 1 FP16, 2 BF16
};

// q and the result in the activation type of the pipeline (FP32 unless FLLM_ACT_DTYPE says otherwise)
__device__ __forceinline__ float a4_load(const void *p, size_t i, int adt) {
#if defined(__HIPCC__)
    (void)adt;
    return static_cast<const float *>(p)[i];
#else
    if (adt == 1) return __half2float(static_cast<const __half *>(p)[i]);
    if (adt == 2) return __bfloat162float(static_cast<const __nv_bfloat16 *>(p)[i]);
    return static_cast<const float *>(p)[i];
#endif
}
__device__ __forceinline__ void a4_store(void *p, size_t i, float v, int adt) {
#if defined(__HIPCC__)
    (void)adt;
    static_cast<float *>(p)[i] = v;
#else
    if (adt == 1) static_cast<__half *>(p)[i] = __float2half_rn(v);
    else if (adt == 2) static_cast<__nv_bfloat16 *>(p)[i] = __float2bfloat16_rn(v);
    else static_cast<float *>(p)[i] = v;
#endif
}

__device__ __forceinline__ void a4_unpack(const uint4 &v, float f[8]) {
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; i++) {
        f[2 * i] = h2f_bits((uint16_t)(w[i] & 0xffffu));
        f[2 * i + 1] = h2f_bits((uint16_t)(w[i] >> 16));
    }
}

// The number of splits row `r` really uses. The grid has `a.splits` blocks per row and head (one launch shape for every context length, which a CUDA graph
// needs: it is recorded for the longest context), the row uses as many of them as its own number of keys asks for: about 512 keys per block, at least
// `min_splits` once it has more than 128 keys. The blocks of the other splits leave at once.
__device__ __forceinline__ int a4_row_splits(const Attn4Args &a, int keys) {
    if (a.splits <= 1 || keys <= 128) return 1;
    const int want = max(a.min_splits, (keys + 511) / 512);
    return max(1, min(min(want, a.splits), (keys + 63) / 64));
}
__device__ __forceinline__ int a4_row_start(const Attn4Args &a, int n) {
    if (a.chunk != 0) return (n - 1) / a.chunk * a.chunk;
    if (a.window != 0 && n > a.window) return n - a.window;
    return 0;
}

template <int D, int G, bool Q8>
__global__ void __launch_bounds__(A4_WARPS * 32) attn4_kernel(Attn4Args a) {
    constexpr int LPK = D / 8;              // lanes that read one key row (8 values each)
    constexpr int GPW = 32 / LPK;           // key groups per warp
    constexpr int NG = A4_WARPS * GPW;      // key groups per block
    __shared__ float sm_m[NG][G], sm_l[NG][G];
    __shared__ float sm_acc[NG][G][D];
    const int kh = blockIdx.x, r = blockIdx.y, sp = blockIdx.z;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int grp = lane / LPK, sub = lane % LPK;
    const int gid = warp * GPW + grp;
    const int n = (int)a.lengths[r];
    const int start = a4_row_start(a, n);
    const int seff = a4_row_splits(a, n - start);
    if (sp >= seff) return;
    const int per_split = (n - start + seff - 1) / seff;
    const int begin = start + sp * per_split;
    const int end = min(n, begin + per_split);

    float qv[G][8], acc[G][8], m[G], l[G];
#pragma unroll
    for (int g = 0; g < G; g++) {
        const size_t qi = ((size_t)r * a.heads + (size_t)kh * G + g) * D + sub * 8;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            qv[g][i] = a4_load(a.q, qi + i, a.adt) * a.scale;
            acc[g][i] = 0.f;
        }
        m[g] = A4_FLOOR;
        l[g] = 0.f;
    }
    const int64_t *table = a.tables + (size_t)r * a.stride;
    const size_t kvw = (size_t)a.kv_heads * D;
    const uint16_t *kbase = a.kpool + (size_t)kh * D + sub * 8;
    const uint16_t *vbase = a.vpool + (size_t)kh * D + sub * 8;

    for (int base = begin; base < end; base += NG * A4_UK) {
        uint4 kvec[A4_UK], vvec[A4_UK];
        float ksc[Q8 ? A4_UK : 1], vsc[Q8 ? A4_UK : 1];
        bool ok[A4_UK];
#pragma unroll
        for (int j = 0; j < A4_UK; j++) {
            const int p = base + gid * A4_UK + j;
            ok[j] = p < end;
            const size_t slot = ok[j] ? (size_t)table[p / a.block_size] * a.block_size + p % a.block_size : 0;
            if (Q8) {
                // 8 int8 values per lane; one f16 scale per 32 values
                const int8_t *kp = reinterpret_cast<const int8_t *>(a.kpool) + slot * kvw + (size_t)kh * D + sub * 8;
                const int8_t *vp = reinterpret_cast<const int8_t *>(a.vpool) + slot * kvw + (size_t)kh * D + sub * 8;
                const uint2 kq = __ldg(reinterpret_cast<const uint2 *>(kp)), vq = __ldg(reinterpret_cast<const uint2 *>(vp));
                kvec[j] = make_uint4(kq.x, kq.y, 0u, 0u);
                vvec[j] = make_uint4(vq.x, vq.y, 0u, 0u);
                const size_t sidx = slot * (kvw / 32) + ((size_t)kh * D + sub * 8) / 32;
                ksc[j] = h2f_bits(a.kpool[a.kscale_base + sidx]);
                vsc[j] = h2f_bits(a.vpool[a.vscale_base + sidx]);
            } else {
                kvec[j] = __ldg(reinterpret_cast<const uint4 *>(kbase + slot * kvw));
                vvec[j] = __ldg(reinterpret_cast<const uint4 *>(vbase + slot * kvw));
            }
        }
#pragma unroll
        for (int j = 0; j < A4_UK; j++) {
            float kf[8], vf[8];
            if (Q8) {
                const uint32_t kw[2] = {kvec[j].x, kvec[j].y}, vw[2] = {vvec[j].x, vvec[j].y};
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    kf[i] = (float)(int8_t)((kw[i >> 2] >> (8 * (i & 3))) & 0xffu) * ksc[j];
                    vf[i] = (float)(int8_t)((vw[i >> 2] >> (8 * (i & 3))) & 0xffu) * vsc[j];
                }
            } else {
                a4_unpack(kvec[j], kf);
                a4_unpack(vvec[j], vf);
            }
            float s[G];
#pragma unroll
            for (int g = 0; g < G; g++) {
                float d = 0.f;
#pragma unroll
                for (int i = 0; i < 8; i++) d = fmaf(qv[g][i], kf[i], d);
                s[g] = d;
            }
#if defined(__HIPCC__)
            // AMD: the sum over the LPK lanes of a key row with DPP row operations; __shfl_xor is a ds_bpermute there (hip/hip_dpp.cuh)
#pragma unroll
            for (int g = 0; g < G; g++) s[g] = fllm_hip::group_sum<LPK>(s[g]);
#else
#pragma unroll
            for (int o = LPK / 2; o > 0; o >>= 1)
#pragma unroll
                for (int g = 0; g < G; g++) s[g] += __shfl_xor_sync(0xffffffffu, s[g], o);
#endif
            if (!ok[j]) continue;
#pragma unroll
            for (int g = 0; g < G; g++) {
                float sg = s[g];
                if (a.softcap != 0.f) sg = a.softcap * tanhf(fminf(fmaxf(sg / a.softcap, -15.f), 15.f));
                const float m_new = fmaxf(m[g], sg);
                const float corr = __expf(m[g] - m_new);
                const float pr = __expf(sg - m_new);
                l[g] = l[g] * corr + pr;
                m[g] = m_new;
#pragma unroll
                for (int i = 0; i < 8; i++) acc[g][i] = fmaf(pr, vf[i], acc[g][i] * corr);
            }
        }
    }

    // merge the eight groups of the block
#pragma unroll
    for (int g = 0; g < G; g++) {
#pragma unroll
        for (int i = 0; i < 8; i++) sm_acc[gid][g][sub * 8 + i] = acc[g][i];
        if (sub == 0) {
            sm_m[gid][g] = m[g];
            sm_l[gid][g] = l[g];
        }
    }
    __syncthreads();
    for (int d = threadIdx.x; d < D; d += A4_WARPS * 32) {
#pragma unroll
        for (int g = 0; g < G; g++) {
            float M = A4_FLOOR;
#pragma unroll
            for (int s = 0; s < NG; s++) M = fmaxf(M, sm_m[s][g]);
            float L = 0.f, v = 0.f;
#pragma unroll
            for (int s = 0; s < NG; s++) {
                const float f = __expf(sm_m[s][g] - M);
                L += sm_l[s][g] * f;
                v += sm_acc[s][g][d] * f;
            }
            const int h = kh * G + g;
            if (seff == 1) {
                if (a.sinks) {
                    const float Mf = fmaxf(M, a.sinks[h]);
                    const float rescale = __expf(M - Mf);
                    v *= rescale;
                    L = L * rescale + __expf(a.sinks[h] - Mf);
                }
                a4_store(a.out, ((size_t)r * a.heads + h) * D + d, v / L, a.adt);
            } else {
                float *dst = a.ws + (((size_t)r * a.heads + h) * a.splits + sp) * (D + 2);
                dst[d] = v;
                if (d == 0) {
                    dst[D] = M;
                    dst[D + 1] = L;
                }
            }
        }
    }
}

#if !defined(__HIPCC__)
// ---- the same decode attention on the tensor cores -----------------------------------------------------------------------------------------------
// attn4_kernel above does 2 x G x 128 multiply-adds per key and KV head on the FP32 units (G = 7 for Qwen2.5-7B: 1792 per key), which at 16 rows x 2100
// keys x 4 KV heads is as long as reading the cache takes (114 us per layer against 58 us of memory time on an A100). Here the products run on `mma.m16n8k16`
// (FP16 inputs, FP32 accumulation): S = Q K^T with the query heads as the 16 rows (G <= 8 of them, the other rows are zero), O += P V with P in the registers
// the S fragments leave it in. The queries and the probabilities are split in a high and a low FP16 part (x = hi + lo, 22 bits), so that the result is the one
// of the FP32 kernel up to rounding: the keys and values are FP16 in the cache anyway. A warp walks its own 16-key tiles, each lane reads its fragments of K and
// V straight from the cache (V is transposed in the registers with `movmatrix`), and the four warps of the block merge as the FP32 kernel's groups do.
// The block covers the same keys of the same (row, KV head, split) as attn4_kernel and writes the same partial results, so the merge kernel is shared.
// Covers: f16 cache, key = value width 128, G <= 8, sliding window / chunk / soft-capping / sinks.
__device__ __forceinline__ void a4_mma(float c[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
    (void)c; (void)a0; (void)a1; (void)a2; (void)a3; (void)b0; (void)b1;
#else
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
#endif
}
__device__ __forceinline__ uint32_t a4_transpose8x8(uint32_t x) {
    uint32_t y = 0;
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)
    asm volatile("movmatrix.sync.aligned.m8n8.trans.b16 %0, %1;\n" : "=r"(y) : "r"(x));
#endif
    return y;
}
__device__ __forceinline__ uint32_t a4_pack(float lo, float hi) {
    const __half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t *>(&v);
}
// the FP16 value nearest to x and the FP16 value nearest to the rest: hi + lo = x to 22 bits
__device__ __forceinline__ void a4_split(float x, __half &hi, __half &lo) {
    hi = __float2half_rn(x);
    lo = __float2half_rn(x - __half2float(hi));
}

template <int G>
__global__ void __launch_bounds__(128) attn4tc_kernel(Attn4Args a) {
    constexpr int D = 128;
    __shared__ uint4 sm_q[8][32][2];  // the A fragments of the queries per 16 dimensions: high part, low part
    __shared__ float sm_acc[4][G][D];
    __shared__ float sm_m[4][G], sm_l[4][G];
    const int kh = blockIdx.x, r = blockIdx.y, sp = blockIdx.z;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, q = lane & 3;
    const int n = (int)a.lengths[r];
    const int start = a4_row_start(a, n);
    const int seff = a4_row_splits(a, n - start);
    if (sp >= seff) return;
    const int per_split = (n - start + seff - 1) / seff;
    const int begin = start + sp * per_split;
    const int end = min(n, begin + per_split);

    for (int e = threadIdx.x; e < 8 * 32; e += 128) {
        const int ks = e >> 5, ln = e & 31, gg = ln >> 2, qq = ln & 3;
        uint32_t hi[4] = {0u, 0u, 0u, 0u}, lo[4] = {0u, 0u, 0u, 0u};  // rows 8..15 (the registers 1 and 3) stay zero
        if (gg < G) {
#pragma unroll
            for (int part = 0; part < 2; part++) {
                const size_t qi = ((size_t)r * a.heads + (size_t)kh * G + gg) * D + 16 * ks + 2 * qq + 8 * part;
                const float x0 = a4_load(a.q, qi, a.adt) * a.scale, x1 = a4_load(a.q, qi + 1, a.adt) * a.scale;
                __half h0, l0, h1, l1;
                a4_split(x0, h0, l0);
                a4_split(x1, h1, l1);
                hi[2 * part] = a4_pack(__half2float(h0), __half2float(h1));
                lo[2 * part] = a4_pack(__half2float(l0), __half2float(l1));
            }
        }
        sm_q[ks][ln][0] = make_uint4(hi[0], hi[1], hi[2], hi[3]);
        sm_q[ks][ln][1] = make_uint4(lo[0], lo[1], lo[2], lo[3]);
    }
    __syncthreads();

    float acc[D / 8][4];
#pragma unroll
    for (int dt = 0; dt < D / 8; dt++) acc[dt][0] = acc[dt][1] = acc[dt][2] = acc[dt][3] = 0.f;
    float m = A4_FLOOR, l = 0.f;  // of the query head g (the lanes of a quad hold the same m; l is summed over the quad at the end)
    const int64_t *table = a.tables + (size_t)r * a.stride;
    const size_t kvw = (size_t)a.kv_heads * D;
    const uint16_t *kbase = a.kpool + (size_t)kh * D;
    const uint16_t *vbase = a.vpool + (size_t)kh * D;

    for (int p0 = begin + warp * 16; p0 < end; p0 += 4 * 16) {
        // the cache rows of the keys g and g + 8 of the tile (K^T fragments: B[k = dimension][n = key]; V: one 8 x 8 block of keys x dimensions per 8 dimensions)
        const uint16_t *kr[2], *vr[2];
        bool live[2];
#pragma unroll
        for (int j = 0; j < 2; j++) {
            const int p = p0 + g + 8 * j;
            live[j] = p < end;
            const size_t slot = live[j] ? (size_t)table[p / a.block_size] * a.block_size + p % a.block_size : 0;
            kr[j] = kbase + slot * kvw + 2 * q;
            vr[j] = vbase + slot * kvw + 2 * q;
        }
        uint32_t kb[2][8][2], vw[D / 8][2];
#pragma unroll
        for (int j = 0; j < 2; j++) {
#pragma unroll
            for (int ks = 0; ks < 8; ks++) {
                kb[j][ks][0] = __ldg(reinterpret_cast<const uint32_t *>(kr[j] + 16 * ks));
                kb[j][ks][1] = __ldg(reinterpret_cast<const uint32_t *>(kr[j] + 16 * ks + 8));
            }
        }
#pragma unroll
        for (int dt = 0; dt < D / 8; dt++) {
#pragma unroll
            for (int j = 0; j < 2; j++) vw[dt][j] = live[j] ? __ldg(reinterpret_cast<const uint32_t *>(vr[j] + 8 * dt)) : 0u;
        }
        float s[2][4];
#pragma unroll
        for (int nt = 0; nt < 2; nt++) s[nt][0] = s[nt][1] = s[nt][2] = s[nt][3] = 0.f;
#pragma unroll
        for (int ks = 0; ks < 8; ks++) {
            const uint4 qh = sm_q[ks][lane][0], ql = sm_q[ks][lane][1];
#pragma unroll
            for (int nt = 0; nt < 2; nt++) {
                a4_mma(s[nt], qh.x, qh.y, qh.z, qh.w, kb[nt][ks][0], kb[nt][ks][1]);
                a4_mma(s[nt], ql.x, ql.y, ql.z, ql.w, kb[nt][ks][0], kb[nt][ks][1]);
            }
        }
        // the scores of head g for the keys 2q, 2q + 1, 8 + 2q, 8 + 2q + 1 of the tile
        float sc[4] = {s[0][0], s[0][1], s[1][0], s[1][1]};
        bool valid[4];
        float tmax = A4_FLOOR;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            valid[i] = p0 + (i >> 1) * 8 + 2 * q + (i & 1) < end;
            if (a.softcap != 0.f) sc[i] = a.softcap * tanhf(fminf(fmaxf(sc[i] / a.softcap, -15.f), 15.f));
            if (valid[i]) tmax = fmaxf(tmax, sc[i]);
        }
        tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 1));
        tmax = fmaxf(tmax, __shfl_xor_sync(0xffffffffu, tmax, 2));
        const float m_new = fmaxf(m, tmax);
        const float corr = __expf(m - m_new);
        float pr[4], psum = 0.f;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            pr[i] = valid[i] ? __expf(sc[i] - m_new) : 0.f;
            psum += pr[i];
        }
        l = l * corr + psum;
        m = m_new;
        uint32_t ph0, ph2, pl0, pl2;
        {
            __half h[4], lo[4];
#pragma unroll
            for (int i = 0; i < 4; i++) a4_split(pr[i], h[i], lo[i]);
            ph0 = a4_pack(__half2float(h[0]), __half2float(h[1]));
            ph2 = a4_pack(__half2float(h[2]), __half2float(h[3]));
            pl0 = a4_pack(__half2float(lo[0]), __half2float(lo[1]));
            pl2 = a4_pack(__half2float(lo[2]), __half2float(lo[3]));
        }
#pragma unroll
        for (int dt = 0; dt < D / 8; dt++) {
            acc[dt][0] *= corr;
            acc[dt][1] *= corr;
            const uint32_t b0 = a4_transpose8x8(vw[dt][0]), b1 = a4_transpose8x8(vw[dt][1]);
            a4_mma(acc[dt], ph0, 0u, ph2, 0u, b0, b1);
            a4_mma(acc[dt], pl0, 0u, pl2, 0u, b0, b1);
        }
    }

    l += __shfl_xor_sync(0xffffffffu, l, 1);
    l += __shfl_xor_sync(0xffffffffu, l, 2);
    if (g < G) {
#pragma unroll
        for (int dt = 0; dt < D / 8; dt++) {
            sm_acc[warp][g][8 * dt + 2 * q] = acc[dt][0];
            sm_acc[warp][g][8 * dt + 2 * q + 1] = acc[dt][1];
        }
        if (q == 0) {
            sm_m[warp][g] = m;
            sm_l[warp][g] = l;
        }
    }
    __syncthreads();
    // merge the four warps of the block (as attn4_kernel merges its eight groups)
    for (int d = threadIdx.x; d < D; d += 128) {
#pragma unroll
        for (int gq = 0; gq < G; gq++) {
            float M = A4_FLOOR;
#pragma unroll
            for (int s = 0; s < 4; s++) M = fmaxf(M, sm_m[s][gq]);
            float L = 0.f, v = 0.f;
#pragma unroll
            for (int s = 0; s < 4; s++) {
                const float f = __expf(sm_m[s][gq] - M);
                L += sm_l[s][gq] * f;
                v += sm_acc[s][gq][d] * f;
            }
            const int h = kh * G + gq;
            if (seff == 1) {
                if (a.sinks) {
                    const float Mf = fmaxf(M, a.sinks[h]);
                    const float rescale = __expf(M - Mf);
                    v *= rescale;
                    L = L * rescale + __expf(a.sinks[h] - Mf);
                }
                a4_store(a.out, ((size_t)r * a.heads + h) * D + d, v / L, a.adt);
            } else {
                float *dst = a.ws + (((size_t)r * a.heads + h) * a.splits + sp) * (D + 2);
                dst[d] = v;
                if (d == 0) {
                    dst[D] = M;
                    dst[D + 1] = L;
                }
            }
        }
    }
}
#endif

template <int D>
__global__ void __launch_bounds__(D) attn4_merge_kernel(Attn4Args a) {
    const int h = blockIdx.x, r = blockIdx.y;
    const int n = (int)a.lengths[r];
    const int splits = a4_row_splits(a, n - a4_row_start(a, n));
    if (splits == 1) return;  // the single block of the row wrote the result itself
    const float *src = a.ws + ((size_t)r * a.heads + h) * a.splits * (D + 2);
    float M = a.sinks ? a.sinks[h] : A4_FLOOR;
    for (int s = 0; s < splits; s++) M = fmaxf(M, src[(size_t)s * (D + 2) + D]);
    float L = a.sinks ? __expf(a.sinks[h] - M) : 0.f, v = 0.f;
    for (int s = 0; s < splits; s++) {
        const float *p = src + (size_t)s * (D + 2);
        const float f = __expf(p[D] - M);
        L += p[D + 1] * f;
        v += p[threadIdx.x] * f;
    }
    a4_store(a.out, ((size_t)r * a.heads + h) * D + threadIdx.x, v / L, a.adt);
}

template <int D, int G, bool Q8>
static int launch4(cudaStream_t st, Attn4Args a, int rows, int max_len, float *ws, size_t ws_bytes) {
    // enough blocks to fill the GPU, and no more than ~512 keys per block. The grid is sized for `max_len` (a CUDA graph is recorded for the longest
    // context); each row takes the splits its own length needs (a4_row_splits), the others return at once. At most about 1024 blocks are launched.
    int splits = 1;
    const int blocks = rows * a.kv_heads;
    a.min_splits = 1;
    if (max_len > 128) {
        a.min_splits = (92 + blocks - 1) / blocks;
        splits = a.min_splits < (max_len + 511) / 512 ? (max_len + 511) / 512 : a.min_splits;
        const int cap = (max_len + 63) / 64;
        if (splits > cap) splits = cap;
        if (splits > 64) splits = 64;
        const int grid_cap = 1024 / blocks > a.min_splits ? 1024 / blocks : a.min_splits;
        if (splits > grid_cap) splits = grid_cap;
        while (splits > 1 && (size_t)rows * a.heads * splits * (D + 2) * 4 > ws_bytes) splits--;
        if (splits < 1) splits = 1;
        if (a.min_splits > splits) a.min_splits = splits;
    }
    a.splits = splits;
    a.ws = ws;
#if !defined(__HIPCC__)
    static const bool no_tc = getenv("FLLM_CUDA_NO_ATTN4TC") != nullptr;
    if constexpr (D == 128 && !Q8 && G <= 8) {
        if (!no_tc && fllm_cc() >= 80 && !fllm_exact_math()) {
            attn4tc_kernel<G><<<dim3(a.kv_heads, rows, splits), 128, 0, st>>>(a);
            if (splits > 1) attn4_merge_kernel<D><<<dim3(a.heads, rows), D, 0, st>>>(a);
            return fllm_check_launch("attention4 tensor cores");
        }
    }
#endif
    attn4_kernel<D, G, Q8><<<dim3(a.kv_heads, rows, splits), A4_WARPS * 32, 0, st>>>(a);
    if (splits > 1) attn4_merge_kernel<D><<<dim3(a.heads, rows), D, 0, st>>>(a);
    return fllm_check_launch("attention4");
}

// FLLM_ACT_DTYPE: the activation type of q and out for the next call (act_api.inc sets it around fllm_attention4_try; 0 = FP32)
thread_local int fllm_attn_act_dt = 0;

// Returns -1 when the shape is not covered.
int fllm_attention4_try(cudaStream_t s, const float *q, const uint16_t *kpool, const uint16_t *vpool, float *out,
                        const int64_t *tables, const int64_t *lengths, int max_len, const float *sinks,
                        const float *alibi, int rows, int heads, int kv_heads, int dk, int dv, int stride,
                        int block_size, int kind, float scale, float softcap, int window, int chunk, float *ws,
                        size_t ws_bytes, int kscale_base, int vscale_base) {
    static const bool disabled = getenv("FLLM_CUDA_NO_ATTN4") != nullptr;
    if (disabled || (kind != 1 && kind != 2) || (dk != 64 && dk != 128 && dk != 256) || dv != dk || alibi || rows <= 0 || kv_heads <= 0 || heads % kv_heads != 0) return -1;
    if ((reinterpret_cast<uintptr_t>(kpool) & 15) != 0 || (reinterpret_cast<uintptr_t>(vpool) & 15) != 0) return -1;
    Attn4Args a{q, kpool, vpool, out, ws, tables, lengths, sinks, heads, kv_heads, stride, block_size, 1, 1, scale,
                softcap, window, chunk, kscale_base, vscale_base};
    a.adt = fllm_attn_act_dt;
    auto run = [&](auto dim) -> int {
        constexpr int D = decltype(dim)::value;
        const int g = heads / kv_heads;
        if (kind == 2) {
            switch (g) {
                case 1: return launch4<D, 1, true>(s, a, rows, max_len, ws, ws_bytes);
                case 2: return launch4<D, 2, true>(s, a, rows, max_len, ws, ws_bytes);
                case 4: return launch4<D, 4, true>(s, a, rows, max_len, ws, ws_bytes);
                case 8: return launch4<D, 8, true>(s, a, rows, max_len, ws, ws_bytes);
                case 3: return launch4<D, 3, true>(s, a, rows, max_len, ws, ws_bytes);
                case 5: return launch4<D, 5, true>(s, a, rows, max_len, ws, ws_bytes);
                case 6: return launch4<D, 6, true>(s, a, rows, max_len, ws, ws_bytes);
                case 7: return launch4<D, 7, true>(s, a, rows, max_len, ws, ws_bytes);
                default: return -1;
            }
        }
        switch (g) {
            case 1: return launch4<D, 1, false>(s, a, rows, max_len, ws, ws_bytes);
            case 2: return launch4<D, 2, false>(s, a, rows, max_len, ws, ws_bytes);
            case 4: return launch4<D, 4, false>(s, a, rows, max_len, ws, ws_bytes);
            case 8: return launch4<D, 8, false>(s, a, rows, max_len, ws, ws_bytes);
            case 3: return launch4<D, 3, false>(s, a, rows, max_len, ws, ws_bytes);
            case 5: return launch4<D, 5, false>(s, a, rows, max_len, ws, ws_bytes);
            case 6: return launch4<D, 6, false>(s, a, rows, max_len, ws, ws_bytes);
            case 7: return launch4<D, 7, false>(s, a, rows, max_len, ws, ws_bytes);
            default: return -1;
        }
    };
    if (dk == 64) return run(std::integral_constant<int, 64>{});
    if (dk == 128) return run(std::integral_constant<int, 128>{});
    return run(std::integral_constant<int, 256>{});
}
