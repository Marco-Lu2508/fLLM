// fLLM CUDA backend: tensor-core attention for prompt processing (many query rows of one sequence).
//
// A "run" is a block of consecutive query rows that share one block table and whose visible lengths
// grow by one per row (the rows of a prefill chunk). One CUDA block handles 128 query rows of one
// head: 8 warps, 16 rows each. Keys and values are copied tile by tile (64 keys) from the paged
// FP16 cache into shared memory with a double-buffered cp.async pipeline. S = Q K^T and O = P V run
// on tensor cores (mma.m16n8k16, FP16 inputs, FP32 accumulation); the softmax is kept online in
// registers, and the probabilities feed the second matrix product straight from the accumulator
// registers (the FlashAttention-2 data flow).
//
// Supported: f16 cache, key = value width 64 or 128, GQA, sinks. Window, chunk, soft-capping and
// ALiBi use the generic kernels.
#include "common.cuh"
#include "fllm_cuda_internal.h"
#include "mma_util.cuh"

#define A3_WARPS 8
#define A3_BQ (A3_WARPS * 16)
#define A3_BK 64
#define A3_NEG_INF (-INFINITY)

__device__ __forceinline__ void a3_cp16(void *smem, const void *gmem, bool valid) { cp_async16(smem, gmem, valid); }
__device__ __forceinline__ void a3_commit() { cp_async_commit(); }
template <int N>
__device__ __forceinline__ void a3_wait() { cp_async_wait<N>(); }

__device__ __forceinline__ void a3_ldsm(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, const void *p) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a));
}
__device__ __forceinline__ void a3_ldsm_t(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, const void *p) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a));
}
__device__ __forceinline__ void a3_mma(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) { mma16816(c, a, b0, b1); }
__device__ __forceinline__ uint32_t a3_pack(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t *>(&h);
}

struct Attn3Args {
    const float *q;
    const uint16_t *kpool, *vpool;
    float *out;
    const int64_t *tables, *lengths;
    const float *sinks;
    int rows, heads, kv_heads, stride, block_size;
    float scale_log2;  // softmax scale * log2(e)
    int kscale_base, vscale_base;  // q8_0 cache: first scale (16-bit words) of the layer's key / value pool
};

template <int D, bool Q8>
__global__ void __launch_bounds__(A3_WARPS * 32, 1) attn3_kernel(Attn3Args a) {
#if FLLM_TC
    constexpr int CH = D / 8;           // 16-byte chunks per key row
    constexpr int KB = D / 16;          // k16 steps of the QK^T product
    constexpr int DB = D / 8;           // 8-wide output blocks
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char *Ks = smem;                                          // [2][64][D] halves, swizzled chunks
    unsigned char *Vs = smem + (size_t)2 * A3_BK * D * 2;
    const int h = blockIdx.y, q0 = blockIdx.x * A3_BQ;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int g = lane >> 2, t4 = lane & 3;
    const int kh = h / (a.heads / a.kv_heads);
    const size_t kvw = (size_t)a.kv_heads * D;
    const int row_a = min(q0 + warp * 16 + g, a.rows - 1), row_b = min(q0 + warp * 16 + g + 8, a.rows - 1);
    const bool live_a = q0 + warp * 16 + g < a.rows, live_b = q0 + warp * 16 + g + 8 < a.rows;
    const int n_a = (int)a.lengths[row_a], n_b = (int)a.lengths[row_b];
    const int last = min(q0 + A3_BQ, a.rows) - 1;
    const int kmax = (int)a.lengths[last];
    const int tiles = (kmax + A3_BK - 1) / A3_BK;
    const int64_t *table = a.tables;  // all rows of a run share the first row's block table

    // Q fragments (scaled), straight from the FP32 activations
    uint32_t qf[KB][4];
#pragma unroll
    for (int kc = 0; kc < KB; kc++) {
        const float *pa = a.q + ((size_t)row_a * a.heads + h) * D + kc * 16 + 2 * t4;
        const float *pb = a.q + ((size_t)row_b * a.heads + h) * D + kc * 16 + 2 * t4;
        qf[kc][0] = a3_pack(pa[0] * a.scale_log2, pa[1] * a.scale_log2);
        qf[kc][1] = a3_pack(pb[0] * a.scale_log2, pb[1] * a.scale_log2);
        qf[kc][2] = a3_pack(pa[8] * a.scale_log2, pa[9] * a.scale_log2);
        qf[kc][3] = a3_pack(pb[8] * a.scale_log2, pb[9] * a.scale_log2);
    }

    float o[DB][4];
#pragma unroll
    for (int i = 0; i < DB; i++)
#pragma unroll
        for (int c = 0; c < 4; c++) o[i][c] = 0.f;
    float m_a = A3_NEG_INF, m_b = A3_NEG_INF, l_a = 0.f, l_b = 0.f;
    if (a.sinks) {
        const float sk = a.sinks[h] * 1.4426950408889634f;
        m_a = m_b = sk;
        l_a = l_b = 1.f;
    }

    auto issue = [&](int tile, int buf) {
        for (int c = tid; c < A3_BK * CH; c += A3_WARPS * 32) {
            const int kr = c / CH, chunk = c % CH;
            const int p = tile * A3_BK + kr;
            const bool ok = p < kmax;
            size_t slot = 0;
            if (ok) slot = (size_t)table[p / a.block_size] * a.block_size + p % a.block_size;
            const size_t off = slot * kvw + (size_t)kh * D + chunk * 8;
            const int sw = (chunk ^ (kr & 7)) * 16;
            a3_cp16(Ks + ((size_t)buf * A3_BK + kr) * D * 2 + sw, a.kpool + off, ok);
            a3_cp16(Vs + ((size_t)buf * A3_BK + kr) * D * 2 + sw, a.vpool + off, ok);
        }
    };

    // q8_0 cache: the int8 rows are expanded to half precision on the way into shared memory
    // (registers hold the next tile while the current one is multiplied)
    constexpr int QCH = A3_BK * CH / (A3_WARPS * 32);  // 8-value chunks per thread and tile
    uint2 kq[Q8 ? QCH : 1], vq[Q8 ? QCH : 1];
    float ksc[Q8 ? QCH : 1], vsc[Q8 ? QCH : 1];
    auto q8_load = [&](int tile) {
#pragma unroll
        for (int i = 0; i < QCH; i++) {
            const int c = tid + i * (A3_WARPS * 32);
            const int kr = c / CH, chunk = c % CH;
            const int p = tile * A3_BK + kr;
            const bool ok = p < kmax;
            size_t slot = 0;
            if (ok) slot = (size_t)table[p / a.block_size] * a.block_size + p % a.block_size;
            const size_t byte = slot * kvw + (size_t)kh * D + chunk * 8;
            kq[i] = ok ? __ldg(reinterpret_cast<const uint2 *>(reinterpret_cast<const int8_t *>(a.kpool) + byte)) : make_uint2(0, 0);
            vq[i] = ok ? __ldg(reinterpret_cast<const uint2 *>(reinterpret_cast<const int8_t *>(a.vpool) + byte)) : make_uint2(0, 0);
            const size_t sidx = slot * (kvw / 32) + ((size_t)kh * D + chunk * 8) / 32;
            ksc[i] = ok ? h2f_bits(a.kpool[a.kscale_base + sidx]) : 0.f;
            vsc[i] = ok ? h2f_bits(a.vpool[a.vscale_base + sidx]) : 0.f;
        }
    };
    auto q8_store = [&](int buf) {
#pragma unroll
        for (int i = 0; i < QCH; i++) {
            const int c = tid + i * (A3_WARPS * 32);
            const int kr = c / CH, chunk = c % CH;
            const int sw = (chunk ^ (kr & 7)) * 16;
#pragma unroll
            for (int w = 0; w < 2; w++) {
                const uint2 &src = w == 0 ? kq[i] : vq[i];
                const float sc = w == 0 ? ksc[i] : vsc[i];
                uint32_t out[4];
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const uint32_t word = j < 2 ? src.x : src.y;
                    const int sh = (j & 1) * 16;
                    const float lo = (float)(int8_t)((word >> sh) & 0xffu) * sc, hi = (float)(int8_t)((word >> (sh + 8)) & 0xffu) * sc;
                    out[j] = a3_pack(lo, hi);
                }
                unsigned char *dst = (w == 0 ? Ks : Vs) + ((size_t)buf * A3_BK + kr) * D * 2 + sw;
                *reinterpret_cast<uint4 *>(dst) = make_uint4(out[0], out[1], out[2], out[3]);
            }
        }
    };

    if (Q8) {
        q8_load(0);
        q8_store(0);
        __syncthreads();
    } else {
        issue(0, 0);
        a3_commit();
    }
    for (int t = 0; t < tiles; t++) {
        if (Q8) {
            if (t + 1 < tiles) q8_load(t + 1);
        } else {
            if (t + 1 < tiles) issue(t + 1, (t + 1) & 1);
            a3_commit();
            a3_wait<1>();
            __syncthreads();
        }
        const int buf = t & 1;
        const unsigned char *kb = Ks + (size_t)buf * A3_BK * D * 2;
        const unsigned char *vb = Vs + (size_t)buf * A3_BK * D * 2;

        float s[8][4];
#pragma unroll
        for (int nb = 0; nb < 8; nb++)
#pragma unroll
            for (int c = 0; c < 4; c++) s[nb][c] = 0.f;
#pragma unroll
        for (int kc = 0; kc < KB; kc++) {
#pragma unroll
            for (int np = 0; np < 4; np++) {
                uint32_t b0, b1, b2, b3;
                const int kr = np * 16 + (lane & 7) + ((lane >> 4) << 3);
                const int chunk = kc * 2 + ((lane >> 3) & 1);
                a3_ldsm(b0, b1, b2, b3, kb + (size_t)kr * D * 2 + ((chunk ^ (kr & 7)) * 16));
                a3_mma(s[np * 2], qf[kc], b0, b1);
                a3_mma(s[np * 2 + 1], qf[kc], b2, b3);
            }
        }

        // causal mask: key p is visible to row i iff p < n_i
        const int pbase = t * A3_BK;
        if (pbase + A3_BK > min(n_a, n_b)) {
#pragma unroll
            for (int nb = 0; nb < 8; nb++)
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    const int p = pbase + nb * 8 + 2 * t4 + (c & 1);
                    const int n = (c >> 1) ? n_b : n_a;
                    if (p >= n) s[nb][c] = A3_NEG_INF;
                }
        }

        float tmax_a = A3_NEG_INF, tmax_b = A3_NEG_INF;
#pragma unroll
        for (int nb = 0; nb < 8; nb++) {
            tmax_a = fmaxf(tmax_a, fmaxf(s[nb][0], s[nb][1]));
            tmax_b = fmaxf(tmax_b, fmaxf(s[nb][2], s[nb][3]));
        }
        tmax_a = fmaxf(tmax_a, __shfl_xor_sync(0xffffffffu, tmax_a, 1));
        tmax_a = fmaxf(tmax_a, __shfl_xor_sync(0xffffffffu, tmax_a, 2));
        tmax_b = fmaxf(tmax_b, __shfl_xor_sync(0xffffffffu, tmax_b, 1));
        tmax_b = fmaxf(tmax_b, __shfl_xor_sync(0xffffffffu, tmax_b, 2));
        const float mn_a = fmaxf(m_a, tmax_a), mn_b = fmaxf(m_b, tmax_b);
        const float ms_a = mn_a == A3_NEG_INF ? 0.f : mn_a, ms_b = mn_b == A3_NEG_INF ? 0.f : mn_b;
        const float corr_a = exp2f(m_a - ms_a), corr_b = exp2f(m_b - ms_b);
        m_a = mn_a;
        m_b = mn_b;
        l_a *= corr_a;
        l_b *= corr_b;
#pragma unroll
        for (int i = 0; i < DB; i++) {
            o[i][0] *= corr_a; o[i][1] *= corr_a;
            o[i][2] *= corr_b; o[i][3] *= corr_b;
        }
        uint32_t pf[4][4];
#pragma unroll
        for (int nb = 0; nb < 8; nb++) {
            const float p0 = exp2f(s[nb][0] - ms_a), p1 = exp2f(s[nb][1] - ms_a);
            const float p2 = exp2f(s[nb][2] - ms_b), p3 = exp2f(s[nb][3] - ms_b);
            l_a += p0 + p1;
            l_b += p2 + p3;
            const int kb2 = nb >> 1, hi = nb & 1;
            pf[kb2][hi ? 2 : 0] = a3_pack(p0, p1);
            pf[kb2][hi ? 3 : 1] = a3_pack(p2, p3);
        }
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
#pragma unroll
            for (int dp = 0; dp < DB / 2; dp++) {
                uint32_t b0, b1, b2, b3;
                const int kr = kk * 16 + (lane & 7) + (((lane >> 3) & 1) << 3);
                const int chunk = dp * 2 + (lane >> 4);
                a3_ldsm_t(b0, b1, b2, b3, vb + (size_t)kr * D * 2 + ((chunk ^ (kr & 7)) * 16));
                a3_mma(o[dp * 2], pf[kk], b0, b1);
                a3_mma(o[dp * 2 + 1], pf[kk], b2, b3);
            }
        }
        if (Q8 && t + 1 < tiles) q8_store((t + 1) & 1);
        __syncthreads();
    }
    a3_wait<0>();

    l_a += __shfl_xor_sync(0xffffffffu, l_a, 1);
    l_a += __shfl_xor_sync(0xffffffffu, l_a, 2);
    l_b += __shfl_xor_sync(0xffffffffu, l_b, 1);
    l_b += __shfl_xor_sync(0xffffffffu, l_b, 2);
    const float inv_a = 1.f / l_a, inv_b = 1.f / l_b;
    float *oa = a.out + ((size_t)row_a * a.heads + h) * D;
    float *ob = a.out + ((size_t)row_b * a.heads + h) * D;
#pragma unroll
    for (int i = 0; i < DB; i++) {
        if (live_a) *reinterpret_cast<float2 *>(oa + i * 8 + 2 * t4) = make_float2(o[i][0] * inv_a, o[i][1] * inv_a);
        if (live_b) *reinterpret_cast<float2 *>(ob + i * 8 + 2 * t4) = make_float2(o[i][2] * inv_b, o[i][3] * inv_b);
    }
#endif
}

template <int D, bool Q8>
static int launch(cudaStream_t s, const Attn3Args &a) {
    constexpr size_t smem = (size_t)2 * 2 * A3_BK * D * 2;
    if (smem + 2048 > (size_t)fllm_smem_limit()) return -1;
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(attn3_kernel<D, Q8>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        configured = true;
    }
    attn3_kernel<D, Q8><<<dim3((a.rows + A3_BQ - 1) / A3_BQ, a.heads), A3_WARPS * 32, smem, s>>>(a);
    return fllm_check_launch("attention3");
}

// `rows` consecutive query rows of one sequence (shared block table `tables`, lengths growing by
// one). Returns -1 when the shape is not covered.
int fllm_attention3_try(cudaStream_t s, const float *q, const uint16_t *kpool, const uint16_t *vpool, float *out,
                        const int64_t *tables, const int64_t *lengths, const float *sinks, int rows, int heads,
                        int kv_heads, int dk, int dv, int stride, int block_size, int kind, float scale,
                        int kscale_base, int vscale_base, int kv_len) {
    if (fllm_cc() < 75) return -1;
    if ((kind != 1 && kind != 2) || dk != dv || (dk != 64 && dk != 128) || rows < 16) return -1;
    if ((reinterpret_cast<uintptr_t>(kpool) & 15) != 0 || (reinterpret_cast<uintptr_t>(vpool) & 15) != 0 ||
        ((size_t)kv_heads * dk * (kind == 1 ? 2 : 1)) % 16 != 0)
        return -1;
    // FlashInfer's prefill kernel (Apache-2.0) is faster where it applies: f16 cache, no attention sinks
    if (kind == 1 && sinks == nullptr) {
        const int rc = fllm_flashinfer_prefill_try(s, q, kpool, vpool, out, tables, rows, kv_len, heads, kv_heads, dk, dv,
                                                   block_size, scale);
        if (rc >= 0) return rc;
    }
    Attn3Args a{q, kpool, vpool, out, tables, lengths, sinks, rows, heads, kv_heads, stride, block_size,
                scale * 1.4426950408889634f, kscale_base, vscale_base};
    if (kind == 2) return dk == 64 ? launch<64, true>(s, a) : launch<128, true>(s, a);
    return dk == 64 ? launch<64, false>(s, a) : launch<128, false>(s, a);
}
