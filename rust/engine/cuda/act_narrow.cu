// fLLM CUDA backend: activations in a narrow type (FLLM_ACT_DTYPE=f16|bf16).
//
// Until now the residual stream, the norm outputs, the attention output and the feed-forward intermediates were FP32 and were converted to FP16 in
// front of every Marlin / tensor-core GEMM and back after it. The kernels here keep them in FP16 or BF16 (vLLM's layout): every kernel reads the narrow
// values, computes in FP32 registers (sums of squares, residual adds, rotations, SiLU) and writes the narrow type once. The residual add of
// `x += scale * branch` is rounded to the narrow type before the norm reads it, so the fused add + norm equals the two kernels it replaces bit for bit.
//
// dt (the activation type code used by all entry points): 0 FP32, 1 FP16, 2 BF16. Buffers are plain arrays of that type; the host side allocates
// them with the FP32 size, so a narrow buffer is half full. The rotary tables, norm weights and biases stay FP32.
//
// Original code of fLLM (Apache-2.0). docs/activation-dtype.md has the tensor-by-tensor account.
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "common.cuh"
#include "fllm_cuda_internal.h"
#include "pdl.cuh"

namespace {

// ---- element types ----------------------------------------------------------------------------------------------------------------------
template <int DT>
struct Elem;
template <>
struct Elem<0> {
    using T = float;
    static __device__ __forceinline__ float ld(const float &v) { return v; }
    static __device__ __forceinline__ float st(float v) { return v; }
};
template <>
struct Elem<1> {
    using T = __half;
    static __device__ __forceinline__ float ld(const __half &v) { return __half2float(v); }
    static __device__ __forceinline__ __half st(float v) { return __float2half_rn(v); }
};
template <>
struct Elem<2> {
    using T = __nv_bfloat16;
    static __device__ __forceinline__ float ld(const __nv_bfloat16 &v) { return __bfloat162float(v); }
    static __device__ __forceinline__ __nv_bfloat16 st(float v) { return __float2bfloat16_rn(v); }
};

// 8 elements of a 16-byte aligned narrow array
template <int DT>
__device__ __forceinline__ void load8(const typename Elem<DT>::T *p, float (&f)[8]) {
    const uint4 u = *reinterpret_cast<const uint4 *>(p);
    const typename Elem<DT>::T *t = reinterpret_cast<const typename Elem<DT>::T *>(&u);
#pragma unroll
    for (int i = 0; i < 8; i++) f[i] = Elem<DT>::ld(t[i]);
}
template <int DT>
__device__ __forceinline__ void store8(typename Elem<DT>::T *p, const float (&f)[8]) {
    uint4 u;
    typename Elem<DT>::T *t = reinterpret_cast<typename Elem<DT>::T *>(&u);
#pragma unroll
    for (int i = 0; i < 8; i++) t[i] = Elem<DT>::st(f[i]);
    *reinterpret_cast<uint4 *>(p) = u;
}
// the value as stored in the narrow type, back in FP32
template <int DT>
__device__ __forceinline__ float round_to(float v) {
    return Elem<DT>::ld(Elem<DT>::st(v));
}

__device__ __forceinline__ float block_sum_act(float v, float *scratch) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float total = 0.f;
    for (int i = 0; i < warps; i++) total += scratch[i];
    return total;
}

unsigned grid_for(size_t n, int block) {
    size_t g = (n + block - 1) / block;
    return (unsigned)(g > 65535 ? 65535 : (g == 0 ? 1 : g));
}

// ---- conversion -------------------------------------------------------------------------------------------------------------------------
template <int S, int D>
__global__ void convert_kernel(const typename Elem<S>::T *__restrict__ src, typename Elem<D>::T *__restrict__ dst, size_t n) {
    fllm_pdl_wait();
    fllm_pdl_trigger();
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        dst[i] = Elem<D>::st(Elem<S>::ld(src[i]));
}

// ---- RMS norm (optionally with the residual add in front) -------------------------------------------------------------------------------
// One block per row; the row is held in registers (8 values per thread and step, PER steps). With ADD: x = round(x + scale * branch) is written back
// and the norm reads the rounded value.
template <int DT, int PER, bool ADD>
__global__ void __launch_bounds__(256)
rms_norm_act_kernel(typename Elem<DT>::T *x, const typename Elem<DT>::T *__restrict__ branch, float scale, const float *__restrict__ w,
                    typename Elem<DT>::T *__restrict__ out, int cols8, float eps, float inv_cols) {
    fllm_pdl_wait();
    fllm_pdl_trigger();
    __shared__ float scratch[32];
    using T = typename Elem<DT>::T;
    T *xr = x + (size_t)blockIdx.x * cols8 * 8;
    T *orow = out + (size_t)blockIdx.x * cols8 * 8;
    const T *br = ADD ? branch + (size_t)blockIdx.x * cols8 * 8 : nullptr;
    float v[PER][8];
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < PER; i++) {
        const int c = threadIdx.x + i * 256;
        if (c < cols8) {
            load8<DT>(xr + (size_t)c * 8, v[i]);
            if (ADD) {
                float b[8];
                load8<DT>(br + (size_t)c * 8, b);
#pragma unroll
                for (int j = 0; j < 8; j++) v[i][j] = round_to<DT>(v[i][j] + scale * b[j]);
                store8<DT>(xr + (size_t)c * 8, v[i]);
            }
        } else {
#pragma unroll
            for (int j = 0; j < 8; j++) v[i][j] = 0.f;
        }
#pragma unroll
        for (int j = 0; j < 8; j++) sum = fmaf(v[i][j], v[i][j], sum);
    }
    const float total = block_sum_act(sum, scratch);
    const float inv = 1.0f / sqrtf(total * inv_cols + eps);
#pragma unroll
    for (int i = 0; i < PER; i++) {
        const int c = threadIdx.x + i * 256;
        if (c < cols8) {
            const float4 g0 = __ldg(reinterpret_cast<const float4 *>(w) + 2 * c), g1 = __ldg(reinterpret_cast<const float4 *>(w) + 2 * c + 1);
            const float g[8] = {g0.x, g0.y, g0.z, g0.w, g1.x, g1.y, g1.z, g1.w};
            float o[8];
#pragma unroll
            for (int j = 0; j < 8; j++) o[j] = v[i][j] * (inv * g[j]);
            store8<DT>(orow + (size_t)c * 8, o);
        }
    }
}

// any width, any alignment
template <int DT, bool ADD>
__global__ void rms_norm_act_generic_kernel(typename Elem<DT>::T *x, const typename Elem<DT>::T *__restrict__ branch, float scale,
                                            const float *__restrict__ w, typename Elem<DT>::T *__restrict__ out, int cols, float eps) {
    fllm_pdl_wait();
    fllm_pdl_trigger();
    __shared__ float scratch[32];
    typename Elem<DT>::T *xr = x + (size_t)blockIdx.x * cols;
    typename Elem<DT>::T *orow = out + (size_t)blockIdx.x * cols;
    float sum = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        float v = Elem<DT>::ld(xr[c]);
        if (ADD) {
            v = round_to<DT>(v + scale * Elem<DT>::ld(branch[(size_t)blockIdx.x * cols + c]));
            xr[c] = Elem<DT>::st(v);
        }
        sum = fmaf(v, v, sum);
    }
    const float total = block_sum_act(sum, scratch);
    const float inv = 1.0f / sqrtf(total / (float)cols + eps);
    for (int c = threadIdx.x; c < cols; c += blockDim.x) orow[c] = Elem<DT>::st(Elem<DT>::ld(xr[c]) * (inv * w[c]));
}

// ---- residual add, gated activation -------------------------------------------------------------------------------------------------------
template <int DT>
__global__ void residual_act_kernel(typename Elem<DT>::T *x, const typename Elem<DT>::T *__restrict__ res, size_t count, size_t vec, float scale) {
    fllm_pdl_wait();
    fllm_pdl_trigger();
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < vec; i += (size_t)gridDim.x * blockDim.x) {
        float a[8], b[8];
        load8<DT>(x + i * 8, a);
        load8<DT>(res + i * 8, b);
#pragma unroll
        for (int j = 0; j < 8; j++) a[j] = a[j] + b[j] * scale;
        store8<DT>(x + i * 8, a);
    }
    for (size_t i = vec * 8 + (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x)
        x[i] = Elem<DT>::st(Elem<DT>::ld(x[i]) + Elem<DT>::ld(res[i]) * scale);
}

template <int DT>
__global__ void glu_act_kernel(const typename Elem<DT>::T *__restrict__ gate, const typename Elem<DT>::T *__restrict__ up,
                               typename Elem<DT>::T *__restrict__ out, size_t count, size_t vec, int mode) {
    fllm_pdl_wait();
    fllm_pdl_trigger();
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < vec; i += (size_t)gridDim.x * blockDim.x) {
        float g[8], u[8], o[8];
        load8<DT>(gate + i * 8, g);
        load8<DT>(up + i * 8, u);
#pragma unroll
        for (int j = 0; j < 8; j++) o[j] = fllm_glu(g[j], u[j], mode);
        store8<DT>(out + i * 8, o);
    }
    for (size_t i = vec * 8 + (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x)
        out[i] = Elem<DT>::st(fllm_glu(Elem<DT>::ld(gate[i]), Elem<DT>::ld(up[i]), mode));
}

// y[b][r] += bias[r] (the bias stays FP32)
template <int DT>
__global__ void bias_act_kernel(typename Elem<DT>::T *y, const float *__restrict__ bias, int rows, size_t count) {
    fllm_pdl_wait();
    fllm_pdl_trigger();
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x)
        y[i] = Elem<DT>::st(Elem<DT>::ld(y[i]) + bias[i % rows]);
}

// ---- rotary embedding, KV cache write ----------------------------------------------------------------------------------------------------
// Standalone rotation of q and k in place; one thread per pair (the layout of rope_kernel).
template <int DT>
__global__ void rope_act_kernel(typename Elem<DT>::T *q, typename Elem<DT>::T *k, const int32_t *pos, const float *cs, const float *sn, int rows, int heads,
                                int kv_heads, int dim, int half_, int neox) {
    fllm_pdl_wait();
    fllm_pdl_trigger();
    const int per_row = (heads + kv_heads) * half_;
    const size_t total = (size_t)rows * per_row;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const int r = (int)(i / per_row), rem = (int)(i % per_row);
        const int h = rem / half_, d = rem % half_;
        const int p = pos[r];
        const float c = cs[(size_t)p * half_ + d], s = sn[(size_t)p * half_ + d];
        const int ia = neox ? d : 2 * d, ib = neox ? d + half_ : 2 * d + 1;
        typename Elem<DT>::T *base = h < heads ? q + ((size_t)r * heads + h) * dim : k + ((size_t)r * kv_heads + (h - heads)) * dim;
        const float a = Elem<DT>::ld(base[ia]), b = Elem<DT>::ld(base[ib]);
        float ra, rb;
        rope_pair(a, b, c, s, ra, rb);
        base[ia] = Elem<DT>::st(ra);
        base[ib] = Elem<DT>::st(rb);
    }
}

// f16 KV pool (16-bit words) from a narrow source
template <int DT>
__global__ void kv_write_act_kernel(const typename Elem<DT>::T *src, uint16_t *pool, const int32_t *slots, int rows, int kv) {
    fllm_pdl_wait();
    fllm_pdl_trigger();
    const size_t total = (size_t)rows * kv;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const size_t slot = (size_t)slots[i / kv];
        pool[slot * kv + i % kv] = f2h_bits(Elem<DT>::ld(src[i]));
    }
}

// QK norm (per head, before the rotation) + rotary embedding + KV cache write, one warp per (row, head). Slots 0 .. heads - 1 are the query heads (read from q_src,
// rotated into q_dst), heads .. heads + kv_heads - 1 the key heads (normed, rotated and written to the f16 pool), the rest the value heads (copied to the pool).
// q, k and v are read with their own row strides (elements per row): dense arrays, or the three column blocks of one merged q/k/v GEMM result (FLLM_MERGE_PROJ).
// `qw` / `kw`: the norm weights of q and k (width dk), nullptr for no norm. `qb` / `kb` / `vb`: FP32 biases of the projections (nullptr for none), added before the
// norm. The norm is that of rms_norm_vec_kernel (x * (inv * w), inv from the sum of squares of the head); the rotation is rope_pair.
constexpr int kMaxPairs = 8, kMaxTail = 16;  // heads of up to 512 values
template <int DT>
__global__ void __launch_bounds__(128)
rope_kv_act_kernel(typename Elem<DT>::T *q_dst, const typename Elem<DT>::T *q_src, int q_stride, const typename Elem<DT>::T *k, int k_stride,
                   const typename Elem<DT>::T *v, int v_stride, uint16_t *pool_k, uint16_t *pool_v, const int32_t *pos, const int32_t *slots, const float *cs,
                   const float *sn, const float *qw, const float *kw, const float *qb, const float *kb, const float *vb, float eps, int rows, int heads,
                   int kv_heads, int dk, int dv, int half_, int neox) {
    fllm_pdl_wait();
    fllm_pdl_trigger();
    const int warp = (int)(((size_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5), lane = threadIdx.x & 31;
    const int per_row = heads + 2 * kv_heads;
    if (warp >= rows * per_row) return;
    const int r = warp / per_row, slot_h = warp % per_row;
    const size_t slot = (size_t)slots[r];
    if (slot_h >= heads + kv_heads) {  // value head: copy
        const int hv = slot_h - heads - kv_heads;
        const typename Elem<DT>::T *src = v + (size_t)r * v_stride + (size_t)hv * dv;
        uint16_t *dst = pool_v + slot * (size_t)kv_heads * dv + (size_t)hv * dv;
        for (int e = lane; e < dv; e += 32) dst[e] = f2h_bits(Elem<DT>::ld(src[e]) + (vb ? vb[(size_t)hv * dv + e] : 0.f));
        return;
    }
    const bool is_q = slot_h < heads;
    const int hk = slot_h - heads;
    const typename Elem<DT>::T *src = is_q ? q_src + (size_t)r * q_stride + (size_t)slot_h * dk : k + (size_t)r * k_stride + (size_t)hk * dk;
    const float *w = is_q ? qw : kw;
    const float *bias = (is_q ? qb : kb) ? (is_q ? qb + (size_t)slot_h * dk : kb + (size_t)hk * dk) : nullptr;
    const int tail = dk - 2 * half_;
    float a[kMaxPairs], b[kMaxPairs], t[kMaxTail];
    float sum = 0.f;
#pragma unroll
    for (int j = 0; j < kMaxPairs; j++) {
        const int d = lane + 32 * j;
        a[j] = b[j] = 0.f;
        if (d < half_) {
            const int ia = neox ? d : 2 * d, ib = neox ? d + half_ : 2 * d + 1;
            a[j] = Elem<DT>::ld(src[ia]) + (bias ? bias[ia] : 0.f);
            b[j] = Elem<DT>::ld(src[ib]) + (bias ? bias[ib] : 0.f);
            sum = fmaf(a[j], a[j], fmaf(b[j], b[j], sum));
        }
    }
#pragma unroll
    for (int j = 0; j < kMaxTail; j++) {
        const int e = lane + 32 * j;
        t[j] = 0.f;
        if (e < tail) {
            t[j] = Elem<DT>::ld(src[2 * half_ + e]) + (bias ? bias[2 * half_ + e] : 0.f);
            sum = fmaf(t[j], t[j], sum);
        }
    }
    if (w != nullptr) {
        const float inv = 1.0f / sqrtf(warp_sum(sum) * (1.0f / (float)dk) + eps);
#pragma unroll
        for (int j = 0; j < kMaxPairs; j++) {
            const int d = lane + 32 * j;
            if (d < half_) {
                const int ia = neox ? d : 2 * d, ib = neox ? d + half_ : 2 * d + 1;
                a[j] = a[j] * (inv * w[ia]);
                b[j] = b[j] * (inv * w[ib]);
            }
        }
#pragma unroll
        for (int j = 0; j < kMaxTail; j++) {
            const int e = lane + 32 * j;
            if (e < tail) t[j] = t[j] * (inv * w[2 * half_ + e]);
        }
    }
    const int p = pos[r];
    typename Elem<DT>::T *qdst = is_q ? q_dst + ((size_t)r * heads + slot_h) * dk : nullptr;
    uint16_t *kdst = is_q ? nullptr : pool_k + slot * (size_t)kv_heads * dk + (size_t)hk * dk;
#pragma unroll
    for (int j = 0; j < kMaxPairs; j++) {
        const int d = lane + 32 * j;
        if (d < half_) {
            const int ia = neox ? d : 2 * d, ib = neox ? d + half_ : 2 * d + 1;
            float ra, rb;
            rope_pair(a[j], b[j], cs[(size_t)p * half_ + d], sn[(size_t)p * half_ + d], ra, rb);
            if (is_q) {
                qdst[ia] = Elem<DT>::st(ra);
                qdst[ib] = Elem<DT>::st(rb);
            } else {
                kdst[ia] = f2h_bits(ra);
                kdst[ib] = f2h_bits(rb);
            }
        }
    }
    if (tail > 0) {
#pragma unroll
        for (int j = 0; j < kMaxTail; j++) {
            const int e = lane + 32 * j;
            if (e < tail) {
                if (is_q) qdst[2 * half_ + e] = Elem<DT>::st(t[j]);
                else kdst[2 * half_ + e] = f2h_bits(t[j]);
            }
        }
    }
}

// Columns of one row-major array [rows, n0 + n1 + n2] to three dense arrays (the q, k and v blocks of a merged projection). Elements of E bytes (2 or 4).
template <typename E>
__global__ void split3_kernel(const E *__restrict__ src, E *__restrict__ d0, E *__restrict__ d1, E *__restrict__ d2, int n0, int n1, int n2, size_t rows) {
    fllm_pdl_wait();
    fllm_pdl_trigger();
    const int n = n0 + n1 + n2;
    const size_t total = rows * (size_t)n;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const size_t r = i / n;
        const int c = (int)(i % n);
        const E value = src[i];
        if (c < n0) d0[r * n0 + c] = value;
        else if (c < n0 + n1) d1[r * n1 + (c - n0)] = value;
        else d2[r * n2 + (c - n0 - n1)] = value;
    }
}

// act[r][i] = glu(gu[r][i], gu[r][ff + i]): gate and up as the two column blocks of one merged GEMM result
template <int DT>
__global__ void glu_merged_act_kernel(const typename Elem<DT>::T *__restrict__ gu, typename Elem<DT>::T *__restrict__ out, size_t rows, int ff, int vec, int mode) {
    fllm_pdl_wait();
    fllm_pdl_trigger();
    const size_t per_row = vec ? (size_t)ff / 8 : (size_t)ff;
    const size_t total = rows * per_row;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const size_t r = i / per_row, c = i % per_row;
        if (vec) {
            float g[8], u[8], o[8];
            load8<DT>(gu + r * 2 * (size_t)ff + c * 8, g);
            load8<DT>(gu + r * 2 * (size_t)ff + ff + c * 8, u);
#pragma unroll
            for (int j = 0; j < 8; j++) o[j] = fllm_glu(g[j], u[j], mode);
            store8<DT>(out + r * (size_t)ff + c * 8, o);
        } else {
            out[r * (size_t)ff + c] = Elem<DT>::st(fllm_glu(Elem<DT>::ld(gu[r * 2 * (size_t)ff + c]), Elem<DT>::ld(gu[r * 2 * (size_t)ff + ff + c]), mode));
        }
    }
}

// ---- activation quantization from the narrow type ------------------------------------------------------------------------------------------
// quantize_x8_kernel (mmvq.cu) reading FP16 / BF16 instead of FP32: xq[n][k] = round(x * 127 / amax) per group of 32 values, xs[n][g] = {amax / 127, amax / 127 *
// sum(xq)}; one warp per group. The same arithmetic as the FP32 kernel, so the integer dot-product GEMVs (mmvq.cu) read the result as they read their own.
template <int DT>
__global__ void quantize_x8_act_kernel(const typename Elem<DT>::T *__restrict__ x, int8_t *__restrict__ xq, float2 *__restrict__ xs, size_t groups) {
    fllm_pdl_wait();
    fllm_pdl_trigger();
    const size_t gid = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (gid >= groups) return;
    const float v = Elem<DT>::ld(x[gid * 32 + lane]);
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float dx = amax / 127.f;
    const int q = amax > 0.f ? __float2int_rn(v * (127.f / amax)) : 0;
    int s = q;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    xq[gid * 32 + lane] = (int8_t)q;
    if (lane == 0) xs[gid] = make_float2(dx, dx * (float)s);
}

// ---- checks --------------------------------------------------------------------------------------------------------------------------
// out[0]: largest |x| of the finite values (float bits, non-negative so the int order is the float order), out[1]: number of NaN / Inf values
template <int DT>
__global__ void check_act_kernel(const typename Elem<DT>::T *x, size_t count, int *out) {
    fllm_pdl_wait();
    fllm_pdl_trigger();
    float m = 0.f;
    int bad = 0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x) {
        const float v = Elem<DT>::ld(x[i]);
        if (isfinite(v)) m = fmaxf(m, fabsf(v));
        else bad++;
    }
    m = warp_max(m);
    bad = (int)warp_sum((float)bad);
    if ((threadIdx.x & 31) == 0) {
        atomicMax(out, __float_as_int(m));
        if (bad) atomicAdd(out + 1, bad);
    }
}

// ---- dispatch helpers ----------------------------------------------------------------------------------------------------------------------
#define FLLM_ACT_SWITCH(dt, CALL) \
    switch (dt) {                 \
        case 1: { constexpr int DT = 1; CALL; break; } \
        case 2: { constexpr int DT = 2; CALL; break; } \
        default: return fllm_set_error("activation type %d is not FP16 or BF16", dt); \
    }

template <int S, int D>
int convert_launch(cudaStream_t s, const void *src, void *dst, size_t n) {
    fllm_launch_pdl(convert_kernel<S, D>, dim3(grid_for(n, 256)), dim3(256), 0, s, static_cast<const typename Elem<S>::T *>(src), static_cast<typename Elem<D>::T *>(dst), n);
    return fllm_check_launch("act convert");
}

}  // namespace

int fllm_na_convert_launch(cudaStream_t s, const void *src, int src_dt, void *dst, int dst_dt, size_t n) {
    if (n == 0) return 0;
    if (src_dt == dst_dt) return fllm_set_error("act convert: source and target type are the same");
#define C(S, D) if (src_dt == S && dst_dt == D) return convert_launch<S, D>(s, src, dst, n)
    C(0, 1); C(0, 2); C(1, 0); C(2, 0); C(1, 2); C(2, 1);
#undef C
    return fllm_set_error("act convert: unsupported types %d -> %d", src_dt, dst_dt);
}

// x (rows x cols, narrow, updated when branch != null: x = round(x + scale * branch)), out = rms_norm(x) * w. Returns 3 when the operands are not supported
// (nothing launched; the caller runs the generic path).
int fllm_na_add_rms_norm_launch(cudaStream_t s, int dt, void *x, const void *branch, float scale, const float *w, void *out, int rows, int cols, float eps) {
    if (rows <= 0) return 0;
    const size_t es = 2;
    const bool aligned = ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(out) | reinterpret_cast<uintptr_t>(w) |
                           (branch ? reinterpret_cast<uintptr_t>(branch) : 0)) & 15) == 0;
    const bool vec = aligned && cols % 8 == 0 && cols <= 8 * 256 * 8;
    const float inv_cols = 1.0f / (float)cols;
    (void)es;
    if (vec) {
        const int c8 = cols / 8;
#define FLLM_N(DTV, PERV, ADDV) \
    fllm_launch_pdl(rms_norm_act_kernel<DTV, PERV, ADDV>, dim3(rows), dim3(256), 0, s, static_cast<typename Elem<DTV>::T *>(x), static_cast<const typename Elem<DTV>::T *>(branch), scale, w, \
                                                              static_cast<typename Elem<DTV>::T *>(out), c8, eps, inv_cols)
        const int per = c8 <= 256 ? 1 : (c8 <= 512 ? 2 : (c8 <= 1024 ? 4 : 8));
        if (dt == 1) {
            if (branch) { if (per == 1) FLLM_N(1, 1, true); else if (per == 2) FLLM_N(1, 2, true); else if (per == 4) FLLM_N(1, 4, true); else FLLM_N(1, 8, true); }
            else { if (per == 1) FLLM_N(1, 1, false); else if (per == 2) FLLM_N(1, 2, false); else if (per == 4) FLLM_N(1, 4, false); else FLLM_N(1, 8, false); }
        } else if (dt == 2) {
            if (branch) { if (per == 1) FLLM_N(2, 1, true); else if (per == 2) FLLM_N(2, 2, true); else if (per == 4) FLLM_N(2, 4, true); else FLLM_N(2, 8, true); }
            else { if (per == 1) FLLM_N(2, 1, false); else if (per == 2) FLLM_N(2, 2, false); else if (per == 4) FLLM_N(2, 4, false); else FLLM_N(2, 8, false); }
        } else {
            return fllm_set_error("activation type %d is not FP16 or BF16", dt);
        }
#undef FLLM_N
        return fllm_check_launch("act rms_norm");
    }
    if (dt == 1) {
        if (branch) fllm_launch_pdl(rms_norm_act_generic_kernel<1, true>, dim3(rows), dim3(256), 0, s, (__half *)x, (const __half *)branch, scale, w, (__half *)out, cols, eps);
        else fllm_launch_pdl(rms_norm_act_generic_kernel<1, false>, dim3(rows), dim3(256), 0, s, (__half *)x, nullptr, scale, w, (__half *)out, cols, eps);
    } else if (dt == 2) {
        if (branch) fllm_launch_pdl(rms_norm_act_generic_kernel<2, true>, dim3(rows), dim3(256), 0, s, (__nv_bfloat16 *)x, (const __nv_bfloat16 *)branch, scale, w, (__nv_bfloat16 *)out, cols, eps);
        else fllm_launch_pdl(rms_norm_act_generic_kernel<2, false>, dim3(rows), dim3(256), 0, s, (__nv_bfloat16 *)x, nullptr, scale, w, (__nv_bfloat16 *)out, cols, eps);
    } else {
        return fllm_set_error("activation type %d is not FP16 or BF16", dt);
    }
    return fllm_check_launch("act rms_norm");
}

int fllm_na_residual_launch(cudaStream_t s, int dt, void *x, const void *res, size_t count, float scale) {
    if (count == 0) return 0;
    const unsigned g = grid_for((count + 7) / 8, 256);
    const size_t vec = (((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(res)) & 15) == 0) ? count / 8 : 0;
    FLLM_ACT_SWITCH(dt, (fllm_launch_pdl(residual_act_kernel<DT>, dim3(g), dim3(256), 0, s, static_cast<typename Elem<DT>::T *>(x), static_cast<const typename Elem<DT>::T *>(res), count, vec, scale)));
    return fllm_check_launch("act residual");
}

int fllm_na_glu_launch(cudaStream_t s, int dt, const void *gate, const void *up, void *out, size_t count, int mode) {
    if (count == 0) return 0;
    const unsigned g = grid_for((count + 7) / 8, 256);
    const size_t vec = (((reinterpret_cast<uintptr_t>(gate) | reinterpret_cast<uintptr_t>(up) | reinterpret_cast<uintptr_t>(out)) & 15) == 0) ? count / 8 : 0;
    FLLM_ACT_SWITCH(dt, (fllm_launch_pdl(glu_act_kernel<DT>, dim3(g), dim3(256), 0, s, static_cast<const typename Elem<DT>::T *>(gate), static_cast<const typename Elem<DT>::T *>(up),
                                                                static_cast<typename Elem<DT>::T *>(out), count, vec, mode)));
    return fllm_check_launch("act glu");
}

int fllm_na_bias_launch(cudaStream_t s, int dt, void *y, const float *bias, int rows, int batch) {
    const size_t count = (size_t)rows * batch;
    if (count == 0) return 0;
    FLLM_ACT_SWITCH(dt, (fllm_launch_pdl(bias_act_kernel<DT>, dim3(grid_for(count, 256)), dim3(256), 0, s, static_cast<typename Elem<DT>::T *>(y), bias, rows, count)));
    return fllm_check_launch("act bias");
}

int fllm_na_rope_launch(cudaStream_t s, int dt, void *q, void *k, const int32_t *pos, const float *cs, const float *sn, int rows, int heads, int kv_heads,
                         int dim, int half_, int neox) {
    const size_t total = (size_t)rows * (heads + kv_heads) * half_;
    if (total == 0) return 0;
    FLLM_ACT_SWITCH(dt, (fllm_launch_pdl(rope_act_kernel<DT>, dim3(grid_for(total, 256)), dim3(256), 0, s, static_cast<typename Elem<DT>::T *>(q), static_cast<typename Elem<DT>::T *>(k), pos, cs,
                                                                                   sn, rows, heads, kv_heads, dim, half_, neox)));
    return fllm_check_launch("act rope");
}

int fllm_na_kv_write_launch(cudaStream_t s, int dt, const void *src, uint16_t *pool, const int32_t *slots, int rows, int kv) {
    const size_t total = (size_t)rows * kv;
    if (total == 0) return 0;
    FLLM_ACT_SWITCH(dt, (fllm_launch_pdl(kv_write_act_kernel<DT>, dim3(grid_for(total, 256)), dim3(256), 0, s, static_cast<const typename Elem<DT>::T *>(src), pool, slots, rows, kv)));
    return fllm_check_launch("act kv write");
}

// dt 0 runs the same kernel on FP32 buffers (the FLLM_FUSE_QK_NORM_ROPE path of an FP32 pipeline). q_src / k / v with their row strides (elements), q rotated into
// q_dst (may be q_src when the stride is the dense one). Returns 3 when the head shape is not covered.
int fllm_na_rope_kv_strided_launch(cudaStream_t s, int dt, void *q_dst, const void *q_src, int q_stride, const void *k, int k_stride, const void *v, int v_stride,
                                   uint16_t *pool_k, uint16_t *pool_v, const int32_t *pos, const int32_t *slots, const float *cs, const float *sn, const float *qw,
                                   const float *kw, const float *qb, const float *kb, const float *vb, float eps, int rows, int heads, int kv_heads, int dk,
                                   int dv, int half_, int neox) {
    if (rows <= 0) return 0;
    if (half_ <= 0 || half_ > 32 * kMaxPairs || dk - 2 * half_ > 32 * kMaxTail || dk - 2 * half_ < 0) return 3;
    const size_t warps = (size_t)rows * (heads + 2 * kv_heads);
    const unsigned blocks = (unsigned)((warps * 32 + 127) / 128);
#define FLLM_RK(DTV) \
    fllm_launch_pdl(rope_kv_act_kernel<DTV>, dim3(blocks), dim3(128), 0, s, static_cast<typename Elem<DTV>::T *>(q_dst), static_cast<const typename Elem<DTV>::T *>(q_src), \
                    q_stride, static_cast<const typename Elem<DTV>::T *>(k), k_stride, static_cast<const typename Elem<DTV>::T *>(v), v_stride, pool_k, pool_v, pos, \
                    slots, cs, sn, qw, kw, qb, kb, vb, eps, rows, heads, kv_heads, dk, dv, half_, neox)
    if (dt == 0) FLLM_RK(0);
    else if (dt == 1) FLLM_RK(1);
    else if (dt == 2) FLLM_RK(2);
    else return fllm_set_error("activation type %d is not FP32, FP16 or BF16", dt);
#undef FLLM_RK
    return fllm_check_launch("act rope+kv");
}

int fllm_na_rope_kv_launch(cudaStream_t s, int dt, void *q, const void *k, const void *v, uint16_t *pool_k, uint16_t *pool_v, const int32_t *pos,
                           const int32_t *slots, const float *cs, const float *sn, const float *qw, const float *kw, float eps, int rows, int heads,
                           int kv_heads, int dk, int dv, int half_, int neox) {
    return fllm_na_rope_kv_strided_launch(s, dt, q, q, heads * dk, k, kv_heads * dk, v, kv_heads * dv, pool_k, pool_v, pos, slots, cs, sn, qw, kw, nullptr,
                                          nullptr, nullptr, eps, rows, heads, kv_heads, dk, dv, half_, neox);
}

// the three column blocks of a merged projection result to dense arrays (dt 1, 2: 16-bit elements, dt 0: 32-bit)
int fllm_na_split3_launch(cudaStream_t s, int dt, const void *src, void *d0, void *d1, void *d2, int rows, int n0, int n1, int n2) {
    const size_t total = (size_t)rows * (n0 + n1 + n2);
    if (total == 0) return 0;
    if (dt == 0)
        fllm_launch_pdl(split3_kernel<float>, dim3(grid_for(total, 256)), dim3(256), 0, s, static_cast<const float *>(src), static_cast<float *>(d0),
                        static_cast<float *>(d1), static_cast<float *>(d2), n0, n1, n2, (size_t)rows);
    else
        fllm_launch_pdl(split3_kernel<uint16_t>, dim3(grid_for(total, 256)), dim3(256), 0, s, static_cast<const uint16_t *>(src), static_cast<uint16_t *>(d0),
                        static_cast<uint16_t *>(d1), static_cast<uint16_t *>(d2), n0, n1, n2, (size_t)rows);
    return fllm_check_launch("act split3");
}

int fllm_na_glu_merged_launch(cudaStream_t s, int dt, const void *gu, void *out, int rows, int ff, int mode) {
    const size_t total = (size_t)rows * ff;
    if (total == 0) return 0;
    const bool vec = ff % 8 == 0 && (((reinterpret_cast<uintptr_t>(gu) | reinterpret_cast<uintptr_t>(out)) & 15) == 0);
    const unsigned g = grid_for(vec ? total / 8 : total, 256);
    FLLM_ACT_SWITCH(dt, (fllm_launch_pdl(glu_merged_act_kernel<DT>, dim3(g), dim3(256), 0, s, static_cast<const typename Elem<DT>::T *>(gu),
                                         static_cast<typename Elem<DT>::T *>(out), (size_t)rows, ff, vec ? 1 : 0, mode)));
    return fllm_check_launch("act glu merged");
}

int fllm_na_quantize_x8_launch(cudaStream_t s, int dt, const void *x, int8_t *xq, float2 *xs, size_t groups) {
    if (groups == 0) return 0;
    const unsigned blocks = (unsigned)((groups * 32 + 255) / 256);
    FLLM_ACT_SWITCH(dt, (fllm_launch_pdl(quantize_x8_act_kernel<DT>, dim3(blocks), dim3(256), 0, s, static_cast<const typename Elem<DT>::T *>(x), xq, xs, groups)));
    return fllm_check_launch("act quantize x8");
}

// out[0] = bits of the largest finite |x|, out[1] = count of NaN / Inf values; both must be zeroed by the caller (device memory)
int fllm_na_check_launch(cudaStream_t s, int dt, const void *x, size_t count, int *out) {
    if (count == 0) return 0;
    if (dt == 0) {
        // FP32: reuse the FP16 kernel through a tiny wrapper would lose range; use the generic instantiation on float
        check_act_kernel<0><<<grid_for(count, 256), 256, 0, s>>>(static_cast<const float *>(x), count, out);
        return fllm_check_launch("act check");
    }
    FLLM_ACT_SWITCH(dt, (check_act_kernel<DT><<<grid_for(count, 256), 256, 0, s>>>(static_cast<const typename Elem<DT>::T *>(x), count, out)));
    return fllm_check_launch("act check");
}

// selftest-gpu: FLLM_PDL on (1), off (0) or back to the environment (negative) for the launches of this file and of the small-M GEMV
extern "C" void fllm_pdl_force(int v) { fllm_pdl_override().store(v, std::memory_order_relaxed); }
extern "C" int fllm_pdl_active(void) { return fllm_pdl_enabled() ? 1 : 0; }
