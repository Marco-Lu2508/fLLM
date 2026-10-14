// fLLM CUDA backend: KV cache formats beyond f16 / q8_0 (bf16, f32, fp8 e4m3 / e5m2, int8, q8_0, q5_0, q5_1, q4_0, q4_1,
// iq4_nl, nvfp4), keys and values in independent formats.
//
// The layout is the one of src/kv_format.rs (the CPU side): per layer a code plane, an optional high-bit plane (5 bit),
// a scale plane and an optional minimum plane, all in 16-bit words; the offsets arrive in FllmKvDesc.
//  - kvq_write: one thread quantizes one group of values (one scale), same arithmetic as the CPU encoder.
//  - kvq_attn: paged attention that decodes keys and values on the fly (split-KV like attention2.cu; any head width up to
//    512, GQA, soft cap, window, chunk, sinks, ALiBi).
//  - prompts (16 or more rows of one sequence): the rows' keys and values are expanded once into an FP16 scratch and the
//    FP16 tensor-core / FlashInfer prompt kernels run on it (memory bound by FLLM_KVQ_SCRATCH_MB, default 256).
//
// Compiled without --use_fast_math: the encoders must agree bit for bit with the Rust ones (IEEE division, no flush to
// zero), see build.rs.
#include "common.cuh"
#include "fllm_cuda.h"
#include "fllm_cuda_internal.h"
#include <map>
#include <mutex>

enum {
    KVQ_F16 = 1, KVQ_Q8_0 = 2, KVQ_BF16 = 3, KVQ_E4M3 = 4, KVQ_E5M2 = 5, KVQ_INT8 = 6, KVQ_Q5_0 = 7, KVQ_Q5_1 = 8,
    KVQ_Q4_0 = 9, KVQ_Q4_1 = 10, KVQ_IQ4NL = 11, KVQ_NVFP4 = 12, KVQ_F32 = 13
};

__constant__ int8_t kIq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
__constant__ float kFp4Mid[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};

// ---------------------------------------------------------------------------------------------------------------------
// element decoding

__device__ __forceinline__ float kvq_e4m3(uint32_t b) {
    const uint32_t e = (b >> 3) & 15u, m = b & 7u;
    const float v = e == 0 ? (float)m * (1.0f / 512.0f) : __uint_as_float(((e + 120u) << 23) | (m << 20));
    return (b & 0x80u) ? -v : v;
}
__device__ __forceinline__ float kvq_e5m2(uint32_t b) {
    const uint32_t e = (b >> 2) & 31u, m = b & 3u;
    const float v = e == 0 ? (float)m * (1.0f / 65536.0f) : __uint_as_float(((e + 112u) << 23) | (m << 21));
    return (b & 0x80u) ? -v : v;
}
__device__ __forceinline__ float kvq_fp4(uint32_t code) {
    const uint32_t e = (code >> 1) & 3u, m = code & 1u;
    const float v = e == 0 ? 0.5f * (float)m : __uint_as_float(((126u + e) << 23) | (m << 22));
    return (code & 8u) ? -v : v;
}
__device__ __forceinline__ int kvq_iq4(uint32_t idx) { return (int)kIq4nl[idx & 15u]; }

__device__ __forceinline__ uint32_t kvq_f8(const uint16_t *w, size_t i) { return (w[i >> 1] >> ((i & 1) * 8)) & 0xffu; }
__device__ __forceinline__ uint32_t kvq_f4(const uint16_t *w, size_t i) { return (w[i >> 2] >> ((i & 3) * 4)) & 15u; }

// The values of one head of one slot: first value index and first group index.
struct KvqRow {
    size_t i0, g0;
};
__device__ __forceinline__ KvqRow kvq_row(const FllmKvDesc &d, size_t slot, int kh, int dim) {
    KvqRow r;
    r.i0 = slot * (size_t)d.kv + (size_t)kh * dim;
    const size_t groups = (size_t)d.gps;
    r.g0 = slot * groups + (d.gshift < 0 ? (size_t)kh : ((size_t)kh * dim) >> d.gshift);
    return r;
}

// Value `e` (0 <= e < dim) of the head `r`.
__device__ __forceinline__ float kvq_at(const uint16_t *w, const FllmKvDesc &d, const KvqRow &r, int e) {
    const size_t i = r.i0 + e;
    const size_t g = r.g0 + (d.gshift < 0 ? 0 : (size_t)(e >> d.gshift));
    switch (d.fmt) {
        case KVQ_F32: return __uint_as_float((uint32_t)w[2 * i] | ((uint32_t)w[2 * i + 1] << 16));
        case KVQ_F16: return h2f_bits(w[i]);
        case KVQ_BF16: return __uint_as_float((uint32_t)w[i] << 16);
        case KVQ_E5M2: return kvq_e5m2(kvq_f8(w, i));
        case KVQ_Q8_0:
        case KVQ_INT8: return (float)(int8_t)kvq_f8(w, i) * h2f_bits(w[d.scale_off + g]);
        case KVQ_E4M3: return kvq_e4m3(kvq_f8(w, i)) * h2f_bits(w[d.scale_off + g]);
        case KVQ_Q4_0: return (float)((int)kvq_f4(w, i) - 8) * h2f_bits(w[d.scale_off + g]);
        case KVQ_Q4_1: return (float)kvq_f4(w, i) * h2f_bits(w[d.scale_off + g]) + h2f_bits(w[d.min_off + g]);
        case KVQ_IQ4NL: return (float)kvq_iq4(kvq_f4(w, i)) * h2f_bits(w[d.scale_off + g]);
        case KVQ_Q5_0:
        case KVQ_Q5_1: {
            const int code = (int)kvq_f4(w, i) | (int)((w[d.hi_off + (i >> 4)] >> (i & 15)) & 1u) << 4;
            const float s = h2f_bits(w[d.scale_off + g]);
            return d.fmt == KVQ_Q5_0 ? (float)(code - 16) * s : (float)code * s + h2f_bits(w[d.min_off + g]);
        }
        default: {  // KVQ_NVFP4
            const uint32_t sb = (w[d.scale_off + (g >> 1)] >> ((g & 1) * 8)) & 0xffu;
            return kvq_fp4(kvq_f4(w, i)) * kvq_e4m3(sb);
        }
    }
}

// Prefetch the code word and the scale word of value `e` of the head `r` into L1 (the decode loops below are otherwise
// latency bound: one dependent global load per iteration and few resident warps).
__device__ __forceinline__ void kvq_prefetch(const uint16_t *w, const FllmKvDesc &d, const KvqRow &r, int e) {
    const size_t i = r.i0 + e;
    const int bits = d.fmt == KVQ_F32 ? 32
                     : (d.fmt == KVQ_F16 || d.fmt == KVQ_BF16) ? 16
                     : (d.fmt == KVQ_Q8_0 || d.fmt == KVQ_INT8 || d.fmt == KVQ_E4M3 || d.fmt == KVQ_E5M2) ? 8
                                                                                                            : 4;
    const uint16_t *p = w + ((i * (size_t)bits) >> 4);
    asm volatile("prefetch.global.L1 [%0];" ::"l"(p));
    if (d.fmt != KVQ_F32 && d.fmt != KVQ_F16 && d.fmt != KVQ_BF16 && d.fmt != KVQ_E5M2) {
        const size_t g = r.g0 + (d.gshift < 0 ? 0 : (size_t)(e >> d.gshift));
        const uint16_t *sp = w + d.scale_off + (d.fmt == KVQ_NVFP4 ? (g >> 1) : g);
        asm volatile("prefetch.global.L1 [%0];" ::"l"(sp));
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// quantization (mirrors kv_format.rs::quantize_group)

__device__ __forceinline__ uint32_t kvq_f32_to_fp8(float x, int mbits, int bias, uint32_t max_code) {
    const uint32_t sign = __float_as_uint(x) >> 31 ? 0x80u : 0u;
    const float a = fabsf(x);
    if (a != a) return 0;
    const float min_normal = __uint_as_float((uint32_t)(1 - bias + 127) << 23);
    uint32_t code;
    if (a < min_normal) {
        const float scaled = a * __uint_as_float((uint32_t)(bias - 1 + mbits + 127) << 23);
        code = (uint32_t)rintf(scaled);
    } else {
        const uint32_t bits = __float_as_uint(a);
        const int shift = 23 - mbits;
        const uint32_t r = (bits + ((1u << (shift - 1)) - 1u) + ((bits >> shift) & 1u)) >> shift;
        code = r - ((uint32_t)(127 - bias) << mbits);
    }
    return sign | min(code, max_code);
}

// Quantize `n` values `x` (a group) into `codes`; returns scale / minimum bits through `s` / `m`.
__device__ void kvq_quant_group(int fmt, const float *x, int n, uint8_t *codes, uint32_t &s, uint32_t &m) {
    m = 0;
    if (fmt == KVQ_Q8_0 || fmt == KVQ_INT8) {
        float amax = 0.f;
        for (int j = 0; j < n; j++) amax = fmaxf(amax, fabsf(x[j]));
        const float d = amax / 127.0f;
        s = f2h_bits(d);
        for (int j = 0; j < n; j++) {
            float q = 0.f;
            if (d != 0.f) {
                const float t = x[j] / d;
                q = fminf(fmaxf(copysignf(floorf(fabsf(t) + 0.5f), t), -127.f), 127.f);
            }
            codes[j] = (uint8_t)(int8_t)(int)q;
        }
    } else if (fmt == KVQ_E4M3) {
        float amax = 0.f;
        for (int j = 0; j < n; j++) amax = fmaxf(amax, fabsf(x[j]));
        s = f2h_bits(amax / 448.0f);
        const float d = h2f_bits((uint16_t)s);
        for (int j = 0; j < n; j++) codes[j] = d == 0.f ? 0 : (uint8_t)kvq_f32_to_fp8(x[j] / d, 3, 7, 126u);
    } else if (fmt == KVQ_Q4_0 || fmt == KVQ_Q5_0) {
        const float zero = fmt == KVQ_Q4_0 ? 8.f : 16.f, levels = fmt == KVQ_Q4_0 ? 15.f : 31.f;
        float v = 0.f;
        for (int j = 0; j < n; j++)
            if (fabsf(x[j]) > fabsf(v)) v = x[j];
        s = f2h_bits(v / -zero);
        const float d = h2f_bits((uint16_t)s);
        for (int j = 0; j < n; j++)
            codes[j] = d == 0.f ? (uint8_t)zero : (uint8_t)fminf(fmaxf(floorf(x[j] / d + (zero + 0.5f)), 0.f), levels);
    } else if (fmt == KVQ_Q4_1 || fmt == KVQ_Q5_1) {
        const float levels = fmt == KVQ_Q4_1 ? 15.f : 31.f;
        float mn = INFINITY, mx = -INFINITY;
        for (int j = 0; j < n; j++) {
            mn = fminf(mn, x[j]);
            mx = fmaxf(mx, x[j]);
        }
        s = f2h_bits((mx - mn) / levels);
        m = f2h_bits(mn);
        const float d = h2f_bits((uint16_t)s), mm = h2f_bits((uint16_t)m);
        for (int j = 0; j < n; j++)
            codes[j] = d == 0.f ? 0 : (uint8_t)fminf(fmaxf(floorf((x[j] - mm) / d + 0.5f), 0.f), levels);
    } else if (fmt == KVQ_IQ4NL) {
        float v = 0.f;
        for (int j = 0; j < n; j++)
            if (fabsf(x[j]) > fabsf(v)) v = x[j];
        if (v == 0.f) {
            for (int j = 0; j < n; j++) codes[j] = 8;
            s = 0;
            return;
        }
        const float d0 = v / -127.0f;
        float sxg = 0.f, sgg = 0.f;
        for (int j = 0; j < n; j++) {
            const float t = x[j] / d0;
            int best = 0;
            float best_err = INFINITY;
#pragma unroll
            for (int i = 0; i < 16; i++) {
                const float err = fabsf(t - (float)kIq4nl[i]);
                if (err < best_err) {
                    best_err = err;
                    best = i;
                }
            }
            codes[j] = (uint8_t)best;
            const float g = (float)kIq4nl[best];
            sxg = fmaf(x[j], g, sxg);
            sgg += g * g;
        }
        s = f2h_bits(sxg / sgg);
    } else {  // KVQ_NVFP4
        float amax = 0.f;
        for (int j = 0; j < n; j++) amax = fmaxf(amax, fabsf(x[j]));
        s = kvq_f32_to_fp8(amax / 6.0f, 3, 7, 126u);
        const float d = kvq_e4m3(s);
        for (int j = 0; j < n; j++) {
            uint32_t idx = 0;
            if (d != 0.f) {
                const float a = fabsf(x[j]) / d;
#pragma unroll
                for (int i = 0; i < 7; i++) idx += a > kFp4Mid[i] ? 1u : 0u;
            }
            codes[j] = (uint8_t)((__float_as_uint(x[j]) >> 31 ? 8u : 0u) | idx);
        }
    }
}

#define KVQ_MAX_UNIT 1024

__global__ void kvq_write_kernel(const float *src, uint16_t *pool, const int32_t *slots, int rows, FllmKvDesc d) {
    const int fmt = d.fmt;
    const size_t kv = (size_t)d.kv;
    if (fmt == KVQ_F32 || fmt == KVQ_F16 || fmt == KVQ_BF16 || fmt == KVQ_E5M2) {
        // unscaled: two values per thread keep every store a whole word for the 8 bit format
        const size_t pairs = (size_t)rows * kv / 2;
        for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < pairs; t += (size_t)gridDim.x * blockDim.x) {
            const size_t i = 2 * t;
            const size_t slot = (size_t)slots[i / kv];
            const size_t idx = slot * kv + i % kv;
            const float a = src[i], b = src[i + 1];
            if (fmt == KVQ_F16) {
                pool[idx] = f2h_bits(a);
                pool[idx + 1] = f2h_bits(b);
            } else if (fmt == KVQ_BF16) {
                for (int u = 0; u < 2; u++) {
                    const float x = u ? b : a;
                    const uint32_t bits = __float_as_uint(x);
                    pool[idx + u] = (uint16_t)(x != x ? ((bits >> 16) | 0x40u) : ((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16));
                }
            } else if (fmt == KVQ_F32) {
                for (int u = 0; u < 2; u++) {
                    const uint32_t bits = __float_as_uint(u ? b : a);
                    pool[2 * (idx + u)] = (uint16_t)bits;
                    pool[2 * (idx + u) + 1] = (uint16_t)(bits >> 16);
                }
            } else {
                pool[idx / 2] = (uint16_t)(kvq_f32_to_fp8(a, 2, 15, 123u) | (kvq_f32_to_fp8(b, 2, 15, 123u) << 8));
            }
        }
        return;
    }
    const int group = d.group;
    const int per_unit = fmt == KVQ_NVFP4 ? 2 : 1;  // nvfp4: two 16-value blocks share a scale word
    const int unit = group * per_unit;
    const size_t units = (size_t)rows * (kv / unit);
    const size_t units_per_row = kv / unit;
    for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < units; t += (size_t)gridDim.x * blockDim.x) {
        const size_t r = t / units_per_row, u = t % units_per_row;
        const size_t slot = (size_t)slots[r];
        const float *x = src + r * kv + u * unit;
        const size_t first = slot * kv + u * unit;  // index of the first value of the unit
        const size_t gfirst = (slot * kv + u * unit) / group;
        uint8_t codes[KVQ_MAX_UNIT];
        uint32_t s[2] = {0, 0}, m[2] = {0, 0};
        for (int sub = 0; sub < per_unit; sub++) kvq_quant_group(fmt, x + sub * group, group, codes + sub * group, s[sub], m[sub]);
        if (fmt == KVQ_Q8_0 || fmt == KVQ_INT8 || fmt == KVQ_E4M3) {
            for (int j = 0; j < unit; j += 2) pool[(first + j) / 2] = (uint16_t)(codes[j] | (codes[j + 1] << 8));
        } else {
            for (int j = 0; j < unit; j += 4)
                pool[(first + j) / 4] = (uint16_t)((codes[j] & 15) | ((codes[j + 1] & 15) << 4) | ((codes[j + 2] & 15) << 8) |
                                                   ((codes[j + 3] & 15) << 12));
            if (fmt == KVQ_Q5_0 || fmt == KVQ_Q5_1)
                for (int j = 0; j < unit; j += 16) {
                    uint32_t bits = 0;
                    for (int k = 0; k < 16; k++) bits |= (uint32_t)((codes[j + k] >> 4) & 1) << k;
                    pool[d.hi_off + (first + j) / 16] = (uint16_t)bits;
                }
        }
        if (fmt == KVQ_NVFP4) {
            pool[d.scale_off + gfirst / 2] = (uint16_t)(s[0] | (s[1] << 8));
        } else {
            pool[d.scale_off + gfirst] = (uint16_t)s[0];
            if (fmt == KVQ_Q4_1 || fmt == KVQ_Q5_1) pool[d.min_off + gfirst] = (uint16_t)m[0];
        }
    }
}

// One warp quantizes one unit (a scale group; two 16-value blocks for nvfp4): lane l holds the values l, l + 32, ...
// The reductions (maximum, minimum, first element of largest magnitude) are warp shuffles and ballots, so the results
// equal those of the sequential CPU encoder; the only order-dependent sums (iq4_nl) are done in element order.
template <int NK>
__global__ void __launch_bounds__(128) kvq_write_warp_kernel(const float *src, uint16_t *pool, const int32_t *slots, int rows,
                                                             FllmKvDesc d) {
    const int lane = threadIdx.x & 31;
    const size_t warp0 = (size_t)blockIdx.x * 4 + (threadIdx.x >> 5);
    const size_t nwarps = (size_t)gridDim.x * 4;
    const int fmt = d.fmt;
    const size_t kv = (size_t)d.kv;
    const int group = d.group;
    const int unit = fmt == KVQ_NVFP4 ? 2 * group : group;
    const size_t units_per_row = kv / unit;
    const size_t units = (size_t)rows * units_per_row;
    const unsigned full = 0xffffffffu;
    for (size_t t = warp0; t < units; t += nwarps) {
        const size_t r = t / units_per_row, u = t % units_per_row;
        const size_t slot = (size_t)slots[r];
        const float *x = src + r * kv + u * unit;
        const size_t first = slot * kv + u * unit;
        const size_t gfirst = first / group;
        float v[NK];
        bool ok[NK];
#pragma unroll
        for (int k = 0; k < NK; k++) {
            ok[k] = lane + 32 * k < unit;
            v[k] = ok[k] ? x[lane + 32 * k] : 0.f;
        }
        uint32_t s = 0, m = 0;
        uint32_t code[NK];
        if (fmt == KVQ_NVFP4) {
            float amax = fabsf(v[0]);
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(full, amax, o));
            s = kvq_f32_to_fp8(amax / 6.0f, 3, 7, 126u);
            const float dd = kvq_e4m3(s);
            uint32_t idx = 0;
            if (dd != 0.f) {
                const float a = fabsf(v[0]) / dd;
#pragma unroll
                for (int i = 0; i < 7; i++) idx += a > kFp4Mid[i] ? 1u : 0u;
            }
            code[0] = (__float_as_uint(v[0]) >> 31 ? 8u : 0u) | idx;
        } else if (fmt == KVQ_Q8_0 || fmt == KVQ_INT8 || fmt == KVQ_E4M3) {
            float amax = 0.f;
#pragma unroll
            for (int k = 0; k < NK; k++) amax = fmaxf(amax, fabsf(v[k]));
            amax = warp_max(amax);
            if (fmt == KVQ_E4M3) {
                s = f2h_bits(amax / 448.0f);
                const float dd = h2f_bits((uint16_t)s);
#pragma unroll
                for (int k = 0; k < NK; k++) code[k] = dd == 0.f ? 0u : kvq_f32_to_fp8(v[k] / dd, 3, 7, 126u);
            } else {
                const float dd = amax / 127.0f;
                s = f2h_bits(dd);
#pragma unroll
                for (int k = 0; k < NK; k++) {
                    float q = 0.f;
                    if (dd != 0.f) {
                        const float tq = v[k] / dd;
                        q = fminf(fmaxf(copysignf(floorf(fabsf(tq) + 0.5f), tq), -127.f), 127.f);
                    }
                    code[k] = (uint32_t)(uint8_t)(int8_t)(int)q;
                }
            }
        } else if (fmt == KVQ_Q4_1 || fmt == KVQ_Q5_1) {
            const float levels = fmt == KVQ_Q4_1 ? 15.f : 31.f;
            float mn = INFINITY, mx = -INFINITY;
#pragma unroll
            for (int k = 0; k < NK; k++) {
                if (ok[k]) {
                    mn = fminf(mn, v[k]);
                    mx = fmaxf(mx, v[k]);
                }
            }
            mn = -warp_max(-mn);
            mx = warp_max(mx);
            s = f2h_bits((mx - mn) / levels);
            m = f2h_bits(mn);
            const float dd = h2f_bits((uint16_t)s), mm = h2f_bits((uint16_t)m);
#pragma unroll
            for (int k = 0; k < NK; k++)
                code[k] = dd == 0.f ? 0u : (uint32_t)fminf(fmaxf(floorf((v[k] - mm) / dd + 0.5f), 0.f), levels);
        } else {  // q4_0, q5_0, iq4_nl: the element of largest magnitude (the first one) with its sign
            float amax = 0.f;
#pragma unroll
            for (int k = 0; k < NK; k++) amax = fmaxf(amax, fabsf(v[k]));
            amax = warp_max(amax);
            float vmax = 0.f;
            bool found = false;
#pragma unroll
            for (int k = 0; k < NK; k++) {
                const unsigned mask = __ballot_sync(full, ok[k] && fabsf(v[k]) == amax);
                const float cand = __shfl_sync(full, v[k], mask ? __ffs(mask) - 1 : 0);
                if (!found && mask && amax > 0.f) {
                    vmax = cand;
                    found = true;
                }
            }
            if (fmt == KVQ_IQ4NL) {
                if (vmax == 0.f) {
                    s = 0;
#pragma unroll
                    for (int k = 0; k < NK; k++) code[k] = 8u;
                } else {
                    const float d0 = vmax / -127.0f;
                    float g[NK];
#pragma unroll
                    for (int k = 0; k < NK; k++) {
                        const float tq = v[k] / d0;
                        int best = 0;
                        float best_err = INFINITY;
#pragma unroll
                        for (int i = 0; i < 16; i++) {
                            const float err = fabsf(tq - (float)kIq4nl[i]);
                            if (err < best_err) {
                                best_err = err;
                                best = i;
                            }
                        }
                        code[k] = (uint32_t)best;
                        g[k] = (float)kIq4nl[best];
                    }
                    float sxg = 0.f, sgg = 0.f;
#pragma unroll
                    for (int k = 0; k < NK; k++) {
#pragma unroll 1
                        for (int l = 0; l < 32; l++) {
                            if (32 * k + l < unit) {
                                const float xj = __shfl_sync(full, v[k], l), gj = __shfl_sync(full, g[k], l);
                                sxg = fmaf(xj, gj, sxg);
                                sgg += gj * gj;
                            }
                        }
                    }
                    s = f2h_bits(sxg / sgg);
                }
            } else {
                const float zero = fmt == KVQ_Q4_0 ? 8.f : 16.f, levels = fmt == KVQ_Q4_0 ? 15.f : 31.f;
                s = f2h_bits(vmax / -zero);
                const float dd = h2f_bits((uint16_t)s);
#pragma unroll
                for (int k = 0; k < NK; k++)
                    code[k] = dd == 0.f ? (uint32_t)zero : (uint32_t)fminf(fmaxf(floorf(v[k] / dd + (zero + 0.5f)), 0.f), levels);
            }
        }
        // codes to words
        const bool eight = fmt == KVQ_Q8_0 || fmt == KVQ_INT8 || fmt == KVQ_E4M3;
        const bool five = fmt == KVQ_Q5_0 || fmt == KVQ_Q5_1;
#pragma unroll
        for (int k = 0; k < NK; k++) {
            const size_t idx0 = first + 32 * k + lane;
            if (eight) {
                const uint32_t c1 = __shfl_down_sync(full, code[k], 1);
                if ((lane & 1) == 0 && ok[k]) pool[idx0 >> 1] = (uint16_t)((code[k] & 0xffu) | (c1 << 8));
            } else {
                const uint32_t c0 = code[k] & 15u;
                const uint32_t c1 = __shfl_down_sync(full, c0, 1), c2 = __shfl_down_sync(full, c0, 2),
                               c3 = __shfl_down_sync(full, c0, 3);
                if ((lane & 3) == 0 && ok[k]) pool[idx0 >> 2] = (uint16_t)(c0 | (c1 << 4) | (c2 << 8) | (c3 << 12));
                if (five) {
                    const unsigned bits = __ballot_sync(full, ok[k] && ((code[k] >> 4) & 1u));
                    const size_t w0 = d.hi_off + ((first + 32 * k) >> 4);
                    if (lane == 0 && 32 * k < unit) pool[w0] = (uint16_t)(bits & 0xffffu);
                    if (lane == 0 && 32 * k + 16 < unit) pool[w0 + 1] = (uint16_t)(bits >> 16);
                }
            }
        }
        if (fmt == KVQ_NVFP4) {
            const uint32_t s1 = __shfl_sync(full, s, 16);
            if (lane == 0) pool[d.scale_off + gfirst / 2] = (uint16_t)((s & 0xffu) | (s1 << 8));
        } else if (lane == 0) {
            pool[d.scale_off + gfirst] = (uint16_t)s;
            if (fmt == KVQ_Q4_1 || fmt == KVQ_Q5_1) pool[d.min_off + gfirst] = (uint16_t)m;
        }
    }
}

int fllm_kvq_write_launch(cudaStream_t s, const float *src, uint16_t *pool, const int32_t *slots, int rows,
                                       const FllmKvDesc &d) {
    if (rows <= 0) return 0;
    if (d.group > KVQ_MAX_UNIT || (d.fmt == KVQ_NVFP4 && d.group * 2 > KVQ_MAX_UNIT))
        return fllm_set_error("kv cache write: group of %d values too large", d.group);
    const bool scaled = !(d.fmt == KVQ_E5M2 || d.fmt == KVQ_F32 || d.fmt == KVQ_F16 || d.fmt == KVQ_BF16);
    const int unit = d.fmt == KVQ_NVFP4 ? d.group * 2 : d.group;
    if (scaled && unit <= 512 && getenv("FLLM_KVQ_THREAD_WRITE") == nullptr) {
        const size_t units = (size_t)rows * d.kv / unit;
        int wb = (int)((units + 3) / 4);
        if (wb > 65535) wb = 65535;
        if (unit <= 32) kvq_write_warp_kernel<1><<<wb, 128, 0, s>>>(src, pool, slots, rows, d);
        else if (unit <= 64) kvq_write_warp_kernel<2><<<wb, 128, 0, s>>>(src, pool, slots, rows, d);
        else if (unit <= 128) kvq_write_warp_kernel<4><<<wb, 128, 0, s>>>(src, pool, slots, rows, d);
        else if (unit <= 256) kvq_write_warp_kernel<8><<<wb, 128, 0, s>>>(src, pool, slots, rows, d);
        else kvq_write_warp_kernel<16><<<wb, 128, 0, s>>>(src, pool, slots, rows, d);
        return fllm_check_launch("kvq_write");
    }
    const size_t work = (size_t)rows * d.kv / (d.fmt == KVQ_E5M2 || d.fmt == KVQ_F32 || d.fmt == KVQ_F16 || d.fmt == KVQ_BF16
                                                   ? 2
                                                   : (d.group * (d.fmt == KVQ_NVFP4 ? 2 : 1)));
    int blocks = (int)((work + 127) / 128);
    if (blocks > 65535) blocks = 65535;
    kvq_write_kernel<<<blocks > 0 ? blocks : 1, 128, 0, s>>>(src, pool, slots, rows, d);
    return fllm_check_launch("kvq_write");
}

// ---------------------------------------------------------------------------------------------------------------------
// attention

#define KVQ_WARPS 4
#define KVQ_KPW 4
#define KVQ_FLOOR (-3.0e38f)

struct KvqAttnArgs {
    const float *q;
    const uint16_t *kpool, *vpool;
    float *out, *ws;
    const int64_t *tables, *lengths;
    const float *sinks, *alibi;
    FllmKvDesc kd, vd;
    int heads, kv_heads, dk, dv, stride, block_size, per_split, splits;
    float scale, softcap;
    int window, chunk;
};

template <int NV>
__global__ void __launch_bounds__(KVQ_WARPS * 32) kvq_attn_kernel(KvqAttnArgs a) {
    __shared__ float wm[KVQ_WARPS], wl[KVQ_WARPS];
    __shared__ float wacc[KVQ_WARPS][NV * 32];
    const int h = blockIdx.x, r = blockIdx.y, sp = blockIdx.z;
    const int kh = h / (a.heads / a.kv_heads);
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int n = (int)a.lengths[r];
    int start = 0;
    if (a.chunk != 0) start = (n - 1) / a.chunk * a.chunk;
    else if (a.window != 0 && n > a.window) start = n - a.window;
    const int begin = start + sp * a.per_split;
    const int end = min(n, begin + a.per_split);

    float qv[NV];
    const float *qp = a.q + ((size_t)r * a.heads + h) * a.dk;
#pragma unroll
    for (int e = 0; e < NV; e++) {
        const int d = lane + 32 * e;
        qv[e] = d < a.dk ? qp[d] * a.scale : 0.f;
    }
    float acc[NV];
#pragma unroll
    for (int e = 0; e < NV; e++) acc[e] = 0.f;
    float m = KVQ_FLOOR, l = 0.f;
    const int64_t *table = a.tables + (size_t)r * a.stride;

    for (int p0 = begin + warp * KVQ_KPW; p0 < end; p0 += KVQ_WARPS * KVQ_KPW) {
        size_t slot[KVQ_KPW];
        bool ok[KVQ_KPW];
        float s[KVQ_KPW];
#pragma unroll
        for (int j = 0; j < KVQ_KPW; j++) {
            const int p = p0 + j;
            ok[j] = p < end;
            slot[j] = ok[j] ? (size_t)table[p / a.block_size] * a.block_size + p % a.block_size : 0;
        }
#pragma unroll
        for (int j = 0; j < KVQ_KPW; j++) {
            const KvqRow kr = kvq_row(a.kd, slot[j], kh, a.dk);
            float part = 0.f;
#pragma unroll
            for (int e = 0; e < NV; e++) {
                const int d = lane + 32 * e;
                if (d < a.dk) part = fmaf(qv[e], kvq_at(a.kpool, a.kd, kr, d), part);
            }
            s[j] = part;
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
#pragma unroll
            for (int j = 0; j < KVQ_KPW; j++) s[j] += __shfl_xor_sync(0xffffffffu, s[j], o);
        float m_new = m;
#pragma unroll
        for (int j = 0; j < KVQ_KPW; j++) {
            if (!ok[j]) {
                s[j] = KVQ_FLOOR;
                continue;
            }
            if (a.alibi) s[j] += a.alibi[h] * ((float)(p0 + j) - (float)(n - 1));
            if (a.softcap != 0.f) s[j] = a.softcap * tanhf(fminf(fmaxf(s[j] / a.softcap, -15.f), 15.f));
            m_new = fmaxf(m_new, s[j]);
        }
        const float corr = __expf(m - m_new);
        float w[KVQ_KPW];
        float esum = 0.f;
#pragma unroll
        for (int j = 0; j < KVQ_KPW; j++) {
            w[j] = ok[j] ? __expf(s[j] - m_new) : 0.f;
            esum += w[j];
        }
        l = l * corr + esum;
#pragma unroll
        for (int e = 0; e < NV; e++) acc[e] *= corr;
#pragma unroll
        for (int j = 0; j < KVQ_KPW; j++) {
            if (!ok[j]) continue;
            const KvqRow vr = kvq_row(a.vd, slot[j], kh, a.dv);
#pragma unroll
            for (int e = 0; e < NV; e++) {
                const int d = lane + 32 * e;
                if (d < a.dv) acc[e] = fmaf(w[j], kvq_at(a.vpool, a.vd, vr, d), acc[e]);
            }
        }
        m = m_new;
    }

#pragma unroll
    for (int e = 0; e < NV; e++) wacc[warp][lane + 32 * e] = acc[e];
    if (lane == 0) {
        wm[warp] = m;
        wl[warp] = l;
    }
    __syncthreads();

    float M = KVQ_FLOOR;
    for (int w = 0; w < KVQ_WARPS; w++) M = fmaxf(M, wm[w]);
    float f[KVQ_WARPS], L = 0.f;
    for (int w = 0; w < KVQ_WARPS; w++) {
        f[w] = __expf(wm[w] - M);
        L += wl[w] * f[w];
    }
    if (a.splits == 1) {
        if (a.sinks) {
            const float Mf = fmaxf(M, a.sinks[h]);
            const float rescale = __expf(M - Mf);
            for (int w = 0; w < KVQ_WARPS; w++) f[w] *= rescale;
            L = L * rescale + __expf(a.sinks[h] - Mf);
        }
        const float inv = 1.0f / L;
        for (int d = threadIdx.x; d < a.dv; d += blockDim.x) {
            float v = 0.f;
            for (int w = 0; w < KVQ_WARPS; w++) v += wacc[w][d] * f[w];
            a.out[((size_t)r * a.heads + h) * a.dv + d] = v * inv;
        }
    } else {
        float *dst = a.ws + (((size_t)r * a.heads + h) * a.splits + sp) * (a.dv + 2);
        for (int d = threadIdx.x; d < a.dv; d += blockDim.x) {
            float v = 0.f;
            for (int w = 0; w < KVQ_WARPS; w++) v += wacc[w][d] * f[w];
            dst[d] = v;
        }
        if (threadIdx.x == 0) {
            dst[a.dv] = M;
            dst[a.dv + 1] = L;
        }
    }
}

__global__ void kvq_merge_kernel(KvqAttnArgs a) {
    const int h = blockIdx.x, r = blockIdx.y;
    const int dv = a.dv;
    const float *src = a.ws + ((size_t)r * a.heads + h) * a.splits * (dv + 2);
    float M = a.sinks ? a.sinks[h] : KVQ_FLOOR;
    for (int s = 0; s < a.splits; s++) M = fmaxf(M, src[(size_t)s * (dv + 2) + dv]);
    float L = a.sinks ? __expf(a.sinks[h] - M) : 0.f, v = 0.f;
    for (int s = 0; s < a.splits; s++) {
        const float *p = src + (size_t)s * (dv + 2);
        const float f = __expf(p[dv] - M);
        L += p[dv + 1] * f;
        v += p[threadIdx.x] * f;
    }
    a.out[((size_t)r * a.heads + h) * dv + threadIdx.x] = v / L;
}

// Attention v2: one block per (kv head, row, split). The keys and values of a tile are decoded once, cooperatively and
// coalesced, into shared memory and shared by all query heads of the kv head (GQA). Scores, the online softmax and the
// value sums then run on shared memory. Partial results use the layout of kvq_merge_kernel.
#define KVQ2_THREADS 256

template <int ACC>
__global__ void __launch_bounds__(KVQ2_THREADS) kvq_attn2_kernel(KvqAttnArgs a, int T) {
    extern __shared__ float sm[];
    const int G = a.heads / a.kv_heads;
    const int dk = a.dk, dv = a.dv;
    const int kh = blockIdx.x, r = blockIdx.y, sp = blockIdx.z;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    float *qs = sm;                 // G * dk, scaled
    float *Ks = qs + G * dk;        // T * (dk + 1)
    float *Vs = Ks + T * (dk + 1);  // T * (dv + 1)
    float *S = Vs + T * (dv + 1);   // G * T scores, then weights
    float *Mx = S + G * T;
    float *Lx = Mx + G;
    float *Cx = Lx + G;
    __shared__ int64_t slot_s[32];
    __shared__ KvqRow rk_s[32], rv_s[32];

    const int n = (int)a.lengths[r];
    int start = 0;
    if (a.chunk != 0) start = (n - 1) / a.chunk * a.chunk;
    else if (a.window != 0 && n > a.window) start = n - a.window;
    const int begin = start + sp * a.per_split;
    const int end = min(n, begin + a.per_split);
    const int64_t *table = a.tables + (size_t)r * a.stride;

    for (int i = tid; i < G * dk; i += KVQ2_THREADS) {
        const int g = i / dk, d = i % dk;
        qs[i] = a.q[((size_t)r * a.heads + kh * G + g) * dk + d] * a.scale;
    }
    for (int g = tid; g < G; g += KVQ2_THREADS) {
        Mx[g] = KVQ_FLOOR;
        Lx[g] = 0.f;
    }
    float acc[ACC];
#pragma unroll
    for (int j = 0; j < ACC; j++) acc[j] = 0.f;
    __syncthreads();

    for (int p0 = begin; p0 < end; p0 += T) {
        const int cnt = min(T, end - p0);
        if (tid < T) {
            const int p = p0 + tid;
            slot_s[tid] = p < end ? (int64_t)table[p / a.block_size] * a.block_size + p % a.block_size : 0;
            rk_s[tid] = kvq_row(a.kd, (size_t)slot_s[tid], kh, dk);
            rv_s[tid] = kvq_row(a.vd, (size_t)slot_s[tid], kh, dv);
        }
        __syncthreads();
        for (int i = tid; i < T * dk; i += KVQ2_THREADS) {
            const int t = i / dk, d = i - t * dk;
            if (t < cnt) kvq_prefetch(a.kpool, a.kd, rk_s[t], d);
        }
        for (int i = tid; i < T * dv; i += KVQ2_THREADS) {
            const int t = i / dv, d = i - t * dv;
            if (t < cnt) kvq_prefetch(a.vpool, a.vd, rv_s[t], d);
        }
        for (int i = tid; i < T * dk; i += KVQ2_THREADS) {
            const int t = i / dk, d = i - t * dk;
            Ks[t * (dk + 1) + d] = t < cnt ? kvq_at(a.kpool, a.kd, rk_s[t], d) : 0.f;
        }
        for (int i = tid; i < T * dv; i += KVQ2_THREADS) {
            const int t = i / dv, d = i - t * dv;
            Vs[t * (dv + 1) + d] = t < cnt ? kvq_at(a.vpool, a.vd, rv_s[t], d) : 0.f;
        }
        __syncthreads();
        for (int item = tid; item < G * T; item += KVQ2_THREADS) {
            const int g = item / T, t = item - g * T;
            float s = -INFINITY;
            if (t < cnt) {
                const float *qg = qs + g * dk, *kt = Ks + t * (dk + 1);
                float dot = 0.f;
                for (int d = 0; d < dk; d++) dot = fmaf(qg[d], kt[d], dot);
                s = dot;
                if (a.alibi) s += a.alibi[kh * G + g] * ((float)(p0 + t) - (float)(n - 1));
                if (a.softcap != 0.f) s = a.softcap * tanhf(fminf(fmaxf(s / a.softcap, -15.f), 15.f));
            }
            S[g * T + t] = s;
        }
        __syncthreads();
        for (int g = warp; g < G; g += KVQ2_THREADS / 32) {
            const float s = lane < T ? S[g * T + lane] : -INFINITY;
            const float m_old = Mx[g];
            const float m_new = fmaxf(m_old, warp_max(s));
            const float corr = __expf(m_old - m_new);
            const float e = s > -INFINITY ? __expf(s - m_new) : 0.f;
            const float esum = warp_sum(e);
            if (lane < T) S[g * T + lane] = e;
            if (lane == 0) {
                Mx[g] = m_new;
                Lx[g] = Lx[g] * corr + esum;
                Cx[g] = corr;
            }
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < ACC; j++) {
            const int i = tid + KVQ2_THREADS * j;
            if (i < G * dv) {
                const int g = i / dv, d = i - g * dv;
                float x = acc[j] * Cx[g];
                const float *w = S + g * T, *vcol = Vs + d;
                for (int t = 0; t < cnt; t++) x = fmaf(w[t], vcol[t * (dv + 1)], x);
                acc[j] = x;
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < ACC; j++) {
        const int i = tid + KVQ2_THREADS * j;
        if (i < G * dv) {
            const int g = i / dv, d = i - g * dv;
            const int h = kh * G + g;
            const float M = Mx[g], L = Lx[g];
            if (a.splits == 1) {
                float af = acc[j], Lf = L;
                if (a.sinks) {
                    const float Mf = fmaxf(M, a.sinks[h]);
                    const float rs = __expf(M - Mf);
                    af *= rs;
                    Lf = L * rs + __expf(a.sinks[h] - Mf);
                }
                a.out[((size_t)r * a.heads + h) * dv + d] = af / Lf;
            } else {
                float *dst = a.ws + (((size_t)r * a.heads + h) * a.splits + sp) * (dv + 2);
                dst[d] = acc[j];
                if (d == 0) {
                    dst[dv] = M;
                    dst[dv + 1] = L;
                }
            }
        }
    }
}

template <int ACC>
static void kvq_attn2_run(cudaStream_t st, const KvqAttnArgs &a, int rows, int T, size_t smem) {
    kvq_attn2_kernel<ACC><<<dim3(a.kv_heads, rows, a.splits), KVQ2_THREADS, smem, st>>>(a, T);
}

// Returns -1 when the shape is not covered (the caller uses kvq_attn_launch).
static int kvq_attn2_launch(cudaStream_t st, KvqAttnArgs a, int rows, int max_len, float *ws, size_t ws_bytes) {
    if (getenv("FLLM_KVQ_OLD_ATTN") != nullptr) return -1;
    const int G = a.heads / a.kv_heads;
    const int mx = a.dk > a.dv ? a.dk : a.dv;
    const int T = mx <= 128 ? 32 : (mx <= 256 ? 16 : 8);
    const size_t smem = ((size_t)G * a.dk + (size_t)T * (a.dk + 1) + (size_t)T * (a.dv + 1) + (size_t)G * T + 3 * G) * 4;
    if (G > 16 || (size_t)G * a.dv > 4096 || smem > 48 * 1024) return -1;
    int splits = 1;
    const int blocks = rows * a.kv_heads;
    if (blocks < 192 && max_len > T) {
        splits = (max_len + T - 1) / T;
        if (splits > 96) splits = 96;
        while (splits > 1 && (size_t)rows * a.heads * splits * (a.dv + 2) * 4 > ws_bytes) splits--;
    }
    a.splits = splits;
    a.per_split = ((max_len + splits - 1) / splits + T - 1) / T * T;
    a.ws = ws;
    const int acc = (G * a.dv + KVQ2_THREADS - 1) / KVQ2_THREADS;
    if (acc <= 1) kvq_attn2_run<1>(st, a, rows, T, smem);
    else if (acc <= 2) kvq_attn2_run<2>(st, a, rows, T, smem);
    else if (acc <= 4) kvq_attn2_run<4>(st, a, rows, T, smem);
    else if (acc <= 8) kvq_attn2_run<8>(st, a, rows, T, smem);
    else if (acc <= 16) kvq_attn2_run<16>(st, a, rows, T, smem);
    else kvq_attn2_run<32>(st, a, rows, T, smem);
    if (splits > 1) kvq_merge_kernel<<<dim3(a.heads, rows), a.dv, 0, st>>>(a);
    return fllm_check_launch("kvq_attention2");
}

template <int NV>
static int kvq_attn_launch(cudaStream_t st, KvqAttnArgs a, int rows, int max_len, float *ws, size_t ws_bytes) {
    int splits = 1;
    const int blocks = rows * a.heads;
    if (blocks < 256 && max_len > 512) {
        splits = (max_len + 255) / 256;
        if (splits > 32) splits = 32;
        while (splits > 1 && (size_t)rows * a.heads * splits * (a.dv + 2) * 4 > ws_bytes) splits--;
    }
    a.splits = splits;
    a.per_split = (max_len + splits - 1) / splits;
    a.ws = ws;
    kvq_attn_kernel<NV><<<dim3(a.heads, rows, splits), KVQ_WARPS * 32, 0, st>>>(a);
    if (splits > 1) kvq_merge_kernel<<<dim3(a.heads, rows), a.dv, 0, st>>>(a);
    return fllm_check_launch("kvq_attention");
}

// ---------------------------------------------------------------------------------------------------------------------
// prompts: expand to FP16, then the FP16 prompt kernels

__global__ void kvq_expand_kernel(const uint16_t *pool, FllmKvDesc d, int kv_heads, int dim, const int64_t *table,
                                  int block_size, int count, uint16_t *out) {
    // out[p * kv + c] for p < count, c < kv_heads * dim
    const size_t kv = (size_t)kv_heads * dim;
    const size_t total = (size_t)count * kv;
    for (size_t t = (size_t)blockIdx.x * blockDim.x + threadIdx.x; t < total; t += (size_t)gridDim.x * blockDim.x) {
        const size_t p = t / kv;
        const int c = (int)(t % kv);
        const size_t slot = (size_t)table[p / block_size] * block_size + p % block_size;
        const int kh = c / dim, e = c % dim;
        out[t] = f2h_bits(kvq_at(pool, d, kvq_row(d, slot, kh, dim), e));
    }
}

__global__ void kvq_iota_kernel(int64_t *out, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) out[i] = i;
}

struct KvqScratch {
    void *k = nullptr, *v = nullptr;
    size_t bytes = 0;
    int64_t *iota = nullptr;
    int iota_n = 0;
};
static std::mutex g_scratch_mutex;
static std::map<void *, KvqScratch> g_scratch;

static size_t kvq_scratch_limit() {
    static const size_t mb = [] {
        const char *e = getenv("FLLM_KVQ_SCRATCH_MB");
        return e ? (size_t)atoll(e) : (size_t)256;
    }();
    return mb << 20;
}

// Run `n` consecutive prompt rows [r0, r0 + n) of one sequence through the FP16 prompt kernels. Returns -1 when the shape
// is not covered.
static int kvq_prompt_run(cudaStream_t s, void *key, const float *q, const uint16_t *kpool, const uint16_t *vpool, float *out,
                          const int64_t *dtables, const int64_t *dlengths, const float *sinks, int n, int heads,
                          int kv_heads, int dk, int block_size, const FllmKvDesc &kd, const FllmKvDesc &vd, float scale,
                          int kv_len) {
    if (fllm_cc() < 75 || (dk != 64 && dk != 128)) return -1;
    const int padded = (kv_len + block_size - 1) / block_size * block_size;
    const size_t bytes = (size_t)padded * kv_heads * dk * 2;
    if (bytes * 2 > kvq_scratch_limit()) return -1;
    KvqScratch *sc;
    {
        std::lock_guard<std::mutex> lock(g_scratch_mutex);
        sc = &g_scratch[key];
        if (sc->bytes < bytes) {
            cudaFree(sc->k);  // implicit device synchronization: no kernel still reads the old buffers
            cudaFree(sc->v);
            sc->k = sc->v = nullptr;
            const size_t want = bytes + bytes / 4;
            if (cudaMalloc(&sc->k, want) != cudaSuccess || cudaMalloc(&sc->v, want) != cudaSuccess) {
                cudaGetLastError();
                cudaFree(sc->k);
                sc->k = sc->v = nullptr;
                sc->bytes = 0;
                return -1;
            }
            sc->bytes = want;
        }
        const int blocks_needed = padded / block_size;
        if (sc->iota_n < blocks_needed) {
            cudaFree(sc->iota);
            sc->iota = nullptr;
            sc->iota_n = 0;
            if (cudaMalloc(&sc->iota, (size_t)(blocks_needed + 1024) * sizeof(int64_t)) != cudaSuccess) {
                cudaGetLastError();
                return -1;
            }
            sc->iota_n = blocks_needed + 1024;
            kvq_iota_kernel<<<(sc->iota_n + 255) / 256, 256, 0, s>>>(sc->iota, sc->iota_n);
        }
    }
    const int blocks = (int)min((size_t)4096, ((size_t)kv_len * kv_heads * dk + 255) / 256);
    kvq_expand_kernel<<<blocks, 256, 0, s>>>(kpool, kd, kv_heads, dk, dtables, block_size, kv_len, (uint16_t *)sc->k);
    kvq_expand_kernel<<<blocks, 256, 0, s>>>(vpool, vd, kv_heads, dk, dtables, block_size, kv_len, (uint16_t *)sc->v);
    if (int e = fllm_check_launch("kvq_expand")) return e;
    // the scratch is a pool of identity-mapped blocks; the rows of the run keep their lengths
    return fllm_attention3_try(s, q, (const uint16_t *)sc->k, (const uint16_t *)sc->v, out, sc->iota, dlengths, sinks, n,
                               heads, kv_heads, dk, dk, 0, block_size, 1, scale, 0, 0, kv_len);
}

// Entry point of api.cu: `dtables` / `dlengths` are the staged device copies, `tables_host` / `lengths_host` the host data.
int fllm_kvq_attention(cudaStream_t s, void *stream_key, const float *q, const uint16_t *kpool, const uint16_t *vpool, float *out,
                       const int64_t *dtables, const int64_t *dlengths, const int64_t *tables_host,
                       const int64_t *lengths_host, const float *sinks, const float *alibi, int rows, int heads, int kv_heads,
                       int dk, int dv, int stride, int block_size, const FllmKvDesc &kd, const FllmKvDesc &vd, float scale,
                       float softcap, int window, int chunk, float *ws, size_t ws_bytes, bool capturing, int capture_max_len) {
    if (rows <= 0) return 0;
    if (dk > 512 || dv > 512) return fllm_set_error("kv cache attention: head width %d/%d exceeds 512", dk, dv);
    if (kd.group > KVQ_MAX_UNIT || vd.group > KVQ_MAX_UNIT) return fllm_set_error("kv cache attention: group too large");
    KvqAttnArgs a{};
    a.q = q; a.kpool = kpool; a.vpool = vpool; a.out = out; a.sinks = sinks; a.alibi = alibi;
    a.kd = kd; a.vd = vd;
    a.heads = heads; a.kv_heads = kv_heads; a.dk = dk; a.dv = dv; a.stride = stride; a.block_size = block_size;
    a.scale = scale; a.softcap = softcap; a.window = window; a.chunk = chunk;

    const bool prompt_ok = !capturing && !alibi && softcap == 0.f && window == 0 && chunk == 0 && rows >= 16 && dk == dv &&
                           !fllm_exact_math() && getenv("FLLM_KVQ_NO_EXPAND") == nullptr;
    int r0 = 0;
    bool any_prompt = false;
    if (prompt_ok) {
        // runs of rows of one sequence (same table, lengths growing by one)
        for (int r = 0; r < rows;) {
            int r1 = r + 1;
            while (r1 < rows && lengths_host[r1] == lengths_host[r1 - 1] + 1 &&
                   memcmp(tables_host + (size_t)r1 * stride, tables_host + (size_t)r * stride, (size_t)stride * sizeof(int64_t)) == 0)
                r1++;
            if (r1 - r >= 16) any_prompt = true;
            r = r1;
        }
    }
    auto generic = [&](int from, int count, int max_len) {
        KvqAttnArgs b = a;
        b.q = q + (size_t)from * heads * dk;
        b.out = out + (size_t)from * heads * dv;
        b.tables = dtables + (size_t)from * stride;
        b.lengths = dlengths + from;
        {
            const int rc2 = kvq_attn2_launch(s, b, count, max_len, ws, ws_bytes);
            if (rc2 >= 0) return rc2;
        }
        const int need = dk > dv ? dk : dv;
        if (need <= 64) return kvq_attn_launch<2>(s, b, count, max_len, ws, ws_bytes);
        if (need <= 128) return kvq_attn_launch<4>(s, b, count, max_len, ws, ws_bytes);
        if (need <= 256) return kvq_attn_launch<8>(s, b, count, max_len, ws, ws_bytes);
        return kvq_attn_launch<16>(s, b, count, max_len, ws, ws_bytes);
    };
    auto longest = [&](int from, int count) {
        int mx = 1;
        for (int r = from; r < from + count; r++) mx = lengths_host[r] > mx ? (int)lengths_host[r] : mx;
        return capturing ? capture_max_len : mx;
    };
    if (!any_prompt) return generic(0, rows, longest(0, rows));
    while (r0 < rows) {
        int r1 = r0 + 1;
        while (r1 < rows && lengths_host[r1] == lengths_host[r1 - 1] + 1 &&
               memcmp(tables_host + (size_t)r1 * stride, tables_host + (size_t)r0 * stride, (size_t)stride * sizeof(int64_t)) == 0)
            r1++;
        const int n = r1 - r0;
        int rc = -1;
        if (n >= 16)
            rc = kvq_prompt_run(s, stream_key, q + (size_t)r0 * heads * dk, kpool, vpool, out + (size_t)r0 * heads * dv,
                                dtables + (size_t)r0 * stride, dlengths + r0, sinks, n, heads, kv_heads, dk, block_size, kd, vd,
                                scale, (int)lengths_host[r1 - 1]);
        if (rc > 0) return rc;
        if (rc < 0 || n < 16) {
            if (int e = generic(r0, n, longest(r0, n))) return e;
        }
        r0 = r1;
    }
    return 0;
}
