// fLLM Metal kernels of the non-matmul operations (package p06b): shared helpers, parameter structs and weight/KV decoders.
//
// This file contains no kernel, so the backend prepends it to every other source unit (see API.md of the Metal core). Every symbol of this file
// starts with `o_` so that it cannot clash with the helpers of the other shader packages that are prepended as well. Parameter structs consist
// of 4-byte scalars (plus leading 8-byte `ulong` fields) and mirror `Params` in `src/backend/metal/ops.rs`; the plan tests compare both.
// Conventions: f32 activations, f32 accumulation, `constant` struct buffers, `FLLM_*` defines are provided by the backend.
//
// Fast-math note: the backend compiles with fast math by default (FLLM_METAL_FAST_MATH=1). The kernels therefore never rely on inf/NaN
// arithmetic: -inf is represented by the finite `o_neg`, argmax compares integer keys, and tanh/exp arguments are clamped.
#ifndef FLLM_OPS_COMMON_METAL
#define FLLM_OPS_COMMON_METAL

#include <metal_stdlib>
using namespace metal;

#ifndef FLLM_HAS_SIMDGROUP_MATRIX
#define FLLM_HAS_SIMDGROUP_MATRIX 1
#endif

#define O_UNROLL _Pragma("clang loop unroll(full)")

// Finite stand-in for -infinity in running maxima: exp(o_neg - m) is exactly 0 for any finite m and never produces inf - inf. Large enough for
// every score, small enough that `o_neg * log2(e)` (what a fast exp computes) does not overflow to -inf.
constant float o_neg = -1.0e30f;

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Parameter structs (mirrors of `Params` in ops.rs; all members are 4-byte scalars unless a `ulong` leads the struct)
// ---------------------------------------------------------------------------------------------------------------------------------------------

struct o_norm_params {
    uint cols;
    uint rows;
    uint x_stride;
    uint x_off;
    uint o_stride;
    uint o_off;
    float eps;
    uint flags;  // bit 0: rms_norm: weight present; layer_norm: bias present
};

struct o_rope_params {
    uint rows;
    uint heads;
    uint kv_heads;
    uint dim;
    uint half_dim;
    uint neox;
};

struct o_act_params {
    uint n_rows;
    uint width;
    uint gate_stride;
    uint gate_off;
    uint up_stride;
    uint up_off;
    uint out_stride;
    float alpha;
    float limit;
};

struct o_count_params {
    uint count;
    float scale;
};

struct o_split_params {
    uint rows;
    uint heads;
    uint dk;
    uint per_head;
};

struct o_argmax_params {
    uint vocab;
    uint rows;
};

struct o_embed_params {
    ulong plane1;    // planar MXFP4: byte offset of the plane of scale bytes (else 0)
    uint n_ids;
    uint cols;
    uint row_bytes;  // bytes of one table row (code plane of one row for the planar layout)
    uint _pad;
};

struct o_kvw_params {
    ulong base;        // layer start in 16-bit words
    ulong scale_base;  // q8: 16-bit words from the layer start to the first scale
    uint rows;
    uint kv;           // kv_heads * dim
    uint group;        // q8: values per scale
    uint _pad;
};

struct o_attn_params {
    ulong k_base;   // layer start in the key pool, 16-bit words
    ulong v_base;
    ulong k_scale;  // q8: 16-bit words from the layer start to the first scale
    ulong v_scale;
    uint heads;
    uint kv_heads;
    uint dk;
    uint dv;
    uint stride;      // block table entries per row
    uint block_size;  // tokens per KV block
    float scale;
    float softcap;  // 0: off
    uint window;    // 0: off
    uint chunk;     // 0: off
    uint flags;     // bit 0: sinks, bit 1: ALiBi
    uint nsplit;    // decode: splits of the key range
    uint rows_total;
    uint row0;      // first global row of this dispatch
    uint rows_end;  // prefill: one past the last global row of this dispatch
    uint k_gshift;  // q8: log2 of the group (5), or 0 when the group is the whole head
    uint v_gshift;
    uint _pad;
};

struct o_merge_params {
    uint heads;
    uint dv;
    uint nsplit;
    uint flags;
    uint row0;
};

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Scalar helpers
// ---------------------------------------------------------------------------------------------------------------------------------------------

inline float o_silu(float x) { return x / (1.0f + exp(-x)); }
inline float o_sigmoid(float x) { return 1.0f / (1.0f + exp(-x)); }

// tanh approximation of GELU as in the CPU decoder (`gelu_tanh`); the argument of tanh is clamped to +-15 (exactly 1 in f32 there) so that a
// tanh computed through exp() cannot produce NaN.
inline float o_gelu_tanh(float x) {
    float t = clamp(0.7978846f * (x + 0.044715f * x * x * x), -15.0f, 15.0f);
    return 0.5f * x * (1.0f + tanh(t));
}

// Logit soft cap: cap * tanh(s / cap)
inline float o_softcap(float s, float cap) { return cap * tanh(clamp(s / cap, -15.0f, 15.0f)); }

// Integer sort key of a float: key(a) > key(b) iff a > b for non-NaN values (-0 equals +0); a NaN gets the smallest key 0 and so never wins an
// argmax. Integer compares are immune to the fast-math "no NaN" assumption.
inline uint o_order_key(float v) {
    uint b = as_type<uint>(v);
    if ((b & 0x7FFFFFFFu) > 0x7F800000u) return 0u;
    if (b == 0x80000000u) b = 0u;
    return (b & 0x80000000u) != 0u ? ~b : (b | 0x80000000u);
}

// first position a row of length n attends to (AttentionParams: chunk, window or 0)
inline uint o_row_start(uint n, uint window, uint chunk) {
    if (chunk != 0u) return (n - 1u) / chunk * chunk;
    if (window != 0u && n > window) return n - window;
    return 0u;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Threadgroup reductions. All threads of the group must call them (they contain barriers); `red` has at least 32 floats (one per simdgroup).
// The partial results are added in a fixed order, so the sum does not depend on the scheduling.
// ---------------------------------------------------------------------------------------------------------------------------------------------

inline float o_tg_sum(float v, threadgroup float *red, uint sg, uint lane, uint nsg) {
    v = simd_sum(v);
    if (lane == 0u) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = 0.0f;
    for (uint i = 0u; i < nsg; i++) r += red[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);  // red may be reused right away
    return r;
}

inline float o_tg_max(float v, threadgroup float *red, uint sg, uint lane, uint nsg) {
    v = simd_max(v);
    if (lane == 0u) red[sg] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float r = red[0];
    for (uint i = 1u; i < nsg; i++) r = max(r, red[i]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Unaligned reads from byte arrays (quantized blocks start at arbitrary even or odd offsets)
// ---------------------------------------------------------------------------------------------------------------------------------------------

inline float o_f16(device const uchar *p) { return float(as_type<half>(ushort(uint(p[0]) | (uint(p[1]) << 8)))); }

inline uint o_u32(device const uchar *p) { return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24); }

inline float o_i8(uchar b) { return float(as_type<char>(b)); }

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Dequantization of one element of a quantized block in the GGUF layout of the file (used by the embedding lookup). `b` points to the block,
// `i` is the element index inside the block. Reference: `compute::dequantize`, `quant_extra`, `quant_iq` of the engine.
// ---------------------------------------------------------------------------------------------------------------------------------------------

#define O_FMT_Q4_0 0
#define O_FMT_Q4_1 1
#define O_FMT_Q5_0 2
#define O_FMT_Q5_1 3
#define O_FMT_Q8_0 4
#define O_FMT_Q2_K 5
#define O_FMT_Q3_K 6
#define O_FMT_Q4_K 7
#define O_FMT_Q5_K 8
#define O_FMT_Q6_K 9
#define O_FMT_IQ4_NL 10
#define O_FMT_MXFP4 11

// values per block
constexpr uint o_fmt_qk(int fmt) { return fmt >= O_FMT_Q2_K && fmt <= O_FMT_Q6_K ? 256u : 32u; }

// bytes per block
constexpr uint o_fmt_bytes(int fmt) {
    return fmt == O_FMT_Q4_0 ? 18u
        : fmt == O_FMT_Q4_1 ? 20u
        : fmt == O_FMT_Q5_0 ? 22u
        : fmt == O_FMT_Q5_1 ? 24u
        : fmt == O_FMT_Q8_0 ? 34u
        : fmt == O_FMT_Q2_K ? 84u
        : fmt == O_FMT_Q3_K ? 110u
        : fmt == O_FMT_Q4_K ? 144u
        : fmt == O_FMT_Q5_K ? 176u
        : fmt == O_FMT_Q6_K ? 210u
        : fmt == O_FMT_IQ4_NL ? 18u
        : 17u;
}

constant char o_kvalues_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};

// FP4 (E2M1) values doubled, as in the engine's `FP4` table (the E8M0 scale is halved to compensate)
constant float o_fp4[16] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f, 0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f};

inline float o_deq_q4_0(device const uchar *b, uint i) {
    uint byte = b[2u + (i & 15u)];
    uint q = i < 16u ? (byte & 15u) : (byte >> 4);
    return o_f16(b) * (float(q) - 8.0f);
}

inline float o_deq_q4_1(device const uchar *b, uint i) {
    uint byte = b[4u + (i & 15u)];
    uint q = i < 16u ? (byte & 15u) : (byte >> 4);
    return o_f16(b) * float(q) + o_f16(b + 2);
}

inline float o_deq_q5_0(device const uchar *b, uint i) {
    uint byte = b[6u + (i & 15u)];
    uint q = i < 16u ? (byte & 15u) : (byte >> 4);
    q |= ((o_u32(b + 2) >> i) & 1u) << 4;
    return o_f16(b) * (float(q) - 16.0f);
}

inline float o_deq_q5_1(device const uchar *b, uint i) {
    uint byte = b[8u + (i & 15u)];
    uint q = i < 16u ? (byte & 15u) : (byte >> 4);
    q |= ((o_u32(b + 4) >> i) & 1u) << 4;
    return o_f16(b) * float(q) + o_f16(b + 2);
}

inline float o_deq_q8_0(device const uchar *b, uint i) { return o_f16(b) * o_i8(b[2u + i]); }

// Q2_K: scales[16] (low nibble: scale, high nibble: min), qs[64], d, dmin
inline float o_deq_q2_k(device const uchar *b, uint i) {
    uint n = i >> 7;
    uint r = i & 127u;
    uint s = r >> 5;
    uint hg = (r >> 4) & 1u;
    uint l = r & 15u;
    uint sc = b[n * 8u + s * 2u + hg];
    uint q = (b[16u + n * 32u + hg * 16u + l] >> (2u * s)) & 3u;
    float d = o_f16(b + 80);
    float dmin = o_f16(b + 82);
    return d * float(sc & 15u) * float(q) - dmin * float(sc >> 4);
}

// Q3_K: hmask[32], qs[64], scales[12] (6 bit), d
inline float o_deq_q3_k(device const uchar *b, uint i) {
    uint outer = i >> 7;
    uint rem = i & 127u;
    uint sub = rem >> 5;
    uint within = rem & 31u;
    uint side = within >> 4;
    uint si = outer * 8u + sub * 2u + side;
    uint i4 = si & 3u;
    uint grp = si >> 2;
    uint hi = b[96u + i4];
    uint lo = b[100u + i4];
    uint packed = b[104u + i4];
    uint src = (grp & 1u) != 0u ? lo : hi;
    uint nib = grp >= 2u ? (src >> 4) : (src & 15u);
    int sc = int(nib | (((packed >> (2u * grp)) & 3u) << 4)) - 32;
    int low = int((b[32u + outer * 32u + within] >> (sub * 2u)) & 3u);
    int high = ((b[within] >> (outer * 4u + sub)) & 1u) != 0u ? 0 : 4;
    return o_f16(b + 108) * float(sc) * float(low - high);
}

// scale and min of group g (0..7) of a Q4_K / Q5_K block
inline void o_scale_min_k4(device const uchar *s, uint g, thread uint &sc, thread uint &mn) {
    if (g < 4u) {
        sc = s[g] & 63u;
        mn = s[g + 4u] & 63u;
    } else {
        sc = (s[g + 4u] & 15u) | ((uint(s[g - 4u]) >> 6) << 4);
        mn = (uint(s[g + 4u]) >> 4) | ((uint(s[g]) >> 6) << 4);
    }
}

// Q4_K: d, dmin, scales[12], qs[128]
inline float o_deq_q4_k(device const uchar *b, uint i) {
    uint g = i >> 5;
    uint j = i & 31u;
    uint sc, mn;
    o_scale_min_k4(b + 4, g, sc, mn);
    uint byte = b[16u + (g >> 1) * 32u + j];
    uint q = (g & 1u) != 0u ? (byte >> 4) : (byte & 15u);
    return o_f16(b) * float(sc) * float(q) - o_f16(b + 2) * float(mn);
}

// Q5_K: d, dmin, scales[12], qh[32], qs[128]
inline float o_deq_q5_k(device const uchar *b, uint i) {
    uint g = i >> 5;
    uint j = i & 31u;
    uint sc, mn;
    o_scale_min_k4(b + 4, g, sc, mn);
    uint byte = b[48u + (g >> 1) * 32u + j];
    uint q = (g & 1u) != 0u ? (byte >> 4) : (byte & 15u);
    q |= ((uint(b[16u + j]) >> g) & 1u) << 4;
    return o_f16(b) * float(sc) * float(q) - o_f16(b + 2) * float(mn);
}

// Q6_K: ql[128], qh[64], scales[16] (int8), d
inline float o_deq_q6_k(device const uchar *b, uint i) {
    uint n = i >> 7;
    uint r = i & 127u;
    uint grp = r >> 5;
    uint j = r & 31u;
    uint lowbyte = b[n * 64u + j + ((grp & 1u) != 0u ? 32u : 0u)];
    uint low = grp >= 2u ? (lowbyte >> 4) : (lowbyte & 15u);
    uint high = (uint(b[128u + n * 32u + j]) >> (2u * grp)) & 3u;
    float scale = o_i8(b[192u + n * 8u + (j >> 4) + grp * 2u]);
    return o_f16(b + 208) * scale * (float(low | (high << 4)) - 32.0f);
}

inline float o_deq_iq4_nl(device const uchar *b, uint i) {
    uint byte = b[2u + (i & 15u)];
    uint q = i < 16u ? (byte & 15u) : (byte >> 4);
    return o_f16(b) * float(o_kvalues_iq4nl[q]);
}

// half of 2^(e - 127) (the E8M0 scale of MXFP4), denormal bit patterns for e < 2 (as the CPU decoder `e8m0_half`)
inline float o_e8m0_half(uint e) { return as_type<float>(e < 2u ? (0x00200000u << e) : ((e - 1u) << 23)); }

// value i of an MXFP4 block given its scale byte and code byte (value j: low nibble of code byte j, value j + 16: high nibble of code byte j)
inline float o_mxfp4_value(uint scale_byte, uint code_byte, uint i) {
    uint q = i < 16u ? (code_byte & 15u) : (code_byte >> 4);
    return o_fp4[q] * o_e8m0_half(scale_byte);
}

// MXFP4 in the layout of the file: E8M0 scale byte, 16 code bytes
inline float o_deq_mxfp4(device const uchar *b, uint i) { return o_mxfp4_value(b[0], b[1u + (i & 15u)], i); }

template <int FMT>
inline float o_deq(device const uchar *b, uint i) {
    if (FMT == O_FMT_Q4_0) return o_deq_q4_0(b, i);
    if (FMT == O_FMT_Q4_1) return o_deq_q4_1(b, i);
    if (FMT == O_FMT_Q5_0) return o_deq_q5_0(b, i);
    if (FMT == O_FMT_Q5_1) return o_deq_q5_1(b, i);
    if (FMT == O_FMT_Q8_0) return o_deq_q8_0(b, i);
    if (FMT == O_FMT_Q2_K) return o_deq_q2_k(b, i);
    if (FMT == O_FMT_Q3_K) return o_deq_q3_k(b, i);
    if (FMT == O_FMT_Q4_K) return o_deq_q4_k(b, i);
    if (FMT == O_FMT_Q5_K) return o_deq_q5_k(b, i);
    if (FMT == O_FMT_Q6_K) return o_deq_q6_k(b, i);
    if (FMT == O_FMT_IQ4_NL) return o_deq_iq4_nl(b, i);
    return o_deq_mxfp4(b, i);
}

// ---------------------------------------------------------------------------------------------------------------------------------------------
// Paged KV pools (DeviceKvCache, backend/mod.rs). The pool is bound as `device char *`; a layer may start at an ODD 16-bit word, so the f16 view
// indexes 2-byte elements and the q8 view indexes bytes:
//   f16: value (slot, c) = half at element  base + slot * kv + c
//   q8:  value (slot, c) = int8 at byte  2 * base + slot * kv + c   *   half scale at element  base + scale_words + slot * gps + c / group
// with kv = kv_heads * dim and gps = kv / group (group = 32 when dim is a multiple of 32, else dim; `gshift` = 5 resp. 0).
// ---------------------------------------------------------------------------------------------------------------------------------------------

// element (slot, c) of a pool; c is the channel inside the slot row (kv_head * dim + d), kh the kv head (used when the group is a whole head)
template <bool Q8>
inline float o_kv_at(device const char *pool, ulong base, ulong scale_words, uint gps, uint gshift, uint kvw, ulong slot, uint c, uint kh) {
    device const half *ph = (device const half *)pool;
    if (!Q8) return float(ph[base + slot * kvw + c]);
    uint si = gshift != 0u ? (c >> gshift) : kh;
    float sc = float(ph[base + scale_words + slot * gps + si]);
    return float(pool[2ul * base + slot * kvw + c]) * sc;
}

// `n` (<= E) consecutive channels c0 .. c0 + n - 1 of slot `slot` into out[0..E) (the rest is zero)
template <bool Q8, int E>
inline void o_kv_load(device const char *pool, ulong base, ulong scale_words, uint gps, uint gshift, uint kvw, ulong slot, uint c0, uint kh, uint n, thread float *out) {
    device const half *ph = (device const half *)pool;
    if (!Q8) {
        device const half *p = ph + base + slot * kvw + c0;
        O_UNROLL
        for (int i = 0; i < E; i++) out[i] = uint(i) < n ? float(p[i]) : 0.0f;
    } else {
        device const char *pb = pool + 2ul * base + slot * kvw + c0;
        device const half *ps = ph + base + scale_words + slot * gps;
        uint last = 0xFFFFFFFFu;
        float sc = 0.0f;
        O_UNROLL
        for (int i = 0; i < E; i++) {
            float v = 0.0f;
            if (uint(i) < n) {
                uint si = gshift != 0u ? ((c0 + uint(i)) >> gshift) : kh;
                if (si != last) {
                    sc = float(ps[si]);
                    last = si;
                }
                v = float(pb[i]) * sc;
            }
            out[i] = v;
        }
    }
}

#endif
