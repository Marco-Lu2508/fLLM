// fLLM Metal kernels, package 6c (MoE, gated attention, DeltaNet, Mamba): helpers shared by moe_*, gate_*, delta_* and mamba_*.metal.
//
// The file name starts with `common`, so the backend prepends this file to every other kernel file (rust/engine/metal/README.md,
// pipelines/source.rs). Everything here is `inline` or a type, nothing is a kernel, and every symbol has the prefix `r_` (macros `R_`) so that
// it cannot collide with the helpers of the other shader packages when FLLM_METAL_SINGLE_LIB=1 puts all files into one library.
//
// Contents:
//   * activation helpers (SiLU, sigmoid, softplus) that survive fast-math (no infinities are produced),
//   * threadgroup reductions in threadgroup memory (they run on every GPU; no simdgroup operations are assumed),
//   * byte-wise readers for GGUF blocks (block sizes are not multiples of 4 for several formats: 17 bytes MXFP4, 18 bytes Q4_0, 34 bytes Q8_0),
//   * the format traits `r_fmt_*` used by moe_linear: how to take the dot product of one PART of one GGUF block with a float vector.
//
// ASSUMPTION about the weight layout (package 6a may differ; see INTERFACE_RECURRENT.md): the weight of one expert is the raw GGUF tensor, row
// major, `rows x (cols / QK)` blocks of exactly the GGUF block structure (little endian), no padding and no repacking.
#include <metal_stdlib>
using namespace metal;

#define R_UNROLL _Pragma("clang loop unroll(full)")
// finite stand-in for -infinity in running maxima: exp(R_NEG - m) is exactly 0 for any finite m and never produces inf - inf
#define R_NEG (-3.0e38f)

// ---------------------------------------------------------------------------------------------------------------------------------------
// activations
// ---------------------------------------------------------------------------------------------------------------------------------------

// exp() arguments are capped at 80 (exp(80) = 5.5e34 is finite) so that no kernel ever computes an infinity, which fast-math may mishandle.
inline float r_sigmoid(float x) { return 1.0f / (1.0f + precise::exp(min(-x, 80.0f))); }
inline float r_silu(float x) { return x / (1.0f + precise::exp(min(-x, 80.0f))); }
// ln(1 + exp(x)), the value itself above 20 (CPU: `if raw > 20 { raw } else { raw.exp().ln_1p() }`)
inline float r_softplus20(float x) { return x > 20.0f ? x : precise::log(1.0f + precise::exp(min(x, 20.0f))); }

// ---------------------------------------------------------------------------------------------------------------------------------------
// threadgroup reductions: EVERY thread of the threadgroup must call them from uniform control flow (they contain barriers). `red` is a
// threadgroup array of at least WG floats. The result is returned to all threads and `red` can be reused right away.
// ---------------------------------------------------------------------------------------------------------------------------------------

template <uint WG>
inline float r_tg_sum(float v, threadgroup float *red, uint lane) {
    red[lane] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = WG / 2u; s > 0u; s >>= 1u) {
        if (lane < s) red[lane] += red[lane + s];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float r = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

template <uint WG>
inline float r_tg_max(float v, threadgroup float *red, uint lane) {
    red[lane] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint s = WG / 2u; s > 0u; s >>= 1u) {
        if (lane < s) red[lane] = max(red[lane], red[lane + s]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const float r = red[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return r;
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// byte-wise readers (GGUF blocks are only 1- or 2-byte aligned)
// ---------------------------------------------------------------------------------------------------------------------------------------

inline uint r_u16(const device uchar *p) { return uint(p[0]) | (uint(p[1]) << 8); }
inline uint r_u32(const device uchar *p) { return uint(p[0]) | (uint(p[1]) << 8) | (uint(p[2]) << 16) | (uint(p[3]) << 24); }
// IEEE binary16 stored little endian
inline float r_f16(const device uchar *p) { return float(as_type<half>(ushort(r_u16(p)))); }
// two's complement byte
inline int r_i8(uchar v) { return v < 128 ? int(v) : int(v) - 256; }

// MXFP4 scale 2^(e - 128) (E8M0 exponent, already halved because the E2M1 table below holds doubled values)
inline float r_e8m0_half(uchar e) {
    const uint bits = e < 2 ? (0x00200000u << uint(e)) : ((uint(e) - 1u) << 23);
    return as_type<float>(bits);
}

// E2M1 values at twice their magnitude (MXFP4, NVFP4)
constant float r_fp4[16] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 6.0f, 8.0f, 12.0f, 0.0f, -1.0f, -2.0f, -3.0f, -4.0f, -6.0f, -8.0f, -12.0f};
// IQ4_NL non-linear 4-bit values
constant float r_iq4nl[16] = {-127.0f, -104.0f, -83.0f, -65.0f, -49.0f, -35.0f, -22.0f, -10.0f, 1.0f, 13.0f, 25.0f, 38.0f, 53.0f, 69.0f, 89.0f, 113.0f};

// scale and minimum of sub-block g (0..7) of a Q4_K / Q5_K block from its 12 packed 6-bit bytes
inline void r_k4_scale_min(const device uchar *sc, uint g, thread uint &s, thread uint &m) {
    if (g < 4u) {
        s = uint(sc[g]) & 63u;
        m = uint(sc[g + 4u]) & 63u;
    } else {
        s = (uint(sc[g + 4u]) & 15u) | ((uint(sc[g - 4u]) >> 6) << 4);
        m = (uint(sc[g + 4u]) >> 4) | ((uint(sc[g]) >> 6) << 4);
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// Format traits for moe_linear. A block of QK weights occupies BYTES bytes and is processed in PARTS parts of PART consecutive weights
// (part p = weights p * PART .. (p + 1) * PART - 1 of the block). dot_part(block, part, x) returns the dot product of the decoded weights of that part
// with x[0 .. PART - 1] (the caller passes x already offset to the first weight of the part). Layouts follow the GGUF block structures; the
// decoded values are the ones of compute.rs `dequantize` / quant_extra.rs / quant_iq.rs (the CPU reference).
// ---------------------------------------------------------------------------------------------------------------------------------------

// Q4_0: { half d; uchar qs[16]; }  w[j] = d * ((qs[j] & 15) - 8) for j < 16, w[j + 16] = d * ((qs[j] >> 4) - 8)
struct r_fmt_q4_0 {
    static constant constexpr const uint QK = 32;
    static constant constexpr const uint BYTES = 18;
    static constant constexpr const uint PARTS = 4;
    static constant constexpr const uint PART = 8;
    static inline float dot_part(const device uchar *b, uint part, const device float *x) {
        const float d = r_f16(b);
        const device uchar *qs = b + 2 + (part & 1u) * 8u;
        const bool hi = part >= 2u;
        float sq = 0.0f;
        float sx = 0.0f;
        R_UNROLL for (uint i = 0; i < 8u; i++) {
            const uint q = hi ? (uint(qs[i]) >> 4) : (uint(qs[i]) & 15u);
            const float xv = x[i];
            sq += float(q) * xv;
            sx += xv;
        }
        return d * (sq - 8.0f * sx);
    }
};

// Q4_1: { half d; half m; uchar qs[16]; }  w = d * q + m
struct r_fmt_q4_1 {
    static constant constexpr const uint QK = 32;
    static constant constexpr const uint BYTES = 20;
    static constant constexpr const uint PARTS = 4;
    static constant constexpr const uint PART = 8;
    static inline float dot_part(const device uchar *b, uint part, const device float *x) {
        const float d = r_f16(b);
        const float m = r_f16(b + 2);
        const device uchar *qs = b + 4 + (part & 1u) * 8u;
        const bool hi = part >= 2u;
        float sq = 0.0f;
        float sx = 0.0f;
        R_UNROLL for (uint i = 0; i < 8u; i++) {
            const uint q = hi ? (uint(qs[i]) >> 4) : (uint(qs[i]) & 15u);
            const float xv = x[i];
            sq += float(q) * xv;
            sx += xv;
        }
        return d * sq + m * sx;
    }
};

// Q5_0: { half d; uchar qh[4]; uchar qs[16]; }  q = nibble | (bit j of qh) << 4, w = d * (q - 16)
struct r_fmt_q5_0 {
    static constant constexpr const uint QK = 32;
    static constant constexpr const uint BYTES = 22;
    static constant constexpr const uint PARTS = 4;
    static constant constexpr const uint PART = 8;
    static inline float dot_part(const device uchar *b, uint part, const device float *x) {
        const float d = r_f16(b);
        const uint qh = r_u32(b + 2);
        const device uchar *qs = b + 6 + (part & 1u) * 8u;
        const bool hi = part >= 2u;
        float sq = 0.0f;
        float sx = 0.0f;
        R_UNROLL for (uint i = 0; i < 8u; i++) {
            const uint j = part * 8u + i;
            const uint nib = hi ? (uint(qs[i]) >> 4) : (uint(qs[i]) & 15u);
            const uint q = nib | (((qh >> j) & 1u) << 4);
            const float xv = x[i];
            sq += float(q) * xv;
            sx += xv;
        }
        return d * (sq - 16.0f * sx);
    }
};

// Q5_1: { half d; half m; uchar qh[4]; uchar qs[16]; }  w = d * q + m
struct r_fmt_q5_1 {
    static constant constexpr const uint QK = 32;
    static constant constexpr const uint BYTES = 24;
    static constant constexpr const uint PARTS = 4;
    static constant constexpr const uint PART = 8;
    static inline float dot_part(const device uchar *b, uint part, const device float *x) {
        const float d = r_f16(b);
        const float m = r_f16(b + 2);
        const uint qh = r_u32(b + 4);
        const device uchar *qs = b + 8 + (part & 1u) * 8u;
        const bool hi = part >= 2u;
        float sq = 0.0f;
        float sx = 0.0f;
        R_UNROLL for (uint i = 0; i < 8u; i++) {
            const uint j = part * 8u + i;
            const uint nib = hi ? (uint(qs[i]) >> 4) : (uint(qs[i]) & 15u);
            const uint q = nib | (((qh >> j) & 1u) << 4);
            const float xv = x[i];
            sq += float(q) * xv;
            sx += xv;
        }
        return d * sq + m * sx;
    }
};

// Q8_0: { half d; char qs[32]; }  w = d * qs[j]
struct r_fmt_q8_0 {
    static constant constexpr const uint QK = 32;
    static constant constexpr const uint BYTES = 34;
    static constant constexpr const uint PARTS = 4;
    static constant constexpr const uint PART = 8;
    static inline float dot_part(const device uchar *b, uint part, const device float *x) {
        const float d = r_f16(b);
        const device uchar *qs = b + 2 + part * 8u;
        float sq = 0.0f;
        R_UNROLL for (uint i = 0; i < 8u; i++) sq += float(r_i8(qs[i])) * x[i];
        return d * sq;
    }
};

// IQ4_NL: { half d; uchar qs[16]; }  w = d * kvalues_iq4nl[nibble], nibble order as Q4_0
struct r_fmt_iq4_nl {
    static constant constexpr const uint QK = 32;
    static constant constexpr const uint BYTES = 18;
    static constant constexpr const uint PARTS = 4;
    static constant constexpr const uint PART = 8;
    static inline float dot_part(const device uchar *b, uint part, const device float *x) {
        const float d = r_f16(b);
        const device uchar *qs = b + 2 + (part & 1u) * 8u;
        const bool hi = part >= 2u;
        float sq = 0.0f;
        R_UNROLL for (uint i = 0; i < 8u; i++) {
            const uint q = hi ? (uint(qs[i]) >> 4) : (uint(qs[i]) & 15u);
            sq += r_iq4nl[q] * x[i];
        }
        return d * sq;
    }
};

// MXFP4: { uchar e; uchar qs[16]; } (17 bytes)  w = e8m0_half(e) * fp4[nibble], nibble order as Q4_0
struct r_fmt_mxfp4 {
    static constant constexpr const uint QK = 32;
    static constant constexpr const uint BYTES = 17;
    static constant constexpr const uint PARTS = 4;
    static constant constexpr const uint PART = 8;
    static inline float dot_part(const device uchar *b, uint part, const device float *x) {
        const float d = r_e8m0_half(b[0]);
        const device uchar *qs = b + 1 + (part & 1u) * 8u;
        const bool hi = part >= 2u;
        float sq = 0.0f;
        R_UNROLL for (uint i = 0; i < 8u; i++) {
            const uint q = hi ? (uint(qs[i]) >> 4) : (uint(qs[i]) & 15u);
            sq += r_fp4[q] * x[i];
        }
        return d * sq;
    }
};

// Q4_K: { half d; half dmin; uchar scales[12]; uchar qs[128]; } (144 bytes), 8 sub-blocks of 32 weights;
// sub-block g reads qs[(g / 2) * 32 + j], low nibble for even g, high nibble for odd g;  w = d * s_g * q - dmin * m_g
struct r_fmt_q4_k {
    static constant constexpr const uint QK = 256;
    static constant constexpr const uint BYTES = 144;
    static constant constexpr const uint PARTS = 8;
    static constant constexpr const uint PART = 32;
    static inline float dot_part(const device uchar *b, uint part, const device float *x) {
        const float d = r_f16(b);
        const float dmin = r_f16(b + 2);
        uint s;
        uint m;
        r_k4_scale_min(b + 4, part, s, m);
        const device uchar *qs = b + 16 + (part >> 1) * 32u;
        const bool hi = (part & 1u) != 0u;
        float sq = 0.0f;
        float sx = 0.0f;
        for (uint j = 0; j < 32u; j++) {
            const uint q = hi ? (uint(qs[j]) >> 4) : (uint(qs[j]) & 15u);
            const float xv = x[j];
            sq += float(q) * xv;
            sx += xv;
        }
        return d * float(s) * sq - dmin * float(m) * sx;
    }
};

// Q5_K: { half d; half dmin; uchar scales[12]; uchar qh[32]; uchar qs[128]; } (176 bytes); as Q4_K plus bit g of qh[j] as the 5th bit
struct r_fmt_q5_k {
    static constant constexpr const uint QK = 256;
    static constant constexpr const uint BYTES = 176;
    static constant constexpr const uint PARTS = 8;
    static constant constexpr const uint PART = 32;
    static inline float dot_part(const device uchar *b, uint part, const device float *x) {
        const float d = r_f16(b);
        const float dmin = r_f16(b + 2);
        uint s;
        uint m;
        r_k4_scale_min(b + 4, part, s, m);
        const device uchar *qh = b + 16;
        const device uchar *qs = b + 48 + (part >> 1) * 32u;
        const bool hi = (part & 1u) != 0u;
        float sq = 0.0f;
        float sx = 0.0f;
        for (uint j = 0; j < 32u; j++) {
            const uint nib = hi ? (uint(qs[j]) >> 4) : (uint(qs[j]) & 15u);
            const uint q = nib | (((uint(qh[j]) >> part) & 1u) << 4);
            const float xv = x[j];
            sq += float(q) * xv;
            sx += xv;
        }
        return d * float(s) * sq - dmin * float(m) * sx;
    }
};

// Q6_K: { uchar ql[128]; uchar qh[64]; char scales[16]; half d; } (210 bytes); two halves n of 128 weights, in each four groups of 32;
// part p = 16 weights: n = p / 8, group = (p % 8) / 2, jh = p % 2, scale index n * 8 + group * 2 + jh;
// q = low/high nibble of ql[n * 64 + (group & 1) * 32 + j] (high for group >= 2) | ((qh[n * 32 + j] >> (2 * group)) & 3) << 4,  w = d * scale * (q - 32)
struct r_fmt_q6_k {
    static constant constexpr const uint QK = 256;
    static constant constexpr const uint BYTES = 210;
    static constant constexpr const uint PARTS = 16;
    static constant constexpr const uint PART = 16;
    static inline float dot_part(const device uchar *b, uint part, const device float *x) {
        const float d = r_f16(b + 208);
        const uint n = part >> 3;
        const uint grp = (part & 7u) >> 1;
        const uint jh = part & 1u;
        const device uchar *ql = b + n * 64u + (grp & 1u) * 32u + jh * 16u;
        const device uchar *qh = b + 128 + n * 32u + jh * 16u;
        const float scale = float(r_i8(b[192 + n * 8u + grp * 2u + jh]));
        const bool hi = grp >= 2u;
        float sq = 0.0f;
        float sx = 0.0f;
        R_UNROLL for (uint i = 0; i < 16u; i++) {
            const uint nib = hi ? (uint(ql[i]) >> 4) : (uint(ql[i]) & 15u);
            const uint q = nib | (((uint(qh[i]) >> (2u * grp)) & 3u) << 4);
            const float xv = x[i];
            sq += float(q) * xv;
            sx += xv;
        }
        return d * scale * (sq - 32.0f * sx);
    }
};
