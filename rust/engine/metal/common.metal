// Shared definitions of the fLLM Metal kernels: weight block layouts and chunk decoders of the quantized linear kernels.
// Owner: package p06a (quantized matvec / matmul). Every symbol of this file starts with `q_` / `Q_`.
//
// Original work (Apache-2.0). The kernel structure (several weight rows per simdgroup with a simd_sum reduction for small
// batches, simdgroup_matrix tiles with the weights dequantized into threadgroup memory for large batches) follows the public
// design of MLX (MIT) and of other Metal LLM engines; no source text of them is copied. The block layouts and dequantization
// equations are those of the GGUF formats as implemented by the CPU decoders of this repository (compute.rs, quant_extra.rs,
// quant_iq.rs; those files credit the ggml authors, MIT, Copyright (c) 2023-2026 The ggml authors); the decoders below are
// checked against `compute::dequantize` by metal/emu (CPU emulation, no GPU involved).
//
// Verification level: L0 (parsed with a clang shim, executed as C++ on the CPU; never compiled by the Metal compiler).
#include <metal_stdlib>
using namespace metal;

#define Q_FUNC inline __attribute__((always_inline))
#define Q_CONST static constant constexpr const

// Pointer to weight bytes. Alignment promised by the backend: the weight buffer (plus offset) is 4-byte aligned, a block of the
// raw GGUF layout is 2-byte aligned for the formats whose block size is not a multiple of 4 (Q4_0, Q5_0, Q8_0, IQ4_NL, Q3_K,
// Q6_K); only these loads are issued with 16 bit granularity.
typedef device const uchar *q_bytes;

Q_FUNC float q_f16(q_bytes p, uint off) { return float(*(device const half *)(p + off)); }
Q_FUNC uint q_u16(q_bytes p, uint off) { return uint(*(device const ushort *)(p + off)); }
Q_FUNC uint q_u32(q_bytes p, uint off) { return *(device const uint *)(p + off); }
// 32 bit value at a 2-byte aligned address
Q_FUNC uint q_u32w(q_bytes p, uint off) { return q_u16(p, off) | (q_u16(p, off + 2u) << 16); }
// sign extension of the low byte
Q_FUNC float q_s8(uint v) { return float(int((v & 0xFFu) ^ 0x80u) - 128); }

// code books
constant char q_kv_iq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
// FP4 E2M1 values times two
constant char q_kv_fp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// bitsandbytes 4-bit code books as float bit patterns (gguf-probe `bnb::NF4_CODE` / `FP4_CODE`; the emulation compares every entry with them)
constant uint q_kv_nf4[16] = {0xbf800000u, 0xbf3239b1u, 0xbf066b30u, 0xbeca32a0u, 0xbe91a24du, 0xbe3d353fu, 0xbdba7871u, 0x00000000u,
                              0x3da2faffu, 0x3e24cae3u, 0x3e7c04ddu, 0x3ead033au, 0x3ee1a4b8u, 0x3f1007abu, 0x3f3913b3u, 0x3f800000u};
constant uint q_kv_bnbfp4[16] = {0x00000000u, 0x3baaaaabu, 0x3f2aaaabu, 0x3f800000u, 0x3eaaaaabu, 0x3f000000u, 0x3e2aaaabu, 0x3e800000u,
                                 0x80000000u, 0xbbaaaaabu, 0xbf2aaaabu, 0xbf800000u, 0xbeaaaaabu, 0xbf000000u, 0xbe2aaaabu, 0xbe800000u};

// half of 2^(x - 127) (the E8M0 scale of MXFP4), denormal bit patterns for x < 2 (as the CPU decoder)
Q_FUNC float q_e8m0_half(uint x) { return as_type<float>(x < 2u ? (0x00200000u << x) : ((x - 1u) << 23)); }

// half of the unsigned E4M3 value of the NVFP4 block scales (0 and 0x7f give 0)
Q_FUNC float q_ue4m3_half(uint x) {
    if (x == 0u || x == 0x7fu) return 0.0f;
    const uint e = (x >> 3) & 15u;
    const uint m = x & 7u;
    if (e == 0u) return float(m) * (1.0f / 1024.0f);
    return 0.5f * as_type<float>(((e + 120u) << 23) | (m << 20));
}

// FP8 E4M3 (finite-only variant, bias 7, NaN = S.1111.111)
Q_FUNC float q_e4m3(uint x) {
    const uint e = (x >> 3) & 15u;
    const uint m = x & 7u;
    float v;
    if (e == 0u) v = float(m) * (1.0f / 512.0f);
    else if (e == 15u && m == 7u) v = as_type<float>(0x7FC00000u);
    else v = as_type<float>(((e + 120u) << 23) | (m << 20));
    return (x & 0x80u) != 0u ? -v : v;
}

// FP8 E5M2 is the upper byte of an IEEE binary16 value
Q_FUNC float q_e5m2(uint x) { return float(as_type<half>(ushort((x & 0xFFu) << 8))); }

// 6-bit scale and min of sub-block j (0..7) of Q4_K / Q5_K; `s` points at the 12 packed bytes
Q_FUNC void q_scale_min_k4(q_bytes s, uint j, thread float &sc, thread float &mn) {
    if (j < 4u) {
        sc = float(uint(s[j]) & 63u);
        mn = float(uint(s[j + 4u]) & 63u);
    } else {
        sc = float((uint(s[j + 4u]) & 15u) | ((uint(s[j - 4u]) >> 6) << 4));
        mn = float((uint(s[j + 4u]) >> 4) | ((uint(s[j]) >> 6) << 4));
    }
}

// ---------------------------------------------------------------------------------------------------------------------------
// Chunk decoders. A "chunk" is 8 consecutive values of a row. dec8(w, g, c, p1, o) writes the 8 weights of chunk `c`
// (0 <= c < CPB = QK / 8) of the block with the global index `g` (= row * blocks_per_row + block) to o[0..8].
// `w` is the start of the weight buffer, `p1` the byte offset of plane 1 for the repacked formats (MXFP4, Q4Z*, FP8M scales).
// row_scale() is a per-row factor applied after the dot product (FP8M); 1 for all other formats.
// ---------------------------------------------------------------------------------------------------------------------------
// ROW_AWARE is false for every format whose chunk decoder needs only the global block index `g`; a format whose chunk depends on its weight row
// and on the parameter block (FP8B: the tile scale) sets it and provides `dec8p` instead of `dec8` (see q_dec8 after q_params).
#define Q_NO_ROW_SCALE                                                                                                    \
    static constant constexpr bool ROW_AWARE = false;                                                                                      \
    static float row_scale(q_bytes, ulong, uint) { return 1.0f; }

// Q4_0: { half d; uchar qs[16]; }  value j<16: low nibble of qs[j], j>=16: high nibble of qs[j-16]; weight = d * (q - 8)
struct q_fmt_q4_0 {
    Q_CONST uint QK = 32;
    Q_CONST uint BYTES = 18;
    Q_CONST uint CPB = 4;
    Q_NO_ROW_SCALE
    static void dec8(q_bytes w, ulong g, uint c, ulong, thread float *o) {
        q_bytes b = w + g * BYTES;
        const float d = q_f16(b, 0);
        const uint sh = (c >> 1) << 2;
        const uint qo = 2u + ((c & 1u) << 3);
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            const uint u = q_u16(b, qo + 2u * i) >> sh;
            o[2 * i] = d * (float(u & 15u) - 8.0f);
            o[2 * i + 1] = d * (float((u >> 8) & 15u) - 8.0f);
        }
    }
};

// Q4_1: { half d, m; uchar qs[16]; }  weight = d * q + m
struct q_fmt_q4_1 {
    Q_CONST uint QK = 32;
    Q_CONST uint BYTES = 20;
    Q_CONST uint CPB = 4;
    Q_NO_ROW_SCALE
    static void dec8(q_bytes w, ulong g, uint c, ulong, thread float *o) {
        q_bytes b = w + g * BYTES;
        const float d = q_f16(b, 0);
        const float m = q_f16(b, 2);
        const uint sh = (c >> 1) << 2;
        const uint qo = 4u + ((c & 1u) << 3);
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 2; ++i) {
            const uint u = q_u32(b, qo + 4u * i) >> sh;
            #pragma clang loop unroll(full)
            for (uint k = 0; k < 4; ++k) o[4 * i + k] = d * float((u >> (8 * k)) & 15u) + m;
        }
    }
};

// Q5_0: { half d; uchar qh[4]; uchar qs[16]; }  q = nibble | bit j of qh << 4; weight = d * (q - 16)
struct q_fmt_q5_0 {
    Q_CONST uint QK = 32;
    Q_CONST uint BYTES = 22;
    Q_CONST uint CPB = 4;
    Q_NO_ROW_SCALE
    static void dec8(q_bytes w, ulong g, uint c, ulong, thread float *o) {
        q_bytes b = w + g * BYTES;
        const float d = q_f16(b, 0);
        const uint qh = q_u32w(b, 2) >> (c << 3);
        const uint sh = (c >> 1) << 2;
        const uint qo = 6u + ((c & 1u) << 3);
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            const uint u = q_u16(b, qo + 2u * i) >> sh;
            const uint q0 = (u & 15u) | (((qh >> (2 * i)) & 1u) << 4);
            const uint q1 = ((u >> 8) & 15u) | (((qh >> (2 * i + 1)) & 1u) << 4);
            o[2 * i] = d * (float(q0) - 16.0f);
            o[2 * i + 1] = d * (float(q1) - 16.0f);
        }
    }
};

// Q5_1: { half d, m; uchar qh[4]; uchar qs[16]; }  weight = d * q + m
struct q_fmt_q5_1 {
    Q_CONST uint QK = 32;
    Q_CONST uint BYTES = 24;
    Q_CONST uint CPB = 4;
    Q_NO_ROW_SCALE
    static void dec8(q_bytes w, ulong g, uint c, ulong, thread float *o) {
        q_bytes b = w + g * BYTES;
        const float d = q_f16(b, 0);
        const float m = q_f16(b, 2);
        const uint qh = q_u32(b, 4) >> (c << 3);
        const uint sh = (c >> 1) << 2;
        const uint qo = 8u + ((c & 1u) << 3);
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 2; ++i) {
            const uint u = q_u32(b, qo + 4u * i) >> sh;
            #pragma clang loop unroll(full)
            for (uint k = 0; k < 4; ++k) {
                const uint q = ((u >> (8 * k)) & 15u) | (((qh >> (4 * i + k)) & 1u) << 4);
                o[4 * i + k] = d * float(q) + m;
            }
        }
    }
};

// Q8_0: { half d; char qs[32]; }  weight = d * q
struct q_fmt_q8_0 {
    Q_CONST uint QK = 32;
    Q_CONST uint BYTES = 34;
    Q_CONST uint CPB = 4;
    Q_NO_ROW_SCALE
    static void dec8(q_bytes w, ulong g, uint c, ulong, thread float *o) {
        q_bytes b = w + g * BYTES;
        const float d = q_f16(b, 0);
        const uint qo = 2u + (c << 3);
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            const uint u = q_u16(b, qo + 2u * i);
            o[2 * i] = d * q_s8(u);
            o[2 * i + 1] = d * q_s8(u >> 8);
        }
    }
};

// IQ4_NL: layout of Q4_0, weight = d * KVALUES_IQ4NL[nibble]
struct q_fmt_iq4_nl {
    Q_CONST uint QK = 32;
    Q_CONST uint BYTES = 18;
    Q_CONST uint CPB = 4;
    Q_NO_ROW_SCALE
    static void dec8(q_bytes w, ulong g, uint c, ulong, thread float *o) {
        q_bytes b = w + g * BYTES;
        const float d = q_f16(b, 0);
        const uint sh = (c >> 1) << 2;
        const uint qo = 2u + ((c & 1u) << 3);
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            const uint u = q_u16(b, qo + 2u * i) >> sh;
            o[2 * i] = d * float(q_kv_iq4nl[u & 15u]);
            o[2 * i + 1] = d * float(q_kv_iq4nl[(u >> 8) & 15u]);
        }
    }
};

// IQ4_XS: { half d; ushort scales_h; uchar scales_l[4]; uchar qs[128]; }  8 sub-blocks of 32 values, 16 bytes each (as IQ4_NL),
// 6-bit sub-block scale ls: weight = d * (ls - 32) * KVALUES_IQ4NL[nibble]
struct q_fmt_iq4_xs {
    Q_CONST uint QK = 256;
    Q_CONST uint BYTES = 136;
    Q_CONST uint CPB = 32;
    Q_NO_ROW_SCALE
    static void dec8(q_bytes w, ulong g, uint c, ulong, thread float *o) {
        q_bytes b = w + g * BYTES;
        const uint ib = c >> 2;
        const uint cc = c & 3u;
        const uint low = (uint(b[4u + (ib >> 1)]) >> ((ib & 1u) << 2)) & 15u;
        const uint ls = low | (((q_u16(b, 2) >> (2 * ib)) & 3u) << 4);
        const float dl = q_f16(b, 0) * (float(ls) - 32.0f);
        const uint sh = (cc >> 1) << 2;
        const uint qo = 8u + ib * 16u + ((cc & 1u) << 3);
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            const uint u = q_u16(b, qo + 2u * i) >> sh;
            o[2 * i] = dl * float(q_kv_iq4nl[u & 15u]);
            o[2 * i + 1] = dl * float(q_kv_iq4nl[(u >> 8) & 15u]);
        }
    }
};

// Q2_K: { uchar scales[16]; uchar qs[64]; half d, dmin; }  16 sub-blocks of 16 values: sub-block si = n*8 + sub*2 + hg,
// q = (qs[n*32 + hg*16 + l] >> 2*sub) & 3, weight = d * (scales[si] & 15) * q - dmin * (scales[si] >> 4)
struct q_fmt_q2_k {
    Q_CONST uint QK = 256;
    Q_CONST uint BYTES = 84;
    Q_CONST uint CPB = 32;
    Q_NO_ROW_SCALE
    static void dec8(q_bytes w, ulong g, uint c, ulong, thread float *o) {
        q_bytes b = w + g * BYTES;
        const uint n = c >> 4;
        const uint sub = (c >> 2) & 3u;
        const uint hg = (c >> 1) & 1u;
        const uint si = n * 8u + sub * 2u + hg;
        const uint sc = uint(b[si]);
        const float ws = q_f16(b, 80) * float(sc & 15u);
        const float wm = -q_f16(b, 82) * float(sc >> 4);
        const uint qo = 16u + n * 32u + hg * 16u + ((c & 1u) << 3);
        const uint sh = sub << 1;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 2; ++i) {
            const uint u = q_u32(b, qo + 4u * i) >> sh;
            #pragma clang loop unroll(full)
            for (uint k = 0; k < 4; ++k) o[4 * i + k] = ws * float((u >> (8 * k)) & 3u) + wm;
        }
    }
};

// Q3_K: { uchar hmask[32]; uchar qs[64]; uchar scales[12]; half d; }
// q = ((qs[outer*32 + ix] >> 2*sub) & 3) - (hmask[ix] bit (outer*4 + sub) ? 0 : 4), ix = side*16 + l,
// sub-block scale index outer*8 + sub*2 + side (6 bits packed in 12 bytes, minus 32); weight = d * scale * q
struct q_fmt_q3_k {
    Q_CONST uint QK = 256;
    Q_CONST uint BYTES = 110;
    Q_CONST uint CPB = 32;
    Q_NO_ROW_SCALE
    static void dec8(q_bytes w, ulong g, uint c, ulong, thread float *o) {
        q_bytes b = w + g * BYTES;
        const uint outer = c >> 4;
        const uint sub = (c >> 2) & 3u;
        const uint j4 = c & 3u;
        const uint si = outer * 8u + sub * 2u + (j4 >> 1);
        const uint k = si & 3u;
        const uint row = si >> 2;
        const uint a = uint(b[96u + (row & 1u) * 4u + k]);
        const uint nib = (row >> 1) != 0u ? (a >> 4) : (a & 15u);
        const uint hi2 = (uint(b[104u + k]) >> (2u * row)) & 3u;
        const float ws = q_f16(b, 108) * (float(nib | (hi2 << 4)) - 32.0f);
        const uint sh = sub << 1;
        const uint hb = outer * 4u + sub;
        const uint qo = 32u + outer * 32u + j4 * 8u;
        const uint ho = j4 * 8u;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            const uint uq = q_u16(b, qo + 2u * i) >> sh;
            const uint uh = q_u16(b, ho + 2u * i) >> hb;
            const uint v0 = (uq & 3u) | ((uh & 1u) << 2);
            const uint v1 = ((uq >> 8) & 3u) | (((uh >> 8) & 1u) << 2);
            o[2 * i] = ws * (float(v0) - 4.0f);
            o[2 * i + 1] = ws * (float(v1) - 4.0f);
        }
    }
};

// Q4_K: { half d, dmin; uchar scales[12]; uchar qs[128]; }  group g8 (32 values) uses qs bytes (g8/2)*32.., low nibbles for even g8,
// high nibbles for odd g8; weight = d * sc[g8] * q - dmin * mn[g8]
struct q_fmt_q4_k {
    Q_CONST uint QK = 256;
    Q_CONST uint BYTES = 144;
    Q_CONST uint CPB = 32;
    Q_NO_ROW_SCALE
    static void dec8(q_bytes w, ulong g, uint c, ulong, thread float *o) {
        q_bytes b = w + g * BYTES;
        const uint g8 = c >> 2;
        const uint j4 = c & 3u;
        float sc, mn;
        q_scale_min_k4(b + 4, g8, sc, mn);
        const float ws = q_f16(b, 0) * sc;
        const float wm = -q_f16(b, 2) * mn;
        const uint qo = 16u + (g8 >> 1) * 32u + j4 * 8u;
        const uint sh = (g8 & 1u) << 2;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 2; ++i) {
            const uint u = q_u32(b, qo + 4u * i) >> sh;
            #pragma clang loop unroll(full)
            for (uint k = 0; k < 4; ++k) o[4 * i + k] = ws * float((u >> (8 * k)) & 15u) + wm;
        }
    }
};

// Q5_K: { half d, dmin; uchar scales[12]; uchar qh[32]; uchar qs[128]; }  as Q4_K, the fifth bit of value (g8, j) is bit g8 of qh[j]
struct q_fmt_q5_k {
    Q_CONST uint QK = 256;
    Q_CONST uint BYTES = 176;
    Q_CONST uint CPB = 32;
    Q_NO_ROW_SCALE
    static void dec8(q_bytes w, ulong g, uint c, ulong, thread float *o) {
        q_bytes b = w + g * BYTES;
        const uint g8 = c >> 2;
        const uint j4 = c & 3u;
        float sc, mn;
        q_scale_min_k4(b + 4, g8, sc, mn);
        const float ws = q_f16(b, 0) * sc;
        const float wm = -q_f16(b, 2) * mn;
        const uint qo = 48u + (g8 >> 1) * 32u + j4 * 8u;
        const uint ho = 16u + j4 * 8u;
        const uint sh = (g8 & 1u) << 2;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 2; ++i) {
            const uint u = q_u32(b, qo + 4u * i) >> sh;
            const uint h = q_u32(b, ho + 4u * i) >> g8;
            #pragma clang loop unroll(full)
            for (uint k = 0; k < 4; ++k) {
                const uint q = ((u >> (8 * k)) & 15u) | (((h >> (8 * k)) & 1u) << 4);
                o[4 * i + k] = ws * float(q) + wm;
            }
        }
    }
};

// Q6_K: { uchar ql[128]; uchar qh[64]; char scales[16]; half d; }  half n (128 values), group grp (32 values), j in 0..31:
// low 4 bits = low (grp < 2) / high (grp >= 2) nibble of ql[n*64 + (grp&1)*32 + j], high 2 bits = bits 2*grp.. of qh[n*32 + j];
// scale index n*8 + j/16 + grp*2; weight = d * scale * (q - 32)
struct q_fmt_q6_k {
    Q_CONST uint QK = 256;
    Q_CONST uint BYTES = 210;
    Q_CONST uint CPB = 32;
    Q_NO_ROW_SCALE
    static void dec8(q_bytes w, ulong g, uint c, ulong, thread float *o) {
        q_bytes b = w + g * BYTES;
        const uint n = c >> 4;
        const uint grp = (c >> 2) & 3u;
        const uint j4 = c & 3u;
        const float ws = q_f16(b, 208) * q_s8(uint(b[192u + n * 8u + (j4 >> 1) + grp * 2u]));
        const uint lo = n * 64u + (grp & 1u) * 32u + j4 * 8u;
        const uint hi = 128u + n * 32u + j4 * 8u;
        const uint sh = (grp >> 1) << 2;
        const uint hs = grp << 1;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 4; ++i) {
            const uint ul = q_u16(b, lo + 2u * i) >> sh;
            const uint uh = q_u16(b, hi + 2u * i) >> hs;
            const uint v0 = (ul & 15u) | ((uh & 3u) << 4);
            const uint v1 = ((ul >> 8) & 15u) | (((uh >> 8) & 3u) << 4);
            o[2 * i] = ws * (float(v0) - 32.0f);
            o[2 * i + 1] = ws * (float(v1) - 32.0f);
        }
    }
};

// NVFP4: { uchar d[4]; uchar qs[32]; } blocks of 64 values = four sub-blocks of 16 with an unsigned E4M3 scale each (halved);
// sub-block s: values 0..7 are the low nibbles of qs[8s..8s+7], values 8..15 the high nibbles; weight = FP4(q) * scale.
// The per-tensor factor is applied through q_params::scale.
struct q_fmt_nvfp4 {
    Q_CONST uint QK = 64;
    Q_CONST uint BYTES = 36;
    Q_CONST uint CPB = 8;
    Q_NO_ROW_SCALE
    static void dec8(q_bytes w, ulong g, uint c, ulong, thread float *o) {
        q_bytes b = w + g * BYTES;
        const uint s = c >> 1;
        const float d = q_ue4m3_half(uint(b[s]));  // halved scale; the FP4 table is doubled
        const uint sh = (c & 1u) << 2;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 2; ++i) {
            const uint u = q_u32(b, 4u + s * 8u + 4u * i) >> sh;
            #pragma clang loop unroll(full)
            for (uint k = 0; k < 4; ++k) o[4 * i + k] = float(q_kv_fp4[(u >> (8 * k)) & 15u]) * d;
        }
    }
};

// ---- repacked formats (plane 0 at the start of the buffer, plane 1 at q_params::plane1) --------------------------------------

// MXFP4 (OCP microscaling FP4): plane 0 = 16 code bytes per block (as in the file), plane 1 = the E8M0 scale byte of every block.
// value j (0..15) = low nibble of byte j, value j+16 = high nibble; weight = FP4(q) * 2^(e - 127)
struct q_fmt_mxfp4 {
    Q_CONST uint QK = 32;
    Q_CONST uint BYTES = 16;
    Q_CONST uint CPB = 4;
    Q_NO_ROW_SCALE
    static void dec8(q_bytes w, ulong g, uint c, ulong p1, thread float *o) {
        q_bytes b = w + g * BYTES;
        const float d = q_e8m0_half(uint(w[p1 + g]));
        const uint sh = (c >> 1) << 2;
        const uint qo = (c & 1u) << 3;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 2; ++i) {
            const uint u = q_u32(b, qo + 4u * i) >> sh;
            #pragma clang loop unroll(full)
            for (uint k = 0; k < 4; ++k) o[4 * i + k] = float(q_kv_fp4[(u >> (8 * k)) & 15u]) * d;
        }
    }
};

// Q4Z (AWQ / GPTQ 4-bit): plane 0 = group/2 code bytes per group (value 2j in the low nibble of byte j, as in the file),
// plane 1 = one word per group: binary16 scale | zero point << 16; weight = (q - zero) * scale.
// Q4ZP (act-order GPTQ) uses the same blocks; the activations are permuted by the prepass kernel.
template <uint GROUP>
struct q_fmt_q4z {
    Q_CONST uint QK = GROUP;
    Q_CONST uint BYTES = GROUP / 2;
    Q_CONST uint CPB = GROUP / 8;
    Q_NO_ROW_SCALE
    static void dec8(q_bytes w, ulong g, uint c, ulong p1, thread float *o) {
        const uint u = q_u32(w + g * BYTES, c * 4u);
        const uint sz = q_u32(w + p1 + g * 4u, 0u);
        const float scale = float(as_type<half>(ushort(sz & 0xFFFFu)));
        const float zero = float((sz >> 16) & 15u);
        #pragma clang loop unroll(full)
        for (uint k = 0; k < 8; ++k) o[k] = (float((u >> (4 * k)) & 15u) - zero) * scale;
    }
};

// MLX affine group quantization (fLLM `MLXA` types): plane 0 = the codes of every group as the MLX words have them (GROUP * BITS / 8 bytes,
// a little-endian bit stream, code i at bit i * BITS), plane 1 = one word per group: scale | bias << 16 as binary16, or as bfloat16 when
// bit 63 of q_params::plane1 is set (Q_MLX_BF16; the planner sets it, the offset of plane 1 is the low 63 bits).
// weight = code * scale + bias. A chunk of 8 codes is BITS bytes at a byte boundary (8 * BITS bits), so a chunk is read byte by byte
// (the offset c * BITS is not 4-byte aligned for odd widths) and the codes are cut out of a 64 bit window.
#define Q_MLX_BF16 0x8000000000000000ul
template <uint BITS, uint GROUP>
struct q_fmt_mlx {
    Q_CONST uint QK = GROUP;
    Q_CONST uint BYTES = GROUP * BITS / 8;
    Q_CONST uint CPB = GROUP / 8;
    Q_NO_ROW_SCALE
    static void dec8(q_bytes w, ulong g, uint c, ulong p1, thread float *o) {
        const bool bf = (p1 & Q_MLX_BF16) != 0ul;
        const uint sb = q_u32(w + (p1 & ~Q_MLX_BF16) + g * 4ul, 0u);
        const uint sraw = sb & 0xFFFFu;
        const uint braw = sb >> 16;
        const float scale = bf ? as_type<float>(sraw << 16) : float(as_type<half>(ushort(sraw)));
        const float bias = bf ? as_type<float>(braw << 16) : float(as_type<half>(ushort(braw)));
        q_bytes b = w + g * BYTES + c * BITS;
        ulong v = 0ul;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < BITS; ++i) v |= ulong(b[i]) << (8u * i);
        #pragma clang loop unroll(full)
        for (uint k = 0; k < 8; ++k) o[k] = float(uint((v >> (BITS * k)) & ((1ul << BITS) - 1ul))) * scale + bias;
    }
};
// bitsandbytes NF4 / FP4 (fLLM types NF4B64, NF4B128, FP4B64, FP4B128, gguf-probe `bnb`), read as they are in the file: a block of BLOCK values is
// { float absmax; uchar codes[BLOCK / 2]; }. Code byte j holds the elements 2j (high nibble) and 2j + 1 (low nibble); weight =
// code_book[nibble] * absmax (the book entries are the bit patterns q_kv_nf4 / q_kv_bnbfp4). The blocks are 36 / 68 bytes, so every block and
// every 4-byte code word of a chunk of 8 values is 4-byte aligned.
template <uint BLOCK, bool FP4>
struct q_fmt_bnb4 {
    Q_CONST uint QK = BLOCK;
    Q_CONST uint BYTES = 4 + BLOCK / 2;
    Q_CONST uint CPB = BLOCK / 8;
    Q_NO_ROW_SCALE
    static void dec8(q_bytes w, ulong g, uint c, ulong, thread float *o) {
        q_bytes b = w + g * BYTES;
        const float d = as_type<float>(q_u32(b, 0u));
        const uint u = q_u32(b, 4u + 4u * c);
        #pragma clang loop unroll(full)
        for (uint k = 0; k < 4; ++k) {
            const uint byte = (u >> (8 * k)) & 255u;
            o[2 * k] = as_type<float>(FP4 ? q_kv_bnbfp4[byte >> 4] : q_kv_nf4[byte >> 4]) * d;
            o[2 * k + 1] = as_type<float>(FP4 ? q_kv_bnbfp4[byte & 15u] : q_kv_nf4[byte & 15u]) * d;
        }
    }
};

// LLM.int8() (fLLM type INT8R): blocks of 64 values { float scale; char codes[64]; }, weight = code * scale (the scale, SCB / 127, is the same in
// every block of a row).
struct q_fmt_int8r {
    Q_CONST uint QK = 64;
    Q_CONST uint BYTES = 68;
    Q_CONST uint CPB = 8;
    Q_NO_ROW_SCALE
    static void dec8(q_bytes w, ulong g, uint c, ulong, thread float *o) {
        q_bytes b = w + g * BYTES;
        const float d = as_type<float>(q_u32(b, 0u));
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 2; ++i) {
            const uint u = q_u32(b, 4u + 8u * c + 4u * i);
            #pragma clang loop unroll(full)
            for (uint k = 0; k < 4; ++k) o[4 * i + k] = q_s8(u >> (8 * k)) * d;
        }
    }
};

typedef q_fmt_bnb4<64, false> q_fmt_nf4b64;
typedef q_fmt_bnb4<128, false> q_fmt_nf4b128;
typedef q_fmt_bnb4<64, true> q_fmt_fp4b64;
typedef q_fmt_bnb4<128, true> q_fmt_fp4b128;

// ---- dense / float formats: "blocks" of 8 values ------------------------------------------------------------------------

struct q_fmt_f32 {
    Q_CONST uint QK = 8;
    Q_CONST uint BYTES = 32;
    Q_CONST uint CPB = 1;
    Q_NO_ROW_SCALE
    static void dec8(q_bytes w, ulong g, uint, ulong, thread float *o) {
        device const float *p = (device const float *)(w + g * BYTES);
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; ++i) o[i] = p[i];
    }
};

struct q_fmt_f16 {
    Q_CONST uint QK = 8;
    Q_CONST uint BYTES = 16;
    Q_CONST uint CPB = 1;
    Q_NO_ROW_SCALE
    static void dec8(q_bytes w, ulong g, uint, ulong, thread float *o) {
        device const half *p = (device const half *)(w + g * BYTES);
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; ++i) o[i] = float(p[i]);
    }
};

// bfloat16 is converted by hand (no `bfloat` type needed): the bits are the upper half of a float
struct q_fmt_bf16 {
    Q_CONST uint QK = 8;
    Q_CONST uint BYTES = 16;
    Q_CONST uint CPB = 1;
    Q_NO_ROW_SCALE
    static void dec8(q_bytes w, ulong g, uint, ulong, thread float *o) {
        device const ushort *p = (device const ushort *)(w + g * BYTES);
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 8; ++i) o[i] = as_type<float>(uint(p[i]) << 16);
    }
};

// FP8 E4M3 / E5M2 bytes (scale: q_params::scale)
struct q_fmt_fp8e4m3 {
    Q_CONST uint QK = 8;
    Q_CONST uint BYTES = 8;
    Q_CONST uint CPB = 1;
    Q_NO_ROW_SCALE
    static void dec8(q_bytes w, ulong g, uint, ulong, thread float *o) {
        q_bytes b = w + g * BYTES;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 2; ++i) {
            const uint u = q_u32(b, 4u * i);
            #pragma clang loop unroll(full)
            for (uint k = 0; k < 4; ++k) o[4 * i + k] = q_e4m3((u >> (8 * k)) & 255u);
        }
    }
};

struct q_fmt_fp8e5m2 {
    Q_CONST uint QK = 8;
    Q_CONST uint BYTES = 8;
    Q_CONST uint CPB = 1;
    Q_NO_ROW_SCALE
    static void dec8(q_bytes w, ulong g, uint, ulong, thread float *o) {
        q_bytes b = w + g * BYTES;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 2; ++i) {
            const uint u = q_u32(b, 4u * i);
            #pragma clang loop unroll(full)
            for (uint k = 0; k < 4; ++k) o[4 * i + k] = q_e5m2((u >> (8 * k)) & 255u);
        }
    }
};

// FP8M (fLLM): rows * cols E4M3 bytes followed by one binary16 scale per row (plane 1 = byte offset of the scales)
struct q_fmt_fp8m {
    Q_CONST uint QK = 8;
    Q_CONST uint BYTES = 8;
    Q_CONST uint CPB = 1;
    Q_CONST bool ROW_AWARE = false;
    static float row_scale(q_bytes w, ulong p1, uint row) { return float(*(device const half *)(w + p1 + ulong(row) * 2u)); }
    static void dec8(q_bytes w, ulong g, uint, ulong, thread float *o) {
        q_bytes b = w + g * BYTES;
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 2; ++i) {
            const uint u = q_u32(b, 4u * i);
            #pragma clang loop unroll(full)
            for (uint k = 0; k < 4; ++k) o[4 * i + k] = q_e4m3((u >> (8 * k)) & 255u);
        }
    }
};

typedef q_fmt_q4z<32> q_fmt_q4z32;
typedef q_fmt_q4z<64> q_fmt_q4z64;
typedef q_fmt_q4z<128> q_fmt_q4z128;
typedef q_fmt_mlx<2, 32> q_fmt_mlx2g32;
typedef q_fmt_mlx<2, 64> q_fmt_mlx2g64;
typedef q_fmt_mlx<2, 128> q_fmt_mlx2g128;
typedef q_fmt_mlx<3, 32> q_fmt_mlx3g32;
typedef q_fmt_mlx<3, 64> q_fmt_mlx3g64;
typedef q_fmt_mlx<3, 128> q_fmt_mlx3g128;
typedef q_fmt_mlx<4, 32> q_fmt_mlx4g32;
typedef q_fmt_mlx<4, 64> q_fmt_mlx4g64;
typedef q_fmt_mlx<4, 128> q_fmt_mlx4g128;
typedef q_fmt_mlx<5, 32> q_fmt_mlx5g32;
typedef q_fmt_mlx<5, 64> q_fmt_mlx5g64;
typedef q_fmt_mlx<5, 128> q_fmt_mlx5g128;
typedef q_fmt_mlx<6, 32> q_fmt_mlx6g32;
typedef q_fmt_mlx<6, 64> q_fmt_mlx6g64;
typedef q_fmt_mlx<6, 128> q_fmt_mlx6g128;
typedef q_fmt_mlx<8, 32> q_fmt_mlx8g32;
typedef q_fmt_mlx<8, 64> q_fmt_mlx8g64;
typedef q_fmt_mlx<8, 128> q_fmt_mlx8g128;

// ---- kernel interface shared by the matrix-vector and matrix-matrix kernels -------------------------------------------------------

// kernel is hidden behind a macro so that the source scanner of the backend does not take shared files for kernel files.
#define Q_KERNEL kernel

// Parameter block (buffer(4)). Must match `QParams` in src/backend/metal/weights/plan.rs byte for byte (48 bytes).
struct q_params {
    ulong plane1;      // byte offset of plane 1 in the weight buffer (repacked formats and FP8M scales), else 0
    uint rows;         // weight rows handled by this dispatch
    uint cols;         // K (multiple of 8 and of the block size)
    uint batch;        // activation rows
    uint flags;        // bit 0: add bias[row_base + r]; bit 1: add the previous contents of out (residual)
    float scale;       // per-tensor factor (NVFP4, scaled FP8); 1.0 otherwise
    uint out_stride;   // floats between the outputs of two tokens
    uint row_base;     // index of the first output feature of this dispatch
    uint x_stride;     // floats between two activation rows
    uint nb_blocks;    // blocks per row (cols / QK)
    uint pad;
};

Q_FUNC float q_finish(float acc, float rs, constant q_params &p, device const float *bias, uint row) {
    float v = acc * (p.scale * rs);
    if ((p.flags & 1u) != 0u) v += bias[p.row_base + row];
    return v;
}


// ---- row-aware formats --------------------------------------------------------------------------------------------------------------

// Block-wise FP8 (fLLM `FP8B` types, gguf-probe `Fp8bLayout`), read as it is in the file: rows * cols E4M3 codes (row-major), then one f32 scale
// per tile of 2^lbr rows x 2^lbc columns (tiles row-major, ceil(cols / 2^lbc) per row of tiles; the file's trailer behind them is not read).
// weight = e4m3(code) * scale[(row >> lbr) * tiles_per_row + (col >> lbc)], the arithmetic of cpu_fp8.rs. The tile of a chunk depends on its row, so
// the format is ROW_AWARE: `dec8p` gets the row and the parameter block. q_params: plane1 = byte offset of the scale table (= rows * cols),
// pad = lbr | lbc << 8, cols = the real number of columns (the last tile may be partial), nb_blocks = cols / 8 (a "block" is one chunk of 8 codes).
// A chunk (8 codes at row * cols + 8 * blk, a multiple of 8 bytes) lies in one tile when 2^lbc >= 8; narrower tiles take one scale per code.
struct q_fmt_fp8b {
    Q_CONST uint QK = 8;
    Q_CONST uint BYTES = 8;
    Q_CONST uint CPB = 1;
    Q_CONST bool ROW_AWARE = true;
    static float row_scale(q_bytes, ulong, uint) { return 1.0f; }
    static void dec8p(q_bytes w, uint row, uint blk, uint, constant q_params &p, thread float *o) {
        const uint lbr = p.pad & 0xFFu;
        const uint lbc = (p.pad >> 8) & 0xFFu;
        const uint col = blk * 8u;
        const uint tiles_per_row = (p.cols + ((1u << lbc) - 1u)) >> lbc;
        device const float *scales = (device const float *)(w + p.plane1) + ulong(row >> lbr) * ulong(tiles_per_row);
        q_bytes b = w + ulong(row) * ulong(p.cols) + ulong(col);
        #pragma clang loop unroll(full)
        for (uint i = 0; i < 2; ++i) {
            const uint u = q_u32(b, 4u * i);
            #pragma clang loop unroll(full)
            for (uint k = 0; k < 4; ++k) {
                const uint at = 4u * i + k;
                const float s = lbc >= 3u ? scales[col >> lbc] : scales[(col + at) >> lbc];
                o[at] = q_e4m3((u >> (8 * k)) & 255u) * s;
            }
        }
    }
};

// The chunk decoder as the kernel bodies call it: the 8 weights of chunk `c` of block `blk` of weight row `row`. Formats that only need the global
// block index (all but ROW_AWARE ones) get g = row * nb_blocks + blk.
template <typename F, bool ROW_AWARE>
struct q_dec_select {
    static Q_FUNC void run(q_bytes w, uint row, uint blk, uint c, constant q_params &p, thread float *o) {
        F::dec8(w, ulong(row) * ulong(p.nb_blocks) + ulong(blk), c, p.plane1, o);
    }
};

template <typename F>
struct q_dec_select<F, true> {
    static Q_FUNC void run(q_bytes w, uint row, uint blk, uint c, constant q_params &p, thread float *o) {
        F::dec8p(w, row, blk, c, p, o);
    }
};

template <typename F>
Q_FUNC void q_dec8(q_bytes w, uint row, uint blk, uint c, constant q_params &p, thread float *o) {
    q_dec_select<F, F::ROW_AWARE>::run(w, row, blk, c, p, o);
}
