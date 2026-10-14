// SPDX-License-Identifier: Apache-2.0
// fLLM Intel GPU backend (SYCL / oneAPI): device-side decoders of the weight formats.
//
// Every format F describes a "unit": the piece of a block one work-item decodes at a time.
//   F::BLK  values per block (the float formats use blocks of 8 values)
//   F::UPB  units per block
//   F::V    values per unit
//   F::TAIL rows may end inside a block (float formats: padded with zeros; reads of x must be guarded)
//   F::pos(u, i)  position (0 .. BLK-1) inside the block of the i-th value of unit u
//   F::decode(row, nbp, blk, u, w)  w[0..V) = the dequantized values (f32) of unit u of block blk
// `row` points at the start of a row in the device layout written by weights.cpp, `nbp` is the number of (padded) blocks per row:
// the layouts keep the scale arrays and the quant arrays of a row separate and 16-byte aligned (legacy formats, Q8_0, IQ4_NL,
// MXFP4) or pad the super-blocks to a multiple of 16 bytes (Q6_K 224, Q2_K 96, Q3_K 112), so every load below is naturally
// aligned. Decoding follows the CPU reference (compute.rs `dequantize`, quant_extra.rs) value by value.
// Verification level: L0.
#pragma once

#include <sycl/sycl.hpp>

#include <cstdint>

namespace fllm {

inline float h2f(uint16_t b) { return static_cast<float>(sycl::bit_cast<sycl::half>(b)); }
inline uint16_t f2h(float v) { return sycl::bit_cast<uint16_t>(sycl::half(v)); }
inline float bf2f(uint16_t b) { return sycl::bit_cast<float>(static_cast<uint32_t>(b) << 16); }
inline uint64_t ld64(const uint8_t* p) { return *reinterpret_cast<const uint64_t*>(p); }
inline uint32_t ld32(const uint8_t* p) { return *reinterpret_cast<const uint32_t*>(p); }
inline uint16_t ld16(const uint8_t* p) { return *reinterpret_cast<const uint16_t*>(p); }
inline int byte8(uint64_t v, int i) { return static_cast<int>((v >> (8 * i)) & 0xffu); }
inline int sbyte8(uint64_t v, int i) { return static_cast<int>(static_cast<int8_t>((v >> (8 * i)) & 0xffu)); }

// Scale and minimum of sub-block g (0..7) of a Q4_K / Q5_K super-block (12 packed bytes).
inline void scale_min_k4(const uint8_t* s, int g, float& sc, float& mn) {
    if (g < 4) {
        sc = static_cast<float>(s[g] & 63);
        mn = static_cast<float>(s[g + 4] & 63);
    } else {
        sc = static_cast<float>((s[g + 4] & 15) | ((s[g - 4] >> 6) << 4));
        mn = static_cast<float>((s[g + 4] >> 4) | ((s[g] >> 6) << 4));
    }
}

// Signed 6-bit scale idx (0..15) of a Q3_K super-block (12 packed bytes).
inline int q3k_scale(const uint8_t* s, int idx) {
    const int i = idx & 3, grp = idx >> 2;
    const int packed = s[8 + i];
    int v;
    if (grp == 0) v = (s[i] & 15) | ((packed & 3) << 4);
    else if (grp == 1) v = (s[4 + i] & 15) | (((packed >> 2) & 3) << 4);
    else if (grp == 2) v = (s[i] >> 4) | (((packed >> 4) & 3) << 4);
    else v = (s[4 + i] >> 4) | (((packed >> 6) & 3) << 4);
    return v - 32;
}

// 16-entry value grids: IQ4_NL (non-linear) and MXFP4 (E2M1 times two, the block scale carries the rest).
constexpr int8_t kIq4Grid[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
constexpr int8_t kMxfp4Grid[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

// ---- float formats ----------------------------------------------------------------------------------------------------
struct FF32 {
    static constexpr int KIND = 0, BLK = 8, UPB = 1, V = 8;
    static constexpr bool TAIL = true;
    static constexpr int pos(int, int i) { return i; }
    static inline void decode(const uint8_t* row, int, int blk, int, float* w) {
        const float* p = reinterpret_cast<const float*>(row) + static_cast<size_t>(blk) * 8;
#pragma unroll
        for (int i = 0; i < 8; i++) w[i] = p[i];
    }
};

struct FF16 {
    static constexpr int KIND = 1, BLK = 8, UPB = 1, V = 8;
    static constexpr bool TAIL = true;
    static constexpr int pos(int, int i) { return i; }
    static inline void decode(const uint8_t* row, int, int blk, int, float* w) {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(row) + static_cast<size_t>(blk) * 8;
#pragma unroll
        for (int i = 0; i < 8; i++) w[i] = h2f(p[i]);
    }
};

struct FBF16 {
    static constexpr int KIND = 30, BLK = 8, UPB = 1, V = 8;
    static constexpr bool TAIL = true;
    static constexpr int pos(int, int i) { return i; }
    static inline void decode(const uint8_t* row, int, int blk, int, float* w) {
        const uint16_t* p = reinterpret_cast<const uint16_t*>(row) + static_cast<size_t>(blk) * 8;
#pragma unroll
        for (int i = 0; i < 8; i++) w[i] = bf2f(p[i]);
    }
};

// ---- Q8_0: [d: f16 x nbp][quants: i8 x 32 x nbp] -----------------------------------------------------------------------
struct FQ8_0 {
    static constexpr int KIND = 8, BLK = 32, UPB = 2, V = 16;
    static constexpr bool TAIL = false;
    static constexpr int pos(int u, int i) { return u * 16 + i; }
    static inline void decode(const uint8_t* row, int nbp, int blk, int u, float* w) {
        const float d = h2f(ld16(row + static_cast<size_t>(blk) * 2));
        const uint8_t* q = row + static_cast<size_t>(nbp) * 2 + static_cast<size_t>(blk) * 32 + u * 16;
        const uint64_t a = ld64(q), b = ld64(q + 8);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            w[i] = d * static_cast<float>(sbyte8(a, i));
            w[8 + i] = d * static_cast<float>(sbyte8(b, i));
        }
    }
};

// ---- Q4_0 / Q4_1 / Q5_0 / Q5_1: separate arrays [d][m][qh][qs], 16 quant bytes per block ----------------------------------
// value j < 16 is the low nibble of qs[j], value j >= 16 the high nibble of qs[j - 16]; the fifth bit of Q5 is bit j of qh.
template <int KIND_, bool AFFINE, bool FIVE>
struct FLegacy {
    static constexpr int KIND = KIND_, BLK = 32, UPB = 1, V = 32;
    static constexpr bool TAIL = false;
    static constexpr int pos(int, int i) { return i; }
    static inline void decode(const uint8_t* row, int nbp, int blk, int, float* w) {
        const size_t n = static_cast<size_t>(nbp);
        const float d = h2f(ld16(row + static_cast<size_t>(blk) * 2));
        float m = 0.f;
        if constexpr (AFFINE) m = h2f(ld16(row + n * 2 + static_cast<size_t>(blk) * 2));
        const size_t qh_off = n * 2 * (AFFINE ? 2 : 1);
        uint32_t qh = 0;
        if constexpr (FIVE) qh = ld32(row + qh_off + static_cast<size_t>(blk) * 4);
        const size_t qs_off = qh_off + (FIVE ? n * 4 : 0);
        const uint8_t* qs = row + qs_off + static_cast<size_t>(blk) * 16;
        const uint64_t a = ld64(qs), b = ld64(qs + 8);
#pragma unroll
        for (int i = 0; i < 16; i++) {
            const int byte = i < 8 ? byte8(a, i) : byte8(b, i - 8);
            int lo = byte & 15, hi = byte >> 4;
            if constexpr (FIVE) {
                lo |= static_cast<int>((qh >> i) & 1u) << 4;
                hi |= static_cast<int>((qh >> (i + 16)) & 1u) << 4;
            }
            if constexpr (AFFINE) {
                w[i] = d * static_cast<float>(lo) + m;
                w[i + 16] = d * static_cast<float>(hi) + m;
            } else {
                const int z = FIVE ? 16 : 8;
                w[i] = d * static_cast<float>(lo - z);
                w[i + 16] = d * static_cast<float>(hi - z);
            }
        }
    }
};
using FQ4_0 = FLegacy<2, false, false>;
using FQ4_1 = FLegacy<3, true, false>;
using FQ5_0 = FLegacy<6, false, true>;
using FQ5_1 = FLegacy<7, true, true>;

// ---- IQ4_NL / MXFP4: [d: f16 x nbp][qs: 16 bytes x nbp], value = d * grid[nibble] ---------------------------------------------
template <int KIND_, bool MX>
struct FGrid {
    static constexpr int KIND = KIND_, BLK = 32, UPB = 1, V = 32;
    static constexpr bool TAIL = false;
    static constexpr int pos(int, int i) { return i; }
    static inline void decode(const uint8_t* row, int nbp, int blk, int, float* w) {
        const float d = h2f(ld16(row + static_cast<size_t>(blk) * 2));
        const uint8_t* qs = row + static_cast<size_t>(nbp) * 2 + static_cast<size_t>(blk) * 16;
        const uint64_t a = ld64(qs), b = ld64(qs + 8);
#pragma unroll
        for (int i = 0; i < 16; i++) {
            const int byte = i < 8 ? byte8(a, i) : byte8(b, i - 8);
            const int lo = MX ? kMxfp4Grid[byte & 15] : kIq4Grid[byte & 15];
            const int hi = MX ? kMxfp4Grid[byte >> 4] : kIq4Grid[byte >> 4];
            w[i] = d * static_cast<float>(lo);
            w[i + 16] = d * static_cast<float>(hi);
        }
    }
};
using FIQ4_NL = FGrid<20, false>;
using FMXFP4 = FGrid<39, true>;

// ---- Q4_K: super-block 144 bytes {d, dmin, scales[12], qs[128]} ---------------------------------------------------------------
// unit u (0..15): g = u / 4 picks 32 quant bytes, w0 = (u % 4) * 8 the byte within; the low nibbles are values g*64 + w0 + i with
// sub-block 2g, the high nibbles values g*64 + 32 + w0 + i with sub-block 2g + 1.
struct FQ4_K {
    static constexpr int KIND = 12, BLK = 256, UPB = 16, V = 16;
    static constexpr bool TAIL = false;
    static constexpr int pos(int u, int i) { return (u >> 2) * 64 + (i < 8 ? 0 : 32) + (u & 3) * 8 + (i & 7); }
    static inline void decode(const uint8_t* row, int, int blk, int u, float* w) {
        const uint8_t* sb = row + static_cast<size_t>(blk) * 144;
        const float d = h2f(ld16(sb)), dmin = h2f(ld16(sb + 2));
        const int g = u >> 2, w0 = (u & 3) * 8;
        const uint64_t q = ld64(sb + 16 + g * 32 + w0);
        float s0, m0, s1, m1;
        scale_min_k4(sb + 4, 2 * g, s0, m0);
        scale_min_k4(sb + 4, 2 * g + 1, s1, m1);
        const float d0 = d * s0, dm0 = dmin * m0, d1 = d * s1, dm1 = dmin * m1;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int byte = byte8(q, i);
            w[i] = d0 * static_cast<float>(byte & 15) - dm0;
            w[8 + i] = d1 * static_cast<float>(byte >> 4) - dm1;
        }
    }
};

// ---- Q5_K: 176 bytes {d, dmin, scales[12], qh[32], qs[128]}; the fifth bit of value (group, j) is bit `group` of qh[j] ----------
struct FQ5_K {
    static constexpr int KIND = 13, BLK = 256, UPB = 16, V = 16;
    static constexpr bool TAIL = false;
    static constexpr int pos(int u, int i) { return FQ4_K::pos(u, i); }
    static inline void decode(const uint8_t* row, int, int blk, int u, float* w) {
        const uint8_t* sb = row + static_cast<size_t>(blk) * 176;
        const float d = h2f(ld16(sb)), dmin = h2f(ld16(sb + 2));
        const int g = u >> 2, w0 = (u & 3) * 8;
        const uint64_t q = ld64(sb + 48 + g * 32 + w0);
        const uint64_t h = ld64(sb + 16 + w0);
        float s0, m0, s1, m1;
        scale_min_k4(sb + 4, 2 * g, s0, m0);
        scale_min_k4(sb + 4, 2 * g + 1, s1, m1);
        const float d0 = d * s0, dm0 = dmin * m0, d1 = d * s1, dm1 = dmin * m1;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int byte = byte8(q, i), hb = byte8(h, i);
            const int lo = (byte & 15) | (((hb >> (2 * g)) & 1) << 4);
            const int hi = (byte >> 4) | (((hb >> (2 * g + 1)) & 1) << 4);
            w[i] = d0 * static_cast<float>(lo) - dm0;
            w[8 + i] = d1 * static_cast<float>(hi) - dm1;
        }
    }
};

// ---- Q6_K: 210 bytes padded to 224 {ql[128], qh[64], scales[16] (i8), d} -----------------------------------------------------
// unit u (0..7): n = u / 4 is the half of the super-block, j0 = (u % 4) * 8; four groups of 8 values: n*128 + grp*32 + j0 + i.
struct FQ6_K {
    static constexpr int KIND = 14, BLK = 256, UPB = 8, V = 32;
    static constexpr bool TAIL = false;
    static constexpr int pos(int u, int i) { return (u >> 2) * 128 + (i >> 3) * 32 + (u & 3) * 8 + (i & 7); }
    static inline void decode(const uint8_t* row, int, int blk, int u, float* w) {
        const uint8_t* sb = row + static_cast<size_t>(blk) * 224;
        const float d = h2f(ld16(sb + 208));
        const int n = u >> 2, j0 = (u & 3) * 8;
        const uint64_t l0 = ld64(sb + n * 64 + j0), l1 = ld64(sb + n * 64 + 32 + j0), h = ld64(sb + 128 + n * 32 + j0);
        const int8_t* sc = reinterpret_cast<const int8_t*>(sb + 192 + n * 8) + (j0 >> 4);
        const float s0 = d * static_cast<float>(sc[0]), s1 = d * static_cast<float>(sc[2]), s2 = d * static_cast<float>(sc[4]),
                    s3 = d * static_cast<float>(sc[6]);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int b0 = byte8(l0, i), b1 = byte8(l1, i), hb = byte8(h, i);
            const int v0 = (b0 & 15) | ((hb & 3) << 4);
            const int v1 = (b1 & 15) | (((hb >> 2) & 3) << 4);
            const int v2 = (b0 >> 4) | (((hb >> 4) & 3) << 4);
            const int v3 = (b1 >> 4) | (((hb >> 6) & 3) << 4);
            w[i] = s0 * static_cast<float>(v0 - 32);
            w[8 + i] = s1 * static_cast<float>(v1 - 32);
            w[16 + i] = s2 * static_cast<float>(v2 - 32);
            w[24 + i] = s3 * static_cast<float>(v3 - 32);
        }
    }
};

// ---- Q2_K: 84 bytes padded to 96 {scales[16], qs[64], d, dmin} ----------------------------------------------------------------
// unit u (0..7): n = u / 4, j0 = (u % 4) * 8; four 2-bit planes s = 0..3 give the values n*128 + s*32 + j0 + i, scale byte
// n*8 + s*2 + j0/16 (low nibble: scale, high nibble: minimum).
struct FQ2_K {
    static constexpr int KIND = 10, BLK = 256, UPB = 8, V = 32;
    static constexpr bool TAIL = false;
    static constexpr int pos(int u, int i) { return (u >> 2) * 128 + (i >> 3) * 32 + (u & 3) * 8 + (i & 7); }
    static inline void decode(const uint8_t* row, int, int blk, int u, float* w) {
        const uint8_t* sb = row + static_cast<size_t>(blk) * 96;
        const float d = h2f(ld16(sb + 80)), dmin = h2f(ld16(sb + 82));
        const int n = u >> 2, j0 = (u & 3) * 8;
        const uint64_t q = ld64(sb + 16 + n * 32 + j0);
#pragma unroll
        for (int s = 0; s < 4; s++) {
            const int sc = sb[n * 8 + s * 2 + (j0 >> 4)];
            const float ds = d * static_cast<float>(sc & 15), dm = dmin * static_cast<float>(sc >> 4);
#pragma unroll
            for (int i = 0; i < 8; i++) w[s * 8 + i] = ds * static_cast<float>((byte8(q, i) >> (2 * s)) & 3) - dm;
        }
    }
};

// ---- Q3_K: 110 bytes padded to 112 {hmask[32], qs[64], scales[12], d} ---------------------------------------------------------
// same unit structure as Q2_K; the high bit of a value is bit (n*4 + s) of hmask[j] (set = no offset), value = d * scale * (low - 4 * !high).
struct FQ3_K {
    static constexpr int KIND = 11, BLK = 256, UPB = 8, V = 32;
    static constexpr bool TAIL = false;
    static constexpr int pos(int u, int i) { return (u >> 2) * 128 + (i >> 3) * 32 + (u & 3) * 8 + (i & 7); }
    static inline void decode(const uint8_t* row, int, int blk, int u, float* w) {
        const uint8_t* sb = row + static_cast<size_t>(blk) * 112;
        const float d = h2f(ld16(sb + 108));
        const int n = u >> 2, j0 = (u & 3) * 8;
        const uint64_t q = ld64(sb + 32 + n * 32 + j0);
        const uint64_t hm = ld64(sb + j0);
#pragma unroll
        for (int s = 0; s < 4; s++) {
            const float ds = d * static_cast<float>(q3k_scale(sb + 96, n * 8 + s * 2 + (j0 >> 4)));
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const int low = (byte8(q, i) >> (2 * s)) & 3;
                const int high = ((byte8(hm, i) >> (n * 4 + s)) & 1) ? 0 : 4;
                w[s * 8 + i] = ds * static_cast<float>(low - high);
            }
        }
    }
};

// The kinds of all formats, for the dispatch tables: X(kind, Format)
#define FLLM_SYCL_FORMATS(X)                                                                                                    \
    X(0, FF32) X(1, FF16) X(30, FBF16) X(2, FQ4_0) X(3, FQ4_1) X(6, FQ5_0) X(7, FQ5_1) X(8, FQ8_0) X(10, FQ2_K) X(11, FQ3_K) \
    X(12, FQ4_K) X(13, FQ5_K) X(14, FQ6_K) X(20, FIQ4_NL) X(39, FMXFP4)

}  // namespace fllm
