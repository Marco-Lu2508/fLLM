// fLLM CUDA backend: device decoders of the GGML K-quant and IQ formats that have no dedicated kernel (Q2_K, Q3_K, IQ1_S,
// IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S). Each decodes 16 consecutive values of a row (`decode16` in common.cuh),
// which serves the generic GEMV (linear.cu), the tensor-core GEMM (tc_gemm.cu) and the embedding lookup.
//
// The block layouts and dequantization equations are the ones of the Rust reference decoders in rust/engine/src/quant_iq.rs and
// quant_extra.rs (adapted from the ggml authors' MIT-licensed llama.cpp implementation, see THIRD_PARTY.md); selftest-gpu
// compares every format with them. Blocks are 2-byte aligned only, so all reads go through ld16 / byte loads.
#pragma once
#include "iq_tables.cuh"

__device__ __forceinline__ uint32_t iq_ld32(const uint8_t *p) { return (uint32_t)ld16(p) | ((uint32_t)ld16(p + 2) << 16); }

// eight values of a grid entry (bytes of `grid`, unsigned) with the signs of bits 0..7 of `signs` applied
__device__ __forceinline__ void iq_put8(float *w, uint64_t grid, uint32_t signs, float scale) {
#pragma unroll
    for (int j = 0; j < 8; j++) {
        const float v = (float)((grid >> (8 * j)) & 0xffu);
        w[j] = scale * ((signs >> j) & 1u ? -v : v);
    }
}
// same for 3-bit grids: two entries of four bytes
__device__ __forceinline__ void iq_put8_3(float *w, uint32_t g0, uint32_t g1, uint32_t signs, float scale) {
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const float a = (float)((g0 >> (8 * j)) & 0xffu), b = (float)((g1 >> (8 * j)) & 0xffu);
        w[j] = scale * ((signs >> j) & 1u ? -a : a);
        w[4 + j] = scale * ((signs >> (4 + j)) & 1u ? -b : b);
    }
}
// IQ1: signed grid bytes plus a shared delta
__device__ __forceinline__ void iq_put8_1(float *w, uint64_t grid, float delta, float scale) {
#pragma unroll
    for (int j = 0; j < 8; j++) w[j] = scale * ((float)(int8_t)((grid >> (8 * j)) & 0xffu) + delta);
}

template <uint32_t KIND>
__device__ __forceinline__ void decode16_iq(const uint8_t *rp, int run, float *w) {
    const uint8_t *b = rp + (size_t)(run >> 4) * kind_iq_block_bytes(KIND);
    const int local = (run & 15) * 16;  // first value inside the 256-value block
    if (KIND == KIND_Q2_K) {
        const float d = h2f_bits(ld16(b + 80)), dmin = h2f_bits(ld16(b + 82));
        const int n = local >> 7, s = (local & 127) >> 5, hg = (local >> 4) & 1;
        const uint8_t sc = b[n * 8 + s * 2 + hg];
        const float ds = d * (float)(sc & 15), dm = dmin * (float)(sc >> 4);
        const uint8_t *q = b + 16 + n * 32 + hg * 16;
#pragma unroll
        for (int l = 0; l < 16; l++) w[l] = ds * (float)((q[l] >> (2 * s)) & 3) - dm;
        return;
    }
    if (KIND == KIND_Q3_K) {
        const float d = h2f_bits(ld16(b + 108));
        const int n = local >> 7, sub = (local & 127) >> 5, side = (local >> 4) & 1;
        const int idx = n * 8 + sub * 2 + side, i = idx & 3, quad = idx >> 2;
        const uint32_t hi = b[96 + i], lo = b[100 + i], packed = b[104 + i];
        const uint32_t nib = (quad & 1 ? lo : hi) >> (quad >= 2 ? 4 : 0);
        const int scale = (int)((nib & 15u) | (((packed >> (2 * quad)) & 3u) << 4)) - 32;
        const float ds = d * (float)scale;
        const uint8_t *q = b + 32 + n * 32 + side * 16, *hm = b + side * 16;
        const uint32_t mask = 1u << (n * 4 + sub);
#pragma unroll
        for (int l = 0; l < 16; l++) {
            const int low = (q[l] >> (2 * sub)) & 3, high = (hm[l] & mask) ? 0 : 4;
            w[l] = ds * (float)(low - high);
        }
        return;
    }
    const float d = KIND == KIND_IQ1_M ? 0.f : h2f_bits(ld16(b));
    const int ib32 = local >> 5, l0 = (local >> 3) & 3;  // 32-value group; first of the two 8-value entries of this run
    if (KIND == KIND_IQ2_XXS) {
        const uint32_t lo = iq_ld32(b + 2 + ib32 * 8), hi = iq_ld32(b + 2 + ib32 * 8 + 4);
        const float db = d * (0.5f + (float)(hi >> 28)) * 0.25f;
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const int l = l0 + e;
            iq_put8(w + 8 * e, kIq2xxsGrid[(lo >> (8 * l)) & 0xffu], kIqSigns[(hi >> (7 * l)) & 0x7fu], db);
        }
        return;
    }
    if (KIND == KIND_IQ2_XS) {
        const uint8_t sc = b[66 + ib32];
        const float db = d * (0.5f + (float)((l0 >= 2 ? sc >> 4 : sc & 15))) * 0.25f;
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const uint32_t q = ld16(b + 2 + ib32 * 8 + (l0 + e) * 2);
            iq_put8(w + 8 * e, kIq2xsGrid[q & 0x1ffu], kIqSigns[q >> 9], db);
        }
        return;
    }
    if (KIND == KIND_IQ2_S) {
        const uint8_t sc = b[74 + ib32];
        const float db = d * (0.5f + (float)((l0 >= 2 ? sc >> 4 : sc & 15))) * 0.25f;
        const uint32_t qh = b[66 + ib32];
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const int l = l0 + e;
            const uint32_t index = b[2 + ib32 * 4 + l] | (((qh >> (2 * l)) & 3u) << 8);
            iq_put8(w + 8 * e, kIq2sGrid[index], b[34 + ib32 * 4 + l], db);
        }
        return;
    }
    if (KIND == KIND_IQ3_XXS) {
        const uint32_t ss = iq_ld32(b + 66 + ib32 * 4);
        const float db = d * (0.5f + (float)(ss >> 28)) * 0.5f;
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const int l = l0 + e;
            iq_put8_3(w + 8 * e, kIq3xxsGrid[b[2 + ib32 * 8 + l * 2]], kIq3xxsGrid[b[2 + ib32 * 8 + l * 2 + 1]],
                      kIqSigns[(ss >> (7 * l)) & 0x7fu], db);
        }
        return;
    }
    if (KIND == KIND_IQ3_S) {
        const int pair = ib32 >> 1;
        const uint32_t s4 = b[106 + pair];
        const float scale = d * (float)(1 + 2 * (int)((ib32 & 1) ? (s4 >> 4) : (s4 & 15u)));
        const uint32_t qh = b[66 + ib32];
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const int l = l0 + e;
            const uint32_t q0 = b[2 + ib32 * 8 + l * 2] | (((qh >> (2 * l)) & 1u) << 8);
            const uint32_t q1 = b[2 + ib32 * 8 + l * 2 + 1] | (((qh >> (2 * l + 1)) & 1u) << 8);
            iq_put8_3(w + 8 * e, kIq3sGrid[q0], kIq3sGrid[q1], b[74 + ib32 * 4 + l], scale);
        }
        return;
    }
    if (KIND == KIND_IQ1_S) {
        const uint32_t qh = ld16(b + 34 + ib32 * 2);
        const float dl = d * (float)(2 * ((qh >> 12) & 7u) + 1);
        const float delta = (qh & 0x8000u) ? -0.125f : 0.125f;
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const int l = l0 + e;
            iq_put8_1(w + 8 * e, kIq1sGrid[b[2 + ib32 * 4 + l] | (((qh >> (3 * l)) & 7u) << 8)], delta, dl);
        }
        return;
    }
    if (KIND == KIND_IQ1_M) {
        const uint32_t sc0 = ld16(b + 48), sc1 = ld16(b + 50), sc2 = ld16(b + 52), sc3 = ld16(b + 54);
        const uint32_t bits = (sc0 >> 12) | ((sc1 >> 8) & 0x00f0u) | ((sc2 >> 4) & 0x0f00u) | (sc3 & 0xf000u);
        const float dm = h2f_bits((uint16_t)bits);
        const uint32_t sc = ld16(b + 48 + (ib32 >> 1) * 2);
        const int shift = 6 * (ib32 & 1);
        const float dl1 = dm * (float)(2 * ((sc >> shift) & 7u) + 1), dl2 = dm * (float)(2 * ((sc >> (shift + 3)) & 7u) + 1);
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const int l = l0 + e;
            const uint32_t qh = b[32 + ib32 * 2 + (l >> 1)];
            const uint32_t high = (l & 1) ? ((qh >> 4) & 7u) : (qh & 7u);
            const float delta = (qh & ((l & 1) ? 0x80u : 0x08u)) ? -0.125f : 0.125f;
            iq_put8_1(w + 8 * e, kIq1sGrid[b[ib32 * 4 + l] | (high << 8)], delta, l < 2 ? dl1 : dl2);
        }
        return;
    }
}
