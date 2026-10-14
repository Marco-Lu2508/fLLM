// fLLM CUDA backend: shared device helpers.
#pragma once
#if defined(__HIPCC__)
#include "hip_compat.h"
#else
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif
#include <stdint.h>
#include <stdio.h>

#define FLLM_WARP 32

// The tensor-core kernels use mma.m16n8k16, ldmatrix and cp.async (compute capability 8.0 on). Turing (7.5) has ldmatrix and
// m16n8k8 only: mma_util.cuh emulates m16n8k16 with two m16n8k8 and cp.async with plain shared-memory stores (the pattern
// of vLLM's Marlin for sm_75). Older GPUs get these kernels emptied; the dispatcher never launches them there.
#if defined(__HIPCC__) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 750))
#define FLLM_TC 0
#else
#define FLLM_TC 1
#endif

__device__ __forceinline__ float h2f_bits(uint16_t b) { return __half2float(__ushort_as_half(b)); }
__device__ __forceinline__ uint16_t f2h_bits(float x) { return __half_as_ushort(__float2half_rn(x)); }

// Quantized blocks start at multiples of their size (18, 22, 34, 110, 210 ... bytes), so they are
// only guaranteed 2-byte aligned.
__device__ __forceinline__ uint16_t ld16(const uint8_t *p) { return *reinterpret_cast<const uint16_t *>(p); }
__device__ __forceinline__ uint32_t ld32_a2(const uint8_t *p) {
    return (uint32_t)ld16(p) | ((uint32_t)ld16(p + 2) << 16);
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = FLLM_WARP / 2; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = FLLM_WARP / 2; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// GGUF tensor type ids (ggml).
enum FllmKind {
    KIND_F32 = 0, KIND_F16 = 1, KIND_Q4_0 = 2, KIND_Q4_1 = 3, KIND_Q5_0 = 6, KIND_Q5_1 = 7,
    KIND_Q8_0 = 8, KIND_IQ4_NL = 20, KIND_MXFP4 = 39, KIND_Q4_K = 12, KIND_Q4_K_MARLIN = 41, KIND_Q5_K = 13, KIND_Q6_K = 14, KIND_BF16 = 30,
    // 256-value formats decoded by iq_decode.cuh
    KIND_Q2_K = 10, KIND_Q3_K = 11, KIND_IQ2_XXS = 16, KIND_IQ2_XS = 17, KIND_IQ3_XXS = 18, KIND_IQ1_S = 19,
    KIND_IQ3_S = 21, KIND_IQ2_S = 22, KIND_IQ1_M = 29
};

__host__ __device__ constexpr bool kind_is_iq_or_kextra(int kind) {
    return kind == KIND_Q2_K || kind == KIND_Q3_K || kind == KIND_IQ2_XXS || kind == KIND_IQ2_XS || kind == KIND_IQ3_XXS ||
           kind == KIND_IQ1_S || kind == KIND_IQ3_S || kind == KIND_IQ2_S || kind == KIND_IQ1_M;
}
__host__ __device__ constexpr int kind_iq_block_bytes(int kind) {
    return kind == KIND_Q2_K ? 84 : kind == KIND_Q3_K ? 110 : kind == KIND_IQ2_XXS ? 66 : kind == KIND_IQ2_XS ? 74
         : kind == KIND_IQ3_XXS ? 98 : kind == KIND_IQ1_S ? 50 : kind == KIND_IQ3_S ? 110 : kind == KIND_IQ2_S ? 82
         : kind == KIND_IQ1_M ? 56 : 0;
}
__host__ __device__ constexpr int kind_block_elems(int kind) {
    return kind == KIND_F32 || kind == KIND_F16 || kind == KIND_BF16 ? 1
           : kind == KIND_Q4_K || kind == KIND_Q5_K || kind == KIND_Q6_K || kind_is_iq_or_kextra(kind) ? 256 : 32;
}
__host__ __device__ constexpr int kind_block_bytes(int kind) {
    switch (kind) {
        case KIND_F32: return 4;
        case KIND_F16: case KIND_BF16: return 2;
        case KIND_Q4_0: case KIND_IQ4_NL: case KIND_MXFP4: return 18;
        case KIND_Q4_1: return 20;
        case KIND_Q5_0: return 22;
        case KIND_Q5_1: return 24;
        case KIND_Q8_0: return 34;
        case KIND_Q4_K: return 144;
        case KIND_Q5_K: return 176;
        case KIND_Q6_K: return 224;
        default: return kind_iq_block_bytes(kind);
    }
}
// Rows of the 32-value formats (Q4_0 ... Q8_0, IQ4_NL, MXFP4) are stored for a width rounded up to a multiple of 256
// (the padding blocks are zero), so that every array of the device layout starts on a 16-byte boundary.
__host__ __device__ constexpr int pad256(int cols) { return (cols + 255) & ~255; }
__host__ __device__ constexpr size_t kind_row_bytes(int kind, int cols) {
    return (size_t)((kind_block_elems(kind) == 32 ? pad256(cols) : cols) / kind_block_elems(kind)) * kind_block_bytes(kind);
}
__host__ __device__ constexpr bool kind_supported(int kind) {
    return kind == KIND_F32 || kind == KIND_F16 || kind == KIND_BF16 || kind == KIND_Q4_0 || kind == KIND_IQ4_NL || kind == KIND_MXFP4 ||
           kind == KIND_Q4_1 || kind == KIND_Q5_0 || kind == KIND_Q5_1 || kind == KIND_Q8_0 ||
           kind == KIND_Q4_K || kind == KIND_Q5_K || kind == KIND_Q6_K || kind_is_iq_or_kextra(kind);
}

// SwiGLU (mode 0) / GeGLU (mode 1) of one element.
__device__ __forceinline__ float fllm_glu(float g, float u, int mode) {
    float a;
    if (mode == 0) a = g / (1.0f + expf(-g));
    else a = 0.5f * g * (1.0f + tanhf(0.7978846f * (g + 0.044715f * g * g * g)));
    return a * u;
}

// IQ4_NL / IQ4_XS: 16-entry non-linear grid (ggml kvalues_iq4nl, int8). The tables hold value + 128 so that the
// looked-up bytes are unsigned; the kernels subtract 128 * scale through their usual offset term.
#define FLLM_LUT_KIND(K) ((K) == KIND_IQ4_NL || (K) == KIND_MXFP4)
// 16-entry value grid of the formats that decode through a table: IQ4_NL (ggml kvalues_iq4nl) and MXFP4 (E2M1 times two;
// the block scale then carries the remaining factor 2^(e-128)).
template <int KIND>
__host__ __device__ constexpr uint32_t lut_word(int i, int bias) {
    constexpr int iq[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
    constexpr int mx[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    return (uint32_t)(((KIND == KIND_MXFP4 ? mx[4 * i] : iq[4 * i]) + bias) & 0xff) |
           ((uint32_t)(((KIND == KIND_MXFP4 ? mx[4 * i + 1] : iq[4 * i + 1]) + bias) & 0xff) << 8) |
           ((uint32_t)(((KIND == KIND_MXFP4 ? mx[4 * i + 2] : iq[4 * i + 2]) + bias) & 0xff) << 16) |
           ((uint32_t)(((KIND == KIND_MXFP4 ? mx[4 * i + 3] : iq[4 * i + 3]) + bias) & 0xff) << 24);
}
// Four 4-bit indices (the low nibbles of the four bytes of `n`) -> four bytes of grid value + BIAS (128: unsigned
// bytes for the fp16 tensor-core path; 0: signed int8 for dp4a).
template <int KIND, int BIAS>
__device__ __forceinline__ uint32_t lut4t(uint32_t n) {
    const uint32_t t = n & 0x07070707u, h = (n >> 1) & 0x04040404u;
    const uint32_t sel = __byte_perm(t | (t >> 4), 0u, 0x4420u);
    const uint32_t hs = __byte_perm(h | (h >> 4), 0u, 0x4420u);
    const uint32_t v1 = __byte_perm(lut_word<KIND>(0, BIAS), lut_word<KIND>(1, BIAS), sel);
    const uint32_t v2 = __byte_perm(lut_word<KIND>(2, BIAS), lut_word<KIND>(3, BIAS), sel);
    return __byte_perm(v1, v2, 0x3210u | hs);
}
__device__ __forceinline__ uint32_t iq4_lut4(uint32_t n) { return lut4t<KIND_IQ4_NL, 128>(n); }

// Scale and minimum of sub-block `g` (0..8) of a Q4_K / Q5_K super-block (ggml get_scale_min_k4).
__device__ __forceinline__ void scale_min_k4(const uint8_t *scales, int g, float &sc, float &mn) {
    if (g < 4) {
        sc = (float)(scales[g] & 63);
        mn = (float)(scales[g + 4] & 63);
    } else {
        sc = (float)((scales[g + 4] & 15) | ((scales[g - 4] >> 6) << 4));
        mn = (float)((scales[g + 4] >> 4) | ((scales[g] >> 6) << 4));
    }
}

#include "iq_decode.cuh"

// Decode the 16 consecutive elements `run * 16 .. run * 16 + 15` of a row into `w`.
// `rp` points at the first byte of the row. Elements at or beyond `cols` decode to zero.
// Byte layouts match the WGSL decoder (backend/wgpu_backend.rs), which is checked against the
// CPU reference by `fllm-rs selftest-gpu`.
template <int KIND>
__device__ __forceinline__ void decode16(const uint8_t *rp, int run, int cols, float *w) {
    const int e0 = run * 16;
    if (KIND == KIND_F32) {
        const float *p = reinterpret_cast<const float *>(rp);
#pragma unroll
        for (int i = 0; i < 16; i++) w[i] = e0 + i < cols ? p[e0 + i] : 0.f;
        return;
    }
    if (KIND == KIND_F16) {
        const uint16_t *p = reinterpret_cast<const uint16_t *>(rp);
#pragma unroll
        for (int i = 0; i < 16; i++) w[i] = e0 + i < cols ? h2f_bits(p[e0 + i]) : 0.f;
        return;
    }
    if (KIND == KIND_BF16) {
        const uint16_t *p = reinterpret_cast<const uint16_t *>(rp);
#pragma unroll
        for (int i = 0; i < 16; i++) w[i] = e0 + i < cols ? __uint_as_float((uint32_t)p[e0 + i] << 16) : 0.f;
        return;
    }
    if (e0 >= cols) {
#pragma unroll
        for (int i = 0; i < 16; i++) w[i] = 0.f;
        return;
    }
    if (kind_is_iq_or_kextra(KIND)) {
        decode16_iq<KIND>(rp, run, w);
        return;
    }
    if (KIND == KIND_Q4_0 || FLLM_LUT_KIND(KIND) || KIND == KIND_Q4_1 || KIND == KIND_Q5_0 || KIND == KIND_Q5_1) {
        constexpr bool affine = KIND == KIND_Q4_1 || KIND == KIND_Q5_1;
        constexpr bool five = KIND == KIND_Q5_0 || KIND == KIND_Q5_1;
        constexpr int hdr = affine ? 4 : 2;
        constexpr int qoff = hdr + (five ? 4 : 0);
        constexpr int bytes = kind_block_bytes(KIND);
        const int half_ = run & 1;  // 0: low nibbles of the 16 bytes, 1: high nibbles
        float d, m;
        uint32_t qh = 0;
        const uint8_t *qs;
        {
            // device layout (backend/cuda.rs): [d][m][qh][quants] as separate arrays, for a width padded to 256
            const int blk = run >> 1, nb = pad256(cols) >> 5;
            d = h2f_bits(ld16(rp + (size_t)blk * 2));
            m = affine ? h2f_bits(ld16(rp + (size_t)nb * 2 + (size_t)blk * 2)) : 0.f;
            const size_t qh_off = (size_t)nb * 2 * (affine ? 2 : 1);
            if (five) qh = ld32_a2(rp + qh_off + (size_t)blk * 4);
            qs = rp + qh_off + (five ? (size_t)nb * 4 : 0) + (size_t)blk * 16;
        }
#pragma unroll
        for (int i = 0; i < 16; i++) {
            uint32_t q = (qs[i] >> (half_ * 4)) & 15u;
            if (five) q |= ((qh >> (half_ * 16 + i)) & 1u) << 4;
            if (affine) w[i] = d * (float)q + m;
            else if (FLLM_LUT_KIND(KIND)) w[i] = d * ((float)(lut4t<KIND, 128>(q) & 0xffu) - 128.f);
            else w[i] = d * ((float)q - (five ? 16.f : 8.f));
        }
        return;
    }
    if (KIND == KIND_Q8_0) {  // device layout: [scales][quants], width padded to 256, see backend/cuda.rs
        const int blk = run >> 1, nb = pad256(cols) >> 5;
        const float d = h2f_bits(ld16(rp + (size_t)blk * 2));
        const int8_t *qs = reinterpret_cast<const int8_t *>(rp + (size_t)nb * 2 + (size_t)blk * 32 + (run & 1) * 16);
#pragma unroll
        for (int i = 0; i < 16; i++) w[i] = d * (float)qs[i];
        return;
    }
    if (KIND == KIND_Q4_K || KIND == KIND_Q5_K) {
        constexpr bool five = KIND == KIND_Q5_K;
        constexpr int bytes = kind_block_bytes(KIND);
        const uint8_t *b = rp + (size_t)(run >> 4) * bytes;
        const int local = (run & 15) * 16;
        const int g = local >> 5;  // sub-block of 32
        const float d = h2f_bits(ld16(b)), dmin = h2f_bits(ld16(b + 2));
        float sc, mn;
        scale_min_k4(b + 4, g, sc, mn);
        const float ds = d * sc, dm = dmin * mn;
        const uint8_t *qs = b + (five ? 48 : 16) + (g >> 1) * 32 + (local & 31);
        const int shift = (g & 1) * 4;
        const uint8_t *qh = b + 16 + (local & 31);
#pragma unroll
        for (int i = 0; i < 16; i++) {
            uint32_t q = (qs[i] >> shift) & 15u;
            if (five) q |= (((uint32_t)qh[i] >> g) & 1u) << 4;
            w[i] = ds * (float)q - dm;
        }
        return;
    }
    if (KIND == KIND_Q6_K) {
        const uint8_t *b = rp + (size_t)(run >> 4) * 224;
        const int local = (run & 15) * 16;
        const int n = local >> 7, g = (local & 127) >> 5, j = local & 31;  // j is 0 or 16
        const float d = h2f_bits(ld16(b + 208));
        const float sc = (float)(int8_t)b[192 + n * 8 + g * 2 + (j >> 4)];
        const uint8_t *ql = b + n * 64 + (g & 1) * 32 + j;
        const uint8_t *qh = b + 128 + n * 32 + j;
        const int lshift = (g >> 1) * 4, hshift = g * 2;
#pragma unroll
        for (int i = 0; i < 16; i++) {
            const uint32_t q = ((ql[i] >> lshift) & 15u) | ((((uint32_t)qh[i] >> hshift) & 3u) << 4);
            w[i] = d * sc * ((float)q - 32.f);
        }
        return;
    }
}

// One element of a row-major weight (index counted over the whole tensor). Used by embedding
// lookups and the up-front conversion kernels, where speed does not matter.
__device__ __forceinline__ float weight_at(const uint8_t *w, int kind, size_t index, int row_cols) {
    const int be = kind_block_elems(kind);
    if (kind == KIND_F32) return reinterpret_cast<const float *>(w)[index];
    if (kind == KIND_F16) return h2f_bits(reinterpret_cast<const uint16_t *>(w)[index]);
    if (kind == KIND_BF16) return __uint_as_float((uint32_t)reinterpret_cast<const uint16_t *>(w)[index] << 16);
    // all remaining kinds decode a run of 16 at a time; pick the right instantiation
    const size_t row = index / row_cols, col = index % row_cols;
    const uint8_t *rp = w + row * kind_row_bytes(kind, row_cols);
    const size_t run = col / 16;
    const int lane = (int)(col % 16);
    float v[16];
    const int cols = row_cols;
    switch (kind) {
        case KIND_Q4_0: decode16<KIND_Q4_0>(rp, (int)run, cols, v); break;
        case KIND_IQ4_NL: decode16<KIND_IQ4_NL>(rp, (int)run, cols, v); break;
        case KIND_MXFP4: decode16<KIND_MXFP4>(rp, (int)run, cols, v); break;
        case KIND_Q4_1: decode16<KIND_Q4_1>(rp, (int)run, cols, v); break;
        case KIND_Q5_0: decode16<KIND_Q5_0>(rp, (int)run, cols, v); break;
        case KIND_Q5_1: decode16<KIND_Q5_1>(rp, (int)run, cols, v); break;
        case KIND_Q8_0: decode16<KIND_Q8_0>(rp, (int)run, cols, v); break;
        case KIND_Q4_K: decode16<KIND_Q4_K>(rp, (int)run, cols, v); break;
        case KIND_Q5_K: decode16<KIND_Q5_K>(rp, (int)run, cols, v); break;
        case KIND_Q2_K: decode16<KIND_Q2_K>(rp, (int)run, cols, v); break;
        case KIND_Q3_K: decode16<KIND_Q3_K>(rp, (int)run, cols, v); break;
        case KIND_IQ2_XXS: decode16<KIND_IQ2_XXS>(rp, (int)run, cols, v); break;
        case KIND_IQ2_XS: decode16<KIND_IQ2_XS>(rp, (int)run, cols, v); break;
        case KIND_IQ3_XXS: decode16<KIND_IQ3_XXS>(rp, (int)run, cols, v); break;
        case KIND_IQ1_S: decode16<KIND_IQ1_S>(rp, (int)run, cols, v); break;
        case KIND_IQ3_S: decode16<KIND_IQ3_S>(rp, (int)run, cols, v); break;
        case KIND_IQ2_S: decode16<KIND_IQ2_S>(rp, (int)run, cols, v); break;
        case KIND_IQ1_M: decode16<KIND_IQ1_M>(rp, (int)run, cols, v); break;
        default: decode16<KIND_Q6_K>(rp, (int)run, cols, v); break;
    }
    (void)be;
    return v[lane];
}
