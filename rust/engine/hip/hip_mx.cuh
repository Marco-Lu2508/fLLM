// fLLM HIP backend: 8-bit and 4-bit floating-point formats in software, the same on every GPU and on the host.
//
//   OCP E4M3 (bias 7, max 448, NaN = S.1111.111, no infinity)   what DeepSeek / Qwen / Llama FP8 checkpoints and NVIDIA GPUs use, and
//                                                               what the matrix cores of gfx950 (MI350 / MI355X) read
//   FNUZ E4M3 (bias 8, max 240, NaN = 1.0000.000, no -0)        what the matrix cores of gfx940 / gfx941 / gfx942 (MI300) read
//   E8M0 (a power of two, 2^(e - 127), 255 = NaN)               the shared block scale of the OCP microscaling (MX) formats
//   E2M1 (FP4: +-{0, 0.5, 1, 1.5, 2, 3, 4, 6})                  MXFP4 elements
//
// The software conversions are the reference for the kernels (and are mirrored value for value by rust/engine/src/mx_ref.rs, which the
// unit tests check exhaustively); the hardware conversion instructions are not used so that all architectures agree bit for bit.
// Level of verification: the Rust twin is tested on the CPU; this header compiles for gfx942 / gfx950 / gfx1100.
#pragma once
#include <stdint.h>
#if defined(__HIPCC__)
#include "../cuda/hip_compat.h"
#elif !defined(__CUDACC__)
// plain C++ (the host test that compares this header with src/mx_ref.rs)
#include <string.h>
#define __host__
#define __device__
static inline float __uint_as_float(uint32_t u) { float f; memcpy(&f, &u, 4); return f; }
static inline uint32_t __float_as_uint(float f) { uint32_t u; memcpy(&u, &f, 4); return u; }
#endif

#if defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__)
#define FLLM_FP8_FNUZ_HW 1  // the matrix cores of this GPU read FNUZ, not OCP, 8-bit floats
#endif

namespace fllm_mx {

// ---- E4M3 (OCP) ------------------------------------------------------------------------------------------------------------------------
__host__ __device__ inline float e4m3_ocp_to_float(uint32_t b) {
    const uint32_t s = b & 0x80u, e = (b >> 3) & 15u, m = b & 7u;
    float v;
    if (e == 15u && m == 7u) {
        v = __uint_as_float(0x7fc00000u);
    } else if (e == 0u) {
        v = (float)m * (1.0f / 512.0f);
    } else {
        v = __uint_as_float(((e + 120u) << 23) | (m << 20));
    }
    return s ? -v : v;
}

// round to nearest even, saturate to +-448 (the behaviour of the MX conversion with saturation; NaN stays NaN)
__host__ __device__ inline uint32_t float_to_e4m3_ocp(float x) {
    const uint32_t bits = __float_as_uint(x);
    const uint32_t sign = (bits >> 24) & 0x80u;
    const uint32_t abs_bits = bits & 0x7fffffffu;
    if (abs_bits > 0x7f800000u) return 0x7fu;  // NaN
    if (abs_bits >= 0x43e00000u) return sign | 0x7eu;  // >= 448 (also +-inf): the largest finite value
    if (abs_bits < 0x3a800000u) {  // below 2^-10 (half the smallest subnormal 2^-9): rounds to zero (ties to even: exactly 2^-10 -> 0)
        return sign;
    }
    const int exp = (int)(abs_bits >> 23) - 127;
    if (exp < -6) {  // subnormal: m * 2^-9
        const float scaled = __uint_as_float(abs_bits) * 512.0f;  // exact
        // round to nearest even by hand: scaled is in (0.5, 8)
        const float fl = (float)(int)scaled;
        const float diff = scaled - fl;
        int m = (int)fl;
        if (diff > 0.5f || (diff == 0.5f && (m & 1))) m++;
        return sign | (uint32_t)m;  // m == 8 carries into the exponent field: 2^-6, which is the smallest normal
    }
    // normal: 3 mantissa bits, round to nearest even on the 20 dropped bits
    uint32_t mant = abs_bits & 0x7fffffu;
    uint32_t e = (uint32_t)(exp + 7);
    uint32_t m = mant >> 20;
    const uint32_t rest = mant & 0xfffffu;
    if (rest > 0x80000u || (rest == 0x80000u && (m & 1u))) {
        m++;
        if (m == 8u) {
            m = 0;
            e++;
        }
    }
    if (e > 15u || (e == 15u && m == 7u)) return sign | 0x7eu;
    return sign | (e << 3) | m;
}

// ---- E4M3 (FNUZ) -----------------------------------------------------------------------------------------------------------------------
__host__ __device__ inline float e4m3_fnuz_to_float(uint32_t b) {
    if (b == 0x80u) return __uint_as_float(0x7fc00000u);
    const uint32_t s = b & 0x80u, e = (b >> 3) & 15u, m = b & 7u;
    const float v = e == 0u ? (float)m * (1.0f / 1024.0f) : __uint_as_float(((e + 119u) << 23) | (m << 20));
    return s ? -v : v;
}

// round to nearest even, saturate to +-240; zero is +0 (there is no negative zero), NaN is 0x80
__host__ __device__ inline uint32_t float_to_e4m3_fnuz(float x) {
    const uint32_t bits = __float_as_uint(x);
    const uint32_t sign = (bits >> 24) & 0x80u;
    const uint32_t abs_bits = bits & 0x7fffffffu;
    if (abs_bits > 0x7f800000u) return 0x80u;
    if (abs_bits >= 0x43700000u) return sign | 0x7fu;  // >= 240
    if (abs_bits < 0x3a000000u) return 0u;  // below 2^-11 (half the smallest subnormal 2^-10)
    const int exp = (int)(abs_bits >> 23) - 127;
    if (exp < -7) {  // subnormal: m * 2^-10
        const float scaled = __uint_as_float(abs_bits) * 1024.0f;
        const float fl = (float)(int)scaled;
        const float diff = scaled - fl;
        int m = (int)fl;
        if (diff > 0.5f || (diff == 0.5f && (m & 1))) m++;
        return m == 0 ? 0u : (sign | (uint32_t)m);
    }
    uint32_t mant = abs_bits & 0x7fffffu;
    uint32_t e = (uint32_t)(exp + 8);
    uint32_t m = mant >> 20;
    const uint32_t rest = mant & 0xfffffu;
    if (rest > 0x80000u || (rest == 0x80000u && (m & 1u))) {
        m++;
        if (m == 8u) {
            m = 0;
            e++;
        }
    }
    if (e > 15u) return sign | 0x7fu;
    return sign | (e << 3) | m;
}

// an OCP E4M3 byte as the FNUZ byte of the same value (checkpoints are OCP; MI300 matrix cores read FNUZ). Values above 240 saturate.
__host__ __device__ inline uint32_t e4m3_ocp_to_fnuz(uint32_t b) { return float_to_e4m3_fnuz(e4m3_ocp_to_float(b)); }
__host__ __device__ inline uint32_t e4m3_fnuz_to_ocp(uint32_t b) { return float_to_e4m3_ocp(e4m3_fnuz_to_float(b)); }

// ---- E5M2 (OCP): decode only ------------------------------------------------------------------------------------------------------------
__host__ __device__ inline float e5m2_ocp_to_float(uint32_t b) {
    const uint32_t s = b & 0x80u, e = (b >> 2) & 31u, m = b & 3u;
    float v;
    if (e == 31u) {
        v = m == 0u ? __uint_as_float(0x7f800000u) : __uint_as_float(0x7fc00000u);
    } else if (e == 0u) {
        v = (float)m * (1.0f / 65536.0f);
    } else {
        v = __uint_as_float(((e + 112u) << 23) | (m << 21));
    }
    return s ? -v : v;
}

// ---- E8M0 and E2M1 (MX) ----------------------------------------------------------------------------------------------------------------
// 2^(e - 127); 255 is NaN. Built from the exponent field directly, so 2^-127 is the denormal 0x00400000.
__host__ __device__ inline float e8m0_to_float(uint32_t e) {
    if (e == 255u) return __uint_as_float(0x7fc00000u);
    if (e == 0u) return __uint_as_float(0x00400000u);
    return __uint_as_float(e << 23);
}

__host__ __device__ inline float e2m1_to_float(uint32_t code) {
    const uint32_t e = (code >> 1) & 3u, m = code & 1u;
    const float v = e == 0u ? 0.5f * (float)m : __uint_as_float((126u + e) << 23 | (m << 22));
    return (code & 8u) ? -v : v;
}

// The shared scale of a block of 32 values whose largest magnitude is `amax`, for E4M3 elements (OCP MX spec: floor(log2(amax)) minus
// the exponent of the largest element, 8). Returns the E8M0 byte; an all-zero block gets scale 1.
__host__ __device__ inline uint32_t mx_scale_for_e4m3(float amax) {
    if (!(amax > 0.0f)) return 127u;
    int exp = (int)((__float_as_uint(amax) >> 23) & 0xffu) - 127;  // floor(log2) for normal numbers
    if (exp < -126) exp = -126;
    int shared = exp - 8;
    if (shared < -127) shared = -127;
    if (shared > 127) shared = 127;
    return (uint32_t)(shared + 127);
}

}  // namespace fllm_mx
