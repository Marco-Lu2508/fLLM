// FLLM Vulkan shaders, package 3 (non-matmul operations): helpers shared by the attn_/norm_/rope_/act_/kv_/moe_/delta_/mamba_/misc_ shaders.
// Included right after `#version 450`. Conventions (INTERFACE_OPS.md): entry `main`, descriptor set 0, storage buffers 0..n, one push-constant block of
// at most 128 bytes made of 32-bit scalars only (so the Rust side can lay it out as a plain word array).
#ifndef FLLM_OPS_COMMON
#define FLLM_OPS_COMMON

#extension GL_EXT_control_flow_attributes : enable

// Finite stand-in for -infinity in running maxima: exp(NEG - m) is exactly 0 for any finite m and never produces inf - inf.
const float NEG = -3.0e38;

float silu(float x) { return x / (1.0 + exp(-x)); }
float sigmoid1(float x) { return 1.0 / (1.0 + exp(-x)); }
// tanh approximation of GELU as in the CPU decoder (`gelu_tanh`); the argument of tanh is clamped to +-15 (exactly 1 in f32 there)
// so that drivers computing tanh through exp() cannot return NaN.
float gelu_tanh(float x) {
    float t = clamp(0.7978846 * (x + 0.044715 * x * x * x), -15.0, 15.0);
    return 0.5 * x * (1.0 + tanh(t));
}
// ln(1 + exp(x)), the value itself above 20 (CPU: `if raw > 20 { raw } else { raw.exp().ln_1p() }`)
// erf as the CPU decoder computes it (`erf` of rust_decoder.rs): Chebyshev fit of erfc (Numerical Recipes), fractional error below 1.2e-7
float erf_fit(float x) {
    float t = 1.0 / (1.0 + 0.5 * abs(x));
    float poly = -1.26551223 + t * (1.00002368 + t * (0.37409196 + t * (0.09678418 + t * (-0.18628806
        + t * (0.27886807 + t * (-1.13520398 + t * (1.48851587 + t * (-0.82215223 + t * 0.17087277))))))));
    float tau = t * exp(-x * x + poly);
    return x >= 0.0 ? 1.0 - tau : tau - 1.0;
}
float gelu_erf(float x) { return 0.5 * x * (1.0 + erf_fit(x * 0.70710678)); }
float softplus20(float x) { return x > 20.0 ? x : log(1.0 + exp(x)); }

// binary16 bits of x, rounded to nearest even. packHalf2x16 is not used: its rounding mode is implementation defined and AMD drivers round
// toward zero (an error of up to one unit in the last place that the KV cache write test of the Radeon Z1 Extreme showed).
uint f16_bits(float x) {
    const uint f = floatBitsToUint(x);
    const uint sign = (f >> 16) & 0x8000u;
    const uint e = (f >> 23) & 0xFFu;
    uint m = f & 0x7FFFFFu;
    if (e == 0xFFu) {  // infinity or NaN (a NaN keeps a non-zero payload)
        return sign | 0x7C00u | (m != 0u ? (0x200u | (m >> 13)) : 0u);
    }
    const int ex = int(e) - 112;  // exponent re-biased from 127 to 15
    if (ex >= 31) return sign | 0x7C00u;
    if (ex <= 0) {  // subnormal binary16 or zero
        if (ex < -10) return sign;
        m |= 0x800000u;
        const uint shift = uint(14 - ex);
        uint h = m >> shift;
        const uint rem = m & ((1u << shift) - 1u);
        const uint half_way = 1u << (shift - 1u);
        if (rem > half_way || (rem == half_way && (h & 1u) != 0u)) h += 1u;
        return sign | h;
    }
    uint h = (uint(ex) << 10) | (m >> 13);
    const uint rem = m & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u) != 0u)) h += 1u;  // a carry into the exponent is the correct result (up to infinity)
    return sign | h;
}

// Index of the workgroup in a 1-D logical grid that was split over x and y (count > 65535): x + y * numGroupsX.
uint group_index() { return gl_WorkGroupID.x + gl_WorkGroupID.y * gl_NumWorkGroups.x; }

#endif
