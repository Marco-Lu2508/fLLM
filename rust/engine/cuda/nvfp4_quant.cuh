// fLLM CUDA backend: NVFP4 (E2M1 values, UE4M3 block scales) conversion helpers shared by the kernels in nvfp4.cu and
// nvfp4_cutlass.cu.
#pragma once

#include <cstdint>

__device__ __forceinline__ float ue4m3_to_f(uint32_t b) {
    const uint32_t e = (b >> 3) & 15u, m = b & 7u;
    return e == 0 ? ldexpf((float)m * 0.125f, -6) : ldexpf(1.f + (float)m * 0.125f, (int)e - 7);
}
// Nearest UE4M3 value (round to nearest, saturating at 448).
__device__ __forceinline__ uint32_t f_to_ue4m3(float x) {
    if (!(x > 0.f)) return 0;
    if (x >= 448.f) return 0x7E;
    int e;
    float f = frexpf(x, &e);  // x = f * 2^e, f in [0.5, 1)
    int ex = e - 1 + 7;       // biased exponent of 1.m form
    if (ex <= 0) {            // subnormal: m * 2^-9
        int m = __float2int_rn(x * 512.f);
        return m >= 8 ? 8u : (uint32_t)m;
    }
    int m = __float2int_rn((f * 2.f - 1.f) * 8.f);
    if (m == 8) { m = 0; ex++; }
    if (ex >= 15) return 0x7E;
    return (uint32_t)(ex << 3 | m);
}
// Nearest E2M1 magnitude index (0..7) of |v| / scale.
__device__ __forceinline__ uint32_t e2m1_idx(float a) {
    // thresholds are midpoints of {0, .5, 1, 1.5, 2, 3, 4, 6}
    return (a > 0.25f) + (a > 0.75f) + (a > 1.25f) + (a > 1.75f) + (a > 2.5f) + (a > 3.5f) + (a > 5.f);
}

// Quantizes 16 values to one UE4M3 scale (returned) and two words of eight nibbles (nibble q of word h = element 8h + q).
__device__ __forceinline__ uint32_t fp4_quantize_values(const float *v, uint32_t words[2]) {
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(v[i]));
    const uint32_t sb = f_to_ue4m3(amax * (1.f / 6.f));
    const float sd = ue4m3_to_f(sb);
    const float inv = sd > 0.f ? 1.f / sd : 0.f;
#pragma unroll
    for (int h = 0; h < 2; h++) {
        uint32_t w = 0;
#pragma unroll
        for (int q = 0; q < 8; q++) {
            const float t = v[8 * h + q] * inv;
            const uint32_t nib = e2m1_idx(fabsf(t)) | (t < 0.f ? 8u : 0u);
            w |= nib << (4 * q);
        }
        words[h] = w;
    }
    return sb;
}
