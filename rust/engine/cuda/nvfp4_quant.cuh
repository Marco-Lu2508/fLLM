// fLLM CUDA backend: NVFP4 (E2M1 values, UE4M3 block scales) conversion helpers shared by the kernels in nvfp4.cu and
// nvfp4_cutlass.cu.
//
// Activation quantization follows NVIDIA's NVFP4 recipe (ModelOpt, as implemented by vLLM `scaled_fp4_quant`, csrc/quantization/fp4/
// nvfp4_quant_kernels.cu, Apache-2.0): with a per-tensor activation factor `gs` (= 2688 / amax of the tensor, 2688 = 448 * 6;
// ModelOpt stores its inverse as `input_scale`) the UE4M3 block scale is  round_e4m3(gs * block_amax / 6)  and the E2M1 values are
// round_rne(x * gs / block_scale_as_float). The GEMM then multiplies by weight_scale_2 / gs. Without `gs` (gs = 1, the behaviour before
// FLLM_NVFP4_ACT_SCALE existed) the block scale of small activations falls into the subnormal range of E4M3 (below 2^-6) and large
// ones saturate at 448, both of which the global factor avoids.
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
    if (ex > 15 || (ex == 15 && m > 6)) return 0x7E;  // E4M3 (fn) has exponent 15 up to mantissa 6 = 448; code 0x7F is NaN
    return (uint32_t)(ex << 3 | m);
}
// Nearest E2M1 magnitude index (0..7) of |v| / scale, ties to even (the codes 0, 2, 4, 6 of {0, .5, 1, 1.5, 2, 3, 4, 6} have a zero mantissa
// bit; this is what cvt.rn.satfinite.e2m1x2.f32 does). The midpoints 0.25, 1.25, 2.5 and 5 round down, 0.75, 1.75 and 3.5 round up.
__device__ __forceinline__ uint32_t e2m1_idx(float a) {
    return (a > 0.25f) + (a >= 0.75f) + (a > 1.25f) + (a >= 1.75f) + (a > 2.5f) + (a >= 3.5f) + (a > 5.f);
}

// Largest |value| an activation tensor may reach before its block scale saturates at 448 (E4M3 maximum) times 6 (E2M1 maximum).
#define FLLM_FP4_ACT_RANGE 2688.f

// Activation factor of a dynamic per-tensor scale: 2688 / amax (1 for an all-zero tensor).
__device__ __forceinline__ float fp4_dynamic_gs(float amax) { return amax > 0.f ? FLLM_FP4_ACT_RANGE / amax : 1.f; }

// Quantizes 16 values to one UE4M3 scale (returned) and two words of eight nibbles (nibble q of word h = element 8h + q).
// gs: activation factor (see above), 1 for none.
__device__ __forceinline__ uint32_t fp4_quantize_values(const float *v, uint32_t words[2], float gs = 1.f) {
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(v[i]));
    const uint32_t sb = f_to_ue4m3(amax * gs * (1.f / 6.f));
    const float sd = ue4m3_to_f(sb);
    const float inv = sd > 0.f ? gs / sd : 0.f;
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

// Mode of FLLM_NVFP4_ACT_SCALE: 0 = off (gs = 1, the behaviour before the switch), 1 = static (the checkpoint's input_scale, stored in the
// weight trailer; gs = 1 for checkpoints without it, auto is the same), 2 = dynamic (amax of the activation tensor of every GEMM).
#include <cstdlib>
#include <cstring>
static inline int fllm_nvfp4_act_scale_mode() {
    static const int mode = [] {
        const char *v = std::getenv("FLLM_NVFP4_ACT_SCALE");
        if (!v || !*v || !std::strcmp(v, "auto") || !std::strcmp(v, "static")) return 1;
        if (!std::strcmp(v, "dynamic")) return 2;
        return 0;  // off, legacy, none, 0 ...
    }();
    return mode;
}

// Trailer behind the swizzled scale factors of a device NVFP4 weight (written by nvfp4_device_layout in backend/cuda.rs):
// four floats {gs = 1 / input_scale, alpha = weight_scale_2 * input_scale, weight_scale_2, 1 if the checkpoint had an input_scale}.
// Checkpoints without input_scale store {1, weight_scale_2, weight_scale_2, 0}, which makes the static mode equal to off.
struct Nvfp4Trailer {
    float gs, alpha, wscale, valid;
};
static inline size_t fllm_nvfp4_trailer_offset(int rows, int cols) {
    const size_t nibble_bytes = (((size_t)rows * (cols >> 1)) + 255) & ~size_t(255);
    const size_t scale_bytes = (size_t)((rows + 127) / 128) * (cols >> 6) * 512;
    return (nibble_bytes + scale_bytes + 255) & ~size_t(255);
}
