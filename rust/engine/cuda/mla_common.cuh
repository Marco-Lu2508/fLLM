// fLLM CUDA backend: helpers shared by the latent-cache MLA kernels (mla_latent.cu, mla_decode.cu): ldmatrix variants, FP16 packing and the
// decoding of one cache element in every format the latent cache can use.
#pragma once
#include "common.cuh"
#include "mma_util.cuh"
#include "fllm_cuda.h"

#if !defined(__HIPCC__)
__device__ __forceinline__ void ldsm_x2(uint32_t &r0, uint32_t &r1, const void *p) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(a));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, const void *p) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n" : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(a));
}
__device__ __forceinline__ uint32_t mla_pack_half2(float a, float b) { return half2_as_u32(__floats2half2_rn(a, b)); }
#endif

// Cache formats of the latent cache (KvFormat codes of src/kv_format.rs).
enum { MLA_F16 = 1, MLA_Q8_0 = 2, MLA_E4M3 = 4, MLA_INT8 = 6 };

__device__ __forceinline__ float mla_e4m3(uint32_t b) {  // same decoder as kvq_e4m3 in kv_quant.cu
    const uint32_t e = (b >> 3) & 15u, m = b & 7u;
    const float v = e == 0 ? (float)m * (1.0f / 512.0f) : __uint_as_float(((e + 120u) << 23) | (m << 20));
    return (b & 0x80u) ? -v : v;
}

// Value `e` of the latent row of `slot` (generic, used by the reference kernel and the tile loader of the scaled formats).
// Same addressing as kvq_at (kv_quant.cu) for one head: element index slot * kv + e, scale group slot * gps + (e >> gshift).
__device__ __forceinline__ float mla_cache_at(const uint16_t *w, const FllmKvDesc &d, size_t slot, int e) {
    const size_t i = slot * (size_t)d.kv + (size_t)e;
    if (d.fmt == MLA_F16) return h2f_bits(w[i]);
    const size_t g = slot * (size_t)d.gps + (d.gshift < 0 ? (size_t)0 : (size_t)(e >> d.gshift));
    const uint32_t code = (w[i >> 1] >> ((i & 1) * 8)) & 0xffu;
    const float scale = h2f_bits(w[d.scale_off + g]);
    return d.fmt == MLA_E4M3 ? mla_e4m3(code) * scale : (float)(int8_t)code * scale;
}
