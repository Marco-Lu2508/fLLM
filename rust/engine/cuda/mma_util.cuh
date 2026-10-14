// fLLM CUDA backend: small PTX wrappers shared by the tensor-core GEMM kernels.
#pragma once
#include "common.cuh"
#if !defined(__HIPCC__)

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
// Turing: no asynchronous copies. Synchronous stores keep the multi-stage loops correct (the data is in shared memory when
// the loop's __syncthreads publishes it); `valid == false` zero-fills like the cp.async src-size operand.
__device__ __forceinline__ void cp_async16(void *smem, const void *gmem, bool valid) {
    *reinterpret_cast<int4 *>(smem) = valid ? *reinterpret_cast<const int4 *>(gmem) : make_int4(0, 0, 0, 0);
}
__device__ __forceinline__ void cp_async8(void *smem, const void *gmem, bool valid) {
    *reinterpret_cast<int2 *>(smem) = valid ? *reinterpret_cast<const int2 *>(gmem) : make_int2(0, 0);
}
__device__ __forceinline__ void cp_async4_ca(void *smem, const void *gmem, bool valid) {
    *reinterpret_cast<int *>(smem) = valid ? *reinterpret_cast<const int *>(gmem) : 0;
}
__device__ __forceinline__ void cp_async_commit() {}
template <int N>
__device__ __forceinline__ void cp_async_wait() {}
#else
__device__ __forceinline__ void cp_async16(void *smem, const void *gmem, bool valid) {
    const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    const int sz = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(gmem), "r"(sz));
}
__device__ __forceinline__ void cp_async4_ca(void *smem, const void *gmem, bool valid) {
    const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    const int sz = valid ? 4 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(d), "l"(gmem), "r"(sz));
}
__device__ __forceinline__ void cp_async8(void *smem, const void *gmem, bool valid) {
    const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    const int sz = valid ? 8 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n" ::"r"(d), "l"(gmem), "r"(sz));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
#endif

__device__ __forceinline__ void ldmatrix4(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, const void *p) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a));
}
__device__ __forceinline__ void mma16816(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
    // m16n8k16 as two m16n8k8: k 0..7 uses (a0, a1, b0), k 8..15 uses (a2, a3, b1)
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(b0));
    asm("mma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[2]), "r"(a[3]), "r"(b1));
#else
    asm(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

__device__ __forceinline__ __half2 u32_as_half2(uint32_t v) { return *reinterpret_cast<const __half2 *>(&v); }
__device__ __forceinline__ uint32_t half2_as_u32(__half2 v) { return *reinterpret_cast<const uint32_t *>(&v); }

// Two nibbles (already isolated in the low bits of bytes 0 and 1 of `bytes01`) as exact FP16 values,
// then n * s - o.
__device__ __forceinline__ uint32_t dequant_pair(uint32_t nibbles_in_bytes, uint32_t sel, __half2 s2, __half2 o2) {
    const uint32_t spread = __byte_perm(nibbles_in_bytes, 0u, sel);  // 16-bit lanes {byte_a, byte_b}
    const __half2 magic = u32_as_half2(spread | 0x64006400u);        // 1024 + n
    const __half2 n = __hsub2(magic, __half2half2(__float2half_rn(1024.f)));
    return half2_as_u32(__hfma2(n, s2, __hneg2(o2)));
}

// (a & b) | c in one instruction; 0xea is that combination in the LOP3 truth table.
__device__ __forceinline__ uint32_t lop3_eax(uint32_t a, uint32_t b, uint32_t c) {
    uint32_t d;
    asm("lop3.b32 %0, %1, %2, %3, 0xea;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// Same result as `dequant_pair`, but the caller leaves the nibbles unmasked: `mask` picks the low
// (0x000f000f) or the high (0x00f000f0) nibble of every byte and is applied together with the
// 1024+n magic number in a single LOP3, which replaces the shift, the mask and the OR.
__device__ __forceinline__ uint32_t dequant_pair_lop3(uint32_t bytes_ab, uint32_t sel, uint32_t mask, __half2 s2,
                                                      __half2 o2) {
    const uint32_t spread = __byte_perm(bytes_ab, 0u, sel);                   // 16-bit lanes {byte_a, byte_b}
    const __half2 magic = u32_as_half2(lop3_eax(spread, mask, 0x64006400u));  // 1024 + n
    const __half2 n = __hsub2(magic, __half2half2(__float2half_rn(1024.f)));
    return half2_as_u32(__hfma2(n, s2, __hneg2(o2)));
}
#endif  // !__HIPCC__
