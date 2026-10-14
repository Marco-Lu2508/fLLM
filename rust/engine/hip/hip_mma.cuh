// fLLM HIP backend: one 16x16x16 matrix-core step (FP16 inputs, FP32 accumulation) behind one interface for CDNA (MFMA), RDNA3 (WMMA) and
// RDNA4 (WMMA, gfx12 encoding).
//
//   D[m][n] += sum_k A[m][k] * B[k][n]          (m, n, k = 0..15)
//
// Every lane of the wavefront holds a slice of the three matrices; the slices differ per architecture. This header gives them one
// name each, so a kernel only has to ask "which k values does my lane hold" and "which (row, col) does accumulator element i belong to":
//
//                     CDNA (wave64)               RDNA3 (wave32)                RDNA4 (wave32)
//   builtin           mfma_f32_16x16x16f16        wmma_f32_16x16x16_f16_w32     wmma_f32_16x16x16_f16_w32_gfx12
//   A / B fragment    4 halves                    16 halves                     8 halves
//   A element         A[l % 16][4 * (l / 16) + e] A[l % 16][e]                  A[l % 16][8 * (l / 16) + e]
//   B element         B[4 * (l / 16) + e][l % 16] B[e][l % 16]                  B[8 * (l / 16) + e][l % 16]
//   accumulator       4 floats                    8 floats                      8 floats
//   D element i       D[4 * (l / 16) + i][l % 16] D[2 * i + l / 16][l % 16]     D[8 * (l / 16) + i][l % 16]
//
// (l = lane id inside the wavefront.) RDNA3 lanes 16..31 repeat the A / B data of lanes 0..15. The builtins and fragment sizes were
// checked against rocWMMA (internal/mfma_impl.hpp, internal/wmma_impl.hpp, MIT); the lane layouts follow the AMD matrix instruction
// calculator and the RDNA3 / RDNA4 ISA guides. Level of verification: L0 (compiles; layouts never run on hardware).
//
// The macros FLLM_HIP_CDNA / FLLM_HIP_RDNA3 / FLLM_HIP_RDNA4 exist only in the device pass of a matching gfx target; FLLM_HIP_MMA is set
// when any of them does. A kernel body without it must be empty (the host pass and gfx targets without matrix cores compile it too).
#pragma once
#include "../cuda/hip_compat.h"

#if defined(FLLM_HIP_CDNA) || defined(FLLM_HIP_RDNA3) || defined(FLLM_HIP_RDNA4)
#define FLLM_HIP_MMA 1
#endif

namespace fllm_hip {

typedef _Float16 f16x4 __attribute__((ext_vector_type(4)));
typedef _Float16 f16x8 __attribute__((ext_vector_type(8)));
typedef _Float16 f16x16 __attribute__((ext_vector_type(16)));
typedef float f32x4 __attribute__((ext_vector_type(4)));
typedef float f32x8 __attribute__((ext_vector_type(8)));

#if defined(FLLM_HIP_CDNA)
constexpr int MMA_WAVE = 64;
constexpr int MMA_FRAG = 4;      // halves per lane in an A / B fragment
constexpr int MMA_ACC = 4;       // floats per lane in the accumulator
typedef f16x4 MmaFrag;
typedef f32x4 MmaAcc;
// first k held by lane `lane` inside a 16-wide k step
__device__ __forceinline__ int mma_k0(int lane) { return 4 * (lane >> 4); }
// accumulator element i of lane `lane` belongs to this row (m) of D; its column (n) is lane % 16
__device__ __forceinline__ int mma_row(int lane, int i) { return 4 * (lane >> 4) + i; }
__device__ __forceinline__ MmaAcc mma16(MmaFrag a, MmaFrag b, MmaAcc c) { return __builtin_amdgcn_mfma_f32_16x16x16f16(a, b, c, 0, 0, 0); }
// 4 consecutive halves of LDS (8-byte aligned)
__device__ __forceinline__ MmaFrag mma_load(const __half *p) { return *reinterpret_cast<const f16x4 *>(p); }
#elif defined(FLLM_HIP_RDNA3)
constexpr int MMA_WAVE = 32;
constexpr int MMA_FRAG = 16;
constexpr int MMA_ACC = 8;
typedef f16x16 MmaFrag;
typedef f32x8 MmaAcc;
__device__ __forceinline__ int mma_k0(int) { return 0; }
__device__ __forceinline__ int mma_row(int lane, int i) { return 2 * i + (lane >> 4); }
__device__ __forceinline__ MmaAcc mma16(MmaFrag a, MmaFrag b, MmaAcc c) { return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c); }
// 16 consecutive halves of LDS, 16-byte aligned (two 128-bit reads: the vector type itself wants 32-byte alignment)
__device__ __forceinline__ MmaFrag mma_load(const __half *p) {
    const f16x8 lo = *reinterpret_cast<const f16x8 *>(p), hi = *reinterpret_cast<const f16x8 *>(p + 8);
    return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
}
#elif defined(FLLM_HIP_RDNA4)
constexpr int MMA_WAVE = 32;
constexpr int MMA_FRAG = 8;
constexpr int MMA_ACC = 8;
typedef f16x8 MmaFrag;
typedef f32x8 MmaAcc;
__device__ __forceinline__ int mma_k0(int lane) { return 8 * (lane >> 4); }
__device__ __forceinline__ int mma_row(int lane, int i) { return 8 * (lane >> 4) + i; }
__device__ __forceinline__ MmaAcc mma16(MmaFrag a, MmaFrag b, MmaAcc c) { return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c); }
__device__ __forceinline__ MmaFrag mma_load(const __half *p) { return *reinterpret_cast<const f16x8 *>(p); }
#endif

}  // namespace fllm_hip
