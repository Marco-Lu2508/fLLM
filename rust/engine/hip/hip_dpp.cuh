// fLLM HIP backend: cheap cross-lane reductions with DPP (data-parallel primitives) instead of __shfl_xor.
//
// On AMD hardware __shfl_xor is a ds_bpermute (it goes through the LDS crossbar: latency and bandwidth of a memory access). DPP modifiers
// read a neighbouring lane's register as an operand of an ordinary VALU instruction, within a row of 16 lanes. The reductions below
// are the usual butterfly built from them:
//
//   step 1  quad_perm [1,0,3,2]   lane i <- lane i ^ 1
//   step 2  quad_perm [2,3,0,1]   lane i <- lane i ^ 2         (after it every lane holds the sum of its quad)
//   step 3  row_half_mirror       lane i <- lane 7 - i of its half row   (quad 0 <-> quad 1: sum of 8)
//   step 4  row_mirror            lane i <- lane 15 - i of its row       (half 0 <-> half 1: sum of 16)
//   step 5  __shfl_xor(v, 16)     the other row of a group of 32
//
// Available on gfx9 and on gfx10 and later (DPP16). Every lane of the row must be active (no divergence around the call).
// Level of verification: L0 (compiles; never run on hardware).
#pragma once
#include "../cuda/hip_compat.h"

namespace fllm_hip {

template <int CTRL>
__device__ __forceinline__ float dpp_f32(float v) {
    return __int_as_float(__builtin_amdgcn_mov_dpp(__float_as_int(v), CTRL, 0xf, 0xf, true));
}

// Sum over aligned groups of N lanes (N = 2, 4, 8, 16 or 32); every lane of a group gets the total.
template <int N>
__device__ __forceinline__ float group_sum(float v) {
    static_assert(N == 1 || N == 2 || N == 4 || N == 8 || N == 16 || N == 32, "group size");
    if constexpr (N >= 2) v += dpp_f32<0xB1>(v);    // quad_perm [1,0,3,2]
    if constexpr (N >= 4) v += dpp_f32<0x4E>(v);    // quad_perm [2,3,0,1]
    if constexpr (N >= 8) v += dpp_f32<0x141>(v);   // row_half_mirror
    if constexpr (N >= 16) v += dpp_f32<0x140>(v);  // row_mirror
    if constexpr (N >= 32) v += __shfl_xor(v, 16);
    return v;
}

// Maximum over aligned groups of N lanes.
template <int N>
__device__ __forceinline__ float group_max(float v) {
    static_assert(N == 1 || N == 2 || N == 4 || N == 8 || N == 16 || N == 32, "group size");
    if constexpr (N >= 2) v = fmaxf(v, dpp_f32<0xB1>(v));
    if constexpr (N >= 4) v = fmaxf(v, dpp_f32<0x4E>(v));
    if constexpr (N >= 8) v = fmaxf(v, dpp_f32<0x141>(v));
    if constexpr (N >= 16) v = fmaxf(v, dpp_f32<0x140>(v));
    if constexpr (N >= 32) v = fmaxf(v, __shfl_xor(v, 16));
    return v;
}

}  // namespace fllm_hip
