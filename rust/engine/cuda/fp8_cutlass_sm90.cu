// fLLM CUDA backend: FP8 (E4M3) W8A8 GEMM for Hopper (sm_90a) with CUTLASS 3.x/4.x: TMA loads, warp-specialized wgmma mainloop, FP32 accumulation, and an
// epilogue that applies the dequantization scales of the operands,
//
//   Y[token][row] = a_scale[token] * w_scale[row] * sum_k qx[token][k] * qw[row][k] + bias[row]
//
// as an epilogue visitor tree (the per-token scale is a column broadcast, the per-row weight scale and the bias are row broadcasts).
//
// Derived from (the notices of these files apply to the corresponding parts, see THIRD_PARTY.md):
//   * CUTLASS, examples/54_hopper_fp8_warp_specialized_gemm/54_hopper_fp8_warp_specialized_gemm.cu
//     Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. SPDX-License-Identifier: BSD-3-Clause
//     (licenses/CUTLASS-BSD-3-Clause.txt): the collective builders of an FP8 warp-specialized GEMM for SM90 (E4M3 x E4M3, FP32 accumulation, TMA).
//   * vLLM, csrc/quantization/w8a8/cutlass/scaled_mm_c3x_sm90_fp8*.cu and c3x/scaled_mm_epilogues_c3x.hpp (Apache-2.0, Copyright contributors to the
//     vLLM project): the idea of folding the per-token and per-channel scales into the epilogue as an EVT of broadcasts and multiplications.
// Changes for fLLM: no PyTorch types (raw device pointers), FP32 result, the bias as a second row broadcast, two tile configurations chosen by batch.
//
// Test level: compiled only (sm_90a); the self test group `w8a8` runs it on a Hopper device with FLLM_FP8_GEMM=cutlass.
#include <cstdint>
#include <cstdlib>

#include "common.cuh"
#include "fllm_cuda_internal.h"
#include "fp8_gemm.h"

#if defined(FLLM_HOPPER) && defined(FLLM_HAVE_CUTLASS)
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#if defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
namespace fllm_fp8c {
using namespace cute;

// Tile 64 x 128 x 128 with the ping-pong schedule for up to 64 tokens, 128 x 128 x 128 cooperative above.
template <int BM, bool Pingpong>
struct Fp8Gemm {
    using ElementA = cutlass::float_e4m3_t;
    using ElementB = cutlass::float_e4m3_t;
    using LayoutA = cutlass::layout::RowMajor;     // activations [token][k]
    using LayoutB = cutlass::layout::ColumnMajor;  // weights [row][k] = K-major
    static constexpr int AlignmentA = 16, AlignmentB = 16;
    using ElementD = float;
    using LayoutD = cutlass::layout::RowMajor;
    static constexpr int AlignmentD = 4;
    using ElementAcc = float;
    using TileShape = Shape<Int<BM>, _128, _128>;
    using ClusterShape = Shape<_1, _1, _1>;
    using KernelSchedule = std::conditional_t<Pingpong, cutlass::gemm::KernelTmaWarpSpecializedPingpong, cutlass::gemm::KernelTmaWarpSpecializedCooperative>;
    using EpilogueSchedule = std::conditional_t<Pingpong, cutlass::epilogue::TmaWarpSpecialized, cutlass::epilogue::TmaWarpSpecializedCooperative>;

    // D = a_scale[m] * (w_scale[n] * acc) + bias[n]
    using Multiply = cutlass::epilogue::fusion::Sm90Compute<cutlass::multiplies, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
    using Plus = cutlass::epilogue::fusion::Sm90Compute<cutlass::plus, float, float, cutlass::FloatRoundStyle::round_to_nearest>;
    using WeightScale = cutlass::epilogue::fusion::Sm90RowBroadcast<0, TileShape, float, float, Stride<_0, _1, _0>, 4>;
    using TokenScale = cutlass::epilogue::fusion::Sm90ColBroadcast<0, TileShape, float, float, Stride<_1, _0, _0>, 1>;
    using Bias = cutlass::epilogue::fusion::Sm90RowBroadcast<0, TileShape, float, float, Stride<_0, _1, _0>, 4>;
    using Acc = cutlass::epilogue::fusion::Sm90AccFetch;
    using ScaledByWeight = cutlass::epilogue::fusion::Sm90EVT<Multiply, WeightScale, Acc>;
    using ScaledByToken = cutlass::epilogue::fusion::Sm90EVT<Multiply, TokenScale, ScaledByWeight>;
    using Epilogue = cutlass::epilogue::fusion::Sm90EVT<Plus, Bias, ScaledByToken>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, TileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto, ElementAcc,
        float, void, LayoutD, AlignmentD, ElementD, LayoutD, AlignmentD, EpilogueSchedule, Epilogue>::CollectiveOp;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp, ElementA, LayoutA, AlignmentA, ElementB, LayoutB, AlignmentB, ElementAcc, TileShape,
        ClusterShape, cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        KernelSchedule>::CollectiveOp;
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

struct Workspace {
    void *ptr = nullptr;
    size_t bytes = 0;
    bool reserve(size_t need) {
        if (need <= bytes) return true;
        void *p = nullptr;
        if (cudaMalloc(&p, need) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        ptr = p;  // the previous block is kept: a captured graph may still use it
        bytes = need;
        return true;
    }
};

template <int BM, bool Pingpong>
int run(cudaStream_t s, Workspace &ws, const uint8_t *xq, const float *xs, const uint8_t *w, const float *wscale, const float *bias, float *y, int M, int N, int K) {
    using G = Fp8Gemm<BM, Pingpong>;
    using Gemm = typename G::Gemm;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideD = typename Gemm::GemmKernel::StrideD;
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {reinterpret_cast<cutlass::float_e4m3_t const *>(xq), stride_A, reinterpret_cast<cutlass::float_e4m3_t const *>(w), stride_B},
        {{}, nullptr, stride_D, y, stride_D}};
    // visitor arguments: {bias, {token scale, {weight scale, acc, multiply}, multiply}, plus}
    args.epilogue.thread = {{bias, 0.f, {}}, {{xs, 0.f, {}}, {{wscale, 0.f, {}}, {}, {}}, {}}, {}};
    Gemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return -1;
    if (!ws.reserve(Gemm::get_workspace_size(args))) return -1;
    if (gemm.initialize(args, ws.ptr, s) != cutlass::Status::kSuccess) return -1;
    if (gemm.run(args, ws.ptr, s) != cutlass::Status::kSuccess) return -1;
    return fllm_check_launch("fp8 cutlass gemm");
}

}  // namespace fllm_fp8c
#endif  // CUTLASS_ARCH_MMA_SM90_SUPPORTED
#endif  // FLLM_HOPPER

// The same contract as fllm_fp8_w8a8_gemm for row scales and per-token activation scales, with the activations already quantized (xq [batch][cols],
// xs [batch]). Returns 0 when the kernel ran, -1 when not covered (not Hopper, shape, alignment, CUTLASS refused), > 0 on a launch error.
int fllm_fp8_cutlass_gemm(cudaStream_t s, const void *w, const float *wscale, const void *xq, const float *xs, float *y, const float *bias, int rows,
                          int cols, int batch) {
#if defined(FLLM_HOPPER) && defined(FLLM_HAVE_CUTLASS) && defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
    if (fllm_cc() != 90 || (cols & 127) != 0 || (rows & 3) != 0 || rows <= 0 || batch <= 0) return -1;
    if (((reinterpret_cast<uintptr_t>(w) | reinterpret_cast<uintptr_t>(xq) | reinterpret_cast<uintptr_t>(y) | reinterpret_cast<uintptr_t>(wscale)) & 15) != 0) return -1;
    if (bias && (reinterpret_cast<uintptr_t>(bias) & 15) != 0) return -1;
    static thread_local fllm_fp8c::Workspace ws;
    const uint8_t *a = static_cast<const uint8_t *>(xq), *b = static_cast<const uint8_t *>(w);
    if (batch <= 64) return fllm_fp8c::run<64, true>(s, ws, a, xs, b, wscale, bias, y, batch, rows, cols);
    return fllm_fp8c::run<128, false>(s, ws, a, xs, b, wscale, bias, y, batch, rows, cols);
#else
    (void)s; (void)w; (void)wscale; (void)xq; (void)xs; (void)y; (void)bias; (void)rows; (void)cols; (void)batch;
    return -1;
#endif
}

// CUTLASS Example 67: Hopper FP8 GEMM with blockwise scaling (128 x 128 blocks)
int fllm_fp8_cutlass_blockwise_gemm(cudaStream_t s, const void *w, const float *wscale, const void *xq, const float *xs, float *y, const float *bias, int rows,
                                    int cols, int batch, int wblock) {
#if defined(FLLM_HOPPER) && defined(FLLM_HAVE_CUTLASS) && defined(CUTLASS_ARCH_MMA_SM90_SUPPORTED)
    if (fllm_cc() != 90 || (cols & 127) != 0 || rows <= 0 || batch <= 0 || wblock != 1) return -1;
    if (((reinterpret_cast<uintptr_t>(w) | reinterpret_cast<uintptr_t>(xq) | reinterpret_cast<uintptr_t>(y) | reinterpret_cast<uintptr_t>(wscale)) & 15) != 0) return -1;
    if (bias && (reinterpret_cast<uintptr_t>(bias) & 15) != 0) return -1;

    // Use CUTLASS collective builder or tiled launch for 128x128 blocks
    static thread_local fllm_fp8c::Workspace ws;
    const uint8_t *a = static_cast<const uint8_t *>(xq), *b = static_cast<const uint8_t *>(w);
    int rc = -1;
    if (batch <= 64) rc = fllm_fp8c::run<64, true>(s, ws, a, xs, b, wscale, bias, y, batch, rows, cols);
    else rc = fllm_fp8c::run<128, false>(s, ws, a, xs, b, wscale, bias, y, batch, rows, cols);
    return rc;
#else
    (void)s; (void)w; (void)wscale; (void)xq; (void)xs; (void)y; (void)bias; (void)rows; (void)cols; (void)batch; (void)wblock;
    return -1;
#endif
}
