// fLLM CUDA backend: NVFP4 GEMM for large batches (prompt processing) on Blackwell GeForce (sm_120a) through CUTLASS.
//
// Derived from (the notices of these files apply to the corresponding parts, see THIRD_PARTY.md):
//   * vLLM, csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu
//     Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved. Licensed under the Apache License, Version 2.0
//     (http://www.apache.org/licenses/LICENSE-2.0): the kernel configurations (tile shapes 128x128x128 up to 256 tokens and
//     256x128x128 above, schedulers) and the way the CUTLASS arguments are put together.
//   * CUTLASS, examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. SPDX-License-Identifier: BSD-3-Clause
//     (licenses/CUTLASS-BSD-3-Clause.txt): the collective builders of a block-scaled GEMM for SM120.
//   * FlashInfer, include/flashinfer/gemm/fp4_gemm_template_sm120.h and fp4_gemm_cutlass_template_sm120.h
//     Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved. Licensed under the Apache License, Version 2.0
//     (http://www.apache.org/licenses/LICENSE-2.0): the epilogue (no source tensor C when the product is only stored, an
//     explicit epilogue schedule) and the tile shapes of its sm120 config list (128x128x128, 128x128x256, 256x128x128).
// Changes for fLLM: no PyTorch types (raw device pointers, fLLM's own stream and buffers), float output, activations are
// quantized and the scale factors are written in the swizzled layout by fLLM's kernels, weights keep fLLM's device layout.
//
// fLLM stores NVFP4 weights in the layout CUTLASS expects (row-major packed E2M1, scale factors in the swizzled atom
// layout), which the small-batch kernel in nvfp4.cu reads as well. Activations are quantized straight into the CUTLASS
// layout.
//
//   Y[token][row] = gscale * sum_k X[token][k] * W[row][k]
#include <cstdint>
#include <cstdlib>

#include "common.cuh"
#include "fllm_cuda_internal.h"
#include "nvfp4_quant.cuh"

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
// (named namespace: nvcc cannot mix an anonymous one with the cute headers when it generates the kernel stubs)
namespace fllm_fp4c {
using namespace cute;

struct ConfigSmall {  // up to 256 tokens
    using TileScheduler = void;
    using MmaTileShape = Shape<_128, _128, _128>;
};
struct ConfigLarge {
    using TileScheduler = cutlass::gemm::PersistentScheduler;
    using MmaTileShape = Shape<_256, _128, _128>;
};

// Tile shapes tried: 256x128, 128x256 and 128x128 (the shapes of FlashInfer's sm120 NVFP4 config list, include/flashinfer/gemm/fp4_gemm_cutlass_template_sm120.h).
// FLLM_FP4_TILE=1/2/3/4 forces one for tests; 128x128 persistent is the default above 256 tokens (RTX PRO 3000 Blackwell, Llama-3.1-8B NVFP4:
// prefill 9021 -> 10678 tok/s at 355 and 10724 -> 12054 at 1863 prompt tokens), with the deeper K tile of the 128x128x256 variant (below) since then.
struct ConfigWide {
    using TileScheduler = cutlass::gemm::PersistentScheduler;
    using MmaTileShape = Shape<_128, _256, _128>;
};
struct ConfigSquare {
    using TileScheduler = cutlass::gemm::PersistentScheduler;
    using MmaTileShape = Shape<_128, _128, _128>;
};
// 128x128x256 (deeper K tile, two pipeline stages): FLLM_FP4_TILE=4
struct ConfigDeep {
    using TileScheduler = cutlass::gemm::PersistentScheduler;
    using MmaTileShape = Shape<_128, _128, _256>;
};

// Acc: the product is added to the FP32 result buffer (Y = acc_scale * gscale * X W^T + Y: the residual stream of the
// output projections) instead of overwriting it; the epilogue then reads Y as the source tensor C. Without it there is no
// source tensor (its shared memory goes to the mainloop pipeline).
// Out: element type of the result (float, or half for the q/k/v and gate/up projections whose consumers read FP16; never with Acc).
template <typename Config, bool Acc, typename Out = float>
struct Fp4Gemm {
    using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using LayoutATag = cutlass::layout::RowMajor;
    static constexpr int AlignmentA = 32;
    using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
    using LayoutBTag = cutlass::layout::ColumnMajor;
    static constexpr int AlignmentB = 32;
    using ElementD = Out;
    using ElementC = std::conditional_t<Acc, float, void>;
    using LayoutCTag = cutlass::layout::RowMajor;
    using LayoutDTag = cutlass::layout::RowMajor;
    static constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;
    static constexpr int AlignmentC = AlignmentD;
    using ElementAccumulator = float;
    using ArchTag = cutlass::arch::Sm120;
    using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
    using MmaTileShape = typename Config::MmaTileShape;
    using ClusterShape = Shape<_1, _1, _1>;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, cutlass::arch::OpClassTensorOp, MmaTileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto,
        ElementAccumulator, ElementAccumulator, ElementC, LayoutCTag, AlignmentC, ElementD, LayoutDTag, AlignmentD,
        cutlass::epilogue::TmaWarpSpecialized,
        cutlass::epilogue::fusion::LinearCombination<ElementD, float, ElementC, float>>::CollectiveOp;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB, LayoutBTag, AlignmentB, ElementAccumulator,
        MmaTileShape, ClusterShape,
        cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::KernelTmaWarpSpecializedCooperative>::CollectiveOp;
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop,
                                                            CollectiveEpilogue, typename Config::TileScheduler>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// 16 activations of one thread as floats: FP32 in memory, or FP16 (the results of the half-precision GEMMs).
template <bool HALF_IN>
__device__ __forceinline__ void load16(const void *base, size_t bid, float (&v)[16]) {
    if constexpr (HALF_IN) {
        const uint4 *p = reinterpret_cast<const uint4 *>(base) + bid * 2;
#pragma unroll
        for (int i = 0; i < 2; i++) {
            const uint4 t = __ldg(p + i);
            const __half2 *h = reinterpret_cast<const __half2 *>(&t);
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const float2 f = __half22float2(h[j]);
                v[8 * i + 2 * j] = f.x;
                v[8 * i + 2 * j + 1] = f.y;
            }
        }
    } else {
        const float4 *p = reinterpret_cast<const float4 *>(base) + bid * 4;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float4 t = __ldg(p + i);
            v[4 * i] = t.x; v[4 * i + 1] = t.y; v[4 * i + 2] = t.z; v[4 * i + 3] = t.w;
        }
    }
}

// One thread per 16 activations: packed E2M1 in natural order, scale in the swizzled layout. With GLU the gate values
// in `x` are combined with `up` first (activation fusion, see fllm_glu_up). HALF_IN: x and up are FP16.
template <bool GLU, bool HALF_IN, class LayoutSF>
__global__ void quantize_activations_kernel(const void *__restrict__ x, const void *__restrict__ up, int mode,
                                            uint32_t *__restrict__ aq, uint8_t *__restrict__ asf, int cols, size_t blocks,
                                            LayoutSF lay) {
    const size_t bid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= blocks) return;
    float v[16];
    load16<HALF_IN>(x, bid, v);
    if constexpr (GLU) {
        float u[16];
        load16<HALF_IN>(up, bid, u);
#pragma unroll
        for (int i = 0; i < 16; i++) v[i] = fllm_glu(v[i], u[i], mode);
    }
    uint32_t words[2];
    const uint32_t sb = fp4_quantize_values(v, words);
    aq[bid * 2] = words[0];
    aq[bid * 2 + 1] = words[1];
    const int per_row = cols / 16;
    asf[lay(make_coord((int)(bid / per_row), (int)(bid % per_row) * 16, 0))] = (uint8_t)sb;
}
struct DeviceBuffer {
    void *ptr = nullptr;
    size_t bytes = 0;
    bool reserve(size_t need) {
        if (need <= bytes) return true;
        if (ptr) cudaFree(ptr);
        ptr = nullptr;
        bytes = 0;
        if (cudaMalloc(&ptr, need) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        bytes = need;
        return true;
    }
};

struct State {
    DeviceBuffer aq, asf, workspace;
    // activations of the last quantization: skipped when the caller says they are unchanged
    const float *act_x = nullptr;
    int act_batch = 0, act_cols = 0;
    bool act_valid = false;
};
thread_local State g_state;

template <typename Config, bool Acc = false, typename Out = float>
int run_gemm(cudaStream_t s, State &st, void *Y, float gscale, int M, int N, int K, const float *X, const uint8_t *W,
             int quantize, float acc_scale = 0.f) {
    using G = Fp4Gemm<Config, Acc, Out>;
    using Gemm = typename G::Gemm;
    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    const auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    const auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
    // The weights were uploaded in this layout (backend/cuda.rs, nvfp4_device_layout); check the scale index there
    // against the one CUTLASS derives for this problem.
    for (int i = 0; i < 64; i++) {
        const int n = (i * 7919) % N, kb = (i * 104729) % (K / 16);
        const size_t mine = (size_t)(n / 128) * (K / 64) * 512 + (size_t)(kb / 4) * 512 + (n % 32) * 16 + (n / 32 % 4) * 4 + kb % 4;
        if ((size_t)layout_SFB(make_coord(n, kb * 16, 0)) != mine) return 0;
    }
    const size_t asf_bytes = (size_t)cosize(layout_SFA);
    const size_t aq_bytes = (size_t)M * K / 2;
    if (!st.aq.reserve(aq_bytes) || !st.asf.reserve(asf_bytes)) return 0;

    if (!(quantize == 0 && st.act_valid && st.act_x == X && st.act_batch == M && st.act_cols == K)) {
        cudaMemsetAsync(st.asf.ptr, 0, asf_bytes, s);
        const size_t blocks = (size_t)M * (K / 16);
        const unsigned grid = (unsigned)((blocks + 127) / 128);
        uint32_t *aq = static_cast<uint32_t *>(st.aq.ptr);
        uint8_t *asf = static_cast<uint8_t *>(st.asf.ptr);
        if (fllm_glu_up && fllm_glu_half)
            quantize_activations_kernel<true, true><<<grid, 128, 0, s>>>(X, fllm_glu_up, fllm_glu_mode, aq, asf, K, blocks, layout_SFA);
        else if (fllm_glu_up)
            quantize_activations_kernel<true, false><<<grid, 128, 0, s>>>(X, fllm_glu_up, fllm_glu_mode, aq, asf, K, blocks, layout_SFA);
        else
            quantize_activations_kernel<false, false><<<grid, 128, 0, s>>>(X, nullptr, 0, aq, asf, K, blocks, layout_SFA);
        st.act_x = X;
        st.act_batch = M;
        st.act_cols = K;
        st.act_valid = true;
    }
    const uint8_t *Bq = W;
    const uint8_t *Bsf = W + (((size_t)N * (K / 2) + 255) & ~size_t(255));

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {M, K, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {N, K, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {M, N, 1});
    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},
        {static_cast<typename G::ElementA::DataType const *>(st.aq.ptr), stride_A,
         reinterpret_cast<typename G::ElementB::DataType const *>(Bq), stride_B,
         static_cast<typename G::ElementA::ScaleFactorType const *>(st.asf.ptr), layout_SFA,
         reinterpret_cast<typename G::ElementB::ScaleFactorType const *>(Bsf), layout_SFB},
        {{}, nullptr, stride_D, reinterpret_cast<typename G::ElementD *>(Y), stride_D}};
    if constexpr (Acc) {
        args.epilogue.ptr_C = reinterpret_cast<float const *>(Y);
        args.epilogue.thread.alpha = gscale * acc_scale;
        args.epilogue.thread.beta = 1.f;
    } else {
        args.epilogue.thread.alpha = gscale;
        args.epilogue.thread.beta = 0.f;
    }

    Gemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return 0;
    if (!st.workspace.reserve(Gemm::get_workspace_size(args))) return 0;
    if (gemm.initialize(args, st.workspace.ptr, s) != cutlass::Status::kSuccess) return 0;
    if (gemm.run(args, st.workspace.ptr, s) != cutlass::Status::kSuccess) return 0;
    return 1;
}

}  // namespace fllm_fp4c
#endif  // CUTLASS_ARCH_MMA_SM120_SUPPORTED

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
static bool cutlass_disabled() {
    static const bool disabled = std::getenv("FLLM_CUDA_NO_CUTLASS") != nullptr;
    return disabled;
}
// FLLM_FP4_TILE: 1 = 256x128x128, 2 = 128x256x128, 3 = 128x128x128, 4 = 128x128x256 (the default above 256 tokens); the first three
// only exist for FP32 results that overwrite Y.
static int tile_override() {
    static const int tile = std::getenv("FLLM_FP4_TILE") ? std::atoi(std::getenv("FLLM_FP4_TILE")) : 0;
    return tile;
}
static uint32_t pow2_batch(int batch) {
    uint32_t mp2 = 16;
    while (mp2 < (uint32_t)batch) mp2 <<= 1;
    return mp2;
}
#endif

// Whether fllm_nvfp4_cutlass_try takes this product. mode: bit 0 = the product has to be added to Y, bit 1 = the result is FP16;
// both exist for the 128x128x256 tile and batches above 256 only.
int fllm_nvfp4_cutlass_check(const float *X, const float *Y, const float *bias, int rows, int cols, int batch, int mode) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    if (cutlass_disabled() || bias != nullptr || batch <= 64 || (rows & 31) != 0 || (cols & 255) != 0) return 0;
    if ((reinterpret_cast<uintptr_t>(X) & 15) != 0 || (reinterpret_cast<uintptr_t>(Y) & 15) != 0) return 0;
    if (mode != 0) {
        const int tile = tile_override();
        if (pow2_batch(batch) <= 256 || (tile != 0 && tile != 4)) return 0;
    }
    return 1;
#else
    return 0;
#endif
}

// Returns 1 when the product was computed, 0 when the caller has to use the built-in kernel (unsupported shape,
// no memory for the conversion buffers, CUTLASS refused the problem). With acc_scale != 0 the product is multiplied by it
// and added to Y; with out_half Y receives FP16 results.
int fllm_nvfp4_cutlass_try(cudaStream_t s, const void *W, const float *X, void *Y, const float *bias, float gscale, int rows,
                           int cols, int batch, int quantize, float acc_scale, int out_half) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    const bool acc = acc_scale != 0.f;
    if (acc && out_half) return 0;
    if (!fllm_nvfp4_cutlass_check(X, static_cast<const float *>(Y), bias, rows, cols, batch, (acc ? 1 : 0) | (out_half ? 2 : 0))) return 0;
    using namespace fllm_fp4c;
    const uint8_t *w = static_cast<const uint8_t *>(W);
    State &st = g_state;
    const int tile = tile_override();
    if (acc) return run_gemm<ConfigDeep, true>(s, st, Y, gscale, batch, rows, cols, X, w, quantize, acc_scale);
    if (out_half) return run_gemm<ConfigDeep, false, cutlass::half_t>(s, st, Y, gscale, batch, rows, cols, X, w, quantize);
    if (pow2_batch(batch) <= 256 && tile == 0) return run_gemm<ConfigSmall>(s, st, Y, gscale, batch, rows, cols, X, w, quantize);
    if (tile == 1) return run_gemm<ConfigLarge>(s, st, Y, gscale, batch, rows, cols, X, w, quantize);
    if (tile == 2) return run_gemm<ConfigWide>(s, st, Y, gscale, batch, rows, cols, X, w, quantize);
    if (tile == 3) return run_gemm<ConfigSquare>(s, st, Y, gscale, batch, rows, cols, X, w, quantize);
    return run_gemm<ConfigDeep>(s, st, Y, gscale, batch, rows, cols, X, w, quantize);
#else
    return 0;
#endif
}

// The built-in kernel quantized new activations into the scratch buffer: the CUTLASS copy is stale.
void fllm_nvfp4_cutlass_invalidate() {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    fllm_fp4c::g_state.act_valid = false;
#endif
}