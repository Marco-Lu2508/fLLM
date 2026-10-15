// fLLM CUDA backend: NVFP4 GEMM for large batches (prompt processing) on Blackwell GeForce (sm_120a) through CUTLASS.
//
// Derived from (the notices of these files apply to the corresponding parts, see THIRD_PARTY.md):
//   * vLLM, csrc/libtorch_stable/quantization/fp4/nvfp4_scaled_mm_sm120_kernels.cu
//     Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved. Licensed under the Apache License, Version 2.0
//     (http://www.apache.org/licenses/LICENSE-2.0): the kernel configurations (tile shapes 128x128x128 up to 256 tokens and
//     256x128x128 above, schedulers) and the way the CUTLASS arguments are put together (the epilogue factor is read from device
//     memory through `alpha_ptr`).
//   * CUTLASS, examples/79_blackwell_geforce_gemm/79a_blackwell_geforce_nvfp4_bf16_gemm.cu
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. SPDX-License-Identifier: BSD-3-Clause
//     (licenses/CUTLASS-BSD-3-Clause.txt): the collective builders of a block-scaled GEMM for SM120.
//   * FlashInfer, include/flashinfer/gemm/fp4_gemm_template_sm120.h, fp4_gemm_cutlass_template_sm120.h and the autotuner of
//     flashinfer/gemm/gemm_base.py (gen_gemm_sm120_module_cutlass_fp4)
//     Copyright (c) 2020-2023, NVIDIA CORPORATION. All rights reserved. Licensed under the Apache License, Version 2.0
//     (http://www.apache.org/licenses/LICENSE-2.0): the epilogue (no source tensor C when the product is only stored, an
//     explicit epilogue schedule), the tile shapes of its sm120 config list (128x128 and 256x128 / 128x256 with 128 or 256 values per
//     K tile, N tiles of 64), the choice between the persistent and the stream-K scheduler, and the idea of timing every candidate once
//     per problem shape and keeping the fastest.
//   * vLLM, csrc/quantization/fp4/nvfp4_quant_kernels.cu (scaled_fp4_quant), Apache-2.0: the per-tensor activation factor, see nvfp4_quant.cuh.
// Changes for fLLM: no PyTorch types (raw device pointers, fLLM's own stream and buffers), float output, activations are
// quantized and the scale factors are written in the swizzled layout by fLLM's kernels, weights keep fLLM's device layout.
//
// fLLM stores NVFP4 weights in the layout CUTLASS expects (row-major packed E2M1, scale factors in the swizzled atom
// layout), which the small-batch kernel in nvfp4.cu reads as well. Activations are quantized straight into the CUTLASS
// layout.
//
//   Y[token][row] = gscale * sum_k X[token][k] * W[row][k]
//
// Environment (all read once):
//   FLLM_NVFP4_TILE   unset = the tile of the original configuration (128x128x128 up to 256 tokens, 128x128x256 above), `auto` = the
//                     cost model in pick_tile (wave quantization), a tile name from kTiles (e.g. 128x128x256-sk) forces that tile.
//   FLLM_NVFP4_TUNE   1 = time every tile once per (rows, cols, batch / 128, result kind) at first use and keep the fastest in memory
//                     (not while a CUDA graph is being captured); FLLM_CUDA_TUNE_LOG=1 prints the timings.
//   FLLM_FP4_TILE     1..4, the older switch (256x128x128, 128x256x128, 128x128x128, 128x128x256).
//   FLLM_NVFP4_ACT_SCALE  off | static (default, = auto) | dynamic: per-tensor activation factor, see nvfp4_quant.cuh.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <tuple>
#include <type_traits>

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

// A tile configuration: CTA tile (M tokens x N rows x K values), the work scheduler and the mainloop schedule. All use one CTA per SM
// (the cluster is 1x1x1 on GeForce Blackwell).
template <int BM, int BN, int BK, class Sched, bool Pingpong>
struct Config {
    using TileScheduler = Sched;
    using MmaTileShape = Shape<Int<BM>, Int<BN>, Int<BK>>;
    using KernelSchedule = std::conditional_t<Pingpong, cutlass::gemm::KernelTmaWarpSpecializedPingpong,
                                              cutlass::gemm::KernelTmaWarpSpecializedCooperative>;
};
using Persistent = cutlass::gemm::PersistentScheduler;
using StreamK = cutlass::gemm::StreamKScheduler;

// The tiles, in the order of kTiles below.
using C256x128x128 = Config<256, 128, 128, Persistent, false>;
using C128x256x128 = Config<128, 256, 128, Persistent, false>;
using C128x128x128 = Config<128, 128, 128, Persistent, false>;
using C128x128x256 = Config<128, 128, 256, Persistent, false>;  // deeper K tile, two pipeline stages: the default above 256 tokens
using C128x128x128sk = Config<128, 128, 128, StreamK, false>;
using C128x128x256sk = Config<128, 128, 256, StreamK, false>;
using C128x128x128pp = Config<128, 128, 128, Persistent, true>;  // ping-pong: two consumer warp groups alternate tiles (the epilogue of one hides behind the mainloop of the other)
using C128x64x256 = Config<128, 64, 256, Persistent, false>;     // narrow tile for few output rows (k and v projections: 1024 rows)

enum Tile { kT256x128x128, kT128x256x128, kT128x128x128, kT128x128x256, kT128x128x128sk, kT128x128x256sk, kT128x128x128pp, kT128x64x256, kTileCount };
struct TileDesc {
    const char *name;
    int bm, bn;
    bool stream_k;
    double eff;  // guess of the relative per-FLOP speed of a full tile (largest = 1); only used by the `auto` cost model, FLLM_NVFP4_TUNE=1 replaces it by timings
};
constexpr TileDesc kTiles[kTileCount] = {
    {"256x128x128", 256, 128, false, 1.00},    {"128x256x128", 128, 256, false, 0.99},    {"128x128x128", 128, 128, false, 0.92},
    {"128x128x256", 128, 128, false, 0.97},    {"128x128x128-sk", 128, 128, true, 0.90}, {"128x128x256-sk", 128, 128, true, 0.94},
    {"128x128x128-pp", 128, 128, false, 0.94}, {"128x64x256", 128, 64, false, 0.84},
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
        typename Config::KernelSchedule>::CollectiveOp;
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop,
                                                            CollectiveEpilogue, typename Config::TileScheduler>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

// The scale factor layouts do not depend on the tile (checked at compile time in launch()); the activation quantization uses this one.
using RefGemm = Fp4Gemm<C128x128x256, false>::Gemm;
using RefSfConfig = typename RefGemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
using RefLayoutSFA = decltype(RefSfConfig::tile_atom_to_shape_SFA(make_shape(1, 1, 1, 1)));

// Per-tensor amax of the activations (FLLM_NVFP4_ACT_SCALE=dynamic) and the epilogue factor derived from it, in device memory so that no
// host synchronization is needed (and a CUDA graph can contain the kernels). One value per process and device: the GEMMs of one device run on one stream.
static __device__ float g_amax;
static __device__ float g_alpha;
__global__ void amax_reset_kernel() { g_amax = 0.f; }
// alpha = mult * amax / 2688 (dyn) or mult * (*src)
__global__ void alpha_kernel(float mult, const float *src, int dyn) { g_alpha = dyn ? mult / fp4_dynamic_gs(g_amax) : mult * *src; }
template <class Sym>
static float *device_symbol(Sym &sym, float *(&cache)[16]) {
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16) dev = 0;
    if (!cache[dev]) {
        void *p = nullptr;
        if (cudaGetSymbolAddress(&p, sym) != cudaSuccess) {
            cudaGetLastError();
            return nullptr;
        }
        cache[dev] = static_cast<float *>(p);
    }
    return cache[dev];
}
static float *alpha_symbol() {
    static float *cache[16] = {};
    return device_symbol(g_alpha, cache);
}

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
// gs_ptr: static activation factor (the weight's trailer), DYN: derived from g_amax, otherwise 1.
template <bool GLU, bool HALF_IN, bool DYN, class LayoutSF>
__global__ void quantize_activations_kernel(const void *__restrict__ x, const void *__restrict__ up, int mode,
                                            uint32_t *__restrict__ aq, uint8_t *__restrict__ asf, int cols, size_t blocks,
                                            LayoutSF lay, const float *__restrict__ gs_ptr) {
    const size_t bid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= blocks) return;
    const float gs = DYN ? fp4_dynamic_gs(g_amax) : (gs_ptr ? *gs_ptr : 1.f);
    float v[16];
    load16<HALF_IN>(x, bid, v);
    if constexpr (GLU) {
        float u[16];
        load16<HALF_IN>(up, bid, u);
#pragma unroll
        for (int i = 0; i < 16; i++) v[i] = fllm_glu(v[i], u[i], mode);
    }
    uint32_t words[2];
    const uint32_t sb = fp4_quantize_values(v, words, gs);
    aq[bid * 2] = words[0];
    aq[bid * 2 + 1] = words[1];
    const int per_row = cols / 16;
    asf[lay(make_coord((int)(bid / per_row), (int)(bid % per_row) * 16, 0))] = (uint8_t)sb;
}

// amax of the (GLU-combined) activations into g_amax (non-negative floats order like their bit patterns)
template <bool GLU, bool HALF_IN>
__global__ void amax_kernel(const void *__restrict__ x, const void *__restrict__ up, int mode, size_t blocks) {
    const size_t bid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    float m = 0.f;
    if (bid < blocks) {
        float v[16];
        load16<HALF_IN>(x, bid, v);
        if constexpr (GLU) {
            float u[16];
            load16<HALF_IN>(up, bid, u);
#pragma unroll
            for (int i = 0; i < 16; i++) v[i] = fllm_glu(v[i], u[i], mode);
        }
#pragma unroll
        for (int i = 0; i < 16; i++) m = fmaxf(m, fabsf(v[i]));
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0 && m > 0.f) atomicMax(reinterpret_cast<int *>(&g_amax), __float_as_int(m));
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
    DeviceBuffer aq, asf, workspace, save;
    // activations of the last quantization: skipped when the caller says they are unchanged
    const float *act_x = nullptr;
    int act_batch = 0, act_cols = 0;
    bool act_valid = false;
    std::map<std::tuple<int, int, int, int>, int> tuned;  // (rows, cols, batch / 128, kind) -> tile
};
thread_local State g_state;

// The epilogue factor: a host value, or a device pointer (static trailer / dynamic amax).
struct AlphaSpec {
    float host = 1.f;
    const float *dev = nullptr;
};

enum Kind { kF32 = 0, kAcc = 1, kF16 = 2 };

// One GEMM of tile configuration Config with the activations already quantized into st.aq / st.asf.
template <typename Config, bool Acc, typename Out>
int launch(cudaStream_t s, State &st, void *Y, const AlphaSpec &al, int M, int N, int K, const uint8_t *W) {
    using G = Fp4Gemm<Config, Acc, Out>;
    using Gemm = typename G::Gemm;
    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    const auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    const auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
    static_assert(std::is_same_v<std::remove_const_t<decltype(layout_SFA)>, RefLayoutSFA>, "all tiles share the scale factor layout of the activations");
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
    if (al.dev) args.epilogue.thread.alpha_ptr = al.dev;
    else args.epilogue.thread.alpha = al.host;
    if constexpr (Acc) {
        args.epilogue.ptr_C = reinterpret_cast<float const *>(Y);
        args.epilogue.thread.beta = 1.f;
    } else {
        args.epilogue.thread.beta = 0.f;
    }

    Gemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return 0;
    if (!st.workspace.reserve(Gemm::get_workspace_size(args))) return 0;
    if (gemm.initialize(args, st.workspace.ptr, s) != cutlass::Status::kSuccess) return 0;
    if (gemm.run(args, st.workspace.ptr, s) != cutlass::Status::kSuccess) return 0;
    return 1;
}

template <typename Config>
int launch_kind(int kind, cudaStream_t s, State &st, void *Y, const AlphaSpec &al, int M, int N, int K, const uint8_t *W) {
    if (kind == kAcc) return launch<Config, true, float>(s, st, Y, al, M, N, K, W);
    if (kind == kF16) return launch<Config, false, cutlass::half_t>(s, st, Y, al, M, N, K, W);
    return launch<Config, false, float>(s, st, Y, al, M, N, K, W);
}

int launch_tile(int tile, int kind, cudaStream_t s, State &st, void *Y, const AlphaSpec &al, int M, int N, int K, const uint8_t *W) {
    switch (tile) {
        case kT256x128x128: return launch_kind<C256x128x128>(kind, s, st, Y, al, M, N, K, W);
        case kT128x256x128: return launch_kind<C128x256x128>(kind, s, st, Y, al, M, N, K, W);
        case kT128x128x128: return launch_kind<C128x128x128>(kind, s, st, Y, al, M, N, K, W);
        case kT128x128x256: return launch_kind<C128x128x256>(kind, s, st, Y, al, M, N, K, W);
        case kT128x128x128sk: return launch_kind<C128x128x128sk>(kind, s, st, Y, al, M, N, K, W);
        case kT128x128x256sk: return launch_kind<C128x128x256sk>(kind, s, st, Y, al, M, N, K, W);
        case kT128x128x128pp: return launch_kind<C128x128x128pp>(kind, s, st, Y, al, M, N, K, W);
        case kT128x64x256: return launch_kind<C128x64x256>(kind, s, st, Y, al, M, N, K, W);
        default: return 0;
    }
}

// Quantizes M x K activations (FP32, or FP16 for a glu of halves / a pending FP16 attention output) into st.aq / st.asf with the factor of `amode`
// (0 off, 1 static through gs_ptr, 2 dynamic: amax first). Returns 0 when the buffers cannot be allocated.
int quantize(cudaStream_t s, State &st, const float *X, int M, int K, int amode, const float *gs_ptr) {
    const auto layout_SFA = RefSfConfig::tile_atom_to_shape_SFA(make_shape(M, 128, K, 1));
    const size_t asf_bytes = (size_t)cosize(layout_SFA);
    cudaMemsetAsync(st.asf.ptr, 0, asf_bytes, s);
    const size_t blocks = (size_t)M * (K / 16);
    const unsigned grid = (unsigned)((blocks + 127) / 128);
    uint32_t *aq = static_cast<uint32_t *>(st.aq.ptr);
    uint8_t *asf = static_cast<uint8_t *>(st.asf.ptr);
    const bool glu = fllm_glu_up != nullptr, half = (glu && fllm_glu_half) || fllm_x_half != nullptr;
    const void *src = fllm_x_half ? fllm_x_half : static_cast<const void *>(X);
    if (amode == 2) {
        amax_reset_kernel<<<1, 1, 0, s>>>();
        if (half && glu) amax_kernel<true, true><<<grid, 128, 0, s>>>(src, fllm_glu_up, fllm_glu_mode, blocks);
        else if (half) amax_kernel<false, true><<<grid, 128, 0, s>>>(src, nullptr, 0, blocks);
        else if (glu) amax_kernel<true, false><<<grid, 128, 0, s>>>(src, fllm_glu_up, fllm_glu_mode, blocks);
        else amax_kernel<false, false><<<grid, 128, 0, s>>>(src, nullptr, 0, blocks);
        if (half && glu) quantize_activations_kernel<true, true, true><<<grid, 128, 0, s>>>(src, fllm_glu_up, fllm_glu_mode, aq, asf, K, blocks, layout_SFA, nullptr);
        else if (half) quantize_activations_kernel<false, true, true><<<grid, 128, 0, s>>>(src, nullptr, 0, aq, asf, K, blocks, layout_SFA, nullptr);
        else if (glu) quantize_activations_kernel<true, false, true><<<grid, 128, 0, s>>>(src, fllm_glu_up, fllm_glu_mode, aq, asf, K, blocks, layout_SFA, nullptr);
        else quantize_activations_kernel<false, false, true><<<grid, 128, 0, s>>>(src, nullptr, 0, aq, asf, K, blocks, layout_SFA, nullptr);
        return 1;
    }
    if (half && glu) quantize_activations_kernel<true, true, false><<<grid, 128, 0, s>>>(src, fllm_glu_up, fllm_glu_mode, aq, asf, K, blocks, layout_SFA, gs_ptr);
    else if (half) quantize_activations_kernel<false, true, false><<<grid, 128, 0, s>>>(src, nullptr, 0, aq, asf, K, blocks, layout_SFA, gs_ptr);
    else if (glu) quantize_activations_kernel<true, false, false><<<grid, 128, 0, s>>>(src, fllm_glu_up, fllm_glu_mode, aq, asf, K, blocks, layout_SFA, gs_ptr);
    else quantize_activations_kernel<false, false, false><<<grid, 128, 0, s>>>(src, nullptr, 0, aq, asf, K, blocks, layout_SFA, gs_ptr);
    return 1;
}

int sm_count() {
    static const int sms = [] {
        int dev = 0, n = 0;
        cudaGetDevice(&dev);
        if (cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) != cudaSuccess || n <= 0) {
            cudaGetLastError();
            n = 132;
        }
        return n;
    }();
    return sms;
}

// Tile choice by a cost model: a persistent tile scheduler hands whole tiles to the SMs, so the time is (waves) x (cost of one tile) with
// waves = ceil(tiles / SMs); stream-K splits the last wave across SMs (cost: a fix-up of about 5 %). The per-FLOP efficiency of a tile shape
// (kTiles[].eff) is a guess, not a measurement. Ties go to the earlier entry of `order` (the 128x128x256 tile first, the original default).
int pick_tile(int M, int N) {
    static const int order[kTileCount] = {kT128x128x256, kT128x128x256sk, kT256x128x128, kT128x256x128, kT128x128x128, kT128x128x128sk, kT128x128x128pp, kT128x64x256};
    const int sms = sm_count();
    int best = kT128x128x256;
    double best_cost = 1e300;
    for (int i = 0; i < kTileCount; i++) {
        const TileDesc &d = kTiles[order[i]];
        const double tiles = (double)((M + d.bm - 1) / d.bm) * (double)((N + d.bn - 1) / d.bn);
        double waves = d.stream_k ? (tiles / sms) * 1.05 : (double)(((long)tiles + sms - 1) / sms);
        if (waves < 1.0) waves = 1.0;
        const double cost = waves * d.bm * d.bn / d.eff;
        if (cost < best_cost * 0.999) {
            best_cost = cost;
            best = order[i];
        }
    }
    return best;
}

uint32_t pow2_batch(int batch) {
    uint32_t mp2 = 16;
    while (mp2 < (uint32_t)batch) mp2 <<= 1;
    return mp2;
}

// FLLM_NVFP4_TILE: -2 = unset (original choice), -1 = auto, >= 0 = a tile of kTiles
int tile_env() {
    static const int v = [] {
        const char *e = std::getenv("FLLM_NVFP4_TILE");
        if (!e || !*e || !std::strcmp(e, "legacy")) return -2;
        if (!std::strcmp(e, "auto")) return -1;
        for (int i = 0; i < kTileCount; i++)
            if (!std::strcmp(e, kTiles[i].name)) return i;
        fprintf(stderr, "FLLM_NVFP4_TILE=%s is not a tile (auto, legacy", e);
        for (int i = 0; i < kTileCount; i++) fprintf(stderr, ", %s", kTiles[i].name);
        fprintf(stderr, "); using the original choice\n");
        return -2;
    }();
    return v;
}
// FLLM_FP4_TILE: 1 = 256x128x128, 2 = 128x256x128, 3 = 128x128x128, 4 = 128x128x256
int tile_override() {
    static const int tile = std::getenv("FLLM_FP4_TILE") ? std::atoi(std::getenv("FLLM_FP4_TILE")) : 0;
    return tile;
}
bool tune_enabled() {
    static const bool on = [] {
        const char *e = std::getenv("FLLM_NVFP4_TUNE");
        return e && *e && std::strcmp(e, "0") != 0;
    }();
    return on;
}

// FLLM_NVFP4_TUNE: times every tile (one warm-up, two timed runs) and keeps the fastest. An accumulating GEMM changes Y, so Y is saved first
// and restored after every run. The caller runs the winner once more for the real result.
int tune_tile(cudaStream_t s, State &st, int kind, void *Y, const AlphaSpec &al, int M, int N, int K, const uint8_t *W) {
    static const bool log = std::getenv("FLLM_CUDA_TUNE_LOG") != nullptr;
    const size_t y_bytes = (size_t)M * N * (kind == kF16 ? 2 : 4);
    const bool restore = kind == kAcc;
    if (restore) {
        if (!st.save.reserve(y_bytes)) return -1;
        cudaMemcpyAsync(st.save.ptr, Y, y_bytes, cudaMemcpyDeviceToDevice, s);
    }
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    int pick = -1;
    float best_ms = 1e30f;
    for (int t = 0; t < kTileCount; t++) {
        cudaGetLastError();
        if (!launch_tile(t, kind, s, st, Y, al, M, N, K, W) || cudaGetLastError() != cudaSuccess) continue;
        if (restore) cudaMemcpyAsync(Y, st.save.ptr, y_bytes, cudaMemcpyDeviceToDevice, s);
        cudaEventRecord(a, s);
        launch_tile(t, kind, s, st, Y, al, M, N, K, W);
        launch_tile(t, kind, s, st, Y, al, M, N, K, W);
        cudaEventRecord(b, s);
        cudaEventSynchronize(b);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        if (restore) cudaMemcpyAsync(Y, st.save.ptr, y_bytes, cudaMemcpyDeviceToDevice, s);
        if (log) fprintf(stderr, "[nvfp4 tune %dx%dx%d kind %d] %-15s %.1f us\n", M, N, K, kind, kTiles[t].name, ms * 500.f);
        if (ms < best_ms) {
            best_ms = ms;
            pick = t;
        }
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    cudaGetLastError();
    if (log && pick >= 0) fprintf(stderr, "[nvfp4 tune %dx%dx%d kind %d] best %s\n", M, N, K, kind, kTiles[pick].name);
    return pick;
}

// Tile of one call. Original choice: the 128x128x256 tile for results that are added or leave FP16 and for batches above 256 tokens,
// 128x128x128 otherwise.
int choose_tile(cudaStream_t s, State &st, int kind, void *Y, const AlphaSpec &al, int M, int N, int K, const uint8_t *W) {
    const int env = tile_env();
    if (env >= 0) return env;
    if (env == -1 || tune_enabled()) {
        if (tune_enabled()) {
            cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
            if (cudaStreamIsCapturing(s, &cs) == cudaSuccess && cs == cudaStreamCaptureStatusNone) {
                const auto key = std::make_tuple(N, K, (M + 127) / 128, (int)kind);
                auto it = st.tuned.find(key);
                if (it != st.tuned.end()) return it->second;
                const int t = tune_tile(s, st, kind, Y, al, M, N, K, W);
                if (t >= 0) {
                    st.tuned[key] = t;
                    return t;
                }
            } else {
                cudaGetLastError();
            }
        }
        if (env == -1) return pick_tile(M, N);
    }
    const int forced = tile_override();
    if (kind != kF32) return kT128x128x256;
    if (pow2_batch(M) <= 256 && forced == 0) return kT128x128x128;
    switch (forced) {
        case 1: return kT256x128x128;
        case 2: return kT128x256x128;
        case 3: return kT128x128x128;
        default: return kT128x128x256;
    }
}


// ---- RMSNorm fused with the activation quantization (FLLM_FUSE_NVFP4_NORM) ----
// The row is normalized exactly as rms_norm_vec_kernel in elementwise.cu does (same reduction order, so `out` is bit-identical) and written to
// `out` (other consumers of the normalized activations stay valid), kept in shared memory, and then quantized in 16-value blocks, one per thread,
// straight into the CUTLASS layout: the separate quantization kernel and its re-read of the FP32 rows disappear.
__device__ __forceinline__ float norm_block_sum(float v, float *scratch) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float total = 0.f;
    for (int i = 0; i < warps; i++) total += scratch[i];
    return total;
}
template <class LayoutSF>
__global__ void __launch_bounds__(256)
rms_norm_quant_kernel(const float *__restrict__ x, const float *__restrict__ w, float *__restrict__ out, int cols, float eps, float inv_cols,
                      uint32_t *__restrict__ aq, uint8_t *__restrict__ asf, LayoutSF lay, const float *__restrict__ gs_ptr) {
    extern __shared__ __align__(16) float row[];
    __shared__ float scratch[32];
    const int cols4 = cols >> 2;
    const float4 *xr = reinterpret_cast<const float4 *>(x) + (size_t)blockIdx.x * cols4;
    float4 *orow = reinterpret_cast<float4 *>(out) + (size_t)blockIdx.x * cols4;
    const float4 *w4 = reinterpret_cast<const float4 *>(w);
    float4 *srow = reinterpret_cast<float4 *>(row);
    float sum = 0.f;
    for (int c = threadIdx.x; c < cols4; c += 256) {
        const float4 v = xr[c];
        sum = fmaf(v.x, v.x, fmaf(v.y, v.y, fmaf(v.z, v.z, fmaf(v.w, v.w, sum))));
    }
    const float total = norm_block_sum(sum, scratch);
    const float inv = 1.0f / sqrtf(total * inv_cols + eps);
    for (int c = threadIdx.x; c < cols4; c += 256) {
        const float4 v = xr[c], g = __ldg(w4 + c);
        const float4 o = make_float4(v.x * (inv * g.x), v.y * (inv * g.y), v.z * (inv * g.z), v.w * (inv * g.w));
        orow[c] = o;
        srow[c] = o;
    }
    __syncthreads();
    const float gs = gs_ptr ? *gs_ptr : 1.f;
    const int per_row = cols >> 4;
    for (int b = threadIdx.x; b < per_row; b += 256) {
        float v[16];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float4 t = srow[b * 4 + i];
            v[4 * i] = t.x; v[4 * i + 1] = t.y; v[4 * i + 2] = t.z; v[4 * i + 3] = t.w;
        }
        uint32_t words[2];
        const uint32_t sb = fp4_quantize_values(v, words, gs);
        const size_t bid = (size_t)blockIdx.x * per_row + b;
        aq[bid * 2] = words[0];
        aq[bid * 2 + 1] = words[1];
        asf[lay(make_coord((int)blockIdx.x, b * 16, 0))] = (uint8_t)sb;
    }
}

}  // namespace fllm_fp4c
#endif  // CUTLASS_ARCH_MMA_SM120_SUPPORTED

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
static bool cutlass_disabled() {
    static const bool disabled = std::getenv("FLLM_CUDA_NO_CUTLASS") != nullptr;
    return disabled;
}
#endif

// Whether fllm_nvfp4_cutlass_try takes this product. mode: bit 0 = the product has to be added to Y, bit 1 = the result is FP16;
// both exist for batches above 256 only (and, with FLLM_FP4_TILE, only for the 128x128x256 tile).
int fllm_nvfp4_cutlass_check(const float *X, const float *Y, const float *bias, int rows, int cols, int batch, int mode) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    if (cutlass_disabled() || bias != nullptr || batch <= 64 || (rows & 31) != 0 || (cols & 255) != 0) return 0;
    if ((reinterpret_cast<uintptr_t>(X) & 15) != 0 || (reinterpret_cast<uintptr_t>(Y) & 15) != 0) return 0;
    if (mode != 0) {
        const int tile = fllm_fp4c::tile_override();
        if (fllm_fp4c::pow2_batch(batch) <= 256 || (tile != 0 && tile != 4)) return 0;
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
    using namespace fllm_fp4c;
    const bool acc = acc_scale != 0.f;
    if (acc && out_half) return 0;
    if (!fllm_nvfp4_cutlass_check(X, static_cast<const float *>(Y), bias, rows, cols, batch, (acc ? 1 : 0) | (out_half ? 2 : 0))) return 0;
    const uint8_t *w = static_cast<const uint8_t *>(W);
    State &st = g_state;
    const int M = batch, N = rows, K = cols;
    const int kind = acc ? kAcc : (out_half ? kF16 : kF32);

    // The weights were uploaded in this layout (backend/cuda.rs, nvfp4_device_layout); check the scale index there
    // against the one CUTLASS derives for this problem.
    {
        const auto layout_SFB = RefSfConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
        for (int i = 0; i < 64; i++) {
            const int n = (i * 7919) % N, kb = (i * 104729) % (K / 16);
            const size_t mine = (size_t)(n / 128) * (K / 64) * 512 + (size_t)(kb / 4) * 512 + (n % 32) * 16 + (n / 32 % 4) * 4 + kb % 4;
            if ((size_t)layout_SFB(make_coord(n, kb * 16, 0)) != mine) return 0;
        }
    }
    const auto layout_SFA = RefSfConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    const size_t asf_bytes = (size_t)cosize(layout_SFA);
    const size_t aq_bytes = (size_t)M * K / 2;
    if (!st.aq.reserve(aq_bytes) || !st.asf.reserve(asf_bytes)) return 0;

    // activation factor: static = the input_scale in the weight's trailer, dynamic = amax of the activations
    const int amode = fllm_nvfp4_act_scale_mode();
    const Nvfp4Trailer *tr = reinterpret_cast<const Nvfp4Trailer *>(w + fllm_nvfp4_trailer_offset(rows, cols));
    if (!(quantize == 0 && st.act_valid && st.act_x == X && st.act_batch == M && st.act_cols == K)) {
        if (!fllm_fp4c::quantize(s, st, X, M, K, amode, amode == 1 ? &tr->gs : nullptr)) return 0;
        st.act_x = X;
        st.act_batch = M;
        st.act_cols = K;
        st.act_valid = true;
    }
    // epilogue factor
    const float mult = acc ? acc_scale : 1.f;
    AlphaSpec al;
    if (amode == 0) {
        al.host = gscale * mult;
    } else if (amode == 1 && mult == 1.f) {
        al.dev = &tr->alpha;
    } else {
        float *sym = alpha_symbol();
        if (!sym) return 0;
        alpha_kernel<<<1, 1, 0, s>>>(amode == 2 ? gscale * mult : mult, &tr->alpha, amode == 2);
        al.dev = sym;
    }
    const int tile = choose_tile(s, st, kind, Y, al, M, N, K, w);
    return launch_tile(tile, kind, s, st, Y, al, M, N, K, w);
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

// RMSNorm of `rows` rows of `cols` values fused with the quantization of its result for the CUTLASS GEMM of `weight` (weight_rows x cols,
// whose input_scale supplies the activation factor). `out` receives the normalized FP32 rows as the plain norm kernel would write them.
// Returns 1 when done: the quantized activations then stay in the CUTLASS state, marked as those of `out`, so that the GEMM calls that read
// `out` skip their own quantization. Returns 0 (nothing launched) when this is not possible; the caller runs the plain norm kernel.
int fllm_nvfp4_norm_quant(cudaStream_t s, const float *x, const float *w, float *out, int rows, int cols, float eps, const void *weight,
                          int weight_rows, int mode) {
#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED)
    using namespace fllm_fp4c;
    const int amode = fllm_nvfp4_act_scale_mode();
    if (amode == 2 || cols % 256 != 0 || cols > 8192 || rows <= 64) return 0;
    if (((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(w) | reinterpret_cast<uintptr_t>(out)) & 15) != 0) return 0;
    if (!fllm_nvfp4_cutlass_check(out, out, nullptr, weight_rows, cols, rows, mode)) return 0;
    State &st = g_state;
    const auto layout_SFA = RefSfConfig::tile_atom_to_shape_SFA(make_shape(rows, 128, cols, 1));
    const size_t asf_bytes = (size_t)cosize(layout_SFA);
    if (!st.aq.reserve((size_t)rows * cols / 2) || !st.asf.reserve(asf_bytes)) return 0;
    const Nvfp4Trailer *tr = reinterpret_cast<const Nvfp4Trailer *>(static_cast<const uint8_t *>(weight) + fllm_nvfp4_trailer_offset(weight_rows, cols));
    cudaMemsetAsync(st.asf.ptr, 0, asf_bytes, s);
    rms_norm_quant_kernel<<<rows, 256, (size_t)cols * sizeof(float), s>>>(x, w, out, cols, eps, 1.0f / (float)cols, static_cast<uint32_t *>(st.aq.ptr),
                                                                          static_cast<uint8_t *>(st.asf.ptr), layout_SFA, amode == 1 ? &tr->gs : nullptr);
    st.act_x = out;
    st.act_batch = rows;
    st.act_cols = cols;
    st.act_valid = true;
    return fllm_check_launch("rms_norm+quantize") == 0 ? 1 : 0;
#else
    (void)s; (void)x; (void)w; (void)out; (void)rows; (void)cols; (void)eps; (void)weight; (void)weight_rows; (void)mode;
    return 0;
#endif
}
