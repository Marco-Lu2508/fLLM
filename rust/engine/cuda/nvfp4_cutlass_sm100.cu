// fLLM CUDA backend: NVFP4 GEMM for datacenter Blackwell (sm_100a: B200 / GB200, sm_103a: B300 / GB300, sm_110a: Thor) through CUTLASS.
//
// These GPUs have no `mma.sync` FP4 instruction (that exists on GeForce Blackwell, sm_120a, and is used by nvfp4.cu): their tensor cores are
// driven by tcgen05 / UMMA with the accumulator in tensor memory, which only CUTLASS' Sm100 collectives (TMA, warp specialization, 1 or 2
// SMs per MMA) express. This file is the Sm100 counterpart of nvfp4_cutlass.cu; the dispatch by compute capability is in nvfp4.cu
// (fllm_nvfp4_try): 12.x uses nvfp4.cu / nvfp4_cutlass.cu, 10.x and 11.0 this file for every batch size, everything else keeps the Marlin FP4
// weights (W4A16) that backend/cuda.rs uploads when no native path is available.
//
// STATUS: compile-only (test level L0). It builds for sm_100a / sm_103a / sm_110a with CUDA 12.8 (sm_100a) and 13.0 (all three); no device with
// these architectures was available. It is off unless FLLM_NVFP4_SM100=1.
//
// Derived from (the notices of these files apply to the corresponding parts, see THIRD_PARTY.md):
//   * CUTLASS, examples/72_blackwell_narrow_precision_gemm/72a_blackwell_nvfp4_bf16_gemm.cu and 72b_blackwell_nvfp4_nvfp4_gemm.cu
//     Copyright (c) 2025 - 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved. SPDX-License-Identifier: BSD-3-Clause
//     (licenses/CUTLASS-BSD-3-Clause.txt): the collective builders of a block-scaled NVFP4 GEMM for SM100 (tile shapes, cluster shapes,
//     kernel schedule auto-selection of the 1 SM / 2 SM variants).
//   * vLLM, csrc/quantization/fp4/nvfp4_scaled_mm_sm100_kernels.cu, Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved. Licensed
//     under the Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0): the choice of the tile by the number of tokens
//     (128 x 128 x 256 for small M, 256 x 128 x 256 with a 2 x 1 cluster for large M, 128 x 256 x 256 for few output rows) and the way the
//     arguments are put together (the epilogue factor is read from device memory through `alpha_ptr`).
//   * FlashInfer, include/flashinfer/gemm/fp4_gemm_template_sm100.h, Copyright (c) 2020-2023, NVIDIA CORPORATION. Licensed under the
//     Apache License, Version 2.0: the epilogue without a source tensor and the list of candidate tiles.
//   * vLLM, csrc/quantization/fp4/nvfp4_quant_kernels.cu (scaled_fp4_quant), Apache-2.0: the per-tensor activation factor, see nvfp4_quant.cuh.
// Changes for fLLM: no PyTorch types (raw device pointers, fLLM's own stream and buffers), FP32 activations quantized by fLLM's kernels
// straight into the swizzled scale layout, FP32 results, the weights keep fLLM's device layout (identical to the one the sm_120 path reads:
// row-major packed E2M1, scale factors in the Sm1xx atom layout, then the 16-byte trailer, see nvfp4_device_layout in backend/cuda.rs).
//
//   Y[token][row] = gscale * sum_k X[token][k] * W[row][k]
//
// Environment (all read once):
//   FLLM_NVFP4_SM100        1 = use this path on a datacenter Blackwell GPU (default: off, the Marlin FP4 weights are used)
//   FLLM_NVFP4_SM100_TILE   force a tile: 128x128x256, 128x256x256, 256x128x256, 256x256x256 (default: chosen by the token count)
//   FLLM_NVFP4_ACT_SCALE    off | static (default) | dynamic: per-tensor activation factor, see nvfp4_quant.cuh.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <tuple>
#include <type_traits>

#include "common.cuh"
#include "fllm_cuda_internal.h"
#include "hopper_util.h"
#include "nvfp4_quant.cuh"

#if defined(FLLM_SM100)
#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/util/packed_stride.hpp"
#endif

#if defined(FLLM_SM100) && defined(CUTLASS_ARCH_MMA_SM100_SUPPORTED)
#define FLLM_SM100_ACTIVE 1

// (named namespace: nvcc cannot mix an anonymous one with the cute headers when it generates the kernel stubs)
namespace fllm_fp4s100 {
using namespace cute;

// CTA tile (M tokens x N rows x K values) and cluster along M. A tile of 256 tokens runs one MMA over two SMs (cluster 2 x 1 x 1); the kernel
// schedule (1 SM / 2 SM) follows from the tile (KernelScheduleAuto, as in the CUTLASS examples).
template <int BM, int BN, int BK, int CM>
struct Config {
    using MmaTileShape = Shape<Int<BM>, Int<BN>, Int<BK>>;
    using ClusterShape = Shape<Int<CM>, _1, _1>;
};
using C128x128x256 = Config<128, 128, 256, 1>;
using C128x256x256 = Config<128, 256, 256, 1>;
using C256x128x256 = Config<256, 128, 256, 2>;
using C256x256x256 = Config<256, 256, 256, 2>;
enum Tile { kT128x128x256, kT128x256x256, kT256x128x256, kT256x256x256, kTileCount };
constexpr const char *kTileNames[kTileCount] = {"128x128x256", "128x256x256", "256x128x256", "256x256x256"};

enum ResultKind { kF32 = 0, kAcc = 1, kF16 = 2 };

template <typename Config, bool Acc = false, typename Out = float>
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
    using ArchTag = cutlass::arch::Sm100;
    using OperatorClass = cutlass::arch::OpClassBlockScaledTensorOp;
    using MmaTileShape = typename Config::MmaTileShape;
    using ClusterShape = typename Config::ClusterShape;

    using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
        ArchTag, OperatorClass, MmaTileShape, ClusterShape, cutlass::epilogue::collective::EpilogueTileAuto, ElementAccumulator,
        ElementAccumulator, ElementC, LayoutCTag, AlignmentC, ElementD, LayoutDTag, AlignmentD,
        cutlass::epilogue::collective::EpilogueScheduleAuto,
        cutlass::epilogue::fusion::LinearCombination<ElementD, float, ElementC, float>>::CollectiveOp;
    using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
        ArchTag, OperatorClass, ElementA, LayoutATag, AlignmentA, ElementB, LayoutBTag, AlignmentB, ElementAccumulator, MmaTileShape,
        ClusterShape, cutlass::gemm::collective::StageCountAutoCarveout<static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
        cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
    using GemmKernel = cutlass::gemm::kernel::GemmUniversal<Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;
};

using RefGemm = Fp4Gemm<C128x128x256>::Gemm;
using RefSfConfig = typename RefGemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

// Per-tensor amax of the activations (FLLM_NVFP4_ACT_SCALE=dynamic) and the epilogue factor derived from it, in device memory (no host
// synchronization; a CUDA graph can contain the kernels).
static __device__ float g_amax;
static __device__ float g_alpha;
__global__ void amax_reset_kernel() { g_amax = 0.f; }
// alpha = mult * amax / 2688 (dyn) or mult * (*src)
__global__ void alpha_kernel(float mult, const float *src, int dyn) { g_alpha = dyn ? mult / fp4_dynamic_gs(g_amax) : mult * *src; }
static float *alpha_symbol() {
    static float *cache[16] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16) dev = 0;
    if (!cache[dev]) {
        void *p = nullptr;
        if (cudaGetSymbolAddress(&p, g_alpha) != cudaSuccess) {
            cudaGetLastError();
            return nullptr;
        }
        cache[dev] = static_cast<float *>(p);
    }
    return cache[dev];
}

// One thread per 16 activations: packed E2M1 in natural order, scale in the swizzled layout. With GLU the gate values in `x` are combined
// with `up` first (activation fusion, see fllm_glu_up).
template <bool GLU, bool DYN, bool HALF_IN, class LayoutSF>
__global__ void quantize_activations_kernel(const void *__restrict__ x, const void *__restrict__ up, int mode, uint32_t *__restrict__ aq,
                                            uint8_t *__restrict__ asf, int cols, size_t blocks, LayoutSF lay, const float *__restrict__ gs_ptr) {
    const size_t bid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= blocks) return;
    const float gs = DYN ? fp4_dynamic_gs(g_amax) : (gs_ptr ? *gs_ptr : 1.f);
    float v[16];
    if constexpr (HALF_IN) {
        const __half *p = reinterpret_cast<const __half *>(x) + bid * 16;
        for (int i = 0; i < 16; i++) v[i] = __half2float(p[i]);
        if constexpr (GLU) {
            const __half *pu = reinterpret_cast<const __half *>(up ? up : x) + bid * 16;
            for (int i = 0; i < 16; i++) v[i] = fllm_glu(v[i], __half2float(pu[i]), mode);
        }
    } else {
        const float4 *p = reinterpret_cast<const float4 *>(reinterpret_cast<const float *>(x)) + bid * 4;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float4 t = __ldg(p + i);
            v[4 * i] = t.x; v[4 * i + 1] = t.y; v[4 * i + 2] = t.z; v[4 * i + 3] = t.w;
        }
        if constexpr (GLU) {
            const float4 *pu = reinterpret_cast<const float4 *>(reinterpret_cast<const float *>(up ? up : x)) + bid * 4;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const float4 t = __ldg(pu + i);
                v[4 * i] = fllm_glu(v[4 * i], t.x, mode);
                v[4 * i + 1] = fllm_glu(v[4 * i + 1], t.y, mode);
                v[4 * i + 2] = fllm_glu(v[4 * i + 2], t.z, mode);
                v[4 * i + 3] = fllm_glu(v[4 * i + 3], t.w, mode);
            }
        }
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
        if constexpr (HALF_IN) {
            const __half *p = reinterpret_cast<const __half *>(x) + bid * 16;
            const __half *pu = reinterpret_cast<const __half *>(GLU && up ? up : x) + bid * 16;
            for (int i = 0; i < 16; i++) {
                float val = __half2float(p[i]);
                if constexpr (GLU) val = fllm_glu(val, __half2float(pu[i]), mode);
                m = fmaxf(m, fabsf(val));
            }
        } else {
            const float4 *p = reinterpret_cast<const float4 *>(reinterpret_cast<const float *>(x)) + bid * 4;
            const float4 *pu = reinterpret_cast<const float4 *>(reinterpret_cast<const float *>(GLU && up ? up : x)) + bid * 4;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const float4 t = __ldg(p + i);
                if constexpr (GLU) {
                    const float4 u = __ldg(pu + i);
                    m = fmaxf(m, fmaxf(fmaxf(fabsf(fllm_glu(t.x, u.x, mode)), fabsf(fllm_glu(t.y, u.y, mode))),
                                       fmaxf(fabsf(fllm_glu(t.z, u.z, mode)), fabsf(fllm_glu(t.w, u.w, mode)))));
                } else {
                    m = fmaxf(m, fmaxf(fmaxf(fabsf(t.x), fabsf(t.y)), fmaxf(fabsf(t.z), fabsf(t.w))));
                }
            }
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    if ((threadIdx.x & 31) == 0 && m > 0.f) atomicMax(reinterpret_cast<int *>(&g_amax), __float_as_int(m));
}

__global__ void add_bias_kernel(float *__restrict__ y, const float *__restrict__ bias, int rows, size_t total) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) y[i] += __ldg(bias + i % rows);
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
};
thread_local State g_state;

struct AlphaSpec {
    float host = 1.f;
    const float *dev = nullptr;
};

// One GEMM of tile configuration Config with the activations already quantized into st.aq / st.asf.
template <typename Config, bool Acc, typename Out>
int launch_typed(cudaStream_t s, State &st, void *Y, const AlphaSpec &al, int M, int N, int K, const uint8_t *W) {
    using G = Fp4Gemm<Config, Acc, Out>;
    using Gemm = typename G::Gemm;
    using Sm1xxBlkScaledConfig = typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

    using StrideA = typename Gemm::GemmKernel::StrideA;
    using StrideB = typename Gemm::GemmKernel::StrideB;
    using StrideD = typename Gemm::GemmKernel::StrideD;

    const auto layout_SFA = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(make_shape(M, N, K, 1));
    const auto layout_SFB = Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(make_shape(M, N, K, 1));
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
        {{}, Acc ? reinterpret_cast<const float *>(Y) : nullptr, stride_D, reinterpret_cast<typename G::ElementD *>(Y), stride_D}};
    if (al.dev) args.epilogue.thread.alpha_ptr = al.dev;
    else args.epilogue.thread.alpha = al.host;
    args.epilogue.thread.beta = Acc ? fllm_fp4_acc_scale : 0.f;

    Gemm gemm;
    if (gemm.can_implement(args) != cutlass::Status::kSuccess) return 0;
    if (!st.workspace.reserve(Gemm::get_workspace_size(args))) return 0;
    if (gemm.initialize(args, st.workspace.ptr, s) != cutlass::Status::kSuccess) return 0;
    if (gemm.run(args, st.workspace.ptr, s) != cutlass::Status::kSuccess) return 0;
    return 1;
}

// The accumulating result (an FP32 C operand next to D) leaves too little shared memory for the two tiles with 256 rows of weights: the CUTLASS
// builder refuses them at compile time. Those combinations are reported as "not implemented" (0) like any tile the kernel cannot run.
template <typename Config>
struct AccFits : std::true_type {};
template <>
struct AccFits<C128x256x256> : std::false_type {};
template <>
struct AccFits<C256x256x256> : std::false_type {};

template <typename Config>
int launch_kind(int kind, cudaStream_t s, State &st, void *Y, const AlphaSpec &al, int M, int N, int K, const uint8_t *W) {
    if (kind == kAcc) {
        if constexpr (AccFits<Config>::value) return launch_typed<Config, true, float>(s, st, Y, al, M, N, K, W);
        else return 0;
    }
    if (kind == kF16) return launch_typed<Config, false, cutlass::half_t>(s, st, Y, al, M, N, K, W);
    return launch_typed<Config, false, float>(s, st, Y, al, M, N, K, W);
}

int launch_tile(int tile, int kind, cudaStream_t s, State &st, void *Y, const AlphaSpec &al, int M, int N, int K, const uint8_t *W) {
    switch (tile) {
        case kT128x128x256: return launch_kind<C128x128x256>(kind, s, st, Y, al, M, N, K, W);
        case kT128x256x256: return launch_kind<C128x256x256>(kind, s, st, Y, al, M, N, K, W);
        case kT256x128x256: return launch_kind<C256x128x256>(kind, s, st, Y, al, M, N, K, W);
        case kT256x256x256: return launch_kind<C256x256x256>(kind, s, st, Y, al, M, N, K, W);
        default: return 0;
    }
}

bool tune_enabled() {
    static const bool on = [] {
        const char *e = std::getenv("FLLM_NVFP4_SM100_TUNE");
        if (e && *e) return std::strcmp(e, "0") != 0;
        const char *g = std::getenv("FLLM_NVFP4_TUNE");
        return g && *g && std::strcmp(g, "0") != 0;
    }();
    return on;
}

int tune_tile(cudaStream_t s, State &st, int kind, void *Y, const AlphaSpec &al, int M, int N, int K, const uint8_t *W) {
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
        cudaEventRecord(b, s);
        cudaEventSynchronize(b);
        float ms = 0.f;
        cudaEventElapsedTime(&ms, a, b);
        if (ms < best_ms) {
            best_ms = ms;
            pick = t;
        }
    }
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    if (restore) cudaMemcpyAsync(Y, st.save.ptr, y_bytes, cudaMemcpyDeviceToDevice, s);
    return pick;
}

// FLLM_NVFP4_SM100_TILE: -1 = unset (chosen by the shape), >= 0 = a tile
int tile_env() {
    static const int v = [] {
        const char *e = std::getenv("FLLM_NVFP4_SM100_TILE");
        if (!e || !*e) return -1;
        for (int i = 0; i < kTileCount; i++)
            if (!std::strcmp(e, kTileNames[i])) return i;
        fprintf(stderr, "FLLM_NVFP4_SM100_TILE=%s is not a tile (128x128x256, 128x256x256, 256x128x256, 256x256x256); choosing by shape\n", e);
        return -1;
    }();
    return v;
}
// Few tokens (decode, small prompts): the 128-token tile reads every weight once at memory speed. Many tokens and many output rows: two SMs
// per MMA. Many tokens, few output rows: the narrow wide-N tile is not used, N stays 128.
int choose_tile(int M, int N) {
    const int env = tile_env();
    if (env >= 0) return env;
    if (M <= 256 || N < 256) return kT128x128x256;
    if (M >= 1024 && N >= 2048) return kT256x256x256;
    return kT256x128x256;
}

int quantize(cudaStream_t s, State &st, const void *src_x, const void *src_up, bool half_in, int M, int K, int amode, const float *gs_ptr) {
    const auto layout_SFA = RefSfConfig::tile_atom_to_shape_SFA(make_shape(M, 128, K, 1));
    const size_t asf_bytes = (size_t)cosize(layout_SFA);
    // padding rows of the scale layout must be defined (zero): the MMA reads whole atoms
    cudaMemsetAsync(st.asf.ptr, 0, asf_bytes, s);
    const size_t blocks = (size_t)M * (K / 16);
    const unsigned grid = (unsigned)((blocks + 127) / 128);
    uint32_t *aq = static_cast<uint32_t *>(st.aq.ptr);
    uint8_t *asf = static_cast<uint8_t *>(st.asf.ptr);
    const bool glu = src_up != nullptr;
    if (amode == 2) {
        amax_reset_kernel<<<1, 1, 0, s>>>();
        if (half_in) {
            if (glu) amax_kernel<true, true><<<grid, 128, 0, s>>>(src_x, src_up, fllm_glu_mode, blocks);
            else amax_kernel<false, true><<<grid, 128, 0, s>>>(src_x, nullptr, 0, blocks);
            if (glu) quantize_activations_kernel<true, true, true><<<grid, 128, 0, s>>>(src_x, src_up, fllm_glu_mode, aq, asf, K, blocks, layout_SFA, nullptr);
            else quantize_activations_kernel<false, true, true><<<grid, 128, 0, s>>>(src_x, nullptr, 0, aq, asf, K, blocks, layout_SFA, nullptr);
        } else {
            if (glu) amax_kernel<true, false><<<grid, 128, 0, s>>>(src_x, src_up, fllm_glu_mode, blocks);
            else amax_kernel<false, false><<<grid, 128, 0, s>>>(src_x, nullptr, 0, blocks);
            if (glu) quantize_activations_kernel<true, true, false><<<grid, 128, 0, s>>>(src_x, src_up, fllm_glu_mode, aq, asf, K, blocks, layout_SFA, nullptr);
            else quantize_activations_kernel<false, true, false><<<grid, 128, 0, s>>>(src_x, nullptr, 0, aq, asf, K, blocks, layout_SFA, nullptr);
        }
        return 1;
    }
    if (half_in) {
        if (glu) quantize_activations_kernel<true, false, true><<<grid, 128, 0, s>>>(src_x, src_up, fllm_glu_mode, aq, asf, K, blocks, layout_SFA, gs_ptr);
        else quantize_activations_kernel<false, false, true><<<grid, 128, 0, s>>>(src_x, nullptr, 0, aq, asf, K, blocks, layout_SFA, gs_ptr);
    } else {
        if (glu) quantize_activations_kernel<true, false, false><<<grid, 128, 0, s>>>(src_x, src_up, fllm_glu_mode, aq, asf, K, blocks, layout_SFA, gs_ptr);
        else quantize_activations_kernel<false, false, false><<<grid, 128, 0, s>>>(src_x, nullptr, 0, aq, asf, K, blocks, layout_SFA, gs_ptr);
    }
    return 1;
}

// Does the binary hold the CUTLASS kernel for the current device? (cudaFuncGetAttributes fails with cudaErrorNoKernelImageForDevice when
// the object was built without this device's architecture-specific target.)
bool kernel_present() {
    cudaFuncAttributes attr;
    const cudaError_t e = cudaFuncGetAttributes(&attr, cutlass::device_kernel<typename Fp4Gemm<C128x128x256>::GemmKernel>);
    cudaGetLastError();
    return e == cudaSuccess;
}

}  // namespace fllm_fp4s100
#endif  // FLLM_SM100 && CUTLASS_ARCH_MMA_SM100_SUPPORTED

// Datacenter Blackwell (sm_100 / sm_103) and Thor (sm_110)
[[maybe_unused]] static bool sm100_family(int cc) { return cc == 100 || cc == 103 || cc == 110; }

// 1 when fllm_nvfp4_sm100_try computes products on device `dev`: FLLM_NVFP4_SM100=1, the build has the kernels (CUTLASS headers at build time and the
// sm_100a / sm_103a / sm_110a target of this device). backend/cuda.rs keeps the Marlin FP4 weights otherwise.
extern "C" int fllm_cuda_nvfp4_sm100_native(int dev) {
#if defined(FLLM_SM100_ACTIVE) && !defined(__HIPCC__)
    static const bool enabled = fllm_env_tri("FLLM_NVFP4_SM100") == 1;
    if (!enabled) return 0;
    int prev = 0;
    cudaDeviceProp p;
    if (cudaGetDevice(&prev) != cudaSuccess || cudaGetDeviceProperties(&p, dev) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    if (!sm100_family(p.major * 10 + p.minor)) return 0;
    if (cudaSetDevice(dev) != cudaSuccess) return 0;
    const bool present = fllm_fp4s100::kernel_present();
    cudaSetDevice(prev);
    return present ? 1 : 0;
#else
    (void)dev;
    return 0;
#endif
}

// Returns 1 when the product was computed, 0 when the caller has to decline (device, shape or build not covered). The result replaces Y;
// bias (rows floats) is added when given. Products that add into Y or leave FP16 are not supported here (fllm_nvfp4_cutlass_ok says no).
int fllm_nvfp4_sm100_try(cudaStream_t s, const void *W, const float *X, float *Y, const float *bias, float gscale, int rows, int cols, int batch) {
#if defined(FLLM_SM100_ACTIVE) && !defined(__HIPCC__)
    using namespace fllm_fp4s100;
    static const bool enabled = fllm_env_tri("FLLM_NVFP4_SM100") == 1;
    if (!enabled || !sm100_family(fllm_cc())) return 0;
    const bool acc = fllm_fp4_acc;
    const bool out_half = fllm_fp4_f16out;
    if (acc && out_half) return 0;
    const int kind = acc ? kAcc : (out_half ? kF16 : kF32);
    const bool glu_half = fllm_glu_up && fllm_glu_half;
    const bool half_in = (fllm_x_half != nullptr) || glu_half;
    const void *src_x = half_in && fllm_x_half ? fllm_x_half : X;
    const void *src_up = fllm_glu_up;
    if (batch < 1 || (cols & 255) != 0 || (rows & 15) != 0) return 0;
    if ((reinterpret_cast<uintptr_t>(src_x) & 15) != 0 || (reinterpret_cast<uintptr_t>(Y) & 15) != 0 || (reinterpret_cast<uintptr_t>(W) & 15) != 0)
        return 0;
    if (!kernel_present()) return 0;
    const uint8_t *w = static_cast<const uint8_t *>(W);
    State &st = g_state;
    const int M = batch, N = rows, K = cols;

    // The weights were uploaded in the Sm1xx scale layout (backend/cuda.rs, nvfp4_device_layout); check the scale index there against the one
    // CUTLASS derives for this problem.
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
    if (!st.aq.reserve((size_t)M * K / 2) || !st.asf.reserve(asf_bytes)) return 0;

    const int amode = fllm_nvfp4_act_scale_mode();
    const Nvfp4Trailer *tr = reinterpret_cast<const Nvfp4Trailer *>(w + fllm_nvfp4_trailer_offset(rows, cols));
    if (!quantize(s, st, src_x, src_up, half_in, M, K, amode, amode == 1 ? &tr->gs : nullptr)) return 0;
    AlphaSpec al;
    if (amode == 0) {
        al.host = gscale;
    } else if (amode == 1) {
        al.dev = &tr->alpha;
    } else {
        float *sym = alpha_symbol();
        if (!sym) return 0;
        alpha_kernel<<<1, 1, 0, s>>>(gscale, &tr->alpha, 1);
        al.dev = sym;
    }

    int tile = -1;
    if (tune_enabled()) {
        static std::map<std::tuple<int, int, int, int>, int> g_tune_cache;
        const auto key = std::make_tuple(rows, cols, batch / 128, kind);
        auto it = g_tune_cache.find(key);
        if (it != g_tune_cache.end()) {
            tile = it->second;
        } else {
            tile = tune_tile(s, st, kind, Y, al, M, N, K, w);
            if (tile >= 0) g_tune_cache[key] = tile;
        }
    }
    if (tile < 0) tile = choose_tile(M, N);

    if (!launch_tile(tile, kind, s, st, Y, al, M, N, K, w)) return 0;
    if (bias && !acc) {
        const size_t total = (size_t)M * N;
        add_bias_kernel<<<(unsigned)((total + 255) / 256 > 4096 ? 4096 : (total + 255) / 256), 256, 0, s>>>(Y, bias, N, total);
    }
    return fllm_check_launch("nvfp4 sm100") == 0 ? 1 : 0;
#else
    (void)s; (void)W; (void)X; (void)Y; (void)bias; (void)gscale; (void)rows; (void)cols; (void)batch;
    return 0;
#endif
}

extern "C" int fllm_cuda_nvfp4_sm100_moe_gemm(cudaStream_t s, const void *const *expert_weights, const float *X, float *Y,
                                              const int *expert_offsets, int num_experts, int rows, int cols) {
#if defined(FLLM_SM100_ACTIVE) && !defined(__HIPCC__)
    if (!fllm_cuda_nvfp4_sm100_native(0) || num_experts <= 0 || !expert_offsets) return -1;
    for (int e = 0; e < num_experts; e++) {
        const int start = expert_offsets[e];
        const int end = expert_offsets[e + 1];
        const int tokens = end - start;
        if (tokens <= 0) continue;
        const void *w = expert_weights[e];
        if (!w) return -1;
        const float *x_e = X + (size_t)start * cols;
        float *y_e = Y + (size_t)start * rows;
        if (!fllm_nvfp4_sm100_try(s, w, x_e, y_e, nullptr, 1.0f, rows, cols, tokens)) return -1;
    }
    return 0;
#else
    (void)s; (void)expert_weights; (void)X; (void)Y; (void)expert_offsets; (void)num_experts; (void)rows; (void)cols;
    return -1;
#endif
}
