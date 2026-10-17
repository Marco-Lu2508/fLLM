// fLLM CUDA backend: FP8 (E4M3) W8A8 GEMM for Ada (sm_89), Hopper (sm_90) and Blackwell (sm_100 / sm_120).
//
//   Y[token][row] = sum_k dq(X[token][k]) * dq(W[row][k]) + bias[row]
//
// Weights stay FP8 (E4M3) in memory; the FP32 activations are quantized per call to E4M3 with a dynamic per-token scale (amax / 448), a static
// per-tensor scale, or, for block-wise weights, one scale per 1 x 128 group (the recipe of vLLM's w8a8_block_fp8: weights with one scale per
// 128 x 128 block, activations per token and 128 values). The products run on the tensor cores with FP32 accumulation
// (mma.sync.aligned.m16n8k32 e4m3 x e4m3 -> f32, PTX ISA 8.4, sm_89 and newer).
//
// The kernel is written for fLLM (a 2-stage cp.async pipeline over K steps of 128 values, a padded shared-memory row of 144 bytes so that the
// fragment loads are free of bank conflicts). The scale conventions follow vLLM (csrc/quantization/w8a8, Apache-2.0, Copyright contributors
// to the vLLM project) and DeepSeek-V3 (`weight_scale_inv`); the mma instruction and its fragment layout are those of the PTX ISA. No code of
// those projects is copied.
//
// Test level of this file: compiled only (no GPU was available when it was written); the self test group `w8a8` checks it on a device.
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include <cuda_fp8.h>

#include "common.cuh"
#include "fllm_cuda_internal.h"
#include "fp8_gemm.h"
#include "hopper_util.h"
#include "mma_util.cuh"

namespace {

constexpr float kE4M3Max = 448.f;
constexpr int kRowBytes = 144;  // 128 data bytes + 16 padding: lanes g = 0..7 (rows) x t = 0..3 hit 32 distinct banks

__device__ __forceinline__ uint32_t pack4_e4m3(float a, float b, float c, float d, float inv) {
    const uint32_t q0 = __nv_cvt_float_to_fp8(a * inv, __NV_SATFINITE, __NV_E4M3);
    const uint32_t q1 = __nv_cvt_float_to_fp8(b * inv, __NV_SATFINITE, __NV_E4M3);
    const uint32_t q2 = __nv_cvt_float_to_fp8(c * inv, __NV_SATFINITE, __NV_E4M3);
    const uint32_t q3 = __nv_cvt_float_to_fp8(d * inv, __NV_SATFINITE, __NV_E4M3);
    return q0 | (q1 << 8) | (q2 << 16) | (q3 << 24);
}
__device__ __forceinline__ float amax4(const float4 v) { return fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))); }

// One block per token: scale = amax / 448 (1 for an all-zero row), q = E4M3(x / scale).
__global__ void __launch_bounds__(128) quant_token_kernel(const float *__restrict__ x, uint8_t *__restrict__ q, float *__restrict__ scale, int cols) {
    const float4 *xr = reinterpret_cast<const float4 *>(x + (size_t)blockIdx.x * cols);
    uint32_t *qr = reinterpret_cast<uint32_t *>(q + (size_t)blockIdx.x * cols);
    const int n4 = cols >> 2;
    float m = 0.f;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) m = fmaxf(m, amax4(__ldg(xr + i)));
    m = warp_max(m);
    __shared__ float red[4];
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    m = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
    const float s = m > 0.f ? __fdiv_rn(m, kE4M3Max) : 1.f;
    const float inv = __frcp_rn(s);
    if (threadIdx.x == 0) scale[blockIdx.x] = s;
    for (int i = threadIdx.x; i < n4; i += blockDim.x) {
        const float4 v = __ldg(xr + i);
        qr[i] = pack4_e4m3(v.x, v.y, v.z, v.w, inv);
    }
}

// One block (4 warps) per token, one warp per group of 128 values at a time: scale[token][group].
__global__ void __launch_bounds__(128) quant_group_kernel(const float *__restrict__ x, uint8_t *__restrict__ q, float *__restrict__ scale, int cols) {
    const int groups = cols >> 7;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int g = warp; g < groups; g += 4) {
        const float4 v = __ldg(reinterpret_cast<const float4 *>(x + (size_t)blockIdx.x * cols + (size_t)g * 128) + lane);
        const float m = warp_max(amax4(v));
        const float s = m > 0.f ? __fdiv_rn(m, kE4M3Max) : 1.f;
        if (lane == 0) scale[(size_t)blockIdx.x * groups + g] = s;
        *reinterpret_cast<uint32_t *>(q + (size_t)blockIdx.x * cols + (size_t)g * 128 + lane * 4) = pack4_e4m3(v.x, v.y, v.z, v.w, __frcp_rn(s));
    }
}

// Static per-tensor scale: q = E4M3(x / scale), saturating.
__global__ void quant_static_kernel(const float *__restrict__ x, uint8_t *__restrict__ q, float *__restrict__ scale_out, size_t n4, float scale) {
    const float inv = __frcp_rn(scale);
    if (blockIdx.x == 0 && threadIdx.x == 0) *scale_out = scale;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += (size_t)gridDim.x * blockDim.x) {
        const float4 v = __ldg(reinterpret_cast<const float4 *>(x) + i);
        reinterpret_cast<uint32_t *>(q)[i] = pack4_e4m3(v.x, v.y, v.z, v.w, inv);
    }
}

[[maybe_unused]] __device__ __forceinline__ void mma_e4m3(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#endif
}

struct GemmArgs {
    const uint8_t *xq;
    const float *xs;  // xs[m * xs_m + kb * xs_k]
    long xs_m, xs_k;
    const uint8_t *w;
    const float *ws;  // per row [N], or per block [ceil(N / 128)][K / 128]
    float *y;
    const float *bias;
    int M, N, K;
};

// The products of one 128-wide K step (4 mma k-steps of 32) of the warp's tile into c.
template <int MT, int NT>
__device__ __forceinline__ void mma_stage(float (&c)[MT][NT][4], const uint8_t *sA, const uint8_t *sB, int wm0, int wn0, int g, int t) {
#pragma unroll
    for (int ks = 0; ks < 4; ks++) {
        uint32_t af[MT][4];
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
            const uint8_t *p = sA + (wm0 + mt * 16 + g) * kRowBytes + ks * 32 + t * 4;
            af[mt][0] = *reinterpret_cast<const uint32_t *>(p);
            af[mt][1] = *reinterpret_cast<const uint32_t *>(p + 8 * kRowBytes);
            af[mt][2] = *reinterpret_cast<const uint32_t *>(p + 16);
            af[mt][3] = *reinterpret_cast<const uint32_t *>(p + 8 * kRowBytes + 16);
        }
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
            const uint8_t *p = sB + (wn0 + nt * 8 + g) * kRowBytes + ks * 32 + t * 4;
            const uint32_t b0 = *reinterpret_cast<const uint32_t *>(p), b1 = *reinterpret_cast<const uint32_t *>(p + 16);
#pragma unroll
            for (int mt = 0; mt < MT; mt++) mma_e4m3(c[mt][nt], af[mt], b0, b1);
        }
    }
}

// BLOCK: weights with one scale per 128 x 128 block and activations with one scale per 128 values: every K step is scaled by its own factors
// (the partial sums of a step are folded into the running sums); otherwise all scales are applied once in the epilogue.
template <int BM, int BN, int WARPS_M, int WARPS_N, bool BLOCK>
__global__ void __launch_bounds__(WARPS_M * WARPS_N * 32) fp8_gemm_kernel(GemmArgs a) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 890
    constexpr int THREADS = WARPS_M * WARPS_N * 32;
    constexpr int MT = BM / (16 * WARPS_M), NT = BN / (8 * WARPS_N);
    static_assert(MT >= 1 && NT >= 1 && BM % (16 * WARPS_M) == 0 && BN % (8 * WARPS_N) == 0 && 128 % BN == 0, "tile shape");
    extern __shared__ __align__(16) uint8_t smem[];
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int wm0 = (warp % WARPS_M) * MT * 16, wn0 = (warp / WARPS_M) * NT * 8;
    const int ksteps = a.K >> 7;
    const int nblocks = (a.N + 127) >> 7;

    float acc[MT][NT][4];
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll
            for (int e = 0; e < 4; e++) acc[mt][nt][e] = 0.f;

    auto load = [&](int kt, int stage) {
        uint8_t *sA = smem + (size_t)stage * (BM + BN) * kRowBytes;
        uint8_t *sB = sA + BM * kRowBytes;
        const size_t k0 = (size_t)kt * 128;
        for (int c = threadIdx.x; c < (BM + BN) * 8; c += THREADS) {
            const int r = c >> 3, ch = c & 7;
            if (r < BM) {
                const int gm = m0 + r;
                const bool ok = gm < a.M;
                cp_async16(sA + r * kRowBytes + ch * 16, a.xq + (size_t)(ok ? gm : 0) * a.K + k0 + ch * 16, ok);
            } else {
                const int gn = n0 + r - BM;
                const bool ok = gn < a.N;
                cp_async16(sB + (r - BM) * kRowBytes + ch * 16, a.w + (size_t)(ok ? gn : 0) * a.K + k0 + ch * 16, ok);
            }
        }
    };

    load(0, 0);
    cp_async_commit();
    for (int kt = 0; kt < ksteps; kt++) {
        if (kt + 1 < ksteps) load(kt + 1, (kt + 1) & 1);
        cp_async_commit();
        cp_async_wait<1>();
        __syncthreads();
        const uint8_t *sA = smem + (size_t)(kt & 1) * (BM + BN) * kRowBytes;
        const uint8_t *sB = sA + BM * kRowBytes;
        if constexpr (BLOCK) {
            float part[MT][NT][4];
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#pragma unroll
                for (int nt = 0; nt < NT; nt++)
#pragma unroll
                    for (int e = 0; e < 4; e++) part[mt][nt][e] = 0.f;
            mma_stage<MT, NT>(part, sA, sB, wm0, wn0, g, t);
#pragma unroll
            for (int nt = 0; nt < NT; nt++) {
                int nb = (n0 + wn0 + nt * 8) >> 7;
                nb = nb < nblocks ? nb : nblocks - 1;
                const float wsv = __ldg(a.ws + (size_t)nb * ksteps + kt);
#pragma unroll
                for (int mt = 0; mt < MT; mt++) {
                    const int ma = m0 + wm0 + mt * 16 + g, mb = ma + 8;
                    const float xa = ma < a.M ? __ldg(a.xs + ma * a.xs_m + kt * a.xs_k) * wsv : 0.f;
                    const float xb = mb < a.M ? __ldg(a.xs + mb * a.xs_m + kt * a.xs_k) * wsv : 0.f;
                    acc[mt][nt][0] += part[mt][nt][0] * xa;
                    acc[mt][nt][1] += part[mt][nt][1] * xa;
                    acc[mt][nt][2] += part[mt][nt][2] * xb;
                    acc[mt][nt][3] += part[mt][nt][3] * xb;
                }
            }
        } else {
            mma_stage<MT, NT>(acc, sA, sB, wm0, wn0, g, t);
        }
        __syncthreads();
    }

    // epilogue: row g / g + 8, columns 2t / 2t + 1 of every m16n8 tile
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int m = m0 + wm0 + mt * 16 + g + ((e >> 1) << 3);
                const int n = n0 + wn0 + nt * 8 + 2 * t + (e & 1);
                if (m >= a.M || n >= a.N) continue;
                float v = acc[mt][nt][e];
                if (!BLOCK) v *= __ldg(a.xs + m * a.xs_m) * __ldg(a.ws + n);
                if (a.bias) v += __ldg(a.bias + n);
                a.y[(size_t)m * a.N + n] = v;
            }
        }
    }
#endif
}

template <int BM, int BN, int WARPS_M, int WARPS_N, bool BLOCK>
int launch(cudaStream_t s, const GemmArgs &a) {
    constexpr size_t smem = (size_t)2 * (BM + BN) * kRowBytes;
    if (smem + 1024 > (size_t)fllm_smem_limit()) return 0;
    static bool configured[16] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 16) dev = 0;
    if (!configured[dev]) {
        cudaFuncSetAttribute(fp8_gemm_kernel<BM, BN, WARPS_M, WARPS_N, BLOCK>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        configured[dev] = true;
    }
    fp8_gemm_kernel<BM, BN, WARPS_M, WARPS_N, BLOCK>
        <<<dim3((a.N + BN - 1) / BN, (a.M + BM - 1) / BM), WARPS_M * WARPS_N * 32, smem, s>>>(a);
    return 1;
}

// Tile candidates by batch: one m16 tile of 32 rows per CTA for decode (many CTAs, the weights are streamed once), 32 / 64 tokens,
// and for prompts 64x64, 128x64 and 128x128 (auto-tuned once per shape).
template <bool BLOCK>
int run_candidate(cudaStream_t s, const GemmArgs &a, int c) {
    switch (c) {
        case 0: return launch<16, 32, 1, 4, BLOCK>(s, a);
        case 1: return launch<32, 64, 2, 2, BLOCK>(s, a);
        case 2: return launch<64, 64, 2, 2, BLOCK>(s, a);
        case 3: return launch<128, 64, 4, 1, BLOCK>(s, a);
        case 4:
            if constexpr (BLOCK) return 0;  // the block-scaled variant needs 255 registers and spills
            else return launch<128, 128, 2, 2, BLOCK>(s, a);
        default: return 0;
    }
}

template <bool BLOCK>
int run(cudaStream_t s, const GemmArgs &a, int rows, int cols) {
    int c;
    if (a.M <= 16) c = 0;
    else if (a.M <= 32) c = 1;
    else if (a.M <= 128) c = 2;
    else {
        cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
        const bool capturing = cudaStreamIsCapturing(s, &cap) != cudaSuccess || cap != cudaStreamCaptureStatusNone;
        cudaGetLastError();
        // candidates 2..4; the tuner times them on the real data (the result is overwritten, not accumulated, so repeating a run is harmless)
        c = 2;
        if (!capturing) {
            const int pick = fllm_autotune(s, BLOCK ? 0xF9 : 0xF8, rows, cols, a.M, 3, [&](int i) { return run_candidate<BLOCK>(s, a, 2 + i); });
            c = 2 + pick;
        }
    }
    if (!run_candidate<BLOCK>(s, a, c)) {
        // the candidate needs more shared memory than the device offers: fall back to the smallest tile
        if (!run_candidate<BLOCK>(s, a, a.M <= 16 ? 0 : 2)) return -1;
    }
    return fllm_check_launch("fp8 gemm");
}

std::atomic_int g_mode{-2};
int env_mode() {
    const char *v = getenv("FLLM_FP8_GEMM");
    // off unless asked for: the kernels have not run on a device yet (docs/changes/hopper.md); auto = on from compute capability 8.9
    if (v == nullptr || *v == 0) return 0;
    if (strcmp(v, "cutlass") == 0) return 2;
    return fllm_env_tri("FLLM_FP8_GEMM");
}

}  // namespace

void fllm_fp8_gemm_set_mode(int mode) { g_mode.store(mode < -1 || mode > 2 ? -1 : mode); }
// -1 auto, 0 off, 1 mma.sync, 2 CUTLASS
int fllm_fp8_gemm_mode() {
    const int forced = g_mode.load();
    if (forced >= 0) return forced;
    static const int env = env_mode();
    return env;
}

bool fllm_fp8_w8a8_enabled(int rows, int cols) {
    const int mode = fllm_fp8_gemm_mode();
    if (mode == 0 || fllm_cc() < 89 || fllm_exact_math()) return false;
    return rows > 0 && cols > 0 && (cols & 127) == 0;
}

int fllm_fp8_w8a8_gemm(cudaStream_t s, const void *w, const float *wscale, int wblock, const float *x, float *y, const float *bias, int rows, int cols,
                       int batch, int amode, float static_scale) {
    if (fllm_cc() < 89 || (cols & 127) != 0 || rows <= 0 || batch <= 0) return -1;
    if ((reinterpret_cast<uintptr_t>(w) & 15) != 0 || (reinterpret_cast<uintptr_t>(x) & 15) != 0 || (reinterpret_cast<uintptr_t>(wscale) & 3) != 0)
        return -1;
    if (wblock ? amode != 2 : amode == 2) return -1;
    if (amode == 1 && !(static_scale > 0.f)) return -1;
    const int groups = cols >> 7;
    const size_t q_bytes = (size_t)batch * cols;
    const size_t s_count = amode == 0 ? (size_t)batch : amode == 1 ? 1 : (size_t)batch * groups;
    W8A8Scratch *sc = w8a8_scratch(s, q_bytes, s_count * sizeof(float));
    if (!sc) return fllm_set_error("FP8 W8A8 GEMM: no memory for %zu activation bytes (or the scratch would have to grow inside a CUDA graph capture)", q_bytes);
    uint8_t *xq = static_cast<uint8_t *>(sc->q);
    float *xs = sc->s;
    GemmArgs a{xq, xs, 0, 0, static_cast<const uint8_t *>(w), wscale, y, bias, batch, rows, cols};
    if (amode == 0) {
        quant_token_kernel<<<batch, 128, 0, s>>>(x, xq, xs, cols);
        a.xs_m = 1;
        a.xs_k = 0;
    } else if (amode == 2) {
        quant_group_kernel<<<batch, 128, 0, s>>>(x, xq, xs, cols);
        a.xs_m = groups;
        a.xs_k = 1;
    } else {
        const size_t n4 = q_bytes >> 2;
        const unsigned grid = (unsigned)((n4 + 255) / 256 > 4096 ? 4096 : (n4 + 255) / 256);
        quant_static_kernel<<<grid, 256, 0, s>>>(x, xq, xs, n4, static_scale);
        a.xs_m = 0;
        a.xs_k = 0;
    }
    if (int e = fllm_check_launch("fp8 activation quantization")) return e;
    // FLLM_FP8_GEMM=cutlass: the Hopper wgmma kernel for per-token scales or blockwise scaling; anything it does not cover continues with the mma.sync kernel
    if (fllm_fp8_gemm_mode() == 2) {
        if (amode == 0 && !wblock) {
            const int rc = fllm_fp8_cutlass_gemm(s, w, wscale, xq, xs, y, bias, rows, cols, batch);
            if (rc >= 0) return rc;
        } else if (wblock) {
            const int rc = fllm_fp8_cutlass_blockwise_gemm(s, w, wscale, xq, xs, y, bias, rows, cols, batch, wblock);
            if (rc >= 0) return rc;
        }
    }
    return wblock ? run<true>(s, a, rows, cols) : run<false>(s, a, rows, cols);
}
