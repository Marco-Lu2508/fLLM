/*
 * fLLM HIP backend: skinny GEMM for F16 / BF16 weights and 1..4 tokens (decode).
 *
 * Derived from vLLM, csrc/rocm/skinny_gemms.cu (wvSplitK_hf_), https://github.com/vllm-project/vllm, commit
 * 7fad7db3309347a9d6288a09ed0993728e474a34. Copyright contributors to the vLLM project.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Changes against the original: the PyTorch glue is replaced by the C++ entry point fllm_hip_skinny_try (raw device pointers, a HIP
 * stream); the activations arrive as FP32 and are converted to the weight type into a scratch buffer first; the output and the bias are
 * FP32 (fLLM keeps activations in FP32); only the `wvSplitK_hf_` kernel is ported (it covers `wvSplitK_hf_sml_` too, the LDS holds as much of
 * the activations as fits and the rest is read from memory), `wvSplitK_hf_big_` and the MFMA-bf16 variant of gfx94x are left out;
 * the wave size is derived from the architecture instead of a launch parameter; kernels whose THRDS does not match the wave size of the
 * compiled architecture are empty (they are instantiated for both on the host).
 *
 * The idea (from the original): a wave owns YTILE rows of the weight matrix, its lanes stream 16-byte chunks of those rows with
 * non-temporal loads, multiply them with the activations held in LDS using packed v_dot2 instructions and reduce across the wave with DPP.
 * One workgroup of 16 waves per CU, persistent over the rows. Targets: gfx90a / gfx94x / gfx950 (wave64) and gfx11 / gfx12 (wave32).
 * FLLM_HIP_NO_SKINNY=1 switches the path off. Level of verification: L0 (compiles; never run on hardware).
 */
#include <cstdlib>
#include <string>
#include <type_traits>

#include <hip/hip_bf16.h>

#include "../cuda/common.cuh"
#include "../cuda/fllm_cuda_internal.h"
#include "fllm_hip.h"

// architectures with a kernel body: v_dot2c (gfx9) or v_dot2 (gfx11 / gfx12)
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__))
#define FLLM_SK_GFX9 1
#define FLLM_SK_WAVE 64
#elif defined(__HIP_DEVICE_COMPILE__) && (defined(__GFX11__) || defined(__GFX12__))
#define FLLM_SK_GFX1X 1
#define FLLM_SK_WAVE 32
#else
#define FLLM_SK_WAVE 0
#endif
#if defined(__gfx950__)
#define FLLM_SK_LDS (160 * 1024)
#else
#define FLLM_SK_LDS (64 * 1024)
#endif

namespace {

using sk_half = __half;
using sk_bf16 = __hip_bfloat16;

#if defined(FLLM_SK_GFX9) || defined(FLLM_SK_GFX1X)
// acc += a.lo * b.lo + a.hi * b.hi on packed 16-bit pairs that travel as the bits of a float
template <typename scalar_t>
__device__ __forceinline__ void dot2c(float &acc, float a, float b) {
    if constexpr (std::is_same_v<scalar_t, sk_half>) {
#if defined(FLLM_SK_GFX9)
        asm("v_dot2c_f32_f16 %0, %2, %3" : "=v"(acc) : "0"(acc), "v"(a), "v"(b));
#else
        asm("v_dot2_f32_f16 %0, %1, %2, %0" : "+v"(acc) : "v"(a), "v"(b));
#endif
    } else {
        const uint32_t ua = __float_as_uint(a), ub = __float_as_uint(b);
        acc += __uint_as_float(ua << 16) * __uint_as_float(ub << 16) + __uint_as_float(ua & 0xffff0000u) * __uint_as_float(ub & 0xffff0000u);
    }
}

template <typename T>
__device__ __forceinline__ T loadnt(T *addr) {
    return __builtin_nontemporal_load(addr);
}

// To avoid LLVM silently upcasting to double
__device__ __forceinline__ unsigned int min__(uint32_t a, uint32_t b) { return min(a, b); }
#endif

// activations FP32 -> weight type
template <typename scalar_t>
__global__ void convert_x_kernel(const float *__restrict__ x, scalar_t *__restrict__ out, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        if constexpr (std::is_same_v<scalar_t, sk_half>) out[i] = __float2half(x[i]);
        else out[i] = __float2bfloat16(x[i]);
    }
}

// B: weights [M][K] (row stride K), A: activations [N][K] as scalar_t, C: result [N][M] (FP32), BIAS: M floats or null.
template <typename scalar_t, int THRDS, int YTILE, int WvPrGrp, int A_CHUNK, int UNRL, int N>
__global__ void __launch_bounds__(WvPrGrp *THRDS)
    wvsplitk_kernel(const int K, const int M, const scalar_t *B, const scalar_t *__restrict__ A, const float *__restrict__ BIAS, float *C,
                    const int _WvPrGrp, const int CuCount) {
#if defined(FLLM_SK_GFX9) || defined(FLLM_SK_GFX1X)
    if constexpr (THRDS == FLLM_SK_WAVE) {
        constexpr int max_lds_len = FLLM_SK_LDS / 2;
        using scalar8 = __attribute__((__vector_size__((A_CHUNK / 2) * sizeof(float)))) float;
        union bigType {
            scalar_t h[A_CHUNK];
            float f[A_CHUNK / 2];
            float2 f2[A_CHUNK / 4];
            double d[A_CHUNK / 4];
            scalar8 h8;
        };
        const int Kap = K, Kbp = K;

        // the activations (as far as they fit) live in LDS for the whole lifetime of the workgroup
        __shared__ scalar_t s[max_lds_len];

        // which of the YTILE rows of the last wave are real (the last wave is shifted back so that it never reads past the matrix)
        uint32_t commitColumn[YTILE];
        for (uint32_t i = 0; i < YTILE; i++) commitColumn[i] = 1;

        uint32_t m = (blockIdx.x * _WvPrGrp + threadIdx.y) * YTILE;
        if (m < (uint32_t)M && (m + YTILE) >= (uint32_t)M) {
            uint32_t startColumn = M - YTILE;
            for (uint32_t i = 0; i < (m - startColumn); i++) commitColumn[i] = 0;
            m = startColumn;
        }

        // each lane copies 8 elements (A_CHUNK) per step
        for (uint32_t k = (threadIdx.y * THRDS + threadIdx.x) * A_CHUNK; k < min__(Kap * N, max_lds_len); k += THRDS * WvPrGrp * A_CHUNK) {
            *((bigType *)(&s[k])) = *((const bigType *)(&A[k]));
        }
        __syncthreads();

        if (threadIdx.y >= _WvPrGrp) return;

        while (m < (uint32_t)M) {
            // 'sum' accumulates the products split across the lanes of the wave
            float sum[N][YTILE] = {};

            for (uint32_t k1 = 0; k1 < (uint32_t)K; k1 += THRDS * A_CHUNK * UNRL) {
                bigType bigA[N][UNRL] = {};
                bigType bigB[YTILE][UNRL];
                // weights from memory
#pragma unroll
                for (uint32_t k2 = 0; k2 < UNRL; k2++) {
                    uint32_t k = k1 + k2 * THRDS * A_CHUNK;
                    uint32_t k_ = k + threadIdx.x * A_CHUNK;
                    const scalar_t *B_ = &B[min__(k_, K - A_CHUNK)];
                    for (int y = 0; y < YTILE; y++) bigB[y][k2].h8 = (loadnt((scalar8 *)(&B_[min__(y + m, M - 1) * (size_t)Kbp])));
                }
                // activations from LDS, or from memory beyond what fits in LDS
#pragma unroll
                for (uint32_t k2 = 0; k2 < UNRL; k2++) {
                    uint32_t k = k1 + k2 * THRDS * A_CHUNK;
                    uint32_t k_ = k + threadIdx.x * A_CHUNK;
                    if (k_ >= (uint32_t)K) break;
                    for (int n = 0; n < N; n++) {
                        if (k_ + Kap * n < (uint32_t)max_lds_len) bigA[n][k2] = *((const bigType *)(&(s[k_ + Kap * n])));
                        else bigA[n][k2] = *((const bigType *)(&(A[k_ + Kap * n])));
                    }
                }
                // multiply in an interleaved manner
                for (uint32_t n = 0; n < N; n++) {
                    for (uint32_t k2 = 0; k2 < UNRL; k2++) {
                        for (int y = 0; y < YTILE; y++) {
                            for (uint32_t b = 0; b < A_CHUNK / 2; b++) dot2c<scalar_t>(sum[n][y], bigA[n][k2].f[b], bigB[y][k2].f[b]);
                        }
                    }
                }
            }

            // final reduction over the wave with DPP (the last lane ends up with the total)
            for (int n = 0; n < N; n++) {
                for (int y = 0; y < YTILE; y++) {
                    sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x118, 0xf, 0xf, 1);  // row_shr8
                    sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x114, 0xf, 0xf, 1);  // row_shr4
                    sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x112, 0xf, 0xf, 1);  // row_shr2
                    sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x111, 0xf, 0xf, 1);  // row_shr1
#if defined(FLLM_SK_GFX9)
                    sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x142, 0xf, 0xf, 1);  // ROW_BCAST15
                    sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x143, 0xf, 0xf, 1);  // ROW_BCAST31
#else
                    sum[n][y] += __shfl_xor(sum[n][y], 16);
#endif
                }
            }

            if (threadIdx.x == (THRDS - 1)) {
                for (int n = 0; n < N; n++) {
                    for (int y = 0; y < YTILE; y++) {
                        if (commitColumn[y]) C[m + y + (size_t)n * M] = sum[n][y] + (BIAS ? BIAS[m + y] : 0.f);
                    }
                }
            }

            m += CuCount * _WvPrGrp * YTILE;

            // the last wave is shifted back again
            if (m < (uint32_t)M && (m + YTILE) >= (uint32_t)M) {
                uint32_t startColumn = M - YTILE;
                for (uint32_t i = 0; i < (m - startColumn); i++) commitColumn[i] = 0;
                m = startColumn;
            }
        }
    }
#endif
}

// how many waves per group keep the persistent loop balanced (original: mindiv)
int mindiv(int N, int div1, int div2) {
    int nPrRnd = div1 * div2;
    int rnds[13];
    for (int i = 0; i < 13; i++) {
        rnds[i] = (N + nPrRnd - 1) / nPrRnd;
        nPrRnd -= div1;
    }
    for (int i = 12; i >= 0; i--)
        if (rnds[0] == rnds[i]) return (div2 - i);
    return 0;
}

// (YTILE, UNRL) combinations of the original's tile tables
enum SkCfg { CFG_1_4, CFG_2_2, CFG_3_2, CFG_4_1, CFG_4_2, CFG_2_4, CFG_1_2, CFG_1_1 };

// The tables of the original WVSPLIT_TILE / WVSPLIT_TILE_CFG macros. sYT: rows per CU quarter, fit: the activations fit in LDS.
SkCfg pick_cfg(bool gfx1151, int sYT, int N, int K, bool fit) {
    if (gfx1151) {
        if (sYT <= 1) return CFG_1_4;
        if ((K % 1024 == 512) && K >= 1536 && (sYT >= 40 || K >= 4096)) return CFG_4_1;
        if (K < 1024) return CFG_2_4;
        if (K <= 2048 && (N >= 2 || sYT <= 26)) return CFG_1_4;
        if (N >= 2 && !fit) return CFG_1_4;
        if (N == 1) return CFG_1_2;
        return CFG_1_1;
    }
    if (sYT <= 1) return CFG_1_4;
    if (N == 1 || !fit || sYT <= 4 * 2) return CFG_2_2;
    if (sYT <= 4 * 3) return CFG_3_2;
    if (N == 4) return CFG_4_1;
    return CFG_4_2;
}

template <typename T, int THRDS, int YTILE, int UNRL, int N>
void launch(cudaStream_t s, const T *W, const T *A, const float *bias, float *C, int K, int M, int cus) {
    constexpr int WVPRGRP = 16;
    const int wv = mindiv(M, cus * YTILE, WVPRGRP);
    wvsplitk_kernel<T, THRDS, YTILE, WVPRGRP, 8, UNRL, N><<<dim3(cus), dim3(THRDS, WVPRGRP), 0, s>>>(K, M, W, A, bias, C, wv, cus);
}

template <typename T, int THRDS, int N>
void dispatch_cfg(SkCfg cfg, cudaStream_t s, const T *W, const T *A, const float *bias, float *C, int K, int M, int cus) {
    switch (cfg) {
        case CFG_1_4: launch<T, THRDS, 1, 4, N>(s, W, A, bias, C, K, M, cus); break;
        case CFG_2_2: launch<T, THRDS, 2, 2, N>(s, W, A, bias, C, K, M, cus); break;
        case CFG_3_2: launch<T, THRDS, 3, 2, N>(s, W, A, bias, C, K, M, cus); break;
        case CFG_4_1: launch<T, THRDS, 4, 1, N>(s, W, A, bias, C, K, M, cus); break;
        case CFG_4_2: launch<T, THRDS, 4, 2, N>(s, W, A, bias, C, K, M, cus); break;
        case CFG_2_4: launch<T, THRDS, 2, 4, N>(s, W, A, bias, C, K, M, cus); break;
        case CFG_1_2: launch<T, THRDS, 1, 2, N>(s, W, A, bias, C, K, M, cus); break;
        case CFG_1_1: launch<T, THRDS, 1, 1, N>(s, W, A, bias, C, K, M, cus); break;
    }
}

template <typename T, int THRDS>
void dispatch_n(int N, SkCfg cfg, cudaStream_t s, const T *W, const T *A, const float *bias, float *C, int K, int M, int cus) {
    switch (N) {
        case 1: dispatch_cfg<T, THRDS, 1>(cfg, s, W, A, bias, C, K, M, cus); break;
        case 2: dispatch_cfg<T, THRDS, 2>(cfg, s, W, A, bias, C, K, M, cus); break;
        case 3: dispatch_cfg<T, THRDS, 3>(cfg, s, W, A, bias, C, K, M, cus); break;
        default: dispatch_cfg<T, THRDS, 4>(cfg, s, W, A, bias, C, K, M, cus); break;
    }
}

template <typename T>
int run(cudaStream_t s, const void *W, const float *X, float *Y, const float *bias, int rows, int cols, int batch, void *scratch) {
    const FllmHipDevice &dev = fllm_hip_device();
    T *a16 = static_cast<T *>(scratch);
    const size_t n = (size_t)batch * cols;
    convert_x_kernel<T><<<(unsigned)((n + 255) / 256 < 1024 ? (n + 255) / 256 : 1024), 256, 0, s>>>(X, a16, n);
    const int max_lds_len = (dev.lds_bytes > 64 * 1024 && std::string(dev.gcn).rfind("gfx95", 0) == 0 ? 160 * 1024 : 64 * 1024) / 2;
    const bool fit = (size_t)cols * batch <= (size_t)max_lds_len;
    const int sYT = (rows + dev.cus * 4 - 1) / (dev.cus * 4);
    const SkCfg cfg = pick_cfg(dev.gfx1151, sYT, batch, cols, fit);
    const T *w = static_cast<const T *>(W);
    if (dev.wave == 64) dispatch_n<T, 64>(batch, cfg, s, w, a16, bias, Y, cols, rows, dev.cus);
    else dispatch_n<T, 32>(batch, cfg, s, w, a16, bias, Y, cols, rows, dev.cus);
    return 1;
}

}  // namespace

int fllm_hip_skinny_try(cudaStream_t s, const void *W, int kind, const float *X, float *Y, const float *bias, int rows, int cols, int batch,
                        void *scratch, size_t scratch_bytes) {
    static const bool enabled = fllm_hip_enabled("SKINNY");
    if (!enabled || (kind != KIND_F16 && kind != KIND_BF16)) return 0;
    const FllmHipDevice &dev = fllm_hip_device();
    // kernel bodies exist for gfx90a / gfx94x / gfx950 and gfx11 / gfx12 (v_dot2); wave size 64 resp. 32
    const bool gfx9 = dev.arch == FLLM_HIP_ARCH_CDNA && dev.dot2;
    const bool gfx1x = dev.arch == FLLM_HIP_ARCH_RDNA3 || dev.arch == FLLM_HIP_ARCH_RDNA4;
    if (!(gfx9 || gfx1x) || dev.cus <= 0) return 0;
    if (batch < 1 || batch > 4 || rows <= 0 || cols <= 0 || (cols & 7) != 0) return 0;
    if ((reinterpret_cast<uintptr_t>(W) & 15) != 0 || scratch == nullptr || (reinterpret_cast<uintptr_t>(scratch) & 15) != 0) return 0;
    if ((size_t)batch * cols * 2 > scratch_bytes) return 0;
    // the kernel keeps the first 64 KiB (160 KiB on gfx950) of the activations in LDS; a few percent more are read from memory, much
    // more is the job of the original's `big` variant, which is not ported
    const int lds_elems = (dev.lds_bytes > 64 * 1024 && std::string(dev.gcn).rfind("gfx95", 0) == 0 ? 160 * 1024 : 64 * 1024) / 2;
    if ((double)cols * batch > 1.2 * lds_elems) return 0;
    return kind == KIND_F16 ? run<sk_half>(s, W, X, Y, bias, rows, cols, batch, scratch)
                            : run<sk_bf16>(s, W, X, Y, bias, rows, cols, batch, scratch);
}
