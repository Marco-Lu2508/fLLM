/*
 * fLLM HIP backend: 4-bit group-quantized weights (Q4Z32 / Q4Z64 / Q4Z128 and the act-order forms Q4ZP*: AWQ and GPTQ checkpoints) on AMD GPUs.
 *
 * On NVIDIA these run through vLLM's Marlin GEMM (../cuda/marlin_vllm_glue.cu), which needs tensor-core PTX. This file provides the same
 * C++ entry points (declared in ../cuda/fllm_cuda_internal.h) for CDNA, RDNA3 and RDNA4, so that Q4Z weights stay 4-bit on the device
 * instead of being expanded to F32:
 *
 *   fllm_q4z_repack   AutoAWQ-ordered int4 words [k][n/8] -> one row of packed words per output row, see "Weight layout"
 *   up to 4 tokens    RDNA3 / RDNA4: wvSplitK_int4_hf_sml_ of vLLM (below); elsewhere a wave-per-row GEMV of our own
 *   more tokens       the matrix-core GEMM of hip_gemm.cuh with a Q4Z decoder (dequantize to FP16 in LDS, MFMA / WMMA)
 *
 * Weight layout. Device buffer of a weight, as built by upload_q4z in backend/cuda.rs (rows = output rows n, cols = inputs k,
 * G = cols / group):
 *   [rows * cols / 2 bytes: packed codes][pad to 256][G * rows halves: scales, permuted per block of 64 rows (marlin_permute_scales)]
 *   [pad to 256][G * rows / 8 words: zero points, 8 nibbles per word in Marlin order]
 * The codes of row n are cols / 8 little-endian words; word w holds k = 8 w .. 8 w + 7, the code of offset t in nibble
 * (t & 1) * 4 + (t >> 1), i.e. nibbles 0..7 hold t = 0, 2, 4, 6, 1, 3, 5, 7 (the order the magic-number dequantization of the vLLM kernel
 * expects: two codes of one half2 are four nibbles apart). Value of element (n, k) = (code - zero) * scale with the scale and zero of
 * group k / group of row n. For rows in the same block of 64 the scale of row n sits at position 8 * (n % 8) + (n % 64) / 8 of the block,
 * its zero point in nibble (s even ? s / 2 : 4 + s / 2) of the word at position / 8, s = position % 8 (the inverse of `marlin_arrays` in
 * backend/q4z.rs).
 *
 * Act-order weights (Q4ZP*) come with `perm`: the code at k pairs with the activation x[perm[k]]. The accumulate (fllm_q4z_acc) and
 * FP16-result (fllm_q4z_f16out) modes of the Marlin path are honoured, a fused gated activation (glu_gate / glu_up) is computed into
 * x first. FLLM_HIP_NO_Q4Z=1 makes fllm_q4z_supported() return false (the Rust side then expands the weights), FLLM_HIP_NO_Q4Z_SKINNY=1
 * switches the vLLM kernel off.
 *
 * The RDNA kernel (q4z_int4_sml_kernel) is derived from vLLM, csrc/rocm/skinny_gemms_int4.cu (wvSplitK_int4_hf_sml_, fp16 activations,
 * per-group scales with zero points), https://github.com/vllm-project/vllm, commit 7fad7db3309347a9d6288a09ed0993728e474a34.
 * Copyright contributors to the vLLM project. SPDX-License-Identifier: Apache-2.0
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
 * Changes against the original: the PyTorch glue is replaced by fllm_q4z_launch; scales and zero points are read through Q4zView (the
 * Marlin-permuted arrays of the fLLM upload) instead of [M][K/group] tensors; the FP32 activations are converted to FP16 (through the
 * act-order permutation) into the scratch buffer of the stream; the result goes through store_result (FP32, bias, accumulate, FP16
 * modes); the dequantized weights and the scale/zero fetch are shared by the tokens of a pass; only the `sml` kernel (activations fit
 * in LDS) with the fp16 path and zero points is ported (no bf16, no `big`), launched with wave32 only; the weight layout is the one above.
 *
 * Level of verification: L0 (compiles; never run on hardware).
 */
#include <cstdlib>

#include "../cuda/fllm_cuda_internal.h"
#include "hip_gemm.cuh"

using namespace fllm_hip;

namespace {

constexpr size_t align256(size_t v) { return (v + 255) & ~size_t(255); }

// nibble of a code word that holds the code of offset t (0..7) of its group of 8 inputs
__host__ __device__ __forceinline__ int nib_of(int t) { return ((t & 1) << 2) | ((t & 7) >> 1); }

struct Q4zView {
    const uint32_t *q;       // [rows][cols / 8] code words
    const uint16_t *scales;  // [groups][rows], permuted per block of 64 rows
    const uint32_t *zeros;   // [groups * rows / 8]
    int rows, cols, group;

    // position of row n inside the permuted scale / zero arrays of one group
    __device__ __forceinline__ static int slot(int n) { return (n & ~63) + 8 * (n & 7) + ((n & 63) >> 3); }

    __device__ __forceinline__ void scale_zero(int n, int g, float &scale, float &zero) const {
        const int pos = slot(n);
        const size_t flat = (size_t)g * rows + pos;
        scale = h2f_bits(scales[flat]);
        const int s = pos & 7;  // rows % 8 == 0, so flat % 8 == pos % 8
        const int nib = (s & 1) ? 4 + (s >> 1) : (s >> 1);
        zero = (float)((zeros[flat >> 3] >> (4 * nib)) & 15u);
    }

    __device__ __forceinline__ const uint32_t *row_words(int n) const { return q + (size_t)n * (cols / 8); }

    // decoder concept of hip_gemm.cuh: 16 weights of row `row`, k = 16 * run ..
    __device__ __forceinline__ void decode(int row, int run, float *w) const {
        const int k0 = run * 16;
        if (k0 >= cols) {
#pragma unroll
            for (int i = 0; i < 16; i++) w[i] = 0.f;
            return;
        }
        float scale, zero;
        scale_zero(row, k0 / group, scale, zero);
        const uint2 raw = *reinterpret_cast<const uint2 *>(row_words(row) + (k0 >> 3));
        const uint32_t word[2] = {raw.x, raw.y};
#pragma unroll
        for (int i = 0; i < 16; i++) w[i] = ((float)((word[i >> 3] >> (4 * nib_of(i))) & 15u) - zero) * scale;
    }
};

Q4zView view_of(const void *w, int rows, int cols, int group) {
    const size_t groups = (size_t)cols / group;
    const size_t scale_at = align256((size_t)rows * cols / 2);
    const size_t zero_at = scale_at + align256(groups * rows * 2);
    const uint8_t *base = static_cast<const uint8_t *>(w);
    return Q4zView{reinterpret_cast<const uint32_t *>(base), reinterpret_cast<const uint16_t *>(base + scale_at),
                   reinterpret_cast<const uint32_t *>(base + zero_at), rows, cols, group};
}

// ---- repack: AutoAWQ words -> code words per output row -------------------------------------------------------------------------
__global__ void q4z_repack_kernel(const uint32_t *__restrict__ awq, uint32_t *__restrict__ out, int k, int n) {
    // nibble of an AutoAWQ word that holds output column c % 8
    const int slot_of[8] = {0, 4, 1, 5, 2, 6, 3, 7};
    const size_t total = (size_t)n * (k / 8);
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const int row = (int)(i / (k / 8)), w = (int)(i % (k / 8));
        const int shift = 4 * slot_of[row & 7];
        uint32_t word = 0;
#pragma unroll
        for (int t = 0; t < 8; t++) {
            const uint32_t code = (awq[(size_t)(8 * w + t) * (n / 8) + row / 8] >> shift) & 15u;
            word |= code << (4 * nib_of(t));
        }
        out[i] = word;
    }
}

// activations FP32 -> FP16, through the act-order permutation when there is one
__global__ void act16_kernel(const float *__restrict__ x, __half *__restrict__ out, const int *__restrict__ perm, int batch, int cols) {
    const size_t total = (size_t)batch * cols;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const size_t b = i / cols;
        const int k = (int)(i % cols);
        out[i] = __float2half(x[b * cols + (perm ? perm[k] : k)]);
    }
}

// ---- GEMV of our own: batch <= 4 on every architecture ----------------------------------------------------------------------------
// One wave per output row; NB tokens at a time (b0 .. b0 + NB - 1).
template <int NB>
__global__ void __launch_bounds__(256) q4z_gemv_kernel(Q4zView v, GemmArgs a, int b0) {
    const int wave = threadIdx.x / warpSize, lane = threadIdx.x % warpSize;
    const int row = blockIdx.x * (blockDim.x / warpSize) + wave;
    if (row >= v.rows) return;  // the whole wave leaves together
    const uint32_t *rw = v.row_words(row);
    bool vec[NB];
#pragma unroll
    for (int b = 0; b < NB; b++) vec[b] = ((reinterpret_cast<uintptr_t>(a.X + (size_t)(b0 + b) * v.cols)) & 15) == 0;
    float acc[NB];
#pragma unroll
    for (int b = 0; b < NB; b++) acc[b] = 0.f;
    const int steps = v.cols / 32;  // 32 codes (4 words, 16 bytes) per step; a step never straddles a group (group % 32 == 0)
    for (int st = lane; st < steps; st += warpSize) {
        const uint4 qw = *reinterpret_cast<const uint4 *>(rw + (size_t)st * 4);
        const uint32_t w[4] = {qw.x, qw.y, qw.z, qw.w};
        const int k0 = st * 32;
        float scale, zero;
        v.scale_zero(row, k0 / v.group, scale, zero);
#pragma unroll
        for (int b = 0; b < NB; b++) {
            if (b0 + b >= a.batch) continue;
            const float *xr = a.X + (size_t)(b0 + b) * v.cols;
            float s1 = 0.f, s2 = 0.f;
            if (a.perm) {
#pragma unroll
                for (int j = 0; j < 32; j++) {
                    const float x = xr[a.perm[k0 + j]];
                    s1 = fmaf(x, (float)((w[j >> 3] >> (4 * nib_of(j))) & 15u), s1);
                    s2 += x;
                }
            } else if (vec[b]) {
                const float4 *xp = reinterpret_cast<const float4 *>(xr + k0);
#pragma unroll
                for (int i = 0; i < 8; i++) {
                    const float4 x4 = xp[i];
                    const float xs[4] = {x4.x, x4.y, x4.z, x4.w};
#pragma unroll
                    for (int c = 0; c < 4; c++) {
                        const int j = 4 * i + c;
                        s1 = fmaf(xs[c], (float)((w[j >> 3] >> (4 * nib_of(j))) & 15u), s1);
                        s2 += xs[c];
                    }
                }
            } else {
#pragma unroll
                for (int j = 0; j < 32; j++) {
                    const float x = xr[k0 + j];
                    s1 = fmaf(x, (float)((w[j >> 3] >> (4 * nib_of(j))) & 15u), s1);
                    s2 += x;
                }
            }
            acc[b] += scale * (s1 - zero * s2);
        }
    }
#pragma unroll
    for (int b = 0; b < NB; b++) {
        for (int o = warpSize / 2; o > 0; o >>= 1) acc[b] += __shfl_xor(acc[b], o);
    }
    if (lane == 0) {
#pragma unroll
        for (int b = 0; b < NB; b++)
            if (b0 + b < a.batch) store_result(a, b0 + b, row, acc[b]);
    }
}

void launch_gemv(cudaStream_t s, const Q4zView &v, const GemmArgs &a) {
    const FllmHipDevice &dev = fllm_hip_device();
    const int waves = 256 / (dev.wave > 0 ? dev.wave : 64);
    const dim3 grid((a.rows + waves - 1) / waves), block(256);
    for (int b0 = 0; b0 < a.batch; b0 += 4) {
        switch (a.batch - b0 < 4 ? a.batch - b0 : 4) {
            case 1: q4z_gemv_kernel<1><<<grid, block, 0, s>>>(v, a, b0); break;
            case 2: q4z_gemv_kernel<2><<<grid, block, 0, s>>>(v, a, b0); break;
            case 3: q4z_gemv_kernel<3><<<grid, block, 0, s>>>(v, a, b0); break;
            default: q4z_gemv_kernel<4><<<grid, block, 0, s>>>(v, a, b0); break;
        }
    }
}

// ---- RDNA3 / RDNA4: wvSplitK_int4_hf_sml_ of vLLM ---------------------------------------------------------------------------------
#if defined(FLLM_HIP_RDNA3) || defined(FLLM_HIP_RDNA4)
#define FLLM_Q4Z_SK 1
#endif
constexpr int SK_THRDS = 32, SK_WVPRGRP = 16, SK_ACHUNK = 16;
constexpr int SK_LDS_ELEMS = 64 * 1024 / 2;  // activations (FP16) that fit in LDS

typedef _Float16 sk_f16x2 __attribute__((ext_vector_type(2)));

template <typename T>
__device__ __forceinline__ T sk_loadnt(T *addr) {
    return __builtin_nontemporal_load(addr);
}

// To avoid LLVM silently upcasting to double
__device__ __forceinline__ unsigned int sk_min(uint32_t a, uint32_t b) { return min(a, b); }

// A: FP16 activations [N][K] (already permuted), the weights, scales and zeros come from `v`.
template <int YTILE, int UNRL, int N, int GS>
__global__ void __launch_bounds__(SK_WVPRGRP *SK_THRDS)
    q4z_int4_sml_kernel(Q4zView v, const __half *__restrict__ A, GemmArgs args, const int _WvPrGrp, const int CuCount) {
#if defined(FLLM_Q4Z_SK)
    constexpr int THRDS = SK_THRDS, WvPrGrp = SK_WVPRGRP, A_CHUNK = SK_ACHUNK, max_lds_len = SK_LDS_ELEMS;
    static_assert(GS % A_CHUNK == 0, "a group holds whole chunks");
    const int K = v.cols, M = v.rows;

    union bigTypeA {
        __half h[A_CHUNK];
        float f[A_CHUNK / 2];
    };
    union bigTypeW {
        uint8_t b[A_CHUNK / 2];
        uint32_t u32[A_CHUNK / 8];
        float f[A_CHUNK / 8];
    };

    // the activations live in LDS for the whole lifetime of the workgroup
    __shared__ __half s[max_lds_len];
    for (uint32_t k = 0; k < sk_min(K * N, max_lds_len); k += THRDS * WvPrGrp * A_CHUNK) {
        uint32_t k_in = k + ((threadIdx.y * THRDS + threadIdx.x) * A_CHUNK);
        if (k_in >= sk_min(K * N, max_lds_len)) break;
        *((bigTypeA *)(&s[k_in])) = *((const bigTypeA *)(&A[k_in]));
    }
    __syncthreads();

    if (threadIdx.y >= _WvPrGrp) return;

    uint32_t m = (blockIdx.x * _WvPrGrp + (threadIdx.y % _WvPrGrp)) * YTILE;

    float sum[N][YTILE];

    while (m < (uint32_t)M) {
        for (int i = 0; i < YTILE; i++)
            for (int n = 0; n < N; n++) sum[n][i] = 0;

        bigTypeA bigA[N][UNRL];
        bigTypeW bigB[YTILE][UNRL];

        for (uint32_t k1 = 0; k1 < (uint32_t)K; k1 += THRDS * A_CHUNK * UNRL) {
            // packed weights from memory (non-temporal)
#pragma unroll
            for (uint32_t k2 = 0; k2 < UNRL; k2++) {
                uint32_t k = k1 + k2 * THRDS * A_CHUNK;
                uint32_t k_ = k + threadIdx.x * A_CHUNK;
                if (k_ >= (uint32_t)K) break;
                for (int y = 0; y < YTILE; y++) {
                    const float *src = (const float *)(v.row_words(m + y) + (k_ >> 3));
#pragma unroll
                    for (int i = 0; i < A_CHUNK / 8; i++) bigB[y][k2].f[i] = sk_loadnt((float *)&src[i]);
                }
            }
            // activations from LDS
#pragma unroll
            for (uint32_t k2 = 0; k2 < UNRL; k2++) {
                uint32_t k = k1 + k2 * THRDS * A_CHUNK;
                uint32_t k_ = k + threadIdx.x * A_CHUNK;
                if (k_ >= (uint32_t)K) break;
                for (int n = 0; n < N; n++) bigA[n][k2] = *((const bigTypeA *)(&(s[k_ + K * n])));
            }
            // dequantize with the magic-number trick, subtract the zero point, dot with the activations
#pragma unroll
            for (uint32_t k2 = 0; k2 < UNRL; k2++) {
                uint32_t k = k1 + k2 * THRDS * A_CHUNK;
                uint32_t k_ = k + threadIdx.x * A_CHUNK;
                if (k_ >= (uint32_t)K) break;
#pragma unroll
                for (int y = 0; y < YTILE; y++) {
                    bigTypeA cvtB;
                    constexpr uint32_t FP16_MAGIC = 0x64006400u;  // two halves 1024
                    constexpr uint32_t BIAS_LO = 0x64006400u;     // 1024: removes the magic from the low nibbles
                    constexpr uint32_t SCALE16 = 0x2C002C00u;     // 1/16
                    constexpr uint32_t BIAS_HI = 0xD400D400u;     // -64: the high nibbles carry 1024 + 16 q
#pragma unroll
                    for (uint32_t w = 0; w < A_CHUNK / 8; w++) {
                        uint32_t qa = bigB[y][k2].u32[w];
                        const uint32_t lo0 = (qa & 0x000F000Fu) | FP16_MAGIC;
                        const uint32_t hi0 = (qa & 0x00F000F0u) | FP16_MAGIC;
                        qa >>= 8;
                        const uint32_t lo1 = (qa & 0x000F000Fu) | FP16_MAGIC;
                        const uint32_t hi1 = (qa & 0x00F000F0u) | FP16_MAGIC;
                        const sk_f16x2 bias_lo = __builtin_bit_cast(sk_f16x2, BIAS_LO), scale16 = __builtin_bit_cast(sk_f16x2, SCALE16),
                                       bias_hi = __builtin_bit_cast(sk_f16x2, BIAS_HI);
                        const sk_f16x2 c0 = __builtin_bit_cast(sk_f16x2, lo0) - bias_lo;
                        const sk_f16x2 c1 = __builtin_bit_cast(sk_f16x2, hi0) * scale16 + bias_hi;
                        const sk_f16x2 c2 = __builtin_bit_cast(sk_f16x2, lo1) - bias_lo;
                        const sk_f16x2 c3 = __builtin_bit_cast(sk_f16x2, hi1) * scale16 + bias_hi;
                        cvtB.f[w * 4 + 0] = __builtin_bit_cast(float, c0);
                        cvtB.f[w * 4 + 1] = __builtin_bit_cast(float, c1);
                        cvtB.f[w * 4 + 2] = __builtin_bit_cast(float, c2);
                        cvtB.f[w * 4 + 3] = __builtin_bit_cast(float, c3);
                    }
                    float scale, zero;
                    v.scale_zero((int)(m + y), (int)(k_ / GS), scale, zero);
                    const __half zp = __float2half(zero);
#pragma unroll
                    for (uint32_t b = 0; b < A_CHUNK; b++) cvtB.h[b] = cvtB.h[b] - zp;
#pragma unroll
                    for (uint32_t n = 0; n < N; n++) {
                        float partial = 0;
#pragma unroll
                        for (uint32_t b = 0; b < A_CHUNK / 2; b++)
                            partial = __builtin_amdgcn_fdot2(__builtin_bit_cast(sk_f16x2, bigA[n][k2].f[b]), __builtin_bit_cast(sk_f16x2, cvtB.f[b]), partial, false);
                        sum[n][y] += partial * scale;
                    }
                }
            }
        }

        // wave32 reduction: row_shr within 16-lane rows, then across the two rows (the last lane holds the total)
        for (int n = 0; n < N; n++)
            for (int y = 0; y < YTILE; y++) {
                sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x118, 0xf, 0xf, 1);  // shr8
                sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x114, 0xf, 0xf, 1);  // shr4
                sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x112, 0xf, 0xf, 1);  // shr2
                sum[n][y] += __builtin_amdgcn_mov_dpp(sum[n][y], 0x111, 0xf, 0xf, 1);  // shr1
                sum[n][y] += __shfl_xor(sum[n][y], 16);
            }

        if (threadIdx.x == (THRDS - 1)) {
            for (int n = 0; n < N; n++)
                for (int i = 0; i < YTILE; i++) store_result(args, n, (int)(m + i), sum[n][i]);
        }
        m += CuCount * _WvPrGrp * YTILE;
    }
#endif
}

// waves per group that keep the persistent loop balanced (original: mindiv_int4)
int mindiv_int4(int N, int div1, int div2) {
    int nPrRnd = div1 * div2;
    int limit = div2 < 13 ? div2 : 13;
    int rnds[16];
    for (int i = 0; i < limit; i++) {
        rnds[i] = (N + nPrRnd - 1) / nPrRnd;
        nPrRnd -= div1;
    }
    for (int i = limit - 1; i >= 0; i--)
        if (rnds[0] == rnds[i]) return (div2 - i);
    return 0;
}

enum SkCfg { SK_4_1, SK_2_4, SK_2_2, SK_1_4 };

// The (YTILE, UNRL) table of the original (WVSPLIT_INT4G_TILE, activations fit in LDS)
SkCfg pick_cfg(int sYT, int N, int K) {
    if (N >= 4 && sYT >= 480) return SK_4_1;
    if (N >= 3 && sYT >= 40) return SK_4_1;
    if (N >= 3 && sYT < 40 && (K <= 2048 || K >= 4096)) return SK_2_4;
    if (N >= 3 && sYT < 40) return SK_2_2;
    if (N >= 2) return SK_2_2;
    if (sYT >= 30) return SK_2_4;
    return SK_1_4;
}

template <int YTILE, int UNRL, int N, int GS>
void launch_sk(cudaStream_t s, const Q4zView &v, const __half *a16, const GemmArgs &args, int cus) {
    const int wv = mindiv_int4(v.rows, cus * YTILE, SK_WVPRGRP);
    q4z_int4_sml_kernel<YTILE, UNRL, N, GS><<<dim3(cus), dim3(SK_THRDS, SK_WVPRGRP), 0, s>>>(v, a16, args, wv, cus);
}

template <int N, int GS>
void dispatch_cfg(SkCfg cfg, cudaStream_t s, const Q4zView &v, const __half *a16, const GemmArgs &args, int cus) {
    switch (cfg) {
        case SK_4_1: launch_sk<4, 1, N, GS>(s, v, a16, args, cus); break;
        case SK_2_4: launch_sk<2, 4, N, GS>(s, v, a16, args, cus); break;
        case SK_2_2: launch_sk<2, 2, N, GS>(s, v, a16, args, cus); break;
        case SK_1_4: launch_sk<1, 4, N, GS>(s, v, a16, args, cus); break;
    }
}

template <int GS>
void dispatch_n(int N, SkCfg cfg, cudaStream_t s, const Q4zView &v, const __half *a16, const GemmArgs &args, int cus) {
    switch (N) {
        case 1: dispatch_cfg<1, GS>(cfg, s, v, a16, args, cus); break;
        case 2: dispatch_cfg<2, GS>(cfg, s, v, a16, args, cus); break;
        case 3: dispatch_cfg<3, GS>(cfg, s, v, a16, args, cus); break;
        default: dispatch_cfg<4, GS>(cfg, s, v, a16, args, cus); break;
    }
}

// true when the vLLM kernel took the call
bool launch_skinny(cudaStream_t s, const Q4zView &v, const GemmArgs &args, void *io) {
    static const bool enabled = fllm_hip_enabled("Q4Z_SKINNY");
    const FllmHipDevice &dev = fllm_hip_device();
    if (!enabled || io == nullptr || (dev.arch != FLLM_HIP_ARCH_RDNA3 && dev.arch != FLLM_HIP_ARCH_RDNA4) || dev.cus <= 0) return false;
    const int N = args.batch, K = v.cols;
    if (N < 1 || N > 4 || (K & 15) != 0 || (size_t)K * N > (size_t)SK_LDS_ELEMS || (v.rows & 7) != 0) return false;
    if ((reinterpret_cast<uintptr_t>(io) & 15) != 0) return false;
    __half *a16 = static_cast<__half *>(io);
    const size_t total = (size_t)N * K;
    act16_kernel<<<(unsigned)((total + 255) / 256 < 1024 ? (total + 255) / 256 : 1024), 256, 0, s>>>(args.X, a16, args.perm, N, K);
    const int sYT = (v.rows + dev.cus * 4 - 1) / (dev.cus * 4);
    const SkCfg cfg = pick_cfg(sYT, N, K);
    // rows % 8 == 0 is enough for YTILE 1, 2 and 4
    if (v.group == 32) dispatch_n<32>(N, cfg, s, v, a16, args, dev.cus);
    else if (v.group == 64) dispatch_n<64>(N, cfg, s, v, a16, args, dev.cus);
    else dispatch_n<128>(N, cfg, s, v, a16, args, dev.cus);
    return true;
}

}  // namespace

// ---- entry points (the ones declared in fllm_cuda_internal.h for the Marlin path) ------------------------------------------------
bool fllm_q4z_supported(int rows, int cols, int group) {
    static const bool enabled = fllm_hip_enabled("Q4Z");
    if (!enabled || fllm_hip_device().arch == FLLM_HIP_ARCH_OTHER) return false;
    if (group != 32 && group != 64 && group != 128) return false;
    // the host-side rearrangement (backend/q4z.rs) works in blocks of 64 rows and 64 inputs
    return rows > 0 && cols > 0 && rows % 64 == 0 && cols % 64 == 0 && cols % group == 0;
}
// FP16 copy of the activations for the vLLM kernel; no split-K partials or lock workspace
size_t fllm_q4z_io_bytes(int, int cols, int batch) { return ((size_t)batch * cols * 2 + 255) & ~size_t(255); }
size_t fllm_q4z_ctmp_bytes(int, int) { return 0; }
size_t fllm_q4z_workspace_bytes(int) { return 0; }

int fllm_q4z_launch(cudaStream_t s, int, int, const void *w, int group, const float *x, float *y, const float *bias, int rows, int cols, int batch,
                    void *io, void *, void *, const float *glu_gate, const float *glu_up, int glu_mode, bool, const int *perm) {
    if (!fllm_q4z_supported(rows, cols, group) || batch <= 0) return -1;
    if (glu_gate && glu_up) {
        // the gated activation feeding this GEMM: compute it into x (the Marlin path folds it into the activation conversion)
        const size_t count = (size_t)batch * cols;
        const int rc = fllm_glu_half ? fllm_glu_launch_f16(s, glu_gate, glu_up, const_cast<float *>(x), count, glu_mode)
                                     : fllm_glu_launch(s, glu_gate, glu_up, const_cast<float *>(x), count, glu_mode);
        if (rc != 0) return rc;
    }
    const Q4zView v = view_of(w, rows, cols, group);
    const int mode = (fllm_q4z_acc ? 1 : 0) | (fllm_q4z_f16out ? 2 : 0);
    const GemmArgs args{x, y, bias, perm, rows, cols, batch, mode, fllm_q4z_acc_scale};
    static const bool no_mma = !fllm_hip_enabled("MMA");
    if (batch <= 4 && launch_skinny(s, v, args, io)) {
        // vLLM wvSplitK_int4 kernel (RDNA3 / RDNA4)
    } else if (batch > 4 && !no_mma && (reinterpret_cast<uintptr_t>(x) & 15) == 0 && (cols & 15) == 0) {
        launch_mma_gemm(s, v, args);
    } else {
        launch_gemv(s, v, args);
    }
    return fllm_check_launch("q4z") != 0 ? 1 : 0;
}

// Synchronous, like the Marlin repack: `awq_words` [k][n / 8] (AutoAWQ order), `out` k * n / 2 bytes (n rows of k / 8 words).
int fllm_q4z_repack(int dev, const void *awq_words, void *out, int k, int n) {
    if (hipSetDevice(dev) != hipSuccess) return fllm_set_error("hipSetDevice(%d) failed", dev);
    if (k <= 0 || n <= 0 || (k & 7) || (n & 7)) return fllm_set_error("Q4Z repack: bad shape k=%d n=%d", k, n);
    const size_t total = (size_t)n * (k / 8);
    const unsigned grid = (unsigned)((total + 255) / 256 < 65535 ? (total + 255) / 256 : 65535);
    q4z_repack_kernel<<<grid, 256>>>(static_cast<const uint32_t *>(awq_words), static_cast<uint32_t *>(out), k, n);
    if (int e = fllm_check_launch("q4z repack")) return e;
    const hipError_t e = hipDeviceSynchronize();
    if (e != hipSuccess) return fllm_set_error("q4z repack: %s", hipGetErrorString(e));
    return 0;
}
