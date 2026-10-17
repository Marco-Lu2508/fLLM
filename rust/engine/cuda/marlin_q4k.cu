/*
 * SPDX-License-Identifier: Apache-2.0
 * Copyright (C) Marlin.2024 Elias Frantar (elias.frantar@ist.ac.at)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *         http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * Adapted for fLLM:
 * - Grouped affine quantization with group_size = 32 (group_blocks = 2) for Q4_K.
 * - Direct register dequantization of unsigned 4-bit nibbles with fused scale and min:
 *     val = q * scale - min = q * scale + neg_min (HFMA2).
 * - Direct FP32 accumulation and FP32 output with optional fused bias addition.
 */

#include "common.cuh"
#include "fllm_cuda_internal.h"
#include <cuda_fp16.h>

#if !defined(__HIPCC__)

// Defined for every architecture: the host code that launches it is compiled for all of them, and nvcc needs the definition in the
// view that generates the launch stub (otherwise the link fails with "undefined hidden symbol" on an sm_75-only build).
__global__ void fp32_to_fp16_kernel(const float2 *__restrict__ x, const float2 *__restrict__ up, int mode,
                                     half2 *__restrict__ out, size_t n2_valid, size_t n2_total) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (size_t i = idx; i < n2_total; i += (size_t)gridDim.x * blockDim.x) {
        if (i < n2_valid) {
            float2 val = x[i];
            if (up) {
                float2 u = up[i];
                val.x = fllm_glu(val.x, u.x, mode);
                val.y = fllm_glu(val.y, u.y, mode);
            }
            out[i] = __floats2half2_rn(val.x, val.y);
        } else {
            out[i] = __float2half2_rn(0.0f);
        }
    }
}

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
// Marlin requires Ampere or newer (sm_80+) for cp.async and m16n8k16.
// Turing and older architectures fall back to other kernels in fLLM.
#else

__host__ __device__ constexpr int ceildiv(int a, int b) {
    return (a + b - 1) / b;
}

template <typename T, int n>
struct Vec {
    T elems[n];
    __device__ T &operator[](int i) { return elems[i]; }
    __device__ const T &operator[](int i) const { return elems[i]; }
};

using I4 = Vec<int, 4>;
using FragA = Vec<half2, 4>;
using FragB = Vec<half2, 2>;
using FragC = Vec<float, 4>;
using FragS = Vec<half2, 1>;

__device__ inline void cp_async4_pred(void *smem_ptr, const void *glob_ptr, bool pred = true) {
    const int BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   .reg .pred p;\n"
        "   setp.ne.b32 p, %0, 0;\n"
        "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
        "}\n" :: "r"((int)pred), "r"(smem), "l"(glob_ptr), "n"(BYTES)
    );
}

__device__ inline void cp_async4_stream(void *smem_ptr, const void *glob_ptr) {
    const int BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{\n"
        "   .reg .b64 p;\n"
        "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;\n"
        "   cp.async.cg.shared.global.L2::cache_hint [%0], [%1], %2, p;\n"
        "}\n" :: "r"(smem), "l"(glob_ptr), "n"(BYTES)
    );
}

__device__ inline void cp_async_fence() {
    asm volatile("cp.async.commit_group;\n" ::);
}

template <int n>
__device__ inline void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(n));
}

__device__ inline void mma(const FragA &a_frag, const FragB &frag_b, FragC &frag_c) {
    const uint32_t *a = reinterpret_cast<const uint32_t *>(&a_frag);
    const uint32_t *b = reinterpret_cast<const uint32_t *>(&frag_b);
    float *c = reinterpret_cast<float *>(&frag_c);
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3])
    );
}

__device__ inline void ldsm4(FragA &frag_a, const void *smem_ptr) {
    uint32_t *a = reinterpret_cast<uint32_t *>(&frag_a);
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(smem)
    );
}

template <int lut>
__device__ inline int lop3(int a, int b, int c) {
    int res;
    asm volatile(
        "lop3.b32 %0, %1, %2, %3, %4;\n"
        : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut)
    );
    return res;
}

// Unsigned 4-bit nibbles dequantized directly into full fp16 in [0.0, 15.0]
__device__ inline FragB dequant_u4(int q) {
    const int LO = 0x000f000f;
    const int HI = 0x00f000f0;
    const int EX = 0x64006400; // 1024.0 in half2
    int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
    int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
    const int SUB = 0x64006400; // 1024.0
    const int MUL = 0x2c002c00; // 1/16
    const int ADD = 0xd400d400; // -64.0
    FragB frag_b;
    frag_b[0] = __hsub2(*reinterpret_cast<half2 *>(&lo), *reinterpret_cast<const half2 *>(&SUB));
    frag_b[1] = __hfma2(*reinterpret_cast<half2 *>(&hi), *reinterpret_cast<const half2 *>(&MUL), *reinterpret_cast<const half2 *>(&ADD));
    return frag_b;
}

// val = q * scale + neg_min (HFMA2: single instruction for q * scale - min)
__device__ inline void scale_and_sub(FragB &frag_b, FragS &frag_s, FragS &frag_m, int i) {
    half2 s = __half2half2(reinterpret_cast<__half *>(&frag_s)[i]);
    half2 neg_m = __half2half2(reinterpret_cast<__half *>(&frag_m)[i]);
    frag_b[0] = __hfma2(frag_b[0], s, neg_m);
    frag_b[1] = __hfma2(frag_b[1], s, neg_m);
}

__device__ inline void barrier_acquire(int *lock, int count) {
    if (threadIdx.x == 0) {
        int state = -1;
        do {
            asm volatile("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
        } while (state != count);
    }
    __syncthreads();
}

__device__ inline void barrier_release(int *lock, bool reset = false) {
    __syncthreads();
    if (threadIdx.x == 0) {
        if (reset) {
            lock[0] = 0;
            return;
        }
        int val = 1;
        asm volatile("fence.acq_rel.gpu;\n");
        asm volatile("red.relaxed.gpu.global.add.s32 [%0], %1;\n" :: "l"(lock), "r"(val));
    }
}

template <
    const int threads,
    const int thread_m_blocks,
    const int thread_n_blocks,
    const int thread_k_blocks,
    const int stages,
    const int group_blocks = 2
>
__global__ void MarlinQ4K(
    const int4 *__restrict__ A,
    const int4 *__restrict__ B,
    float *__restrict__ C,
    const int4 *__restrict__ s,
    const int4 *__restrict__ m,
    const float *__restrict__ bias,
    int prob_m,
    int prob_n,
    int prob_k,
    int *locks,
    float *C_tmp
) {
    int parallel = 1;
    if (prob_m > 16 * thread_m_blocks) {
        parallel = prob_m / (16 * thread_m_blocks);
        prob_m = 16 * thread_m_blocks;
    }

    int k_tiles = prob_k / 16 / thread_k_blocks;
    int n_tiles = prob_n / 16 / thread_n_blocks;
    int iters = k_tiles;

    int slice_row = (iters * blockIdx.x) % k_tiles;
    int slice_col_par = (iters * blockIdx.x) / k_tiles;
    int slice_col = slice_col_par;
    int slice_iters;
    int slice_count = 0;
    int slice_idx;

    if (slice_col_par >= n_tiles) {
        A += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_k / 8;
        C += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_n;
        locks += (slice_col_par / n_tiles) * n_tiles;
        slice_col = slice_col_par % n_tiles;
    }

    auto init_slice = [&]() {
        slice_iters = iters * (blockIdx.x + 1) - (k_tiles * slice_col_par + slice_row);
        if (slice_iters < 0 || slice_col_par >= n_tiles * parallel)
            slice_iters = 0;
        if (slice_iters == 0) return;
        if (slice_row + slice_iters > k_tiles)
            slice_iters = k_tiles - slice_row;
        slice_count = 1;
        slice_idx = 0;
        int col_first = iters * ceildiv(k_tiles * slice_col_par, iters);
        if (col_first <= k_tiles * (slice_col_par + 1)) {
            int col_off = col_first - k_tiles * slice_col_par;
            slice_count = ceildiv(k_tiles - col_off, iters);
            if (col_off > 0) slice_count++;
            int delta_first = iters * blockIdx.x - col_first;
            if (delta_first < 0 || (col_off == 0 && delta_first == 0))
                slice_idx = slice_count - 1;
            else {
                slice_idx = slice_count - 1 - delta_first / iters;
                if (col_off > 0) slice_idx--;
            }
        }
        if (slice_col == n_tiles) {
            A += 16 * thread_m_blocks * prob_k / 8;
            C += 16 * thread_m_blocks * prob_n;
            locks += n_tiles;
            slice_col = 0;
        }
    };
    init_slice();

    int a_gl_stride = prob_k / 8;
    constexpr int a_sh_stride = 16 * thread_k_blocks / 8;
    constexpr int a_gl_rd_delta_o = 16 * thread_k_blocks / 8;
    int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o);
    constexpr int a_sh_wr_delta = a_sh_stride * (threads / a_gl_rd_delta_o);
    constexpr int a_sh_rd_delta_o = 2 * ((threads / 32) / (thread_n_blocks / 4));
    constexpr int a_sh_rd_delta_i = a_sh_stride * 16;
    constexpr int a_sh_stage = a_sh_stride * (16 * thread_m_blocks);
    constexpr int a_sh_wr_iters = ceildiv(a_sh_stage, a_sh_wr_delta);

    int b_gl_stride = 16 * prob_n / 32;
    constexpr int b_sh_stride = 32 * thread_n_blocks / 4;
    int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks;
    int b_gl_rd_delta_i = b_gl_stride * (threads / b_sh_stride);
    constexpr int b_sh_wr_delta = threads;
    constexpr int b_sh_rd_delta = threads;
    constexpr int b_sh_stage = b_sh_stride * thread_k_blocks;
    constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;

    int s_gl_stride = prob_n / 8;
    constexpr int s_sh_stride = 16 * thread_n_blocks / 8;
    constexpr int s_groups_per_tile = (thread_k_blocks >= 2) ? (thread_k_blocks / 2) : 1;
    constexpr int s_sh_stage = s_sh_stride * s_groups_per_tile;
    int s_gl_rd_delta = s_gl_stride * s_groups_per_tile;

    int a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
    a_gl_rd += a_gl_rd_delta_o * slice_row;
    int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
    int a_sh_rd = a_sh_stride * ((threadIdx.x % 32) % 16) + (threadIdx.x % 32) / 16;
    a_sh_rd += 2 * ((threadIdx.x / 32) / (thread_n_blocks / 4));

    int b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);
    b_gl_rd += b_sh_stride * slice_col;
    b_gl_rd += b_gl_rd_delta_o * slice_row;
    int b_sh_wr = threadIdx.x;
    int b_sh_rd = threadIdx.x;

    int s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / 2) + s_sh_stride * slice_col + threadIdx.x;
    int s_sh_wr = threadIdx.x;
    int s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + (threadIdx.x % 32) / 4;

    bool a_sh_wr_pred[a_sh_wr_iters];
    #pragma unroll
    for (int i = 0; i < a_sh_wr_iters; i++)
        a_sh_wr_pred[i] = a_sh_wr_delta * i + a_sh_wr < a_sh_stride * prob_m;
    bool s_sh_wr_pred = threadIdx.x < s_sh_stride;

    auto transform_a = [&](int i) {
        int row = i / a_gl_rd_delta_o;
        return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ row;
    };

    int a_sh_wr_trans[a_sh_wr_iters];
    #pragma unroll
    for (int i = 0; i < a_sh_wr_iters; i++)
        a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);
    int a_sh_rd_trans[b_sh_wr_iters][thread_m_blocks];
    #pragma unroll
    for (int i = 0; i < b_sh_wr_iters; i++) {
        #pragma unroll
        for (int j = 0; j < thread_m_blocks; j++)
            a_sh_rd_trans[i][j] = transform_a(a_sh_rd_delta_o * i + a_sh_rd_delta_i * j + a_sh_rd);
    }

    const int4 *B_ptr[b_sh_wr_iters];
    #pragma unroll
    for (int i = 0; i < b_sh_wr_iters; i++)
        B_ptr[i] = B + b_gl_rd_delta_i * i + b_gl_rd;

    extern __shared__ int4 sh[];
    int4 *sh_a = sh;
    int4 *sh_b = sh_a + (stages * a_sh_stage);
    int4 *sh_s = sh_b + (stages * b_sh_stage);
    int4 *sh_m = sh_s + (stages * s_sh_stage);

    FragA frag_a[2][thread_m_blocks];
    I4 frag_b_quant[2];
    FragC frag_c[thread_m_blocks][4][2];
    FragS frag_s[2][4];
    FragS frag_m[2][4];

    auto zero_accums = [&]() {
        #pragma unroll
        for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++)
            reinterpret_cast<float *>(frag_c)[i] = 0.0f;
    };

    auto fetch_to_shared = [&](int pipe, int a_off, bool pred = true) {
        if (pred) {
            int4 *sh_a_stage = sh_a + a_sh_stage * pipe;
            #pragma unroll
            for (int i = 0; i < a_sh_wr_iters; i++) {
                cp_async4_pred(
                    &sh_a_stage[a_sh_wr_trans[i]],
                    &A[a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * a_off],
                    a_sh_wr_pred[i]
                );
            }
            int4 *sh_b_stage = sh_b + b_sh_stage * pipe;
            #pragma unroll
            for (int i = 0; i < b_sh_wr_iters; i++) {
                cp_async4_stream(&sh_b_stage[b_sh_wr_delta * i + b_sh_wr], B_ptr[i]);
                B_ptr[i] += b_gl_rd_delta_o;
            }
            int4 *sh_s_stage = sh_s + s_sh_stage * pipe;
            int4 *sh_m_stage = sh_m + s_sh_stage * pipe;
            if (s_sh_wr_pred) {
                #pragma unroll
                for (int g = 0; g < s_groups_per_tile; g++) {
                    cp_async4_stream(&sh_s_stage[s_sh_stride * g + s_sh_wr], &s[s_gl_rd + s_gl_stride * g]);
                    cp_async4_stream(&sh_m_stage[s_sh_stride * g + s_sh_wr], &m[s_gl_rd + s_gl_stride * g]);
                }
            }
            s_gl_rd += s_gl_rd_delta;
        }
        cp_async_fence();
    };

    auto wait_for_stage = [&]() {
        cp_async_wait<stages - 2>();
        __syncthreads();
    };

    auto fetch_to_registers = [&](int k, int pipe) {
        int cur_grp = (k % s_groups_per_tile);
        int4 *sh_s_stage = sh_s + s_sh_stage * pipe;
        int4 *sh_m_stage = sh_m + s_sh_stage * pipe;
        reinterpret_cast<int4 *>(&frag_s[k % 2])[0] = sh_s_stage[s_sh_rd + cur_grp * s_sh_stride];
        reinterpret_cast<int4 *>(&frag_m[k % 2])[0] = sh_m_stage[s_sh_rd + cur_grp * s_sh_stride];

        int4 *sh_a_stage = sh_a + a_sh_stage * pipe;
        #pragma unroll
        for (int i = 0; i < thread_m_blocks; i++)
            ldsm4(frag_a[k % 2][i], &sh_a_stage[a_sh_rd_trans[k % b_sh_wr_iters][i]]);
        int4 *sh_b_stage = sh_b + b_sh_stage * pipe;
        frag_b_quant[k % 2] = *reinterpret_cast<I4 *>(&sh_b_stage[b_sh_rd_delta * (k % b_sh_wr_iters) + b_sh_rd]);
    };

    auto matmul = [&](int k) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
            int b_quant = frag_b_quant[k % 2][j];
            int b_quant_shift = b_quant >> 8;
            FragB frag_b0 = dequant_u4(b_quant);
            scale_and_sub(frag_b0, frag_s[k % 2][j], frag_m[k % 2][j], 0);
            FragB frag_b1 = dequant_u4(b_quant_shift);
            scale_and_sub(frag_b1, frag_s[k % 2][j], frag_m[k % 2][j], 1);
            #pragma unroll
            for (int i = 0; i < thread_m_blocks; i++) {
                mma(frag_a[k % 2][i], frag_b0, frag_c[i][j][0]);
                mma(frag_a[k % 2][i], frag_b1, frag_c[i][j][1]);
            }
        }
    };

    auto thread_block_reduce = [&]() {
        constexpr int red_off = threads / b_sh_stride / 2;
        if (red_off >= 1) {
            int red_idx = threadIdx.x / b_sh_stride;
            constexpr int red_sh_stride = b_sh_stride * 4 * 2;
            constexpr int red_sh_delta = b_sh_stride;
            int red_sh_rd = red_sh_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);

            #pragma unroll
            for (int m_block = 0; m_block < thread_m_blocks; m_block++) {
                #pragma unroll
                for (int i = red_off; i > 0; i /= 2) {
                    if (i <= red_idx && red_idx < 2 * i) {
                        #pragma unroll
                        for (int j = 0; j < 4 * 2; j++) {
                            int red_sh_wr = red_sh_delta * j + (red_sh_rd - red_sh_stride * i);
                            if (i < red_off) {
                                float *c_rd = reinterpret_cast<float *>(&sh[red_sh_delta * j + red_sh_rd]);
                                float *c_wr = reinterpret_cast<float *>(&sh[red_sh_wr]);
                                #pragma unroll
                                for (int k = 0; k < 4; k++)
                                    reinterpret_cast<FragC *>(frag_c)[4 * 2 * m_block + j][k] += c_rd[k] + c_wr[k];
                            }
                            sh[red_sh_wr] = reinterpret_cast<int4 *>(&frag_c)[4 * 2 * m_block + j];
                        }
                    }
                    __syncthreads();
                }
                if (red_idx == 0) {
                    #pragma unroll
                    for (int i = 0; i < 4 * 2; i++) {
                        float *c_rd = reinterpret_cast<float *>(&sh[red_sh_delta * i + red_sh_rd]);
                        #pragma unroll
                        for (int j = 0; j < 4; j++)
                            reinterpret_cast<FragC *>(frag_c)[4 * 2 * m_block + i][j] += c_rd[j];
                    }
                }
                __syncthreads();
            }
        }
    };

    auto write_result = [&]() {
        int c_gl_stride = prob_n / 2;
        constexpr int c_sh_stride = 8 * thread_n_blocks + 1;
        int c_gl_wr_delta = c_gl_stride * (threads / (8 * thread_n_blocks));
        constexpr int c_sh_rd_delta = c_sh_stride * (threads / (8 * thread_n_blocks));

        int c_gl_wr = c_gl_stride * (threadIdx.x / (8 * thread_n_blocks)) + (threadIdx.x % (8 * thread_n_blocks));
        c_gl_wr += (8 * thread_n_blocks) * slice_col;
        int c_sh_rd = c_sh_stride * (threadIdx.x / (8 * thread_n_blocks)) + (threadIdx.x % (8 * thread_n_blocks));

        int c_gl_wr_end = c_gl_stride * prob_m;

        int warp_id = threadIdx.x / 32;
        int lane = threadIdx.x % 32;
        int row_in_warp = lane / 4;
        int col_in_warp = lane % 4;
        int c_sh_wr = row_in_warp * c_sh_stride + warp_id * 32 + col_in_warp;

        auto write = [&](int idx, float c0, float c1) {
            ((float2 *)sh)[idx] = make_float2(c0, c1);
        };

        if (warp_id < thread_n_blocks / 4) {
            #pragma unroll
            for (int i = 0; i < thread_m_blocks; i++) {
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    int wr = c_sh_wr + (j * 8);
                    write(wr + 0 * c_sh_stride + 0, frag_c[i][j][0][0], frag_c[i][j][0][1]);
                    write(wr + 8 * c_sh_stride + 0, frag_c[i][j][0][2], frag_c[i][j][0][3]);
                    write(wr + 0 * c_sh_stride + 4, frag_c[i][j][1][0], frag_c[i][j][1][1]);
                    write(wr + 8 * c_sh_stride + 4, frag_c[i][j][1][2], frag_c[i][j][1][3]);
                }
                c_sh_wr += 16 * c_sh_stride;
            }
        }
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < ceildiv(16 * thread_m_blocks, threads / (8 * thread_n_blocks)); i++) {
            if (c_gl_wr < c_gl_wr_end) {
                float2 res = ((float2 *)sh)[c_sh_rd];
                if (bias) {
                    int col0 = 2 * (c_gl_wr % c_gl_stride);
                    res.x += bias[col0];
                    res.y += bias[col0 + 1];
                }
                reinterpret_cast<float2 *>(C)[c_gl_wr] = res;
                c_gl_wr += c_gl_wr_delta;
                c_sh_rd += c_sh_rd_delta;
            }
        }
    };

    auto start_pipes = [&]() {
        #pragma unroll
        for (int i = 0; i < stages - 1; i++)
            fetch_to_shared(i, i, i < slice_iters);
        zero_accums();
        wait_for_stage();
        fetch_to_registers(0, 0);
        a_gl_rd += a_gl_rd_delta_o * (stages - 1);
    };
    start_pipes();

    while (slice_iters) {
        #pragma unroll
        for (int pipe = 0; pipe < stages;) {
            #pragma unroll
            for (int k = 0; k < b_sh_wr_iters; k++) {
                fetch_to_registers(k + 1, pipe % stages);
                if (k == b_sh_wr_iters - 2) {
                    fetch_to_shared((pipe + stages - 1) % stages, pipe, slice_iters >= stages);
                    pipe++;
                    wait_for_stage();
                }
                matmul(k);
            }
            slice_iters--;
            if (slice_iters == 0) break;
        }
        a_gl_rd += a_gl_rd_delta_o * stages;

        if (slice_iters == 0) {
            cp_async_wait<0>();
            bool last = slice_idx == slice_count - 1;
            thread_block_reduce();
            if (slice_count > 1) {
                constexpr int active_threads = 32 * thread_n_blocks / 4;
                barrier_acquire(&locks[slice_col], slice_idx);
                if (threadIdx.x < active_threads) {
                    constexpr int red_elems = thread_m_blocks * 4 * 2 * 4;
                    float *frag_c_ptr = reinterpret_cast<float *>(&frag_c);
                    float *tmp_slice = C_tmp + (size_t)slice_col * active_threads * red_elems;
                    if (slice_idx > 0) {
                        #pragma unroll
                        for (int r = 0; r < red_elems; r++)
                            frag_c_ptr[r] += tmp_slice[threadIdx.x * red_elems + r];
                    }
                    if (!last) {
                        #pragma unroll
                        for (int r = 0; r < red_elems; r++)
                            tmp_slice[threadIdx.x * red_elems + r] = frag_c_ptr[r];
                    }
                }
                barrier_release(&locks[slice_col], last);
            }
            if (last)
                write_result();
            slice_row = 0;
            slice_col_par++;
            slice_col++;
            init_slice();
            if (slice_iters) {
                a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
                #pragma unroll
                for (int i = 0; i < b_sh_wr_iters; i++)
                    B_ptr[i] += b_sh_stride - b_gl_rd_delta_o * k_tiles;
                if (slice_col == 0) {
                    #pragma unroll
                    for (int i = 0; i < b_sh_wr_iters; i++)
                        B_ptr[i] -= b_gl_stride;
                }
                s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / 2) + s_sh_stride * slice_col + threadIdx.x;
                start_pipes();
            }
        }
    }
}

const int MARLIN_THREADS = 256;
const int MARLIN_STAGES = 4;
const int MARLIN_SHARED_MEM = 96 * 1024;

#define CALL_MARLIN_Q4K(THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS) \
    else if (thread_m_blocks == THREAD_M_BLOCKS && thread_n_blocks == THREAD_N_BLOCKS && thread_k_blocks == THREAD_K_BLOCKS) { \
        cudaFuncSetAttribute( \
            MarlinQ4K<MARLIN_THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, MARLIN_STAGES, 2>, \
            cudaFuncAttributeMaxDynamicSharedMemorySize, \
            MARLIN_SHARED_MEM \
        ); \
        MarlinQ4K< \
            MARLIN_THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, MARLIN_STAGES, 2 \
        ><<<blocks, MARLIN_THREADS, MARLIN_SHARED_MEM, s>>>( \
            A_ptr, B_ptr, C_ptr, s_ptr, m_ptr, bias, \
            prob_m, prob_n, prob_k, \
            locks, C_tmp \
        ); \
    }

#endif // sm_80+
#endif // !__HIPCC__

int fllm_marlin_q4k_try(cudaStream_t s, const void *W, const float *X, float *Y, const float *bias,
                        int rows, int cols, int batch, void *x16, size_t x16_bytes) {
#if defined(__HIPCC__) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
    // (the sm_75 device pass of nvcc parses this body too, without the kernels and helpers above)
    return 0;
#else
    if (fllm_cc() < 80) return 0;
    static const bool disabled = getenv("FLLM_CUDA_NO_MARLIN") != nullptr;
    if (disabled) return 0;
    if (batch < 1 || (rows & 63) != 0 || (cols & 63) != 0) return 0;
    if ((reinterpret_cast<uintptr_t>(W) & 15) != 0 || (reinterpret_cast<uintptr_t>(X) & 15) != 0) return 0;
    if (!x16 || x16_bytes < (size_t)batch * cols * 2) return 0;

    int prob_m = batch;
    int prob_n = rows;
    int prob_k = cols;

    int dev = 0;
    cudaGetDevice(&dev);
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    if (sms <= 0) sms = 36;

    int thread_k = 64;
    int thread_n = 256;
    if (prob_n % thread_n != 0 || prob_k % thread_k != 0) return 0;

    int thread_k_blocks = thread_k / 16;
    int thread_n_blocks = thread_n / 16;

    int tot_m = prob_m;
    int tot_m_blocks = ceildiv(tot_m, 16);
    int pad = 16 * tot_m_blocks - tot_m;

    // Convert activations X (FP32) to x16 (FP16) with zero-padding up to a multiple of 16 rows
    size_t x_elems = (size_t)batch * cols;
    size_t a_padded_elems = (size_t)tot_m_blocks * 16 * cols;
    half *A_half = static_cast<half *>(x16);
    // the activations in x16 are in this kernel's own layout (padded rows): never reuse another kernel's, never offer ours
    {
        const size_t n2_valid = x_elems / 2;
        const size_t n2_total = a_padded_elems / 2;
        const int blk = 256;
        const unsigned grid = (unsigned)((n2_total + blk - 1) / blk > 4096 ? 4096 : (n2_total + blk - 1) / blk);
        const float2 *up_ptr = fllm_glu_up ? reinterpret_cast<const float2 *>(fllm_glu_up) : nullptr;
        fp32_to_fp16_kernel<<<grid, blk, 0, s>>>(reinterpret_cast<const float2 *>(X), up_ptr, fllm_glu_mode,
                                                 reinterpret_cast<half2 *>(A_half), n2_valid, n2_total);
    }
    fllm_x16_filled = false;

    // Scratch workspace after A_half padded buffer: locks followed by C_tmp
    size_t a_bytes = ((a_padded_elems * 2 + 255) & ~size_t(255));
    char *scratch = static_cast<char *>(x16) + a_bytes;
    size_t scratch_avail = x16_bytes > a_bytes ? x16_bytes - a_bytes : 0;

    int n_tiles = prob_n / thread_n;
    size_t locks_bytes = ((size_t)n_tiles * 16 * sizeof(int) + 255) & ~size_t(255);
    if (scratch_avail < locks_bytes) return 0;

    int *locks = reinterpret_cast<int *>(scratch);
    // locks not used with slice_count == 1

    float *C_tmp = reinterpret_cast<float *>(scratch + locks_bytes);
    size_t c_tmp_avail = scratch_avail > locks_bytes ? scratch_avail - locks_bytes : 0;
    constexpr int red_elems_max = 4 * 4 * 2 * 4;
    size_t c_tmp_need = (size_t)n_tiles * MARLIN_THREADS * red_elems_max * sizeof(float);
    if (c_tmp_avail < c_tmp_need && prob_m <= 64) return 0;

    // Weight buffer layout: repacked weights B, permuted scales s, permuted negative mins m
    const int4 *B_ptr = reinterpret_cast<const int4 *>(W);
    const int4 *s_ptr = reinterpret_cast<const int4 *>(static_cast<const char *>(W) + (size_t)cols * rows / 2);
    const int4 *m_ptr = reinterpret_cast<const int4 *>(static_cast<const char *>(W) + (size_t)cols * rows / 2 + (size_t)cols * rows / 16);

    const int4 *A_ptr = reinterpret_cast<const int4 *>(A_half);
    float *C_ptr = Y;

    int max_par = 16;

    for (int i = 0; i < tot_m_blocks; i += 4) {
        int thread_m_blocks = tot_m_blocks - i;
        prob_m = tot_m - 16 * i;
        int par = 1;
        if (thread_m_blocks > 4) {
            par = (16 * thread_m_blocks - pad) / 64;
            if (par > max_par) par = max_par;
            prob_m = 64 * par;
            i += 4 * (par - 1);
            thread_m_blocks = 4;
        }
        int blocks = n_tiles * par;

        if (false) {}
        CALL_MARLIN_Q4K(1, 16, 4)
        CALL_MARLIN_Q4K(2, 16, 4)
        CALL_MARLIN_Q4K(3, 16, 4)
        CALL_MARLIN_Q4K(4, 16, 4)
        else return 0;

        A_ptr += 16 * thread_m_blocks * (prob_k / 8) * par;
        C_ptr += 16 * thread_m_blocks * prob_n * par;
    }

    return 1;
#endif
}
