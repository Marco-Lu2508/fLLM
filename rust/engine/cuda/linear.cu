// fLLM CUDA backend: quantized linear layers (y = W x [+ bias]) and weight conversion.
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <utility>

#include "common.cuh"
#include "fllm_cuda_internal.h"
#if !defined(__HIPCC__)
#include <cuda_bf16.h>

#include "fp8_gemm.h"
#include "hopper_util.h"
int fllm_cublaslt_gemm_try(cudaStream_t s, const void *W, int kind, const float *X, float *Y, const float *bias, int rows, int cols, int batch);
#endif

// ---------------------------------------------------------------------------------------------
// GEMV for small batches: one warp per output row, every lane decodes runs of 16 weights and
// multiplies them with up to NB activation rows. Exact FP32 arithmetic (activations stay float).
// ---------------------------------------------------------------------------------------------
template <int NB>
__device__ __forceinline__ void load16(const float *p, int e0, int cols, float *x, bool vec) {
    if (vec && e0 + 16 <= cols) {
        const float4 *q = reinterpret_cast<const float4 *>(p + e0);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float4 v = __ldg(q + i);
            x[4 * i] = v.x; x[4 * i + 1] = v.y; x[4 * i + 2] = v.z; x[4 * i + 3] = v.w;
        }
    } else {
#pragma unroll
        for (int i = 0; i < 16; i++) x[i] = e0 + i < cols ? __ldg(p + e0 + i) : 0.f;
    }
}

template <uint32_t KIND, int NB>
__global__ void __launch_bounds__(128)
gemv_kernel(const uint8_t *__restrict__ W, const float *__restrict__ X, float *__restrict__ Y,
            const float *__restrict__ bias, int rows, int cols, int batch, int b0) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= rows) return;
    const uint8_t *rp = W + (size_t)row * kind_row_bytes(KIND, cols);
    bool vec[NB];
#pragma unroll
    for (int b = 0; b < NB; b++) vec[b] = ((reinterpret_cast<uintptr_t>(X + (size_t)(b0 + b) * cols)) & 15) == 0;
    float acc[NB];
#pragma unroll
    for (int b = 0; b < NB; b++) acc[b] = 0.f;
    const int runs = (cols + 15) / 16;
    for (int run = lane; run < runs; run += FLLM_WARP) {
        float w[16];
        decode16<(uint32_t)KIND>(rp, run, cols, w);
#pragma unroll
        for (int b = 0; b < NB; b++) {
            if (b0 + b < batch) {
                float x[16];
                load16<NB>(X + (size_t)(b0 + b) * cols, run * 16, cols, x, vec[b]);
#pragma unroll
                for (int i = 0; i < 16; i++) acc[b] = fmaf(w[i], x[i], acc[b]);
            }
        }
    }
#pragma unroll
    for (int b = 0; b < NB; b++) acc[b] = warp_sum(acc[b]);
    if (lane == 0) {
        const float bv = bias ? bias[row] : 0.f;
#pragma unroll
        for (int b = 0; b < NB; b++)
            if (b0 + b < batch) Y[(size_t)(b0 + b) * rows + row] = acc[b] + bv;
    }
}

// ---------------------------------------------------------------------------------------------
// GEMM for larger batches (prefill): 64x64 output tile per block, K step 64, FP32 FMA.
// Weights are decoded straight into shared memory, one run of 16 per thread.
// ---------------------------------------------------------------------------------------------
template <uint32_t KIND>
__global__ void __launch_bounds__(256)
gemm_kernel(const uint8_t *__restrict__ W, const float *__restrict__ X, float *__restrict__ Y,
            const float *__restrict__ bias, int rows, int cols, int batch) {
    constexpr int BM = 64, BN = 64, BK = 64;
    __shared__ float Ws[BK][BM + 1];
    __shared__ float Xs[BK][BN + 1];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
    const int ld_row = tid >> 2, ld_q = tid & 3;  // 64 rows x 4 runs of 16
    for (int k0 = 0; k0 < cols; k0 += BK) {
        {   // weights: row m0 + ld_row, run (k0 / 16 + ld_q)
            float w[16];
            const int m = m0 + ld_row;
            if (m < rows) {
                const uint8_t *rp = W + (size_t)m * kind_row_bytes(KIND, cols);
                decode16<(uint32_t)KIND>(rp, k0 / 16 + ld_q, cols, w);
            } else {
#pragma unroll
                for (int i = 0; i < 16; i++) w[i] = 0.f;
            }
#pragma unroll
            for (int i = 0; i < 16; i++) Ws[ld_q * 16 + i][ld_row] = w[i];
        }
        {   // activations: token n0 + ld_row, elements k0 + ld_q * 16 ..
            const int n = n0 + ld_row;
            const int e0 = k0 + ld_q * 16;
            float x[16];
            if (n < batch) {
                const float *p = X + (size_t)n * cols;
                load16<1>(p, e0, cols, x, ((reinterpret_cast<uintptr_t>(p)) & 15) == 0);
            } else {
#pragma unroll
                for (int i = 0; i < 16; i++) x[i] = 0.f;
            }
#pragma unroll
            for (int i = 0; i < 16; i++) Xs[ld_q * 16 + i][ld_row] = x[i];
        }
        __syncthreads();
#pragma unroll 8
        for (int kk = 0; kk < BK; kk++) {
            float a[4], b[4];
#pragma unroll
            for (int i = 0; i < 4; i++) a[i] = Ws[kk][tx + 16 * i];
#pragma unroll
            for (int j = 0; j < 4; j++) b[j] = Xs[kk][ty + 16 * j];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] = fmaf(a[i], b[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const int n = n0 + ty + 16 * j;
        if (n >= batch) continue;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const int m = m0 + tx + 16 * i;
            if (m < rows) Y[(size_t)n * rows + m] = acc[i][j] + (bias ? bias[m] : 0.f);
        }
    }
}

#if !defined(__HIPCC__)
// The smallest batch the tensor-core GEMV (dense_skinny_kernel) takes when FLLM_CUDA_DENSE_SKINNY is not set; smaller batches use the CUDA-core GEMV below.
#ifndef FLLM_DENSE_SKINNY_MIN_BATCH
#define FLLM_DENSE_SKINNY_MIN_BATCH 1
#endif
// ... and the largest (17 tokens and more go to cuBLASLt)
#ifndef FLLM_DENSE_SKINNY_MAX_BATCH
#define FLLM_DENSE_SKINNY_MAX_BATCH 16
#endif
// ---------------------------------------------------------------------------------------------
// Dense 16-bit weights (F16 / BF16), one to eight tokens: `gemv_kernel` with the weights of four runs in flight.
//
// The generic kernel decodes one run of 16 weights per lane and waits for it before it asks for the next, so a lane has two 16-byte loads in flight
// and the warp reads about 45 % of the memory bandwidth on an A100 (0.096 ms for the average projection of a 7B model, 0.69 TB/s of 1.55). This kernel
// issues the loads of four runs first and multiplies afterwards. Every lane still visits the runs `lane, lane + 32, ...` in increasing order and runs the
// 16 FMAs of each in the same order, and the warp sum is the same, so the result is the bit pattern of `gemv_kernel` (the self test compares them).
// Needs cols % 16 == 0 and 16-byte aligned weights and activations; anything else keeps the generic kernel.
// ---------------------------------------------------------------------------------------------
__device__ __forceinline__ void dense16_unpack(uint32_t w, bool bf16, float &lo, float &hi) {
    if (bf16) {
        lo = __uint_as_float(w << 16);
        hi = __uint_as_float(w & 0xffff0000u);
    } else {
        lo = h2f_bits((uint16_t)(w & 0xffffu));
        hi = h2f_bits((uint16_t)(w >> 16));
    }
}

template <uint32_t KIND, int NB>
__global__ void __launch_bounds__(128)
gemv_dense_kernel(const uint8_t *__restrict__ W, const float *__restrict__ X, float *__restrict__ Y,
                  const float *__restrict__ bias, int rows, int cols, int batch, int b0) {
    constexpr int U = 4;
    constexpr bool bf16 = KIND == KIND_BF16;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= rows) return;
    const uint4 *rp = reinterpret_cast<const uint4 *>(W + (size_t)row * cols * 2);
    const int runs = cols >> 4;
    float acc[NB];
#pragma unroll
    for (int b = 0; b < NB; b++) acc[b] = 0.f;
    for (int base = lane; base < runs; base += FLLM_WARP * U) {
        uint4 raw[U][2];
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int run = base + u * FLLM_WARP;
            if (run < runs) {
                raw[u][0] = __ldg(rp + 2 * run);
                raw[u][1] = __ldg(rp + 2 * run + 1);
            }
        }
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int run = base + u * FLLM_WARP;
            if (run < runs) {
                float w[16];
                const uint32_t words[8] = {raw[u][0].x, raw[u][0].y, raw[u][0].z, raw[u][0].w, raw[u][1].x, raw[u][1].y, raw[u][1].z, raw[u][1].w};
#pragma unroll
                for (int i = 0; i < 8; i++) dense16_unpack(words[i], bf16, w[2 * i], w[2 * i + 1]);
#pragma unroll
                for (int b = 0; b < NB; b++) {
                    const float4 *q = reinterpret_cast<const float4 *>(X + (size_t)(b0 + b) * cols + (size_t)run * 16);
                    float x[16];
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        const float4 v = __ldg(q + i);
                        x[4 * i] = v.x; x[4 * i + 1] = v.y; x[4 * i + 2] = v.z; x[4 * i + 3] = v.w;
                    }
#pragma unroll
                    for (int i = 0; i < 16; i++) acc[b] = fmaf(w[i], x[i], acc[b]);
                }
            }
        }
    }
#pragma unroll
    for (int b = 0; b < NB; b++) acc[b] = warp_sum(acc[b]);
    if (lane == 0) {
        const float bv = bias ? bias[row] : 0.f;
#pragma unroll
        for (int b = 0; b < NB; b++) Y[(size_t)(b0 + b) * rows + row] = acc[b] + bv;
    }
}

// ---------------------------------------------------------------------------------------------
// Dense 16-bit weights, one to sixteen tokens on the tensor cores: the weights stream through registers once, nothing else touches them.
//
// Y^T (rows x tokens) = W (rows x cols) * X^T with `mma.m16n8k16`: the weights are the A operand (16 rows per block), the tokens the B operand (one or two
// groups of eight), the activations are rounded to the weight's 16-bit format in registers and the products accumulate in FP32 (the rounding that the
// cuBLASLt path and tc_gemm.cu apply as well). The eight warps of a block split the columns in slabs of 32, take their weights as 16-byte loads (a lane holds 8
// consecutive columns of two rows, so nothing is staged in shared memory), keep four slabs in flight, and add their partial sums at the end, so the
// whole product is one launch with no split-K reduction kernel, no conversion kernel for the activations and the bias folded into the write.
// The columns of a slab are assigned to the lanes in the order the loads deliver them and the same order is used for the activations; the sum over the
// 16 columns of an `mma` does not depend on that order.
// Needs cols % 32 == 0 and 16-byte aligned weights and activations.
// ---------------------------------------------------------------------------------------------
template <bool BF16>
__device__ __forceinline__ void mma_dense16(float c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800)
    // (Turing has no m16n8k16: the host never launches the kernel below compute capability 8.0, `fllm_cc() < 80` in dense_skinny_try)
    (void)c; (void)a; (void)b0; (void)b1;
#else
    if (BF16) {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    } else {
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                     : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                     : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
    }
#endif
}

template <bool BF16>
__device__ __forceinline__ uint32_t pack_dense16(float lo, float hi) {
    if (BF16) {
        const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
        return *reinterpret_cast<const uint32_t *>(&v);
    }
    const __half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t *>(&v);
}

// MT 16-row tiles per warp share one set of activation fragments (the activations are read once per 16 * MT rows, which halves their traffic from the L2
// cache for MT = 2); the WARPS warps of a block split the columns.
// (the slabs [s_begin, s_end) of 32 columns: all of them, or one part of a split product)
__host__ __device__ constexpr int dense_uf(int mt) { return mt == 1 ? 4 : 2; }

// XN: the activations are narrow (BF16 when BF16, else FP16: the type of the weights, FLLM_ACT_DTYPE) and are read as the bits the tensor cores take; YN: the
// result is written narrow (the bias is added in FP32 before the rounding)
template <bool BF16, int NT, int WARPS, int MT, int UF = dense_uf(MT), bool XN = false, bool YN = false>
__device__ __forceinline__ void dense_skinny_core(const uint8_t *__restrict__ W, const void *__restrict__ X, void *__restrict__ Y, const float *__restrict__ bias,
                                                  int rows, int cols, int batch, int blk, int s_begin, int s_end) {
    constexpr int U = UF;  // slabs in flight per warp
    constexpr int OUT = MT * NT * 4 * 32;
    __shared__ float partial[WARPS][OUT];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int q = lane & 3, g = lane >> 2;
    const int base = blk * 16 * MT;
    const uint4 *w0[MT], *w1[MT];
#pragma unroll
    for (int m = 0; m < MT; m++) {
        const int row0 = base + 16 * m + g, row1 = row0 + 8;
        w0[m] = reinterpret_cast<const uint4 *>(W + (size_t)(row0 < rows ? row0 : rows - 1) * cols * 2);
        w1[m] = reinterpret_cast<const uint4 *>(W + (size_t)(row1 < rows ? row1 : rows - 1) * cols * 2);
    }
    const float4 *xp[NT];
    const uint4 *xq[NT];
    bool live[NT];
#pragma unroll
    for (int t = 0; t < NT; t++) {
        const int tok = 8 * t + g;
        live[t] = tok < batch;
        if constexpr (XN) {
            xq[t] = reinterpret_cast<const uint4 *>(static_cast<const uint16_t *>(X) + (size_t)(live[t] ? tok : 0) * cols) + q;
        } else {
            xp[t] = reinterpret_cast<const float4 *>(static_cast<const float *>(X) + (size_t)(live[t] ? tok : 0) * cols) + q * 2;
        }
    }
    float c[MT][NT][4];
#pragma unroll
    for (int m = 0; m < MT; m++)
#pragma unroll
        for (int t = 0; t < NT; t++)
#pragma unroll
            for (int i = 0; i < 4; i++) c[m][t][i] = 0.f;
    const int slabs = s_end;
    for (int s0 = s_begin + warp; s0 < slabs; s0 += WARPS * U) {
        uint4 wa[U][MT], wb[U][MT];
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int s = s0 + u * WARPS;
            if (s < slabs) {
#pragma unroll
                for (int m = 0; m < MT; m++) {
                    wa[u][m] = __ldg(w0[m] + s * 4 + q);
                    wb[u][m] = __ldg(w1[m] + s * 4 + q);
                }
            }
        }
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int s = s0 + u * WARPS;
            if (s < slabs) {
                uint32_t b[NT][4];
#pragma unroll
                for (int t = 0; t < NT; t++) {
                    if constexpr (XN) {
                        // eight consecutive columns of the lane's slab as they are stored: the pairs the mma takes
                        uint4 xv = make_uint4(0u, 0u, 0u, 0u);
                        if (live[t]) xv = __ldg(xq[t] + s * 4);
                        b[t][0] = xv.x;
                        b[t][1] = xv.y;
                        b[t][2] = xv.z;
                        b[t][3] = xv.w;
                    } else {
                        float4 x0 = make_float4(0.f, 0.f, 0.f, 0.f), x1 = x0;
                        if (live[t]) {
                            x0 = __ldg(xp[t] + s * 8);
                            x1 = __ldg(xp[t] + s * 8 + 1);
                        }
                        b[t][0] = pack_dense16<BF16>(x0.x, x0.y);
                        b[t][1] = pack_dense16<BF16>(x0.z, x0.w);
                        b[t][2] = pack_dense16<BF16>(x1.x, x1.y);
                        b[t][3] = pack_dense16<BF16>(x1.z, x1.w);
                    }
                }
#pragma unroll
                for (int m = 0; m < MT; m++) {
                    const uint32_t a1[4] = {wa[u][m].x, wb[u][m].x, wa[u][m].y, wb[u][m].y};
                    const uint32_t a2[4] = {wa[u][m].z, wb[u][m].z, wa[u][m].w, wb[u][m].w};
#pragma unroll
                    for (int t = 0; t < NT; t++) {
                        mma_dense16<BF16>(c[m][t], a1, b[t][0], b[t][1]);
                        mma_dense16<BF16>(c[m][t], a2, b[t][2], b[t][3]);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int m = 0; m < MT; m++)
#pragma unroll
        for (int t = 0; t < NT; t++)
#pragma unroll
            for (int i = 0; i < 4; i++) partial[warp][((m * NT + t) * 4 + i) * 32 + lane] = c[m][t][i];
    __syncthreads();
    // every thread adds the partial sums of all warps for one or more of the MT * 16 x (8 * NT) results and writes them
    for (int idx = threadIdx.x; idx < OUT; idx += WARPS * 32) {
        float sum = 0.f;
#pragma unroll
        for (int w = 0; w < WARPS; w++) sum += partial[w][idx];
        const int ln = idx & 31, i = (idx >> 5) & 3, t = (idx >> 7) % NT, m = (idx >> 7) / NT;
        const int row = base + 16 * m + (ln >> 2) + 8 * (i >> 1);
        const int tok = 8 * t + 2 * (ln & 3) + (i & 1);
        if (row < rows && tok < batch) {
            const float value = sum + (bias ? bias[row] : 0.f);
            const size_t at = (size_t)tok * rows + row;
            if constexpr (!YN) static_cast<float *>(Y)[at] = value;
            else if constexpr (BF16) static_cast<__nv_bfloat16 *>(Y)[at] = __float2bfloat16(value);
            else static_cast<__half *>(Y)[at] = __float2half(value);
        }
    }
}

template <bool BF16, int NT, int WARPS, int MT, int UF = dense_uf(MT), bool XN = false, bool YN = false>
__global__ void __launch_bounds__(WARPS * 32)
dense_skinny_kernel(const uint8_t *__restrict__ W, const void *__restrict__ X, void *__restrict__ Y, const float *__restrict__ bias, int rows, int cols,
                    int batch) {
    dense_skinny_core<BF16, NT, WARPS, MT, UF, XN, YN>(W, X, Y, bias, rows, cols, batch, blockIdx.x, 0, cols >> 5);
}

// Up to three projections that read the same activations (q, k and v) in one launch: the blocks of the segments follow each other, so the small ones (k and v
// are 16 blocks each) run in the last wave of the big one instead of after it.
struct DenseSegs {
    const uint8_t *W[3];
    float *Y[3];
    const float *bias[3];
    int rows[3];
    int end[3];  // first block after the segment
    int n;
};
template <bool BF16, int NT, int WARPS, int MT>
__global__ void __launch_bounds__(WARPS * 32) dense_skinny_group_kernel(DenseSegs sg, const float *__restrict__ X, int cols, int batch) {
    int s = 0;
    while (s + 1 < sg.n && (int)blockIdx.x >= sg.end[s]) s++;
    const int begin = s ? sg.end[s - 1] : 0;
    dense_skinny_core<BF16, NT, WARPS, MT>(sg.W[s], X, sg.Y[s], sg.bias[s], sg.rows[s], cols, batch, (int)blockIdx.x - begin, 0, cols >> 5);
}

// Split along the columns: block (x, y) sums the slabs of part y of the row tile x into its own slice of `part` (splits x batch x rows values), and
// dense_skinny_reduce_kernel adds the parts in a fixed order (plus the bias). Few row tiles (3584 rows are 224 tiles on 108 SMs: the last wave of blocks
// leaves a third of the card idle) become many blocks of equal size; the result does not depend on the schedule.
template <bool BF16, int NT, int WARPS, int MT, bool XN = false>
__global__ void __launch_bounds__(WARPS * 32)
dense_skinny_split_kernel(const uint8_t *__restrict__ W, const void *__restrict__ X, float *__restrict__ part, int rows, int cols, int batch, int splits) {
    const int slabs = cols >> 5;
    const int per = (slabs + splits - 1) / splits;
    const int s_begin = blockIdx.y * per, s_end = min(slabs, s_begin + per);
    dense_skinny_core<BF16, NT, WARPS, MT, dense_uf(MT), XN, false>(W, X, part + (size_t)blockIdx.y * batch * rows, nullptr, rows, cols, batch, blockIdx.x, s_begin, s_end);
}
template <bool BF16, bool YN>
__global__ void dense_skinny_reduce_kernel(const float *__restrict__ part, void *__restrict__ Y, const float *__restrict__ bias, int rows, int batch, int splits) {
    const size_t total = (size_t)batch * rows;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        float sum = bias ? bias[i % rows] : 0.f;
        for (int p = 0; p < splits; p++) sum += part[(size_t)p * total + i];
        if constexpr (!YN) static_cast<float *>(Y)[i] = sum;
        else if constexpr (BF16) static_cast<__nv_bfloat16 *>(Y)[i] = __float2bfloat16(sum);
        else static_cast<__half *>(Y)[i] = __float2half(sum);
    }
}

// (fllm_skinny_part, the scratch for the parts of a split product, is set by fllm_cuda_linear: the attention workspace of the stream)
static std::atomic<int> g_dense_skinny_mode{-1};
// -1: environment (FLLM_CUDA_DENSE_SKINNY = auto | 0 | 1), 0: never, 1: always where the shape fits; used by the self test to compare the paths
extern "C" void fllm_dense_skinny_set_mode(int mode) { g_dense_skinny_mode.store(mode < -1 || mode > 2 ? -1 : mode); }

// 0: the tensor-core GEMV does not take a dense 16-bit projection of this shape (or is switched off); 1: one 16-row tile per warp; 2: two tiles per warp
int fllm_dense_skinny_variant(int rows, int cols, int batch) {
    static const int env = [] {
        const char *v = getenv("FLLM_CUDA_DENSE_SKINNY");
        return v == nullptr || *v == 0 ? -1 : fllm_env_tri("FLLM_CUDA_DENSE_SKINNY");
    }();
    const int forced = g_dense_skinny_mode.load();
    const int mode = forced >= 0 ? forced : env;
    if (mode == 0 || batch < 1 || batch > 16 || (cols & 31) != 0 || rows <= 0 || fllm_exact_math() || fllm_cc() < 80) return 0;
    if (mode < 0 && (batch < FLLM_DENSE_SKINNY_MIN_BATCH || batch > FLLM_DENSE_SKINNY_MAX_BATCH)) return 0;
    // automatic choice for 9-16 tokens, measured on an A100 (bf16, 16 tokens, GB/s of weights; one tile per warp / two tiles / cuBLASLt):
    //   3584 x 3584: 1161 / 1053 / 869,  18944 x 3584: 1005 / 1145 / 1046,  3584 x 18944: 808 / 824 / 1167,  512 x 3584: 253 / 315 / 282.
    // A long column range over many blocks but few rows leaves SMs without work in the last wave (224 or 112 blocks on 108 SMs): cuBLASLt splits those
    // columns. Under 2048 rows there are less blocks than SMs and the rest of the card is idle anyway, so the single launch (no conversion, no
    // split-K reduction, no bias kernel) wins.
    int variant = mode == 2 ? 2 : 1;
    if (mode < 0 && batch > 8) {
        if (rows >= 2048 && cols >= 4 * (size_t)rows) return 0;
        variant = rows >= 8192 || rows < 2048 ? 2 : 1;
    }
    return variant;
}

template <bool bf16, bool XN, bool YN>
static bool dense_skinny_launch(cudaStream_t s, const uint8_t *W, const void *X, void *Y, const float *bias, int rows, int cols, int batch) {
    const int variant = fllm_dense_skinny_variant(rows, cols, batch);
    if (variant == 0) return false;
    if ((reinterpret_cast<uintptr_t>(W) & 15) != 0 || (reinterpret_cast<uintptr_t>(X) & 15) != 0) return false;
    // two row tiles per warp and sixteen warps per block
    const int tiles = batch > 8 && variant == 2 ? (rows + 31) / 32 : (rows + 15) / 16;
    // few tiles: split the columns so that about a thousand blocks of equal size are launched (never less than four slabs of 32 columns per part)
    static const bool no_split = getenv("FLLM_CUDA_NO_SKINNY_SPLIT") != nullptr;
    int splits = 1;
    if (!no_split && tiles < 512 && fllm_skinny_part != nullptr) {
        splits = 1024 / tiles;
        if (splits > 8) splits = 8;
        if (splits > (cols >> 5) / 4) splits = (cols >> 5) / 4;
        while (splits > 1 && (size_t)splits * batch * rows * sizeof(float) > fllm_skinny_part_bytes) splits--;
        if (splits < 1) splits = 1;
    }
    if (splits > 1) {
        const dim3 grid(tiles, splits);
        if (batch > 8 && variant == 2) dense_skinny_split_kernel<bf16, 2, 16, 2, XN><<<grid, 512, 0, s>>>(W, X, fllm_skinny_part, rows, cols, batch, splits);
        else if (batch <= 8) dense_skinny_split_kernel<bf16, 1, 8, 1, XN><<<grid, 256, 0, s>>>(W, X, fllm_skinny_part, rows, cols, batch, splits);
        else dense_skinny_split_kernel<bf16, 2, 8, 1, XN><<<grid, 256, 0, s>>>(W, X, fllm_skinny_part, rows, cols, batch, splits);
        const size_t total = (size_t)batch * rows;
        dense_skinny_reduce_kernel<bf16, YN><<<(unsigned)((total + 255) / 256 > 1024 ? 1024 : (total + 255) / 256), 256, 0, s>>>(fllm_skinny_part, Y, bias, rows, batch, splits);
        return true;
    }
    static const bool no_wide = getenv("FLLM_CUDA_NO_SKINNY_WIDE") != nullptr;
    if (batch > 8 && variant == 2 && rows >= 8192 && !no_wide) {
        // tall projections (gate, up): four 16-row tiles per warp read each slice of the activations once for 64 rows instead of 32, one slab in flight
        // per tile (A100, 18944 x 3584, 16 tokens: 119 -> 110 us)
        dense_skinny_kernel<bf16, 2, 4, 4, 1, XN, YN><<<dim3((rows + 63) / 64), 128, 0, s>>>(W, X, Y, bias, rows, cols, batch);
    } else if (batch > 8 && variant == 2) {
        dense_skinny_kernel<bf16, 2, 16, 2, dense_uf(2), XN, YN><<<dim3((rows + 31) / 32), 512, 0, s>>>(W, X, Y, bias, rows, cols, batch);
    } else if (batch <= 8) {
        dense_skinny_kernel<bf16, 1, 8, 1, dense_uf(1), XN, YN><<<dim3((rows + 15) / 16), 256, 0, s>>>(W, X, Y, bias, rows, cols, batch);
    } else {
        dense_skinny_kernel<bf16, 2, 8, 1, dense_uf(1), XN, YN><<<dim3((rows + 15) / 16), 256, 0, s>>>(W, X, Y, bias, rows, cols, batch);
    }
    return true;
}

template <uint32_t KIND>
static bool dense_skinny_try(cudaStream_t s, const uint8_t *W, const float *X, float *Y, const float *bias, int rows, int cols, int batch) {
    return dense_skinny_launch<KIND == KIND_BF16, false, false>(s, W, X, Y, bias, rows, cols, batch);
}

// FLLM_ACT_DTYPE: dense 16-bit weights on narrow activations (BF16 weights with BF16 activations, FP16 with FP16): the activations are read and the result is written
// narrow, no conversion kernels. false (nothing launched) when the tensor-core GEMV does not take the shape (cuBLASLt does, from 9 tokens on).
bool fllm_dense_skinny_narrow_try(cudaStream_t s, bool bf16, const void *W, const void *X, void *Y, const float *bias, int rows, int cols, int batch) {
    return bf16 ? dense_skinny_launch<true, true, true>(s, static_cast<const uint8_t *>(W), X, Y, bias, rows, cols, batch)
                : dense_skinny_launch<false, true, true>(s, static_cast<const uint8_t *>(W), X, Y, bias, rows, cols, batch);
}

// q, k, v as one launch: every segment must be a shape the single projections would give to the tensor-core GEMV, the activations and weights aligned; returns
// false (nothing launched) otherwise. All segments run with the tile of the smallest variant any of them asked for (the results are those of the separate
// launches up to the order of the sums over the columns).
bool fllm_dense_skinny_group_try(cudaStream_t s, uint32_t kind, int n, const void *const *W, void *const *Y, const void *const *bias, const int *rows, int cols,
                                 int batch, const float *X) {
    if (n < 2 || n > 3 || (kind != KIND_BF16 && kind != KIND_F16) || (reinterpret_cast<uintptr_t>(X) & 15) != 0) return false;
    int variant = 2;
    for (int i = 0; i < n; i++) {
        const int v = fllm_dense_skinny_variant(rows[i], cols, batch);
        if (v == 0 || (reinterpret_cast<uintptr_t>(W[i]) & 15) != 0) return false;
        if (v == 1) variant = 1;
    }
    const bool big = batch > 8 && variant == 2;
    DenseSegs sg{};
    sg.n = n;
    int blocks = 0;
    for (int i = 0; i < n; i++) {
        sg.W[i] = static_cast<const uint8_t *>(W[i]);
        sg.Y[i] = static_cast<float *>(Y[i]);
        sg.bias[i] = static_cast<const float *>(bias[i]);
        sg.rows[i] = rows[i];
        blocks += big ? (rows[i] + 31) / 32 : (rows[i] + 15) / 16;
        sg.end[i] = blocks;
    }
    const bool bf16 = kind == KIND_BF16;
    if (big) {
        if (bf16) dense_skinny_group_kernel<true, 2, 16, 2><<<dim3(blocks), 512, 0, s>>>(sg, X, cols, batch);
        else dense_skinny_group_kernel<false, 2, 16, 2><<<dim3(blocks), 512, 0, s>>>(sg, X, cols, batch);
    } else if (batch <= 8) {
        if (bf16) dense_skinny_group_kernel<true, 1, 8, 1><<<dim3(blocks), 256, 0, s>>>(sg, X, cols, batch);
        else dense_skinny_group_kernel<false, 1, 8, 1><<<dim3(blocks), 256, 0, s>>>(sg, X, cols, batch);
    } else {
        if (bf16) dense_skinny_group_kernel<true, 2, 8, 1><<<dim3(blocks), 256, 0, s>>>(sg, X, cols, batch);
        else dense_skinny_group_kernel<false, 2, 8, 1><<<dim3(blocks), 256, 0, s>>>(sg, X, cols, batch);
    }
    return true;
}

// 1..15 tokens in passes of up to eight; returns false (nothing launched) when the shape or the alignment does not fit.
template <uint32_t KIND>
static bool dense_gemv_try(cudaStream_t s, const uint8_t *W, const float *X, float *Y, const float *bias, int rows, int cols, int batch) {
    static const bool off = getenv("FLLM_CUDA_NO_DENSE_GEMV") != nullptr;
    if (off || batch < 1 || batch > 15 || (cols & 15) != 0 || rows <= 0) return false;
    if ((reinterpret_cast<uintptr_t>(W) & 15) != 0 || (reinterpret_cast<uintptr_t>(X) & 15) != 0) return false;
    const dim3 block(128), grid((rows + 3) / 4);
    for (int b0 = 0; b0 < batch; b0 += 8) {
        switch (batch - b0 < 8 ? batch - b0 : 8) {
            case 1: gemv_dense_kernel<KIND, 1><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, b0); break;
            case 2: gemv_dense_kernel<KIND, 2><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, b0); break;
            case 3: gemv_dense_kernel<KIND, 3><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, b0); break;
            case 4: gemv_dense_kernel<KIND, 4><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, b0); break;
            case 5: gemv_dense_kernel<KIND, 5><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, b0); break;
            case 6: gemv_dense_kernel<KIND, 6><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, b0); break;
            case 7: gemv_dense_kernel<KIND, 7><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, b0); break;
            default: gemv_dense_kernel<KIND, 8><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, b0); break;
        }
    }
    return true;
}
#endif

// ---------------------------------------------------------------------------------------------
// Launchers
// ---------------------------------------------------------------------------------------------
#include "bnb.cuh"

// The caller handed over a gated activation as (gate = X, up = fllm_glu_up) because the GEMM it expected folds the GLU into the conversion of its
// activations. When that GEMM did not run (it declines a shape, or the device's shared memory is too small for any of its tile shapes, which it only finds
// out after the conversion) the next kernel would read the raw gate: compute the activation into the buffer it belongs in and continue with that.
static const float *glu_recover(cudaStream_t s, const float *X, int batch, int cols) {
    // (FLLM_CUDA_NO_GLU_RECOVER=1: the behaviour before this function existed, for bisecting and for checking that the self test sees the defect)
    static const bool off = getenv("FLLM_CUDA_NO_GLU_RECOVER") != nullptr;
    if (!fllm_glu_up || !fllm_glu_dst || off) return X;
    const size_t count = (size_t)batch * cols;
    const int rc = fllm_glu_half ? fllm_glu_launch_f16(s, X, fllm_glu_up, fllm_glu_dst, count, fllm_glu_mode)
                                 : fllm_glu_launch(s, X, fllm_glu_up, fllm_glu_dst, count, fllm_glu_mode);
    fllm_glu_up = nullptr;
    return rc == 0 ? fllm_glu_dst : X;
}

template <int KIND>
static int launch_kind(cudaStream_t s, const uint8_t *W, const float *X, float *Y, const float *bias,
                       int rows, int cols, int batch, void *x16, size_t x16_bytes) {
    if (KIND == KIND_Q4_K_MARLIN && !fllm_exact_math() &&
        fllm_marlin_q4k_try(s, W, X, Y, bias, rows, cols, batch, x16, x16_bytes))
        return fllm_check_launch("linear");
    if (KIND == KIND_Q4_K_MARLIN) {
        // The GEMM declined (a shape it does not tile, an alignment, too little scratch room for a tall projection): the dual layout holds the plain
        // Q4_K blocks in front of the Marlin ones (`W` points behind them, backend/cuda.rs), so run the Q4_K kernels on those. Failing the step here took
        // Gemma 4 E4B down at 16 concurrent requests (the output head of a 262k vocabulary needs more split-K room than the activation scratch had).
        const uint8_t *plain = W - (size_t)rows * (cols / 256) * 144;
        return launch_kind<KIND_Q4_K>(s, plain, X, Y, bias, rows, cols, batch, x16, x16_bytes);
    }
    static const bool old_gemv = getenv("FLLM_CUDA_OLD_GEMV") != nullptr;
#if defined(__HIPCC__)
    // AMD: F16 / BF16 weights and 1..4 tokens go to the skinny GEMM after vLLM's wvSplitK (hip/hip_skinny.cu)
    if ((KIND == KIND_F16 || KIND == KIND_BF16) && batch <= 4 && !fllm_exact_math() &&
        fllm_hip_skinny_try(s, W, KIND, X, Y, bias, rows, cols, batch, x16, x16_bytes))
        return fllm_check_launch("linear");
#endif
    // single tokens and a few concurrent users on the 32-value formats: integer dot products with int8 activations
    static const bool no_mmvq_leg = getenv("FLLM_CUDA_NO_MMVQ") != nullptr || getenv("FLLM_CUDA_NO_MMVQ_LEG") != nullptr;
    if (!no_mmvq_leg && (KIND == KIND_Q4_0 || KIND == KIND_Q4_1 || KIND == KIND_Q5_0 || KIND == KIND_Q5_1 || FLLM_LUT_KIND(KIND) ||
                         KIND == KIND_Q8_0) &&
        batch <= 4 && !fllm_exact_math() && fllm_mmvq_leg_try(s, W, KIND, X, Y, bias, rows, cols, batch, x16, x16_bytes))
        return fllm_check_launch("linear");
    if (kind_is_iq_or_kextra(KIND) && KIND != KIND_IQ1_M && batch <= fllm_mmvq_x_maxb() && !fllm_exact_math() &&
        fllm_mmvq_x_try(s, W, KIND, X, Y, bias, rows, cols, batch, x16, x16_bytes))
        return fllm_check_launch("linear");
    static const bool mmvq1 = getenv("FLLM_CUDA_NO_MMVQ1") == nullptr && getenv("FLLM_CUDA_NO_MMVQ") == nullptr;
    if (mmvq1 && (KIND == KIND_Q4_K || KIND == KIND_Q5_K || KIND == KIND_Q6_K) && batch == 1 && !fllm_exact_math() &&
        fllm_mmvq_try(s, W, KIND, X, Y, bias, rows, cols, batch, x16, x16_bytes))
        return fllm_check_launch("linear");
    // narrow batches (decode, a few concurrent users) are bandwidth-bound: decode the weights once per pass
    static const int gemv_max_batch = getenv("FLLM_CUDA_GEMV_MAXB") ? atoi(getenv("FLLM_CUDA_GEMV_MAXB")) : 1;
    if (!old_gemv && (KIND == KIND_Q4_K || KIND == KIND_Q5_K || KIND == KIND_Q6_K || KIND == KIND_Q8_0 || KIND == KIND_Q4_0 || FLLM_LUT_KIND(KIND) || KIND == KIND_Q4_1 ||
         KIND == KIND_Q5_0 || KIND == KIND_Q5_1) &&
        batch <= gemv_max_batch &&
        fllm_gemv2_try(s, W, KIND, X, Y, bias, rows, cols, batch))
        return fllm_check_launch("linear");
    // 2..4 tokens: integer dot products (int8 activations) are the fastest way to stream the weights
    static const bool no_mmvq = getenv("FLLM_CUDA_NO_MMVQ") != nullptr;
    if (!no_mmvq && (KIND == KIND_Q4_K || KIND == KIND_Q5_K || KIND == KIND_Q6_K) && batch >= 2 && batch <= fllm_mmvq_maxb() && !fllm_exact_math() &&
        fllm_mmvq_try(s, W, KIND, X, Y, bias, rows, cols, batch, x16, x16_bytes))
        return fllm_check_launch("linear");
#if !defined(__HIPCC__)
    // F16 / BF16 weights, up to sixteen tokens: the tensor-core GEMV reads the activation itself, so a gated activation is computed first
    if constexpr (KIND == KIND_F16 || KIND == KIND_BF16) {
        if (batch <= 16) {
            X = glu_recover(s, X, batch, cols);
            if (dense_skinny_try<KIND>(s, W, X, Y, bias, rows, cols, batch)) return fllm_check_launch("linear");
        }
    }
    // (the int8 GEMM below does not fold a gated activation into its input)
    if (KIND == KIND_Q8_0 && batch >= 16) X = glu_recover(s, X, batch, cols);
    // dense 16-bit weights, prompt batches: cuBLASLt loaded at run time (cublaslt_gemm.cu; Hopper / Blackwell kernels), which folds a pending gated
    // activation into its conversion of the input; else the built-in GEMM below
    if ((KIND == KIND_F16 || KIND == KIND_BF16) && batch >= 9) {
        const int rc = fllm_cublaslt_gemm_try(s, W, KIND, X, Y, bias, rows, cols, batch);
        if (rc >= 0) return rc;
    }
    if ((KIND == KIND_F16 || KIND == KIND_BF16) && batch >= 16) X = glu_recover(s, X, batch, cols);
    // Q8_0 prompts on the int8 tensor cores (int8_gemm.cu, FLLM_INT8_GEMM=1)
    if (KIND == KIND_Q8_0 && batch >= 16) {
        const int rc = fllm_int8_q8_try(s, W, X, Y, bias, rows, cols, batch);
        if (rc >= 0) return rc;
    }
#endif
    if (KIND == KIND_Q4_K && batch >= 2 && !fllm_exact_math() &&
        fllm_qgemm4k_try(s, W, X, Y, bias, rows, cols, batch, x16, x16_bytes))
        return fllm_check_launch("linear");
    if ((KIND == KIND_Q4_0 || FLLM_LUT_KIND(KIND) || KIND == KIND_Q4_1 || KIND == KIND_Q5_0 || KIND == KIND_Q5_1) && batch >= 2 && !fllm_exact_math() &&
        (cols & 31) == 0 && fllm_qgemm_leg_try(s, W, KIND, X, Y, bias, rows, cols, batch, x16, x16_bytes))
        return fllm_check_launch("linear");
    if (KIND == KIND_Q8_0 && batch >= 2 && !fllm_exact_math() && (cols & 31) == 0 &&
        fllm_qgemm8_try(s, W, X, Y, bias, rows, cols, batch, x16, x16_bytes))
        return fllm_check_launch("linear");
    if (KIND == KIND_Q5_K && batch >= 2 && !fllm_exact_math() &&
        fllm_qgemm5k_try(s, W, X, Y, bias, rows, cols, batch, x16, x16_bytes))
        return fllm_check_launch("linear");
    if (KIND == KIND_Q6_K && batch >= 2 && !fllm_exact_math() &&
        fllm_qgemm6k_try(s, W, X, Y, bias, rows, cols, batch, x16, x16_bytes))
        return fllm_check_launch("linear");
    X = glu_recover(s, X, batch, cols);  // none of the GEMMs above ran: the kernels below read the activation itself
    if (!old_gemv && (KIND == KIND_Q4_K || KIND == KIND_Q6_K) && fllm_gemv2_try(s, W, KIND, X, Y, bias, rows, cols, batch))
        return fllm_check_launch("linear");
#if defined(__HIPCC__)
    // AMD matrix-core GEMM with the activations converted to FP16 once (hip/hip_gemm.cu)
    if (batch >= 16 && !fllm_exact_math() && fllm_hip_gemm_x16_try(s, W, KIND, X, Y, bias, rows, cols, batch, x16, x16_bytes)) return fllm_check_launch("linear");
#endif
#if !defined(__HIPCC__)
    // F16 / BF16 weights, one to fifteen tokens: the bandwidth-bound GEMV with several runs in flight (bits of the generic kernel below)
    if constexpr (KIND == KIND_F16 || KIND == KIND_BF16) {
        if (dense_gemv_try<KIND>(s, W, X, Y, bias, rows, cols, batch)) return fllm_check_launch("linear");
    }
#endif
    if (batch >= 16 && !fllm_exact_math() && fllm_tc_gemm_try(s, W, KIND, X, Y, bias, rows, cols, batch)) return fllm_check_launch("linear");
    if (batch <= 8) {
        const dim3 block(128), grid((rows + 3) / 4);
        for (int b0 = 0; b0 < batch; b0 += 4) {
            const int nb = batch - b0 < 4 ? batch - b0 : 4;
            switch (nb) {
                case 1: gemv_kernel<KIND, 1><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, b0); break;
                case 2: gemv_kernel<KIND, 2><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, b0); break;
                case 3: gemv_kernel<KIND, 3><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, b0); break;
                default: gemv_kernel<KIND, 4><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, b0); break;
            }
        }
    } else {
        const dim3 grid((rows + 63) / 64, (batch + 63) / 64), block(256);
        gemm_kernel<KIND><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch);
    }
    return fllm_check_launch("linear");
}

// bitsandbytes NF4 / FP4 / int8 (ids above INT_MAX, so not a launch_kind instantiation): the decode GEMV of bnb.cuh for up to eight tokens, the
// tensor-core GEMM of tc_gemm.cu (blocks decoded into FP16 tiles) from sixteen, else the generic FP32 kernels on decode16
template <uint32_t KIND>
static int launch_bnb(cudaStream_t s, const uint8_t *W, const float *X, float *Y, const float *bias, int rows, int cols, int batch) {
    if (batch <= 8 && fllm_bnb_gemv_try(s, W, KIND, X, Y, bias, rows, cols, batch)) return fllm_check_launch("linear");
    if (batch >= 16 && !fllm_exact_math() && fllm_tc_gemm_try(s, W, (int)KIND, X, Y, bias, rows, cols, batch)) return fllm_check_launch("linear");
    if (batch <= 8) {
        const dim3 block(128), grid((rows + 3) / 4);
        for (int b0 = 0; b0 < batch; b0 += 4) {
            const int nb = batch - b0 < 4 ? batch - b0 : 4;
            switch (nb) {
                case 1: gemv_kernel<KIND, 1><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, b0); break;
                case 2: gemv_kernel<KIND, 2><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, b0); break;
                case 3: gemv_kernel<KIND, 3><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, b0); break;
                default: gemv_kernel<KIND, 4><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, b0); break;
            }
        }
    } else {
        const dim3 grid((rows + 63) / 64, (batch + 63) / 64), block(256);
        gemm_kernel<KIND><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch);
    }
    return fllm_check_launch("linear");
}

static int g_exact = -1;  // -1: not read yet
bool fllm_exact_math() {
    if (g_exact < 0) g_exact = getenv("FLLM_CUDA_EXACT") != nullptr ? 1 : 0;
    return g_exact != 0;
}
// selftest-gpu runs its checks twice: with the exact FP32 kernels and with the fast ones
extern "C" int fllm_cuda_set_exact(int on) {
    g_exact = on ? 1 : 0;
    return 0;
}

static int fllm_linear_launch_impl(cudaStream_t s, const void *W, int kind, const float *X, float *Y, const float *bias,
                                   int rows, int cols, int batch, void *x16, size_t x16_bytes);

// FLLM_CUDA_PROFILE=1: GPU time of every weight projection (events around each launch, so the stream is synchronised after each of them: the
// run is slower than normal), summed per weight type and batch class and printed to stderr every 2000 projections
int fllm_linear_launch(cudaStream_t s, const void *W, int kind, const float *X, float *Y, const float *bias,
                       int rows, int cols, int batch, void *x16, size_t x16_bytes) {
    static const bool profile = getenv("FLLM_CUDA_PROFILE") != nullptr && getenv("FLLM_CUDA_PROFILE")[0] != '0';
    if (!profile) return fllm_linear_launch_impl(s, W, kind, X, Y, bias, rows, cols, batch, x16, x16_bytes);
    struct Acc { double ms = 0; long calls = 0; double macs = 0; };
    static std::map<std::pair<int, int>, Acc> table;
    static long total_calls = 0;
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    cudaEventRecord(a, s);
    const int rc = fllm_linear_launch_impl(s, W, kind, X, Y, bias, rows, cols, batch, x16, x16_bytes);
    cudaEventRecord(b, s);
    cudaEventSynchronize(b);
    float ms = 0.f;
    cudaEventElapsedTime(&ms, a, b);
    cudaEventDestroy(a);
    cudaEventDestroy(b);
    const int cls = batch == 1 ? 1 : batch <= 4 ? 4 : batch <= 8 ? 8 : batch <= 16 ? 16 : batch <= 32 ? 32 : batch <= 64 ? 64 : batch <= 256 ? 256 : 1024;
    Acc &acc = table[{kind, cls}];
    acc.ms += ms;
    acc.calls++;
    acc.macs += (double)rows * cols * batch;
    if (++total_calls % 2000 == 0) {
        fprintf(stderr, "[fllm-gpu-profile] projections: weight type, batch up to, calls, ms per call, GMAC/s\n");
        for (const auto &e : table)
            fprintf(stderr, "[fllm-gpu-profile] kind %d batch<=%d calls %ld %.3f ms %.1f GMAC/s total %.0f ms\n", e.first.first, e.first.second, e.second.calls,
                    e.second.ms / e.second.calls, e.second.macs / (e.second.ms * 1e6), e.second.ms);
    }
    return rc;
}

static int fllm_linear_launch_impl(cudaStream_t s, const void *W, int kind, const float *X, float *Y, const float *bias,
                                   int rows, int cols, int batch, void *x16, size_t x16_bytes) {
    const uint8_t *w = static_cast<const uint8_t *>(W);
    if (fllm_kind_is_mlx(kind)) return fllm_mlx_launch(s, W, kind, X, Y, bias, rows, cols, batch);
    switch ((uint32_t)kind) {
        case KIND_F32: return launch_kind<KIND_F32>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_F16: return launch_kind<KIND_F16>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_BF16: return launch_kind<KIND_BF16>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_Q4_0: return launch_kind<KIND_Q4_0>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_IQ4_NL: return launch_kind<KIND_IQ4_NL>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_MXFP4: return launch_kind<KIND_MXFP4>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_Q4_1: return launch_kind<KIND_Q4_1>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_Q5_0: return launch_kind<KIND_Q5_0>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_Q5_1: return launch_kind<KIND_Q5_1>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_Q8_0: return launch_kind<KIND_Q8_0>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_Q4_K: return launch_kind<KIND_Q4_K>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_Q4_K_MARLIN: return launch_kind<KIND_Q4_K_MARLIN>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_Q5_K: return launch_kind<KIND_Q5_K>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_Q6_K: return launch_kind<KIND_Q6_K>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_Q2_K: return launch_kind<KIND_Q2_K>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_Q2_0: return launch_kind<KIND_Q2_0>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_Q3_K: return launch_kind<KIND_Q3_K>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_IQ2_XXS: return launch_kind<KIND_IQ2_XXS>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_IQ2_XS: return launch_kind<KIND_IQ2_XS>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_IQ3_XXS: return launch_kind<KIND_IQ3_XXS>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_IQ1_S: return launch_kind<KIND_IQ1_S>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_IQ3_S: return launch_kind<KIND_IQ3_S>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_IQ2_S: return launch_kind<KIND_IQ2_S>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_IQ1_M: return launch_kind<KIND_IQ1_M>(s, w, X, Y, bias, rows, cols, batch, x16, x16_bytes);
        case KIND_NF4B64: return launch_bnb<KIND_NF4B64>(s, w, X, Y, bias, rows, cols, batch);
        case KIND_NF4B128: return launch_bnb<KIND_NF4B128>(s, w, X, Y, bias, rows, cols, batch);
        case KIND_FP4B64: return launch_bnb<KIND_FP4B64>(s, w, X, Y, bias, rows, cols, batch);
        case KIND_FP4B128: return launch_bnb<KIND_FP4B128>(s, w, X, Y, bias, rows, cols, batch);
        case KIND_INT8R: return launch_bnb<KIND_INT8R>(s, w, X, Y, bias, rows, cols, batch);
        default: return fllm_set_error("linear: unsupported weight type %d", kind);
    }
}

// Embedding lookup: out[t][c] = weight[ids[t]][c]
__global__ void embedding_kernel(const uint8_t *W, int kind, const int64_t *ids, float *out, int cols, int n) {
    const size_t total = (size_t)n * cols;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const size_t t = i / cols, c = i % cols;
        out[i] = weight_at(W, kind, (size_t)ids[t] * cols + c, cols);
    }
}

int fllm_embedding_launch(cudaStream_t s, const void *W, int kind, const int64_t *ids_dev, float *out, int cols, int n) {
    if (!kind_supported(kind)) return fllm_set_error("embedding: unsupported weight type %d", kind);
    const size_t total = (size_t)n * cols;
    const int block = 256;
    const unsigned grid = (unsigned)((total + block - 1) / block < 65535 ? (total + block - 1) / block : 65535);
    embedding_kernel<<<grid, block, 0, s>>>(static_cast<const uint8_t *>(W), kind, ids_dev, out, cols, n);
    return fllm_check_launch("embedding");
}
