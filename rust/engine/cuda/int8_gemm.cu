// fLLM CUDA backend: int8 W8A8 GEMM on the tensor cores (mma.sync m16n8k32 s8 x s8 -> s32 from sm_80; Turing: four m8n8k16).
//
//   Y[token][row] = sum_k (sx * qx[token][k]) * (sw * qw[row][k]) + bias[row]
//
// Two weight layouts:
//   * Q8_0 (the format the engine converts int8 checkpoints to), in the device layout per row [FP16 scales][int8 values] (padded to 256 values). Activations are
//     quantized per token and 32 values (scale = amax / 127, the usual Q8_0 activation format); every block is one mma k-step whose
//     int32 result is scaled by (activation scale x weight scale) and added to the FP32 running sum.
//   * Plain int8 [rows][cols] with one FP32 scale per row (SmoothQuant / compressed-tensors W8A8 per-channel layout): activations are
//     quantized per token (one scale per row), the int32 accumulation runs over all of K and the scales are applied once in the epilogue.
//
// The kernel reads the operands straight from global memory into the mma fragments (no shared memory staging): the layouts of the
// operands are k-contiguous, so a fragment register is one 32-bit load per lane.
// It is a simple kernel written for fLLM; the fragment layouts are those of the PTX ISA.
//
// Test level of this file: compiled only; the self test group `w8a8` checks it on a device.
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "common.cuh"
#include "fllm_cuda_internal.h"
#include "fp8_gemm.h"
#include "hopper_util.h"

namespace {

// s32 += s8 x s8 over k = 32 (A: 4 registers, B: 2 registers, the fragment layout of mma.m16n8k32; Turing: four m8n8k16)
__device__ __forceinline__ void mma_s8(int c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 750
    // Turing has only m8n8k16: four of them cover the m16n8k32 tile (rows 0..7 / 8..15, k 0..15 / 16..31)
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n" : "+r"(c[0]), "+r"(c[1]) : "r"(a[0]), "r"(b0));
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n" : "+r"(c[0]), "+r"(c[1]) : "r"(a[2]), "r"(b1));
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n" : "+r"(c[2]), "+r"(c[3]) : "r"(a[1]), "r"(b0));
    asm volatile("mma.sync.aligned.m8n8k16.row.col.s32.s8.s8.s32 {%0,%1}, {%2}, {%3}, {%0,%1};\n" : "+r"(c[2]), "+r"(c[3]) : "r"(a[3]), "r"(b1));
#endif
}

__device__ __forceinline__ int8_t q8_round(float v) {
    const float r = rintf(v);
    return (int8_t)(r > 127.f ? 127.f : (r < -127.f ? -127.f : r));
}

// One block (4 warps) per token; every warp quantizes 32 values at a time (one per lane): scale[token][block] = amax / 127.
__global__ void __launch_bounds__(128) quant_blocks_kernel(const float *__restrict__ x, int8_t *__restrict__ q, float *__restrict__ scale, int cols) {
    const int nb = cols >> 5;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int b = warp; b < nb; b += 4) {
        const size_t at = (size_t)blockIdx.x * cols + (size_t)b * 32 + lane;
        const float v = x[at];
        const float m = warp_max(fabsf(v));
        const float d = __fdiv_rn(m, 127.f);
        if (lane == 0) scale[(size_t)blockIdx.x * nb + b] = d;
        q[at] = m > 0.f ? q8_round(__fdiv_rn(v, d)) : (int8_t)0;
    }
}

// One block per token, one scale for the whole row.
__global__ void __launch_bounds__(128) quant_row_kernel(const float *__restrict__ x, int8_t *__restrict__ q, float *__restrict__ scale, int cols) {
    const float *xr = x + (size_t)blockIdx.x * cols;
    float m = 0.f;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) m = fmaxf(m, fabsf(xr[i]));
    m = warp_max(m);
    __shared__ float red[4];
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = m;
    __syncthreads();
    m = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
    const float d = m > 0.f ? __fdiv_rn(m, 127.f) : 1.f;
    if (threadIdx.x == 0) scale[blockIdx.x] = d;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) q[(size_t)blockIdx.x * cols + i] = q8_round(__fdiv_rn(xr[i], d));
}

struct Args {
    const int8_t *xq;
    const float *xs;  // [M][K / 32] (Q8_0) or [M]
    const uint8_t *w;
    const float *ws;  // plain layout: [N]
    float *y;
    const float *bias;
    int M, N, K;
    int padded;  // Q8_0: blocks per row of the device layout (the width rounded up to 256 values)
};

// Q8_0 selects the weight layout. Each of the 4 warps computes MT x NT m16n8 tiles; the block covers 16 * MT tokens and 32 * NT rows.
template <bool Q8_0, int MT, int NT>
__global__ void __launch_bounds__(128) int8_gemm_kernel(Args a) {
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 750
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
    const int m0 = blockIdx.y * MT * 16;
    const int nbase = blockIdx.x * (4 * NT * 8) + warp * NT * 8;
    const int nblk = a.K >> 5;
    int mrow[MT][2];
#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
        mrow[mt][0] = min(m0 + mt * 16 + g, a.M - 1);
        mrow[mt][1] = min(m0 + mt * 16 + g + 8, a.M - 1);
    }
    int nrow[NT];      // row of the B fragment (n = g of the tile)
    int ncol[NT][2];   // rows of the accumulator columns 2t, 2t + 1
#pragma unroll
    for (int nt = 0; nt < NT; nt++) {
        nrow[nt] = min(nbase + nt * 8 + g, a.N - 1);
        ncol[nt][0] = min(nbase + nt * 8 + 2 * t, a.N - 1);
        ncol[nt][1] = min(nbase + nt * 8 + 2 * t + 1, a.N - 1);
    }
    float accf[MT][NT][4];
    int acci[MT][NT][4];
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int nt = 0; nt < NT; nt++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                accf[mt][nt][e] = 0.f;
                acci[mt][nt][e] = 0;
            }

    for (int kb = 0; kb < nblk; kb++) {
        uint32_t af[MT][4];
#pragma unroll
        for (int mt = 0; mt < MT; mt++) {
            const int8_t *pa = a.xq + (size_t)mrow[mt][0] * a.K + kb * 32 + t * 4;
            const int8_t *pb = a.xq + (size_t)mrow[mt][1] * a.K + kb * 32 + t * 4;
            af[mt][0] = *reinterpret_cast<const uint32_t *>(pa);
            af[mt][1] = *reinterpret_cast<const uint32_t *>(pb);
            af[mt][2] = *reinterpret_cast<const uint32_t *>(pa + 16);
            af[mt][3] = *reinterpret_cast<const uint32_t *>(pb + 16);
        }
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
            uint32_t b0, b1;
            float d0 = 1.f, d1 = 1.f;
            if (Q8_0) {
                // device layout of a Q8_0 row (backend/cuda.rs, q8_0_device_layout): [scales: padded * 2 bytes][quants: padded * 32 bytes]
                const size_t row_bytes = (size_t)a.padded * 34;
                const uint8_t *qs = a.w + (size_t)nrow[nt] * row_bytes + (size_t)a.padded * 2 + kb * 32 + t * 4;
                b0 = *reinterpret_cast<const uint32_t *>(qs);
                b1 = *reinterpret_cast<const uint32_t *>(qs + 16);
                d0 = h2f_bits(ld16(a.w + (size_t)ncol[nt][0] * row_bytes + kb * 2));
                d1 = h2f_bits(ld16(a.w + (size_t)ncol[nt][1] * row_bytes + kb * 2));
            } else {
                const uint8_t *p = a.w + (size_t)nrow[nt] * a.K + kb * 32 + t * 4;
                b0 = *reinterpret_cast<const uint32_t *>(p);
                b1 = *reinterpret_cast<const uint32_t *>(p + 16);
            }
#pragma unroll
            for (int mt = 0; mt < MT; mt++) {
                if (Q8_0) {
                    int c[4] = {0, 0, 0, 0};
                    mma_s8(c, af[mt], b0, b1);
                    const float xa = a.xs[(size_t)mrow[mt][0] * nblk + kb], xb = a.xs[(size_t)mrow[mt][1] * nblk + kb];
                    accf[mt][nt][0] += (float)c[0] * (xa * d0);
                    accf[mt][nt][1] += (float)c[1] * (xa * d1);
                    accf[mt][nt][2] += (float)c[2] * (xb * d0);
                    accf[mt][nt][3] += (float)c[3] * (xb * d1);
                } else {
                    mma_s8(acci[mt][nt], af[mt], b0, b1);
                }
            }
        }
    }

#pragma unroll
    for (int mt = 0; mt < MT; mt++) {
#pragma unroll
        for (int nt = 0; nt < NT; nt++) {
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int m = m0 + mt * 16 + g + ((e >> 1) << 3);
                const int n = nbase + nt * 8 + 2 * t + (e & 1);
                if (m >= a.M || n >= a.N) continue;
                float v = Q8_0 ? accf[mt][nt][e] : (float)acci[mt][nt][e] * (a.xs[m] * a.ws[n]);
                if (a.bias) v += a.bias[n];
                a.y[(size_t)m * a.N + n] = v;
            }
        }
    }
#endif
}

template <bool Q8_0>
int launch(cudaStream_t s, const Args &a) {
    if (a.M <= 16) {
        int8_gemm_kernel<Q8_0, 1, 1><<<dim3((a.N + 127) / 128, (a.M + 15) / 16), 128, 0, s>>>(a);
    } else if (a.M <= 64) {
        int8_gemm_kernel<Q8_0, 2, 2><<<dim3((a.N + 255) / 256, (a.M + 31) / 32), 128, 0, s>>>(a);
    } else {
        int8_gemm_kernel<Q8_0, 4, 2><<<dim3((a.N + 255) / 256, (a.M + 63) / 64), 128, 0, s>>>(a);
    }
    return fllm_check_launch("int8 gemm");
}

std::atomic_int g_mode{-2};

}  // namespace

void fllm_int8_gemm_set_mode(int mode) { g_mode.store(mode < -1 || mode > 1 ? -1 : mode); }

static int int8_mode() {
    const int forced = g_mode.load();
    if (forced >= 0) return forced;
    static const int env = fllm_env_tri("FLLM_INT8_GEMM");
    return env;
}

int fllm_int8_q8_try(cudaStream_t s, const void *w, const float *x, float *y, const float *bias, int rows, int cols, int batch) {
    // auto (-1) keeps the FP16 tensor-core kernels until the int8 GEMM has been measured; FLLM_INT8_GEMM=1 turns it on for prompts
    if (int8_mode() != 1 || fllm_exact_math() || fllm_cc() < 75 || batch < 16 || (cols & 31) != 0 || rows <= 0) return -1;
    if ((reinterpret_cast<uintptr_t>(w) & 15) != 0 || (reinterpret_cast<uintptr_t>(x) & 3) != 0) return -1;
    const int nb = cols >> 5;
    W8A8Scratch *sc = w8a8_scratch(s, (size_t)batch * cols, (size_t)batch * nb * sizeof(float));
    if (!sc) return -1;
    int8_t *xq = static_cast<int8_t *>(sc->q);
    quant_blocks_kernel<<<batch, 128, 0, s>>>(x, xq, sc->s, cols);
    Args a{xq, sc->s, static_cast<const uint8_t *>(w), nullptr, y, bias, batch, rows, cols, ((cols + 255) / 256) * 8};
    return launch<true>(s, a);
}

int fllm_int8_plain_gemm(cudaStream_t s, const void *w, const float *wscale, const float *x, float *y, const float *bias, int rows, int cols, int batch) {
    if (fllm_cc() < 75 || (cols & 31) != 0 || rows <= 0 || batch <= 0) return -1;
    if ((reinterpret_cast<uintptr_t>(w) & 3) != 0 || (reinterpret_cast<uintptr_t>(x) & 3) != 0) return -1;
    W8A8Scratch *sc = w8a8_scratch(s, (size_t)batch * cols, (size_t)batch * sizeof(float));
    if (!sc) return fllm_set_error("int8 W8A8 GEMM: no memory for the quantized activations (or the scratch would have to grow inside a CUDA graph capture)");
    int8_t *xq = static_cast<int8_t *>(sc->q);
    quant_row_kernel<<<batch, 128, 0, s>>>(x, xq, sc->s, cols);
    Args a{xq, sc->s, static_cast<const uint8_t *>(w), wscale, y, bias, batch, rows, cols, 0};
    return launch<false>(s, a);
}
