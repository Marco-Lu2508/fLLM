// fLLM CUDA backend: bandwidth-oriented GEMV for the K-quant formats that dominate Q4_K_M models.
//
// Differences to the generic GEMV in linear.cu:
//  - a lane works on a whole 32-element chunk and reads its quantized bytes with 16-byte loads
//    (the 2-byte aligned Q6_K blocks use aligned 32-bit loads and a funnel shift),
//  - the loads of all rows handled by a warp are issued before any arithmetic, so many bytes are
//    in flight per lane,
//  - one warp handles R rows at once so each activation vector is read once for R rows,
//  - the affine part (dmin * min) is applied to the sum of the activations of a sub-block.
// Arithmetic stays FP32 with FP32 activations, matching the generic kernel's tolerance.
#include "common.cuh"
#include "fllm_cuda_internal.h"

#ifndef GEMV_U
#define GEMV_U 2  // chunks per lane in flight
#endif
#ifndef GEMV_R
#define GEMV_R 2  // rows per warp (batch 1)
#endif

// 16 bytes at a 2-byte aligned address as four 32-bit words (little endian).
__device__ __forceinline__ void ld16_a2(const uint8_t *p, uint32_t v[4]) {
    const uintptr_t a = reinterpret_cast<uintptr_t>(p);
    const uint32_t *q = reinterpret_cast<const uint32_t *>(a & ~uintptr_t(3));
    const int sh = (int)(a & 3) * 8;
    uint32_t w[5];
#pragma unroll
    for (int i = 0; i < 5; i++) w[i] = __ldg(q + i);
#pragma unroll
    for (int i = 0; i < 4; i++) v[i] = __funnelshift_r(w[i], w[i + 1], sh);
}


// Byte `m` (0..3) of `w` as a float without an int->float conversion (quarter rate on NVIDIA GPUs):
// the byte becomes the low mantissa bits of 2^23, and 2^23 is subtracted again. `sub` lets the
// caller fold a constant offset into the subtraction (2^23 + 32 for Q6_K).
template <int M>
__device__ __forceinline__ float byte_f(uint32_t w, float sub) {
    return __uint_as_float(__byte_perm(w, 0x4B000000u, 0x7440u | M)) - sub;
}

__device__ __forceinline__ uint32_t byte_of(const uint4 &h, int i) {  // scale bytes live in h.y..h.w
    const uint32_t w = i < 4 ? h.y : (i < 8 ? h.z : h.w);
    return (w >> ((i & 3) * 8)) & 0xffu;
}

__device__ __forceinline__ void sc_min(const uint4 &h, int g, float &sc, float &mn) {
    if (g < 4) {
        sc = (float)(byte_of(h, g) & 63u);
        mn = (float)(byte_of(h, g + 4) & 63u);
    } else {
        sc = (float)((byte_of(h, g + 4) & 15u) | ((byte_of(h, g - 4) >> 6) << 4));
        mn = (float)((byte_of(h, g + 4) >> 4) | ((byte_of(h, g) >> 6) << 4));
    }
}

// ---------------------------------------------------------------------------------------------
// Q4_K: 144-byte blocks (d, dmin, 12 scale bytes, 128 nibble bytes). Chunk = 16 bytes of nibbles:
// low nibbles are 16 elements of sub-block 2j, high nibbles 16 elements of sub-block 2j + 1.
// ---------------------------------------------------------------------------------------------
template <int NB, int R, bool FIVE>
__global__ void __launch_bounds__(128)
gemv_q4k_kernel(const uint8_t *__restrict__ W, const float *__restrict__ X, float *__restrict__ Y,
                const float *__restrict__ bias, int rows, int cols, int batch, int b0) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * 4 + warp) * R;
    if (row0 >= rows) return;
    const int nb = cols >> 8, nchunks = nb * 8;
    constexpr int BB = FIVE ? 176 : 144;  // Q5_K adds 32 bytes of high bits after the scales
    const size_t rowb = (size_t)nb * BB;
    const uint8_t *rp[R];
#pragma unroll
    for (int r = 0; r < R; r++) rp[r] = W + (size_t)min(row0 + r, rows - 1) * rowb;
    float acc[R][NB];
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int b = 0; b < NB; b++) acc[r][b] = 0.f;

    constexpr int U = GEMV_U;
    for (int c0 = lane; c0 < nchunks; c0 += 32 * U) {
        uint4 q[U][R], h[U][R], hb[U][R];
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int c = c0 + 32 * u;
            if (c < nchunks) {
                const int blk = c >> 3, cc = c & 7;
#pragma unroll
                for (int r = 0; r < R; r++) {
                    const uint8_t *bp = rp[r] + (size_t)blk * BB;
                    h[u][r] = __ldg(reinterpret_cast<const uint4 *>(bp));
                    q[u][r] = __ldg(reinterpret_cast<const uint4 *>(bp + (FIVE ? 48 : 16) + (cc >> 1) * 32 + (cc & 1) * 16));
                    if (FIVE) hb[u][r] = __ldg(reinterpret_cast<const uint4 *>(bp + 16 + (cc & 1) * 16));
                }
            }
        }
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int c = c0 + 32 * u;
            if (c >= nchunks) break;
            const int blk = c >> 3, cc = c & 7, j = cc >> 1, t = (cc & 1) * 16;
            const int g0 = 2 * j, g1 = 2 * j + 1;
            // decode the weights once; every token of the batch reuses them
            float wl[R][16], wh[R][16], ds0[R], ds1[R], dm0[R], dm1[R];
#pragma unroll
            for (int r = 0; r < R; r++) {
                const uint32_t qw[4] = {q[u][r].x, q[u][r].y, q[u][r].z, q[u][r].w};
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    uint32_t lo4 = qw[k] & 0x0F0F0F0Fu, hi4 = (qw[k] >> 4) & 0x0F0F0F0Fu;
                    if (FIVE) {
                        const uint32_t hw = k == 0 ? hb[u][r].x : (k == 1 ? hb[u][r].y : (k == 2 ? hb[u][r].z : hb[u][r].w));
                        lo4 |= ((hw >> (2 * j)) & 0x01010101u) << 4;
                        hi4 |= ((hw >> (2 * j + 1)) & 0x01010101u) << 4;
                    }
                    wl[r][4 * k] = byte_f<0>(lo4, 8388608.f);
                    wl[r][4 * k + 1] = byte_f<1>(lo4, 8388608.f);
                    wl[r][4 * k + 2] = byte_f<2>(lo4, 8388608.f);
                    wl[r][4 * k + 3] = byte_f<3>(lo4, 8388608.f);
                    wh[r][4 * k] = byte_f<0>(hi4, 8388608.f);
                    wh[r][4 * k + 1] = byte_f<1>(hi4, 8388608.f);
                    wh[r][4 * k + 2] = byte_f<2>(hi4, 8388608.f);
                    wh[r][4 * k + 3] = byte_f<3>(hi4, 8388608.f);
                }
                const float d = h2f_bits((uint16_t)(h[u][r].x & 0xffffu));
                const float dmin = h2f_bits((uint16_t)(h[u][r].x >> 16));
                float sc0, mn0, sc1, mn1;
                sc_min(h[u][r], g0, sc0, mn0);
                sc_min(h[u][r], g1, sc1, mn1);
                ds0[r] = d * sc0;
                ds1[r] = d * sc1;
                dm0[r] = dmin * mn0;
                dm1[r] = dmin * mn1;
            }
#pragma unroll
            for (int b = 0; b < NB; b++) {
                if (b0 + b >= batch) break;
                const float *xp = X + (size_t)(b0 + b) * cols + blk * 256;
                float4 xl[4], xh[4];
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    xl[i] = __ldg(reinterpret_cast<const float4 *>(xp + g0 * 32 + t) + i);
                    xh[i] = __ldg(reinterpret_cast<const float4 *>(xp + g1 * 32 + t) + i);
                }
                float slo = 0.f, shi = 0.f;
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    slo += xl[i].x + xl[i].y + xl[i].z + xl[i].w;
                    shi += xh[i].x + xh[i].y + xh[i].z + xh[i].w;
                }
#pragma unroll
                for (int r = 0; r < R; r++) {
                    float dlo = 0.f, dhi = 0.f;
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        dlo = fmaf(wl[r][4 * k], xl[k].x, dlo);
                        dlo = fmaf(wl[r][4 * k + 1], xl[k].y, dlo);
                        dlo = fmaf(wl[r][4 * k + 2], xl[k].z, dlo);
                        dlo = fmaf(wl[r][4 * k + 3], xl[k].w, dlo);
                        dhi = fmaf(wh[r][4 * k], xh[k].x, dhi);
                        dhi = fmaf(wh[r][4 * k + 1], xh[k].y, dhi);
                        dhi = fmaf(wh[r][4 * k + 2], xh[k].z, dhi);
                        dhi = fmaf(wh[r][4 * k + 3], xh[k].w, dhi);
                    }
                    acc[r][b] += ds0[r] * dlo + ds1[r] * dhi - dm0[r] * slo - dm1[r] * shi;
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int b = 0; b < NB; b++) acc[r][b] = warp_sum(acc[r][b]);
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; r++) {
            if (row0 + r >= rows) break;
            const float bv = bias ? bias[row0 + r] : 0.f;
#pragma unroll
            for (int b = 0; b < NB; b++)
                if (b0 + b < batch) Y[(size_t)(b0 + b) * rows + row0 + r] = acc[r][b] + bv;
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Q6_K: 224-byte blocks (ql[128], qh[64], scales[16], d, 14 bytes padding). Chunk = (half n, group g) = 32 elements:
// ql bytes n*64 + (g&1)*32 .. +32 (nibble (g>>1)), qh bytes n*32 .. +32 (2 bits at g*2).
// ---------------------------------------------------------------------------------------------
template <int NB, int R>
__global__ void __launch_bounds__(128)
gemv_q6k_kernel(const uint8_t *__restrict__ W, const float *__restrict__ X, float *__restrict__ Y,
                const float *__restrict__ bias, int rows, int cols, int batch, int b0) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * 4 + warp) * R;
    if (row0 >= rows) return;
    const int nb = cols >> 8, nchunks = nb * 8;
    const size_t rowb = (size_t)nb * 224;
    const uint8_t *rp[R];
#pragma unroll
    for (int r = 0; r < R; r++) rp[r] = W + (size_t)min(row0 + r, rows - 1) * rowb;
    float acc[R][NB];
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int b = 0; b < NB; b++) acc[r][b] = 0.f;

    for (int c = lane; c < nchunks; c += 32) {
        const int blk = c >> 3, cc = c & 7, n = cc >> 2, g = cc & 3;
        uint4 ql[R][2], qh[R][2];
        uint32_t sc2[R];
        float d[R];
#pragma unroll
        for (int r = 0; r < R; r++) {
            const uint8_t *bp = rp[r] + (size_t)blk * 224;
            ql[r][0] = __ldg(reinterpret_cast<const uint4 *>(bp + n * 64 + (g & 1) * 32));
            ql[r][1] = __ldg(reinterpret_cast<const uint4 *>(bp + n * 64 + (g & 1) * 32 + 16));
            qh[r][0] = __ldg(reinterpret_cast<const uint4 *>(bp + 128 + n * 32));
            qh[r][1] = __ldg(reinterpret_cast<const uint4 *>(bp + 128 + n * 32 + 16));
            sc2[r] = ld16(bp + 192 + n * 8 + g * 2);
            d[r] = h2f_bits(ld16(bp + 208));
        }
        const int lsh = (g >> 1) * 4, hsh = g * 2;
        float w[R][32];
#pragma unroll
        for (int r = 0; r < R; r++) {
            const uint32_t qlw[8] = {ql[r][0].x, ql[r][0].y, ql[r][0].z, ql[r][0].w, ql[r][1].x, ql[r][1].y, ql[r][1].z, ql[r][1].w};
            const uint32_t qhw[8] = {qh[r][0].x, qh[r][0].y, qh[r][0].z, qh[r][0].w, qh[r][1].x, qh[r][1].y, qh[r][1].z, qh[r][1].w};
#pragma unroll
            for (int k = 0; k < 8; k++) {
                const uint32_t q4 = ((qlw[k] >> lsh) & 0x0F0F0F0Fu) | (((qhw[k] >> hsh) & 0x03030303u) << 4);
                constexpr float kSub = 8388608.f + 32.f;
                w[r][4 * k] = byte_f<0>(q4, kSub);
                w[r][4 * k + 1] = byte_f<1>(q4, kSub);
                w[r][4 * k + 2] = byte_f<2>(q4, kSub);
                w[r][4 * k + 3] = byte_f<3>(q4, kSub);
            }
        }
#pragma unroll
        for (int b = 0; b < NB; b++) {
            if (b0 + b >= batch) break;
            const float *xp = X + (size_t)(b0 + b) * cols + blk * 256 + n * 128 + g * 32;
            float4 xv[8];
#pragma unroll
            for (int i = 0; i < 8; i++) xv[i] = __ldg(reinterpret_cast<const float4 *>(xp) + i);
#pragma unroll
            for (int r = 0; r < R; r++) {
                float part[2] = {0.f, 0.f};
#pragma unroll
                for (int k = 0; k < 8; k++) {
                    part[k >> 2] = fmaf(w[r][4 * k], xv[k].x, part[k >> 2]);
                    part[k >> 2] = fmaf(w[r][4 * k + 1], xv[k].y, part[k >> 2]);
                    part[k >> 2] = fmaf(w[r][4 * k + 2], xv[k].z, part[k >> 2]);
                    part[k >> 2] = fmaf(w[r][4 * k + 3], xv[k].w, part[k >> 2]);
                }
                const float s0 = (float)(int8_t)(sc2[r] & 0xffu), s1 = (float)(int8_t)(sc2[r] >> 8);
                acc[r][b] += d[r] * (s0 * part[0] + s1 * part[1]);
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int b = 0; b < NB; b++) acc[r][b] = warp_sum(acc[r][b]);
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; r++) {
            if (row0 + r >= rows) break;
            const float bv = bias ? bias[row0 + r] : 0.f;
#pragma unroll
            for (int b = 0; b < NB; b++)
                if (b0 + b < batch) Y[(size_t)(b0 + b) * rows + row0 + r] = acc[r][b] + bv;
        }
    }
}


// ---------------------------------------------------------------------------------------------
// Q8_0 in the device layout of backend/cuda.rs (width a multiple of 256): `[f16 scale x blocks][int8 x 32 per block]`.
// A lane owns one block of 32 values; two blocks per lane are in flight.
// ---------------------------------------------------------------------------------------------
template <int NB, int R>
__global__ void __launch_bounds__(128)
gemv_q8_kernel(const uint8_t *__restrict__ W, const float *__restrict__ X, float *__restrict__ Y,
               const float *__restrict__ bias, int rows, int cols, int batch, int b0) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * 4 + warp) * R;
    if (row0 >= rows) return;
    const int nb = cols >> 5, nbp = pad256(cols) >> 5;  // blocks of the row / of its padded layout
    const size_t rowb = (size_t)nbp * 34;
    const uint8_t *rp[R];
#pragma unroll
    for (int r = 0; r < R; r++) rp[r] = W + (size_t)min(row0 + r, rows - 1) * rowb;
    float acc[R][NB];
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int b = 0; b < NB; b++) acc[r][b] = 0.f;

    for (int blk0 = lane; blk0 < nb; blk0 += 64) {
        uint4 q[2][R][2];
        float d[2][R];
#pragma unroll
        for (int u = 0; u < 2; u++) {
            const int blk = blk0 + 32 * u;
            if (blk < nb) {
#pragma unroll
                for (int r = 0; r < R; r++) {
                    const uint8_t *qp = rp[r] + (size_t)nbp * 2 + (size_t)blk * 32;
                    q[u][r][0] = __ldg(reinterpret_cast<const uint4 *>(qp));
                    q[u][r][1] = __ldg(reinterpret_cast<const uint4 *>(qp + 16));
                    d[u][r] = h2f_bits(ld16(rp[r] + (size_t)blk * 2));
                }
            }
        }
#pragma unroll
        for (int u = 0; u < 2; u++) {
            const int blk = blk0 + 32 * u;
            if (blk >= nb) break;
            float w[R][32];
#pragma unroll
            for (int r = 0; r < R; r++) {
                const uint32_t qw[8] = {q[u][r][0].x, q[u][r][0].y, q[u][r][0].z, q[u][r][0].w,
                                        q[u][r][1].x, q[u][r][1].y, q[u][r][1].z, q[u][r][1].w};
#pragma unroll
                for (int k = 0; k < 8; k++) {
                    const uint32_t v = qw[k] ^ 0x80808080u;  // bytes q + 128
                    constexpr float kSub = 8388608.f + 128.f;
                    w[r][4 * k] = byte_f<0>(v, kSub);
                    w[r][4 * k + 1] = byte_f<1>(v, kSub);
                    w[r][4 * k + 2] = byte_f<2>(v, kSub);
                    w[r][4 * k + 3] = byte_f<3>(v, kSub);
                }
            }
#pragma unroll
            for (int b = 0; b < NB; b++) {
                if (b0 + b >= batch) break;
                const float4 *xp = reinterpret_cast<const float4 *>(X + (size_t)(b0 + b) * cols + (size_t)blk * 32);
                float4 xv[8];
#pragma unroll
                for (int i = 0; i < 8; i++) xv[i] = __ldg(xp + i);
#pragma unroll
                for (int r = 0; r < R; r++) {
                    float sum = 0.f;
#pragma unroll
                    for (int k = 0; k < 8; k++) {
                        sum = fmaf(w[r][4 * k], xv[k].x, sum);
                        sum = fmaf(w[r][4 * k + 1], xv[k].y, sum);
                        sum = fmaf(w[r][4 * k + 2], xv[k].z, sum);
                        sum = fmaf(w[r][4 * k + 3], xv[k].w, sum);
                    }
                    acc[r][b] = fmaf(d[u][r], sum, acc[r][b]);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int b = 0; b < NB; b++) acc[r][b] = warp_sum(acc[r][b]);
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; r++) {
            if (row0 + r >= rows) break;
            const float bv = bias ? bias[row0 + r] : 0.f;
#pragma unroll
            for (int b = 0; b < NB; b++)
                if (b0 + b < batch) Y[(size_t)(b0 + b) * rows + row0 + r] = acc[r][b] + bv;
        }
    }
}


// ---------------------------------------------------------------------------------------------
// Q4_0 / Q4_1 / Q5_0 / Q5_1 in the device layout of backend/cuda.rs (width a multiple of 256):
// per row [d: f16 x blocks][m: f16 x blocks (affine)][qh: u32 x blocks (5-bit)][qs: 16 bytes per block].
// A lane owns one block of 32 values (low nibbles: elements 0..15, high nibbles: 16..31).
// ---------------------------------------------------------------------------------------------
#ifndef LEG_U
#define LEG_U 2
#endif
template <int KIND, int NB, int R>
__global__ void __launch_bounds__(128)
gemv_legacy_kernel(const uint8_t *__restrict__ W, const float *__restrict__ X, float *__restrict__ Y,
                   const float *__restrict__ bias, int rows, int cols, int batch, int b0) {
    constexpr bool AFF = KIND == KIND_Q4_1 || KIND == KIND_Q5_1;
    constexpr bool FIVE = KIND == KIND_Q5_0 || KIND == KIND_Q5_1;
    constexpr float OFF = KIND == KIND_Q4_0 ? 8.f : (KIND == KIND_Q5_0 ? 16.f : (FLLM_LUT_KIND(KIND) ? 128.f : 0.f));
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * 4 + warp) * R;
    if (row0 >= rows) return;
    const int nb = cols >> 5, nbp = pad256(cols) >> 5;
    const size_t rowb = (size_t)nbp * kind_block_bytes(KIND);
    const size_t m_off = (size_t)nbp * 2;
    const size_t qh_off = (size_t)nbp * 2 * (AFF ? 2 : 1);
    const size_t qs_off = qh_off + (FIVE ? (size_t)nbp * 4 : 0);
    const uint8_t *rp[R];
#pragma unroll
    for (int r = 0; r < R; r++) rp[r] = W + (size_t)min(row0 + r, rows - 1) * rowb;
    float acc[R][NB];
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int b = 0; b < NB; b++) acc[r][b] = 0.f;

    for (int blk0 = lane; blk0 < nb; blk0 += 32 * LEG_U) {
        uint4 q[LEG_U][R];
        float d[LEG_U][R], mm[LEG_U][R];
        uint32_t qh[LEG_U][R];
#pragma unroll
        for (int u = 0; u < LEG_U; u++) {
            const int blk = blk0 + 32 * u;
            if (blk < nb) {
#pragma unroll
                for (int r = 0; r < R; r++) {
                    q[u][r] = __ldg(reinterpret_cast<const uint4 *>(rp[r] + qs_off + (size_t)blk * 16));
                    d[u][r] = h2f_bits(ld16(rp[r] + (size_t)blk * 2));
                    mm[u][r] = AFF ? h2f_bits(ld16(rp[r] + m_off + (size_t)blk * 2)) : 0.f;
                    qh[u][r] = FIVE ? __ldg(reinterpret_cast<const uint32_t *>(rp[r] + qh_off + (size_t)blk * 4)) : 0u;
                }
            }
        }
#pragma unroll
        for (int u = 0; u < LEG_U; u++) {
            const int blk = blk0 + 32 * u;
            if (blk >= nb) break;
            float w[R][32];
#pragma unroll
            for (int r = 0; r < R; r++) {
                const uint32_t qw[4] = {q[u][r].x, q[u][r].y, q[u][r].z, q[u][r].w};
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    uint32_t lo = qw[k] & 0x0F0F0F0Fu, hi = (qw[k] >> 4) & 0x0F0F0F0Fu;
                    if (FLLM_LUT_KIND(KIND)) {
                        lo = lut4t<KIND, 128>(lo);
                        hi = lut4t<KIND, 128>(hi);
                    }
                    if (FIVE) {
                        // bits 4k..4k+3 (low half) and 16+4k.. (high half) of qh go to bit 4 of the four bytes
                        lo |= (((((qh[u][r] >> (4 * k)) & 0xFu) * 0x00204081u) & 0x01010101u) << 4);
                        hi |= (((((qh[u][r] >> (16 + 4 * k)) & 0xFu) * 0x00204081u) & 0x01010101u) << 4);
                    }
                    w[r][4 * k] = byte_f<0>(lo, 8388608.f);
                    w[r][4 * k + 1] = byte_f<1>(lo, 8388608.f);
                    w[r][4 * k + 2] = byte_f<2>(lo, 8388608.f);
                    w[r][4 * k + 3] = byte_f<3>(lo, 8388608.f);
                    w[r][16 + 4 * k] = byte_f<0>(hi, 8388608.f);
                    w[r][16 + 4 * k + 1] = byte_f<1>(hi, 8388608.f);
                    w[r][16 + 4 * k + 2] = byte_f<2>(hi, 8388608.f);
                    w[r][16 + 4 * k + 3] = byte_f<3>(hi, 8388608.f);
                }
            }
#pragma unroll
            for (int b = 0; b < NB; b++) {
                if (b0 + b >= batch) break;
                const float4 *xp = reinterpret_cast<const float4 *>(X + (size_t)(b0 + b) * cols + (size_t)blk * 32);
                float4 xv[8];
#pragma unroll
                for (int i = 0; i < 8; i++) xv[i] = __ldg(xp + i);
                float sumx = 0.f;
                if (OFF != 0.f || AFF) {
#pragma unroll
                    for (int i = 0; i < 8; i++) sumx += xv[i].x + xv[i].y + xv[i].z + xv[i].w;
                }
#pragma unroll
                for (int r = 0; r < R; r++) {
                    float sum = 0.f;
#pragma unroll
                    for (int k = 0; k < 8; k++) {
                        sum = fmaf(w[r][4 * k], xv[k].x, sum);
                        sum = fmaf(w[r][4 * k + 1], xv[k].y, sum);
                        sum = fmaf(w[r][4 * k + 2], xv[k].z, sum);
                        sum = fmaf(w[r][4 * k + 3], xv[k].w, sum);
                    }
                    acc[r][b] = fmaf(d[u][r], sum, acc[r][b]);
                    if (OFF != 0.f || AFF) acc[r][b] = fmaf(mm[u][r] - d[u][r] * OFF, sumx, acc[r][b]);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int b = 0; b < NB; b++) acc[r][b] = warp_sum(acc[r][b]);
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < R; r++) {
            if (row0 + r >= rows) break;
            const float bv = bias ? bias[row0 + r] : 0.f;
#pragma unroll
            for (int b = 0; b < NB; b++)
                if (b0 + b < batch) Y[(size_t)(b0 + b) * rows + row0 + r] = acc[r][b] + bv;
        }
    }
}

// Returns 1 when the fast path handled the call (batch <= 8, aligned activations).
template <int KIND>
static int run(cudaStream_t s, const uint8_t *W, const float *X, float *Y, const float *bias, int rows, int cols,
               int batch) {
    const dim3 block(128);
#define LAUNCH(NBV, RV)                                                                                   \
    do {                                                                                                  \
        const dim3 grid((rows + 4 * (RV) - 1) / (4 * (RV)));                                               \
        if (KIND == KIND_Q4_K) gemv_q4k_kernel<NBV, RV, false><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, 0); \
        else if (KIND == KIND_Q8_0) gemv_q8_kernel<NBV, RV><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, 0); \
        else if (KIND == KIND_Q4_0 || FLLM_LUT_KIND(KIND) || KIND == KIND_Q4_1 || KIND == KIND_Q5_0 || KIND == KIND_Q5_1) gemv_legacy_kernel<KIND, NBV, RV><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, 0); \
        else if (KIND == KIND_Q5_K) gemv_q4k_kernel<NBV, RV, true><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, 0); \
        else gemv_q6k_kernel<NBV, RV><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, 0);         \
    } while (0)
    switch (batch) {
        case 1: LAUNCH(1, GEMV_R); break;
        case 2: LAUNCH(2, 1); break;
        case 3: LAUNCH(3, 1); break;
        case 4: LAUNCH(4, 1); break;
        case 5: LAUNCH(5, 1); break;
        case 6: LAUNCH(6, 1); break;
        case 7: LAUNCH(7, 1); break;
        default: LAUNCH(8, 1); break;
    }
#undef LAUNCH
    return 1;
}

int fllm_gemv2_try(cudaStream_t s, const void *W, int kind, const float *X, float *Y, const float *bias, int rows,
                   int cols, int batch) {
    if (batch > 8) return 0;
    const bool narrow_ok = kind == KIND_Q4_0 || kind == KIND_Q4_1 || kind == KIND_Q5_0 || kind == KIND_Q5_1 ||
                           kind == KIND_Q8_0 || kind == KIND_IQ4_NL || kind == KIND_MXFP4;  // widths of any multiple of 32
    if ((cols & (narrow_ok ? 31 : 255)) != 0) return 0;
    if ((reinterpret_cast<uintptr_t>(X) & 15) != 0 || (reinterpret_cast<uintptr_t>(W) & 15) != 0) return 0;
    const uint8_t *w = static_cast<const uint8_t *>(W);
    if (kind == KIND_Q4_K) return run<KIND_Q4_K>(s, w, X, Y, bias, rows, cols, batch);
    if (kind == KIND_Q5_K) return run<KIND_Q5_K>(s, w, X, Y, bias, rows, cols, batch);
    if (kind == KIND_Q8_0) return run<KIND_Q8_0>(s, w, X, Y, bias, rows, cols, batch);
    if (kind == KIND_Q4_0) return run<KIND_Q4_0>(s, w, X, Y, bias, rows, cols, batch);
    if (kind == KIND_IQ4_NL) return run<KIND_IQ4_NL>(s, w, X, Y, bias, rows, cols, batch);
    if (kind == KIND_MXFP4) return run<KIND_MXFP4>(s, w, X, Y, bias, rows, cols, batch);
    if (kind == KIND_Q4_1) return run<KIND_Q4_1>(s, w, X, Y, bias, rows, cols, batch);
    if (kind == KIND_Q5_0) return run<KIND_Q5_0>(s, w, X, Y, bias, rows, cols, batch);
    if (kind == KIND_Q5_1) return run<KIND_Q5_1>(s, w, X, Y, bias, rows, cols, batch);
    if (kind == KIND_Q6_K) return run<KIND_Q6_K>(s, w, X, Y, bias, rows, cols, batch);
    return 0;
}
