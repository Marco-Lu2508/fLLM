// fLLM CUDA backend: integer dot-product kernels for narrow batches (2..4 tokens) of Q4_K and Q6_K.
//
// The activations are quantized to int8 with one scale per 32 values (the Q8_1 layout) so that the
// weight nibbles can be multiplied with `dp4a`: four multiply-adds per instruction and no float
// conversion of the weights. One decoded weight word serves all tokens of the batch, and the int8
// activations are a quarter of the L1 traffic of the FP32 kernels in gemv2.cu. The 8-bit rounding of
// the activations costs about 1% relative error per output, the same trade llama.cpp makes; the
// single-token path keeps FP32 activations.
//
//   sum_k w x = dx * (d * sc * sum(q xq) - dmin * mn * sum(xq))          (Q4_K, per 32-value sub-block)
//   sum_k w x = dx * d * sc * (sum(q xq) - 32 * sum(xq))                 (Q6_K, per 16 values)
#include "common.cuh"
#include "fllm_cuda_internal.h"

// xq[n][k] = round(x / dx) per 32-group, xs[n][g] = {dx, dx * sum(xq)}. One warp per group.
__global__ void quantize_x8_kernel(const float *__restrict__ x, int8_t *__restrict__ xq, float2 *__restrict__ xs,
                                   size_t groups) {
    const size_t gid = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (gid >= groups) return;
    const float v = x[gid * 32 + lane];
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float dx = amax / 127.f;
    const int q = amax > 0.f ? __float2int_rn(v * (127.f / amax)) : 0;
    int s = q;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    xq[gid * 32 + lane] = (int8_t)q;
    if (lane == 0) xs[gid] = make_float2(dx, dx * (float)s);
}

__device__ __forceinline__ uint32_t mv_byte(const uint4 &h, int i) {  // scale bytes live in h.y..h.w
    const uint32_t w = i < 4 ? h.y : (i < 8 ? h.z : h.w);
    return (w >> ((i & 3) * 8)) & 0xffu;
}
__device__ __forceinline__ void mv_scmin(const uint4 &h, int g, float &sc, float &mn) {
    if (g < 4) {
        sc = (float)(mv_byte(h, g) & 63u);
        mn = (float)(mv_byte(h, g + 4) & 63u);
    } else {
        sc = (float)((mv_byte(h, g + 4) & 15u) | ((mv_byte(h, g - 4) >> 6) << 4));
        mn = (float)((mv_byte(h, g + 4) >> 4) | ((mv_byte(h, g) >> 6) << 4));
    }
}

// Q4_K: chunk = 16 bytes of nibbles = 16 values of sub-block 2j (low) and of sub-block 2j + 1 (high).
template <int NB, int R, bool FIVE>
__global__ void __launch_bounds__(128)
mmvq_q4k_kernel(const uint8_t *W_, const int8_t *Xq_, const float2 *Xs_, float *Y_, const float *bias_,
                int rows, int cols, int batch, const uint8_t *const *wtab = nullptr, const int32_t *ids = nullptr,
                int xdiv = 1, const float *const *btab = nullptr) {
    const float *bias = bias_;
    // mixture-of-experts: blockIdx.y picks the slot; its expert's weights come from `wtab`, its input vector is the
    // slot's token (slot / xdiv), its output row block follows the slot
    const uint8_t *W = W_;
    const int8_t *Xq = Xq_;
    const float2 *Xs = Xs_;
    float *Y = Y_;
    if (wtab) {
        const int slot = blockIdx.y;
        W = wtab[ids[slot]];
        Xq += (size_t)(slot / xdiv) * cols;
        Xs += (size_t)(slot / xdiv) * (cols >> 5);
        Y += (size_t)slot * rows;
        if (btab) bias = btab[ids[slot]];
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * 4 + warp) * R;
    if (row0 >= rows) return;
    const int nb = cols >> 8, nchunks = nb * 8, sg = cols >> 5;
    constexpr int BB = FIVE ? 176 : 144;
    const size_t rowb = (size_t)nb * BB;
    const uint8_t *rp[R];
#pragma unroll
    for (int r = 0; r < R; r++) rp[r] = W + (size_t)min(row0 + r, rows - 1) * rowb;
    float acc[R][NB];
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int b = 0; b < NB; b++) acc[r][b] = 0.f;

    for (int c0 = lane; c0 < nchunks; c0 += 64) {
        uint4 q[2][R], h[2][R], hb[2][R];
#pragma unroll
        for (int u = 0; u < 2; u++) {
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
        for (int u = 0; u < 2; u++) {
            const int c = c0 + 32 * u;
            if (c >= nchunks) break;
            const int blk = c >> 3, cc = c & 7, j = cc >> 1, t = (cc & 1) * 16;
            const int g0 = 2 * j, g1 = 2 * j + 1;
            const bool first = (cc & 1) == 0;  // this lane also carries the minimum term of its sub-blocks
            uint32_t lo[R][4], hi[R][4];
            float ds0[R], ds1[R], dm0[R], dm1[R];
#pragma unroll
            for (int r = 0; r < R; r++) {
                const uint32_t qw[4] = {q[u][r].x, q[u][r].y, q[u][r].z, q[u][r].w};
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    lo[r][k] = qw[k] & 0x0F0F0F0Fu;
                    hi[r][k] = (qw[k] >> 4) & 0x0F0F0F0Fu;
                    if (FIVE) {
                        const uint32_t hw = k == 0 ? hb[u][r].x : (k == 1 ? hb[u][r].y : (k == 2 ? hb[u][r].z : hb[u][r].w));
                        lo[r][k] |= ((hw >> (2 * j)) & 0x01010101u) << 4;
                        hi[r][k] |= ((hw >> (2 * j + 1)) & 0x01010101u) << 4;
                    }
                }
                const float d = h2f_bits((uint16_t)(h[u][r].x & 0xffffu));
                const float dmin = h2f_bits((uint16_t)(h[u][r].x >> 16));
                float sc0, mn0, sc1, mn1;
                mv_scmin(h[u][r], g0, sc0, mn0);
                mv_scmin(h[u][r], g1, sc1, mn1);
                ds0[r] = d * sc0;
                ds1[r] = d * sc1;
                dm0[r] = first ? dmin * mn0 : 0.f;
                dm1[r] = first ? dmin * mn1 : 0.f;
            }
#pragma unroll
            for (int b = 0; b < NB; b++) {
                if (b >= batch) break;
                const int8_t *xp = Xq + (size_t)b * cols + blk * 256;
                const uint4 xl = __ldg(reinterpret_cast<const uint4 *>(xp + g0 * 32 + t));
                const uint4 xh = __ldg(reinterpret_cast<const uint4 *>(xp + g1 * 32 + t));
                const float2 s0 = __ldg(Xs + (size_t)b * sg + blk * 8 + g0);
                const float2 s1 = __ldg(Xs + (size_t)b * sg + blk * 8 + g1);
                const uint32_t xlw[4] = {xl.x, xl.y, xl.z, xl.w}, xhw[4] = {xh.x, xh.y, xh.z, xh.w};
#pragma unroll
                for (int r = 0; r < R; r++) {
                    int dl = 0, dh = 0;
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        dl = __dp4a((int)lo[r][k], (int)xlw[k], dl);
                        dh = __dp4a((int)hi[r][k], (int)xhw[k], dh);
                    }
                    acc[r][b] += ds0[r] * s0.x * (float)dl + ds1[r] * s1.x * (float)dh - dm0[r] * s0.y - dm1[r] * s1.y;
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
                if (b < batch) Y[(size_t)b * rows + row0 + r] = acc[r][b] + bv;
        }
    }
}

// Q6_K (224-byte blocks): chunk = (half n, group g) = 32 values; ql bytes n*64 + (g&1)*32 (nibble g>>1),
// qh bytes n*32 (2 bits at 2g), two scales per chunk (16 values each).
template <int NB, int R>
__global__ void __launch_bounds__(128)
mmvq_q6k_kernel(const uint8_t *W_, const int8_t *Xq_, const float2 *Xs_, float *Y_, const float *bias_,
                int rows, int cols, int batch, const uint8_t *const *wtab = nullptr, const int32_t *ids = nullptr,
                int xdiv = 1, const float *const *btab = nullptr) {
    const float *bias = bias_;
    // mixture-of-experts: blockIdx.y picks the slot; its expert's weights come from `wtab`, its input vector is the
    // slot's token (slot / xdiv), its output row block follows the slot
    const uint8_t *W = W_;
    const int8_t *Xq = Xq_;
    const float2 *Xs = Xs_;
    float *Y = Y_;
    if (wtab) {
        const int slot = blockIdx.y;
        W = wtab[ids[slot]];
        Xq += (size_t)(slot / xdiv) * cols;
        Xs += (size_t)(slot / xdiv) * (cols >> 5);
        Y += (size_t)slot * rows;
        if (btab) bias = btab[ids[slot]];
    }
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * 4 + warp) * R;
    if (row0 >= rows) return;
    const int nb = cols >> 8, nchunks = nb * 8, sg = cols >> 5;
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
        uint32_t q4[R][8];
#pragma unroll
        for (int r = 0; r < R; r++) {
            const uint32_t qlw[8] = {ql[r][0].x, ql[r][0].y, ql[r][0].z, ql[r][0].w, ql[r][1].x, ql[r][1].y, ql[r][1].z, ql[r][1].w};
            const uint32_t qhw[8] = {qh[r][0].x, qh[r][0].y, qh[r][0].z, qh[r][0].w, qh[r][1].x, qh[r][1].y, qh[r][1].z, qh[r][1].w};
#pragma unroll
            for (int k = 0; k < 8; k++)
                q4[r][k] = ((qlw[k] >> lsh) & 0x0F0F0F0Fu) | (((qhw[k] >> hsh) & 0x03030303u) << 4);
        }
#pragma unroll
        for (int b = 0; b < NB; b++) {
            if (b >= batch) break;
            const int8_t *xp = Xq + (size_t)b * cols + blk * 256 + n * 128 + g * 32;
            const uint4 x0 = __ldg(reinterpret_cast<const uint4 *>(xp));
            const uint4 x1 = __ldg(reinterpret_cast<const uint4 *>(xp + 16));
            const uint32_t xw[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
            const float2 s = __ldg(Xs + (size_t)b * sg + blk * 8 + n * 4 + g);
            // (q - 32) x needs the sums of the two 16-value halves of x: dp4a against the constant 1
            int s0 = 0, s1 = 0;
#pragma unroll
            for (int k = 0; k < 4; k++) {
                s0 = __dp4a(0x01010101, (int)xw[k], s0);
                s1 = __dp4a(0x01010101, (int)xw[k + 4], s1);
            }
#pragma unroll
            for (int r = 0; r < R; r++) {
                int p0 = 0, p1 = 0;
#pragma unroll
                for (int k = 0; k < 4; k++) {
                    p0 = __dp4a((int)q4[r][k], (int)xw[k], p0);
                    p1 = __dp4a((int)q4[r][k + 4], (int)xw[k + 4], p1);
                }
                const float sc0 = (float)(int8_t)(sc2[r] & 0xffu), sc1 = (float)(int8_t)(sc2[r] >> 8);
                acc[r][b] += d[r] * s.x * (sc0 * (float)(p0 - 32 * s0) + sc1 * (float)(p1 - 32 * s1));
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
                if (b < batch) Y[(size_t)b * rows + row0 + r] = acc[r][b] + bv;
        }
    }
}

int fllm_mmvq_maxb() {
#if defined(__HIPCC__)
    // AMD: the integer-dot kernel beats the FP32 GEMV (4x on a Radeon 840M at 8 tokens) and the matrix-core GEMM (2x at 16 tokens) up to 16 tokens
    static const int v = getenv("FLLM_CUDA_MMVQ_MAXB") ? atoi(getenv("FLLM_CUDA_MMVQ_MAXB")) : 16;
#else
    static const int v = getenv("FLLM_CUDA_MMVQ_MAXB") ? atoi(getenv("FLLM_CUDA_MMVQ_MAXB")) : 4;
#endif
    return v > 16 ? 16 : v;
}
#ifndef MMVQ_MAXB
#define MMVQ_MAXB 4
#endif
#ifndef MMVQ_R
#define MMVQ_R 2
#endif

template <int KIND, int NB>
static void mmvq_launch(cudaStream_t s, const uint8_t *W, const int8_t *xq, const float2 *xs, float *Y,
                        const float *bias, int rows, int cols, int batch) {
    const dim3 block(128), grid((rows + 4 * MMVQ_R - 1) / (4 * MMVQ_R));
    if (KIND == KIND_Q4_K) mmvq_q4k_kernel<NB, MMVQ_R, false><<<grid, block, 0, s>>>(W, xq, xs, Y, bias, rows, cols, batch);
    else if (KIND == KIND_Q5_K) mmvq_q4k_kernel<NB, MMVQ_R, true><<<grid, block, 0, s>>>(W, xq, xs, Y, bias, rows, cols, batch);
    else mmvq_q6k_kernel<NB, MMVQ_R><<<grid, block, 0, s>>>(W, xq, xs, Y, bias, rows, cols, batch);
}

template <int KIND>
static int mmvq_run(cudaStream_t s, const uint8_t *W, const int8_t *xq, const float2 *xs, float *Y, const float *bias,
                    int rows, int cols, int batch) {
    switch (batch) {
        case 1: mmvq_launch<KIND, 1>(s, W, xq, xs, Y, bias, rows, cols, batch); break;
        case 2: mmvq_launch<KIND, 2>(s, W, xq, xs, Y, bias, rows, cols, batch); break;
        case 3: mmvq_launch<KIND, 3>(s, W, xq, xs, Y, bias, rows, cols, batch); break;
        case 4: mmvq_launch<KIND, 4>(s, W, xq, xs, Y, bias, rows, cols, batch); break;
        case 5: mmvq_launch<KIND, 5>(s, W, xq, xs, Y, bias, rows, cols, batch); break;
        case 6: mmvq_launch<KIND, 6>(s, W, xq, xs, Y, bias, rows, cols, batch); break;
        case 7: mmvq_launch<KIND, 7>(s, W, xq, xs, Y, bias, rows, cols, batch); break;
        case 8: mmvq_launch<KIND, 8>(s, W, xq, xs, Y, bias, rows, cols, batch); break;
        case 9: case 10: case 11: case 12: mmvq_launch<KIND, 12>(s, W, xq, xs, Y, bias, rows, cols, batch); break;
        default: mmvq_launch<KIND, 16>(s, W, xq, xs, Y, bias, rows, cols, batch); break;
    }
    return 1;
}

// `scratch` holds batch * cols bytes of int8 followed by batch * cols / 32 float2 scales (it is the same
// buffer the FP16 GEMM uses for its activations, which is always large enough).
int fllm_mmvq_try(cudaStream_t s, const void *W, int kind, const float *X, float *Y, const float *bias, int rows,
                  int cols, int batch, void *scratch, size_t scratch_bytes) {
    if (batch < 1 || batch > fllm_mmvq_maxb() || (cols & 255) != 0) return 0;
    if (kind != KIND_Q4_K && kind != KIND_Q5_K && kind != KIND_Q6_K) return 0;
    const size_t xq_bytes = ((size_t)batch * cols + 255) & ~size_t(255);
    const size_t need = xq_bytes + (size_t)batch * (cols / 32) * sizeof(float2);
    if ((reinterpret_cast<uintptr_t>(X) & 15) != 0 || (reinterpret_cast<uintptr_t>(W) & 15) != 0 || need > scratch_bytes)
        return 0;
    fllm_x16_filled = false;
    int8_t *xq = static_cast<int8_t *>(scratch);
    float2 *xs = reinterpret_cast<float2 *>(static_cast<char *>(scratch) + xq_bytes);
    fllm_x8_filled = true;
    if (!fllm_x8_reuse) {
        const size_t groups = (size_t)batch * cols / 32;
#if !defined(__HIPCC__)
        if (fllm_glu_up) {  // X is the gate of a gated activation that this quantization computes (FLLM_FUSE_GLU_Q8)
            fllm_quantize_x8_glu_launch(s, X, fllm_glu_up, fllm_glu_mode, xq, xs, groups, fllm_glu_half);
        } else
#endif
        {
            quantize_x8_kernel<<<(unsigned)((groups * 32 + 255) / 256), 256, 0, s>>>(X, xq, xs, groups);
        }
    }
    const uint8_t *w = static_cast<const uint8_t *>(W);
    if (kind == KIND_Q5_K) return mmvq_run<KIND_Q5_K>(s, w, xq, xs, Y, bias, rows, cols, batch);
    return kind == KIND_Q4_K ? mmvq_run<KIND_Q4_K>(s, w, xq, xs, Y, bias, rows, cols, batch)
                             : mmvq_run<KIND_Q6_K>(s, w, xq, xs, Y, bias, rows, cols, batch);
}

// Will fllm_mmvq_try take a K-quant product of this width and batch (the order of the branches in launch_kind, linear.cu)? Used by the
// kernels that quantize the activations of the next GEMM themselves (fused.cu).
bool fllm_mmvq_prep_ok(int kind, int cols, int batch) {
#if defined(__HIPCC__)
    (void)kind; (void)cols; (void)batch;
    return false;
#else
    if (fllm_exact_math() || (kind != KIND_Q4_K && kind != KIND_Q5_K && kind != KIND_Q6_K) || (cols & 255) != 0 || batch < 1 ||
        batch > fllm_mmvq_maxb())
        return false;
    static const bool no_mmvq = getenv("FLLM_CUDA_NO_MMVQ") != nullptr;
    static const bool mmvq1 = getenv("FLLM_CUDA_NO_MMVQ1") == nullptr && !no_mmvq;
    static const bool old_gemv = getenv("FLLM_CUDA_OLD_GEMV") != nullptr;
    static const int gemv_max_batch = getenv("FLLM_CUDA_GEMV_MAXB") ? atoi(getenv("FLLM_CUDA_GEMV_MAXB")) : 1;
    if (batch == 1) return mmvq1;
    return !no_mmvq && !(!old_gemv && batch <= gemv_max_batch);
#endif
}

// ---------------------------------------------------------------------------------------------
// Q4_0 / Q4_1 / Q5_0 / Q5_1 / IQ4_NL / Q8_0 in the device layouts of backend/cuda.rs (width a multiple of 256). A lane
// owns one block of 32 weights: 8 dp4a with the 32 int8 activations of the block.
//   Q4_0: d (dx sum(q xq) - 8 dx sum(xq))   Q5_0: ... - 16 ...   Q4_1 / Q5_1: d dx sum(q xq) + m dx sum(xq)
//   IQ4_NL: d dx sum(grid[q] xq)             Q8_0: d dx sum(q xq)
// ---------------------------------------------------------------------------------------------
template <int KIND, int NB, int R>
__global__ void __launch_bounds__(128)
mmvq_leg_kernel(const uint8_t *W_, const int8_t *Xq_, const float2 *Xs_, float *Y_, const float *bias_, int rows,
                int cols, int batch, const uint8_t *const *wtab = nullptr, const int32_t *ids = nullptr, int xdiv = 1,
                const float *const *btab = nullptr) {
    // mixture-of-experts: see mmvq_q4k_kernel
    const uint8_t *W = W_;
    const int8_t *Xq = Xq_;
    const float2 *Xs = Xs_;
    float *Y = Y_;
    const float *bias = bias_;
    if (wtab) {
        const int slot = blockIdx.y;
        W = wtab[ids[slot]];
        Xq += (size_t)(slot / xdiv) * cols;
        Xs += (size_t)(slot / xdiv) * (cols >> 5);
        Y += (size_t)slot * rows;
        if (btab) bias = btab[ids[slot]];
    }
    constexpr bool AFF = KIND == KIND_Q4_1 || KIND == KIND_Q5_1;
    constexpr bool FIVE = KIND == KIND_Q5_0 || KIND == KIND_Q5_1;
    constexpr bool Q8 = KIND == KIND_Q8_0;
    constexpr float OFF = KIND == KIND_Q4_0 ? 8.f : (KIND == KIND_Q5_0 ? 16.f : 0.f);
    constexpr int U = 2;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row0 = (blockIdx.x * 4 + warp) * R;
    if (row0 >= rows) return;
    const int nb = cols >> 5, nbp = pad256(cols) >> 5;
    const size_t rowb = (size_t)nbp * kind_block_bytes(KIND);
    const size_t m_off = (size_t)nbp * 2;
    const size_t qh_off = (size_t)nbp * 2 * (AFF ? 2 : 1);
    const size_t qs_off = Q8 ? (size_t)nbp * 2 : qh_off + (FIVE ? (size_t)nbp * 4 : 0);
    const uint8_t *rp[R];
#pragma unroll
    for (int r = 0; r < R; r++) rp[r] = W + (size_t)min(row0 + r, rows - 1) * rowb;
    float acc[R][NB];
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
        for (int b = 0; b < NB; b++) acc[r][b] = 0.f;

    for (int blk0 = lane; blk0 < nb; blk0 += 32 * U) {
        uint4 q[U][R][Q8 ? 2 : 1];
        float d[U][R], mm[U][R];
        uint32_t qh[U][R];
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int blk = blk0 + 32 * u;
            if (blk < nb) {
#pragma unroll
                for (int r = 0; r < R; r++) {
                    if (Q8) {
                        q[u][r][0] = __ldg(reinterpret_cast<const uint4 *>(rp[r] + qs_off + (size_t)blk * 32));
                        q[u][r][Q8 ? 1 : 0] = __ldg(reinterpret_cast<const uint4 *>(rp[r] + qs_off + (size_t)blk * 32 + 16));
                    } else {
                        q[u][r][0] = __ldg(reinterpret_cast<const uint4 *>(rp[r] + qs_off + (size_t)blk * 16));
                    }
                    d[u][r] = h2f_bits(ld16(rp[r] + (size_t)blk * 2));
                    mm[u][r] = AFF ? h2f_bits(ld16(rp[r] + m_off + (size_t)blk * 2)) : 0.f;
                    qh[u][r] = FIVE ? __ldg(reinterpret_cast<const uint32_t *>(rp[r] + qh_off + (size_t)blk * 4)) : 0u;
                }
            }
        }
#pragma unroll
        for (int u = 0; u < U; u++) {
            const int blk = blk0 + 32 * u;
            if (blk >= nb) break;
            uint32_t qa[R][8];  // the 32 weights of each row as int8 (or unsigned nibbles) in 8 words
#pragma unroll
            for (int r = 0; r < R; r++) {
                if (Q8) {
                    qa[r][0] = q[u][r][0].x; qa[r][1] = q[u][r][0].y; qa[r][2] = q[u][r][0].z; qa[r][3] = q[u][r][0].w;
                    qa[r][4] = q[u][r][Q8 ? 1 : 0].x; qa[r][5] = q[u][r][Q8 ? 1 : 0].y;
                    qa[r][6] = q[u][r][Q8 ? 1 : 0].z; qa[r][7] = q[u][r][Q8 ? 1 : 0].w;
                } else {
                    const uint32_t qw[4] = {q[u][r][0].x, q[u][r][0].y, q[u][r][0].z, q[u][r][0].w};
#pragma unroll
                    for (int k = 0; k < 4; k++) {
                        uint32_t lo = qw[k] & 0x0F0F0F0Fu, hi = (qw[k] >> 4) & 0x0F0F0F0Fu;
                        if (FIVE) {
                            lo |= (((((qh[u][r] >> (4 * k)) & 0xFu) * 0x00204081u) & 0x01010101u) << 4);
                            hi |= (((((qh[u][r] >> (16 + 4 * k)) & 0xFu) * 0x00204081u) & 0x01010101u) << 4);
                        }
                        if (FLLM_LUT_KIND(KIND)) {
                            lo = lut4t<KIND, 0>(lo);
                            hi = lut4t<KIND, 0>(hi);
                        }
                        qa[r][k] = lo;
                        qa[r][4 + k] = hi;
                    }
                }
            }
#pragma unroll
            for (int b = 0; b < NB; b++) {
                if (b >= batch) break;
                const int8_t *xp = Xq + (size_t)b * cols + (size_t)blk * 32;
                const uint4 x0 = __ldg(reinterpret_cast<const uint4 *>(xp)), x1 = __ldg(reinterpret_cast<const uint4 *>(xp + 16));
                const float2 sx = __ldg(Xs + (size_t)b * nb + blk);
                const uint32_t xw[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
                for (int r = 0; r < R; r++) {
                    int dot = 0;
#pragma unroll
                    for (int k = 0; k < 8; k++) dot = __dp4a((int)qa[r][k], (int)xw[k], dot);
                    float v = d[u][r] * (sx.x * (float)dot);
                    if (OFF != 0.f) v -= d[u][r] * OFF * sx.y;
                    if (AFF) v += mm[u][r] * sx.y;
                    acc[r][b] += v;
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
                if (b < batch) Y[(size_t)b * rows + row0 + r] = acc[r][b] + bv;
        }
    }
}

template <int KIND>
static int mmvq_leg_run(cudaStream_t s, const uint8_t *W, const int8_t *xq, const float2 *xs, float *Y, const float *bias,
                        int rows, int cols, int batch) {
    const dim3 block(128);
    switch (batch) {
        case 1: mmvq_leg_kernel<KIND, 1, 2><<<(rows + 7) / 8, block, 0, s>>>(W, xq, xs, Y, bias, rows, cols, batch); break;
        case 2: mmvq_leg_kernel<KIND, 2, 1><<<(rows + 3) / 4, block, 0, s>>>(W, xq, xs, Y, bias, rows, cols, batch); break;
        case 3: mmvq_leg_kernel<KIND, 3, 1><<<(rows + 3) / 4, block, 0, s>>>(W, xq, xs, Y, bias, rows, cols, batch); break;
        default: mmvq_leg_kernel<KIND, 4, 1><<<(rows + 3) / 4, block, 0, s>>>(W, xq, xs, Y, bias, rows, cols, batch); break;
    }
    return 1;
}

int fllm_mmvq_leg_try(cudaStream_t s, const void *W, int kind, const float *X, float *Y, const float *bias, int rows,
                      int cols, int batch, void *scratch, size_t scratch_bytes) {
    if (batch < 1 || batch > 4 || (cols & 31) != 0) return 0;
    const size_t xq_bytes = ((size_t)batch * cols + 255) & ~size_t(255);
    const size_t need = xq_bytes + (size_t)batch * (cols / 32) * sizeof(float2);
    if ((reinterpret_cast<uintptr_t>(X) & 15) != 0 || (reinterpret_cast<uintptr_t>(W) & 15) != 0 || need > scratch_bytes)
        return 0;
    switch (kind) {
        case KIND_Q4_0: case KIND_Q4_1: case KIND_Q5_0: case KIND_Q5_1: case KIND_IQ4_NL: case KIND_MXFP4: case KIND_Q8_0: break;
        default: return 0;
    }
    fllm_x16_filled = false;
    int8_t *xq = static_cast<int8_t *>(scratch);
    float2 *xs = reinterpret_cast<float2 *>(static_cast<char *>(scratch) + xq_bytes);
    fllm_x8_filled = true;
    if (!fllm_x8_reuse) {
        const size_t groups = (size_t)batch * cols / 32;
        quantize_x8_kernel<<<(unsigned)((groups * 32 + 255) / 256), 256, 0, s>>>(X, xq, xs, groups);
    }
    const uint8_t *w = static_cast<const uint8_t *>(W);
    switch (kind) {
        case KIND_Q4_0: return mmvq_leg_run<KIND_Q4_0>(s, w, xq, xs, Y, bias, rows, cols, batch);
        case KIND_Q4_1: return mmvq_leg_run<KIND_Q4_1>(s, w, xq, xs, Y, bias, rows, cols, batch);
        case KIND_Q5_0: return mmvq_leg_run<KIND_Q5_0>(s, w, xq, xs, Y, bias, rows, cols, batch);
        case KIND_Q5_1: return mmvq_leg_run<KIND_Q5_1>(s, w, xq, xs, Y, bias, rows, cols, batch);
        case KIND_IQ4_NL: return mmvq_leg_run<KIND_IQ4_NL>(s, w, xq, xs, Y, bias, rows, cols, batch);
        case KIND_MXFP4: return mmvq_leg_run<KIND_MXFP4>(s, w, xq, xs, Y, bias, rows, cols, batch);
        default: return mmvq_leg_run<KIND_Q8_0>(s, w, xq, xs, Y, bias, rows, cols, batch);
    }
}

// ---------------------------------------------------------------------------------------------
// Mixture-of-experts: y[slot] = W[ids[slot]] x[slot / xdiv] for Q4_K / Q5_K / Q6_K experts whose device weights are
// listed in a pointer table; the int8 activations are quantized once per input vector.
// ---------------------------------------------------------------------------------------------
int fllm_moe_linear_launch(cudaStream_t s, const void *const *tab, int kind, const int32_t *ids, const float *X, float *Y,
                           int rows, int cols, int slots, int xdiv, const void *const *btab, void *scratch, size_t scratch_bytes) {
    const bool k_quant = kind == KIND_Q4_K || kind == KIND_Q5_K || kind == KIND_Q6_K;
    if ((cols & (k_quant ? 255 : 31)) != 0 || slots <= 0 || xdiv <= 0) return fllm_set_error("moe_linear: unsupported shape");
    const int inputs = (slots + xdiv - 1) / xdiv;
    const size_t xq_bytes = ((size_t)inputs * cols + 255) & ~size_t(255);
    const size_t need = xq_bytes + (size_t)inputs * (cols / 32) * sizeof(float2);
    if (need > scratch_bytes || (reinterpret_cast<uintptr_t>(X) & 15) != 0)
        return fllm_set_error("moe_linear: scratch too small or activations unaligned");
    fllm_x16_filled = false;
    fllm_x8_filled = false;
    int8_t *xq = static_cast<int8_t *>(scratch);
    float2 *xs = reinterpret_cast<float2 *>(static_cast<char *>(scratch) + xq_bytes);
    const size_t groups = (size_t)inputs * cols / 32;
    quantize_x8_kernel<<<(unsigned)((groups * 32 + 255) / 256), 256, 0, s>>>(X, xq, xs, groups);
    const dim3 block(128), grid((rows + 4 * MMVQ_R - 1) / (4 * MMVQ_R), slots);
    const uint8_t *const *wt = reinterpret_cast<const uint8_t *const *>(tab);
    const float *const *bt = reinterpret_cast<const float *const *>(btab);
    if (kind == KIND_Q4_K) mmvq_q4k_kernel<1, MMVQ_R, false><<<grid, block, 0, s>>>(nullptr, xq, xs, Y, nullptr, rows, cols, 1, wt, ids, xdiv, bt);
    else if (kind == KIND_Q5_K) mmvq_q4k_kernel<1, MMVQ_R, true><<<grid, block, 0, s>>>(nullptr, xq, xs, Y, nullptr, rows, cols, 1, wt, ids, xdiv, bt);
    else if (kind == KIND_Q6_K) mmvq_q6k_kernel<1, MMVQ_R><<<grid, block, 0, s>>>(nullptr, xq, xs, Y, nullptr, rows, cols, 1, wt, ids, xdiv, bt);
    else {
        // 32-value formats: one lane per block, two rows per warp
        const dim3 lgrid((rows + 7) / 8, slots);
#define MOE_LEG(K) mmvq_leg_kernel<K, 1, 2><<<lgrid, block, 0, s>>>(nullptr, xq, xs, Y, nullptr, rows, cols, 1, wt, ids, xdiv, bt)
        switch (kind) {
            case KIND_Q4_0: MOE_LEG(KIND_Q4_0); break;
            case KIND_Q4_1: MOE_LEG(KIND_Q4_1); break;
            case KIND_Q5_0: MOE_LEG(KIND_Q5_0); break;
            case KIND_Q5_1: MOE_LEG(KIND_Q5_1); break;
            case KIND_IQ4_NL: MOE_LEG(KIND_IQ4_NL); break;
            case KIND_MXFP4: MOE_LEG(KIND_MXFP4); break;
            case KIND_Q8_0: MOE_LEG(KIND_Q8_0); break;
            default: return fllm_set_error("moe_linear: unsupported weight type %d", kind);
        }
#undef MOE_LEG
    }
    return fllm_check_launch("moe_linear");
}

#include "mmvq_x.cuh"
