// fLLM CUDA backend: int8 tensor-core GEMM for Q4_K weights (prompt processing).
//
//   Y[token][row] = sum_k X[token][k] * W[row][k]  (+ bias[row])
//
// The activations are quantized to int8 with one scale per 32 values (the Q8_1 layout used by
// llama.cpp's MMQ kernels). The Q4_K nibbles then go to the tensor cores unchanged: a 32-value
// sub-block of one row is exactly one k32 step of `mma.m16n8k32.s8`, and the sub-block scale and
// minimum are applied to the int32 result in registers:
//
//   w = d * sc * q - dmin * mn,   x = dx * xq
//   sum_k w x = dx * (d * sc * sum(q xq) - dmin * mn * sum(xq))
//
// Weights and activations travel to shared memory through multi-stage `cp.async` pipelines; nothing
// is decoded in shared memory. The int8 tensor rate is twice the FP16 rate, and unlike the FP16
// kernel in qgemm4k.cu the weights are never converted. Rounding the activations to 8 bits costs
// about 1% relative error per output (the same trade llama.cpp makes).
#include <cstdlib>

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

#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)
__device__ __forceinline__ void qm_cp16(void *smem, const void *gmem, bool valid) {
    const uint32_t d = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
    const int sz = valid ? 16 : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(d), "l"(gmem), "r"(sz));
}
__device__ __forceinline__ void qm_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void qm_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }
#else
// Turing has neither cp.async nor m16n8k32.s8: the kernel is compiled empty there and never launched (fllm_qmma4k_try)
__device__ __forceinline__ void qm_cp16(void *, const void *, bool) {}
__device__ __forceinline__ void qm_commit() {}
template <int N>
__device__ __forceinline__ void qm_wait() {}
#endif
__device__ __forceinline__ void qm_ldsm(uint32_t &r0, uint32_t &r1, uint32_t &r2, uint32_t &r3, const void *p) {
    const uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a));
}
#if !defined(__CUDA_ARCH__) || (__CUDA_ARCH__ >= 800)
__device__ __forceinline__ void qm_mma(int c[4], const uint32_t a[4], uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0));
}

#else
__device__ __forceinline__ void qm_mma(int[4], const uint32_t[4], uint32_t, uint32_t) {}
#endif

constexpr int QM_A = 48;  // per row and k-tile: 16 header bytes + 32 nibble bytes
constexpr int QM_B = 80;  // per token and k-tile: 64 int8 + 2 x {dx, dx * sum}

template <int BM, int BN, int WM, int WN, int STAGES>
__global__ void __launch_bounds__(WM * WN * 32)
qmma4k_kernel(const uint8_t *__restrict__ W, const int8_t *__restrict__ Xq, const float2 *__restrict__ Xs,
              float *__restrict__ Y, const float *__restrict__ bias, int rows, int cols, int batch) {
    constexpr int NT = WM * WN * 32;
    constexpr int MI = BM / WM / 16;
    constexpr int NI = BN / WN / 8;
    static_assert(NI % 2 == 0 && MI >= 1, "tile shape");
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char *As = smem;                                    // [STAGES][BM][48]
    unsigned char *Bs = smem + (size_t)STAGES * BM * QM_A;       // [STAGES][BN][80]
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp % WM, wn = warp / WM;
    const int g = lane >> 2, t4 = lane & 3;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    const int nsb = cols >> 8;
    const size_t row_bytes = (size_t)nsb * 144;
    const int ktiles = cols >> 6;
    const int sgroups = cols >> 5;

    auto issue = [&](int kt, int stage) {
        const int sb = kt >> 2, sub = kt & 3;
        for (int c = tid; c < BM * 3; c += NT) {
            const int r = c / 3, part = c % 3;
            const int row = m0 + r;
            const bool ok = row < rows;
            const uint8_t *src = W + (size_t)(ok ? row : 0) * row_bytes + (size_t)sb * 144 +
                                 (part == 0 ? 0 : 16 + sub * 32 + (part - 1) * 16);
            qm_cp16(As + ((size_t)stage * BM + r) * QM_A + part * 16, src, ok);
        }
        for (int c = tid; c < BN * 5; c += NT) {  // 4 chunks of int8 values + 1 chunk of scales per token
            const int n = c / 5, part = c % 5;
            const int tok = n0 + n;
            const bool ok = tok < batch;
            const void *src = part < 4 ? static_cast<const void *>(Xq + (size_t)(ok ? tok : 0) * cols + (size_t)kt * 64 + part * 16)
                                       : static_cast<const void *>(Xs + (size_t)(ok ? tok : 0) * sgroups + (size_t)kt * 2);
            qm_cp16(Bs + ((size_t)stage * BN + n) * QM_B + part * 16, src, ok);
        }
    };

    float acc[MI][NI][4];
#pragma unroll
    for (int a = 0; a < MI; a++)
#pragma unroll
        for (int b = 0; b < NI; b++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[a][b][c] = 0.f;

#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < ktiles) issue(s, s);
        qm_commit();
    }

    for (int kt = 0; kt < ktiles; kt++) {
        qm_wait<STAGES - 2>();
        __syncthreads();
        if (kt + STAGES - 1 < ktiles) issue(kt + STAGES - 1, (kt + STAGES - 1) % STAGES);
        qm_commit();

        const int stage = kt % STAGES;
        const unsigned char *a_stage = As + (size_t)stage * BM * QM_A;
        const unsigned char *b_stage = Bs + (size_t)stage * BN * QM_B;
        const int g0 = 2 * (kt & 3);

        float sw[MI][2][2], mw[MI][2][2];  // [mi][row half][sub-block]: d * sc and dmin * mn
#pragma unroll
        for (int mi = 0; mi < MI; mi++)
#pragma unroll
            for (int rh = 0; rh < 2; rh++) {
                const int r = wm * (BM / WM) + mi * 16 + g + rh * 8;
                const uint4 h = *reinterpret_cast<const uint4 *>(a_stage + (size_t)r * QM_A);
                const float d = h2f_bits((uint16_t)(h.x & 0xffffu)), dmin = h2f_bits((uint16_t)(h.x >> 16));
#pragma unroll
                for (int sbk = 0; sbk < 2; sbk++) {
                    const int gi = g0 + sbk;
                    auto byte_of = [&](int i) -> uint32_t {
                        const uint32_t w = i < 4 ? h.y : (i < 8 ? h.z : h.w);
                        return (w >> ((i & 3) * 8)) & 0xffu;
                    };
                    float sc, mn;
                    if (gi < 4) {
                        sc = (float)(byte_of(gi) & 63u);
                        mn = (float)(byte_of(gi + 4) & 63u);
                    } else {
                        sc = (float)((byte_of(gi + 4) & 15u) | ((byte_of(gi - 4) >> 6) << 4));
                        mn = (float)((byte_of(gi + 4) >> 4) | ((byte_of(gi) >> 6) << 4));
                    }
                    sw[mi][rh][sbk] = d * sc;
                    mw[mi][rh][sbk] = dmin * mn;
                }
            }

#pragma unroll
        for (int grp = 0; grp < 2; grp++) {
            uint32_t af[MI][4];
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
                for (int rh = 0; rh < 2; rh++) {
                    const int r = wm * (BM / WM) + mi * 16 + g + rh * 8;
                    const unsigned char *row = a_stage + (size_t)r * QM_A + 16 + 4 * t4;
                    uint32_t w0 = *reinterpret_cast<const uint32_t *>(row);
                    uint32_t w1 = *reinterpret_cast<const uint32_t *>(row + 16);
                    if (grp) { w0 >>= 4; w1 >>= 4; }
                    af[mi][rh] = w0 & 0x0F0F0F0Fu;
                    af[mi][2 + rh] = w1 & 0x0F0F0F0Fu;
                }
#pragma unroll
            for (int np = 0; np < NI / 2; np++) {
                uint32_t bf[4];
                const int n = wn * (BN / WN) + np * 16 + (lane & 7) + ((lane >> 4) << 3);
                qm_ldsm(bf[0], bf[1], bf[2], bf[3], b_stage + (size_t)n * QM_B + grp * 32 + ((lane >> 3) & 1) * 16);
#pragma unroll
                for (int h2 = 0; h2 < 2; h2++) {
                    const int ni = np * 2 + h2;
                    const int col = wn * (BN / WN) + ni * 8 + 2 * t4;
                    const float2 s0 = *reinterpret_cast<const float2 *>(b_stage + (size_t)col * QM_B + 64 + grp * 8);
                    const float2 s1 = *reinterpret_cast<const float2 *>(b_stage + (size_t)(col + 1) * QM_B + 64 + grp * 8);
#pragma unroll
                    for (int mi = 0; mi < MI; mi++) {
                        int c[4];
                        qm_mma(c, af[mi], bf[h2 * 2], bf[h2 * 2 + 1]);
                        const float swa = sw[mi][0][grp], swb = sw[mi][1][grp];
                        const float mwa = mw[mi][0][grp], mwb = mw[mi][1][grp];
                        acc[mi][ni][0] = fmaf(-mwa, s0.y, fmaf((float)c[0] * swa, s0.x, acc[mi][ni][0]));
                        acc[mi][ni][1] = fmaf(-mwa, s1.y, fmaf((float)c[1] * swa, s1.x, acc[mi][ni][1]));
                        acc[mi][ni][2] = fmaf(-mwb, s0.y, fmaf((float)c[2] * swb, s0.x, acc[mi][ni][2]));
                        acc[mi][ni][3] = fmaf(-mwb, s1.y, fmaf((float)c[3] * swb, s1.x, acc[mi][ni][3]));
                    }
                }
            }
        }
    }
    qm_wait<0>();

#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int ni = 0; ni < NI; ni++)
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int m = m0 + wm * (BM / WM) + mi * 16 + g + (c >> 1) * 8;
                const int n = n0 + wn * (BN / WN) + ni * 8 + 2 * t4 + (c & 1);
                if (m < rows && n < batch) Y[(size_t)n * rows + m] = acc[mi][ni][c] + (bias ? bias[m] : 0.f);
            }
}

template <int BM, int BN, int WM, int WN, int STAGES>
static int qm_launch(cudaStream_t s, const uint8_t *W, const int8_t *xq, const float2 *xs, float *Y, const float *bias,
                     int rows, int cols, int batch) {
    constexpr size_t smem = (size_t)STAGES * (BM * QM_A + BN * QM_B);
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(qmma4k_kernel<BM, BN, WM, WN, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        configured = true;
    }
    const dim3 grid((rows + BM - 1) / BM, (batch + BN - 1) / BN), block(WM * WN * 32);
    qmma4k_kernel<BM, BN, WM, WN, STAGES><<<grid, block, smem, s>>>(W, xq, xs, Y, bias, rows, cols, batch);
    return 1;
}

#ifndef QM_BM
#define QM_BM 128
#endif
#ifndef QM_BN
#define QM_BN 128
#endif
#ifndef QM_WM
#define QM_WM 4
#endif
#ifndef QM_WN
#define QM_WN 2
#endif
#ifndef QM_ST
#define QM_ST 4
#endif

// `scratch` holds batch * cols bytes of int8 followed by batch * cols / 32 float2 scales. Returns 1
// when handled, 0 otherwise.
int fllm_qmma4k_try(cudaStream_t s, const void *W, const float *X, float *Y, const float *bias, int rows, int cols,
                    int batch, void *scratch, size_t scratch_bytes) {
    if (fllm_cc() < 80) return 0;
    const size_t xq_bytes = ((size_t)batch * cols + 255) & ~size_t(255);
    const size_t need = xq_bytes + (size_t)batch * (cols / 32) * sizeof(float2);
    if (batch < 2 || (cols & 255) != 0 || (reinterpret_cast<uintptr_t>(X) & 15) != 0 ||
        (reinterpret_cast<uintptr_t>(W) & 15) != 0 || need > scratch_bytes)
        return 0;
    int8_t *xq = static_cast<int8_t *>(scratch);
    float2 *xs = reinterpret_cast<float2 *>(static_cast<char *>(scratch) + xq_bytes);
    const size_t groups = (size_t)batch * cols / 32;
    quantize_x8_kernel<<<(unsigned)((groups * 32 + 255) / 256), 256, 0, s>>>(X, xq, xs, groups);
    return qm_launch<QM_BM, QM_BN, QM_WM, QM_WN, QM_ST>(s, static_cast<const uint8_t *>(W), xq, xs, Y, bias, rows, cols,
                                                       batch);
}
