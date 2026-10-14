// fLLM CUDA backend: tensor-core GEMM for the 32-value legacy formats Q4_0, Q4_1, Q5_0 and Q5_1 (see qgemm4k.cu).
//
// Device layout of a row (width a multiple of 256, built by backend/cuda.rs):
//   [d: f16 x blocks][m: f16 x blocks (Q4_1, Q5_1)][qh: u32 x blocks (Q5_0, Q5_1)][quants: 16 bytes x blocks]
// A pipeline stage covers G blocks (G = 8: 256 values for narrow batches, G = 4: 128 values for large ones). Block
// b holds element e (0..15) in the low and element e + 16 in the high nibble of byte e; the fifth bit of element i is
// bit i of qh. The value is d * (n - 8), d * (n - 16) or d * n + m.
#include <cstdlib>

#include "common.cuh"
#include "fllm_cuda_internal.h"
#include "mma_util.cuh"

__device__ __forceinline__ void cp_async8x(void *smem, const void *gmem, bool valid) { cp_async8(smem, gmem, valid); }

constexpr int leg_round16(int v) { return (v + 15) & ~15; }

template <int KIND, int G>
struct LegLayout {
    static constexpr bool AFF = KIND == KIND_Q4_1 || KIND == KIND_Q5_1;
    static constexpr bool FIVE = KIND == KIND_Q5_0 || KIND == KIND_Q5_1;
    static constexpr int D_OFF = 0;
    static constexpr int M_OFF = leg_round16(G * 2);
    static constexpr int QH_OFF = M_OFF + (AFF ? leg_round16(G * 2) : 0);
    static constexpr int QS_OFF = QH_OFF + (FIVE ? leg_round16(G * 4) : 0);
    static constexpr int SA = QS_OFF + G * 16;  // bytes per weight row and stage
    static constexpr int SB = G * 64;           // bytes per token and stage (G * 32 halves)
};

template <int KIND, int G, int BM, int BN, int WM, int WN, int STAGES>
__global__ void __launch_bounds__(WM * WN * 32)
qgemm_leg_kernel(const uint8_t *__restrict__ W, const __half *__restrict__ X16, float *__restrict__ Y,
                 const float *__restrict__ bias, int rows, int cols, int batch) {
#if FLLM_TC
    using L = LegLayout<KIND, G>;
    constexpr int NT = WM * WN * 32;
    constexpr int MI = BM / WM / 16;
    constexpr int NI = BN / WN / 8;
    constexpr int XCH = G * 4;  // 16-byte chunks of one token's activations in a stage
    static_assert(NI % 2 == 0 && MI >= 1, "tile shape");
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char *As = smem;
    unsigned char *Bs = smem + (size_t)STAGES * BM * L::SA;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp % WM, wn = warp / WM;
    const int g = lane >> 2, t4 = lane & 3;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    const int nb = cols >> 5;
    const size_t row_bytes = (size_t)nb * kind_block_bytes(KIND);
    const size_t m_off = (size_t)nb * 2;
    const size_t qh_off = (size_t)nb * 2 * (L::AFF ? 2 : 1);
    const size_t qs_off = qh_off + (L::FIVE ? (size_t)nb * 4 : 0);
    const int nst_all = nb / G;
    const int splits = gridDim.z;
    const int per = (nst_all + splits - 1) / splits;
    const int st0 = blockIdx.z * per;
    const int nst = max(0, min(nst_all, st0 + per) - st0);

    constexpr int D_OPS = 1, M_OPS = L::AFF ? 1 : 0, Q_OPS = L::FIVE ? (G * 4 + 15) / 16 : 0, S_OPS = G;
    constexpr int OPS = D_OPS + M_OPS + Q_OPS + S_OPS;

    auto issue = [&](int st, int stage) {
        for (int c = tid; c < BM * OPS; c += NT) {
            const int r = c / OPS, op = c % OPS;
            const int row = m0 + r;
            const bool ok = row < rows;
            const uint8_t *rowp = W + (size_t)(ok ? row : 0) * row_bytes;
            unsigned char *dst = As + ((size_t)stage * BM + r) * L::SA;
            int o = op;
            if (o < D_OPS) {
                if (G * 2 >= 16) cp_async16(dst + L::D_OFF, rowp + (size_t)st * (G * 2), ok);
                else cp_async8x(dst + L::D_OFF, rowp + (size_t)st * (G * 2), ok);
                continue;
            }
            o -= D_OPS;
            if (o < M_OPS) {
                if (G * 2 >= 16) cp_async16(dst + L::M_OFF, rowp + m_off + (size_t)st * (G * 2), ok);
                else cp_async8x(dst + L::M_OFF, rowp + m_off + (size_t)st * (G * 2), ok);
                continue;
            }
            o -= M_OPS;
            if (o < Q_OPS) {
                cp_async16(dst + L::QH_OFF + o * 16, rowp + qh_off + (size_t)st * (G * 4) + o * 16, ok);
                continue;
            }
            o -= Q_OPS;
            cp_async16(dst + L::QS_OFF + o * 16, rowp + qs_off + (size_t)st * (G * 16) + o * 16, ok);
        }
        for (int c = tid; c < BN * XCH; c += NT) {
            const int n = c / XCH, chunk = c % XCH;
            const int tok = n0 + n;
            const bool ok = tok < batch;
            const __half *src = X16 + (size_t)(ok ? tok : 0) * cols + (size_t)st * (G * 32) + chunk * 8;
            cp_async16(Bs + ((size_t)stage * BN + n) * L::SB + ((chunk ^ (n & 7)) * 16), src, ok);
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
        if (s < nst) issue(st0 + s, s);
        cp_async_commit();
    }

    for (int sti = 0; sti < nst; sti++) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        if (sti + STAGES - 1 < nst) issue(st0 + sti + STAGES - 1, (sti + STAGES - 1) % STAGES);
        cp_async_commit();

        const int stage = sti % STAGES;
        const unsigned char *a_stage = As + (size_t)stage * BM * L::SA;
        const unsigned char *b_stage = Bs + (size_t)stage * BN * L::SB;

#pragma unroll
        for (int bi = 0; bi < G; bi++) {
#pragma unroll
            for (int step = 0; step < 2; step++) {
                uint32_t af[MI][4];
#pragma unroll
                for (int mi = 0; mi < MI; mi++)
#pragma unroll
                    for (int rh = 0; rh < 2; rh++) {
                        const int r = wm * (BM / WM) + mi * 16 + g + rh * 8;
                        const unsigned char *row = a_stage + (size_t)r * L::SA;
                        const uint32_t w = *reinterpret_cast<const uint32_t *>(row + L::QS_OFF + bi * 16 + 4 * t4);
                        uint32_t n = (step == 0 ? w : (w >> 4)) & 0x0F0F0F0Fu;
                        if (FLLM_LUT_KIND(KIND)) n = lut4t<KIND, 128>(n);
                        if (L::FIVE) {
                            const uint32_t qh = *reinterpret_cast<const uint32_t *>(row + L::QH_OFF + bi * 4);
                            n |= (((((qh >> (step * 16 + 4 * t4)) & 0xFu) * 0x00204081u) & 0x01010101u) << 4);
                        }
                        const float d = h2f_bits(*reinterpret_cast<const uint16_t *>(row + L::D_OFF + bi * 2));
                        float off;
                        if (L::AFF) off = -h2f_bits(*reinterpret_cast<const uint16_t *>(row + L::M_OFF + bi * 2));
                        else off = (FLLM_LUT_KIND(KIND) ? 128.f : (L::FIVE ? 16.f : 8.f)) * d;
                        const __half2 s2 = __half2half2(__float2half_rn(d));
                        const __half2 o2 = __half2half2(__float2half_rn(off));
                        af[mi][rh] = dequant_pair(n, 0x4140u, s2, o2);
                        af[mi][2 + rh] = dequant_pair(n, 0x4342u, s2, o2);
                    }
                uint32_t bf[NI][2];
#pragma unroll
                for (int np = 0; np < NI / 2; np++) {
                    const int n = wn * (BN / WN) + np * 16 + (lane & 7) + ((lane >> 4) << 3);
                    const int chunk = (bi * 2 + step) * 2 + ((lane >> 3) & 1);
                    ldmatrix4(bf[np * 2][0], bf[np * 2][1], bf[np * 2 + 1][0], bf[np * 2 + 1][1],
                              b_stage + (size_t)n * L::SB + ((chunk ^ (n & 7)) * 16));
                }
#pragma unroll
                for (int mi = 0; mi < MI; mi++)
#pragma unroll
                    for (int ni = 0; ni < NI; ni++) mma16816(acc[mi][ni], af[mi], bf[ni][0], bf[ni][1]);
            }
        }
    }
    cp_async_wait<0>();

#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int ni = 0; ni < NI; ni++)
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int m = m0 + wm * (BM / WM) + mi * 16 + g + (c >> 1) * 8;
                const int n = n0 + wn * (BN / WN) + ni * 8 + 2 * t4 + (c & 1);
                if (m < rows && n < batch) {
                    if (splits > 1) Y[(size_t)blockIdx.z * batch * rows + (size_t)n * rows + m] = acc[mi][ni][c];
                    else Y[(size_t)n * rows + m] = acc[mi][ni][c] + (bias ? bias[m] : 0.f);
                }
            }
#endif
}

__global__ void splitk_reduce_leg_kernel(const float *__restrict__ part, float *__restrict__ Y, const float *__restrict__ bias,
                                         int rows, size_t count, int splits) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x) {
        float s = bias ? bias[i % rows] : 0.f;
        for (int z = 0; z < splits; z++) s += part[(size_t)z * count + i];
        Y[i] = s;
    }
}

static int splitk_choice_leg(int grid_blocks, int units, int batch, int rows, size_t spare_bytes) {
    if (grid_blocks >= 92) return 1;
    int splits = (92 + grid_blocks - 1) / grid_blocks;
    if (splits > 8) splits = 8;
    while (splits > 1 && (units / splits < 2 || (size_t)splits * batch * rows * 4 > spare_bytes)) splits--;
    return splits;
}

template <int KIND, int G, int BM, int BN, int WM, int WN, int STAGES>
static int leg_launch(cudaStream_t s, const uint8_t *W, const __half *X16, float *Y, const float *bias, int rows, int cols,
                      int batch, float *part, size_t part_bytes) {
    using L = LegLayout<KIND, G>;
    constexpr size_t smem = (size_t)STAGES * (BM * L::SA + BN * L::SB);
    if (smem > (size_t)fllm_smem_limit()) {
        if constexpr (STAGES > 2) return leg_launch<KIND, G, BM, BN, WM, WN, 2>(s, W, X16, Y, bias, rows, cols, batch, part, part_bytes);
        else return 0;
    }
    static_assert(smem <= 99 * 1024, "shared memory budget");
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(qgemm_leg_kernel<KIND, G, BM, BN, WM, WN, STAGES>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        configured = true;
    }
    const int row_blocks = (rows + BM - 1) / BM;
    const int splits = splitk_choice_leg(row_blocks * ((batch + BN - 1) / BN), (cols >> 5) / G, batch, rows, part_bytes);
    const dim3 grid(row_blocks, (batch + BN - 1) / BN, splits), block(WM * WN * 32);
    qgemm_leg_kernel<KIND, G, BM, BN, WM, WN, STAGES><<<grid, block, smem, s>>>(W, X16, splits > 1 ? part : Y, bias, rows, cols, batch);
    if (splits > 1) {
        const size_t count = (size_t)batch * rows;
        splitk_reduce_leg_kernel<<<(unsigned)((count + 255) / 256), 256, 0, s>>>(part, Y, bias, rows, count, splits);
    }
    return 1;
}

template <int KIND>
static int leg_dispatch(cudaStream_t s, const uint8_t *w, const __half *x16, float *Y, const float *bias, int rows, int cols,
                        int batch, float *part, size_t part_bytes) {
    if (batch <= 16) return leg_launch<KIND, 8, 64, 16, 4, 1, 4>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
    if (batch <= 32) return leg_launch<KIND, 8, 64, 32, 4, 1, 3>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
    if (batch <= 64) return leg_launch<KIND, 8, 64, 64, 4, 1, 2>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
    return leg_launch<KIND, 4, 128, 128, 8, 1, 2>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
}

// GEMM for `batch` >= 2 tokens; the weights must be in the device layout (width a multiple of 256).
int fllm_qgemm_leg_try(cudaStream_t s, const void *W, int kind, const float *X, float *Y, const float *bias, int rows,
                       int cols, int batch, void *x16_raw, size_t x16_bytes) {
    if (fllm_cc() < 75) return 0;
    const int cols_real = cols;
    cols = pad256(cols);
    static const bool disabled = getenv("FLLM_CUDA_NO_QGEMM") != nullptr;
    if (disabled || batch < 2 || (cols_real & 31) != 0 || (reinterpret_cast<uintptr_t>(X) & 15) != 0 ||
        (reinterpret_cast<uintptr_t>(W) & 15) != 0 || (size_t)batch * cols * 2 > x16_bytes)
        return 0;
    fllm_convert_x16_pad(s, X, x16_raw, batch, cols_real, cols);
    const __half *x16 = static_cast<const __half *>(x16_raw);
    const uint8_t *w = static_cast<const uint8_t *>(W);
    const size_t x_used = ((size_t)batch * cols * 2 + 255) & ~size_t(255);
    float *part = reinterpret_cast<float *>(static_cast<char *>(x16_raw) + x_used);
    const size_t part_bytes = x16_bytes > x_used ? x16_bytes - x_used : 0;
    switch (kind) {
        case KIND_Q4_0: return leg_dispatch<KIND_Q4_0>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
        case KIND_IQ4_NL: return leg_dispatch<KIND_IQ4_NL>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
        case KIND_MXFP4: return leg_dispatch<KIND_MXFP4>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
        case KIND_Q4_1: return leg_dispatch<KIND_Q4_1>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
        case KIND_Q5_0: return leg_dispatch<KIND_Q5_0>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
        case KIND_Q5_1: return leg_dispatch<KIND_Q5_1>(s, w, x16, Y, bias, rows, cols, batch, part, part_bytes);
        default: return 0;
    }
}
