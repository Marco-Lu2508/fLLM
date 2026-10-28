// fLLM CUDA backend: MLX affine group-quantized weights (the `MLXA` tensor types, fllm_gguf_probe::MlxAffine).
//
//   Y[token][row] = sum_k X[token][k] * W[row][k] + bias[row],   W[row][k] = code * scale + bias_of_group
//
// Device image of a weight (built by backend/mlx.rs, `device_planes`): three planes in one buffer,
//     codes   rows x (cols * BITS / 8) bytes   the MLX words of each row: code i at bit i * BITS of a little-endian bit stream
//     scales  rows x (cols / group)    u16     half or bfloat16 (`bf16` flag)
//     biases  rows x (cols / group)    u16
// `scales` starts at rows * cols * BITS / 8 (a multiple of 4), `biases` right behind it. A chunk of 32 codes is exactly BITS 32-bit words,
// so every kernel works in chunks of 32 and a code that straddles two words (3, 5 and 6 bits) never leaves its chunk. The group size is any
// multiple of 32 (a chunk lies inside one group). The type id (see the `MLXA` description in gguf-probe) carries bits, group and the
// scale format: 0x80010000 | bf16 << 12 | (group / 32) << 4 | bits.
//
// Three kernels, all reading the packed words directly:
//   * mlx_gemv_kernel   batches of 1..8 tokens: one warp per row, lane l takes chunks l, l + 32, ...; the weights of a chunk are unpacked
//                       once in registers and used for up to 4 tokens; FP32 arithmetic on FP32 activations.
//   * mlx_gemm_kernel   larger batches: 64 x 64 output tile, K step 32, the words of the tile are staged in shared memory and unpacked
//                       into an FP32 tile; FP32 FMA (the exact path, FLLM_CUDA_EXACT, and devices without tensor cores).
//   * mlx_wmma_kernel   prompts on tensor cores (sm_75+): the same staging, but the tile is dequantized into FP16 fragments and the
//                       activations are converted to FP16 on the fly; FP32 accumulation (about 1e-3 relative error, like tc_gemm.cu).
//
// Test level: compiled with nvcc (sm_80/89/90/120); the arithmetic and the indexing are mirrored by a host model in backend/mlx.rs
// (`model_gemv`) that is tested against the CPU reference; the device self test group `mlx` checks all three kernels against the CPU
// reference on a device (not yet run on a GPU).
#include <cstdlib>

#include "common.cuh"
#include "fllm_cuda_internal.h"
#if !defined(__HIPCC__)
#include <mma.h>
#endif

namespace {

constexpr uint32_t kMlxBase = 0x80010000u;

struct MlxType {
    int bits, group;
    bool bf16;
};

// Decodes a type id; false for ids outside the family or with unsupported fields.
bool mlx_decode(uint32_t id, MlxType &t) {
    if ((id & ~0x1fffu) != kMlxBase) return false;
    t.bits = (int)(id & 15u);
    t.group = (int)((id >> 4) & 255u) * 32;
    t.bf16 = (id & 0x1000u) != 0;
    return (t.bits == 2 || t.bits == 3 || t.bits == 4 || t.bits == 5 || t.bits == 6 || t.bits == 8) && t.group > 0;
}

__device__ __forceinline__ float mlx_half(uint16_t raw, int bf16) {
    return bf16 ? __uint_as_float((uint32_t)raw << 16) : __half2float(__ushort_as_half(raw));
}

// Code `i` (0..31) of a chunk whose BITS words start at `w`. A code that crosses a word boundary takes its high bits from the next
// word; that word is read only then, so the read stays inside the chunk (the last code ends exactly at bit 32 * BITS).
template <int BITS>
__device__ __forceinline__ uint32_t mlx_code(const uint32_t *w, int i) {
    const int bit = i * BITS, wi = bit >> 5, sh = bit & 31;
    uint32_t v = w[wi] >> sh;
    if (sh + BITS > 32) v |= w[wi + 1] << (32 - sh);
    return v & ((1u << BITS) - 1u);
}

__device__ __forceinline__ void load32(const float *p, bool vec, float *x) {
    if (vec) {
        const float4 *q = reinterpret_cast<const float4 *>(p);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const float4 v = __ldg(q + i);
            x[4 * i] = v.x; x[4 * i + 1] = v.y; x[4 * i + 2] = v.z; x[4 * i + 3] = v.w;
        }
    } else {
#pragma unroll
        for (int i = 0; i < 32; i++) x[i] = __ldg(p + i);
    }
}

// ---------------------------------------------------------------------------------------------
// GEMV: one warp per row.
// ---------------------------------------------------------------------------------------------
template <int BITS, int NB>
__global__ void __launch_bounds__(128)
mlx_gemv_kernel(const uint8_t *__restrict__ W, const float *__restrict__ X, float *__restrict__ Y, const float *__restrict__ bias,
                int rows, int cols, int batch, int b0, int group, int bf16) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= rows) return;  // whole warp: row depends on the warp only
    const size_t row_words = (size_t)(cols / 32) * BITS, groups_per_row = (size_t)(cols / group);
    const uint32_t *cw = reinterpret_cast<const uint32_t *>(W) + (size_t)row * row_words;
    const uint16_t *sc = reinterpret_cast<const uint16_t *>(W + (size_t)rows * row_words * 4) + (size_t)row * groups_per_row;
    const uint16_t *bi = sc + (size_t)rows * groups_per_row;
    bool vec[NB];
#pragma unroll
    for (int b = 0; b < NB; b++) vec[b] = ((reinterpret_cast<uintptr_t>(X + (size_t)(b0 + b) * cols)) & 15) == 0;
    float acc[NB];
#pragma unroll
    for (int b = 0; b < NB; b++) acc[b] = 0.f;
    const int chunks = cols / 32, chunks_per_group = group / 32;
    for (int chunk = lane; chunk < chunks; chunk += FLLM_WARP) {
        const int g = chunk / chunks_per_group;
        const float s = mlx_half(__ldg(sc + g), bf16), z = mlx_half(__ldg(bi + g), bf16);
        uint32_t words[BITS];
#pragma unroll
        for (int k = 0; k < BITS; k++) words[k] = __ldg(cw + (size_t)chunk * BITS + k);
        float w[32];
#pragma unroll
        for (int i = 0; i < 32; i++) w[i] = fmaf((float)mlx_code<BITS>(words, i), s, z);
#pragma unroll
        for (int b = 0; b < NB; b++) {
            if (b0 + b < batch) {
                float x[32];
                load32(X + (size_t)(b0 + b) * cols + (size_t)chunk * 32, vec[b], x);
#pragma unroll
                for (int i = 0; i < 32; i++) acc[b] = fmaf(w[i], x[i], acc[b]);
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
// GEMM tiles: 64 rows x 64 tokens, K step = one chunk of 32.
// ---------------------------------------------------------------------------------------------
constexpr int BM = 64, BN = 64, BK = 32;

// Stage the words and the group scale / bias of the chunk `chunk` of the 64 rows from `m0`; rows past the end read as zero (code 0, scale 0, bias 0).
template <int BITS, int NT>
__device__ __forceinline__ void stage_chunk(const uint8_t *W, int rows, int cols, int group, int bf16, int m0, int chunk,
                                            uint32_t (*Cw)[9], float *Sc, float *Bi) {
    const size_t row_words = (size_t)(cols / 32) * BITS, groups_per_row = (size_t)(cols / group);
    const uint32_t *cw = reinterpret_cast<const uint32_t *>(W);
    const uint16_t *sc = reinterpret_cast<const uint16_t *>(W + (size_t)rows * row_words * 4);
    const uint16_t *bi = sc + (size_t)rows * groups_per_row;
    for (int e = threadIdx.x; e < BM * BITS; e += NT) {
        const int r = e / BITS, k = e % BITS, m = m0 + r;
        Cw[r][k] = m < rows ? __ldg(cw + (size_t)m * row_words + (size_t)chunk * BITS + k) : 0u;
    }
    if (threadIdx.x < BM) {
        const int m = m0 + threadIdx.x, g = chunk / (group / 32);
        Sc[threadIdx.x] = m < rows ? mlx_half(__ldg(sc + (size_t)m * groups_per_row + g), bf16) : 0.f;
        Bi[threadIdx.x] = m < rows ? mlx_half(__ldg(bi + (size_t)m * groups_per_row + g), bf16) : 0.f;
    }
}

template <int BITS>
__global__ void __launch_bounds__(256)
mlx_gemm_kernel(const uint8_t *__restrict__ W, const float *__restrict__ X, float *__restrict__ Y, const float *__restrict__ bias,
                int rows, int cols, int batch, int group, int bf16) {
    __shared__ float Ws[BK][BM + 1];
    __shared__ float Xs[BK][BN + 1];
    __shared__ uint32_t Cw[BM][9];
    __shared__ float Sc[BM], Bi[BM];
    const int tid = threadIdx.x, tx = tid & 15, ty = tid >> 4;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    const int ld_row = tid >> 2, ld_q = tid & 3;  // 64 rows (tokens) x 4 pieces of 8 values
    float acc[4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j] = 0.f;
    for (int chunk = 0; chunk < cols / 32; chunk++) {
        stage_chunk<BITS, 256>(W, rows, cols, group, bf16, m0, chunk, Cw, Sc, Bi);
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const int i = ld_q * 8 + j;
            Ws[i][ld_row] = fmaf((float)mlx_code<BITS>(Cw[ld_row], i), Sc[ld_row], Bi[ld_row]);
        }
        {
            const int n = n0 + ld_row;
            const float *p = X + (size_t)n * cols + (size_t)chunk * 32 + ld_q * 8;
#pragma unroll
            for (int j = 0; j < 8; j++) Xs[ld_q * 8 + j][ld_row] = n < batch ? __ldg(p + j) : 0.f;
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
// Tensor cores: A = activations (token x k, row major), B = the dequantized weights (k x row, column major = W[row][k] row major),
// 128 threads = 4 warps, each warp owns 32 tokens x 32 rows (2 x 2 fragments of 16 x 16 x 16).
template <int BITS>
__global__ void __launch_bounds__(128)
mlx_wmma_kernel(const uint8_t *__restrict__ W, const float *__restrict__ X, float *__restrict__ Y, const float *__restrict__ bias,
                int rows, int cols, int batch, int group, int bf16) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 700)
    // never launched on such a device (the launcher needs sm_75); keeps the binary buildable for older targets
    return;
#else
    using namespace nvcuda;
    constexpr int LDH = BK + 8, LDC = BM + 4;
    __shared__ __align__(32) __half Xs[BN * LDH];
    __shared__ __align__(32) __half Ws[BM * LDH];
    __shared__ __align__(32) float Cs[BN * LDC];
    __shared__ uint32_t Cw[BM][9];
    __shared__ float Sc[BM], Bi[BM];
    const int tid = threadIdx.x, warp = tid >> 5;
    const int wm = warp >> 1, wn = warp & 1;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    const int ld_row = tid >> 1, ld_h = tid & 1;  // 64 rows (tokens) x 2 halves of 16 values
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c[2][2];
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 2; j++) wmma::fill_fragment(c[i][j], 0.f);
    for (int chunk = 0; chunk < cols / 32; chunk++) {
        stage_chunk<BITS, 128>(W, rows, cols, group, bf16, m0, chunk, Cw, Sc, Bi);
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 16; j++) {
            const int i = ld_h * 16 + j;
            Ws[ld_row * LDH + i] = __float2half_rn(fmaf((float)mlx_code<BITS>(Cw[ld_row], i), Sc[ld_row], Bi[ld_row]));
        }
        {
            const int n = n0 + ld_row;
            const float *p = X + (size_t)n * cols + (size_t)chunk * 32 + ld_h * 16;
#pragma unroll
            for (int j = 0; j < 16; j++) Xs[ld_row * LDH + ld_h * 16 + j] = __float2half_rn(n < batch ? __ldg(p + j) : 0.f);
        }
        __syncthreads();
#pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a[2];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> b[2];
#pragma unroll
            for (int i = 0; i < 2; i++) wmma::load_matrix_sync(a[i], Xs + (wm * 32 + i * 16) * LDH + kk, LDH);
#pragma unroll
            for (int j = 0; j < 2; j++) wmma::load_matrix_sync(b[j], Ws + (wn * 32 + j * 16) * LDH + kk, LDH);
#pragma unroll
            for (int i = 0; i < 2; i++)
#pragma unroll
                for (int j = 0; j < 2; j++) wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 2; j++)
            wmma::store_matrix_sync(Cs + (wm * 32 + i * 16) * LDC + wn * 32 + j * 16, c[i][j], LDC, wmma::mem_row_major);
    __syncthreads();
    for (int idx = tid; idx < BN * BM; idx += 128) {
        const int n = idx / BM, m = idx % BM;
        if (n0 + n < batch && m0 + m < rows) Y[(size_t)(n0 + n) * rows + m0 + m] = Cs[n * LDC + m] + (bias ? bias[m0 + m] : 0.f);
    }
#endif
}
#endif

// Which kernel runs: -1 by batch size, 0 GEMV, 1 FP32 tile, 2 tensor cores (self test).
int g_force = -1;

template <int BITS>
int launch_bits(cudaStream_t s, const uint8_t *W, const float *X, float *Y, const float *bias, int rows, int cols, int batch, int group, int bf16) {
    int path = g_force;
    if (path < 0) {
        static const bool no_tc = getenv("FLLM_CUDA_NO_TC") != nullptr;
        path = batch <= 8 ? 0 : (!fllm_exact_math() && !no_tc && fllm_cc() >= 75) ? 2 : 1;
    }
#if defined(__HIPCC__)
    if (path == 2) path = 1;
#else
    if (path == 2 && fllm_cc() < 75) path = 1;
#endif
    if (path == 0) {
        const dim3 block(128), grid((rows + 3) / 4);
        for (int b0 = 0; b0 < batch; b0 += 4) {
            const int nb = batch - b0 < 4 ? batch - b0 : 4;
            switch (nb) {
                case 1: mlx_gemv_kernel<BITS, 1><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, b0, group, bf16); break;
                case 2: mlx_gemv_kernel<BITS, 2><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, b0, group, bf16); break;
                case 3: mlx_gemv_kernel<BITS, 3><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, b0, group, bf16); break;
                default: mlx_gemv_kernel<BITS, 4><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, b0, group, bf16); break;
            }
        }
    } else if (path == 2) {
#if !defined(__HIPCC__)
        const dim3 grid((rows + BM - 1) / BM, (batch + BN - 1) / BN), block(128);
        mlx_wmma_kernel<BITS><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, group, bf16);
#endif
    } else {
        const dim3 grid((rows + BM - 1) / BM, (batch + BN - 1) / BN), block(256);
        mlx_gemm_kernel<BITS><<<grid, block, 0, s>>>(W, X, Y, bias, rows, cols, batch, group, bf16);
    }
    return fllm_check_launch("linear (MLX affine)");
}

}  // namespace

// `W`: the device image described above, `kind`: the MLXA type id.
int fllm_mlx_launch(cudaStream_t s, const void *W, int kind, const float *X, float *Y, const float *bias, int rows, int cols, int batch) {
    MlxType t;
    if (!mlx_decode((uint32_t)kind, t)) return fllm_set_error("linear: %d is not an MLX affine weight type", kind);
    if (rows <= 0 || cols <= 0 || batch <= 0 || (cols % 32) != 0 || (cols % t.group) != 0)
        return fllm_set_error("linear: MLX affine weight %d x %d (group %d) is not a whole number of groups", rows, cols, t.group);
    const uint8_t *w = static_cast<const uint8_t *>(W);
    const int bf16 = t.bf16 ? 1 : 0;
    switch (t.bits) {
        case 2: return launch_bits<2>(s, w, X, Y, bias, rows, cols, batch, t.group, bf16);
        case 3: return launch_bits<3>(s, w, X, Y, bias, rows, cols, batch, t.group, bf16);
        case 4: return launch_bits<4>(s, w, X, Y, bias, rows, cols, batch, t.group, bf16);
        case 5: return launch_bits<5>(s, w, X, Y, bias, rows, cols, batch, t.group, bf16);
        case 6: return launch_bits<6>(s, w, X, Y, bias, rows, cols, batch, t.group, bf16);
        default: return launch_bits<8>(s, w, X, Y, bias, rows, cols, batch, t.group, bf16);
    }
}

// Self test: force one kernel (0 GEMV, 1 FP32 tile, 2 tensor cores) for every following call; -1 returns to the choice by batch size.
extern "C" int fllm_cuda_mlx_force(int path) {
    if (path < -1 || path > 2) return fllm_set_error("mlx: unknown kernel %d", path);
    g_force = path;
    return 0;
}
