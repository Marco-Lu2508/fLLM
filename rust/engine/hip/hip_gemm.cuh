// fLLM HIP backend: matrix-core GEMM kernel shared by the quantized-weight GEMMs (hip_gemm.cu) and the Q4Z GEMM (hip_q4z.cu).
//
//   Y[token][row] = sum_k X[token][k] * W[row][k]  (+ bias[row])
//
// Same scheme as the tensor-core GEMM of ../cuda/tc_gemm.cu: a block of 256 threads owns a 64 (weight rows) x 64 (tokens) output tile and
// walks K in steps of 64; every thread decodes one run of 16 weights and loads one run of 16 activations per step into registers, the
// runs are converted to FP16 into LDS (double buffered: the next step is fetched while the current one is multiplied) and the waves
// multiply their sub-tiles with 16x16x16 matrix-core steps (hip_mma.cuh). The decoder `Dec` provides
//
//     __device__ void decode(int row, int run, float *w16) const;   // weights row*, k = 16*run .. 16*run+15; 0 past the end of the row
//
// so one kernel serves every GGUF kind (KindDecoder) and the group-quantized Q4Z weights.
//
// Waves: CDNA 4 waves in a 2x2 grid, 2x2 sub-tiles of 16x16 each; RDNA 8 waves in a 2x4 grid, 2x1 sub-tiles each. Rounding the operands
// to FP16 costs about 1e-3 relative error, like tc_gemm.cu. Level of verification: L0 (compiles; never run on hardware).
#pragma once
#include <cstdlib>

#include "../cuda/common.cuh"
#include "fllm_hip.h"
#include "hip_mma.cuh"

namespace fllm_hip {

constexpr int G_BM = 64, G_BK = 64;   // the token tile BN is a template parameter of the kernel: 64, or 128 for long prompts (halves the weight decoding per token)
constexpr int G_LD = G_BK + 8;          // halves per LDS row (144 bytes: 16-byte aligned rows, spread banks)
constexpr int G_NT = 256;
constexpr int G_RPK = G_BK / 16;        // runs of 16 per tile row
static_assert(G_NT == G_BM * G_RPK, "one weight run per thread and step");

#if defined(FLLM_HIP_MMA)
constexpr int G_WGM = 2;                                  // waves along the weight rows
constexpr int G_WGN = G_NT / MMA_WAVE / G_WGM;            // waves along the tokens
constexpr int G_TM = G_BM / G_WGM / 16;                   // 16-row sub-tiles per wave
static_assert(G_TM >= 1 && G_WGM * G_TM * 16 == G_BM, "wave grid");
#endif

// One call of the GEMM kernel: Y[token][row] for `batch` tokens.
struct GemmArgs {
    const float *X;     // activations [batch][cols], 16-byte aligned
    float *Y;           // result [batch][rows]
    const float *bias;  // rows floats or null
    const int *perm;    // act-order permutation (x'[k] = x[perm[k]]) or null
    int rows, cols, batch;
    int mode;           // bit 0: Y += scale * (result + bias); bit 1: the result is FP16 (batch x rows halves), no bias
    float scale;        // multiplier of the accumulate mode
};

// result of one output element in the mode of GemmArgs
__device__ __forceinline__ void store_result(const GemmArgs &a, int n, int m, float acc) {
    const size_t idx = (size_t)n * a.rows + m;
    if (a.mode & 2) {
        reinterpret_cast<uint16_t *>(a.Y)[idx] = f2h_bits(acc);
        return;
    }
    float v = acc + (a.bias ? a.bias[m] : 0.f);
    if (a.mode & 1) v = a.Y[idx] + a.scale * v;
    a.Y[idx] = v;
}

__device__ __forceinline__ uint32_t pack_half2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t *>(&h);
}

// 16 activations (zero past `cols`); the row is 16-byte aligned (the caller checks X and cols % 4)
__device__ __forceinline__ void load_x16(const float *p, int e0, int cols, float *x) {
    if (e0 + 16 <= cols) {
        const float4 *q = reinterpret_cast<const float4 *>(p + e0);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float4 v = q[i];
            x[4 * i] = v.x; x[4 * i + 1] = v.y; x[4 * i + 2] = v.z; x[4 * i + 3] = v.w;
        }
    } else {
#pragma unroll
        for (int i = 0; i < 16; i++) x[i] = e0 + i < cols ? p[e0 + i] : 0.f;
    }
}

// the same through a permutation of the input index
__device__ __forceinline__ void load_x16_perm(const float *p, const int *perm, int e0, int cols, float *x) {
#pragma unroll
    for (int i = 0; i < 16; i++) x[i] = e0 + i < cols ? p[perm[e0 + i]] : 0.f;
}

__device__ __forceinline__ void store_h16(__half *dst, const float *v) {
    uint4 a, b;
    a.x = pack_half2(v[0], v[1]);   a.y = pack_half2(v[2], v[3]);
    a.z = pack_half2(v[4], v[5]);   a.w = pack_half2(v[6], v[7]);
    b.x = pack_half2(v[8], v[9]);   b.y = pack_half2(v[10], v[11]);
    b.z = pack_half2(v[12], v[13]); b.w = pack_half2(v[14], v[15]);
    uint4 *d = reinterpret_cast<uint4 *>(dst);
    d[0] = a;
    d[1] = b;
}

// Decoder of the GGUF kinds of common.cuh (decode16).
template <int KIND>
struct KindDecoder {
    const uint8_t *W;
    size_t row_bytes;
    int cols;
    __device__ __forceinline__ void decode(int row, int run, float *w) const { decode16<KIND>(W + (size_t)row * row_bytes, run, cols, w); }
};

template <class Dec, int G_BN>
__global__ void __launch_bounds__(G_NT)
mma_gemm_kernel(Dec dec, GemmArgs a) {
#if defined(FLLM_HIP_MMA)
    constexpr int G_TN = G_BN / G_WGN / 16;               // 16-token sub-tiles per wave
    constexpr int G_XR = G_BN / G_BM;                     // activation runs per thread and step
    static_assert(G_TN >= 1 && G_WGN * G_TN * 16 == G_BN && G_XR * G_BM == G_BN, "wave grid");
    __shared__ __align__(16) __half As[2][G_BM * G_LD];  // decoded weights  [row][k]
    __shared__ __align__(16) __half Bs[2][G_BN * G_LD];  // activations      [token][k]
    const int rows = a.rows, cols = a.cols, batch = a.batch;
    const int tid = threadIdx.x, wave = tid / MMA_WAVE, lane = tid % MMA_WAVE;
    const int wm = wave % G_WGM, wn = wave / G_WGM;
    const int m0 = blockIdx.x * G_BM, n0 = blockIdx.y * G_BN;
    const int ktiles = (cols + G_BK - 1) / G_BK;
    const int ld_row = tid / G_RPK, ld_q = tid % G_RPK;

    MmaAcc acc[G_TM][G_TN];
#pragma unroll
    for (int i = 0; i < G_TM; i++)
#pragma unroll
        for (int j = 0; j < G_TN; j++)
#pragma unroll
            for (int e = 0; e < MMA_ACC; e++) acc[i][j][e] = 0.f;

    float wv[16], xv[G_XR][16];
    auto fetch = [&](int kt) {
        const int k = kt * G_BK + ld_q * 16;
        const int m = m0 + ld_row;
        if (m < rows) {
            dec.decode(m, k >> 4, wv);
        } else {
#pragma unroll
            for (int j = 0; j < 16; j++) wv[j] = 0.f;
        }
#pragma unroll
        for (int r = 0; r < G_XR; r++) {
            const int n = n0 + ld_row + r * G_BM;
            if (n < batch) {
                if (a.perm) load_x16_perm(a.X + (size_t)n * cols, a.perm, k, cols, xv[r]);
                else load_x16(a.X + (size_t)n * cols, k, cols, xv[r]);
            } else {
#pragma unroll
                for (int j = 0; j < 16; j++) xv[r][j] = 0.f;
            }
        }
    };
    auto stash = [&](int buf) {
        store_h16(&As[buf][ld_row * G_LD + ld_q * 16], wv);
#pragma unroll
        for (int r = 0; r < G_XR; r++) store_h16(&Bs[buf][(ld_row + r * G_BM) * G_LD + ld_q * 16], xv[r]);
    };

    fetch(0);
    stash(0);
    __syncthreads();
    const int frag_row = lane & 15, k0 = mma_k0(lane);
    for (int kt = 0; kt < ktiles; kt++) {
        const int buf = kt & 1;
        if (kt + 1 < ktiles) fetch(kt + 1);
        const __half *as = As[buf], *bs = Bs[buf];
#pragma unroll
        for (int kk = 0; kk < G_BK; kk += 16) {
            MmaFrag af[G_TM], bf[G_TN];
#pragma unroll
            for (int i = 0; i < G_TM; i++) af[i] = mma_load(as + (size_t)(wm * G_TM * 16 + i * 16 + frag_row) * G_LD + kk + k0);
#pragma unroll
            for (int j = 0; j < G_TN; j++) bf[j] = mma_load(bs + (size_t)(wn * G_TN * 16 + j * 16 + frag_row) * G_LD + kk + k0);
#pragma unroll
            for (int i = 0; i < G_TM; i++)
#pragma unroll
                for (int j = 0; j < G_TN; j++) acc[i][j] = mma16(af[i], bf[j], acc[i][j]);
        }
        if (kt + 1 < ktiles) stash(buf ^ 1);
        __syncthreads();
    }

    // accumulator element e of this lane: weight row m0 + ... + mma_row(lane, e), token n0 + ... + lane % 16
#pragma unroll
    for (int i = 0; i < G_TM; i++)
#pragma unroll
        for (int j = 0; j < G_TN; j++) {
            const int n = n0 + wn * G_TN * 16 + j * 16 + frag_row;
#pragma unroll
            for (int e = 0; e < MMA_ACC; e++) {
                const int m = m0 + wm * G_TM * 16 + i * 16 + mma_row(lane, e);
                if (m < rows && n < batch) store_result(a, n, m, acc[i][j][e]);
            }
        }
#endif
}

// Launch the kernel for `dec` on a rows x cols weight and `batch` tokens.
template <class Dec>
static inline void launch_mma_gemm(cudaStream_t s, const Dec &dec, const GemmArgs &args) {
    // a long prompt takes the wide token tile: every decoded weight feeds twice the products
    static const bool wide = !std::getenv("FLLM_HIP_NARROW_TILE");
    if (wide && args.batch > 96) {
        const dim3 grid((args.rows + G_BM - 1) / G_BM, (args.batch + 127) / 128), block(G_NT);
        mma_gemm_kernel<Dec, 128><<<grid, block, 0, s>>>(dec, args);
    } else {
        const dim3 grid((args.rows + G_BM - 1) / G_BM, (args.batch + 63) / 64), block(G_NT);
        mma_gemm_kernel<Dec, 64><<<grid, block, 0, s>>>(dec, args);
    }
}

}  // namespace fllm_hip
