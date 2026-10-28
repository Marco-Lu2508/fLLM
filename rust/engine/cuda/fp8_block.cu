// fLLM CUDA backend: block-wise FP8 weights (GGUF type FP8B, the `weight_scale_inv` format of DeepSeek-V3, Qwen3-FP8, GLM-FP8).
//
//   Y[token][row] = sum_k X[token][k] * dq(W[row][k]) + bias[row],   dq(W[r][k]) = e4m3(code[r][k]) * scale[r >> lbr][k >> lbc]
//
// The device buffer of such a weight is the tensor as the GGUF file holds it (backend/cuda.rs `upload_fp8_block`): rows * cols E4M3 codes
// (row-major), then one FP32 scale per tile (row-major over the tile grid, ceil(cols / 2^lbc) tiles per tile row), then a 16-byte trailer
// the kernels do not read. The weights stay 8-bit in device memory for every kernel in this file:
//
//   * fp8blk_gemv_kernel (decode, 1 .. 16 tokens): one warp per output row, the lanes stream 16 codes each per step, expand them to FP32
//     in registers with the scale of their tile and accumulate in FP32 against the FP32 activations (no activation rounding).
//     Works on every GPU the other kernels do (sm_70 and newer; nothing here needs a tensor core).
//   * fp8blk_tc_kernel (prompts, more than 16 tokens, sm_75 and newer): the tensor-core GEMM of tc_gemm.cu with the weight tile expanded to
//     FP16 in registers (code * scale, rounded once) before it is stored to shared memory. Same rounding as the other W16 tensor-core
//     paths: weights and activations are rounded to FP16, accumulation in FP32.
//   * fllm_fp8_w8a8_gemm (fp8_gemm.cu, sm_89 and newer, 128 x 128 tiles): FP8 x FP8 on the FP8 tensor cores with the activations
//     quantized per token and 128 values (the recipe of vLLM's block-wise FP8 GEMM). Used for prompts by default, for all batch
//     sizes with FLLM_FP8_BLOCK_W8A8=1, never with FLLM_FP8_BLOCK_W8A8=0 or in exact mode.
//
// Scales: a 16-code run lies inside one tile when the tile is at least 16 wide (lbc >= 4, every real checkpoint: 128); narrower tiles take
// the slower per-element path of `fb_decode_run`.
//
// Test level of this file: compiled only (no GPU was available when it was written); the self test group `fp8blk` checks it on a device
// against the CPU kernel's reference (cpu_fp8.rs), and `host_mirror` in backend/cuda/hopper_fp8.rs mirrors the arithmetic on the host.
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "common.cuh"
#include "fllm_cuda_internal.h"
#include "fp8_gemm.h"
#include "mma_util.cuh"

namespace {

// E4M3FN code to float (the formula of decode16 in common.cuh; 0x7f / 0xff, the NaN codes, give +-480 here: they do not occur in checkpoints)
__device__ __forceinline__ float fb_e4m3(uint32_t b) {
    const uint32_t e = (b >> 3) & 15u, m = b & 7u;
    const float v = e == 0 ? (float)m * (1.0f / 512.0f) : __uint_as_float(((e + 120u) << 23) | (m << 20));
    return (b & 0x80u) ? -v : v;
}

// The 16 weights of `run` (codes k .. k + 15 of one row) with their tile scales applied. `vec`: the row start is 16-byte aligned.
// Elements past the end of the row (k + i >= cols) are 0. `srow`: the row of tile scales of this weight row.
__device__ __forceinline__ void fb_decode_run(const uint8_t *__restrict__ wr, int k, int cols, bool vec, const float *__restrict__ srow, int lbc,
                                              float (&w)[16]) {
    uint32_t q[4];
    if (vec && k + 16 <= cols) {
        const uint4 v = __ldg(reinterpret_cast<const uint4 *>(wr + k));
        q[0] = v.x; q[1] = v.y; q[2] = v.z; q[3] = v.w;
    } else {
#pragma unroll
        for (int j = 0; j < 4; j++) {
            uint32_t word = 0;
#pragma unroll
            for (int t = 0; t < 4; t++) {
                const int i = k + 4 * j + t;
                if (i < cols) word |= (uint32_t)__ldg(wr + i) << (8 * t);
            }
            q[j] = word;
        }
    }
#pragma unroll
    for (int i = 0; i < 16; i++) w[i] = fb_e4m3((q[i >> 2] >> (8 * (i & 3))) & 0xffu);
    if (lbc >= 4) {
        const float s = __ldg(srow + (k >> lbc));
#pragma unroll
        for (int i = 0; i < 16; i++) w[i] *= s;
    } else {
#pragma unroll
        for (int i = 0; i < 16; i++)
            if (k + i < cols) w[i] *= __ldg(srow + ((k + i) >> lbc));
    }
}

__device__ __forceinline__ void fb_load16_x(const float *p, int e0, int cols, bool vec, float (&x)[16]) {
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

// ---------------------------------------------------------------------------------------------
// Decode: one warp per output row (blockDim.x is a multiple of 32; a warp whose row is past the end exits as a whole before any shuffle).
// ---------------------------------------------------------------------------------------------
template <int NB>
__global__ void __launch_bounds__(128)
fp8blk_gemv_kernel(const uint8_t *__restrict__ W, const float *__restrict__ S, const float *__restrict__ X, float *__restrict__ Y,
                   const float *__restrict__ bias, int rows, int cols, int batch, int lbr, int lbc, int tiles_c) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + warp;
    if (row >= rows) return;
    const uint8_t *wr = W + (size_t)row * cols;
    const float *srow = S + (size_t)(row >> lbr) * tiles_c;
    const bool vec = (reinterpret_cast<uintptr_t>(wr) & 15) == 0;
    bool xvec[NB];
#pragma unroll
    for (int b = 0; b < NB; b++) xvec[b] = (reinterpret_cast<uintptr_t>(X + (size_t)b * cols) & 15) == 0;
    float acc[NB];
#pragma unroll
    for (int b = 0; b < NB; b++) acc[b] = 0.f;
    const int runs = (cols + 15) >> 4;
    for (int run = lane; run < runs; run += FLLM_WARP) {
        float w[16];
        fb_decode_run(wr, run << 4, cols, vec, srow, lbc, w);
#pragma unroll
        for (int b = 0; b < NB; b++) {
            if (b < batch) {
                float x[16];
                fb_load16_x(X + (size_t)b * cols, run << 4, cols, xvec[b], x);
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
            if (b < batch) Y[(size_t)b * rows + row] = acc[b] + bv;
    }
}

// ---------------------------------------------------------------------------------------------
// Prompts: tensor-core GEMM, 128 weight rows x 128 tokens per block, K step 64, the next tile is fetched (and expanded) while the current one
// is multiplied. Structure of tc_gemm.cu.
// ---------------------------------------------------------------------------------------------
constexpr int FB_BM = 128, FB_BN = 128, FB_BK = 64, FB_WM = 2, FB_WN = 4;
constexpr int FB_LD = FB_BK + 8;
constexpr int FB_NT = FB_WM * FB_WN * 32;
constexpr int FB_MI = FB_BM / FB_WM / 16;
constexpr int FB_NI = FB_BN / FB_WN / 8;
constexpr int FB_APT = FB_BM * FB_BK / 16 / FB_NT;
constexpr int FB_BPT = FB_BN * FB_BK / 16 / FB_NT;
constexpr int FB_RPK = FB_BK / 16;
static_assert(FB_APT >= 1 && FB_BPT >= 1 && FB_NI % 2 == 0 && FB_BK % 16 == 0 && FB_MI >= 1, "tile shape");
constexpr size_t FB_SMEM = (size_t)2 * (FB_BM + FB_BN) * FB_LD * sizeof(__half);

__device__ __forceinline__ uint32_t fb_pack_half2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t *>(&h);
}

__device__ __forceinline__ void fb_store16(__half *dst, const float *v) {
    uint4 a, b;
    a.x = fb_pack_half2(v[0], v[1]);   a.y = fb_pack_half2(v[2], v[3]);
    a.z = fb_pack_half2(v[4], v[5]);   a.w = fb_pack_half2(v[6], v[7]);
    b.x = fb_pack_half2(v[8], v[9]);   b.y = fb_pack_half2(v[10], v[11]);
    b.z = fb_pack_half2(v[12], v[13]); b.w = fb_pack_half2(v[14], v[15]);
    uint4 *d = reinterpret_cast<uint4 *>(dst);
    d[0] = a;
    d[1] = b;
}

// cols is a multiple of 16 (checked by the launcher), so every run is complete.
__global__ void __launch_bounds__(FB_NT)
fp8blk_tc_kernel(const uint8_t *__restrict__ W, const float *__restrict__ S, const float *__restrict__ X, float *__restrict__ Y,
                 const float *__restrict__ bias, int rows, int cols, int batch, int lbr, int lbc, int tiles_c) {
#if FLLM_TC
    extern __shared__ __align__(16) unsigned char fb_smem[];
    __half *As = reinterpret_cast<__half *>(fb_smem);  // [2][FB_BM][FB_LD]
    __half *Bs = As + (size_t)2 * FB_BM * FB_LD;       // [2][FB_BN][FB_LD]
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp % FB_WM, wn = warp / FB_WM;
    const int m0 = blockIdx.x * FB_BM, n0 = blockIdx.y * FB_BN;
    const int ktiles = (cols + FB_BK - 1) / FB_BK;

    float acc[FB_MI][FB_NI][4];
#pragma unroll
    for (int a = 0; a < FB_MI; a++)
#pragma unroll
        for (int b = 0; b < FB_NI; b++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[a][b][c] = 0.f;

    float wv[FB_APT][16], xv[FB_BPT][16];
    auto fetch = [&](int kt) {
#pragma unroll
        for (int i = 0; i < FB_APT; i++) {
            const int run = tid + i * FB_NT;
            const int m = m0 + run / FB_RPK;
            const int k = kt * FB_BK + (run % FB_RPK) * 16;
            if (m < rows && k < cols) {
                const uint8_t *wr = W + (size_t)m * cols;
                fb_decode_run(wr, k, cols, (reinterpret_cast<uintptr_t>(wr) & 15) == 0, S + (size_t)(m >> lbr) * tiles_c, lbc, wv[i]);
            } else {
#pragma unroll
                for (int j = 0; j < 16; j++) wv[i][j] = 0.f;
            }
        }
#pragma unroll
        for (int i = 0; i < FB_BPT; i++) {
            const int run = tid + i * FB_NT;
            const int n = n0 + run / FB_RPK;
            const int k = kt * FB_BK + (run % FB_RPK) * 16;
            if (n < batch && k < cols) {
                fb_load16_x(X + (size_t)n * cols, k, cols, true, xv[i]);
            } else {
#pragma unroll
                for (int j = 0; j < 16; j++) xv[i][j] = 0.f;
            }
        }
    };
    auto stash = [&](int buf) {
#pragma unroll
        for (int i = 0; i < FB_APT; i++) {
            const int run = tid + i * FB_NT;
            fb_store16(As + ((size_t)buf * FB_BM + run / FB_RPK) * FB_LD + (run % FB_RPK) * 16, wv[i]);
        }
#pragma unroll
        for (int i = 0; i < FB_BPT; i++) {
            const int run = tid + i * FB_NT;
            fb_store16(Bs + ((size_t)buf * FB_BN + run / FB_RPK) * FB_LD + (run % FB_RPK) * 16, xv[i]);
        }
    };

    fetch(0);
    stash(0);
    __syncthreads();
    for (int kt = 0; kt < ktiles; kt++) {
        const int buf = kt & 1;
        if (kt + 1 < ktiles) fetch(kt + 1);
        const __half *a_base = As + (size_t)buf * FB_BM * FB_LD;
        const __half *b_base = Bs + (size_t)buf * FB_BN * FB_LD;
#pragma unroll
        for (int kk = 0; kk < FB_BK; kk += 16) {
            uint32_t af[FB_MI][4], bf[FB_NI][2];
#pragma unroll
            for (int mi = 0; mi < FB_MI; mi++)
                ldmatrix4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                          a_base + (size_t)(wm * (FB_BM / FB_WM) + mi * 16 + (lane & 15)) * FB_LD + kk + (lane >> 4) * 8);
#pragma unroll
            for (int np = 0; np < FB_NI / 2; np++)
                ldmatrix4(bf[np * 2][0], bf[np * 2][1], bf[np * 2 + 1][0], bf[np * 2 + 1][1],
                          b_base + (size_t)(wn * (FB_BN / FB_WN) + np * 16 + (lane & 7) + ((lane >> 4) << 3)) * FB_LD + kk +
                              ((lane >> 3) & 1) * 8);
#pragma unroll
            for (int mi = 0; mi < FB_MI; mi++)
#pragma unroll
                for (int ni = 0; ni < FB_NI; ni++) mma16816(acc[mi][ni], af[mi], bf[ni][0], bf[ni][1]);
        }
        if (kt + 1 < ktiles) stash(buf ^ 1);
        __syncthreads();
    }

    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int mi = 0; mi < FB_MI; mi++)
#pragma unroll
        for (int ni = 0; ni < FB_NI; ni++)
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int m = m0 + wm * (FB_BM / FB_WM) + mi * 16 + g + (c >> 1) * 8;
                const int n = n0 + wn * (FB_BN / FB_WN) + ni * 8 + 2 * t + (c & 1);
                if (m < rows && n < batch) Y[(size_t)n * rows + m] = acc[mi][ni][c] + (bias ? bias[m] : 0.f);
            }
#endif
}

template <int NB>
int launch_gemv(cudaStream_t s, const uint8_t *w, const float *scales, const float *x, float *y, const float *bias, int rows, int cols, int batch,
                int lbr, int lbc, int tiles_c) {
    const dim3 block(128), grid((rows + 3) / 4);
    fp8blk_gemv_kernel<NB><<<grid, block, 0, s>>>(w, scales, x, y, bias, rows, cols, batch, lbr, lbc, tiles_c);
    return fllm_check_launch("fp8 block gemv");
}

// 1 when the tensor-core kernel ran (or failed to launch: *rc), 0 when the caller continues with the GEMV
int try_tc(cudaStream_t s, const uint8_t *w, const float *scales, const float *x, float *y, const float *bias, int rows, int cols, int batch, int lbr,
           int lbc, int tiles_c, int *rc) {
    static const bool disabled = getenv("FLLM_CUDA_NO_TC") != nullptr;
    if (disabled || batch <= 16 || fllm_exact_math() || fllm_cc() < 75 || (cols & 15) != 0 || (reinterpret_cast<uintptr_t>(x) & 15) != 0) return 0;
    if (FB_SMEM > (size_t)fllm_smem_limit()) return 0;
    static std::atomic<unsigned> configured{0};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 0 || dev >= 32) dev = 0;
    if (!(configured.load() & (1u << dev))) {
        cudaFuncSetAttribute(fp8blk_tc_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)FB_SMEM);
        configured.fetch_or(1u << dev);
    }
    const dim3 grid((rows + FB_BM - 1) / FB_BM, (batch + FB_BN - 1) / FB_BN), block(FB_NT);
    fp8blk_tc_kernel<<<grid, block, FB_SMEM, s>>>(w, scales, x, y, bias, rows, cols, batch, lbr, lbc, tiles_c);
    *rc = fllm_check_launch("fp8 block tensor-core gemm");
    return 1;
}

// FP8 tensor cores (W8A8) for 128 x 128 tiles: FLLM_FP8_BLOCK_W8A8 = auto (prompts of more than 16 tokens) | 1 (every batch) | 0 (never); the
// self test overrides it through fllm_fp8_gemm_set_mode (0 never, 1 and 2 always).
bool w8a8_wanted(int batch, int cols, int lbr, int lbc) {
    if (lbr != 7 || lbc != 7 || (cols & 127) != 0 || fllm_cc() < 89 || fllm_exact_math()) return false;
    const int forced = fllm_fp8_gemm_forced_mode();
    if (forced == 0) return false;
    if (forced > 0) return true;
    static const int env = [] {
        const char *v = getenv("FLLM_FP8_BLOCK_W8A8");
        if (v == nullptr || *v == 0 || strcmp(v, "auto") == 0) return -1;
        return (strcmp(v, "0") == 0 || strcmp(v, "off") == 0 || strcmp(v, "false") == 0) ? 0 : 1;
    }();
    if (env >= 0) return env == 1;
    return batch > 16;
}

}  // namespace

// 0 when the product ran, > 0 on error (fllm_set_error). `w` is the device buffer of the weight (codes, then the FP32 tile scales).
int fllm_fp8_block_gemm(cudaStream_t s, const void *w, const float *x, float *y, const float *bias, int rows, int cols, int batch, int lbr, int lbc) {
    if (rows <= 0 || cols <= 0 || batch <= 0 || lbr < 0 || lbr > 15 || lbc < 0 || lbc > 15)
        return fllm_set_error("FP8 block GEMM: invalid shape %d x %d, batch %d, tiles 2^%d x 2^%d", rows, cols, batch, lbr, lbc);
    const size_t count = (size_t)rows * (size_t)cols;
    if ((count & 3) != 0) return fllm_set_error("FP8 block GEMM: %d x %d codes leave the scale table unaligned", rows, cols);
    const uint8_t *codes = static_cast<const uint8_t *>(w);
    const float *scales = reinterpret_cast<const float *>(codes + count);
    if ((reinterpret_cast<uintptr_t>(scales) & 3) != 0) return fllm_set_error("FP8 block GEMM: unaligned weight buffer");
    const int tiles_c = (int)(((size_t)cols + ((size_t)1 << lbc) - 1) >> lbc);
    if (w8a8_wanted(batch, cols, lbr, lbc)) {
        const int rc = fllm_fp8_w8a8_gemm(s, w, scales, 1, x, y, bias, rows, cols, batch, 2, 0.f);
        if (rc >= 0) return rc;  // -1: alignment or device not covered, continue with the kernels below
    }
    int rc = 0;
    if (try_tc(s, codes, scales, x, y, bias, rows, cols, batch, lbr, lbc, tiles_c, &rc)) return rc;
    for (int b0 = 0; b0 < batch; b0 += 16) {
        const int n = batch - b0 < 16 ? batch - b0 : 16;
        const float *xb = x + (size_t)b0 * cols;
        float *yb = y + (size_t)b0 * rows;
        int e;
        if (n == 1) e = launch_gemv<1>(s, codes, scales, xb, yb, bias, rows, cols, n, lbr, lbc, tiles_c);
        else if (n <= 2) e = launch_gemv<2>(s, codes, scales, xb, yb, bias, rows, cols, n, lbr, lbc, tiles_c);
        else if (n <= 4) e = launch_gemv<4>(s, codes, scales, xb, yb, bias, rows, cols, n, lbr, lbc, tiles_c);
        else if (n <= 8) e = launch_gemv<8>(s, codes, scales, xb, yb, bias, rows, cols, n, lbr, lbc, tiles_c);
        else e = launch_gemv<16>(s, codes, scales, xb, yb, bias, rows, cols, n, lbr, lbc, tiles_c);
        if (e) return e;
    }
    return 0;
}
