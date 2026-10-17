// fLLM: GEMV for Marlin-layout 4-bit weights, M <= 4 rows (decode).
//
// The Marlin GEMM of vLLM (marlin_vllm.cu) is built for tiles of 16..64 rows: even the 16x8 variant launches `sms` persistent blocks, runs a
// 4-stage cp.async pipeline per block, reduces the split-K partials through a lock workspace and the FP32 scratch `C_tmp`, and the
// activations are staged in shared memory. For one to four rows of a matrix of a few MB (an attention projection of a 3B model) that
// machinery costs more than the weight traffic: a call takes about 12 us where 5.5 MB at the memory bandwidth need 3 us.
//
// This kernel reads the Marlin tiles directly, with one 128-bit load per lane and tile, dequantizes in registers and accumulates in FP32:
//   * grid (N / 64, splits); a block of 4 warps; warp `w` takes the "units" w, w + 4, ... of the block's K range; a unit is U tiles of 16 rows
//     of K that lie in one quantization group, so that scale and zero point are read once per unit;
//   * per unit and lane: sum_k x[k] * q and sum_k x[k] over the lane's 4 K rows of every tile, then s * (sum x*q - z * sum x) per column;
//   * the four lanes that share columns and the four warps are reduced through shuffles and shared memory; the K splits of the grid
//     through an FP32 scratch and a ticket counter (the last block of a column tile adds the partials in a fixed order and writes the
//     result, so the output is deterministic and the kernel is safe inside a CUDA graph).
// The layout of the weights is the one gptq_marlin_repack_kernel / awq_marlin_repack_kernel write (see marlin_vllm_gptq_repack.cu):
// per 16 x 64 tile 128 words, word `th * 4 + w` holds the eight codes of lane `th` of warp `w`: with r = (th % 4) * 2 and c = w * 16 + th / 4
// the nibbles 0..7 are (k, n) = (r, c) (r+8, c) (r, c+8) (r+8, c+8) (r+1, c) (r+9, c) (r+1, c+8) (r+9, c+8). Scales are permuted by
// marlin_permute_scales (64 columns: position 8 * (c % 8) + c / 8), the zero points likewise and packed with the order 0 2 4 6 1 3 5 7
// (backend/q4z.rs builds both).
//
// fllm_marlin_gemv_bench times this kernel against the Marlin GEMM on synthetic weights and compares the results: the same bytes are a valid
// weight matrix for both, so no repack is needed. The dispatcher in marlin_vllm.cu (fllm_marlin_small_m) uses the same two routines to decide
// per shape in FLLM_MARLIN_SMALL_M=auto.
//
// Original code of fLLM (Apache-2.0); the layout facts come from the Marlin kernels of IST-DASLab/marlin and vLLM (Apache-2.0).
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <map>
#include <mutex>
#include <tuple>

#include "../pdl.cuh"

namespace fllm_gemv {

template <typename T>
struct Num;
template <>
struct Num<__half> {
    using T2 = __half2;
    static __device__ __forceinline__ float2 to2(uint32_t bits) {
        __half2 h;
        memcpy(&h, &bits, 4);
        return __half22float2(h);
    }
    static __device__ __forceinline__ float one(const __half &v) { return __half2float(v); }
    static __device__ __forceinline__ __half make(float v) { return __float2half_rn(v); }
};
template <>
struct Num<__nv_bfloat16> {
    using T2 = __nv_bfloat162;
    static __device__ __forceinline__ float2 to2(uint32_t bits) {
        __nv_bfloat162 h;
        memcpy(&h, &bits, 4);
        return __bfloat1622float2(h);
    }
    static __device__ __forceinline__ float one(const __nv_bfloat16 &v) { return __bfloat162float(v); }
    static __device__ __forceinline__ __nv_bfloat16 make(float v) { return __float2bfloat16_rn(v); }
};

constexpr int kWarps = 4;

// M rows (1..4), U tiles per unit (2, 4), ZP: AWQ zero points (else the symmetric zero point 8 of GPTQ)
template <typename T, int M, int U, bool ZP>
__global__ void __launch_bounds__(32 * kWarps)
gemv_kernel(const T *__restrict__ A, const uint4 *__restrict__ B, T *__restrict__ C, float *__restrict__ part, int *__restrict__ tickets,
            const T *__restrict__ S, const uint32_t *__restrict__ Z, int n, int k, int lda, int gtiles, int splits) {
    fllm_pdl_wait();  // (a no-op unless launched with FLLM_PDL=1 on sm_90+; the activations A come from the kernel before)
    __shared__ float red[kWarps][M][64];
    __shared__ int is_last;
    const int nt = blockIdx.x, n_tiles = gridDim.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = (lane & 3) * 2, c8 = lane >> 2;
    const int units = (k >> 4) / U;
    const int u0 = (int)((long long)blockIdx.y * units / splits), u1 = (int)((long long)(blockIdx.y + 1) * units / splits);

    float total[M][4][2];
#pragma unroll
    for (int m = 0; m < M; m++)
#pragma unroll
        for (int w = 0; w < 4; w++) total[m][w][0] = total[m][w][1] = 0.f;

    for (int u = u0 + warp; u < u1; u += kWarps) {
        const int kt0 = u * U;
        const int g = kt0 / gtiles;
        // independent loads first: the weights of U tiles, the scales and zero points of the group
        uint4 wq[U];
#pragma unroll
        for (int i = 0; i < U; i++) wq[i] = __ldcs(B + ((size_t)(kt0 + i) * n_tiles + nt) * 32 + lane);
        const uint4 sv = *reinterpret_cast<const uint4 *>(S + (size_t)g * n + (size_t)nt * 64 + 8 * c8);
        uint32_t zw = 0;
        if (ZP) zw = Z[(size_t)g * (n >> 3) + (size_t)nt * 8 + c8];

        float accg[M][4][2];
        float xs[M];
#pragma unroll
        for (int m = 0; m < M; m++) {
            xs[m] = 0.f;
#pragma unroll
            for (int w = 0; w < 4; w++) accg[m][w][0] = accg[m][w][1] = 0.f;
        }
#pragma unroll
        for (int i = 0; i < U; i++) {
            const int kk = (kt0 + i) * 16 + r;
#pragma unroll
            for (int m = 0; m < M; m++) {
                const float2 xa = Num<T>::to2(*reinterpret_cast<const uint32_t *>(A + (size_t)m * lda + kk));
                const float2 xb = Num<T>::to2(*reinterpret_cast<const uint32_t *>(A + (size_t)m * lda + kk + 8));
                xs[m] += (xa.x + xa.y) + (xb.x + xb.y);
                const uint32_t words[4] = {wq[i].x, wq[i].y, wq[i].z, wq[i].w};
#pragma unroll
                for (int w = 0; w < 4; w++) {
                    const uint32_t v = words[w];
                    const float q0 = (float)(v & 15u), q1 = (float)((v >> 4) & 15u), q2 = (float)((v >> 8) & 15u), q3 = (float)((v >> 12) & 15u);
                    const float q4 = (float)((v >> 16) & 15u), q5 = (float)((v >> 20) & 15u), q6 = (float)((v >> 24) & 15u), q7 = v >> 28;
                    accg[m][w][0] = fmaf(q0, xa.x, fmaf(q1, xb.x, fmaf(q4, xa.y, fmaf(q5, xb.y, accg[m][w][0]))));
                    accg[m][w][1] = fmaf(q2, xa.x, fmaf(q3, xb.x, fmaf(q6, xa.y, fmaf(q7, xb.y, accg[m][w][1]))));
                }
            }
        }
        const uint32_t sw[4] = {sv.x, sv.y, sv.z, sv.w};
#pragma unroll
        for (int w = 0; w < 4; w++) {
            const float2 s = Num<T>::to2(sw[w]);
            const float z0 = ZP ? (float)((zw >> (4 * w)) & 15u) : 8.f;
            const float z1 = ZP ? (float)((zw >> (4 * (w + 4))) & 15u) : 8.f;
#pragma unroll
            for (int m = 0; m < M; m++) {
                total[m][w][0] = fmaf(s.x, accg[m][w][0] - z0 * xs[m], total[m][w][0]);
                total[m][w][1] = fmaf(s.y, accg[m][w][1] - z1 * xs[m], total[m][w][1]);
            }
        }
    }

    // the four lanes of a column group (lane % 4) hold different K rows of the same columns
#pragma unroll
    for (int m = 0; m < M; m++)
#pragma unroll
        for (int w = 0; w < 4; w++)
#pragma unroll
            for (int e = 0; e < 2; e++) {
                float v = total[m][w][e];
                v += __shfl_xor_sync(0xffffffffu, v, 1);
                v += __shfl_xor_sync(0xffffffffu, v, 2);
                total[m][w][e] = v;
            }
    if ((lane & 3) == 0) {
#pragma unroll
        for (int m = 0; m < M; m++)
#pragma unroll
            for (int w = 0; w < 4; w++) {
                red[warp][m][w * 16 + c8] = total[m][w][0];
                red[warp][m][w * 16 + c8 + 8] = total[m][w][1];
            }
    }
    __syncthreads();
    const int t = threadIdx.x;
    const int cols = nt * 64;
    if (splits == 1) {
        for (int idx = t; idx < M * 64; idx += 32 * kWarps) {
            const int m = idx >> 6, c = idx & 63;
            float v = 0.f;
#pragma unroll
            for (int w = 0; w < kWarps; w++) v += red[w][m][c];
            C[(size_t)m * n + cols + c] = Num<T>::make(v);
        }
        return;
    }
    for (int idx = t; idx < M * 64; idx += 32 * kWarps) {
        const int m = idx >> 6, c = idx & 63;
        float v = 0.f;
#pragma unroll
        for (int w = 0; w < kWarps; w++) v += red[w][m][c];
        part[((size_t)blockIdx.y * M + m) * n + cols + c] = v;
    }
    __threadfence();
    __syncthreads();
    if (t == 0) is_last = atomicAdd(&tickets[nt], 1) == splits - 1;
    __syncthreads();
    if (!is_last) return;
    __threadfence();
    for (int idx = t; idx < M * 64; idx += 32 * kWarps) {
        const int m = idx >> 6, c = idx & 63;
        float v = 0.f;
        for (int s = 0; s < splits; s++) v += __ldcg(part + ((size_t)s * M + m) * n + cols + c);
        C[(size_t)m * n + cols + c] = Num<T>::make(v);
    }
    if (t == 0) tickets[nt] = 0;
}

template <typename T, int M, int U, bool ZP>
static void launch(const void *A, const void *B, void *C, float *part, int *tickets, const void *S, const void *Z, int n, int k, int lda, int gtiles,
                   int splits, cudaStream_t s) {
    dim3 grid(n / 64, splits);
    fllm_launch_pdl(gemv_kernel<T, M, U, ZP>, grid, dim3(32 * kWarps), 0, s, static_cast<const T *>(A), static_cast<const uint4 *>(B), static_cast<T *>(C), part,
                    tickets, static_cast<const T *>(S), static_cast<const uint32_t *>(Z), n, k, lda, gtiles, splits);
}

template <typename T, int U, bool ZP>
static void launch_m(int m, const void *A, const void *B, void *C, float *part, int *tickets, const void *S, const void *Z, int n, int k, int lda,
                     int gtiles, int splits, cudaStream_t s) {
    switch (m) {
        case 1: launch<T, 1, U, ZP>(A, B, C, part, tickets, S, Z, n, k, lda, gtiles, splits, s); break;
        case 2: launch<T, 2, U, ZP>(A, B, C, part, tickets, S, Z, n, k, lda, gtiles, splits, s); break;
        case 3: launch<T, 3, U, ZP>(A, B, C, part, tickets, S, Z, n, k, lda, gtiles, splits, s); break;
        default: launch<T, 4, U, ZP>(A, B, C, part, tickets, S, Z, n, k, lda, gtiles, splits, s); break;
    }
}

template <typename T, bool ZP>
static void launch_u(int u, int m, const void *A, const void *B, void *C, float *part, int *tickets, const void *S, const void *Z, int n, int k,
                     int lda, int gtiles, int splits, cudaStream_t s) {
    if (u == 2) launch_m<T, 2, ZP>(m, A, B, C, part, tickets, S, Z, n, k, lda, gtiles, splits, s);
    else launch_m<T, 4, ZP>(m, A, B, C, part, tickets, S, Z, n, k, lda, gtiles, splits, s);
}

}  // namespace fllm_gemv

// Is this product one the kernel handles? kind: 0 AWQ (zero points), 1 GPTQ symmetric. group: 32, 64 or 128.
extern "C" int fllm_marlin_gemv_supported(int m, int n, int k, int kind, int group) {
    if (m < 1 || m > 4 || (kind != 0 && kind != 1)) return 0;
    if (group != 32 && group != 64 && group != 128) return 0;
    if (n % 64 != 0 || k % group != 0) return 0;
    const int u = group == 32 ? 2 : 4;
    return k % (16 * u) == 0 ? 1 : 0;
}

// Number of K splits for a shape: enough blocks to cover the device twice over, but at least one unit per warp.
static int gemv_splits(int n, int k, int group, int sms) {
    const int u = group == 32 ? 2 : 4;
    const int units = (k / 16) / u, n_tiles = n / 64;
    const char *forced = getenv("FLLM_MARLIN_GEMV_SPLITS");
    int want = (2 * sms + n_tiles - 1) / n_tiles;
    if (forced && atoi(forced) > 0) want = atoi(forced);
    int max_splits = units / fllm_gemv::kWarps;
    if (max_splits < 1) max_splits = 1;
    return want < 1 ? 1 : (want > max_splits ? max_splits : want);
}

// Launches the GEMV. A: [m, k] activations (row stride lda, half or bf16 per `bf16`), B: Marlin-tiled codes, C: [m, n], scales [k / group, n] and
// zero points [k / group, n / 8] in the Marlin order, `part` (FP32, part_bytes) and `tickets` (n / 64 ints, zero before and after every call) for
// the split-K reduction. Returns 0 when launched, -1 when the shape or an alignment is not supported.
extern "C" int fllm_marlin_gemv(const void *A, const void *B, void *C, void *part, size_t part_bytes, void *tickets, const void *scales, const void *zeros,
                                int m, int n, int k, int lda, int kind, int group, int bf16, int sms, cudaStream_t stream) {
    if (!fllm_marlin_gemv_supported(m, n, k, kind, group)) return -1;
    if ((reinterpret_cast<uintptr_t>(A) & 3) || (lda & 1) || (reinterpret_cast<uintptr_t>(B) & 15) || (reinterpret_cast<uintptr_t>(scales) & 15) ||
        (kind == 0 && (zeros == nullptr || (reinterpret_cast<uintptr_t>(zeros) & 3))))
        return -1;
    int splits = gemv_splits(n, k, group, sms);
    while (splits > 1 && (part == nullptr || tickets == nullptr || (size_t)splits * m * n * sizeof(float) > part_bytes)) splits--;
    float *p = static_cast<float *>(part);
    int *t = static_cast<int *>(tickets);
    const int gtiles = group / 16;
    if (kind == 0) {
        if (bf16) fllm_gemv::launch_u<__nv_bfloat16, true>(group == 32 ? 2 : 4, m, A, B, C, p, t, scales, zeros, n, k, lda, gtiles, splits, stream);
        else fllm_gemv::launch_u<__half, true>(group == 32 ? 2 : 4, m, A, B, C, p, t, scales, zeros, n, k, lda, gtiles, splits, stream);
    } else {
        if (bf16) fllm_gemv::launch_u<__nv_bfloat16, false>(group == 32 ? 2 : 4, m, A, B, C, p, t, scales, zeros, n, k, lda, gtiles, splits, stream);
        else fllm_gemv::launch_u<__half, false>(group == 32 ? 2 : 4, m, A, B, C, p, t, scales, zeros, n, k, lda, gtiles, splits, stream);
    }
    return cudaPeekAtLastError() == cudaSuccess ? 0 : -2;
}

// ---- comparison and timing --------------------------------------------------------------------------------------------------------------------

namespace {
template <typename T>
__global__ void diff_kernel(const T *a, const T *b, size_t count, float *out /* [max |a - b|, max |a|] as ordered ints */) {
    float d = 0.f, r = 0.f;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x) {
        const float x = fllm_gemv::Num<T>::one(a[i]), y = fllm_gemv::Num<T>::one(b[i]);
        const float dd = fabsf(x - y);
        d = fmaxf(d, isnan(dd) ? 1e30f : dd);
        r = fmaxf(r, fabsf(x));
    }
    atomicMax(reinterpret_cast<int *>(out), __float_as_int(d));
    atomicMax(reinterpret_cast<int *>(out) + 1, __float_as_int(r));
}
__global__ void fill_random_kernel(uint32_t *p, size_t words, uint32_t seed) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < words; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t)i * 2654435761u + seed;
        x ^= x >> 15;
        x *= 2246822519u;
        x ^= x >> 13;
        x *= 3266489917u;
        x ^= x >> 16;
        p[i] = x;
    }
}
template <typename T>
__global__ void fill_values_kernel(T *p, size_t count, uint32_t seed, float amp, float bias) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = (uint32_t)i * 2654435761u + seed;
        x ^= x >> 15;
        x *= 2246822519u;
        x ^= x >> 13;
        x *= 3266489917u;
        x ^= x >> 16;
        p[i] = fllm_gemv::Num<T>::make(bias + amp * ((float)(x & 0xffff) / 65535.f - 0.5f));
    }
}
}  // namespace

// max |a - b| and max |a| of two FP16 (or BF16) arrays on the device (synchronizes `s`); host_out[0], host_out[1]
extern "C" int fllm_marlin_gemv_diff(const void *a, const void *b, size_t count, float *host_out, cudaStream_t s, int bf16) {
    static float *dev = nullptr;
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    if (!dev && cudaMalloc(&dev, 64) != cudaSuccess) return -2;
    cudaMemsetAsync(dev, 0, 64, s);
    if (bf16) diff_kernel<__nv_bfloat16><<<64, 256, 0, s>>>(static_cast<const __nv_bfloat16 *>(a), static_cast<const __nv_bfloat16 *>(b), count, dev);
    else diff_kernel<__half><<<64, 256, 0, s>>>(static_cast<const __half *>(a), static_cast<const __half *>(b), count, dev);
    cudaMemcpyAsync(host_out, dev, 2 * sizeof(float), cudaMemcpyDeviceToHost, s);
    return cudaStreamSynchronize(s) == cudaSuccess ? 0 : -2;
}

extern "C" int fllm_vllm_marlin_gemm(const void *A, const void *B, void *C, void *C_tmp, const void *scales, const void *zeros, const void *bias,
                                     const void *global_scale, int m, int n, int k, int lda, void *workspace, int kind, int group_size, int dev,
                                     int sms, cudaStream_t stream);
extern "C" size_t fllm_vllm_marlin_workspace_bytes(int n);
extern "C" size_t fllm_vllm_marlin_ctmp_bytes(int m, int sms);

// Synthetic-weight comparison of the Marlin GEMM and the GEMV for an [m, k] x [k, n] product (FP16 only):
//   out[0] Marlin us/call, out[1] GEMV us/call, out[2] max |marlin - gemv|, out[3] max |marlin|, out[4] weight bytes, out[5] GB/s of the GEMV
//   out[6] Marlin with a half-precision reduction (no C_tmp) us/call (the vLLM option use_fp32_reduce=False)
// Returns 0, or a negative code (-1 shape not supported, -2 allocation or launch failure).
extern "C" int fllm_marlin_gemv_bench(int dev, int m, int n, int k, int kind, int group, int iters, double *out) {
    if (!fllm_marlin_gemv_supported(m, n, k, kind, group)) return -1;
    if (cudaSetDevice(dev) != cudaSuccess) return -2;
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    const size_t qbytes = (size_t)n * k / 2, sbytes = (size_t)(k / group) * n * 2, zbytes = (size_t)(k / group) * n / 2;
    const size_t cbytes = (size_t)m * n * 2, abytes = (size_t)m * k * 2;
    const size_t ctmp = fllm_vllm_marlin_ctmp_bytes(m, sms), wsb = fllm_vllm_marlin_workspace_bytes(n);
    const size_t part_bytes = (size_t)sms * 64 * 256 * 4;
    char *buf = nullptr;
    const size_t total = qbytes + sbytes + zbytes + 3 * cbytes + abytes + ctmp + wsb + part_bytes + (n / 64) * 4 + 4096 * 8;
    if (cudaMalloc(&buf, total) != cudaSuccess) return -2;
    char *p = buf;
    auto take = [&](size_t bytes) {
        char *r = p;
        p += (bytes + 255) & ~size_t(255);
        return r;
    };
    void *B = take(qbytes), *S = take(sbytes), *Z = take(zbytes), *A = take(abytes), *C1 = take(cbytes), *C2 = take(cbytes), *C3 = take(cbytes);
    void *tmp = take(ctmp), *ws = take(wsb), *part = take(part_bytes), *tickets = take((size_t)(n / 64) * 4 + 256), *red = take(64);
    cudaMemset(ws, 0, wsb);
    cudaMemset(tickets, 0, (size_t)(n / 64) * 4);
    cudaMemset(red, 0, 64);
    fill_random_kernel<<<1024, 256>>>(static_cast<uint32_t *>(B), qbytes / 4, 12345u);
    fill_random_kernel<<<256, 256>>>(static_cast<uint32_t *>(Z), zbytes / 4, 777u);
    fill_values_kernel<__half><<<256, 256>>>(static_cast<__half *>(S), sbytes / 2, 99u, 0.02f, 0.03f);
    fill_values_kernel<__half><<<64, 256>>>(static_cast<__half *>(A), abytes / 2, 5u, 2.0f, 0.f);
    cudaStream_t s;
    cudaStreamCreate(&s);
    const void *zeros = kind == 0 ? Z : nullptr;
    auto marlin = [&](void *C, void *ctmp_ptr) {
        return fllm_vllm_marlin_gemm(A, B, C, ctmp_ptr, S, zeros, nullptr, nullptr, m, n, k, k, ws, kind, group, dev, sms, s);
    };
    auto gemv = [&](void *C) { return fllm_marlin_gemv(A, B, C, part, part_bytes, tickets, S, zeros, m, n, k, k, kind, group, 0, sms, s); };
    int rc = 0;
    if (marlin(C1, tmp) != 0 || gemv(C2) != 0 || marlin(C3, nullptr) != 0) rc = -2;
    cudaEvent_t e[4];
    for (auto &ev : e) cudaEventCreate(&ev);
    if (rc == 0) {
        if (iters < 1) iters = 1;
        auto time = [&](auto &&fn) {
            for (int i = 0; i < 5; i++) fn();
            cudaEventRecord(e[0], s);
            for (int i = 0; i < iters; i++) fn();
            cudaEventRecord(e[1], s);
            cudaEventSynchronize(e[1]);
            float ms = 0.f;
            cudaEventElapsedTime(&ms, e[0], e[1]);
            return (double)ms * 1000.0 / iters;
        };
        out[0] = time([&] { marlin(C1, tmp); });
        out[1] = time([&] { gemv(C2); });
        out[6] = time([&] { marlin(C3, nullptr); });
        diff_kernel<__half><<<64, 256, 0, s>>>(static_cast<const __half *>(C1), static_cast<const __half *>(C2), (size_t)m * n, static_cast<float *>(red));
        float h[2] = {0.f, 0.f};
        cudaMemcpyAsync(h, red, sizeof(h), cudaMemcpyDeviceToHost, s);
        cudaStreamSynchronize(s);
        out[2] = h[0];
        out[3] = h[1];
        out[4] = (double)(qbytes + sbytes + zbytes);
        out[5] = out[4] / (out[1] * 1e-6) / 1e9;
        if (cudaGetLastError() != cudaSuccess) rc = -2;
    }
    for (auto &ev : e) cudaEventDestroy(ev);
    cudaStreamDestroy(s);
    cudaFree(buf);
    return rc;
}
