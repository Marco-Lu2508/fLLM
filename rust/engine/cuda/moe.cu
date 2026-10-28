// fLLM CUDA backend: mixture-of-experts on the device. Routing for every gating variant the host path knows (softmax or sigmoid
// scores, selection bias, group-limited selection, renormalisation, scale), the sort of token-expert pairs into expert-major
// blocks, a grouped tensor-core GEMM that serves all experts of a projection in one launch, and the weighted combination with the
// shared expert.
//
// The algorithms follow the structure of vLLM's MoE path (Apache-2.0, https://github.com/vllm-project/vllm): `grouped_topk` and
// `topk_softmax` for the routing, `moe_align_block_size` for the sort/pad step (sorted_token_ids, expert_ids per block,
// num_tokens_post_padded) and `fused_moe` for the "one launch, expert-major tiles" GEMM. No code is copied; the kernels are written for
// this engine's weight layouts (the `decode16` decoders of common.cuh feed shared memory exactly as in tc_gemm.cu) and activations
// (FP32 in, FP32 out). All shapes that depend on the routing are padded to upper bounds, nothing is read back by the host, so every
// launch can be recorded in a CUDA graph.
//
//   routing:   logits[row][E] -> ids[row][k], weights[row][k]
//   align:     ids -> sorted[pair slots] (pair p = token * k + j, padded per expert to multiples of the tile), block_expert[block],
//              num_post (the padded length)
//   gemm:      Y[p][m] = sum_c W[e(p)][m][c] * X[p / xdiv][c]  (optionally: gate/up in one pass with the GLU applied in the epilogue)
//   combine:   out[t] = sum_j w[t][j] * y[t * k + j] (+ gate[t] * shared[t])
#include <float.h>
#include <limits.h>

#include "common.cuh"
#include "fllm_cuda_internal.h"
#include "mma_util.cuh"

// ---------------------------------------------------------------------------------------------
// Routing
// ---------------------------------------------------------------------------------------------
constexpr int MOE_ROUTE_WARPS = 4;
constexpr int MOE_MAX_EXPERTS = 1024;
constexpr int MOE_MAX_GROUPS = 64;
constexpr int MOE_MAX_K = 32;

__host__ __device__ constexpr size_t moe_route_warp_bytes(int experts) {
    return (size_t)(2 * experts + MOE_MAX_GROUPS + MOE_MAX_K) * 4 + (size_t)((experts + 3) & ~3);
}

// One warp per token. Scores = softmax or sigmoid of the logits; the selection value adds the bias; with groups the best
// `groups_used` groups (by their top score, or the sum of their two best when `top2`) stay eligible; the k best selection values win
// (ties to the lower expert index); the weights are the scores without bias, optionally divided by their sum (+ eps), times `scale`.
__global__ void __launch_bounds__(MOE_ROUTE_WARPS * 32)
moe_route2_kernel(const float *__restrict__ logits, const float *__restrict__ bias, int32_t *__restrict__ ids, float *__restrict__ w,
                  int rows, int experts, int k, int sigmoid, int normalize, float scale, float eps, int groups, int groups_used,
                  int top2) {
    extern __shared__ __align__(16) unsigned char moe_route_smem[];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * MOE_ROUTE_WARPS + warp;
    if (row >= rows) return;
    unsigned char *base = moe_route_smem + (size_t)warp * moe_route_warp_bytes(experts);
    float *scores = reinterpret_cast<float *>(base);
    float *sel = scores + experts;
    float *gval = sel + experts;
    int32_t *chosen = reinterpret_cast<int32_t *>(gval + MOE_MAX_GROUPS);
    unsigned char *taken = reinterpret_cast<unsigned char *>(chosen + MOE_MAX_K);
    const float *l = logits + (size_t)row * experts;

    if (!sigmoid) {
        float m = -3.0e38f;
        for (int e = lane; e < experts; e += 32) m = fmaxf(m, l[e]);
        m = warp_max(m);
        float sum = 0.f;
        for (int e = lane; e < experts; e += 32) {
            const float x = expf(l[e] - m);
            scores[e] = x;
            sum += x;
        }
        sum = warp_sum(sum);
        for (int e = lane; e < experts; e += 32) scores[e] = scores[e] / sum;
    } else {
        for (int e = lane; e < experts; e += 32) scores[e] = 1.0f / (1.0f + expf(-l[e]));
    }
    for (int e = lane; e < experts; e += 32) {
        sel[e] = scores[e] + (bias ? bias[e] : 0.f);
        taken[e] = 0;
    }
    __syncwarp();

    if (groups > 1) {
        const int per = experts / groups;
        for (int g = lane; g < groups; g += 32) {
            float t1 = -INFINITY, t2 = -INFINITY;
            for (int j = 0; j < per; j++) {
                const float v = sel[g * per + j];
                if (v > t1) {
                    t2 = t1;
                    t1 = v;
                } else if (v > t2) {
                    t2 = v;
                }
            }
            gval[g] = (top2 && per >= 2) ? t1 + t2 : t1;
        }
        __syncwarp();
        const int keep = groups_used < groups ? groups_used : groups;
        for (int g = lane; g < groups; g += 32) {
            int rank = 0;
            for (int h = 0; h < groups; h++)
                if (gval[h] > gval[g] || (gval[h] == gval[g] && h < g)) rank++;
            if (rank >= keep)
                for (int j = 0; j < per; j++) sel[g * per + j] = -INFINITY;
        }
        __syncwarp();
    }

    for (int j = 0; j < k; j++) {
        float best = -INFINITY;
        int bi = INT_MAX;
        for (int e = lane; e < experts; e += 32) {
            if (taken[e]) continue;
            const float v = sel[e];
            if (v > best || (v == best && e < bi)) {
                best = v;
                bi = e;
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ob > best || (ob == best && oi < bi)) {
                best = ob;
                bi = oi;
            }
        }
        if (lane == 0) {
            if (bi < experts) taken[bi] = 1;
            chosen[j] = bi < experts ? bi : 0;
        }
        __syncwarp();
    }

    float total = 0.f;
    for (int j = 0; j < k; j++) total += scores[chosen[j]];
    total += eps;
    if (lane < k) {
        const int id = chosen[lane];
        const float sc = scores[id];
        ids[(size_t)row * k + lane] = id;
        w[(size_t)row * k + lane] = (normalize ? sc / total : sc) * scale;
    }
}

int fllm_moe_route2_launch(cudaStream_t s, const float *logits, const float *bias, int32_t *ids, float *w, int rows, int experts, int k,
                           int sigmoid, int normalize, float scale, float eps, int groups, int groups_used, int top2) {
    if (rows <= 0) return 0;
    if (k < 1 || k > MOE_MAX_K || experts < k || experts > MOE_MAX_EXPERTS) return fllm_set_error("moe_route: unsupported expert counts (%d experts, k %d)", experts, k);
    if (groups > 1 && (groups > MOE_MAX_GROUPS || experts % groups != 0 || groups_used < 1))
        return fllm_set_error("moe_route: unsupported group counts (%d experts, %d groups, %d used)", experts, groups, groups_used);
    const size_t smem = moe_route_warp_bytes(experts) * MOE_ROUTE_WARPS;
    if (smem > 48 * 1024) return fllm_set_error("moe_route: too many experts");
    moe_route2_kernel<<<(rows + MOE_ROUTE_WARPS - 1) / MOE_ROUTE_WARPS, MOE_ROUTE_WARPS * 32, smem, s>>>(
        logits, bias, ids, w, rows, experts, k, sigmoid, normalize, scale, eps, groups, groups_used, top2);
    return fllm_check_launch("moe_route2");
}

// ---------------------------------------------------------------------------------------------
// Sort / align (one block): pairs grouped by (local) expert, every expert padded to a multiple of `block`.
// Experts outside [expert_offset, expert_offset + num_local) are dropped (their rows of y stay as they were).
// ---------------------------------------------------------------------------------------------
__global__ void moe_align_kernel(const int32_t *__restrict__ ids, int32_t *__restrict__ sorted, int32_t *__restrict__ block_expert,
                                 int32_t *__restrict__ num_post, int pairs, int expert_offset, int num_local, int block,
                                 int max_padded) {
    extern __shared__ int moe_align_smem[];
    int *cnt = moe_align_smem;               // [num_local]
    int *start = cnt + num_local;            // [num_local + 1]
    int *cur = start + num_local + 1;        // [num_local]
    for (int e = threadIdx.x; e < num_local; e += blockDim.x) {
        cnt[e] = 0;
        cur[e] = 0;
    }
    __syncthreads();
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) {
        const int e = ids[p] - expert_offset;
        if ((unsigned)e < (unsigned)num_local) atomicAdd(&cnt[e], 1);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int acc = 0;
        for (int e = 0; e < num_local; e++) {
            start[e] = acc;
            acc += ((cnt[e] + block - 1) / block) * block;
        }
        start[num_local] = acc;
        num_post[0] = acc;
    }
    __syncthreads();
    const int max_blocks = max_padded / block;
    for (int i = threadIdx.x; i < max_padded; i += blockDim.x) sorted[i] = pairs;
    for (int b = threadIdx.x; b < max_blocks; b += blockDim.x) block_expert[b] = -1;
    __syncthreads();
    for (int e = threadIdx.x; e < num_local; e += blockDim.x)
        for (int b = start[e] / block; b < start[e + 1] / block; b++) block_expert[b] = e;
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) {
        const int e = ids[p] - expert_offset;
        if ((unsigned)e < (unsigned)num_local) sorted[start[e] + atomicAdd(&cur[e], 1)] = p;
    }
}

int fllm_moe_align_launch(cudaStream_t s, const int32_t *ids, int32_t *sorted, int32_t *block_expert, int32_t *num_post, int pairs,
                          int expert_offset, int num_local, int block) {
    if (pairs <= 0) return 0;
    if (num_local < 1 || block < 1) return fllm_set_error("moe_align: invalid expert count or block size");
    const size_t smem = (size_t)(3 * num_local + 1) * sizeof(int);
    if (smem > 48 * 1024) return fllm_set_error("moe_align: too many experts");
    const int max_padded = (int)((((size_t)pairs + (size_t)num_local * (block - 1)) + block - 1) / block * block);
    moe_align_kernel<<<1, 1024, smem, s>>>(ids, sorted, block_expert, num_post, pairs, expert_offset, num_local, block, max_padded);
    return fllm_check_launch("moe_align");
}

// ---------------------------------------------------------------------------------------------
// Grouped GEMM over the experts (tensor cores, FP16 inputs, FP32 accumulation). Same scheme as tc_gemm.cu: the weight tile of the
// block is decoded straight into shared memory, the activation tile is converted from FP32, mma.sync m16n8k16. The block takes one
// run of `BN` sorted pairs of one expert; blocks past num_post and padding slots do nothing.
// ---------------------------------------------------------------------------------------------
template <int BN>
struct MoeTile {
    static constexpr int BM = 128, BK = 64, LD = BK + 8;
    static constexpr int WM = BN >= 64 ? 2 : 4;
    static constexpr int WN = BN / 16;
    static constexpr int NT = WM * WN * 32;
    static constexpr int MI = BM / WM / 16;
    static constexpr int NI = BN / WN / 8;
    static constexpr int APT = BM * BK / 16 / NT;
    static constexpr int RPK = BK / 16;
    static constexpr int BRUNS = BN * RPK;
    static constexpr int BPT = (BRUNS + NT - 1) / NT;
    static constexpr size_t SMEM = (size_t)2 * (BM + BN) * LD * sizeof(__half);
    static_assert(NI == 2 && MI >= 2 && MI % 2 == 0 && APT >= 1, "tile shape");
};

__device__ __forceinline__ uint32_t moe_pack_half2(float lo, float hi) {
    const __half2 h = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t *>(&h);
}
__device__ __forceinline__ void moe_load16_x(const float *p, int e0, int cols, float *x) {
    if (e0 + 16 <= cols) {
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
__device__ __forceinline__ void moe_store16(__half *dst, const float *v) {
    uint4 a, b;
    a.x = moe_pack_half2(v[0], v[1]);   a.y = moe_pack_half2(v[2], v[3]);
    a.z = moe_pack_half2(v[4], v[5]);   a.w = moe_pack_half2(v[6], v[7]);
    b.x = moe_pack_half2(v[8], v[9]);   b.y = moe_pack_half2(v[10], v[11]);
    b.z = moe_pack_half2(v[12], v[13]); b.w = moe_pack_half2(v[14], v[15]);
    uint4 *d = reinterpret_cast<uint4 *>(dst);
    d[0] = a;
    d[1] = b;
}

// glu_mode 0: SwiGLU, 1: GeGLU (tanh), 2: gpt-oss clamped SwiGLU (alpha, limit)
__device__ __forceinline__ float moe_glu(float g, float u, int mode, float alpha, float limit) {
    if (mode == 2) {
        const float gc = fminf(g, limit);
        const float uc = fminf(fmaxf(u, -limit), limit);
        return (uc + 1.f) * gc / (1.f + expf(-alpha * gc));
    }
    return fllm_glu(g, u, mode);
}

template <uint32_t KIND, int BN, bool GLU>
__global__ void __launch_bounds__(MoeTile<BN>::NT)
moe_tc_kernel(const uint8_t *const *__restrict__ wtab, const uint8_t *const *__restrict__ wtab2, const float *const *__restrict__ btab,
              const float *const *__restrict__ btab2, const float *__restrict__ X, const float *__restrict__ pw, float *__restrict__ Y,
              const int32_t *__restrict__ sorted, const int32_t *__restrict__ block_expert, const int32_t *__restrict__ num_post,
              int rows, int cols, int pairs, int xdiv, int glu_mode, float alpha, float limit) {
#if FLLM_TC
    using T = MoeTile<BN>;
    if ((int)blockIdx.y * BN >= num_post[0]) return;
    const int e = block_expert[blockIdx.y];
    if (e < 0) return;
    extern __shared__ __align__(16) unsigned char moe_tc_smem[];
    __half *As = reinterpret_cast<__half *>(moe_tc_smem);       // [2][BM][LD]
    __half *Bs = As + (size_t)2 * T::BM * T::LD;                // [2][BN][LD]
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp % T::WM, wn = warp / T::WM;
    constexpr int HALF = T::BM / T::WM / 2;  // gate rows (and up rows) per warp when the two projections share the tile
    const int m0 = blockIdx.x * (GLU ? T::BM / 2 : T::BM);
    const size_t row_bytes = kind_row_bytes(KIND, cols);
    const int ktiles = (cols + T::BK - 1) / T::BK;
    const int32_t *my_sorted = sorted + (size_t)blockIdx.y * BN;

    const uint8_t *wrow[T::APT];
    const float *srow[T::APT];  // FP8B: the tile scales (128 x 128 tiles) of the row, behind the codes of the expert matrix
    {
        const uint8_t *Wg = wtab[e];
        const uint8_t *Wu = GLU ? wtab2[e] : nullptr;
#pragma unroll
        for (int i = 0; i < T::APT; i++) {
            const int s = (tid + i * T::NT) / T::RPK;
            int row;
            const uint8_t *Wb = Wg;
            if (GLU) {
                const int wmi = s / (2 * HALF), tt = s % (2 * HALF);
                row = m0 + wmi * HALF + tt % HALF;
                if (tt >= HALF) Wb = Wu;
            } else {
                row = m0 + s;
            }
            wrow[i] = row < rows ? Wb + (size_t)row * row_bytes : nullptr;
            srow[i] = nullptr;
            if constexpr (KIND == KIND_FP8B) {
                if (row < rows) srow[i] = reinterpret_cast<const float *>(Wb + (size_t)rows * cols) + (size_t)(row >> 7) * ((cols + 127) >> 7);
            }
        }
    }
    const float *xrow[T::BPT];
    float xsc[T::BPT];
#pragma unroll
    for (int i = 0; i < T::BPT; i++) {
        const int run = tid + i * T::NT;
        const int p = run < T::BRUNS ? my_sorted[run / T::RPK] : pairs;
        xrow[i] = p < pairs ? X + (size_t)(p / xdiv) * cols : nullptr;
        xsc[i] = (pw && p < pairs) ? pw[p] : 1.f;
    }

    float acc[T::MI][T::NI][4];
#pragma unroll
    for (int a = 0; a < T::MI; a++)
#pragma unroll
        for (int b = 0; b < T::NI; b++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[a][b][c] = 0.f;

    float wv[T::APT][16], xv[T::BPT][16];
    auto fetch = [&](int kt) {
#pragma unroll
        for (int i = 0; i < T::APT; i++) {
            const int k = kt * T::BK + ((tid + i * T::NT) % T::RPK) * 16;
            if (wrow[i]) {
                decode16<KIND>(wrow[i], k >> 4, cols, wv[i]);
                if constexpr (KIND == KIND_FP8B) {
                    // the tile scale of the 16 codes (a run never straddles two 128-wide tiles); past the end of the row the activations are zero
                    const float sc = k < cols ? __ldg(srow[i] + (k >> 7)) : 0.f;
#pragma unroll
                    for (int j = 0; j < 16; j++) wv[i][j] *= sc;
                }
            } else {
#pragma unroll
                for (int j = 0; j < 16; j++) wv[i][j] = 0.f;
            }
        }
#pragma unroll
        for (int i = 0; i < T::BPT; i++) {
            const int k = kt * T::BK + ((tid + i * T::NT) % T::RPK) * 16;
            if (xrow[i]) {
                moe_load16_x(xrow[i], k, cols, xv[i]);
                if (pw) {
#pragma unroll
                    for (int j = 0; j < 16; j++) xv[i][j] *= xsc[i];
                }
            } else {
#pragma unroll
                for (int j = 0; j < 16; j++) xv[i][j] = 0.f;
            }
        }
    };
    auto stash = [&](int buf) {
#pragma unroll
        for (int i = 0; i < T::APT; i++) {
            const int run = tid + i * T::NT;
            moe_store16(As + ((size_t)buf * T::BM + run / T::RPK) * T::LD + (run % T::RPK) * 16, wv[i]);
        }
#pragma unroll
        for (int i = 0; i < T::BPT; i++) {
            const int run = tid + i * T::NT;
            if (run < T::BRUNS) moe_store16(Bs + ((size_t)buf * BN + run / T::RPK) * T::LD + (run % T::RPK) * 16, xv[i]);
        }
    };

    fetch(0);
    stash(0);
    __syncthreads();
    for (int kt = 0; kt < ktiles; kt++) {
        const int buf = kt & 1;
        if (kt + 1 < ktiles) fetch(kt + 1);
        const __half *a_base = As + (size_t)buf * T::BM * T::LD;
        const __half *b_base = Bs + (size_t)buf * BN * T::LD;
#pragma unroll
        for (int kk = 0; kk < T::BK; kk += 16) {
            uint32_t af[T::MI][4], bf[T::NI][2];
#pragma unroll
            for (int mi = 0; mi < T::MI; mi++)
                ldmatrix4(af[mi][0], af[mi][1], af[mi][2], af[mi][3],
                          a_base + (size_t)(wm * (T::BM / T::WM) + mi * 16 + (lane & 15)) * T::LD + kk + (lane >> 4) * 8);
#pragma unroll
            for (int np = 0; np < T::NI / 2; np++)
                ldmatrix4(bf[np * 2][0], bf[np * 2][1], bf[np * 2 + 1][0], bf[np * 2 + 1][1],
                          b_base + (size_t)(wn * (BN / T::WN) + np * 16 + (lane & 7) + ((lane >> 4) << 3)) * T::LD + kk +
                              ((lane >> 3) & 1) * 8);
#pragma unroll
            for (int mi = 0; mi < T::MI; mi++)
#pragma unroll
                for (int ni = 0; ni < T::NI; ni++) mma16816(acc[mi][ni], af[mi], bf[ni][0], bf[ni][1]);
        }
        if (kt + 1 < ktiles) stash(buf ^ 1);
        __syncthreads();
    }

    const int g = lane >> 2, t = lane & 3;
    const float *bg = btab ? btab[e] : nullptr;
    if (!GLU) {
#pragma unroll
        for (int mi = 0; mi < T::MI; mi++)
#pragma unroll
            for (int ni = 0; ni < T::NI; ni++)
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    const int m = m0 + wm * (T::BM / T::WM) + mi * 16 + g + (c >> 1) * 8;
                    const int p = my_sorted[wn * (BN / T::WN) + ni * 8 + 2 * t + (c & 1)];
                    if (m < rows && p < pairs) Y[(size_t)p * rows + m] = acc[mi][ni][c] + (bg ? bg[m] : 0.f);
                }
    } else {
        const float *bu = btab2 ? btab2[e] : nullptr;
#pragma unroll
        for (int mi = 0; mi < T::MI / 2; mi++)
#pragma unroll
            for (int ni = 0; ni < T::NI; ni++)
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    const int m = m0 + wm * HALF + mi * 16 + g + (c >> 1) * 8;
                    const int p = my_sorted[wn * (BN / T::WN) + ni * 8 + 2 * t + (c & 1)];
                    if (m < rows && p < pairs) {
                        const float gv = acc[mi][ni][c] + (bg ? bg[m] : 0.f);
                        const float uv = acc[mi + T::MI / 2][ni][c] + (bu ? bu[m] : 0.f);
                        Y[(size_t)p * rows + m] = moe_glu(gv, uv, glu_mode, alpha, limit);
                    }
                }
    }
#endif
}

struct MoeGemmArgs {
    const void *const *wtab, *const *wtab2, *const *btab, *const *btab2;
    const float *X, *pw;
    float *Y;
    const int32_t *sorted, *block_expert, *num_post;
    int rows, cols, pairs, xdiv, max_blocks, glu_mode;
    float alpha, limit;
};

template <uint32_t KIND, int BN, bool GLU>
static int moe_tc_run(cudaStream_t s, const MoeGemmArgs &a) {
    using T = MoeTile<BN>;
    if (T::SMEM > (size_t)fllm_smem_limit()) return fllm_set_error("moe_grouped: the device offers too little shared memory");
    static bool configured[16] = {false};
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev >= 0 && dev < 16 && !configured[dev]) {
        cudaFuncSetAttribute(moe_tc_kernel<KIND, BN, GLU>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)T::SMEM);
        configured[dev] = true;
    }
    const int per_block = GLU ? T::BM / 2 : T::BM;
    const dim3 grid((a.rows + per_block - 1) / per_block, a.max_blocks), block(T::NT);
    moe_tc_kernel<KIND, BN, GLU><<<grid, block, T::SMEM, s>>>(
        reinterpret_cast<const uint8_t *const *>(a.wtab), reinterpret_cast<const uint8_t *const *>(a.wtab2),
        reinterpret_cast<const float *const *>(a.btab), reinterpret_cast<const float *const *>(a.btab2), a.X, a.pw, a.Y, a.sorted,
        a.block_expert, a.num_post, a.rows, a.cols, a.pairs, a.xdiv, a.glu_mode, a.alpha, a.limit);
    return fllm_check_launch("moe_grouped");
}

template <int BN, bool GLU>
static int moe_tc_kind(cudaStream_t s, uint32_t kind, const MoeGemmArgs &a) {
    switch (kind) {
        case KIND_F16: return moe_tc_run<KIND_F16, BN, GLU>(s, a);
        case KIND_BF16: return moe_tc_run<KIND_BF16, BN, GLU>(s, a);
        case KIND_Q4_0: return moe_tc_run<KIND_Q4_0, BN, GLU>(s, a);
        case KIND_Q4_1: return moe_tc_run<KIND_Q4_1, BN, GLU>(s, a);
        case KIND_Q5_0: return moe_tc_run<KIND_Q5_0, BN, GLU>(s, a);
        case KIND_Q5_1: return moe_tc_run<KIND_Q5_1, BN, GLU>(s, a);
        case KIND_Q8_0: return moe_tc_run<KIND_Q8_0, BN, GLU>(s, a);
        case KIND_IQ4_NL: return moe_tc_run<KIND_IQ4_NL, BN, GLU>(s, a);
        case KIND_MXFP4: return moe_tc_run<KIND_MXFP4, BN, GLU>(s, a);
        case KIND_Q4_K: return moe_tc_run<KIND_Q4_K, BN, GLU>(s, a);
        case KIND_Q5_K: return moe_tc_run<KIND_Q5_K, BN, GLU>(s, a);
        case KIND_Q6_K: return moe_tc_run<KIND_Q6_K, BN, GLU>(s, a);
        case KIND_Q2_K: return moe_tc_run<KIND_Q2_K, BN, GLU>(s, a);
        case KIND_Q3_K: return moe_tc_run<KIND_Q3_K, BN, GLU>(s, a);
        case KIND_IQ2_XXS: return moe_tc_run<KIND_IQ2_XXS, BN, GLU>(s, a);
        case KIND_IQ2_XS: return moe_tc_run<KIND_IQ2_XS, BN, GLU>(s, a);
        case KIND_IQ3_XXS: return moe_tc_run<KIND_IQ3_XXS, BN, GLU>(s, a);
        case KIND_IQ1_S: return moe_tc_run<KIND_IQ1_S, BN, GLU>(s, a);
        case KIND_IQ3_S: return moe_tc_run<KIND_IQ3_S, BN, GLU>(s, a);
        case KIND_IQ2_S: return moe_tc_run<KIND_IQ2_S, BN, GLU>(s, a);
        case KIND_IQ1_M: return moe_tc_run<KIND_IQ1_M, BN, GLU>(s, a);
        case KIND_NVFP4: return moe_tc_run<KIND_NVFP4, BN, GLU>(s, a);
        case KIND_Q4Z32: return moe_tc_run<KIND_Q4Z32, BN, GLU>(s, a);
        case KIND_Q4Z64: return moe_tc_run<KIND_Q4Z64, BN, GLU>(s, a);
        case KIND_Q4Z128: return moe_tc_run<KIND_Q4Z128, BN, GLU>(s, a);
        case KIND_FP8B: return moe_tc_run<KIND_FP8B, BN, GLU>(s, a);
        default: return fllm_set_error("moe_grouped: unsupported weight type %d", kind);
    }
}

int fllm_moe_grouped_supported(uint32_t kind, int cols, int block) {
    switch (kind) {
        case KIND_F16: case KIND_BF16: case KIND_Q4_0: case KIND_Q4_1: case KIND_Q5_0: case KIND_Q5_1: case KIND_Q8_0:
        case KIND_IQ4_NL: case KIND_MXFP4: case KIND_Q4_K: case KIND_Q5_K: case KIND_Q6_K:
        case KIND_Q2_K: case KIND_Q3_K: case KIND_IQ2_XXS: case KIND_IQ2_XS: case KIND_IQ3_XXS: case KIND_IQ1_S:
        case KIND_IQ3_S: case KIND_IQ2_S: case KIND_IQ1_M: case KIND_NVFP4:
        case KIND_Q4Z32: case KIND_Q4Z64: case KIND_Q4Z128:
        case KIND_FP8B: break;  // 128 x 128 tiles; FP8M (row scales behind the codes, or the W8A8 / Marlin layouts) has no grouped kernel
        default: return 0;
    }
    return (block == 16 || block == 64) && (cols & 15) == 0 && (cols % kind_block_elems(kind)) == 0 && fllm_cc() >= 75;
}

int fllm_moe_grouped_launch(cudaStream_t s, uint32_t kind, int glu, const void *const *wtab, const void *const *wtab2,
                            const void *const *btab, const void *const *btab2, const float *X, const float *pw, float *Y,
                            const int32_t *sorted, const int32_t *block_expert, const int32_t *num_post, int rows, int cols, int pairs,
                            int xdiv, int block, int max_blocks, int glu_mode, float alpha, float limit) {
    if (pairs <= 0 || rows <= 0) return 0;
    if (!fllm_moe_grouped_supported(kind, cols, block) || xdiv < 1 || max_blocks < 1 || (glu && !wtab2))
        return fllm_set_error("moe_grouped: unsupported shape or weight type (kind %d, cols %d, block %d)", kind, cols, block);
    if ((reinterpret_cast<uintptr_t>(X) & 15) != 0) return fllm_set_error("moe_grouped: unaligned activations");
    MoeGemmArgs a{wtab, wtab2, btab, btab2, X, pw, Y, sorted, block_expert, num_post, rows, cols, pairs, xdiv, max_blocks, glu_mode, alpha, limit};
    if (glu) return block == 16 ? moe_tc_kind<16, true>(s, kind, a) : moe_tc_kind<64, true>(s, kind, a);
    return block == 16 ? moe_tc_kind<16, false>(s, kind, a) : moe_tc_kind<64, false>(s, kind, a);
}

// ---------------------------------------------------------------------------------------------
// Combination: out[t][h] = sum_j w[t][j] * y[t * k + j][h] (+ shared[t][h] * gate[t]). A null `w` means weight 1 (Llama 4 scales the
// expert input instead); a null `gate` means 1. Summation order is fixed, so the result does not depend on the sort.
// ---------------------------------------------------------------------------------------------
__global__ void moe_combine2_kernel(float *__restrict__ out, const float *__restrict__ y, const float *__restrict__ w,
                                    const float *__restrict__ shared, const float *__restrict__ gate, int tokens, int k, int hidden) {
    const size_t total = (size_t)tokens * hidden;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const int t = (int)(i / hidden), h = (int)(i % hidden);
        float acc = 0.f;
        for (int j = 0; j < k; j++) acc += (w ? w[(size_t)t * k + j] : 1.f) * y[((size_t)t * k + j) * hidden + h];
        if (shared) acc += shared[i] * (gate ? gate[t] : 1.f);
        out[i] = acc;
    }
}

int fllm_moe_combine2_launch(cudaStream_t s, float *out, const float *y, const float *w, const float *shared, const float *gate, int tokens,
                             int k, int hidden) {
    const size_t total = (size_t)tokens * hidden;
    if (total == 0) return 0;
    const size_t blocks = (total + 255) / 256;
    moe_combine2_kernel<<<(unsigned)(blocks < 65535 * 4 ? blocks : 65535 * 4), 256, 0, s>>>(out, y, w, shared, gate, tokens, k, hidden);
    return fllm_check_launch("moe_combine2");
}

// gate[t] = sigmoid(g . x[t]) for the gated shared expert (Qwen2-MoE, Qwen3-Next)
__global__ void moe_gate_scale_kernel(float *__restrict__ gate, const float *__restrict__ x, const float *__restrict__ g, int hidden) {
    const float *xt = x + (size_t)blockIdx.x * hidden;
    float sum = 0.f;
    for (int i = threadIdx.x; i < hidden; i += blockDim.x) sum += xt[i] * g[i];
    sum = warp_sum(sum);
    __shared__ float part[8];
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float total = 0.f;
        for (int i = 0; i < (int)(blockDim.x >> 5); i++) total += part[i];
        gate[blockIdx.x] = 1.0f / (1.0f + expf(-total));
    }
}

int fllm_moe_gate_scale_launch(cudaStream_t s, float *gate, const float *x, const float *g, int tokens, int hidden) {
    if (tokens <= 0) return 0;
    moe_gate_scale_kernel<<<tokens, 256, 0, s>>>(gate, x, g, hidden);
    return fllm_check_launch("moe_gate_scale");
}
