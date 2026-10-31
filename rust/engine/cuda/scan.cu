// fLLM CUDA / HIP backend: chunk-parallel prompt processing of the recurrent layers.
//
// Gated DeltaNet, Kimi Delta Attention, RWKV-7, RWKV-6 and Mamba-2 are all instances of one linear recurrence per head,
//     P_t = (diag(w_t) - b_t a_t^T) P_{t-1} + k_t v_t^T,      o_t = P_t^T q_t + c_t v_t,      w_t = exp(lw_t), lw_t <= 0,
// whose chunked form (rust/engine/src/chunked_scan.rs proves it equal to the sequential recurrence in f64, and documents the derivation) turns
// a run of tokens into steps of 64 tokens. The serial kernels of deltanet.cu / kda.cu / rwkv.cu / mamba2.cu walk the tokens one by one, so a
// prompt costs one dependent chain of `n` steps per layer; here only the chain of `n / 64` state updates is serial.
//
// Per layer: a small per-architecture kernel brings the inputs of the run into the canonical layout (scan_*_prep), five generic kernels do
// the chunk algebra (scan_cum, scan_pairs, scan_solve, scan_out, scan_state), a small per-architecture kernel applies the output norm / gate
// (scan_*_post). Everything is FP32 CUDA-core code with static shared memory, no warp primitives: it runs on Turing and on the HIP targets.
//
// Layout. Rows of a run sit in the scratch at `lr = run.pad * 64 + t` (`run.pad` = the run's first chunk slot); the arrays are
// [lr][head][width]: q, kp, a, b `hk` wide, v and o `hv` wide, lw `hk` wide (flag LWV) or one value, c one value. The chunk workspace of
// chunk slot `cs`, head `h` is `ws + (cs * heads + h) * chunk_floats` (see ScanP for the parts).
#include <string.h>

#include "common.cuh"
#include "fllm_cuda_internal.h"

#define SC 64       // tokens per chunk
#define SCW 32      // value columns of the state per block of scan_state
#define SKT 16      // channels per tile of scan_pairs
#define SPS 33      // row stride (floats) of the state tile in shared memory (odd: no bank conflicts)
#define SMAX_HK 128 // widest key dimension of scan_state
#define SLW_MIN (-30.0f)  // log decays are clamped here: exp(-30) = 1e-13 is zero for a state, and a cumulative sum never leaves the float range

enum { SF_RANK1 = 1, SF_BSAME = 2, SF_LWV = 4, SF_QPREV = 8, SF_HASC = 16 };

// Same record as DeltaRun in backend/mod.rs; `pad` is the first chunk slot of the run.
typedef FllmRun ScanRun;

struct ScanP {
    int hk, hv, heads, flags;
    int gw;  // width of a row of G: hk (LWV) or 1
    size_t chunk_floats;
    size_t o_g, o_bm, o_xi, o_qk, o_qb, o_r0, o_wr, o_qe, o_o0, o_kd, o_bd, o_de;
    size_t state_off;                       // bytes into the run's state buffer
    size_t hstride, kstride, vstride;       // state element (h, i, j) at h * hstride + i * kstride + j * vstride floats
};

struct ScanIn {
    float *q, *kp, *b, *a, *lw, *c, *v, *o;
    float *ws;
};

struct ScanPlan {
    ScanP p;
    ScanIn in;
    size_t floats;
};

static size_t up4(size_t x) { return (x + 3) & ~size_t(3); }

// Carves the scratch of `slots` chunk slots out of `base` (null: only sizes).
static ScanPlan scan_plan(float *base, int hk, int hv, int heads, int flags, int slots) {
    ScanPlan sp;
    memset(&sp, 0, sizeof(sp));
    ScanP &p = sp.p;
    p.hk = hk;
    p.hv = hv;
    p.heads = heads;
    p.flags = flags;
    p.gw = (flags & SF_LWV) ? hk : 1;
    const bool r1 = flags & SF_RANK1;
    size_t off = 0;
    auto take = [&](size_t n) {
        const size_t o = off;
        off += up4(n);
        return o;
    };
    // chunk workspace
    size_t c = 0;
    auto part = [&](size_t n) {
        const size_t o = c;
        c += up4(n);
        return o;
    };
    p.o_g = part((size_t)SC * p.gw);
    p.o_bm = part(r1 ? SC * SC : 0);
    p.o_xi = part(r1 ? SC * SC : 0);
    p.o_qk = part(SC * SC);
    p.o_qb = part(r1 ? SC * SC : 0);
    p.o_r0 = part(r1 ? (size_t)SC * hv : 0);
    p.o_wr = part(r1 ? (size_t)SC * hk : 0);
    p.o_qe = part((size_t)SC * hk);
    p.o_o0 = part((size_t)SC * hv);
    p.o_kd = part((size_t)SC * hk);
    p.o_bd = part(r1 ? (size_t)SC * hk : 0);
    p.o_de = part(hk);
    p.chunk_floats = c;
    const size_t rows = (size_t)slots * SC, rh = rows * heads;
    const size_t oq = take(rh * hk), okp = take(rh * hk), oa = take(r1 ? rh * hk : 0), ob = take(r1 && !(flags & SF_BSAME) ? rh * hk : 0);
    const size_t olw = take(rh * p.gw), oc = take(rh), ov = take(rh * hv), oo = take(rh * hv);
    const size_t ows = take((size_t)slots * heads * c);
    sp.floats = off;
    if (base) {
        sp.in.q = base + oq;
        sp.in.kp = base + okp;
        sp.in.a = r1 ? base + oa : nullptr;
        sp.in.b = r1 ? ((flags & SF_BSAME) ? base + okp : base + ob) : nullptr;
        sp.in.lw = base + olw;
        sp.in.c = base + oc;
        sp.in.v = base + ov;
        sp.in.o = base + oo;
        sp.in.ws = base + ows;
    }
    return sp;
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// generic chunk algebra
// ---------------------------------------------------------------------------------------------------------------------------------------

__device__ __forceinline__ size_t scan_row(const ScanP &p, int lr, int h, int width) { return ((size_t)lr * p.heads + h) * width; }

// grid (chunks, runs, heads), block hk. Cumulative log decay G of the chunk, the decay-scaled factors of the state update (Kd, Bd, dE) and the
// state-independent part of the output weights (Qe = q * exp(G), or exp(G - lw) for the early readout).
__global__ void __launch_bounds__(128) scan_cum_kernel(ScanP p, ScanIn in, const ScanRun *__restrict__ runs) {
    const ScanRun run = runs[blockIdx.y];
    const int t0 = blockIdx.x * SC;
    if (t0 >= run.n) return;
    const int L = min(SC, run.n - t0), h = blockIdx.z, i = threadIdx.x, hk = p.hk;
    const int lr0 = (run.pad + blockIdx.x) * SC;
    float *w = in.ws + ((size_t)(run.pad + blockIdx.x) * p.heads + h) * p.chunk_floats;
    float *G = w + p.o_g;
    const bool lwv = p.flags & SF_LWV, r1 = p.flags & SF_RANK1, qp = p.flags & SF_QPREV;
    const int lww = lwv ? hk : 1;
    const float *lw = in.lw + scan_row(p, lr0, h, lww);
    const size_t lws = (size_t)p.heads * lww;
    if (lwv) {
        float acc = 0.f;
        for (int t = 0; t < SC; t++) {
            if (t < L) acc += lw[t * lws + i];
            G[(size_t)t * hk + i] = acc;
        }
    } else if (i == 0) {
        float acc = 0.f;
        for (int t = 0; t < SC; t++) {
            if (t < L) acc += lw[t * lws];
            G[t] = acc;
        }
    }
    __syncthreads();
    const float gend = lwv ? G[(size_t)(L - 1) * hk + i] : G[L - 1];
    (w + p.o_de)[i] = expf(gend);
    const float *q = in.q + scan_row(p, lr0, h, hk), *kp = in.kp + scan_row(p, lr0, h, hk);
    const float *b = r1 ? in.b + scan_row(p, lr0, h, hk) : nullptr;
    const size_t rs = (size_t)p.heads * hk;
    for (int t = 0; t < L; t++) {
        const float gt = lwv ? G[(size_t)t * hk + i] : G[t];
        float gq = gt;
        if (qp) gq -= lw[t * lws + (lwv ? i : 0)];
        const float dec = expf(gend - gt);
        (w + p.o_kd)[(size_t)t * hk + i] = kp[t * rs + i] * dec;
        if (r1) (w + p.o_bd)[(size_t)t * hk + i] = b[t * rs + i] * dec;
        (w + p.o_qe)[(size_t)t * hk + i] = q[t * rs + i] * expf(gq);
    }
}

// grid (chunks, runs, heads), block 256: the four 64 x 64 matrices of pairwise decayed products (see chunked_scan.rs):
//   Bm[t][s] = sum_i a[t][i] b[s][i]  exp(Gp[t][i] - G[s][i])   s < t         Xi: the same with kp instead of b
//   Qk[t][s] = sum_i q[t][i] kp[s][i] exp(Gq[t][i] - G[s][i])   s <= t (s < t for the early readout)    Qb: the same with b
// Thread (ty, tx) of 16 x 16 owns the 4 x 4 pairs t = ty + 16 i, s = tx + 16 j; the channels come in tiles of SKT through shared memory.
template <int FL>
__global__ void __launch_bounds__(256) scan_pairs_kernel(ScanP p, ScanIn in, const ScanRun *__restrict__ runs) {
    constexpr bool R1 = FL & SF_RANK1, BS = FL & SF_BSAME, LV = FL & SF_LWV, QP = FL & SF_QPREV;
    __shared__ float sq[SC][SKT + 1], sk[SC][SKT + 1], sa[R1 ? SC : 1][SKT + 1], sb[(R1 && !BS) ? SC : 1][SKT + 1];
    __shared__ float sg[LV ? SC : 1][SKT + 1], sgp[(LV && (R1 || QP)) ? SC : 1][SKT + 1];
    __shared__ float gs[LV ? 1 : SC], gps[LV ? 1 : SC];
    const ScanRun run = runs[blockIdx.y];
    const int t0 = blockIdx.x * SC;
    if (t0 >= run.n) return;
    const int L = min(SC, run.n - t0), h = blockIdx.z, hk = p.hk, tid = threadIdx.x;
    const int tx = tid & 15, ty = tid >> 4;
    const int lr0 = (run.pad + blockIdx.x) * SC;
    float *w = in.ws + ((size_t)(run.pad + blockIdx.x) * p.heads + h) * p.chunk_floats;
    const float *G = w + p.o_g;
    const int lww = LV ? hk : 1;
    const float *lw = in.lw + scan_row(p, lr0, h, lww);
    const size_t lws = (size_t)p.heads * lww, rs = (size_t)p.heads * hk;
    const float *q = in.q + scan_row(p, lr0, h, hk), *kp = in.kp + scan_row(p, lr0, h, hk);
    const float *av = R1 ? in.a + scan_row(p, lr0, h, hk) : nullptr, *bv = R1 ? in.b + scan_row(p, lr0, h, hk) : nullptr;
    if (!LV) {
        for (int r = tid; r < SC; r += 256) {
            const float g = r < L ? G[r] : 0.f;
            gs[r] = g;
            gps[r] = r < L ? g - lw[r * lws] : 0.f;
        }
    }
    float aq[4][4], ab[4][4], ax[4][4], aqb[4][4];
#pragma unroll
    for (int x = 0; x < 4; x++)
#pragma unroll
        for (int y = 0; y < 4; y++) aq[x][y] = ab[x][y] = ax[x][y] = aqb[x][y] = 0.f;

    for (int i0 = 0; i0 < hk; i0 += SKT) {
        __syncthreads();
        for (int e = tid; e < SC * SKT; e += 256) {
            const int r = e >> 4, ci = e & 15;
            const bool ok = r < L;
            sq[r][ci] = ok ? q[r * rs + i0 + ci] : 0.f;
            sk[r][ci] = ok ? kp[r * rs + i0 + ci] : 0.f;
            if (R1) {
                sa[r][ci] = ok ? av[r * rs + i0 + ci] : 0.f;
                if (!BS) sb[r][ci] = ok ? bv[r * rs + i0 + ci] : 0.f;
            }
            if (LV) {
                const float g = ok ? G[(size_t)r * hk + i0 + ci] : 0.f;
                sg[r][ci] = g;
                if (R1 || QP) sgp[r][ci] = ok ? g - lw[r * lws + i0 + ci] : 0.f;
            }
        }
        __syncthreads();
#pragma unroll
        for (int ci = 0; ci < SKT; ci++) {
            float qt[4], at[4], gt[4], gpt[4], ku[4], bu[4], gu[4];
#pragma unroll
            for (int x = 0; x < 4; x++) {
                const int t = ty + 16 * x;
                qt[x] = sq[t][ci];
                at[x] = R1 ? sa[t][ci] : 0.f;
                gt[x] = LV ? sg[t][ci] : 0.f;
                gpt[x] = (LV && (R1 || QP)) ? sgp[t][ci] : 0.f;
                const int u = tx + 16 * x;
                ku[x] = sk[u][ci];
                bu[x] = R1 ? (BS ? ku[x] : sb[u][ci]) : 0.f;
                gu[x] = LV ? sg[u][ci] : 0.f;
            }
#pragma unroll
            for (int x = 0; x < 4; x++)
#pragma unroll
                for (int y = 0; y < 4; y++) {
                    if (LV) {
                        const float gq = QP ? gpt[x] : gt[x];  // early readout: the decay stops before the token
                        const float e2 = expf(gq - gu[y]);
                        if (R1) {
                            aq[x][y] += qt[x] * ku[y] * e2;
                            if (!BS) aqb[x][y] += qt[x] * bu[y] * e2;
                            const float e1 = expf(gpt[x] - gu[y]);
                            ab[x][y] += at[x] * bu[y] * e1;
                            if (!BS) ax[x][y] += at[x] * ku[y] * e1;
                        } else {
                            aq[x][y] += qt[x] * ku[y] * e2;
                        }
                    } else {
                        aq[x][y] += qt[x] * ku[y];
                        if (R1) {
                            ab[x][y] += at[x] * bu[y];
                            if (!BS) {
                                aqb[x][y] += qt[x] * bu[y];
                                ax[x][y] += at[x] * ku[y];
                            }
                        }
                    }
                }
        }
    }
    float *bm = w + p.o_bm, *xi = w + p.o_xi, *qk = w + p.o_qk, *qb = w + p.o_qb;
#pragma unroll
    for (int x = 0; x < 4; x++)
#pragma unroll
        for (int y = 0; y < 4; y++) {
            const int t = ty + 16 * x, u = tx + 16 * y;
            const bool live = t < L;
            float sc1 = 1.f, sc2 = 1.f;
            if (!LV && live && u <= t) {
                sc2 = expf((QP ? gps[t] : gs[t]) - gs[u]);
                sc1 = expf(gps[t] - gs[u]);
            }
            qk[t * SC + u] = (live && (QP ? u < t : u <= t)) ? aq[x][y] * sc2 : 0.f;
            if (R1) {
                bm[t * SC + u] = (live && u < t) ? ab[x][y] * sc1 : 0.f;
                xi[t * SC + u] = (live && u < t) ? (BS ? ab[x][y] : ax[x][y]) * sc1 : 0.f;
                qb[t * SC + u] = (live && u <= t) ? (BS ? aq[x][y] : aqb[x][y]) * sc2 : 0.f;
            }
        }
}

// grid (chunks, runs, heads * ceil((hv + hk) / 64)), block 64, only with rank-one terms. One column per thread of Y = (I + Bm)^-1 [Xi V | a exp(Gp)]
// by forward substitution: columns < hv give R0, the others Wr. A thread reads back only what it wrote itself.
__global__ void __launch_bounds__(64) scan_solve_kernel(ScanP p, ScanIn in, const ScanRun *__restrict__ runs, int colblocks) {
    const ScanRun run = runs[blockIdx.y];
    const int t0 = blockIdx.x * SC;
    if (t0 >= run.n) return;
    const int L = min(SC, run.n - t0), h = blockIdx.z / colblocks, hk = p.hk, hv = p.hv;
    const int col = (blockIdx.z % colblocks) * 64 + threadIdx.x;
    if (col >= hv + hk) return;
    const int lr0 = (run.pad + blockIdx.x) * SC;
    float *w = in.ws + ((size_t)(run.pad + blockIdx.x) * p.heads + h) * p.chunk_floats;
    const float *bm = w + p.o_bm, *xi = w + p.o_xi, *G = w + p.o_g;
    const bool lwv = p.flags & SF_LWV;
    const int lww = lwv ? hk : 1;
    const float *lw = in.lw + scan_row(p, lr0, h, lww);
    const size_t lws = (size_t)p.heads * lww;
    const bool is_v = col < hv;
    const int j = is_v ? col : col - hv;
    float *y = is_v ? w + p.o_r0 + j : w + p.o_wr + j;
    const int yw = is_v ? hv : hk;
    const float *v = in.v + scan_row(p, lr0, h, hv) + j;
    const size_t vs = (size_t)p.heads * hv;
    const float *av = in.a + scan_row(p, lr0, h, hk) + j;
    const size_t as = (size_t)p.heads * hk;
    for (int t = 0; t < L; t++) {
        float x = 0.f;
        if (is_v) {
            for (int u = 0; u < t; u++) x += xi[t * SC + u] * v[u * vs];
        } else {
            const float gp = lwv ? G[(size_t)t * hk + j] - lw[t * lws + j] : G[t] - lw[t * lws];
            x = av[t * as] * expf(gp);
        }
        for (int u = 0; u < t; u++) x -= bm[t * SC + u] * y[(size_t)u * yw];
        y[(size_t)t * yw] = x;
    }
}

// grid (chunks, runs, heads * ceil(ncols / 64)), block 64; ncols = hk + hv with rank-one terms (the first hk columns correct Qe), else hv.
//   Qe[t][i] -= sum_{s<=t} Qb[t][s] Wr[s][i]
//   O0[t][j]  = c_t v[t][j] + sum_{s<=t} Qk[t][s] v[s][j] - sum_{s<=t} Qb[t][s] R0[s][j]
__global__ void __launch_bounds__(64) scan_out_kernel(ScanP p, ScanIn in, const ScanRun *__restrict__ runs, int colblocks) {
    const ScanRun run = runs[blockIdx.y];
    const int t0 = blockIdx.x * SC;
    if (t0 >= run.n) return;
    const bool r1 = p.flags & SF_RANK1;
    const int L = min(SC, run.n - t0), h = blockIdx.z / colblocks, hk = p.hk, hv = p.hv;
    int col = (blockIdx.z % colblocks) * 64 + threadIdx.x;
    if (col >= (r1 ? hk + hv : hv)) return;
    const int lr0 = (run.pad + blockIdx.x) * SC;
    float *w = in.ws + ((size_t)(run.pad + blockIdx.x) * p.heads + h) * p.chunk_floats;
    const float *qk = w + p.o_qk, *qb = w + p.o_qb;
    if (r1 && col < hk) {
        float *qe = w + p.o_qe + col;
        const float *wr = w + p.o_wr + col;
        for (int t = 0; t < L; t++) {
            float x = qe[(size_t)t * hk];
            for (int s = 0; s <= t; s++) x -= qb[t * SC + s] * wr[(size_t)s * hk];
            qe[(size_t)t * hk] = x;
        }
        return;
    }
    const int j = r1 ? col - hk : col;
    float *o0 = w + p.o_o0 + j;
    const float *r0 = r1 ? w + p.o_r0 + j : nullptr;
    const float *v = in.v + scan_row(p, lr0, h, hv) + j;
    const size_t vs = (size_t)p.heads * hv;
    const bool hc = p.flags & SF_HASC;
    const float *cc = in.c + (size_t)lr0 * p.heads + h;
    const int early = (p.flags & SF_QPREV) ? 1 : 0;
    for (int t = 0; t < L; t++) {
        float x = hc ? cc[(size_t)t * p.heads] * v[t * vs] : 0.f;
        for (int s = 0; s <= t - early; s++) x += qk[t * SC + s] * v[s * vs];
        if (r1)
            for (int s = 0; s <= t; s++) x -= qb[t * SC + s] * r0[(size_t)s * hv];
        o0[(size_t)t * hv] = x;
    }
}

// grid (heads, ceil(hv / 32), runs), block 256: the serial part. The state of 32 value columns lives in shared memory while the block walks the
// chunks of its run:   R = R0 + Wr P,   O = O0 + Qe P,   P = diag(dE) P + Kd^T V - Bd^T R.
__global__ void __launch_bounds__(256) scan_state_kernel(ScanP p, ScanIn in, const ScanRun *__restrict__ runs) {
    __shared__ float P[SMAX_HK * SPS];
    __shared__ float Rm[SC * SPS];
    __shared__ float Vs[SC * SPS];
    const ScanRun run = runs[blockIdx.z];
    const int h = blockIdx.x, c0 = blockIdx.y * SCW, hk = p.hk, hv = p.hv, tid = threadIdx.x;
    const int ncol = min(SCW, hv - c0);
    const bool r1 = p.flags & SF_RANK1;
    float *S = reinterpret_cast<float *>(reinterpret_cast<char *>(run.state) + p.state_off) + (size_t)h * p.hstride;
    for (int e = tid; e < hk * SCW; e += 256) {
        const int i = e / SCW, c = e % SCW;
        P[i * SPS + c] = (run.reset || c >= ncol) ? 0.f : S[(size_t)i * p.kstride + (size_t)(c0 + c) * p.vstride];
    }
    const int nch = (run.n + SC - 1) / SC;
    const int t = tid & 63, cg = tid >> 6;
    for (int ch = 0; ch < nch; ch++) {
        const int L = min(SC, run.n - ch * SC);
        const int lr0 = (run.pad + ch) * SC;
        const float *w = in.ws + ((size_t)(run.pad + ch) * p.heads + h) * p.chunk_floats;
        __syncthreads();  // the state (and the previous chunk's readers of Vs / Rm) are done
        for (int e = tid; e < SC * SCW; e += 256) {
            const int s = e >> 5, c = e & 31;
            Vs[s * SPS + c] = (s < L && c < ncol) ? in.v[scan_row(p, lr0 + s, h, hv) + c0 + c] : 0.f;
        }
        // O and R for token t, columns cg * 8 .. cg * 8 + 7
        float ar[8], ao[8];
#pragma unroll
        for (int k = 0; k < 8; k++) {
            const int c = cg * 8 + k;
            ar[k] = (r1 && c < ncol) ? w[p.o_r0 + (size_t)t * hv + c0 + c] : 0.f;
            ao[k] = (c < ncol) ? w[p.o_o0 + (size_t)t * hv + c0 + c] : 0.f;
        }
        const float *wr = r1 ? w + p.o_wr + (size_t)t * hk : nullptr;
        const float *qe = w + p.o_qe + (size_t)t * hk;
        for (int i = 0; i < hk; i += 4) {
            const float4 q4 = *reinterpret_cast<const float4 *>(qe + i);
            const float4 w4 = r1 ? *reinterpret_cast<const float4 *>(wr + i) : make_float4(0.f, 0.f, 0.f, 0.f);
            const float qv[4] = {q4.x, q4.y, q4.z, q4.w}, wv[4] = {w4.x, w4.y, w4.z, w4.w};
#pragma unroll
            for (int d = 0; d < 4; d++) {
#pragma unroll
                for (int k = 0; k < 8; k++) {
                    const float pv = P[(i + d) * SPS + cg * 8 + k];
                    ao[k] += qv[d] * pv;
                    ar[k] += wv[d] * pv;
                }
            }
        }
#pragma unroll
        for (int k = 0; k < 8; k++) {
            const int c = cg * 8 + k;
            Rm[t * SPS + c] = ar[k];
            if (t < L && c < ncol) in.o[scan_row(p, lr0 + t, h, hv) + c0 + c] = ao[k];
        }
        __syncthreads();
        const float *kd = w + p.o_kd, *bd = w + p.o_bd, *de = w + p.o_de;
        for (int item = tid; item < hk * 4; item += 256) {
            const int i = item % hk, g = item / hk;
            float acc[8];
#pragma unroll
            for (int k = 0; k < 8; k++) acc[k] = P[i * SPS + g * 8 + k] * de[i];
            for (int s = 0; s < L; s++) {
                const float kv = kd[(size_t)s * hk + i];
                const float bv = r1 ? bd[(size_t)s * hk + i] : 0.f;
#pragma unroll
                for (int k = 0; k < 8; k++) acc[k] += kv * Vs[s * SPS + g * 8 + k] - bv * Rm[s * SPS + g * 8 + k];
            }
#pragma unroll
            for (int k = 0; k < 8; k++) P[i * SPS + g * 8 + k] = acc[k];
        }
    }
    __syncthreads();
    if (nch > 0)
        for (int e = tid; e < hk * SCW; e += 256) {
            const int i = e / SCW, c = e % SCW;
            if (c < ncol) S[(size_t)i * p.kstride + (size_t)(c0 + c) * p.vstride] = P[i * SPS + c];
        }
}

static int scan_supported_core(int hk, int hv, int heads) {
    return (hk == 16 || hk == 32 || hk == 64 || hk == 128) && hv >= 1 && hv <= 256 && heads > 0;
}

// Runs the chunk algebra for `nruns` runs of one window (at most `max_chunks` chunks each).
static int scan_core(cudaStream_t s, const ScanP &p, const ScanIn &in, const ScanRun *runs, int nruns, int max_chunks) {
    const dim3 g(max_chunks, nruns, p.heads);
    const bool r1 = p.flags & SF_RANK1;
    scan_cum_kernel<<<g, p.hk, 0, s>>>(p, in, runs);
    if (int e = fllm_check_launch("scan_cum")) return e;
    switch (p.flags & (SF_RANK1 | SF_BSAME | SF_LWV | SF_QPREV)) {
#define SCAN_PAIRS(F) \
    case F: scan_pairs_kernel<F><<<g, 256, 0, s>>>(p, in, runs); break
        SCAN_PAIRS(SF_RANK1 | SF_BSAME);
        SCAN_PAIRS(SF_RANK1 | SF_BSAME | SF_LWV);
        SCAN_PAIRS(SF_RANK1 | SF_LWV);
        SCAN_PAIRS(SF_LWV | SF_QPREV);
        SCAN_PAIRS(0);
#undef SCAN_PAIRS
        default: return fllm_set_error("scan: unsupported flag combination %d", p.flags);
    }
    if (int e = fllm_check_launch("scan_pairs")) return e;
    if (r1) {
        const int cb = (p.hv + p.hk + 63) / 64;
        scan_solve_kernel<<<dim3(max_chunks, nruns, p.heads * cb), 64, 0, s>>>(p, in, runs, cb);
        if (int e = fllm_check_launch("scan_solve")) return e;
    }
    {
        const int cb = ((r1 ? p.hk + p.hv : p.hv) + 63) / 64;
        scan_out_kernel<<<dim3(max_chunks, nruns, p.heads * cb), 64, 0, s>>>(p, in, runs, cb);
        if (int e = fllm_check_launch("scan_out")) return e;
    }
    scan_state_kernel<<<dim3(p.heads, (p.hv + SCW - 1) / SCW, nruns), 256, 0, s>>>(p, in, runs);
    return fllm_check_launch("scan_state");
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// shared helpers of the per-architecture kernels
// ---------------------------------------------------------------------------------------------------------------------------------------

__device__ __forceinline__ float sc_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float sc_silu(float x) { return x / (1.0f + expf(-x)); }

// Sum over the threads of the block through shared memory (`scratch` holds blockDim.x floats; any block size).
__device__ __forceinline__ float sc_block_sum(float v, float *scratch) {
    __syncthreads();
    scratch[threadIdx.x] = v;
    __syncthreads();
    for (int n = (int)blockDim.x; n > 1;) {
        const int half = (n + 1) >> 1;  // any block size: the upper n - half values fold onto the lower ones
        if ((int)threadIdx.x < n - half) scratch[threadIdx.x] += scratch[threadIdx.x + half];
        __syncthreads();
        n = half;
    }
    const float total = scratch[0];
    __syncthreads();
    return total;
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// Gated DeltaNet
// ---------------------------------------------------------------------------------------------------------------------------------------

#define DELTA_MAX_VH 256

// grid (max_n, runs), block 256. P_t = (I - sigma k k^T) g P_{t-1} + sigma k v^T:  w = g, a = g k, b = kp = sigma k.
__global__ void __launch_bounds__(256) scan_delta_prep_kernel(ScanP p, ScanIn in, const ScanRun *__restrict__ runs, const float *__restrict__ mixed,
                                                              const float *__restrict__ beta, const float *__restrict__ alpha,
                                                              const float *__restrict__ dt_bias, const float *__restrict__ decay, int kh, int conv_dim,
                                                              int tiled, float beta_scale) {
    __shared__ float s_lw[DELTA_MAX_VH], s_sig[DELTA_MAX_VH];
    const ScanRun run = runs[blockIdx.y];
    const int t = blockIdx.x;
    if (t >= run.n) return;
    const int vh = p.heads, hk = p.hk, hv = p.hv, row = run.row0 + t, lr = run.pad * SC + t;
    const int key_dim = kh * hk;
    for (int j = threadIdx.x; j < vh; j += blockDim.x) {
        const float raw = alpha[(size_t)row * vh + j] + dt_bias[j];
        const float sp = raw > 20.f ? raw : log1pf(expf(raw));
        const float lw = fmaxf(decay[j] * sp, SLW_MIN);
        s_lw[j] = lw;
        s_sig[j] = beta_scale / (1.0f + expf(-beta[(size_t)row * vh + j]));
        in.lw[(size_t)lr * vh + j] = lw;
    }
    __syncthreads();
    const float *m = mixed + (size_t)row * conv_dim;
    for (int e = threadIdx.x; e < vh * hk; e += blockDim.x) {
        const int j = e / hk, i = e % hk;
        const int src = tiled ? j % kh : j / (vh / kh);
        const float k = m[key_dim + src * hk + i];
        const size_t at = (size_t)lr * vh * hk + e;
        in.q[at] = m[src * hk + i];
        in.kp[at] = k * s_sig[j];
        in.a[at] = k * expf(s_lw[j]);
    }
    for (int e = threadIdx.x; e < vh * hv; e += blockDim.x) in.v[(size_t)lr * vh * hv + e] = m[2 * key_dim + e];
}

// grid (vh, max_n, runs), block hv: the gated RMS norm of the head output.
__global__ void __launch_bounds__(256) scan_delta_post_kernel(ScanP p, ScanIn in, const ScanRun *__restrict__ runs, const float *__restrict__ z,
                                                              const float *__restrict__ norm, float *__restrict__ out, float eps) {
    __shared__ float scratch[256];
    const ScanRun run = runs[blockIdx.z];
    const int t = blockIdx.y;
    if (t >= run.n) return;
    const int j = blockIdx.x, hv = p.hv, c = threadIdx.x, vh = p.heads, value_dim = vh * hv;
    const size_t row = (size_t)run.row0 + t, lr = (size_t)run.pad * SC + t;
    const float acc = in.o[lr * value_dim + (size_t)j * hv + c];
    const float ss = sc_block_sum(acc * acc, scratch);
    const float inv = 1.0f / sqrtf(ss / (float)hv + eps);
    out[row * value_dim + (size_t)j * hv + c] = acc * inv * norm[c] * sc_silu(z[row * value_dim + (size_t)j * hv + c]);
}

static int scan_slots(const ScanRun *runs, int nruns) {
    int slots = 0;
    for (int i = 0; i < nruns; i++) slots += (runs[i].n + SC - 1) / SC;
    return slots;
}

int fllm_scan_delta_supported(int kh, int vh, int hk, int hv) {
    return scan_supported_core(hk, hv, vh) && hv % 32 == 0 && kh > 0 && vh <= DELTA_MAX_VH && vh % kh == 0;
}

// Floats of scratch that a window of `slots` chunk slots needs, per architecture (0: shape not supported).
size_t fllm_scan_delta_floats(int vh, int hk, int hv, int slots) { return scan_plan(nullptr, hk, hv, vh, SF_RANK1 | SF_BSAME, slots).floats; }

// `d_runs`: the window's runs on the device (row0 = first row of the window, n <= 64 * max_chunks, pad = first chunk slot), `h_runs` the same on the host.
int fllm_scan_delta_launch(cudaStream_t s, float *ws, const FllmRun *h_runs, const void *d_runs, int nruns, int max_chunks, int kh, int vh, int hk, int hv,
                           int tiled, float beta_scale, float eps, size_t state_off, const float *mixed, const float *z, const float *beta,
                           const float *alpha, const float *dt_bias, const float *decay, const float *norm, float *out) {
    const int slots = scan_slots(h_runs, nruns);
    ScanPlan sp = scan_plan(ws, hk, hv, vh, SF_RANK1 | SF_BSAME, slots);
    sp.p.state_off = state_off;
    sp.p.hstride = (size_t)hk * hv;
    sp.p.kstride = hv;
    sp.p.vstride = 1;
    const ScanRun *r = static_cast<const ScanRun *>(d_runs);
    const int conv_dim = 2 * kh * hk + vh * hv;
    scan_delta_prep_kernel<<<dim3(max_chunks * SC, nruns), 256, 0, s>>>(sp.p, sp.in, r, mixed, beta, alpha, dt_bias, decay, kh, conv_dim, tiled, beta_scale);
    if (int e = fllm_check_launch("scan_delta_prep")) return e;
    if (int e = scan_core(s, sp.p, sp.in, r, nruns, max_chunks)) return e;
    scan_delta_post_kernel<<<dim3(vh, max_chunks * SC, nruns), hv, 0, s>>>(sp.p, sp.in, r, z, norm, out, eps);
    return fllm_check_launch("scan_delta_post");
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// Kimi Delta Attention: the delta rule with a decay per key channel (head_dim = hk = hv)
// ---------------------------------------------------------------------------------------------------------------------------------------

// grid (max_n, runs), block 256. w = g (a vector), a = g * k, b = kp = sigma k.
__global__ void __launch_bounds__(256) scan_kda_prep_kernel(ScanP p, ScanIn in, const ScanRun *__restrict__ runs, const float *__restrict__ mixed,
                                                            const float *__restrict__ forget, const float *__restrict__ beta,
                                                            const float *__restrict__ dt_bias, const float *__restrict__ rate, int conv_dim, int bounded,
                                                            float lower_bound) {
    const ScanRun run = runs[blockIdx.y];
    const int t = blockIdx.x;
    if (t >= run.n) return;
    const int heads = p.heads, hd = p.hk, key_dim = heads * hd;
    const size_t row = (size_t)run.row0 + t, lr = (size_t)run.pad * SC + t;
    const float *m = mixed + row * conv_dim;
    for (int e = threadIdx.x; e < key_dim; e += blockDim.x) {
        const int h = e / hd;
        const float raw = forget[row * key_dim + e] + dt_bias[e];
        float g;
        if (bounded) {
            g = lower_bound / (1.0f + expf(-rate[h] * raw));
        } else {
            const float sp = raw > 20.f ? raw : log1pf(expf(raw));
            g = -rate[h] * sp;
        }
        g = fmaxf(g, SLW_MIN);
        const float sigma = 1.0f / (1.0f + expf(-beta[row * heads + h]));
        const float k = m[key_dim + e];
        const size_t at = lr * key_dim + e;
        in.lw[at] = g;
        in.q[at] = m[e];
        in.kp[at] = k * sigma;
        in.a[at] = k * expf(g);
        in.v[at] = m[2 * key_dim + e];
    }
}

// grid (heads, max_n, runs), block head_dim: RMS norm of the head output times sigmoid(gate).
__global__ void __launch_bounds__(256) scan_kda_post_kernel(ScanP p, ScanIn in, const ScanRun *__restrict__ runs, const float *__restrict__ gate,
                                                            const float *__restrict__ norm, float *__restrict__ out, float eps) {
    __shared__ float scratch[256];
    const ScanRun run = runs[blockIdx.z];
    const int t = blockIdx.y;
    if (t >= run.n) return;
    const int h = blockIdx.x, hd = p.hk, c = threadIdx.x, key_dim = p.heads * hd;
    const size_t row = (size_t)run.row0 + t, lr = (size_t)run.pad * SC + t;
    const float acc = in.o[lr * key_dim + (size_t)h * hd + c];
    const float ss = sc_block_sum(acc * acc, scratch);
    const float inv = 1.0f / sqrtf(ss / (float)hd + eps);
    out[row * key_dim + (size_t)h * hd + c] = acc * inv * norm[c] * sc_sigmoid(gate[row * key_dim + (size_t)h * hd + c]);
}

#define KDA_FLAGS (SF_RANK1 | SF_BSAME | SF_LWV)

int fllm_scan_kda_supported(int heads, int head_dim) { return scan_supported_core(head_dim, head_dim, heads); }
size_t fllm_scan_kda_floats(int heads, int head_dim, int slots) { return scan_plan(nullptr, head_dim, head_dim, heads, KDA_FLAGS, slots).floats; }

int fllm_scan_kda_launch(cudaStream_t s, float *ws, const FllmRun *h_runs, const void *d_runs, int nruns, int max_chunks, int heads, int head_dim, int bounded,
                         float lower_bound, float eps, size_t state_off, const float *mixed, const float *forget, const float *beta, const float *gate,
                         const float *dt_bias, const float *rate, const float *norm, float *out) {
    ScanPlan sp = scan_plan(ws, head_dim, head_dim, heads, KDA_FLAGS, scan_slots(h_runs, nruns));
    sp.p.state_off = state_off;
    sp.p.hstride = (size_t)head_dim * head_dim;
    sp.p.kstride = head_dim;
    sp.p.vstride = 1;
    const ScanRun *r = static_cast<const ScanRun *>(d_runs);
    scan_kda_prep_kernel<<<dim3(max_chunks * SC, nruns), 256, 0, s>>>(sp.p, sp.in, r, mixed, forget, beta, dt_bias, rate, 3 * heads * head_dim, bounded,
                                                                      lower_bound);
    if (int e = fllm_check_launch("scan_kda_prep")) return e;
    if (int e = scan_core(s, sp.p, sp.in, r, nruns, max_chunks)) return e;
    scan_kda_post_kernel<<<dim3(heads, max_chunks * SC, nruns), head_dim, 0, s>>>(sp.p, sp.in, r, gate, norm, out, eps);
    return fllm_check_launch("scan_kda_post");
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// RWKV-7 and RWKV-6 (head size N = hk = hv = block size)
// ---------------------------------------------------------------------------------------------------------------------------------------

// grid (heads, runs, max_n), block N. P_t = (diag(w) - (kk a) kk^T) P + k v^T.
__global__ void __launch_bounds__(128) scan_rwkv7_prep_kernel(ScanP p, ScanIn in, const ScanRun *__restrict__ runs, const float *__restrict__ rr,
                                                              const float *__restrict__ kk_in, const float *__restrict__ vv, const float *__restrict__ ww,
                                                              const float *__restrict__ aa, const float *__restrict__ vres, const float *__restrict__ k_k,
                                                              const float *__restrict__ k_a, const float *__restrict__ r_k, float *__restrict__ v_first,
                                                              int hidden) {
    __shared__ float red[128];
    const ScanRun run = runs[blockIdx.y];
    const int t = blockIdx.z;
    if (t >= run.n) return;
    const int h = blockIdx.x, i = threadIdx.x, N = p.hk, ch = h * N + i;
    const size_t at = ((size_t)run.row0 + t) * hidden + ch, lr = (size_t)run.pad * SC + t, la = lr * hidden + ch;
    const float a = sc_sigmoid(aa[at]);
    const float lw = -0.6065306597126334f * sc_sigmoid(ww[at]);
    const float k_in = kk_in[at];
    float v = vv[at];
    if (vres) {
        v += (v_first[at] - v) * sc_sigmoid(vres[at]);
    } else {
        v_first[at] = v;
    }
    float kk = k_in * k_k[ch];
    const float ss = sc_block_sum(kk * kk, red);
    kk *= 1.0f / fmaxf(sqrtf(ss), 1e-12f);
    const float k = k_in * (1.0f + (a - 1.0f) * k_a[ch]);
    const float r = rr[at];
    const float bonus = sc_block_sum(r * k * r_k[ch], red);
    in.q[la] = r;
    in.kp[la] = k;
    in.a[la] = kk;
    in.b[la] = kk * a;
    in.lw[la] = lw;
    in.v[la] = v;
    if (i == 0) in.c[lr * p.heads + h] = bonus;
}

// grid (heads, runs, max_n), block N: group norm of the head output, the r k r_k bonus and the output gate.
__global__ void __launch_bounds__(128) scan_rwkv7_post_kernel(ScanP p, ScanIn in, const ScanRun *__restrict__ runs, const float *__restrict__ gate,
                                                              const float *__restrict__ norm_w, const float *__restrict__ norm_b, float *__restrict__ out,
                                                              int hidden, float eps) {
    __shared__ float red[128];
    const ScanRun run = runs[blockIdx.y];
    const int t = blockIdx.z;
    if (t >= run.n) return;
    const int h = blockIdx.x, i = threadIdx.x, N = p.hk, ch = h * N + i;
    const size_t at = ((size_t)run.row0 + t) * hidden + ch, lr = (size_t)run.pad * SC + t, la = lr * hidden + ch;
    const float acc = in.o[la];
    const float mean = sc_block_sum(acc, red) / (float)N;
    const float d = acc - mean;
    const float var = sc_block_sum(d * d, red) / (float)N;
    const float normed = d * (1.0f / sqrtf(var + eps)) * norm_w[ch] + norm_b[ch];
    out[at] = (normed + in.c[lr * p.heads + h] * in.v[la]) * gate[at];
}

// grid (heads, runs, max_n), block N. P_t = diag(exp(-exp(w))) P + k v^T, readout of the state before the token plus the u bonus.
__global__ void __launch_bounds__(128) scan_rwkv6_prep_kernel(ScanP p, ScanIn in, const ScanRun *__restrict__ runs, const float *__restrict__ rr,
                                                              const float *__restrict__ kk, const float *__restrict__ vv, const float *__restrict__ ww,
                                                              const float *__restrict__ first, int hidden) {
    __shared__ float red[128];
    const ScanRun run = runs[blockIdx.y];
    const int t = blockIdx.z;
    if (t >= run.n) return;
    const int h = blockIdx.x, i = threadIdx.x, N = p.hk, ch = h * N + i;
    const size_t at = ((size_t)run.row0 + t) * hidden + ch, lr = (size_t)run.pad * SC + t, la = lr * hidden + ch;
    const float r = rr[at], k = kk[at];
    const float bonus = sc_block_sum(r * first[ch] * k, red);
    in.q[la] = r;
    in.kp[la] = k;
    in.lw[la] = fmaxf(-expf(ww[at]), SLW_MIN);
    in.v[la] = vv[at];
    if (i == 0) in.c[lr * p.heads + h] = bonus;
}

__global__ void __launch_bounds__(128) scan_rwkv6_post_kernel(ScanP p, ScanIn in, const ScanRun *__restrict__ runs, const float *__restrict__ gate,
                                                              const float *__restrict__ norm_w, const float *__restrict__ norm_b, float *__restrict__ out,
                                                              int hidden, float eps) {
    __shared__ float red[128];
    const ScanRun run = runs[blockIdx.y];
    const int t = blockIdx.z;
    if (t >= run.n) return;
    const int h = blockIdx.x, i = threadIdx.x, N = p.hk, ch = h * N + i;
    const size_t at = ((size_t)run.row0 + t) * hidden + ch, lr = (size_t)run.pad * SC + t, la = lr * hidden + ch;
    const float acc = in.o[la];
    const float mean = sc_block_sum(acc, red) / (float)N;
    const float d = acc - mean;
    const float var = sc_block_sum(d * d, red) / (float)N;
    out[at] = (d * (1.0f / sqrtf(var + eps)) * norm_w[ch] + norm_b[ch]) * gate[at];
}

#define RWKV7_FLAGS (SF_RANK1 | SF_LWV)
#define RWKV6_FLAGS (SF_LWV | SF_QPREV | SF_HASC)

int fllm_scan_rwkv_supported(int hidden, int head_size) {
    return (head_size == 16 || head_size == 32 || head_size == 64 || head_size == 128) && hidden > 0 && hidden % head_size == 0;
}
size_t fllm_scan_rwkv7_floats(int heads, int n, int slots) { return scan_plan(nullptr, n, n, heads, RWKV7_FLAGS, slots).floats; }
size_t fllm_scan_rwkv6_floats(int heads, int n, int slots) { return scan_plan(nullptr, n, n, heads, RWKV6_FLAGS, slots).floats; }

// State of RWKV-7: S[h][value i][key j]; of RWKV-6: S[h][key j][value i].
int fllm_scan_rwkv7_launch(cudaStream_t s, float *ws, const FllmRun *h_runs, const void *d_runs, int nruns, int max_chunks, int heads, int n, int hidden,
                           float eps, size_t state_off, const float *r, const float *k, const float *v, const float *w, const float *a, const float *vres,
                           const float *gate, const float *norm_w, const float *norm_b, const float *k_k, const float *k_a, const float *r_k, float *v_first,
                           float *out) {
    ScanPlan sp = scan_plan(ws, n, n, heads, RWKV7_FLAGS, scan_slots(h_runs, nruns));
    sp.p.state_off = state_off;
    sp.p.hstride = (size_t)n * n;
    sp.p.kstride = 1;
    sp.p.vstride = n;
    const ScanRun *rn = static_cast<const ScanRun *>(d_runs);
    const dim3 grid(heads, nruns, max_chunks * SC);
    scan_rwkv7_prep_kernel<<<grid, n, 0, s>>>(sp.p, sp.in, rn, r, k, v, w, a, vres, k_k, k_a, r_k, v_first, hidden);
    if (int e = fllm_check_launch("scan_rwkv7_prep")) return e;
    if (int e = scan_core(s, sp.p, sp.in, rn, nruns, max_chunks)) return e;
    scan_rwkv7_post_kernel<<<grid, n, 0, s>>>(sp.p, sp.in, rn, gate, norm_w, norm_b, out, hidden, eps);
    return fllm_check_launch("scan_rwkv7_post");
}

int fllm_scan_rwkv6_launch(cudaStream_t s, float *ws, const FllmRun *h_runs, const void *d_runs, int nruns, int max_chunks, int heads, int n, int hidden,
                           float eps, size_t state_off, const float *r, const float *k, const float *v, const float *w, const float *gate,
                           const float *norm_w, const float *norm_b, const float *first, float *out) {
    ScanPlan sp = scan_plan(ws, n, n, heads, RWKV6_FLAGS, scan_slots(h_runs, nruns));
    sp.p.state_off = state_off;
    sp.p.hstride = (size_t)n * n;
    sp.p.kstride = n;
    sp.p.vstride = 1;
    const ScanRun *rn = static_cast<const ScanRun *>(d_runs);
    const dim3 grid(heads, nruns, max_chunks * SC);
    scan_rwkv6_prep_kernel<<<grid, n, 0, s>>>(sp.p, sp.in, rn, r, k, v, w, first, hidden);
    if (int e = fllm_check_launch("scan_rwkv6_prep")) return e;
    if (int e = scan_core(s, sp.p, sp.in, rn, nruns, max_chunks)) return e;
    scan_rwkv6_post_kernel<<<grid, n, 0, s>>>(sp.p, sp.in, rn, gate, norm_w, norm_b, out, hidden, eps);
    return fllm_check_launch("scan_rwkv6_post");
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// Mamba-2: P[n][p] (state n, head channel p), scalar decay exp(dt A) per head and token, B and C shared by the heads of a group
// ---------------------------------------------------------------------------------------------------------------------------------------

#define MAMBA_MAX_HEADS 512

// grid (max_n, runs), block 256. `proj` rows are `width` wide with the dt logits at column dt_off; `mixed` rows are [x | B | C] (conv_dim wide).
__global__ void __launch_bounds__(256) scan_mamba2_prep_kernel(ScanP p, ScanIn in, const ScanRun *__restrict__ runs, const float *__restrict__ proj, int width,
                                                               int dt_off, const float *__restrict__ mixed, const float *__restrict__ dt_bias,
                                                               const float *__restrict__ decay, const float *__restrict__ skip, int inner, int conv_dim,
                                                               int groups, float step_min) {
    __shared__ float s_dt[MAMBA_MAX_HEADS];
    const ScanRun run = runs[blockIdx.y];
    const int t = blockIdx.x;
    if (t >= run.n) return;
    const int heads = p.heads, N = p.hk;
    const size_t row = (size_t)run.row0 + t, lr = (size_t)run.pad * SC + t;
    for (int h = threadIdx.x; h < heads; h += blockDim.x) {
        const float raw = proj[row * width + dt_off + h] + dt_bias[h];
        float dt = raw > 20.f ? raw : log1pf(expf(raw));
        dt = fmaxf(dt, step_min);
        s_dt[h] = dt;
        in.lw[lr * heads + h] = fmaxf(dt * decay[h], SLW_MIN);
        in.c[lr * heads + h] = skip[h];
    }
    __syncthreads();
    const float *m = mixed + row * conv_dim;
    for (int e = threadIdx.x; e < heads * N; e += blockDim.x) {
        const int h = e / N, n = e % N, g = h / (heads / groups);
        in.q[lr * heads * N + e] = m[inner + groups * N + g * N + n];
        in.kp[lr * heads * N + e] = s_dt[h] * m[inner + g * N + n];
    }
    for (int e = threadIdx.x; e < inner; e += blockDim.x) in.v[lr * inner + e] = m[e];
}

// grid (max_n, runs), block 256: y = O (the skip term D x is already in it)
__global__ void __launch_bounds__(256) scan_copy_out_kernel(ScanP p, ScanIn in, const ScanRun *__restrict__ runs, float *__restrict__ out, int width) {
    const ScanRun run = runs[blockIdx.y];
    const int t = blockIdx.x;
    if (t >= run.n) return;
    const size_t row = (size_t)run.row0 + t, lr = (size_t)run.pad * SC + t;
    for (int e = threadIdx.x; e < width; e += blockDim.x) out[row * width + e] = in.o[lr * width + e];
}

#define MAMBA2_FLAGS (SF_HASC)

int fllm_scan_mamba2_supported(int inner, int state, int heads, int groups) {
    return heads > 0 && heads <= MAMBA_MAX_HEADS && groups > 0 && heads % groups == 0 && inner % heads == 0 && scan_supported_core(state, inner / heads, heads);
}
size_t fllm_scan_mamba2_floats(int inner, int state, int heads, int slots) { return scan_plan(nullptr, state, inner / heads, heads, MAMBA2_FLAGS, slots).floats; }

int fllm_scan_mamba2_launch(cudaStream_t s, float *ws, const FllmRun *h_runs, const void *d_runs, int nruns, int max_chunks, int inner, int state, int heads,
                            int groups, float step_min, size_t state_off, const float *proj, int width, int dt_off, const float *mixed, int conv_dim,
                            const float *dt_bias, const float *decay, const float *skip, float *out) {
    const int hd = inner / heads;
    ScanPlan sp = scan_plan(ws, state, hd, heads, MAMBA2_FLAGS, scan_slots(h_runs, nruns));
    sp.p.state_off = state_off;
    sp.p.hstride = (size_t)hd * state;
    sp.p.kstride = 1;
    sp.p.vstride = state;
    const ScanRun *r = static_cast<const ScanRun *>(d_runs);
    scan_mamba2_prep_kernel<<<dim3(max_chunks * SC, nruns), 256, 0, s>>>(sp.p, sp.in, r, proj, width, dt_off, mixed, dt_bias, decay, skip, inner, conv_dim,
                                                                         groups, step_min);
    if (int e = fllm_check_launch("scan_mamba2_prep")) return e;
    if (int e = scan_core(s, sp.p, sp.in, r, nruns, max_chunks)) return e;
    scan_copy_out_kernel<<<dim3(max_chunks * SC, nruns), 256, 0, s>>>(sp.p, sp.in, r, out, inner);
    return fllm_check_launch("scan_copy_out");
}
