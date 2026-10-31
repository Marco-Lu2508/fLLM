// fLLM CUDA / HIP backend: the generic part of the chunk-parallel scan (included by scan.cu).
//
// Per head the recurrence is  P_t = (diag(w_t) - b_t a_t^T) P_{t-1} + k_t v_t^T,  o_t = P_t^T q_t + c_t v_t  (see scan.cu and
// rust/engine/src/chunked_scan.rs, which proves the chunked form used here equal to it). For a chunk of 64 tokens everything that does not depend on the
// incoming state is computed for all chunks at once, as small matrix products on shared-memory tiles:
//
//   scan_cum    G (cumulative log decay), AG = a exp(Gp), Kd = kp exp(Gend - G), Bd = b exp(Gend - G), Qe = q exp(G)           [per channel]
//   scan_pairs  Bm, Xi, Qk, Qb: the 64 x 64 matrices of pairwise decayed products
//   scan_tm     T = (I + Bm)^-1, M^T (M = T Xi), W2 = Qb T, N = Qk - Qb M                                                           [rank-one terms only]
//   scan_wide   Wr = T AG,  Qe -= W2 AG,  Kd -= M^T Bd,  O1 = N V + c V                                                      (64 x 64 products per column tile)
//   scan_pm     Pc = Kd^T V,  Mst = diag(dE) - Bd^T Wr                                                                       (state-sized products)
//
// and one block per (head, 32 value columns) walks the chunks of its run in order:   O = O1 + Qe P,   P = Mst P + Pc
// (without rank-one terms Mst is diagonal and P = dE P + Pc). Everything is FP32 on shared-memory tiles without warp primitives.
#pragma once

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
    size_t o_g, o_bm, o_xi, o_qk, o_qb, o_tt, o_mt, o_nn, o_w2, o_ag, o_qe, o_wr, o_kd, o_bd, o_o1, o_pc, o_mst, o_de;
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
    const bool r1 = flags & SF_RANK1, bs = r1 && (flags & SF_BSAME);
    size_t off = 0;
    auto take = [&](size_t n) {
        const size_t o = off;
        off += up4(n);
        return o;
    };
    // chunk workspace; with b = kp the matrices Xi and Qb are Bm and Qk
    size_t c = 0;
    auto part = [&](size_t n) {
        const size_t o = c;
        c += up4(n);
        return o;
    };
    const size_t m64 = r1 ? SC * SC : 0, rk = r1 ? (size_t)SC * hk : 0;
    p.o_g = part((size_t)SC * p.gw);
    p.o_bm = part(m64);
    p.o_xi = bs ? p.o_bm : part(m64);
    p.o_qk = part(SC * SC);
    p.o_qb = bs ? p.o_qk : part(m64);
    p.o_tt = part(m64);
    p.o_mt = part(m64);
    p.o_nn = part(m64);
    p.o_w2 = part(m64);
    p.o_ag = part(rk);
    p.o_qe = part((size_t)SC * hk);
    p.o_wr = part(rk);
    p.o_kd = part((size_t)SC * hk);
    p.o_bd = part(rk);
    p.o_o1 = part((size_t)SC * hv);
    p.o_pc = part((size_t)hk * hv);
    p.o_mst = part(r1 ? (size_t)hk * hk : 0);
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

__device__ __forceinline__ size_t scan_row(const ScanP &p, int lr, int h, int width) { return ((size_t)lr * p.heads + h) * width; }

// ---------------------------------------------------------------------------------------------------------------------------------------
// 64 x 64 tile products on shared memory (256 threads)
// ---------------------------------------------------------------------------------------------------------------------------------------

// dst[k][n] = src[k * ld + n] for k < rows, n < cols (else 0); a 64 x 64 tile, lanes along n.
__device__ __forceinline__ void tile_rows(float *dst, const float *src, size_t ld, int rows, int cols) {
    for (int e = threadIdx.x; e < 4096; e += 256) {
        const int k = e >> 6, n = e & 63;
        dst[e] = (k < rows && n < cols) ? src[(size_t)k * ld + n] : 0.f;
    }
}

// dst[k][m] = src[m * ld + k] for m < rows, k < cols (else 0): the transpose of a row-major tile, lanes along m.
__device__ __forceinline__ void tile_trans(float *dst, const float *src, size_t ld, int rows, int cols) {
    for (int e = threadIdx.x; e < 4096; e += 256) {
        const int k = e >> 6, m = e & 63;
        dst[e] = (m < rows && k < cols) ? src[(size_t)m * ld + k] : 0.f;
    }
}

// acc[i][j] += sum_k As[k][ty * 4 + i] * Bs[k][tx * 4 + j] with (ty, tx) = (tid >> 4, tid & 15): As and Bs are [K][64] tiles.
__device__ __forceinline__ void tile_gemm(const float *As, const float *Bs, float acc[4][4]) {
    const int ty = threadIdx.x >> 4, tx = threadIdx.x & 15;
#pragma unroll 8
    for (int k = 0; k < 64; k++) {
        const float4 a = *reinterpret_cast<const float4 *>(As + k * 64 + ty * 4);
        const float4 b = *reinterpret_cast<const float4 *>(Bs + k * 64 + tx * 4);
        const float av[4] = {a.x, a.y, a.z, a.w}, bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) acc[i][j] += av[i] * bv[j];
    }
}

// acc[i][j] += sum_k A[(ty * 4 + i) * lda + k] * Bs[k][tx * 4 + j]: A is a row-major matrix in global memory (rows ty * 4 .. +3 of this thread, four
// consecutive k per load; the 16 threads of a row group read the same addresses), Bs a [64][64] tile in shared memory.
__device__ __forceinline__ void tile_gemm_ga(const float *A, size_t lda, const float *Bs, float acc[4][4]) {
    const int ty = threadIdx.x >> 4, tx = threadIdx.x & 15;
    const float *a0 = A + (size_t)(ty * 4) * lda;
#pragma unroll 4
    for (int k = 0; k < 64; k += 4) {
        float4 a[4];
#pragma unroll
        for (int i = 0; i < 4; i++) a[i] = *reinterpret_cast<const float4 *>(a0 + (size_t)i * lda + k);
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            const float4 b = *reinterpret_cast<const float4 *>(Bs + (k + kk) * 64 + tx * 4);
            const float bv[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const float ai = kk == 0 ? a[i].x : kk == 1 ? a[i].y : kk == 2 ? a[i].z : a[i].w;
#pragma unroll
                for (int j = 0; j < 4; j++) acc[i][j] += ai * bv[j];
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// the kernels
// ---------------------------------------------------------------------------------------------------------------------------------------

// grid (chunks, runs, heads), block hk. Cumulative log decay G of the chunk and the decay-scaled factors (see the table above).
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
    float gend;
    if (lwv) {
        float acc = 0.f;
#pragma unroll 8
        for (int t = 0; t < SC; t++) {
            if (t < L) acc += lw[t * lws + i];
            G[(size_t)t * hk + i] = acc;
        }
        gend = acc;
    } else {
        __shared__ float sg[SC];
        if (i == 0) {
            float acc = 0.f;
            for (int t = 0; t < SC; t++) {
                if (t < L) acc += lw[t * lws];
                sg[t] = acc;
                G[t] = acc;
            }
        }
        __syncthreads();
        gend = sg[L - 1];
    }
    (w + p.o_de)[i] = expf(gend);
    const float *q = in.q + scan_row(p, lr0, h, hk), *kp = in.kp + scan_row(p, lr0, h, hk);
    const float *a = r1 ? in.a + scan_row(p, lr0, h, hk) : nullptr, *b = r1 ? in.b + scan_row(p, lr0, h, hk) : nullptr;
    const size_t rs = (size_t)p.heads * hk;
    float *kd = w + p.o_kd, *bd = w + p.o_bd, *ag = w + p.o_ag, *qe = w + p.o_qe;
#pragma unroll 4
    for (int t = 0; t < L; t++) {
        const float gt = lwv ? G[(size_t)t * hk + i] : G[t];
        const float lwt = lw[t * lws + (lwv ? i : 0)];
        const float dec = expf(gend - gt);
        kd[(size_t)t * hk + i] = kp[t * rs + i] * dec;
        if (r1) {
            bd[(size_t)t * hk + i] = b[t * rs + i] * dec;
            ag[(size_t)t * hk + i] = a[t * rs + i] * expf(gt - lwt);
        }
        qe[(size_t)t * hk + i] = q[t * rs + i] * expf(qp ? gt - lwt : gt);
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
                        aq[x][y] += qt[x] * ku[y] * e2;
                        if (R1) {
                            if (!BS) aqb[x][y] += qt[x] * bu[y] * e2;
                            const float e1 = expf(gpt[x] - gu[y]);
                            ab[x][y] += at[x] * bu[y] * e1;
                            if (!BS) ax[x][y] += at[x] * ku[y] * e1;
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
                if (!BS) {
                    xi[t * SC + u] = (live && u < t) ? ax[x][y] * sc1 : 0.f;
                    qb[t * SC + u] = (live && u <= t) ? aqb[x][y] * sc2 : 0.f;
                }
            }
        }
}

// grid (chunks, runs, heads), block 256, only with rank-one terms: T = (I + Bm)^-1 (forward substitution, one column per thread), M = T Xi, W2 = Qb T,
// N = Qk - Qb M, all 64 x 64, written to the workspace (the matrices of a short last chunk are identity / zero outside the live rows); also M^T.
__global__ void __launch_bounds__(256) scan_tm_kernel(ScanP p, ScanIn in, const ScanRun *__restrict__ runs) {
    __shared__ float X[4096], Y[4096];
    const ScanRun run = runs[blockIdx.y];
    const int t0 = blockIdx.x * SC;
    if (t0 >= run.n) return;
    const int h = blockIdx.z, tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;
    float *w = in.ws + ((size_t)(run.pad + blockIdx.x) * p.heads + h) * p.chunk_floats;
    const float *bm = w + p.o_bm, *xi = w + p.o_xi, *qk = w + p.o_qk, *qb = w + p.o_qb;
    float *tt = w + p.o_tt, *mt = w + p.o_mt, *nn = w + p.o_nn, *w2 = w + p.o_w2;
    tile_rows(X, bm, 64, 64, 64);
    __syncthreads();
    if (tid < 64) {  // column `tid` of T into Y[t][tid]
        for (int t = 0; t < SC; t++) {
            float acc = (t == tid) ? 1.f : 0.f;
            for (int u = tid; u < t; u++) acc -= X[t * 64 + u] * Y[u * 64 + tid];
            Y[t * 64 + tid] = t >= tid ? acc : 0.f;
        }
    }
    __syncthreads();
    for (int e = tid; e < 4096; e += 256) tt[e] = Y[e];
    tile_rows(X, xi, 64, 64, 64);
    __syncthreads();
    float am[4][4] = {}, a2[4][4] = {}, an[4][4] = {};
    tile_gemm_ga(tt, 64, X, am);  // M = T Xi (T read back from the workspace: written above by other threads of this block)
    __syncthreads();
    tile_rows(X, tt, 64, 64, 64);  // B = T, row-major
    __syncthreads();
    tile_gemm_ga(qb, 64, X, a2);  // W2 = Qb T
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; i++) {
        *reinterpret_cast<float4 *>(X + (ty * 4 + i) * 64 + tx * 4) = make_float4(am[i][0], am[i][1], am[i][2], am[i][3]);
        *reinterpret_cast<float4 *>(w2 + (ty * 4 + i) * 64 + tx * 4) = make_float4(a2[i][0], a2[i][1], a2[i][2], a2[i][3]);
#pragma unroll
        for (int j = 0; j < 4; j++) mt[(size_t)(tx * 4 + j) * 64 + ty * 4 + i] = am[i][j];  // M^T
    }
    __syncthreads();
    tile_gemm_ga(qb, 64, X, an);  // Qb M
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const float4 k4 = *reinterpret_cast<const float4 *>(qk + (ty * 4 + i) * 64 + tx * 4);
        *reinterpret_cast<float4 *>(nn + (ty * 4 + i) * 64 + tx * 4) = make_float4(k4.x - an[i][0], k4.y - an[i][1], k4.z - an[i][2], k4.w - an[i][3]);
    }
}

// grid (chunks, runs, heads * tiles), block 256: tile `z % tiles` (columns n0 .. n0 + 63) of one of the products
//   op 0  Wr = T AG        op 1  Qe -= W2 AG        op 2  Kd -= M^T Bd        (rank-one terms; hk columns each)
//   op 3  O1 = N V + c V   (N = Qk without rank-one terms; hv columns)
// `tk` = tiles over hk, `tv` = tiles over hv; the ops 0 .. 2 come first, tk tiles each.
__global__ void __launch_bounds__(256) scan_wide_kernel(ScanP p, ScanIn in, const ScanRun *__restrict__ runs, int tk, int tv) {
    __shared__ float X[4096];
    const ScanRun run = runs[blockIdx.y];
    const int t0 = blockIdx.x * SC;
    if (t0 >= run.n) return;
    const bool r1 = p.flags & SF_RANK1;
    const int tiles = r1 ? 3 * tk + tv : tv;
    const int h = blockIdx.z / tiles;
    int z = blockIdx.z % tiles;
    const int L = min(SC, run.n - t0), hk = p.hk, hv = p.hv, tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;
    const int lr0 = (run.pad + blockIdx.x) * SC;
    float *w = in.ws + ((size_t)(run.pad + blockIdx.x) * p.heads + h) * p.chunk_floats;
    int op = 3;
    if (r1 && z < 3 * tk) {
        op = z / tk;
        z -= op * tk;
    } else if (r1) {
        z -= 3 * tk;
    }
    const int n0 = z * 64;
    const int cols = (op == 3 ? hv : hk) - n0;
    float *out;
    const float *a_src, *b_src;
    switch (op) {
        case 0: a_src = w + p.o_tt; b_src = w + p.o_ag; out = w + p.o_wr; break;
        case 1: a_src = w + p.o_w2; b_src = w + p.o_ag; out = w + p.o_qe; break;
        case 2: a_src = w + p.o_mt; b_src = w + p.o_bd; out = w + p.o_kd; break;  // Mt[m][k] = M[k][m]
        default: a_src = r1 ? w + p.o_nn : w + p.o_qk; b_src = in.v + scan_row(p, lr0, h, hv); out = w + p.o_o1; break;
    }
    const size_t b_ld = op == 3 ? (size_t)p.heads * hv : (size_t)hk;
    tile_rows(X, b_src + n0, b_ld, L, cols);
    __syncthreads();
    float acc[4][4] = {};
    tile_gemm_ga(a_src, 64, X, acc);
    const int width = op == 3 ? hv : hk;
    const bool hc = op == 3 && (p.flags & SF_HASC);
    const float *cc = in.c + (size_t)lr0 * p.heads + h;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int m = ty * 4 + i;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int n = n0 + tx * 4 + j;
            if (n >= width) continue;
            float v = acc[i][j];
            if (op == 0) {
                out[(size_t)m * width + n] = v;
            } else if (op == 3) {
                if (hc && m < L) v += cc[(size_t)m * p.heads] * b_src[(size_t)m * b_ld + n];
                out[(size_t)m * width + n] = v;
            } else {
                out[(size_t)m * width + n] -= v;
            }
        }
    }
}

// grid (chunks, runs, heads * (pc tiles + mst tiles)), block 256: the state-sized products of a chunk,
//   Pc[i][j] = sum_t Kd[t][i] V[t][j]          Mst[i][i'] = [i == i'] dE[i] - sum_t Bd[t][i] Wr[t][i']  (rank-one terms only)
__global__ void __launch_bounds__(256) scan_pm_kernel(ScanP p, ScanIn in, const ScanRun *__restrict__ runs) {
    __shared__ float X[4096], Y[4096];
    const ScanRun run = runs[blockIdx.y];
    const int t0 = blockIdx.x * SC;
    if (t0 >= run.n) return;
    const bool r1 = p.flags & SF_RANK1;
    const int hk = p.hk, hv = p.hv, tk = (hk + 63) / 64, tv = (hv + 63) / 64;
    const int npc = tk * tv, tiles = npc + (r1 ? tk * tk : 0);
    const int h = blockIdx.z / tiles, z = blockIdx.z % tiles;
    const int L = min(SC, run.n - t0), tid = threadIdx.x, ty = tid >> 4, tx = tid & 15;
    const int lr0 = (run.pad + blockIdx.x) * SC;
    float *w = in.ws + ((size_t)(run.pad + blockIdx.x) * p.heads + h) * p.chunk_floats;
    const bool pc = z < npc;
    const int zi = pc ? z / tv : (z - npc) / tk, zj = pc ? z % tv : (z - npc) % tk;
    const int i0 = zi * 64, j0 = zj * 64;
    tile_rows(Y, (pc ? w + p.o_kd : w + p.o_bd) + i0, hk, L, hk - i0);
    if (pc) tile_rows(X, in.v + scan_row(p, lr0, h, hv) + j0, (size_t)p.heads * hv, L, hv - j0);
    else tile_rows(X, w + p.o_wr + j0, hk, L, hk - j0);
    __syncthreads();
    float acc[4][4] = {};
    tile_gemm(Y, X, acc);
    const int width = pc ? hv : hk;
    float *out = pc ? w + p.o_pc : w + p.o_mst;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int m = i0 + ty * 4 + i;
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int n = j0 + tx * 4 + j;
            if (m >= hk || n >= width) continue;
            out[(size_t)m * width + n] = pc ? acc[i][j] : ((m == n ? (w + p.o_de)[m] : 0.f) - acc[i][j]);
        }
    }
}

// grid (heads, ceil(hv / 32), runs), block 256: the serial part. The state of 32 value columns lives in shared memory while the block walks the chunks of its
// run:   O = O1 + Qe P,   P = Mst P + Pc   (P = dE P + Pc without rank-one terms). The left operands are read from global memory (four consecutive k per load,
// the threads of a row group share the addresses), the state is the right operand in shared memory.
__global__ void __launch_bounds__(256) scan_state_kernel(ScanP p, ScanIn in, const ScanRun *__restrict__ runs) {
    __shared__ float P[SMAX_HK * SPS];
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
    for (int ch = 0; ch < nch; ch++) {
        const int L = min(SC, run.n - ch * SC);
        const int lr0 = (run.pad + ch) * SC;
        const float *w = in.ws + ((size_t)(run.pad + ch) * p.heads + h) * p.chunk_floats;
        __syncthreads();  // the state is complete
        {  // O = O1 + Qe P: thread (ty, tx) owns rows ty * 4 .. +3 and columns tx * 2, tx * 2 + 1
            const int ty = tid >> 4, tx = tid & 15;
            float acc[4][2] = {};
            const float *a0 = w + p.o_qe + (size_t)(ty * 4) * hk;
#pragma unroll 2
            for (int k = 0; k < hk; k += 4) {
                float4 a[4];
#pragma unroll
                for (int i = 0; i < 4; i++) a[i] = *reinterpret_cast<const float4 *>(a0 + (size_t)i * hk + k);
#pragma unroll
                for (int kk = 0; kk < 4; kk++) {
                    const float b0 = P[(k + kk) * SPS + tx * 2], b1 = P[(k + kk) * SPS + tx * 2 + 1];
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        const float ai = kk == 0 ? a[i].x : kk == 1 ? a[i].y : kk == 2 ? a[i].z : a[i].w;
                        acc[i][0] += ai * b0;
                        acc[i][1] += ai * b1;
                    }
                }
            }
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int t = ty * 4 + i;
#pragma unroll
                for (int j = 0; j < 2; j++) {
                    const int c = tx * 2 + j;
                    if (t < L && c < ncol) in.o[scan_row(p, lr0 + t, h, hv) + c0 + c] = acc[i][j] + w[p.o_o1 + (size_t)t * hv + c0 + c];
                }
            }
        }
        if (r1) {  // P = Mst P + Pc: thread (ty, tx) owns rows ty * 4 .. +3 and columns tx * 4 .. +3
            const int ty = tid >> 3, tx = tid & 7;
            float acc[4][4] = {};
            const float *a0 = w + p.o_mst + (size_t)(ty * 4) * hk;
            if (ty * 4 < hk) {
#pragma unroll 2
                for (int k = 0; k < hk; k += 4) {
                    float4 a[4];
#pragma unroll
                    for (int i = 0; i < 4; i++) a[i] = *reinterpret_cast<const float4 *>(a0 + (size_t)i * hk + k);
#pragma unroll
                    for (int kk = 0; kk < 4; kk++) {
                        float bv[4];
#pragma unroll
                        for (int j = 0; j < 4; j++) bv[j] = P[(k + kk) * SPS + tx * 4 + j];
#pragma unroll
                        for (int i = 0; i < 4; i++) {
                            const float ai = kk == 0 ? a[i].x : kk == 1 ? a[i].y : kk == 2 ? a[i].z : a[i].w;
#pragma unroll
                            for (int j = 0; j < 4; j++) acc[i][j] += ai * bv[j];
                        }
                    }
                }
            }
            __syncthreads();  // everybody is done with the old state
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int m = ty * 4 + i;
                if (m >= hk) continue;
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int c = tx * 4 + j;
                    P[m * SPS + c] = (c0 + c < hv) ? acc[i][j] + w[p.o_pc + (size_t)m * hv + c0 + c] : 0.f;
                }
            }
        } else {
            __syncthreads();
            for (int e = tid; e < hk * SCW; e += 256) {
                const int i = e / SCW, c = e % SCW;
                P[i * SPS + c] = (c0 + c < hv) ? w[p.o_de + i] * P[i * SPS + c] + w[p.o_pc + (size_t)i * hv + c0 + c] : 0.f;
            }
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
    const int tk = (p.hk + 63) / 64, tv = (p.hv + 63) / 64;
    if (r1) {
        scan_tm_kernel<<<g, 256, 0, s>>>(p, in, runs);
        if (int e = fllm_check_launch("scan_tm")) return e;
    }
    scan_wide_kernel<<<dim3(max_chunks, nruns, p.heads * (r1 ? 3 * tk + tv : tv)), 256, 0, s>>>(p, in, runs, tk, tv);
    if (int e = fllm_check_launch("scan_wide")) return e;
    scan_pm_kernel<<<dim3(max_chunks, nruns, p.heads * (tk * tv + (r1 ? tk * tk : 0))), 256, 0, s>>>(p, in, runs);
    if (int e = fllm_check_launch("scan_pm")) return e;
    scan_state_kernel<<<dim3(p.heads, (p.hv + SCW - 1) / SCW, nruns), 256, 0, s>>>(p, in, runs);
    return fllm_check_launch("scan_state");
}
