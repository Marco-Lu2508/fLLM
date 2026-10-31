// fLLM CUDA / HIP backend: chunk-parallel prompt processing of the recurrent layers.
//
// Gated DeltaNet, Kimi Delta Attention, RWKV-7, RWKV-6 and Mamba-2 are all instances of one linear recurrence per head,
//     P_t = (diag(w_t) - b_t a_t^T) P_{t-1} + k_t v_t^T,      o_t = P_t^T q_t + c_t v_t,      w_t = exp(lw_t), lw_t <= 0,
// whose chunked form (rust/engine/src/chunked_scan.rs proves it equal to the sequential recurrence in f64, and documents the derivation) turns
// a run of tokens into steps of 64 tokens. The serial kernels of deltanet.cu / kda.cu / rwkv.cu / mamba2.cu walk the tokens one by one, so a
// prompt costs one dependent chain of `n` steps per layer; here only the chain of `n / 64` state updates is serial.
//
// Per layer: a small per-architecture kernel brings the inputs of the run into the canonical layout (scan_*_prep), six generic kernels do
// the chunk algebra (scan_cum, scan_pairs, scan_tm, scan_wide, scan_pm, scan_state; scan_core.cuh), a small per-architecture kernel applies the output norm / gate
// (scan_*_post). Everything is FP32 CUDA-core code with static shared memory, no warp primitives: it runs on Turing and on the HIP targets.
//
// Layout. Rows of a run sit in the scratch at `lr = run.pad * 64 + t` (`run.pad` = the run's first chunk slot); the arrays are
// [lr][head][width]: q, kp, a, b `hk` wide, v and o `hv` wide, lw `hk` wide (flag LWV) or one value, c one value. The chunk workspace of
// chunk slot `cs`, head `h` is `ws + (cs * heads + h) * chunk_floats` (see ScanP for the parts).
#include <string.h>

#include "common.cuh"
#include "fllm_cuda_internal.h"

#include "scan_core.cuh"

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
