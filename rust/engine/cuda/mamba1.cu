// fLLM CUDA backend: Mamba-1 selective state space layer (Mamba, Falcon-Mamba, Jamba) on the device.
//
// Mirrors `mamba1_forward` in rust/engine/src/rust_decoder.rs (the reference). The input projection, x_proj, dt_proj and the output
// projection run as ordinary GEMMs; the kernels in between:
//   1. the depthwise causal convolution with SiLU (fllm_ssm_conv_launch, mamba2.cu), history in the sequence's state buffer;
//   2. mamba1_prep_kernel: optional RMS normalisation of the dt / B / C parts of x_proj's output (Falcon-Mamba) and the compact copy of
//      the dt part that feeds dt_proj;
//   3. mamba1_scan_kernel<N>: one thread per channel keeps its N state values in registers while it walks the tokens of a run:
//      s = s * exp(step * A) + step * B * x;  y = (s . C + D * x) * silu(gate).
// State per layer and sequence: [inner][N] FP32, then the convolution history. Plain FP32 CUDA-core code, runs unchanged on Turing.
#include "common.cuh"
#include "fllm_cuda_internal.h"

// Rows of one sequence that are handled together (mirrors DeltaRun in backend/mod.rs).
struct SsmRun1 {
    unsigned long long state;
    int row0;
    int n;
    int reset;
    int pad;
};

int fllm_mamba1_supported(int inner, int state, int rank, int ksz) {
    return inner > 0 && rank > 0 && ksz >= 2 && ksz <= 8 && (state == 16 || state == 32);
}

// grid rows, block 128. split: rows x (rank + 2 state); dt_in: rows x rank. eps <= 0: no normalisation.
__global__ void __launch_bounds__(128) mamba1_prep_kernel(float *__restrict__ split, float *__restrict__ dt_in, int rank, int state, float eps,
                                                          const float *w_dt, const float *w_b, const float *w_c) {
    __shared__ float scratch[32];
    const size_t row = blockIdx.x;
    float *r = split + row * (size_t)(rank + 2 * state);
    const int warps = blockDim.x >> 5, warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    for (int seg = 0; seg < 3; seg++) {
        const int off = seg == 0 ? 0 : (seg == 1 ? rank : rank + state);
        const int len = seg == 0 ? rank : state;
        float inv = 1.0f;
        if (eps > 0.f) {
            float sum = 0.f;
            for (int i = threadIdx.x; i < len; i += blockDim.x) sum += r[off + i] * r[off + i];
            sum = warp_sum(sum);
            __syncthreads();
            if (lane == 0) scratch[warp] = sum;
            __syncthreads();
            float total = 0.f;
            for (int i = 0; i < warps; i++) total += scratch[i];
            inv = 1.0f / sqrtf(total / (float)len + eps);
        }
        for (int i = threadIdx.x; i < len; i += blockDim.x) {
            // Jamba: learned weights behind the RMS norms (null: none)
            const float *wt = seg == 0 ? w_dt : (seg == 1 ? w_b : w_c);
            const float v = r[off + i] * inv * (wt ? wt[i] : 1.0f);
            r[off + i] = v;
            if (seg == 0) dt_in[row * rank + i] = v;
        }
        __syncthreads();
    }
}

// grid (ceil(inner / 128), runs), block 128
template <int N>
__global__ void __launch_bounds__(128) mamba1_scan_kernel(const float *__restrict__ split, int split_w, int rank, const float *__restrict__ dt,
                                                          const float *__restrict__ mixed, const float *__restrict__ proj,
                                                          const float *__restrict__ decay, const float *__restrict__ skip,
                                                          const SsmRun1 *__restrict__ runs, float *__restrict__ y, int inner,
                                                          size_t state_off) {
    const SsmRun1 run = runs[blockIdx.y];
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= inner) return;
    float *rec = reinterpret_cast<float *>(reinterpret_cast<char *>(run.state) + state_off) + (size_t)c * N;
    float s[N], a[N];
    for (int k = 0; k < N; k++) {
        s[k] = run.reset ? 0.f : rec[k];
        a[k] = decay[(size_t)c * N + k];
    }
    const float sk = skip[c];
    for (int t = 0; t < run.n; t++) {
        const size_t row = (size_t)(run.row0 + t);
        const float raw = dt[row * inner + c];
        const float step = raw > 20.f ? raw : log1pf(expf(raw));
        const float xc = mixed[row * inner + c];
        const float *b = split + row * split_w + rank;
        const float *cv = b + N;
        float acc = 0.f;
#pragma unroll
        for (int k = 0; k < N; k++) {
            s[k] = s[k] * expf(step * a[k]) + step * b[k] * xc;
            acc += s[k] * cv[k];
        }
        const float z = proj[row * 2 * (size_t)inner + inner + c];
        y[row * inner + c] = (acc + sk * xc) * (z / (1.0f + expf(-z)));
    }
    for (int k = 0; k < N; k++) rec[k] = s[k];
}

int fllm_mamba1_conv_launch(cudaStream_t s, const void *runs, int nruns, int inner, int ksz, size_t state_off, size_t rec_bytes, const float *proj,
                            const float *convw, const float *convb, float *mixed) {
    if (nruns <= 0) return 0;
    return fllm_ssm_conv_launch(s, runs, nruns, proj, 2 * inner, 0, convw, convb, mixed, inner, ksz, state_off, rec_bytes);
}

int fllm_mamba1_prep_launch(cudaStream_t s, float *split, float *dt_in, int rows, int rank, int state, float eps, const float *w_dt, const float *w_b,
                            const float *w_c) {
    if (rows <= 0) return 0;
    mamba1_prep_kernel<<<rows, 128, 0, s>>>(split, dt_in, rank, state, eps, w_dt, w_b, w_c);
    return fllm_check_launch("mamba1_prep");
}

int fllm_mamba1_scan_launch(cudaStream_t s, const void *runs, int nruns, int inner, int state, int rank, size_t state_off, const float *split,
                            const float *dt, const float *mixed, const float *proj, const float *decay, const float *skip, float *out) {
    if (nruns <= 0) return 0;
    const SsmRun1 *r = static_cast<const SsmRun1 *>(runs);
    const dim3 grid((inner + 127) / 128, nruns);
    if (state == 16) mamba1_scan_kernel<16><<<grid, 128, 0, s>>>(split, rank + 2 * state, rank, dt, mixed, proj, decay, skip, r, out, inner, state_off);
    else if (state == 32) mamba1_scan_kernel<32><<<grid, 128, 0, s>>>(split, rank + 2 * state, rank, dt, mixed, proj, decay, skip, r, out, inner, state_off);
    else return fllm_set_error("mamba1: unsupported state size %d", state);
    return fllm_check_launch("mamba1_scan");
}

// ---- xLSTM (mLSTM) cell -------------------------------------------------------------------------------------------------------------------
// Mirrors `xlstm_forward` in rust_decoder.rs. One block per (head, run); the block walks the tokens of the run. Per head the state is the matrix
// C [dqk][dv], the normaliser n [dqk] and the stabiliser m (FP32, in this order). Thread t owns the columns t, t + blockDim, ... of C.
//   f_act = exp(logsigmoid(f) + m_old - m_new), i_act = exp(i - m_new), m_new = max(logsigmoid(f) + m_old, i)
//   C = f_act * C + i_act * k v^T;  n = f_act * n + i_act * k;  h = (q / sqrt(dqk)) . C / (max(|(q / sqrt(dqk)) . n|, exp(-m_new)) + eps)
//   out = sigmoid(o) * layer_norm_head(h) * weight
#define XL_BLOCK 256

int fllm_xlstm_supported(int heads, int dqk, int dv) { return heads > 0 && dqk > 0 && dqk <= 1024 && dv > 0 && dv <= 4096; }

__global__ void xlstm_cell_kernel(const float *__restrict__ q, const float *__restrict__ k, const float *__restrict__ v, const float *__restrict__ o,
                                  const float *__restrict__ ig, const float *__restrict__ fg, const float *__restrict__ nw,
                                  const SsmRun1 *__restrict__ runs, float *__restrict__ y, int heads, int dqk, int dv, size_t state_off, float cap,
                                  float eps, float norm_eps) {
    const SsmRun1 run = runs[blockIdx.y];
    const int h = blockIdx.x;
    const int tid = threadIdx.x;
    extern __shared__ float xl_sh[];
    float *qs = xl_sh;
    float *ks = qs + dqk;
    float *hv = ks + dqk;
    float *red = hv + dv;       // XL_BLOCK partial sums
    float *misc = red + XL_BLOCK;  // 0: f_act, 1: i_act, 2: m_new, 3: qn, 4: mean, 5: inverse deviation
    const size_t head_floats = (size_t)dqk * dv + dqk + 1;
    float *st = reinterpret_cast<float *>(reinterpret_cast<char *>(run.state) + state_off) + (size_t)h * head_floats;
    float *C = st;
    float *N = st + (size_t)dqk * dv;
    float *M = N + dqk;
    if (run.reset) {
        for (size_t i = tid; i < head_floats; i += XL_BLOCK) st[i] = 0.f;
    }
    __syncthreads();
    const float qscale = rsqrtf((float)dqk);
    for (int t = 0; t < run.n; t++) {
        const size_t row = (size_t)(run.row0 + t);
        for (int a = tid; a < dqk; a += XL_BLOCK) {
            qs[a] = q[(row * heads + h) * dqk + a] * qscale;
            ks[a] = k[(row * heads + h) * dqk + a];
        }
        if (tid == 0) {
            const float ip = cap * tanhf(ig[row * heads + h] / cap);
            const float fp = cap * tanhf(fg[row * heads + h] / cap);
            const float log_f = fminf(fp, 0.f) - log1pf(expf(-fabsf(fp)));
            const float m_new = fmaxf(log_f + *M, ip);
            misc[0] = expf(log_f + *M - m_new);
            misc[1] = expf(ip - m_new);
            misc[2] = m_new;
            *M = m_new;
        }
        __syncthreads();
        const float f_act = misc[0], i_act = misc[1];
        float part = 0.f;
        for (int a = tid; a < dqk; a += XL_BLOCK) {
            const float nv = f_act * N[a] + i_act * ks[a];
            N[a] = nv;
            part += qs[a] * nv;
        }
        red[tid] = part;
        __syncthreads();
        for (int s2 = XL_BLOCK / 2; s2 > 0; s2 >>= 1) {
            if (tid < s2) red[tid] += red[tid + s2];
            __syncthreads();
        }
        if (tid == 0) misc[3] = red[0];
        __syncthreads();
        const float denom = fmaxf(fabsf(misc[3]), expf(-misc[2])) + eps;
        float lsum = 0.f;
        for (int b = tid; b < dv; b += XL_BLOCK) {
            const float vb = v[(row * heads + h) * dv + b];
            float acc = 0.f;
            for (int a = 0; a < dqk; a++) {
                const size_t at = (size_t)a * dv + b;
                const float c = f_act * C[at] + i_act * ks[a] * vb;
                C[at] = c;
                acc += qs[a] * c;
            }
            const float hval = acc / denom;
            hv[b] = hval;
            lsum += hval;
        }
        red[tid] = lsum;
        __syncthreads();
        for (int s2 = XL_BLOCK / 2; s2 > 0; s2 >>= 1) {
            if (tid < s2) red[tid] += red[tid + s2];
            __syncthreads();
        }
        if (tid == 0) misc[4] = red[0] / (float)dv;
        __syncthreads();
        const float mean = misc[4];
        float vsum = 0.f;
        for (int b = tid; b < dv; b += XL_BLOCK) {
            const float d = hv[b] - mean;
            vsum += d * d;
        }
        red[tid] = vsum;
        __syncthreads();
        for (int s2 = XL_BLOCK / 2; s2 > 0; s2 >>= 1) {
            if (tid < s2) red[tid] += red[tid + s2];
            __syncthreads();
        }
        if (tid == 0) misc[5] = rsqrtf(red[0] / (float)dv + norm_eps);
        __syncthreads();
        const float inv = misc[5];
        for (int b = tid; b < dv; b += XL_BLOCK) {
            const float gate = 1.f / (1.f + expf(-o[(row * heads + h) * dv + b]));
            y[(row * heads + h) * dv + b] = gate * (hv[b] - mean) * inv * nw[(size_t)h * dv + b];
        }
        __syncthreads();
    }
}

int fllm_xlstm_cell_launch(cudaStream_t s, const void *runs, int nruns, int heads, int dqk, int dv, size_t state_off, const float *q, const float *k,
                           const float *v, const float *o, const float *ig, const float *fg, const float *nw, float cap, float eps, float norm_eps,
                           float *out) {
    if (nruns <= 0) return 0;
    const SsmRun1 *r = static_cast<const SsmRun1 *>(runs);
    const dim3 grid(heads, nruns);
    const size_t shared = ((size_t)2 * dqk + dv + XL_BLOCK + 8) * sizeof(float);
    xlstm_cell_kernel<<<grid, XL_BLOCK, shared, s>>>(q, k, v, o, ig, fg, nw, r, out, heads, dqk, dv, state_off, cap, eps, norm_eps);
    return fllm_check_launch("xlstm_cell");
}
