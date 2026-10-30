// fLLM CUDA / HIP backend: RWKV-6 and RWKV-7 operations on the device.
//
// Mirrors rust/engine/src/cpu_rwkv.rs (the reference of selftest-gpu). The projections and the low-rank adapters are ordinary linear
// operations of the decoder; this file has the token shift, the interpolation, the small activations and the two recurrences. A recurrence
// kernel handles one head of one run per block (`n` threads, one per state row/column; the state lives in registers for the whole run, so
// a prompt is one launch per layer and run, not one per token).
#include "common.cuh"
#include "fllm_cuda_internal.h"

struct DeltaRun {
    unsigned long long state;  // device address of the sequence's state buffer
    int row0;                  // first row of the run in the activations
    int n;                     // number of rows (consecutive positions)
    int reset;                 // 1: the sequence starts here, the old state is ignored
    int pad;
};

__device__ __forceinline__ float rw_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// Sum over the `n` threads of the block through shared memory (`scratch` holds blockDim.x floats; works for any block size).
__device__ __forceinline__ float rw_block_sum(float v, float *scratch) {
    __syncthreads();
    scratch[threadIdx.x] = v;
    __syncthreads();
    for (int o = (int)blockDim.x >> 1; o > 0; o >>= 1) {
        if ((int)threadIdx.x < o) scratch[threadIdx.x] += scratch[threadIdx.x + o];
        __syncthreads();
    }
    const float total = scratch[0];
    __syncthreads();
    return total;
}

// grid-stride over rows x hidden: shift[row] = x[row - 1] inside a run, the stored vector at the start of a run.
__global__ void rwkv_shift_kernel(const float *__restrict__ x, float *__restrict__ shift, const DeltaRun *__restrict__ runs, int nruns,
                                  int hidden, size_t shift_off, size_t total) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const int row = (int)(i / hidden), c = (int)(i % hidden);
        for (int r = 0; r < nruns; r++) {
            const DeltaRun run = runs[r];
            if (row < run.row0 || row >= run.row0 + run.n) continue;
            if (row > run.row0) {
                shift[i] = x[i - hidden];
            } else {
                const float *prev = reinterpret_cast<const float *>(reinterpret_cast<const char *>(run.state) + shift_off);
                shift[i] = run.reset ? 0.f : prev[c];
            }
            break;
        }
    }
}

// grid (ceil(hidden / 256), nruns): the stored vector becomes the last row of the run (after the shift kernel read it).
__global__ void rwkv_shift_state_kernel(const float *__restrict__ x, const DeltaRun *__restrict__ runs, int hidden, size_t shift_off) {
    const DeltaRun run = runs[blockIdx.y];
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= hidden || run.n <= 0) return;
    float *prev = reinterpret_cast<float *>(reinterpret_cast<char *>(run.state) + shift_off);
    prev[c] = x[(size_t)(run.row0 + run.n - 1) * hidden + c];
}

__global__ void rwkv_lerp_kernel(const float *__restrict__ x, const float *__restrict__ shift, const float *__restrict__ mix,
                                 const float *__restrict__ delta, float *__restrict__ out, size_t count, int hidden) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x) {
        const float m = mix[i % hidden] + (delta ? delta[i] : 0.f);
        out[i] = x[i] + (shift[i] - x[i]) * m;
    }
}

__global__ void rwkv_act_kernel(float *__restrict__ x, const float *__restrict__ gate, size_t count, int mode) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x) {
        const float v = x[i];
        float o;
        switch (mode) {
            case 0: o = tanhf(v); break;
            case 1: o = rw_sigmoid(v); break;
            case 2: o = v * rw_sigmoid(v); break;
            case 3: o = fmaxf(v, 0.f) * fmaxf(v, 0.f); break;
            case 5: o = v * 0.5f; break;
            default: o = v * rw_sigmoid(gate[i]); break;
        }
        x[i] = o;
    }
}

// RWKV-7. grid (heads, runs), block N threads: thread i owns row i of the head's state S[i][0..N) (value channel i, key channels j).
template <int N>
__global__ void __launch_bounds__(N) rwkv7_kernel(const float *__restrict__ rr, const float *__restrict__ kk_in, const float *__restrict__ vv,
                                                  const float *__restrict__ ww, const float *__restrict__ aa, const float *__restrict__ vres,
                                                  const float *__restrict__ gate, const float *__restrict__ norm_w,
                                                  const float *__restrict__ norm_b, const float *__restrict__ k_k,
                                                  const float *__restrict__ k_a, const float *__restrict__ r_k,
                                                  float *__restrict__ v_first, const DeltaRun *__restrict__ runs, float *__restrict__ out,
                                                  int hidden, size_t state_off, float eps) {
    __shared__ float s_r[N], s_decay[N], s_k[N], s_kk[N], s_a[N], s_red[N];
    const DeltaRun run = runs[blockIdx.y];
    const int h = blockIdx.x;
    const int i = threadIdx.x;
    const int ch = h * N + i;
    float *S = reinterpret_cast<float *>(reinterpret_cast<char *>(run.state) + state_off) + ((size_t)h * N + i) * N;
    float s[N];
#pragma unroll
    for (int j = 0; j < N; j++) s[j] = run.reset ? 0.f : S[j];
    const float lnw = norm_w[ch], lnb = norm_b[ch], kkw = k_k[ch], kaw = k_a[ch];

    for (int t = 0; t < run.n; t++) {
        const size_t at = (size_t)(run.row0 + t) * hidden + ch;
        const float a = rw_sigmoid(aa[at]);
        const float decay = expf(-0.6065306597126334f * rw_sigmoid(ww[at]));
        const float k_in = kk_in[at];
        float v = vv[at];
        if (vres) {
            v += (v_first[at] - v) * rw_sigmoid(vres[at]);
        } else {
            v_first[at] = v;
        }
        float kk = k_in * kkw;
        const float ss = rw_block_sum(kk * kk, s_red);
        kk *= 1.0f / fmaxf(sqrtf(ss), 1e-12f);
        const float k = k_in * (1.0f + (a - 1.0f) * kaw);
        const float r = rr[at];
        s_r[i] = r;
        s_decay[i] = decay;
        s_k[i] = k;
        s_kk[i] = kk;
        s_a[i] = a;
        const float bonus = rw_block_sum(r * k * r_k[ch], s_red);
        // s_* are complete (the block sum synchronised)
        float sa = 0.f;
#pragma unroll
        for (int j = 0; j < N; j++) sa += s[j] * s_kk[j];
        sa = -sa;
        float acc = 0.f;
#pragma unroll
        for (int j = 0; j < N; j++) {
            s[j] = s[j] * s_decay[j] + sa * s_kk[j] * s_a[j] + v * s_k[j];
            acc += s[j] * s_r[j];
        }
        const float mean = rw_block_sum(acc, s_red) / (float)N;
        const float d = acc - mean;
        const float var = rw_block_sum(d * d, s_red) / (float)N;
        const float normed = d * (1.0f / sqrtf(var + eps)) * lnw + lnb;
        out[at] = (normed + bonus * v) * gate[at];
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < N; j++) S[j] = s[j];
}

// RWKV-6. grid (heads, runs), block N threads: thread i owns value column i, S[j][i] for the key channels j.
template <int N>
__global__ void __launch_bounds__(N) rwkv6_kernel(const float *__restrict__ rr, const float *__restrict__ kk, const float *__restrict__ vv,
                                                  const float *__restrict__ ww, const float *__restrict__ gate,
                                                  const float *__restrict__ norm_w, const float *__restrict__ norm_b,
                                                  const float *__restrict__ first, const DeltaRun *__restrict__ runs,
                                                  float *__restrict__ out, int hidden, size_t state_off, float eps) {
    __shared__ float s_r[N], s_w[N], s_k[N], s_u[N], s_red[N];
    const DeltaRun run = runs[blockIdx.y];
    const int h = blockIdx.x;
    const int i = threadIdx.x;
    const int ch = h * N + i;
    float *S = reinterpret_cast<float *>(reinterpret_cast<char *>(run.state) + state_off) + (size_t)h * N * N + i;
    float s[N];
#pragma unroll
    for (int j = 0; j < N; j++) s[j] = run.reset ? 0.f : S[(size_t)j * N];
    const float lnw = norm_w[ch], lnb = norm_b[ch];
    s_u[i] = first[ch];

    for (int t = 0; t < run.n; t++) {
        const size_t at = (size_t)(run.row0 + t) * hidden + ch;
        s_r[i] = rr[at];
        s_k[i] = kk[at];
        s_w[i] = expf(-expf(ww[at]));
        __syncthreads();
        const float v = vv[at];
        float acc = 0.f;
#pragma unroll
        for (int j = 0; j < N; j++) {
            const float kv = s_k[j] * v;
            acc += s_r[j] * (s_u[j] * kv + s[j]);
            s[j] = kv + s_w[j] * s[j];
        }
        const float mean = rw_block_sum(acc, s_red) / (float)N;
        const float d = acc - mean;
        const float var = rw_block_sum(d * d, s_red) / (float)N;
        out[at] = (d * (1.0f / sqrtf(var + eps)) * lnw + lnb) * gate[at];
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < N; j++) S[(size_t)j * N] = s[j];
}

int fllm_rwkv_supported(int hidden, int head_size) {
    return head_size > 0 && hidden % head_size == 0 && (head_size == 16 || head_size == 32 || head_size == 64 || head_size == 128);
}

int fllm_rwkv_shift_launch(cudaStream_t s, const void *runs, int nruns, int total_rows, int hidden, size_t shift_off, const float *x, float *shift) {
    if (nruns <= 0 || total_rows <= 0) return 0;
    const DeltaRun *r = static_cast<const DeltaRun *>(runs);
    const size_t total = (size_t)total_rows * hidden;
    rwkv_shift_kernel<<<(int)min((total + 255) / 256, (size_t)65535), 256, 0, s>>>(x, shift, r, nruns, hidden, shift_off, total);
    if (int e = fllm_check_launch("rwkv_shift")) return e;
    const dim3 g2((hidden + 255) / 256, nruns);
    rwkv_shift_state_kernel<<<g2, 256, 0, s>>>(x, r, hidden, shift_off);
    return fllm_check_launch("rwkv_shift_state");
}

int fllm_rwkv_lerp_launch(cudaStream_t s, const float *x, const float *shift, const float *mix, const float *delta, float *out, size_t count, int hidden) {
    if (count == 0) return 0;
    const int blocks = (int)min((count + 255) / 256, (size_t)65535);
    rwkv_lerp_kernel<<<blocks, 256, 0, s>>>(x, shift, mix, delta, out, count, hidden);
    return fllm_check_launch("rwkv_lerp");
}

int fllm_rwkv_act_launch(cudaStream_t s, float *x, const float *gate, size_t count, int mode) {
    if (count == 0) return 0;
    if (mode < 0 || mode > 5 || (mode == 4 && !gate)) return fllm_set_error("rwkv_act: bad mode");
    const int blocks = (int)min((count + 255) / 256, (size_t)65535);
    rwkv_act_kernel<<<blocks, 256, 0, s>>>(x, gate, count, mode);
    return fllm_check_launch("rwkv_act");
}

int fllm_rwkv7_launch(cudaStream_t s, const void *runs, int nruns, int heads, int head_size, int hidden, float eps, size_t state_off,
                      const float *r, const float *k, const float *v, const float *w, const float *a, const float *vres, const float *gate,
                      const float *norm_w, const float *norm_b, const float *k_k, const float *k_a, const float *r_k, float *v_first,
                      float *out) {
    if (nruns <= 0) return 0;
    if (!fllm_rwkv_supported(hidden, head_size)) return fllm_set_error("RWKV shape not supported by the CUDA kernels");
    const DeltaRun *rn = static_cast<const DeltaRun *>(runs);
    const dim3 grid(heads, nruns);
#define RW7(N) rwkv7_kernel<N><<<grid, N, 0, s>>>(r, k, v, w, a, vres, gate, norm_w, norm_b, k_k, k_a, r_k, v_first, rn, out, hidden, state_off, eps)
    if (head_size == 128) RW7(128);
    else if (head_size == 64) RW7(64);
    else if (head_size == 32) RW7(32);
    else RW7(16);
#undef RW7
    return fllm_check_launch("rwkv7");
}

int fllm_rwkv6_launch(cudaStream_t s, const void *runs, int nruns, int heads, int head_size, int hidden, float eps, size_t state_off,
                      const float *r, const float *k, const float *v, const float *w, const float *gate, const float *norm_w,
                      const float *norm_b, const float *first, float *out) {
    if (nruns <= 0) return 0;
    if (!fllm_rwkv_supported(hidden, head_size)) return fllm_set_error("RWKV shape not supported by the CUDA kernels");
    const DeltaRun *rn = static_cast<const DeltaRun *>(runs);
    const dim3 grid(heads, nruns);
#define RW6(N) rwkv6_kernel<N><<<grid, N, 0, s>>>(r, k, v, w, gate, norm_w, norm_b, first, rn, out, hidden, state_off, eps)
    if (head_size == 128) RW6(128);
    else if (head_size == 64) RW6(64);
    else if (head_size == 32) RW6(32);
    else RW6(16);
#undef RW6
    return fllm_check_launch("rwkv6");
}
