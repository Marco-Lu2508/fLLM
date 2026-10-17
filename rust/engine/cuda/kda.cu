// fLLM CUDA backend: Kimi Delta Attention (KDA) layer on the device.
//
// Mirrors rust/engine/src/cpu_kda.rs: causal depthwise convolution with SiLU over the channels [q|k|v], L2 normalisation of q
// and k per head (with 1/sqrt(head_dim) scaling on q), per-channel decay recurrence with write strength sigmoid(beta), and
// output RMS norm with sigmoid output gate.
#include "common.cuh"
#include "fllm_cuda_internal.h"

struct DeltaRun {
    unsigned long long state;  // device address of the sequence's state buffer
    int row0;                  // first row of the run in the activations
    int n;                     // number of rows (consecutive positions)
    int reset;                 // 1: the sequence starts here, the old state is ignored
    int pad;
};

#define DELTA_CHUNK 16

__device__ __forceinline__ float block_sum_n(float v, float *scratch) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float total = 0.f;
    for (int i = 0; i < warps; i++) total += scratch[i];
    return total;
}

__device__ __forceinline__ float silu_f(float x) { return x / (1.0f + expf(-x)); }

// grid (ceil(conv_dim / hk), runs, ceil(max_n / DELTA_CHUNK)), block hk threads.
// Block x < heads normalises a query head, x in [heads, 2 * heads) a key head.
__global__ void __launch_bounds__(128) kda_conv_kernel(const float *__restrict__ qkv, const float *__restrict__ convw,
                                                       const DeltaRun *__restrict__ runs, float *__restrict__ mixed,
                                                       int conv_dim, int heads, int hk, int ksz, size_t state_off, size_t rec_bytes,
                                                       float qscale) {
    __shared__ float scratch[32];
    const DeltaRun run = runs[blockIdx.y];
    const int t0 = blockIdx.z * DELTA_CHUNK;
    if (t0 >= run.n) return;
    const int t1 = min(run.n, t0 + DELTA_CHUNK);
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const bool valid = c < conv_dim;
    const bool head = blockIdx.x < 2 * heads;
    if (!valid && !head) return;
    const int hist_len = ksz - 1;
    float *hist = reinterpret_cast<float *>(reinterpret_cast<char *>(run.state) + state_off + rec_bytes) + (size_t)c * hist_len;
    const float *x = qkv + (size_t)run.row0 * conv_dim + c;

    float w[8], win[8];
    for (int k = 0; k < ksz; k++) w[k] = valid ? convw[(size_t)c * ksz + k] : 0.f;
    for (int j = 0; j < hist_len; j++) {
        const int idx = t0 - hist_len + j;
        win[j] = idx >= 0 ? x[(size_t)idx * conv_dim] : (run.reset ? 0.f : hist[idx + hist_len]);
    }
    for (int t = t0; t < t1; t++) {
        const float cur = x[(size_t)t * conv_dim];
        float acc = w[hist_len] * cur;
        for (int k = 0; k < hist_len; k++) acc += w[k] * win[k];
        acc = silu_f(acc);
        if (head) {
            const float ss = block_sum_n(acc * acc, scratch);
            const float inv = 1.0f / sqrtf(ss + 1e-6f);
            acc = blockIdx.x < heads ? acc * inv * qscale : acc * inv;
        }
        mixed[(size_t)(run.row0 + t) * conv_dim + c] = acc;
        for (int k = 0; k + 1 < hist_len; k++) win[k] = win[k + 1];
        win[hist_len - 1] = cur;
    }
    if (blockIdx.z == 0 && valid) {
        float fresh[8];
        for (int k = 0; k < hist_len; k++) {
            const int idx = run.n + k - hist_len;
            fresh[k] = idx >= 0 ? x[(size_t)idx * conv_dim] : (run.reset ? 0.f : hist[idx + hist_len]);
        }
        for (int k = 0; k < hist_len; k++) hist[k] = fresh[k];
    }
}

// grid (heads, runs), block 2 * HK threads. Value column c = tid / 2 is shared by lane pair:
// lane half 0 holds the first HK / 2 rows of the column's state in registers, half 1 the others.
template <int HK>
__global__ void __launch_bounds__(256) kda_rec_kernel(const float *__restrict__ mixed,
                                                      const float *__restrict__ forget,
                                                      const float *__restrict__ beta,
                                                      const float *__restrict__ gate,
                                                      const float *__restrict__ dt_bias,
                                                      const float *__restrict__ rate,
                                                      const float *__restrict__ norm,
                                                      const DeltaRun *__restrict__ runs,
                                                      float *__restrict__ out,
                                                      int heads, int conv_dim,
                                                      float lower_bound,
                                                      size_t state_off, float eps) {
    constexpr int HALF = HK / 2;
    __shared__ float scratch[32];
    const DeltaRun run = runs[blockIdx.y];
    const int h = blockIdx.x;
    const int hv = HK;
    const int c = threadIdx.x >> 1;
    const int half = threadIdx.x & 1;
    const int key_dim = heads * HK;
    const float *bias_ptr = dt_bias + h * HK + half * HALF;
    float *S = reinterpret_cast<float *>(reinterpret_cast<char *>(run.state) + state_off) + ((size_t)h * HK + half * HALF) * hv + c;

    float s[HALF];
#pragma unroll
    for (int i = 0; i < HALF; i++) s[i] = run.reset ? 0.f : S[(size_t)i * hv];

    const float rate_h = rate[h], nw = norm[c];
    for (int t = 0; t < run.n; t++) {
        const size_t row = (size_t)(run.row0 + t);
        const float *m = mixed + row * conv_dim;
        const float4 *q4 = reinterpret_cast<const float4 *>(m + h * HK + half * HALF);
        const float4 *k4 = reinterpret_cast<const float4 *>(m + key_dim + h * HK + half * HALF);
        const float *f_tok = forget + row * key_dim + h * HK + half * HALF;
        const float v = m[2 * key_dim + h * hv + c];

        float memory = 0.f;
#pragma unroll
        for (int i = 0; i < HALF / 4; i++) {
            const float4 k = k4[i];
            const float k_arr[4] = {k.x, k.y, k.z, k.w};
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int idx = 4 * i + e;
                const float raw = f_tok[idx] + bias_ptr[idx];
                float g;
                if (lower_bound >= 0.f) {
                    g = lower_bound / (1.0f + expf(-rate_h * raw));
                } else {
                    const float sp = raw > 20.f ? raw : log1pf(expf(raw));
                    g = -rate_h * sp;
                }
                const float dec = expf(g);
                s[idx] *= dec;
                memory += s[idx] * k_arr[e];
            }
        }
        memory += __shfl_xor_sync(0xffffffffu, memory, 1);
        const float sigma = 1.0f / (1.0f + expf(-beta[row * heads + h]));
        const float delta = (v - memory) * sigma;
        float acc = 0.f;
#pragma unroll
        for (int i = 0; i < HALF / 4; i++) {
            const float4 k = k4[i], q = q4[i];
            s[4 * i + 0] += k.x * delta;
            acc += s[4 * i + 0] * q.x;
            s[4 * i + 1] += k.y * delta;
            acc += s[4 * i + 1] * q.y;
            s[4 * i + 2] += k.z * delta;
            acc += s[4 * i + 2] * q.z;
            s[4 * i + 3] += k.w * delta;
            acc += s[4 * i + 3] * q.w;
        }
        acc += __shfl_xor_sync(0xffffffffu, acc, 1);
        // gated RMS norm of the head output (every column counted once)
        const float ss = block_sum_n(half == 0 ? acc * acc : 0.f, scratch);
        const float inv = 1.0f / sqrtf(ss / (float)hv + eps);
        if (half == 0) {
            const float g_val = gate[row * key_dim + h * hv + c];
            const float g_sig = 1.0f / (1.0f + expf(-g_val));
            out[row * key_dim + h * hv + c] = acc * inv * nw * g_sig;
        }
    }
#pragma unroll
    for (int i = 0; i < HALF; i++) S[(size_t)i * hv] = s[i];
}

int fllm_kda_supported(int heads, int head_dim, int ksz) {
    return heads > 0 && (head_dim == 16 || head_dim == 32 || head_dim == 64 || head_dim == 128) && ksz >= 2 && ksz <= 8;
}

int fllm_kda_launch(cudaStream_t s, const void *runs, int nruns, int max_n, int heads, int head_dim, int ksz,
                    float lower_bound, float eps, size_t state_off, size_t rec_bytes,
                    const float *qkv, const float *forget, const float *beta, const float *gate,
                    const float *convw, const float *dt_bias, const float *rate, const float *norm,
                    float *mixed, float *out) {
    if (nruns <= 0 || max_n <= 0) return 0;
    if (!fllm_kda_supported(heads, head_dim, ksz)) return fllm_set_error("KDA shape not supported by the CUDA kernels");
    const int conv_dim = 3 * heads * head_dim;
    const DeltaRun *r = static_cast<const DeltaRun *>(runs);
    const dim3 grid((conv_dim + head_dim - 1) / head_dim, nruns, (max_n + DELTA_CHUNK - 1) / DELTA_CHUNK);
    kda_conv_kernel<<<grid, head_dim, 0, s>>>(qkv, convw, r, mixed, conv_dim, heads, head_dim, ksz, state_off, rec_bytes, 1.0f / sqrtf((float)head_dim));
    if (int e = fllm_check_launch("kda_conv")) return e;
    const dim3 g2(heads, nruns);
    if (head_dim == 128)
        kda_rec_kernel<128><<<g2, 2 * 128, 0, s>>>(mixed, forget, beta, gate, dt_bias, rate, norm, r, out, heads, conv_dim, lower_bound, state_off, eps);
    else if (head_dim == 64)
        kda_rec_kernel<64><<<g2, 2 * 64, 0, s>>>(mixed, forget, beta, gate, dt_bias, rate, norm, r, out, heads, conv_dim, lower_bound, state_off, eps);
    else if (head_dim == 32)
        kda_rec_kernel<32><<<g2, 2 * 32, 0, s>>>(mixed, forget, beta, gate, dt_bias, rate, norm, r, out, heads, conv_dim, lower_bound, state_off, eps);
    else if (head_dim == 16)
        kda_rec_kernel<16><<<g2, 2 * 16, 0, s>>>(mixed, forget, beta, gate, dt_bias, rate, norm, r, out, heads, conv_dim, lower_bound, state_off, eps);
    return fllm_check_launch("kda_rec");
}
