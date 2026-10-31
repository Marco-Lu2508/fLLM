// fLLM CUDA backend: the convolutions and the relative-position attention of Inkling (the reference is `inkling_layers` of rust_decoder/inkling.rs).
//
//   conv_resid   out = x + depthwise causal convolution of x; the last taps - 1 inputs of every channel are the layer's state in the sequence's device buffer
//                (one thread per channel walks the tokens of a run, like shortconv.cu)
//   rel_attn     attention of (row, head) over an f16 KV cache without rotary embeddings: score = dot(q * tau, k) * scale + tau * sum_d rel[d] * profile[d][distance]
//                (the bias exists for distances below `extent`), optionally over the last `window` keys only; tau = 1 + alpha * ln(max(1, (position + 1) / floor))
//
// FP32 CUDA-core code with block-level primitives only, so the same file builds for AMD GPUs.
#include <math.h>
#include "common.cuh"
#include "fllm_cuda_internal.h"

// Rows of one sequence that are handled together (mirrors DeltaRun in backend/mod.rs).
struct InkRun {
    unsigned long long state;
    int row0;
    int n;
    int reset;
    int pad;
};

#define INK_MAX_TAPS 8
#define INK_THREADS 128
#define INK_MAX_DK 512
#define INK_MAX_REL 128

int fllm_conv_resid_supported(int channels, int taps) { return channels > 0 && taps >= 2 && taps <= INK_MAX_TAPS; }

// grid (ceil(channels / 128), runs), block 128
__global__ void __launch_bounds__(INK_THREADS) conv_resid_kernel(const float *__restrict__ x, const float *__restrict__ convw, const InkRun *__restrict__ runs,
                                                                 float *__restrict__ out, int channels, int taps, size_t state_off) {
    const InkRun run = runs[blockIdx.y];
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float *hist = reinterpret_cast<float *>(reinterpret_cast<char *>(run.state) + state_off) + (size_t)c * (taps - 1);
    float w[INK_MAX_TAPS], h[INK_MAX_TAPS - 1];
    for (int k = 0; k < taps; k++) w[k] = convw[(size_t)c * taps + k];
    for (int j = 0; j < taps - 1; j++) h[j] = run.reset ? 0.f : hist[j];
    for (int t = 0; t < run.n; t++) {
        const float cur = x[(size_t)(run.row0 + t) * channels + c];
        float value = 0.f;
        for (int j = 0; j < taps - 1; j++) value += w[j] * h[j];
        value += w[taps - 1] * cur;
        out[(size_t)(run.row0 + t) * channels + c] = value + cur;
        for (int j = 0; j + 1 < taps - 1; j++) h[j] = h[j + 1];
        h[taps - 2] = cur;
    }
    for (int j = 0; j < taps - 1; j++) hist[j] = h[j];
}

int fllm_conv_resid_launch(cudaStream_t s, const void *runs, int nruns, int channels, int taps, size_t state_off, const float *x, const float *convw, float *out) {
    if (!fllm_conv_resid_supported(channels, taps)) return fllm_set_error("conv_resid: unsupported shape");
    if (nruns <= 0) return 0;
    conv_resid_kernel<<<dim3((channels + INK_THREADS - 1) / INK_THREADS, nruns), INK_THREADS, 0, s>>>(x, convw, static_cast<const InkRun *>(runs), out, channels, taps,
                                                                                                   state_off);
    return fllm_check_launch("conv_resid");
}

__device__ __forceinline__ float ink_block_max(float v, float *red) {
    const int t = threadIdx.x;
    red[t] = v;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (t < off) red[t] = fmaxf(red[t], red[t + off]);
        __syncthreads();
    }
    const float out = red[0];
    __syncthreads();
    return out;
}

__device__ __forceinline__ float ink_block_sum(float v, float *red) {
    const int t = threadIdx.x;
    red[t] = v;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (t < off) red[t] += red[t + off];
        __syncthreads();
    }
    const float out = red[0];
    __syncthreads();
    return out;
}

int fllm_rel_attention_supported(int hd, int d_rel) { return hd > 0 && hd <= INK_MAX_DK && d_rel > 0 && d_rel <= INK_MAX_REL; }

// grid (heads, rows), block 128: tiles of 128 keys with an online softmax
__global__ void __launch_bounds__(INK_THREADS)
rel_attn_kernel(const float *__restrict__ q, const float *__restrict__ rel, const float *__restrict__ profile, const uint16_t *__restrict__ kpool,
                const uint16_t *__restrict__ vpool, float *__restrict__ out, const int64_t *__restrict__ tables, const int64_t *__restrict__ lengths, int heads,
                int kv_heads, int hd, int d_rel, int extent, int window, int log_floor, float log_alpha, float scale, int stride, int block_size) {
    __shared__ float qs[INK_MAX_DK];
    __shared__ float rs[INK_MAX_REL];
    __shared__ float tile_p[INK_THREADS];
    __shared__ int tile_slot[INK_THREADS];
    __shared__ float red[INK_THREADS];
    const int h = blockIdx.x, r = blockIdx.y, t = threadIdx.x;
    const int kh = h / (heads / kv_heads);
    const int n = (int)lengths[r];
    const int position = n - 1;
    const float tau = log_floor > 0 ? 1.f + log_alpha * logf(fmaxf((float)(position + 1) / (float)log_floor, 1.f)) : 1.f;
    const int first = (window > 0 && n > window) ? n - window : 0;
    for (int i = t; i < hd; i += blockDim.x) qs[i] = q[((size_t)r * heads + h) * hd + i] * tau;
    for (int i = t; i < d_rel; i += blockDim.x) rs[i] = rel[((size_t)r * heads + h) * d_rel + i];
    __syncthreads();
    float m = -3.0e38f, l = 0.f;
    float acc[4] = {0.f, 0.f, 0.f, 0.f};
    for (int t0 = first; t0 < n; t0 += INK_THREADS) {
        const int j = t0 + t;
        float s = -3.0e38f;
        int slot = 0;
        if (j < n) {
            slot = (int)(tables[(size_t)r * stride + j / block_size] * block_size + j % block_size);
            const uint16_t *key = kpool + (size_t)slot * kv_heads * hd + (size_t)kh * hd;
            float dot = 0.f;
            for (int d = 0; d < hd; d++) dot += qs[d] * h2f_bits(key[d]);
            const int dist = position - j;
            float bias = 0.f;
            if (dist < extent) {
                for (int d = 0; d < d_rel; d++) bias += rs[d] * profile[(size_t)d * extent + dist];
                bias *= tau;
            }
            s = dot * scale + bias;
        }
        tile_slot[t] = slot;
        const float tmax = ink_block_max(s, red);
        const float mn = fmaxf(m, tmax);
        const float corr = expf(m - mn);
        const float pj = j < n ? expf(s - mn) : 0.f;
        tile_p[t] = pj;
        const float tsum = ink_block_sum(pj, red);
        l = l * corr + tsum;
        for (int i = 0; i < 4; i++) {
            const int d = t + i * INK_THREADS;
            if (d < hd) {
                float a = acc[i] * corr;
                for (int jj = 0; jj < INK_THREADS; jj++) {
                    const float pw = tile_p[jj];
                    if (pw != 0.f) a += pw * h2f_bits(vpool[(size_t)tile_slot[jj] * kv_heads * hd + (size_t)kh * hd + d]);
                }
                acc[i] = a;
            }
        }
        __syncthreads();
        m = mn;
    }
    for (int i = 0; i < 4; i++) {
        const int d = t + i * INK_THREADS;
        if (d < hd) out[((size_t)r * heads + h) * hd + d] = l > 0.f ? acc[i] / l : 0.f;
    }
}

int fllm_rel_attention_launch(cudaStream_t s, const float *q, const float *rel, const float *profile, const uint16_t *kpool, const uint16_t *vpool, float *out,
                              const int64_t *tables, const int64_t *lengths, int rows, int heads, int kv_heads, int hd, int d_rel, int extent, int window,
                              int log_floor, float log_alpha, float scale, int stride, int block_size) {
    if (rows <= 0) return 0;
    if (!fllm_rel_attention_supported(hd, d_rel)) return fllm_set_error("rel_attention: head width %d or %d relative channels are not supported", hd, d_rel);
    dim3 grid(heads, rows);
    rel_attn_kernel<<<grid, INK_THREADS, 0, s>>>(q, rel, profile, kpool, vpool, out, tables, lengths, heads, kv_heads, hd, d_rel, extent, window, log_floor, log_alpha,
                                                scale, stride, block_size);
    return fllm_check_launch("rel_attention");
}
