// fLLM CUDA backend: hyper-connections (Hy4, GLM-5-Next, DeepSeek-V4, Qwen3.8-Flash-Next): the residual stream is `hc` parallel streams per token.
// Reference: `Hc::compute`, `Hc::compute_independent`, `Hc::collapse_head` and `combine_streams` of rust_decoder/deepseek4.rs.
//
//   hc_mix      turns the linear output `mixes` of the normalised streams into the mixing weights of the block and collapses the streams:
//                 mode 0 independent (Hy4):   pre = sigmoid(m[n] s0 + b[n]) + eps, post = magnitude sigmoid(m[hc+n] s1 + b[hc+n]) + eps
//                 mode 1 Sinkhorn (GLM-5-Next, mHC): the same pre, post = 2 sigmoid(..), comb = Sinkhorn-normalised softmax rows
//                 mode 2 head: pre only (the final collapse of the model)
//                 mode 3 mean: pre = 1 / hc (no weights)
//                 collapsed = sum_n pre[n] streams[n]
//   hc_combine  streams[k] = post[k] * block + sum_j comb[j][k] * old[j]   (without comb: + old[k])
#include "common.cuh"
#include "fllm_cuda_internal.h"

#define HC_MAX 8

__device__ __forceinline__ float hc_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

__global__ void hc_mix_kernel(const float *__restrict__ streams, const float *__restrict__ mixes, const float *__restrict__ base,
                              const float *__restrict__ scale, float *__restrict__ post_out, float *__restrict__ comb_out, float *__restrict__ collapsed,
                              int hc, int hidden, int mode, int iters, float eps, float magnitude) {
    __shared__ float pre[HC_MAX];
    const int r = blockIdx.x;
    if (threadIdx.x == 0) {
        const int width = mode == 0 ? 2 * hc : (mode == 1 ? (2 + hc) * hc : hc);
        const float *m = mixes + (size_t)r * width;
        for (int n = 0; n < hc; n++) {
            pre[n] = mode == 3 ? 1.0f / hc : hc_sigmoid(m[n] * scale[0] + base[n]) + eps;
            if (mode == 0) post_out[r * hc + n] = magnitude * hc_sigmoid(m[hc + n] * scale[1] + base[hc + n]) + eps;
            if (mode == 1) post_out[r * hc + n] = 2.0f * hc_sigmoid(m[hc + n] * scale[1] + base[hc + n]);
        }
        if (mode == 1) {
            float c[HC_MAX * HC_MAX];
            for (int j = 0; j < hc; j++) {
                float mx = -3.0e38f;
                for (int k = 0; k < hc; k++) {
                    c[j * hc + k] = m[2 * hc + j * hc + k] * scale[2] + base[2 * hc + j * hc + k];
                    mx = fmaxf(mx, c[j * hc + k]);
                }
                float sum = 0.f;
                for (int k = 0; k < hc; k++) {
                    c[j * hc + k] = expf(c[j * hc + k] - mx);
                    sum += c[j * hc + k];
                }
                for (int k = 0; k < hc; k++) c[j * hc + k] = c[j * hc + k] / sum + eps;
            }
            for (int k = 0; k < hc; k++) {
                float s = 0.f;
                for (int j = 0; j < hc; j++) s += c[j * hc + k];
                for (int j = 0; j < hc; j++) c[j * hc + k] /= s + eps;
            }
            for (int it = 1; it < iters; it++) {
                for (int j = 0; j < hc; j++) {
                    float s = 0.f;
                    for (int k = 0; k < hc; k++) s += c[j * hc + k];
                    for (int k = 0; k < hc; k++) c[j * hc + k] /= s + eps;
                }
                for (int k = 0; k < hc; k++) {
                    float s = 0.f;
                    for (int j = 0; j < hc; j++) s += c[j * hc + k];
                    for (int j = 0; j < hc; j++) c[j * hc + k] /= s + eps;
                }
            }
            for (int i = 0; i < hc * hc; i++) comb_out[(size_t)r * hc * hc + i] = c[i];
        }
    }
    __syncthreads();
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) {
        float acc = 0.f;
        for (int n = 0; n < hc; n++) acc += pre[n] * streams[((size_t)r * hc + n) * hidden + d];
        collapsed[(size_t)r * hidden + d] = acc;
    }
}

__global__ void hc_combine_kernel(float *__restrict__ streams, const float *__restrict__ post, const float *__restrict__ comb, const float *__restrict__ block,
                                  int hc, int hidden, size_t total) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const size_t r = i / hidden, d = i % hidden;
        float old[HC_MAX];
        for (int j = 0; j < hc; j++) old[j] = streams[(r * hc + j) * hidden + d];
        const float b = block[i];
        for (int k = 0; k < hc; k++) {
            float mixed = old[k];
            if (comb) {
                mixed = 0.f;
                for (int j = 0; j < hc; j++) mixed += comb[(r * hc + j) * hc + k] * old[j];
            }
            streams[(r * hc + k) * hidden + d] = post[r * hc + k] * b + mixed;
        }
    }
}

int fllm_hc_supported(int hc) { return hc >= 1 && hc <= HC_MAX; }

int fllm_hc_mix_launch(cudaStream_t s, const float *streams, const float *mixes, const float *base, const float *scale, float *post, float *comb,
                       float *collapsed, int rows, int hc, int hidden, int mode, int iters, float eps, float magnitude) {
    if (rows <= 0) return 0;
    if (!fllm_hc_supported(hc)) return fllm_set_error("hc_mix: %d streams are more than %d", hc, HC_MAX);
    hc_mix_kernel<<<rows, 256, 0, s>>>(streams, mixes, base, scale, post, comb, collapsed, hc, hidden, mode, iters, eps, magnitude);
    return fllm_check_launch("hc_mix");
}

int fllm_hc_combine_launch(cudaStream_t s, float *streams, const float *post, const float *comb, const float *block, int rows, int hc, int hidden) {
    if (rows <= 0) return 0;
    if (!fllm_hc_supported(hc)) return fllm_set_error("hc_combine: %d streams are more than %d", hc, HC_MAX);
    const size_t total = (size_t)rows * hidden;
    size_t g = (total + 255) / 256;
    hc_combine_kernel<<<(unsigned)(g > 65535 ? 65535 : g), 256, 0, s>>>(streams, post, comb, block, hc, hidden, total);
    return fllm_check_launch("hc_combine");
}
