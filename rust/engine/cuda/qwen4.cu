// fLLM CUDA backend: the pieces of Qwen3.8-Flash-Next (qwen4_exp) that its Qwen3.5 layers do not have. The reference is rust_decoder/qwen4.rs.
//
//   q4_act        x = factor * act(x * scale) in place, act = SiLU (0) or sigmoid (1)             (gate down-projection, injection weights)
//   q4_gate_mix   out[r][d] = sum_n sigmoid(gates[r][n][d]) * normed[r][n][d] / hc                  (block input from the normalised streams)
//   q4_ple_gate   out[r][n][d] = sigmoid(ssqrt(dot(key[r][n], query[r][n]) / sqrt(hidden))) * value[r][d]    (n-gram embedding gate)
//   q4_ple_conv   out[t][c] = gated[t][c] + silu(sum_j w[c][j] * ext[t + j * dilation][c]), ext = the channel's history, then the run
//   q4_pool_keys  token indexer: the pooled, RMS-normalised, rotated key of the block a row completes (the other rows keep their raw key)
//
// FP32 code with block-level primitives only (no warp intrinsics), so the same file builds for AMD GPUs.
#include "common.cuh"
#include "fllm_cuda_internal.h"

#define Q4_BLOCK 256
// longest history of the n-gram convolution: (kernel - 1) * dilation values per channel
#define Q4_HIST_MAX 32
// widest indexer key one block normalises and rotates
#define Q4_POOL_MAX_DIM 256
#define Q4_HC_MAX 8

__device__ __forceinline__ float q4_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// sum over the threads of the block (blockDim.x a power of two, at most Q4_BLOCK); every thread gets the total
__device__ __forceinline__ float q4_block_sum(float v, float *buf) {
    buf[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }
    const float total = buf[0];
    __syncthreads();
    return total;
}

__global__ void q4_act_kernel(float *__restrict__ x, size_t n, int mode, float scale, float factor) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const float v = x[i] * scale;
        x[i] = factor * (mode == 0 ? v / (1.0f + expf(-v)) : q4_sigmoid(v));
    }
}

__global__ void q4_gate_mix_kernel(float *__restrict__ out, const float *__restrict__ normed, const float *__restrict__ gates, int hc, int hidden, size_t total) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const size_t r = i / hidden, d = i % hidden;
        float acc = 0.f;
        for (int n = 0; n < hc; n++) {
            const size_t at = (r * hc + n) * hidden + d;
            acc += q4_sigmoid(gates[at]) * normed[at] / (float)hc;
        }
        out[i] = acc;
    }
}

// one block per (row, stream)
__global__ void q4_ple_gate_kernel(float *__restrict__ out, const float *__restrict__ key, const float *__restrict__ query, const float *__restrict__ value,
                                   int hc, int hidden) {
    __shared__ float buf[Q4_BLOCK];
    const size_t rn = blockIdx.x, r = rn / hc;
    float dot = 0.f;
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) dot += key[rn * hidden + d] * query[rn * hidden + d];
    const float total = q4_block_sum(dot, buf);
    float g = total / sqrtf((float)hidden);
    g = sqrtf(fmaxf(fabsf(g), 1e-6f)) * (g == 0.f ? 0.f : (g > 0.f ? 1.f : -1.f));
    const float gate = q4_sigmoid(g);
    for (int d = threadIdx.x; d < hidden; d += blockDim.x) out[rn * hidden + d] = gate * value[r * hidden + d];
}

// Rows of one sequence that are handled together (mirrors DeltaRun in backend/mod.rs and ConvRun in shortconv.cu).
struct Q4Run {
    unsigned long long state;
    int row0;
    int n;
    int reset;
    int pad;
};

// grid (ceil(width / 128), runs), block 128. The history of channel c is hist[c * H .. (c + 1) * H), H = (kernel - 1) * dilation, oldest first.
__global__ void __launch_bounds__(128) q4_ple_conv_kernel(float *__restrict__ out, const float *__restrict__ gated, const float *__restrict__ normed,
                                                          const float *__restrict__ convw, const Q4Run *__restrict__ runs, int width, int kernel,
                                                          int dilation, size_t state_off) {
    const Q4Run run = runs[blockIdx.y];
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= width) return;
    const int H = (kernel - 1) * dilation;
    float *hist = reinterpret_cast<float *>(reinterpret_cast<char *>(run.state) + state_off) + (size_t)c * H;
    float h[Q4_HIST_MAX];
    for (int i = 0; i < H; i++) h[i] = run.reset ? 0.f : hist[i];
    for (int t = 0; t < run.n; t++) {
        const size_t at = (size_t)(run.row0 + t) * width + c;
        const float cur = normed[at];
        float value = 0.f;
        for (int j = 0; j < kernel - 1; j++) value += convw[(size_t)c * kernel + j] * h[j * dilation];
        value += convw[(size_t)c * kernel + kernel - 1] * cur;
        out[at] = gated[at] + value / (1.0f + expf(-value));
        for (int i = 0; i + 1 < H; i++) h[i] = h[i + 1];
        h[H - 1] = cur;
    }
    for (int i = 0; i < H; i++) hist[i] = h[i];
}

// One block per row, blockDim.x a power of two >= dim. meta (ratio + 1 ints per row): the rotary position of the first token of the block the row
// completes, or -1 (the row keeps its key), then the source of each of the `ratio` keys of the block: >= 0 a row of `k`, < 0 the store slot -1 - value.
__global__ void q4_pool_keys_kernel(float *__restrict__ out, const float *__restrict__ k, const float *__restrict__ store, const float *__restrict__ norm,
                                    const float *__restrict__ cs, const float *__restrict__ sn, const int32_t *__restrict__ meta, int dim, int width,
                                    int ratio, int rope_half, int neox, float eps) {
    __shared__ float buf[Q4_BLOCK];
    __shared__ float rot[Q4_POOL_MAX_DIM];
    const size_t r = blockIdx.x;
    const int c = threadIdx.x;
    const int32_t *m = meta + r * (size_t)(ratio + 1);
    if (m[0] < 0) {
        if (c < dim) out[r * dim + c] = k[r * dim + c];
        return;
    }
    float mean = 0.f;
    if (c < dim) {
        for (int j = 0; j < ratio; j++) {
            const int32_t src = m[1 + j];
            mean += src >= 0 ? k[(size_t)src * dim + c] : store[(size_t)(-1 - src) * width + c];
        }
        mean /= (float)ratio;
    }
    const float ss = q4_block_sum(mean * mean, buf);
    const float inv = 1.0f / sqrtf(ss / (float)dim + eps);
    if (c < dim) rot[c] = mean * inv * norm[c];
    __syncthreads();
    if (c >= dim) return;
    float v = rot[c];
    if (c < 2 * rope_half) {
        const int p = m[0];
        if (neox) {
            const int i = c < rope_half ? c : c - rope_half;
            const float a = rot[i], b = rot[i + rope_half];
            const float co = cs[(size_t)p * rope_half + i], si = sn[(size_t)p * rope_half + i];
            float ra, rb;
            rope_pair(a, b, co, si, ra, rb);
            v = c < rope_half ? ra : rb;
        } else {
            const int i = c / 2;
            const float a = rot[2 * i], b = rot[2 * i + 1];
            const float co = cs[(size_t)p * rope_half + i], si = sn[(size_t)p * rope_half + i];
            float ra, rb;
            rope_pair(a, b, co, si, ra, rb);
            v = (c & 1) ? rb : ra;
        }
    }
    out[r * dim + c] = v;
}

static unsigned q4_grid(size_t n) {
    size_t g = (n + 255) / 256;
    return (unsigned)(g > 65535 ? 65535 : (g == 0 ? 1 : g));
}

int fllm_q4_supported(int hc, int kernel, int dilation) {
    return hc >= 1 && hc <= Q4_HC_MAX && kernel >= 1 && dilation >= 1 && (kernel - 1) * dilation <= Q4_HIST_MAX;
}

int fllm_q4_pool_supported(int dim, int rope_half) { return dim >= 2 && dim <= Q4_POOL_MAX_DIM && rope_half >= 0 && 2 * rope_half <= dim; }

int fllm_q4_act_launch(cudaStream_t s, float *x, size_t n, int mode, float scale, float factor) {
    if (n == 0) return 0;
    q4_act_kernel<<<q4_grid(n), 256, 0, s>>>(x, n, mode, scale, factor);
    return fllm_check_launch("q4_act");
}

int fllm_q4_gate_mix_launch(cudaStream_t s, float *out, const float *normed, const float *gates, int rows, int hc, int hidden) {
    if (rows <= 0) return 0;
    if (!fllm_q4_supported(hc, 1, 1)) return fllm_set_error("q4_gate_mix: %d streams", hc);
    const size_t total = (size_t)rows * hidden;
    q4_gate_mix_kernel<<<q4_grid(total), 256, 0, s>>>(out, normed, gates, hc, hidden, total);
    return fllm_check_launch("q4_gate_mix");
}

int fllm_q4_ple_gate_launch(cudaStream_t s, float *out, const float *key, const float *query, const float *value, int rows, int hc, int hidden) {
    if (rows <= 0) return 0;
    q4_ple_gate_kernel<<<(unsigned)((size_t)rows * hc), Q4_BLOCK, 0, s>>>(out, key, query, value, hc, hidden);
    return fllm_check_launch("q4_ple_gate");
}

int fllm_q4_ple_conv_launch(cudaStream_t s, const void *runs, int nruns, int width, int kernel, int dilation, size_t state_off, const float *gated,
                            const float *normed, const float *convw, float *out) {
    if (!fllm_q4_supported(1, kernel, dilation)) return fllm_set_error("q4_ple_conv: unsupported kernel %d dilation %d", kernel, dilation);
    if (nruns <= 0) return 0;
    q4_ple_conv_kernel<<<dim3((width + 127) / 128, nruns), 128, 0, s>>>(out, gated, normed, convw, static_cast<const Q4Run *>(runs), width, kernel, dilation,
                                                                        state_off);
    return fllm_check_launch("q4_ple_conv");
}

int fllm_q4_pool_keys_launch(cudaStream_t s, float *out, const float *k, const float *store, const float *norm, const float *cs, const float *sn,
                             const int32_t *meta, int rows, int dim, int width, int ratio, int rope_half, int neox, float eps) {
    if (rows <= 0) return 0;
    if (!fllm_q4_pool_supported(dim, rope_half) || ratio < 1) return fllm_set_error("q4_pool_keys: unsupported key width %d", dim);
    int threads = 32;
    while (threads < dim) threads *= 2;
    q4_pool_keys_kernel<<<rows, threads, 0, s>>>(out, k, store, norm, cs, sn, meta, dim, width, ratio, rope_half, neox, eps);
    return fllm_check_launch("q4_pool_keys");
}
