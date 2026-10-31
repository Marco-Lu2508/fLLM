// fLLM CUDA backend: Gated DeltaNet layer (Qwen3.5 / Qwen3.6 / Qwen3-Next) on the device.
//
// Mirrors rust/engine/src/cpu_delta.rs (the reference): causal convolution with SiLU over the channels [q|k|v], L2 normalisation of q
// and k per head, decay exp(A * softplus(alpha + dt_bias)), write strength beta_scale * sigmoid(beta), the delta-rule update
//     S = S * decay;  S += k (x) ((v - S^T k) * strength);  o = S^T q
// and the gated RMS norm norm(o) * weight * silu(z). Two kernels per layer:
//   1. delta_conv_kernel: convolution + SiLU + head normalisation, tokens of a run in chunks; keeps the convolution history in VRAM.
//   2. delta_rec_kernel<HK>: one block per (run, value head), one thread per value column; the column of the state (HK values) lives
//      in registers while the block walks the tokens of the run in order. Gates, readout and the gated norm are fused in.
// The recurrent state is FP32, S[head][key row][value column], one buffer per sequence (see rust_decoder.rs). Plain FP32 CUDA-core
// code (shared memory, warp shuffles): runs unchanged on Turing.
#include "common.cuh"
#include "fllm_cuda_internal.h"

// Rows of one sequence that are handled together (mirrors DeltaRun in backend/mod.rs).
struct DeltaRun {
    unsigned long long state;  // device address of the sequence's state buffer
    int row0;                  // first row of the run in the activations
    int n;                     // number of rows (consecutive positions)
    int reset;                 // 1: the sequence starts here, the old state is ignored
    int pad;
};

#define DELTA_CHUNK 16  // tokens per conv block (must be >= conv_kernel - 1)

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

// grid (ceil(conv_dim / hk), runs, ceil(max_n / DELTA_CHUNK)), block hk threads. Block x < kh normalises a query head, x in [kh, 2 kh)
// a key head (blocks line up with the heads, so the L2 norm is a block reduction).
__global__ void __launch_bounds__(128) delta_conv_kernel(const float *__restrict__ qkv, const float *__restrict__ convw,
                                                         const DeltaRun *__restrict__ runs, float *__restrict__ mixed,
                                                         int conv_dim, int kh, int ksz, size_t state_off, size_t rec_bytes,
                                                         float qscale) {
    __shared__ float scratch[32];
    const DeltaRun run = runs[blockIdx.y];
    const int t0 = blockIdx.z * DELTA_CHUNK;
    if (t0 >= run.n) return;
    const int t1 = min(run.n, t0 + DELTA_CHUNK);
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    const bool valid = c < conv_dim;
    const bool head = blockIdx.x < 2 * kh;
    if (!valid && !head) return;
    const int hist_len = ksz - 1;
    float *hist = reinterpret_cast<float *>(reinterpret_cast<char *>(run.state) + state_off + rec_bytes) + (size_t)c * hist_len;
    const float *x = qkv + (size_t)run.row0 * conv_dim + c;

    float w[8], win[8];
    for (int k = 0; k < ksz; k++) w[k] = valid ? convw[(size_t)c * ksz + k] : 0.f;
    // window[j] = input of token (t - hist_len + j)
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
            acc = blockIdx.x < kh ? acc * inv * qscale : acc * inv;
        }
        mixed[(size_t)(run.row0 + t) * conv_dim + c] = acc;
        for (int k = 0; k + 1 < hist_len; k++) win[k] = win[k + 1];
        win[hist_len - 1] = cur;
    }
    // the first chunk's threads own the history of their channel: all readers of the old values are these same threads
    if (blockIdx.z == 0 && valid) {
        float fresh[8];
        for (int k = 0; k < hist_len; k++) {
            const int idx = run.n + k - hist_len;
            fresh[k] = idx >= 0 ? x[(size_t)idx * conv_dim] : (run.reset ? 0.f : hist[idx + hist_len]);
        }
        for (int k = 0; k < hist_len; k++) hist[k] = fresh[k];
    }
}

// grid (v_heads, runs), block 2 * hv threads. Value column c = tid / 2 is shared by a pair of neighbouring lanes: lane half 0 holds the
// first HK / 2 rows of the column's state in registers, half 1 the others (the dot products over the key dimension meet through a shuffle).
template <int HK>
__global__ void __launch_bounds__(256) delta_rec_kernel(const float *__restrict__ mixed, const float *__restrict__ z,
                                                        const float *__restrict__ beta, const float *__restrict__ alpha,
                                                        const float *__restrict__ dt_bias, const float *__restrict__ decay,
                                                        const float *__restrict__ norm, const DeltaRun *__restrict__ runs,
                                                        float *__restrict__ out, int kh, int vh, int conv_dim, int tiled,
                                                        size_t state_off, float beta_scale, float eps) {
    constexpr int HALF = HK / 2;
    __shared__ float scratch[32];
    const DeltaRun run = runs[blockIdx.y];
    const int j = blockIdx.x;
    const int hv = blockDim.x / 2;
    const int c = threadIdx.x >> 1;
    const int half = threadIdx.x & 1;
    const int key_dim = kh * HK;
    const int value_dim = vh * hv;
    const int src = tiled ? j % kh : j / (vh / kh);
    float *S = reinterpret_cast<float *>(reinterpret_cast<char *>(run.state) + state_off) + ((size_t)j * HK + half * HALF) * hv + c;

    float s[HALF];
#pragma unroll
    for (int i = 0; i < HALF; i++) s[i] = run.reset ? 0.f : S[(size_t)i * hv];

    const float dtb = dt_bias[j], dec = decay[j], nw = norm[c];
    for (int t = 0; t < run.n; t++) {
        const size_t row = (size_t)(run.row0 + t);
        const float *m = mixed + row * conv_dim;
        const float4 *q4 = reinterpret_cast<const float4 *>(m + src * HK + half * HALF);
        const float4 *k4 = reinterpret_cast<const float4 *>(m + key_dim + src * HK + half * HALF);
        const float raw = alpha[row * vh + j] + dtb;
        const float softplus = raw > 20.f ? raw : log1pf(expf(raw));
        const float forget = expf(dec * softplus);
        const float sigma = beta_scale / (1.0f + expf(-beta[row * vh + j]));
        const float v = m[2 * key_dim + j * hv + c];

        float memory = 0.f;
#pragma unroll
        for (int i = 0; i < HALF / 4; i++) {
            const float4 k = k4[i];
            s[4 * i + 0] *= forget;
            memory += s[4 * i + 0] * k.x;
            s[4 * i + 1] *= forget;
            memory += s[4 * i + 1] * k.y;
            s[4 * i + 2] *= forget;
            memory += s[4 * i + 2] * k.z;
            s[4 * i + 3] *= forget;
            memory += s[4 * i + 3] * k.w;
        }
        memory += __shfl_xor_sync(0xffffffffu, memory, 1);
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
            const float gate = silu_f(z[row * value_dim + j * hv + c]);
            out[row * value_dim + j * hv + c] = acc * inv * nw * gate;
        }
    }
#pragma unroll
    for (int i = 0; i < HALF; i++) S[(size_t)i * hv] = s[i];
}

int fllm_delta_supported(int kh, int vh, int hk, int hv, int ksz) {
    return (hk == 64 || hk == 128) && hv >= 32 && hv <= 128 && hv % 32 == 0 && kh > 0 && vh > 0 && vh % kh == 0 && ksz >= 2 && ksz <= 8;
}

// `runs`: every run (the convolution handles all of them); `rec_runs`: the runs the serial recurrence handles (the others take the chunked scan of scan.cu).
int fllm_delta_launch(cudaStream_t s, const void *runs, int nruns, int max_n, const void *rec_runs, int nrec, int kh, int vh, int hk, int hv, int ksz,
                      int tiled, float beta_scale, float eps, size_t state_off, size_t rec_bytes, const float *qkv, const float *z, const float *beta,
                      const float *alpha, const float *convw, const float *dt_bias, const float *decay, const float *norm, float *mixed,
                      float *out) {
    if (nruns <= 0 || max_n <= 0) return 0;
    if (!fllm_delta_supported(kh, vh, hk, hv, ksz)) return fllm_set_error("DeltaNet shape not supported by the CUDA kernels");
    const int key_dim = kh * hk, value_dim = vh * hv;
    const int conv_dim = 2 * key_dim + value_dim;
    const DeltaRun *r = static_cast<const DeltaRun *>(runs);
    const dim3 grid((conv_dim + hk - 1) / hk, nruns, (max_n + DELTA_CHUNK - 1) / DELTA_CHUNK);
    delta_conv_kernel<<<grid, hk, 0, s>>>(qkv, convw, r, mixed, conv_dim, kh, ksz, state_off, rec_bytes, 1.0f / sqrtf((float)hk));
    if (int e = fllm_check_launch("delta_conv")) return e;
    if (nrec <= 0) return 0;
    const DeltaRun *rr = static_cast<const DeltaRun *>(rec_runs);
    const dim3 g2(vh, nrec);
    if (hk == 128)
        delta_rec_kernel<128><<<g2, 2 * hv, 0, s>>>(mixed, z, beta, alpha, dt_bias, decay, norm, rr, out, kh, vh, conv_dim, tiled, state_off,
                                                beta_scale, eps);
    else
        delta_rec_kernel<64><<<g2, 2 * hv, 0, s>>>(mixed, z, beta, alpha, dt_bias, decay, norm, rr, out, kh, vh, conv_dim, tiled, state_off,
                                               beta_scale, eps);
    return fllm_check_launch("delta_rec");
}
