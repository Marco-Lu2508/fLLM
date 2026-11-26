// fLLM CUDA backend: small element-wise operations of the older / less common architectures (Falcon, GPT-NeoX, Phi, StarCoder2,
// Bloom, OPT, ...): plain (non-gated) activations, gated activations beyond SwiGLU / GeGLU, and value clamping.
// Mirrors `activate` in rust_decoder.rs (the reference). FP32 CUDA-core code, runs unchanged on Turing.
#include "common.cuh"
#include "fllm_cuda_internal.h"

// 0 SiLU, 1 GELU (tanh), 2 GELU (erf), 3 ReLU, 4 ReLU^2
__device__ __forceinline__ float arch_act(float x, int mode) {
    switch (mode) {
        case 0: return x / (1.0f + expf(-x));
        case 1: return 0.5f * x * (1.0f + tanhf(0.7978846f * (x + 0.044715f * x * x * x)));
        case 2: return 0.5f * x * (1.0f + erff(x * 0.70710678f));
        case 3: return fmaxf(x, 0.0f);
        default: {
            const float r = fmaxf(x, 0.0f);
            return r * r;
        }
    }
}

__global__ void arch_act_kernel(const float *__restrict__ x, float *__restrict__ out, size_t n, int mode) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) out[i] = arch_act(x[i], mode);
}

__global__ void arch_act_glu_kernel(const float *__restrict__ gate, const float *__restrict__ up, float *__restrict__ out, size_t n,
                                    int mode) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        out[i] = arch_act(gate[i], mode) * up[i];
}

__global__ void arch_clamp_kernel(float *__restrict__ x, size_t n, float limit) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        x[i] = fminf(fmaxf(x[i], -limit), limit);
}

// x *= softplus(beta * g) / beta (K2 Horizon attention output gate); like AttnGate::Softplus
__global__ void arch_softplus_mul_kernel(float *__restrict__ x, const float *__restrict__ g, size_t n, float beta) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const float scaled = beta * g[i];
        x[i] *= scaled > 20.f ? g[i] : log1pf(expf(scaled)) / beta;
    }
}

// x[i] *= act(g[i / dv]): per-head attention output gate (Laguna, Step-3.5); beta <= 0 selects the sigmoid, beta > 0 softplus(beta g) / beta
__global__ void arch_head_gate_kernel(float *__restrict__ x, const float *__restrict__ g, size_t n, size_t dv, float beta) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const float gv = g[i / dv];
        float a;
        if (beta > 0.f) {
            const float scaled = beta * gv;
            a = scaled > 20.f ? gv : log1pf(expf(scaled)) / beta;
        } else {
            a = 1.0f / (1.0f + expf(-gv));
        }
        x[i] *= a;
    }
}

__global__ void arch_scale_kernel(float *__restrict__ x, size_t n, float factor) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) x[i] *= factor;
}

// Final logits in the order of the host path (rust_decoder.rs): x = x * scale, then cap * tanh(x / cap) when cap > 0 (final_logit_softcapping).
__global__ void arch_logit_cap_kernel(float *__restrict__ x, size_t n, float scale, float cap) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        // the scale in f32 like the host; the cap in double: with --use_fast_math tanhf is the hardware approximation (about 1e-5 relative,
        // enough to swap two near-tied tokens against the host path), the double tanh is exact to the f32 rounding
        const float v = x[i] * scale;
        x[i] = cap > 0.f ? (float)((double)cap * tanh((double)v / (double)cap)) : v;
    }
}

// Clamped gated activations (Hy4 / GLM-5-Next clamped SwiGLU, Kimi K3 SiTU), like `gate_activation_clamped` of rust_decoder.rs:
//   kind 0: g = min(gate, p0); out = g * sigmoid(g) * clamp(up, -p0, p0)
//   kind 1: out = p0 * tanh(gate / p0) * sigmoid(gate) * u, u = p1 > 0 ? p1 * tanh(up / p1) : up
__global__ void arch_glu_ext_kernel(const float *__restrict__ gate, const float *__restrict__ up, float *__restrict__ out, size_t n, int kind, float p0, float p1) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const float g0 = gate[i];
        if (kind == 0) {
            const float g = fminf(g0, p0);
            out[i] = g / (1.0f + expf(-g)) * fminf(fmaxf(up[i], -p0), p0);
        } else {
            const float u = p1 > 0.f ? p1 * tanhf(up[i] / p1) : up[i];
            out[i] = p0 * tanhf(g0 / p0) / (1.0f + expf(-g0)) * u;
        }
    }
}

// xIELU of Apertus' plain feed-forward (like the host path of rust_decoder.rs): x > 0: ap * x^2 + beta * x, else an * (expm1(min(x, eps)) - x) + beta * x
__global__ void arch_xielu_kernel(const float *__restrict__ x, float *__restrict__ out, size_t n, float ap, float an, float beta, float eps) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        const float v = x[i];
        out[i] = v > 0.f ? ap * v * v + beta * v : (expm1f(fminf(v, eps)) - v) * an + beta * v;
    }
}

static unsigned arch_grid(size_t n) {
    size_t g = (n + 255) / 256;
    return (unsigned)(g > 65535 ? 65535 : (g == 0 ? 1 : g));
}

int fllm_xielu_launch(cudaStream_t s, const float *x, float *out, size_t n, float ap, float an, float beta, float eps) {
    if (n == 0) return 0;
    arch_xielu_kernel<<<arch_grid(n), 256, 0, s>>>(x, out, n, ap, an, beta, eps);
    return fllm_check_launch("xielu");
}
int fllm_act_launch(cudaStream_t s, const float *x, float *out, size_t n, int mode) {
    if (n == 0) return 0;
    arch_act_kernel<<<arch_grid(n), 256, 0, s>>>(x, out, n, mode);
    return fllm_check_launch("act");
}
int fllm_act_glu_launch(cudaStream_t s, const float *gate, const float *up, float *out, size_t n, int mode) {
    if (n == 0) return 0;
    arch_act_glu_kernel<<<arch_grid(n), 256, 0, s>>>(gate, up, out, n, mode);
    return fllm_check_launch("act_glu");
}
int fllm_clamp_launch(cudaStream_t s, float *x, size_t n, float limit) {
    if (n == 0) return 0;
    arch_clamp_kernel<<<arch_grid(n), 256, 0, s>>>(x, n, limit);
    return fllm_check_launch("clamp");
}
int fllm_softplus_mul_launch(cudaStream_t s, float *x, const float *g, size_t n, float beta) {
    if (n == 0) return 0;
    arch_softplus_mul_kernel<<<arch_grid(n), 256, 0, s>>>(x, g, n, beta);
    return fllm_check_launch("softplus_mul");
}
int fllm_head_gate_launch(cudaStream_t s, float *x, const float *g, size_t n, size_t dv, float beta) {
    if (n == 0) return 0;
    arch_head_gate_kernel<<<arch_grid(n), 256, 0, s>>>(x, g, n, dv, beta);
    return fllm_check_launch("head_gate");
}
int fllm_glu_ext_launch(cudaStream_t s, const float *gate, const float *up, float *out, size_t n, int kind, float p0, float p1) {
    if (n == 0) return 0;
    arch_glu_ext_kernel<<<arch_grid(n), 256, 0, s>>>(gate, up, out, n, kind, p0, p1);
    return fllm_check_launch("glu_ext");
}
int fllm_scale_launch(cudaStream_t s, float *x, size_t n, float factor) {
    if (n == 0) return 0;
    arch_scale_kernel<<<arch_grid(n), 256, 0, s>>>(x, n, factor);
    return fllm_check_launch("scale");
}
int fllm_logit_cap_launch(cudaStream_t s, float *x, size_t n, float scale, float cap) {
    if (n == 0) return 0;
    arch_logit_cap_kernel<<<arch_grid(n), 256, 0, s>>>(x, n, scale, cap);
    return fllm_check_launch("logit_cap");
}

// DiffLlama (differential attention): the value of key head g is [v[g % m], v[m + g % m]] with m = kv_heads / 2 value heads of width d.
__global__ void diff_expand_v_kernel(const float *__restrict__ v, float *__restrict__ out, size_t rows, int kv_heads, int d) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = rows * kv_heads * 2 * (size_t)d;
    if (i >= total) return;
    const int j = (int)(i % d);
    const size_t rest = i / d;
    const int half = (int)(rest % 2);
    const size_t rg = rest / 2;
    const int g = (int)(rg % kv_heads);
    const size_t r = rg / kv_heads;
    const int m = kv_heads / 2;
    const int src = (half == 0 ? 0 : m) + g % m;
    out[i] = v[(r * kv_heads + src) * d + j];
}
int fllm_diff_expand_v_launch(cudaStream_t s, float *out, const float *v, size_t rows, int kv_heads, int d) {
    const size_t total = rows * kv_heads * 2 * (size_t)d;
    if (total == 0) return 0;
    diff_expand_v_kernel<<<arch_grid(total), 256, 0, s>>>(v, out, rows, kv_heads, d);
    return fllm_check_launch("diff_expand_v");
}

// block (row, pair p): z = a[p] - lambda a[p + half]; out = z * rsqrt(mean(z^2) + eps) * scale
__global__ void diff_attn_mix_kernel(const float *__restrict__ a, float *__restrict__ out, int heads, int width, float lambda, float scale, float eps) {
    const int half = heads / 2;
    const size_t r = blockIdx.x / half;
    const int p = blockIdx.x % half;
    const float *x = a + (r * heads + p) * (size_t)width;
    const float *y = a + (r * heads + p + half) * (size_t)width;
    float *o = out + (r * half + p) * (size_t)width;
    __shared__ float red[256];
    float sum = 0.f;
    for (int j = threadIdx.x; j < width; j += blockDim.x) {
        const float z = x[j] - lambda * y[j];
        sum += z * z;
    }
    red[threadIdx.x] = sum;
    __syncthreads();
    for (int s2 = blockDim.x / 2; s2 > 0; s2 >>= 1) {
        if ((int)threadIdx.x < s2) red[threadIdx.x] += red[threadIdx.x + s2];
        __syncthreads();
    }
    const float inv = rsqrtf(red[0] / (float)width + eps) * scale;
    for (int j = threadIdx.x; j < width; j += blockDim.x) o[j] = (x[j] - lambda * y[j]) * inv;
}
int fllm_diff_attn_mix_launch(cudaStream_t s, float *out, const float *attn, size_t rows, int heads, int width, float lambda, float scale, float eps) {
    if (rows == 0) return 0;
    diff_attn_mix_kernel<<<(unsigned)(rows * (heads / 2)), 256, 0, s>>>(attn, out, heads, width, lambda, scale, eps);
    return fllm_check_launch("diff_attn_mix");
}
