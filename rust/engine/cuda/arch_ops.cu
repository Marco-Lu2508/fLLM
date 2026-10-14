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

static unsigned arch_grid(size_t n) {
    size_t g = (n + 255) / 256;
    return (unsigned)(g > 65535 ? 65535 : (g == 0 ? 1 : g));
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
