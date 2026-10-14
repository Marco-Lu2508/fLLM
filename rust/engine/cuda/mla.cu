// fLLM CUDA backend: multi-head latent attention (DeepSeek V2/V3, Kimi, GLM-MoE-DSA ...), expanded form.
//
// Mirrors `mla_qkv` in rust_decoder.rs (the reference): the projections run as ordinary GEMMs; these kernels do the data movement
// between them. Queries are laid out [rotary | content] per head, keys [shared rotary part | content], values as projected; the
// cache holds the rebuilt per-head keys and values (like the CPU path). Plain FP32 CUDA-core code, runs unchanged on Turing.
#include "common.cuh"
#include "fllm_cuda_internal.h"

// latent[r][c] = kv_a[r][c] for c < kv_lora (kv_a rows are kv_lora + rope wide)
__global__ void mla_latent_kernel(const float *__restrict__ kv_a, float *__restrict__ latent, size_t total, int kv_lora, int width) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const size_t r = i / kv_lora, c = i % kv_lora;
        latent[i] = kv_a[r * width + c];
    }
}

// q_full: rows x heads x (nope + rope) as projected ([content | rotary] per head); kv_a: rows x (kv_lora + rope) (the shared rotary key
// part follows the latent); kv: rows x heads x (nope + value).
__global__ void mla_assemble_kernel(const float *__restrict__ q_full, const float *__restrict__ kv_a, const float *__restrict__ kv,
                                    float *__restrict__ q, float *__restrict__ k, float *__restrict__ v, int rows, int heads, int nope,
                                    int rope, int value, int kv_lora) {
    const int dk = nope + rope;
    const size_t heads_rows = (size_t)rows * heads;
    const size_t total_qk = heads_rows * dk, total_v = heads_rows * value;
    const size_t total = total_qk > total_v ? total_qk : total_v;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        if (i < total_qk) {
            const size_t at = i / dk;
            const int d = (int)(i % dk);
            const size_t r = at / heads;
            if (d < rope) {
                q[i] = q_full[at * dk + nope + d];
                k[i] = kv_a[r * (kv_lora + rope) + kv_lora + d];
            } else {
                q[i] = q_full[at * dk + (d - rope)];
                k[i] = kv[at * (nope + value) + (d - rope)];
            }
        }
        if (i < total_v) {
            const size_t at = i / value;
            const int e = (int)(i % value);
            v[i] = kv[at * (nope + value) + nope + e];
        }
    }
}

static unsigned mla_grid(size_t n) {
    size_t g = (n + 255) / 256;
    return (unsigned)(g > 65535 ? 65535 : (g == 0 ? 1 : g));
}

int fllm_mla_latent_launch(cudaStream_t s, const float *kv_a, float *latent, int rows, int kv_lora, int width) {
    const size_t total = (size_t)rows * kv_lora;
    if (total == 0) return 0;
    mla_latent_kernel<<<mla_grid(total), 256, 0, s>>>(kv_a, latent, total, kv_lora, width);
    return fllm_check_launch("mla_latent");
}

int fllm_mla_assemble_launch(cudaStream_t s, const float *q_full, const float *kv_a, const float *kv, float *q, float *k, float *v, int rows,
                             int heads, int nope, int rope, int value, int kv_lora) {
    const size_t total = (size_t)rows * heads * ((nope + rope) > value ? (nope + rope) : value);
    if (total == 0) return 0;
    mla_assemble_kernel<<<mla_grid(total), 256, 0, s>>>(q_full, kv_a, kv, q, k, v, rows, heads, nope, rope, value, kv_lora);
    return fllm_check_launch("mla_assemble");
}
