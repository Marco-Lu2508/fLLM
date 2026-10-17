// FLLM-KERNEL: arch_clamp | constants: count,limit | buffers: 1 x(rw) | threads: 256,1,1
// FLLM-KERNEL: arch_softplus_mul | constants: count,beta | buffers: 1 x(rw), 2 gate(r) | threads: 256,1,1
// FLLM-KERNEL: arch_mla_latent | constants: total,kv_lora,width | buffers: 1 kv_a(r), 2 latent(w) | threads: 256,1,1
// FLLM-KERNEL: arch_mla_assemble | constants: rows,heads,nope,rope,value,kv_lora,total_qk,total_v | buffers: 1 q_full(r), 2 kv_a(r), 3 kv(r), 4 q(w), 5 k(w), 6 v(w) | threads: 256,1,1
// Element-wise kernels of the less common architectures: value clamping, the softplus attention output gate (K2 Horizon) and the data movement of
// multi-head latent attention in its expanded form (DeepSeek V2/V3, Kimi, GLM-MoE-DSA). They are the counterparts of cuda/arch_ops.cu and cuda/mla.cu
// and follow the same indexing. The constants struct is buffer 0; all four use grid-stride loops over `threads_per_grid`, launched with
// [min(ceil(items / 256), 65535), 1, 1] threadgroups of 256 threads.
//
// arch_clamp          Stream::clamp: x[i] = clamp(x[i], -limit, limit) in place over `count` floats.
// arch_softplus_mul   Stream::softplus_gate_mul: x[i] *= softplus(beta * g) / beta in place, with the CPU reference's shortcut softplus(s) = s for s > 20
//                     (here: x *= g, since s / beta = g). log1p does not exist in MSL; log(1 + exp(s)) differs from it by less than 1e-7 absolute for s > -15
//                     and is 0 instead of exp(s) below, where the gate output is negligible anyway.
// arch_mla_latent     Stream::mla_latent: latent[r][c] = kv_a[r][c] for c < kv_lora, rows of kv_a are `width` = kv_lora + rope floats; total = rows * kv_lora.
// arch_mla_assemble   Stream::mla_assemble: q_full is rows x heads x (nope + rope) as projected ([content | rotary] per head), kv_a rows x (kv_lora + rope)
//                     (the shared rotary key part follows the latent), kv rows x heads x (nope + value). Output per (row, head): q = [rotary | content],
//                     k = [shared rotary | content], v = value part. total_qk = rows * heads * (nope + rope), total_v = rows * heads * value.
#include <metal_stdlib>
using namespace metal;

struct arch_clamp_params {
    uint count;
    float limit;
};

kernel void arch_clamp(
    constant arch_clamp_params &p [[buffer(0)]],
    device float *x [[buffer(1)]],
    uint gid [[thread_position_in_grid]],
    uint grid_items [[threads_per_grid]]) {
    for (uint i = gid; i < p.count; i += grid_items) {
        x[i] = min(max(x[i], -p.limit), p.limit);
    }
}

struct arch_softplus_mul_params {
    uint count;
    float beta;
};

kernel void arch_softplus_mul(
    constant arch_softplus_mul_params &p [[buffer(0)]],
    device float *x [[buffer(1)]],
    device const float *gate [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint grid_items [[threads_per_grid]]) {
    for (uint i = gid; i < p.count; i += grid_items) {
        const float g = gate[i];
        const float scaled = p.beta * g;
        x[i] = x[i] * (scaled > 20.0f ? g : log(1.0f + exp(scaled)) / p.beta);
    }
}

struct arch_mla_latent_params {
    uint total;
    uint kv_lora;
    uint width;
};

kernel void arch_mla_latent(
    constant arch_mla_latent_params &p [[buffer(0)]],
    device const float *kv_a [[buffer(1)]],
    device float *latent [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint grid_items [[threads_per_grid]]) {
    for (uint i = gid; i < p.total; i += grid_items) {
        const ulong r = i / p.kv_lora;
        const uint c = i - uint(r) * p.kv_lora;
        latent[i] = kv_a[r * p.width + c];
    }
}

struct arch_mla_assemble_params {
    uint rows;
    uint heads;
    uint nope;
    uint rope;
    uint value;
    uint kv_lora;
    uint total_qk;
    uint total_v;
};

kernel void arch_mla_assemble(
    constant arch_mla_assemble_params &p [[buffer(0)]],
    device const float *q_full [[buffer(1)]],
    device const float *kv_a [[buffer(2)]],
    device const float *kv [[buffer(3)]],
    device float *q [[buffer(4)]],
    device float *k [[buffer(5)]],
    device float *v [[buffer(6)]],
    uint gid [[thread_position_in_grid]],
    uint grid_items [[threads_per_grid]]) {
    const uint dk = p.nope + p.rope;
    const uint total = max(p.total_qk, p.total_v);
    for (uint i = gid; i < total; i += grid_items) {
        if (i < p.total_qk) {
            const ulong at = i / dk;  // row * heads + head
            const uint d = i - uint(at) * dk;
            const ulong r = at / p.heads;
            if (d < p.rope) {
                q[i] = q_full[at * dk + p.nope + d];
                k[i] = kv_a[r * (p.kv_lora + p.rope) + p.kv_lora + d];
            } else {
                q[i] = q_full[at * dk + (d - p.rope)];
                k[i] = kv[at * (p.nope + p.value) + (d - p.rope)];
            }
        }
        if (i < p.total_v) {
            const ulong at = i / p.value;
            const uint e = i - uint(at) * p.value;
            v[i] = kv[at * (p.nope + p.value) + p.nope + e];
        }
    }
}
