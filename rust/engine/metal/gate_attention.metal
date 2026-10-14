// FLLM-KERNEL: gate_split_q | constants: total,heads,dk,per_head | buffers: 1 projected(r), 2 q(w), 3 gate(w) | threads: 256,1,1
// FLLM-KERNEL: gate_sigmoid_mul | constants: count | buffers: 1 x(rw), 2 gate(r) | threads: 256,1,1
// Gated attention (Qwen3-Next / Qwen3.5 style: the query projection also holds an output gate per head). The constants struct is buffer 0; grid of both:
// [min(ceil(items/256), 65535), 1, 1] threadgroups.
//
// gate_split_q
//   Stream::split_q_gate: row r of `projected` has 2 * heads * dk floats:
//     per_head != 0: [q_h0 (dk), gate_h0 (dk), q_h1, gate_h1, ...]     otherwise: [q_h0 .. q_h(heads-1), gate_h0 .. gate_h(heads-1)].
//   q and gate are `rows x heads x dk`; total = rows * heads * dk. Grid-stride loop, one element per iteration.
// gate_sigmoid_mul
//   Stream::sigmoid_gate_mul: x[i] = x[i] * sigmoid(gate[i]) in place (the gated attention output before the output projection).
// Ports of shaders/vulkan/misc_split_q_gate.comp and act_sigmoid_mul.comp (package 3).

struct gate_split_q_params {
    uint total;
    uint heads;
    uint dk;
    uint per_head;
};

kernel void gate_split_q(
    constant gate_split_q_params &p [[buffer(0)]],
    device const float *projected [[buffer(1)]],
    device float *q [[buffer(2)]],
    device float *gate [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint grid_items [[threads_per_grid]]) {
    const uint width = p.heads * p.dk;
    for (ulong i = gid; i < p.total; i += grid_items) {
        const uint r = uint(i / width);
        const uint rem = uint(i - ulong(r) * width);
        const uint h = rem / p.dk;
        const uint d = rem - h * p.dk;
        const ulong base = ulong(r) * 2u * width;
        const ulong sq = p.per_head != 0u ? base + h * 2u * p.dk + d : base + h * p.dk + d;
        const ulong sg = p.per_head != 0u ? sq + p.dk : base + width + h * p.dk + d;
        q[i] = projected[sq];
        gate[i] = projected[sg];
    }
}

struct gate_sigmoid_mul_params {
    uint count;
};

kernel void gate_sigmoid_mul(
    constant gate_sigmoid_mul_params &p [[buffer(0)]],
    device float *x [[buffer(1)]],
    device const float *gate [[buffer(2)]],
    uint gid [[thread_position_in_grid]],
    uint grid_items [[threads_per_grid]]) {
    for (uint i = gid; i < p.count; i += grid_items) {
        x[i] = x[i] * r_sigmoid(gate[i]);
    }
}
