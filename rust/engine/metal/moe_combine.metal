// FLLM-KERNEL: moe_combine | constants: tokens,k,hidden | buffers: 1 out(w), 2 y(r), 3 weights(r) | threads: 256,1,1
// (the constants struct is buffer 0; grid: [min(ceil(tokens*hidden/256), 65535), 1, 1] threadgroups)
// Stream::moe_combine: out[t][h] = sum_j weights[t * k + j] * y[(t * k + j) * hidden + h] for t < tokens, h < hidden (grid-stride loop, so any grid works).
// Port of shaders/vulkan/moe_combine.comp (package 3).

struct moe_combine_params {
    uint tokens;
    uint k;
    uint hidden;
};

kernel void moe_combine(
    constant moe_combine_params &p [[buffer(0)]],
    device float *o [[buffer(1)]],
    device const float *y [[buffer(2)]],
    device const float *weights [[buffer(3)]],
    uint gid [[thread_position_in_grid]],
    uint grid_items [[threads_per_grid]]) {
    const ulong total = ulong(p.tokens) * p.hidden;
    for (ulong i = gid; i < total; i += grid_items) {
        const ulong t = i / p.hidden;
        const ulong h = i - t * p.hidden;
        float acc = 0.0f;
        for (uint j = 0; j < p.k; j++) acc += weights[t * p.k + j] * y[(t * p.k + j) * p.hidden + h];
        o[i] = acc;
    }
}
