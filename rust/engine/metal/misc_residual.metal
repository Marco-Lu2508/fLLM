// FLLM-KERNEL: fllm_residual_add_f32 | buffers: x, residual | params: o_count_params | grid: [ceil(count / 256), 1, 1] x [256, 1, 1]
// Stream::residual_add: x[i] += residual[i] * scale in place (count floats).
#include <metal_stdlib>
using namespace metal;

kernel void fllm_residual_add_f32(
    device float *x [[buffer(1)]],
    device const float *residual [[buffer(2)]],
    constant o_count_params &p [[buffer(0)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= p.count) return;
    x[i] = x[i] + residual[i] * p.scale;
}
