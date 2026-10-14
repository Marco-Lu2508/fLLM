// FLLM-KERNEL: fllm_sigmoid_mul_f32 | buffers: x, gate | params: o_count_params | grid: [ceil(count / 256), 1, 1] x [256, 1, 1]
// Gated attention output (Stream::sigmoid_gate_mul): x[i] = x[i] * sigmoid(gate[i]) in place (count floats; `scale` is not used).
#include <metal_stdlib>
using namespace metal;

kernel void fllm_sigmoid_mul_f32(
    device float *x [[buffer(1)]],
    device const float *gate [[buffer(2)]],
    constant o_count_params &p [[buffer(0)]],
    uint i [[thread_position_in_grid]]) {
    if (i >= p.count) return;
    x[i] = x[i] * o_sigmoid(gate[i]);
}
