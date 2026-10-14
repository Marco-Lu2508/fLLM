// FLLM-KERNEL: fllm_split_q_gate_f32 | buffers: projected, q, gate | params: o_split_params | grid: [ceil(dk / threads), heads, rows] x [threads, 1, 1]
// Gated attention (Stream::split_q_gate): the query projection holds query and gate of every head. Row r of `projected` has 2 * heads * dk floats:
//   per_head != 0: [q_h0 (dk), gate_h0 (dk), q_h1, gate_h1, ...]     otherwise: [q_h0 .. q_h(heads-1), gate_h0 .. gate_h(heads-1)].
// q and gate are `rows x heads x dk`. One thread per output element (x = d, y = head, z = row).
#include <metal_stdlib>
using namespace metal;

kernel void fllm_split_q_gate_f32(
    device const float *projected [[buffer(1)]],
    device float *q [[buffer(2)]],
    device float *gate [[buffer(3)]],
    constant o_split_params &p [[buffer(0)]],
    uint3 gid [[thread_position_in_grid]]) {
    const uint d = gid.x;
    if (d >= p.dk) return;
    const ulong h = gid.y;
    const ulong r = gid.z;
    const ulong width = ulong(p.heads) * p.dk;
    const ulong base = r * 2ul * width;
    const ulong sq = p.per_head != 0u ? base + h * 2ul * p.dk + d : base + h * p.dk + d;
    const ulong sg = p.per_head != 0u ? sq + p.dk : base + width + h * p.dk + d;
    const ulong dst = r * width + h * p.dk + d;
    q[dst] = projected[sq];
    gate[dst] = projected[sg];
}
