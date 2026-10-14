// FLLM-KERNEL: fllm_rope_f32 | buffers: q, k, positions, cos, sin | params: o_rope_params | grid: [ceil(half_dim / threads), heads + kv_heads, rows] x [threads, 1, 1]
// Rotary embedding in place (Stream::rope). Row r has position positions[r]; cos / sin are `table_rows x half_dim` tables (half_dim = rope_dim / 2).
// One thread rotates one pair of one head: (d, d + half_dim) for NeoX, (2d, 2d + 1) otherwise; query heads first (grid y < heads), then key heads.
// The dimensions of a head from rope_dim on (partial rotary) are not touched. q is `rows x heads x dim`, k is `rows x kv_heads x dim` (f32).
// Like CPU: a' = a cos - b sin, b' = b cos + a sin.
//
// The one-thread-per-pair layout with the head in the second grid dimension follows the MLX kernel rope.metal.
// Copyright © 2023-2024 Apple Inc. MIT License (the full text is in norm_rms.metal and in THIRD_PARTY.md).
#include <metal_stdlib>
using namespace metal;

kernel void fllm_rope_f32(
    device float *q [[buffer(1)]],
    device float *k [[buffer(2)]],
    device const uint *positions [[buffer(3)]],
    device const float *cs [[buffer(4)]],
    device const float *sn [[buffer(5)]],
    constant o_rope_params &p [[buffer(0)]],
    uint3 gid [[thread_position_in_grid]]) {
    const uint d = gid.x;
    if (d >= p.half_dim) return;
    const uint h = gid.y;
    const uint r = gid.z;
    const ulong at = ulong(positions[r]) * p.half_dim + d;
    const float c = cs[at];
    const float s = sn[at];
    const bool neox = p.neox != 0u;
    const uint ia = neox ? d : 2u * d;
    const uint ib = neox ? d + p.half_dim : 2u * d + 1u;
    if (h < p.heads) {
        const ulong base = (ulong(r) * p.heads + h) * p.dim;
        const float a = q[base + ia];
        const float b = q[base + ib];
        q[base + ia] = a * c - b * s;
        q[base + ib] = b * c + a * s;
    } else {
        const ulong base = (ulong(r) * p.kv_heads + (h - p.heads)) * p.dim;
        const float a = k[base + ia];
        const float b = k[base + ib];
        k[base + ia] = a * c - b * s;
        k[base + ib] = b * c + a * s;
    }
}
