// FLLM-KERNEL: fllm_rms_norm_f32, fllm_rms_norm_f32x4 | buffers: x, weight, out | params: o_norm_params | grid: [rows, 1, 1] x [threads, 1, 1]
// RMSNorm (Stream::rms_norm): out[r][c] = x[r][c] * (1 / sqrt(mean(x[r]^2) + eps) * weight[c]); one threadgroup per row. x_stride / o_stride:
// floats between rows, x_off / o_off: first float of row 0 (so a slice of a wider row, e.g. Mamba B/C, can be normalised). flags bit 0: weight
// present (otherwise `weight` is not read and 1 is used). `out` may be the same buffer range as `x`. The `f32x4` variant (cols % 4 == 0) loads
// `packed_float4` (alignment 4, so any x_off works). threads: a multiple of 32, at most 512 (a 32-entry array holds one partial per simdgroup).
//
// The reduction structure (one row per threadgroup, simd_sum, partials combined through threadgroup memory) follows the MLX kernel rms_norm.metal.
// Copyright © 2023-2024 Apple Inc.
//
// MIT License
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the
// following conditions: The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
// Software. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE
// OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#include <metal_stdlib>
using namespace metal;

template <bool VEC>
[[kernel]] void o_rms_norm(
    device const float *x [[buffer(1)]],
    device const float *weight [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant o_norm_params &p [[buffer(0)]],
    uint row [[threadgroup_position_in_grid]],
    uint tid [[thread_position_in_threadgroup]],
    uint tg_size [[threads_per_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]],
    uint nsg [[simdgroups_per_threadgroup]]) {
    threadgroup float red[32];
    device const float *xr = x + p.x_off + ulong(row) * p.x_stride;
    device float *orow = out + p.o_off + ulong(row) * p.o_stride;
    const bool has_w = (p.flags & 1u) != 0u;

    float ss = 0.0f;
    if (VEC) {
        device const packed_float4 *x4 = (device const packed_float4 *)xr;
        for (uint c = tid; c < p.cols / 4u; c += tg_size) {
            const float4 v = float4(x4[c]);
            ss += dot(v, v);
        }
    } else {
        for (uint c = tid; c < p.cols; c += tg_size) {
            const float v = xr[c];
            ss += v * v;
        }
    }
    ss = o_tg_sum(ss, red, sg, lane, nsg);
    const float inv = 1.0f / sqrt(ss / float(p.cols) + p.eps);

    if (VEC) {
        device const packed_float4 *x4 = (device const packed_float4 *)xr;
        device const packed_float4 *w4 = (device const packed_float4 *)weight;
        device packed_float4 *o4 = (device packed_float4 *)orow;
        for (uint c = tid; c < p.cols / 4u; c += tg_size) {
            const float4 wv = has_w ? float4(w4[c]) : float4(1.0f);
            o4[c] = packed_float4(float4(x4[c]) * (inv * wv));
        }
    } else {
        for (uint c = tid; c < p.cols; c += tg_size) {
            const float wv = has_w ? weight[c] : 1.0f;
            orow[c] = xr[c] * (inv * wv);
        }
    }
}

template [[host_name("fllm_rms_norm_f32")]] [[kernel]] decltype(o_rms_norm<false>) o_rms_norm<false>;
template [[host_name("fllm_rms_norm_f32x4")]] [[kernel]] decltype(o_rms_norm<true>) o_rms_norm<true>;
