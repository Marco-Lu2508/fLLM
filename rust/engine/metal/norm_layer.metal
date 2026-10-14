// FLLM-KERNEL: fllm_layer_norm_f32, fllm_layer_norm_f32x4 | buffers: x, weight, bias, out | params: o_norm_params | grid: [rows, 1, 1] x [threads, 1, 1]
// LayerNorm (Stream::layer_norm): out = (x - mean) * (1 / sqrt(var + eps)) * weight + bias with the biased variance (CPU backend: two passes,
// mean first, then the sum of squared deviations); one threadgroup per row. flags bit 0: bias present (otherwise `bias` is not read).
// `out` may be the same buffer range as `x`. The `f32x4` variant needs cols % 4 == 0. threads: a multiple of 32, at most 512.
#include <metal_stdlib>
using namespace metal;

template <bool VEC>
[[kernel]] void o_layer_norm(
    device const float *x [[buffer(1)]],
    device const float *weight [[buffer(2)]],
    device const float *bias [[buffer(3)]],
    device float *out [[buffer(4)]],
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
    const bool has_b = (p.flags & 1u) != 0u;
    device const packed_float4 *x4 = (device const packed_float4 *)xr;

    float s = 0.0f;
    if (VEC) {
        for (uint c = tid; c < p.cols / 4u; c += tg_size) {
            const float4 v = float4(x4[c]);
            s += v.x + v.y + v.z + v.w;
        }
    } else {
        for (uint c = tid; c < p.cols; c += tg_size) s += xr[c];
    }
    const float mean = o_tg_sum(s, red, sg, lane, nsg) / float(p.cols);

    float sq = 0.0f;
    if (VEC) {
        for (uint c = tid; c < p.cols / 4u; c += tg_size) {
            const float4 v = float4(x4[c]) - mean;
            sq += dot(v, v);
        }
    } else {
        for (uint c = tid; c < p.cols; c += tg_size) {
            const float v = xr[c] - mean;
            sq += v * v;
        }
    }
    const float var = o_tg_sum(sq, red, sg, lane, nsg) / float(p.cols);
    const float inv = 1.0f / sqrt(var + p.eps);

    if (VEC) {
        device const packed_float4 *w4 = (device const packed_float4 *)weight;
        device const packed_float4 *b4 = (device const packed_float4 *)bias;
        device packed_float4 *o4 = (device packed_float4 *)orow;
        for (uint c = tid; c < p.cols / 4u; c += tg_size) {
            const float4 bv = has_b ? float4(b4[c]) : float4(0.0f);
            o4[c] = packed_float4((float4(x4[c]) - mean) * inv * float4(w4[c]) + bv);
        }
    } else {
        for (uint c = tid; c < p.cols; c += tg_size) {
            const float bv = has_b ? bias[c] : 0.0f;
            orow[c] = (xr[c] - mean) * inv * weight[c] + bv;
        }
    }
}

template [[host_name("fllm_layer_norm_f32")]] [[kernel]] decltype(o_layer_norm<false>) o_layer_norm<false>;
template [[host_name("fllm_layer_norm_f32x4")]] [[kernel]] decltype(o_layer_norm<true>) o_layer_norm<true>;
