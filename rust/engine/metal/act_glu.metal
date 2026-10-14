// FLLM-KERNEL: fllm_act_swiglu, fllm_act_geglu, fllm_act_swiglu_oai, fllm_act_silu, fllm_act_gelu, fllm_act_relu2, fllm_act_relu | buffers: gate, up, out | params: o_act_params | grid: [ceil(width / threads), n_rows, 1] x [threads, 1, 1]
// Gated and plain activations, selected by the template parameter MODE (one kernel name per mode):
//   0 SWIGLU      out = silu(g) * u                                      (Stream::swiglu)
//   1 GEGLU       out = gelu_tanh(g) * u                                 (Stream::geglu; CPU: 0.5 g (1 + tanh(..)) u)
//   2 SWIGLU_OAI  g' = min(g, limit), u' = clamp(u, -limit, limit), out = g' / (1 + exp(-alpha g')) * (u' + 1)   (Stream::swiglu_oai)
//   3 SILU        out = silu(g)               (u is not read)
//   4 GELU        out = gelu_tanh(g)          (u is not read)
//   5 RELU2       out = max(g, 0)^2           (u is not read)
//   6 RELU        out = max(g, 0)             (u is not read)
// Element (r, c) with r = grid y, c = grid x: gate[gate_off + r * gate_stride + c], up[up_off + r * up_stride + c], out[r * out_stride + c],
// r < n_rows, c < width. Gate and up may be two ranges of the same buffer (fused gate/up projection); `out` may alias `gate`.
#include <metal_stdlib>
using namespace metal;

template <int MODE>
[[kernel]] void o_act_glu(
    device const float *gate [[buffer(1)]],
    device const float *up [[buffer(2)]],
    device float *out [[buffer(3)]],
    constant o_act_params &p [[buffer(0)]],
    uint2 gid [[thread_position_in_grid]]) {
    const uint c = gid.x;
    if (c >= p.width) return;
    const ulong r = gid.y;
    const float g = gate[p.gate_off + r * p.gate_stride + c];
    float result;
    if (MODE == 0) {
        result = o_silu(g) * up[p.up_off + r * p.up_stride + c];
    } else if (MODE == 1) {
        result = o_gelu_tanh(g) * up[p.up_off + r * p.up_stride + c];
    } else if (MODE == 2) {
        const float gc = min(g, p.limit);
        const float uc = clamp(up[p.up_off + r * p.up_stride + c], -p.limit, p.limit);
        result = gc / (1.0f + exp(-p.alpha * gc)) * (uc + 1.0f);
    } else if (MODE == 3) {
        result = o_silu(g);
    } else if (MODE == 4) {
        result = o_gelu_tanh(g);
    } else if (MODE == 5) {
        const float rl = max(g, 0.0f);
        result = rl * rl;
    } else {
        result = max(g, 0.0f);
    }
    out[r * p.out_stride + c] = result;
}

template [[host_name("fllm_act_swiglu")]] [[kernel]] decltype(o_act_glu<0>) o_act_glu<0>;
template [[host_name("fllm_act_geglu")]] [[kernel]] decltype(o_act_glu<1>) o_act_glu<1>;
template [[host_name("fllm_act_swiglu_oai")]] [[kernel]] decltype(o_act_glu<2>) o_act_glu<2>;
template [[host_name("fllm_act_silu")]] [[kernel]] decltype(o_act_glu<3>) o_act_glu<3>;
template [[host_name("fllm_act_gelu")]] [[kernel]] decltype(o_act_glu<4>) o_act_glu<4>;
template [[host_name("fllm_act_relu2")]] [[kernel]] decltype(o_act_glu<5>) o_act_glu<5>;
template [[host_name("fllm_act_relu")]] [[kernel]] decltype(o_act_glu<6>) o_act_glu<6>;
