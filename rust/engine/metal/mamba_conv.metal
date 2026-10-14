// FLLM-KERNEL: mamba_conv | constants: channels,ksz,in_stride,in_off,row0,n,reset,hist_off,has_bias | buffers: 1 x(r), 2 convw(r), 3 convb(r), 4 state(rw), 5 mixed(w) | threads: 64,1,1
// (the constants struct is buffer 0; grid per run: [ceil(channels/64), ceil(n/16), 1] threadgroups)
// Mamba (1 and 2) causal depthwise convolution with SiLU over one run (n consecutive tokens of one sequence), reference: the loop over `mixed` in
// `mamba_forward` / `mamba1_forward` of rust_decoder.rs (port of shaders/vulkan/mamba_conv.comp, package 3):
//   mixed[t][c] = silu( bias[c] + sum_{k < ksz-1} w[c][k] * x[t - (ksz-1) + k][c] + w[c][ksz-1] * x[t][c] )
// with x before the run taken from the stored history (oldest first), or 0 when reset != 0. x[t][c] = x[(row0 + t) * in_stride + in_off + c] (a slice of
// the projected row: Mamba-2 xBC = in_off inner, Mamba-1 xs = in_off 0); mixed is dense, mixed[(row0 + t) * channels + c]. The history of channel c is
// state[hist_off + c * (ksz-1) + k]; the first chunk of 16 tokens (>= ksz - 1) rewrites it at the end. ksz in 2..8, has_bias != 0: convb[c] is added.

#define MAMBA_CONV_WG 64u

struct mamba_conv_params {
    uint channels;
    uint ksz;
    uint in_stride;
    uint in_off;
    uint row0;
    uint n;
    uint reset;
    uint hist_off;
    uint has_bias;
};

kernel void mamba_conv(
    constant mamba_conv_params &p [[buffer(0)]],
    device const float *x [[buffer(1)]],
    device const float *convw [[buffer(2)]],
    device const float *convb [[buffer(3)]],
    device float *state [[buffer(4)]],
    device float *mixed [[buffer(5)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    const uint c = tgp.x * MAMBA_CONV_WG + lane;
    const uint t0 = tgp.y * 16u;
    if (c >= p.channels || t0 >= p.n) return;  // no barriers below: threads may leave independently
    const uint t1 = min(p.n, t0 + 16u);
    const uint hist_len = p.ksz - 1u;
    const uint hist = p.hist_off + c * hist_len;
    const float bias = p.has_bias != 0u ? convb[c] : 0.0f;

    float wk[8];
    float win[8];  // win[j] = input of token (t - hist_len + j)
    R_UNROLL for (uint k = 0u; k < 8u; k++) {
        wk[k] = k < p.ksz ? convw[c * p.ksz + k] : 0.0f;
        win[k] = 0.0f;
    }
    R_UNROLL for (uint j = 0u; j < 7u; j++) {
        if (j < hist_len) {
            float v = 0.0f;
            if (t0 + j >= hist_len) {
                v = x[ulong(p.row0 + t0 + j - hist_len) * p.in_stride + p.in_off + c];
            } else if (p.reset == 0u) {
                v = state[hist + t0 + j];
            }
            win[j] = v;
        }
    }
    for (uint t = t0; t < t1; t++) {
        const float cur = x[ulong(p.row0 + t) * p.in_stride + p.in_off + c];
        float acc = bias;
        R_UNROLL for (uint k = 0u; k < 8u; k++) {
            if (k == hist_len) acc += wk[k] * cur;
            else if (k < hist_len) acc += wk[k] * win[k];
        }
        mixed[ulong(p.row0 + t) * p.channels + c] = r_silu(acc);
        R_UNROLL for (uint k = 0u; k < 7u; k++) {
            if (k + 1u < hist_len) win[k] = win[k + 1u];
            else if (k + 1u == hist_len) win[k] = cur;
        }
    }
    if (tgp.y == 0u) {
        float fresh[7];
        R_UNROLL for (uint k = 0u; k < 7u; k++) {
            float v = 0.0f;
            if (k < hist_len) {
                if (p.n + k >= hist_len) {
                    v = x[ulong(p.row0 + p.n + k - hist_len) * p.in_stride + p.in_off + c];
                } else if (p.reset == 0u) {
                    v = state[hist + p.n + k];
                }
            }
            fresh[k] = v;
        }
        R_UNROLL for (uint k = 0u; k < 7u; k++) {
            if (k < hist_len) state[hist + k] = fresh[k];
        }
    }
}
