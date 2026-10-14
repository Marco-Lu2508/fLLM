// FLLM-KERNEL: delta_conv | constants: conv_dim,kh,hk,ksz,row0,n,reset,hist_off,qscale | buffers: 1 qkv(r), 2 conv(r), 3 state(rw), 4 mixed(w) | threads: 128,1,1
// (the constants struct is buffer 0; grid per run: [2*kh + ceil((conv_dim - 2*kh*hk)/128), ceil(n/16), 1] threadgroups)
// Gated DeltaNet, part 1 (port of shaders/vulkan/delta_conv.comp and delta_conv_kernel in cuda/deltanet.cu, reference cpu_delta.rs): causal convolution
// over the channels [q | k | v] of one run (n consecutive tokens of one sequence, rows row0 .. row0+n-1 of qkv), SiLU, and the L2 normalisation of every
// query / key head:
//   mixed[t][c] = silu( sum_{k < ksz-1} w[c][k] * x[t - (ksz-1) + k][c] + w[c][ksz-1] * x[t][c] )     (x before the run = the stored history, or 0 when reset)
//   query heads: * 1/sqrt(sum_head(mixed^2) + 1e-6) * qscale (= hk^-0.5), key heads: * 1/sqrt(sum_head(mixed^2) + 1e-6), value channels unchanged.
// Threadgroup x: b < kh a query head, b < 2 kh a key head (channels b * hk .., hk threads, the L2 norm is the threadgroup reduction), otherwise a block of
// 128 value channels. Threadgroup y: a chunk of 16 tokens (>= ksz - 1 so that only chunk 0 ever reads the history); chunk 0 rewrites the history at the end.
// The state buffer `state` (float view of the sequence's state buffer): the history of channel c is state[hist_off + c * (ksz-1) + k] (the k-th last input,
// oldest first), hist_off = (state_offset + rec_bytes) / 4 of the layer. hk <= 128, ksz in 2..8. One dispatch per run (every run has its own state buffer).

#define DELTA_CONV_WG 128u

struct delta_conv_params {
    uint conv_dim;
    uint kh;
    uint hk;
    uint ksz;
    uint row0;
    uint n;
    uint reset;
    uint hist_off;
    float qscale;
};

kernel void delta_conv(
    constant delta_conv_params &p [[buffer(0)]],
    device const float *qkv [[buffer(1)]],
    device const float *convw [[buffer(2)]],
    device float *state [[buffer(3)]],
    device float *mixed [[buffer(4)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float red[DELTA_CONV_WG];
    const uint b = tgp.x;
    const uint t0 = tgp.y * 16u;
    if (t0 >= p.n) return;  // uniform over the threadgroup
    const uint t1 = min(p.n, t0 + 16u);
    const bool is_head = b < 2u * p.kh;
    uint base;
    uint width;
    if (is_head) {
        base = b * p.hk;
        width = p.hk;
    } else {
        base = 2u * p.kh * p.hk + (b - 2u * p.kh) * DELTA_CONV_WG;
        width = min(DELTA_CONV_WG, p.conv_dim - base);
    }
    const bool valid = lane < width;
    const uint c = base + lane;
    const uint hist_len = p.ksz - 1u;
    const uint hist = p.hist_off + c * hist_len;

    float wk[8];
    float win[8];  // win[j] = input of token (t - hist_len + j)
    R_UNROLL for (uint k = 0u; k < 8u; k++) {
        wk[k] = (valid && k < p.ksz) ? convw[c * p.ksz + k] : 0.0f;
        win[k] = 0.0f;
    }
    R_UNROLL for (uint j = 0u; j < 7u; j++) {
        if (j < hist_len) {
            float v = 0.0f;
            if (valid) {
                if (t0 + j >= hist_len) {
                    v = qkv[ulong(p.row0 + t0 + j - hist_len) * p.conv_dim + c];
                } else if (p.reset == 0u) {
                    v = state[hist + t0 + j];  // idx + hist_len with idx = t0 + j - hist_len
                }
            }
            win[j] = v;
        }
    }
    for (uint t = t0; t < t1; t++) {
        const float cur = valid ? qkv[ulong(p.row0 + t) * p.conv_dim + c] : 0.0f;
        float acc = 0.0f;
        R_UNROLL for (uint k = 0u; k < 8u; k++) {
            if (k == hist_len) acc += wk[k] * cur;
            else if (k < hist_len) acc += wk[k] * win[k];
        }
        acc = valid ? r_silu(acc) : 0.0f;
        if (is_head) {
            const float ss = r_tg_sum<DELTA_CONV_WG>(acc * acc, red, lane);
            const float inv = 1.0f / sqrt(ss + 1e-6f);
            acc = b < p.kh ? acc * inv * p.qscale : acc * inv;
        }
        if (valid) mixed[ulong(p.row0 + t) * p.conv_dim + c] = acc;
        // slide the window by one token; the new last entry is `cur`
        R_UNROLL for (uint k = 0u; k < 7u; k++) {
            if (k + 1u < hist_len) win[k] = win[k + 1u];
            else if (k + 1u == hist_len) win[k] = cur;
        }
    }
    // chunk 0 owns the history of its channels: every reader of the old values is this very thread
    if (tgp.y == 0u && valid) {
        float fresh[7];
        R_UNROLL for (uint k = 0u; k < 7u; k++) {
            float v = 0.0f;
            if (k < hist_len) {
                // input of token (n + k - hist_len)
                if (p.n + k >= hist_len) {
                    v = qkv[ulong(p.row0 + p.n + k - hist_len) * p.conv_dim + c];
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
