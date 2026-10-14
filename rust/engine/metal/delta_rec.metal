// FLLM-KERNEL: delta_rec_hk<HK> | constants: kh,vh,hv,conv_dim,tiled,row0,n,reset,state_off,beta_scale,eps | buffers: 1 mixed(r), 2 z(r), 3 beta(r), 4 alpha(r), 5 dt_bias(r), 6 decay(r), 7 norm(r), 8 state(rw), 9 out(w) | threads: 256,1,1
// (HK = 32, 64, 96, 128; the constants struct is buffer 0; grid per run: [vh, 1, 1] threadgroups)
// Gated DeltaNet, part 2 (port of shaders/vulkan/delta_rec.comp and delta_rec_kernel in cuda/deltanet.cu, reference cpu_delta.rs): the delta-rule recurrence
// of one run (n consecutive tokens of one sequence) for one value head per threadgroup, per token t:
//   forget = exp(decay[j] * softplus(alpha[t][j] + dt_bias[j])),  sigma = beta_scale * sigmoid(beta[t][j])
//   S = S * forget;  S += k (x) ((v - S^T k) * sigma);  o = S^T q            (q, k: the normalised query / key head of mixed, v: its value channels)
//   out[t][j][c] = o[c] * (1 / sqrt(mean_c(o^2) + eps)) * norm[c] * silu(z[t][j][c])        (gated RMS norm of the head output)
// The recurrent state S[j][i][c] (value head j, key row i < HK, value column c < hv) is FP32 at state[state_off + (j * HK + i) * hv + c] of the sequence's
// state buffer (float view; state_off = state_offset / 4 of the layer). Value column c is handled by two neighbouring threads (lane = 2 c + half): half 0
// holds the first HK/2 rows of the column in registers, half 1 the others; the dot products over the key dimension meet in threadgroup memory. Threads with
// c >= hv idle but join every barrier. HK is a template parameter (one kernel per value), hv <= 128 at run time. `mixed` is the output of delta_conv.
// Key head of value head j: tiled != 0 ? j % kh : j / (vh / kh). reset != 0: the old state is ignored.

#define DELTA_REC_WG 256u

struct delta_rec_params {
    uint kh;
    uint vh;
    uint hv;
    uint conv_dim;
    uint tiled;
    uint row0;
    uint n;
    uint reset;
    uint state_off;
    float beta_scale;
    float eps;
};

template <uint HK>
[[kernel]] void delta_rec_t(
    constant delta_rec_params &p [[buffer(0)]],
    device const float *mixed [[buffer(1)]],
    device const float *z [[buffer(2)]],
    device const float *beta [[buffer(3)]],
    device const float *alpha [[buffer(4)]],
    device const float *dt_bias [[buffer(5)]],
    device const float *decay [[buffer(6)]],
    device const float *norm_w [[buffer(7)]],
    device float *state [[buffer(8)]],
    device float *outv [[buffer(9)]],
    uint j [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    constexpr uint HALF_HK = HK / 2u;
    threadgroup float red[DELTA_REC_WG];
    threadgroup float s_qk[2u * HK];
    threadgroup float xch_a[DELTA_REC_WG];
    threadgroup float xch_b[DELTA_REC_WG];

    const uint c = lane >> 1;
    const uint half_id = lane & 1u;
    const bool col_ok = c < p.hv;
    const uint key_dim = p.kh * HK;
    const uint value_dim = p.vh * p.hv;
    const uint src = p.tiled != 0u ? j % p.kh : j / (p.vh / p.kh);
    const uint sbase = p.state_off + (j * HK + half_id * HALF_HK) * p.hv + c;

    float s[HALF_HK];
    R_UNROLL for (uint i = 0u; i < HALF_HK; i++) s[i] = (col_ok && p.reset == 0u) ? state[sbase + i * p.hv] : 0.0f;

    const float dtb = dt_bias[j];
    const float dec = decay[j];
    const float nw = col_ok ? norm_w[c] : 0.0f;

    for (uint t = 0u; t < p.n; t++) {
        const ulong row = ulong(p.row0) + t;
        // the normalised query and key head of the source head, shared by all columns
        if (lane < 2u * HK) {
            const uint off = lane < HK ? src * HK + lane : key_dim + src * HK + (lane - HK);
            s_qk[lane] = mixed[row * p.conv_dim + off];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float raw = alpha[row * p.vh + j] + dtb;
        const float forget = precise::exp(dec * r_softplus20(raw));
        const float sigma = p.beta_scale * r_sigmoid(beta[row * p.vh + j]);
        const float v = col_ok ? mixed[row * p.conv_dim + 2u * key_dim + j * p.hv + c] : 0.0f;

        float memory = 0.0f;
        R_UNROLL for (uint i = 0u; i < HALF_HK; i++) {
            s[i] *= forget;
            memory += s[i] * s_qk[HK + half_id * HALF_HK + i];
        }
        xch_a[lane] = memory;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        memory += xch_a[lane ^ 1u];
        const float delta = (v - memory) * sigma;
        float acc = 0.0f;
        R_UNROLL for (uint i = 0u; i < HALF_HK; i++) {
            s[i] += s_qk[HK + half_id * HALF_HK + i] * delta;
            acc += s[i] * s_qk[half_id * HALF_HK + i];
        }
        xch_b[lane] = acc;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        acc += xch_b[lane ^ 1u];
        // gated RMS norm of the head output (every column counted once)
        const float ss = r_tg_sum<DELTA_REC_WG>((half_id == 0u && col_ok) ? acc * acc : 0.0f, red, lane);
        const float inv = 1.0f / sqrt(ss / float(p.hv) + p.eps);
        if (half_id == 0u && col_ok) {
            const float gate = r_silu(z[row * value_dim + j * p.hv + c]);
            outv[row * value_dim + j * p.hv + c] = acc * inv * nw * gate;
        }
    }
    if (col_ok) {
        R_UNROLL for (uint i = 0u; i < HALF_HK; i++) state[sbase + i * p.hv] = s[i];
    }
}

template [[host_name("delta_rec_hk32")]] [[kernel]] decltype(delta_rec_t<32>) delta_rec_t<32>;
template [[host_name("delta_rec_hk64")]] [[kernel]] decltype(delta_rec_t<64>) delta_rec_t<64>;
template [[host_name("delta_rec_hk96")]] [[kernel]] decltype(delta_rec_t<96>) delta_rec_t<96>;
template [[host_name("delta_rec_hk128")]] [[kernel]] decltype(delta_rec_t<128>) delta_rec_t<128>;
