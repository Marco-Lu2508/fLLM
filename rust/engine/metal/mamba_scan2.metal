// FLLM-KERNEL: mamba_scan2_ns<N>, mamba_scan2_gen | constants: inner,ns,heads,groups,head_dim,conv_dim,proj_stride,proj_dt_off,row0,n,reset,state_off,step_min | buffers: 1 mixed(r), 2 projected(r), 3 dt_bias(r), 4 decay(r), 5 skip(r), 6 state(rw), 7 out(w) | threads: 64,1,1
// (N = 16, 32, 64, 128; the constants struct is buffer 0; grid per run: [heads, ceil(head_dim/64), 1] threadgroups)
// Mamba-2 selective state update (reference: the scan loop of `mamba_forward` in rust_decoder.rs; port of shaders/vulkan/mamba_scan2.comp, package 3) of one
// run (n consecutive tokens of one sequence). Per token, head h, channel p of the head (xp = mixed[t][h * head_dim + p]) and state index k < ns:
//   dt = max(softplus(dt_raw[t][h] + dt_bias[h]), step_min);  da = exp(dt * decay[h])     (decay = A < 0; dt_raw = projected[t * proj_stride + proj_dt_off + h])
//   cell[h][p][k] = cell[h][p][k] * da + dt * xp * B[g][k];   out[t][h * head_dim + p] = sum_k cell[h][p][k] * C[g][k] + skip[h] * xp
// with g = h / (heads / groups); B[g] = mixed[t][inner + g * ns ..], C[g] = mixed[t][inner + groups * ns + g * ns ..] (mixed = output of mamba_conv, conv_dim
// channels per row: [x (inner) | B | C]). The SSM state is state[state_off + (h * head_dim + p) * ns + k] (float view of the sequence's state buffer; reset != 0
// ignores the old state). One thread per channel p, the tokens in order. Template parameter NS = state size: NS > 0 keeps the cell row in registers for the
// whole run; NS = 0 (generic) reads and writes the state in memory for every token. ns <= 256 (threadgroup staging of B and C).

#define MAMBA_SCAN_WG 64u
#define MAMBA_SCAN_MAX_STATE 256u

struct mamba_scan2_params {
    uint inner;
    uint ns;
    uint heads;
    uint groups;
    uint head_dim;
    uint conv_dim;
    uint proj_stride;
    uint proj_dt_off;
    uint row0;
    uint n;
    uint reset;
    uint state_off;
    float step_min;
};

template <uint NS>
[[kernel]] void mamba_scan2_t(
    constant mamba_scan2_params &p [[buffer(0)]],
    device const float *mixed [[buffer(1)]],
    device const float *projected [[buffer(2)]],
    device const float *dt_bias [[buffer(3)]],
    device const float *decay [[buffer(4)]],
    device const float *skip_d [[buffer(5)]],
    device float *state [[buffer(6)]],
    device float *outv [[buffer(7)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float sb[MAMBA_SCAN_MAX_STATE];
    threadgroup float sc[MAMBA_SCAN_MAX_STATE];
    const uint h = tgp.x;
    const uint pp = tgp.y * MAMBA_SCAN_WG + lane;
    const bool ch_ok = pp < p.head_dim;
    const uint g = h / max(p.heads / p.groups, 1u);
    const ulong cbase = ulong(p.state_off) + ulong(h * p.head_dim + pp) * p.ns;
    float cell[NS > 0u ? NS : 1u];
    if (NS > 0u) {
        R_UNROLL for (uint k = 0u; k < NS; k++) cell[k] = (ch_ok && p.reset == 0u) ? state[cbase + k] : 0.0f;
    } else {
        cell[0] = 0.0f;
        if (ch_ok && p.reset != 0u) {
            for (uint k = 0u; k < p.ns; k++) state[cbase + k] = 0.0f;
        }
    }
    for (uint t = 0u; t < p.n; t++) {
        const ulong row = ulong(p.row0) + t;
        for (uint k = lane; k < p.ns; k += MAMBA_SCAN_WG) {
            sb[k] = mixed[row * p.conv_dim + p.inner + g * p.ns + k];
            sc[k] = mixed[row * p.conv_dim + p.inner + p.groups * p.ns + g * p.ns + k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float raw = projected[row * p.proj_stride + p.proj_dt_off + h] + dt_bias[h];
        const float dt = max(r_softplus20(raw), p.step_min);
        const float da = precise::exp(dt * decay[h]);
        if (ch_ok) {
            const float xp = mixed[row * p.conv_dim + h * p.head_dim + pp];
            float acc = 0.0f;
            if (NS > 0u) {
                R_UNROLL for (uint k = 0u; k < NS; k++) {
                    cell[k] = cell[k] * da + dt * xp * sb[k];
                    acc += cell[k] * sc[k];
                }
            } else {
                for (uint k = 0u; k < p.ns; k++) {
                    const float cv = state[cbase + k] * da + dt * xp * sb[k];
                    state[cbase + k] = cv;
                    acc += cv * sc[k];
                }
            }
            outv[row * p.inner + h * p.head_dim + pp] = acc + skip_d[h] * xp;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);  // sb / sc are rewritten by the next token
    }
    if (NS > 0u) {
        if (ch_ok) {
            R_UNROLL for (uint k = 0u; k < NS; k++) state[cbase + k] = cell[k];
        }
    }
}

template [[host_name("mamba_scan2_ns16")]] [[kernel]] decltype(mamba_scan2_t<16>) mamba_scan2_t<16>;
template [[host_name("mamba_scan2_ns32")]] [[kernel]] decltype(mamba_scan2_t<32>) mamba_scan2_t<32>;
template [[host_name("mamba_scan2_ns64")]] [[kernel]] decltype(mamba_scan2_t<64>) mamba_scan2_t<64>;
template [[host_name("mamba_scan2_ns128")]] [[kernel]] decltype(mamba_scan2_t<128>) mamba_scan2_t<128>;
template [[host_name("mamba_scan2_gen")]] [[kernel]] decltype(mamba_scan2_t<0>) mamba_scan2_t<0>;
