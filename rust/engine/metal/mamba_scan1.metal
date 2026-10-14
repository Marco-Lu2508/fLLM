// FLLM-KERNEL: mamba_scan1_ns<N>, mamba_scan1_gen | constants: inner,ns,rank,split_stride,proj_stride,proj_gate_off,row0,n,reset,state_off | buffers: 1 mixed(r), 2 dt(r), 3 split(r), 4 projected(r), 5 decay(r), 6 skip(r), 7 state(rw), 8 out(w) | threads: 64,1,1
// (N = 8, 16, 32; the constants struct is buffer 0; grid per run: [ceil(inner/64), 1, 1] threadgroups)
// Mamba-1 selective scan (reference: `mamba1_forward` in rust_decoder.rs; port of shaders/vulkan/mamba_scan1.comp, package 3) of one run (n consecutive
// tokens of one sequence), per channel ch < inner:
//   step = softplus(dt[t][ch]);  s[k] = s[k] * exp(step * decay[ch * ns + k]) + step * B[k] * xc;  y = (sum_k s[k] * C[k] + skip[ch] * xc) * silu(gate[ch])
// with xc = mixed[t][ch] (output of mamba_conv, `inner` channels), dt = the output of dt_proj (`inner` per token), B = split[t][rank + k], C = split[t][rank + ns + k]
// (split = output of x_proj, split_stride = rank + 2 ns floats per token, B and C normalised beforehand when the model asks for it: kernel mamba_bc_norm) and
// gate[ch] = projected[t * proj_stride + proj_gate_off + ch]. The state s of channel ch is state[state_off + ch * ns + k]. Template NS > 0: the state row stays in
// registers for the whole run; NS = 0 (generic): read and written in memory per token. ns <= 256.

#define MAMBA1_WG 64u
#define MAMBA1_MAX_STATE 256u

struct mamba_scan1_params {
    uint inner;
    uint ns;
    uint rank;
    uint split_stride;
    uint proj_stride;
    uint proj_gate_off;
    uint row0;
    uint n;
    uint reset;
    uint state_off;
};

template <uint NS>
[[kernel]] void mamba_scan1_t(
    constant mamba_scan1_params &p [[buffer(0)]],
    device const float *mixed [[buffer(1)]],
    device const float *dt [[buffer(2)]],
    device const float *split_bc [[buffer(3)]],
    device const float *projected [[buffer(4)]],
    device const float *decay [[buffer(5)]],
    device const float *skip_d [[buffer(6)]],
    device float *state [[buffer(7)]],
    device float *outv [[buffer(8)]],
    uint tg [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float sb[MAMBA1_MAX_STATE];
    threadgroup float sc[MAMBA1_MAX_STATE];
    const uint ch = tg * MAMBA1_WG + lane;
    const bool ch_ok = ch < p.inner;
    const ulong cbase = ulong(p.state_off) + ulong(ch) * p.ns;
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
        for (uint k = lane; k < p.ns; k += MAMBA1_WG) {
            sb[k] = split_bc[row * p.split_stride + p.rank + k];
            sc[k] = split_bc[row * p.split_stride + p.rank + p.ns + k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (ch_ok) {
            const float step_v = r_softplus20(dt[row * p.inner + ch]);
            const float xc = mixed[row * p.inner + ch];
            float acc = 0.0f;
            if (NS > 0u) {
                R_UNROLL for (uint k = 0u; k < NS; k++) {
                    cell[k] = cell[k] * precise::exp(step_v * decay[ch * p.ns + k]) + step_v * sb[k] * xc;
                    acc += cell[k] * sc[k];
                }
            } else {
                for (uint k = 0u; k < p.ns; k++) {
                    const float cv = state[cbase + k] * precise::exp(step_v * decay[ch * p.ns + k]) + step_v * sb[k] * xc;
                    state[cbase + k] = cv;
                    acc += cv * sc[k];
                }
            }
            const float z = projected[row * p.proj_stride + p.proj_gate_off + ch];
            outv[row * p.inner + ch] = (acc + skip_d[ch] * xc) * r_silu(z);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (NS > 0u) {
        if (ch_ok) {
            R_UNROLL for (uint k = 0u; k < NS; k++) state[cbase + k] = cell[k];
        }
    }
}

template [[host_name("mamba_scan1_ns8")]] [[kernel]] decltype(mamba_scan1_t<8>) mamba_scan1_t<8>;
template [[host_name("mamba_scan1_ns16")]] [[kernel]] decltype(mamba_scan1_t<16>) mamba_scan1_t<16>;
template [[host_name("mamba_scan1_ns32")]] [[kernel]] decltype(mamba_scan1_t<32>) mamba_scan1_t<32>;
template [[host_name("mamba_scan1_gen")]] [[kernel]] decltype(mamba_scan1_t<0>) mamba_scan1_t<0>;
