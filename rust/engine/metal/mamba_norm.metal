// FLLM-KERNEL: mamba_gate_norm | constants: inner,norm_groups,rows,proj_stride,proj_gate_off,eps | buffers: 1 y(rw), 2 projected(r), 3 norm(r) | threads: 256,1,1
// FLLM-KERNEL: mamba_bc_norm | constants: rows,rank,state,split_stride,eps | buffers: 1 split(rw) | threads: 128,1,1
// Mamba normalisations around the scans (both for the whole batch at once, no recurrent state involved). The constants struct is buffer 0.
//
// mamba_gate_norm: grid [norm_groups, rows, 1] threadgroups
//   Mamba-2 gated group RMS norm (end of `mamba_forward` in rust_decoder.rs), in place on y (rows x inner): v = y * silu(gate), then for every group of
//   size = inner / norm_groups values  y = v * (1 / sqrt(mean(v^2) + eps)) * norm[g * size + i]. gate = projected[t * proj_stride + proj_gate_off + i].
//   Port of shaders/vulkan/mamba_gate_norm.comp (package 3).
// mamba_bc_norm: grid [3, rows, 1] threadgroups
//   Mamba-1 `bc_rms_eps`: the three parts of a row of the x_proj output (dt input [0, rank), B [rank, rank + state), C [rank + state, rank + 2 state)) are
//   RMS-normalised separately without weight, in place: part *= 1 / sqrt(mean(part^2) + eps) (`mamba1_forward`). Threadgroup x = part, y = row.

#define MAMBA_NORM_WG 256u
#define MAMBA_BC_WG 128u

struct mamba_gate_norm_params {
    uint inner;
    uint norm_groups;
    uint rows;
    uint proj_stride;
    uint proj_gate_off;
    float eps;
};

kernel void mamba_gate_norm(
    constant mamba_gate_norm_params &p [[buffer(0)]],
    device float *y [[buffer(1)]],
    device const float *projected [[buffer(2)]],
    device const float *norm_w [[buffer(3)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float red[MAMBA_NORM_WG];
    const uint g = tgp.x;
    const uint row = tgp.y;
    if (row >= p.rows) return;  // uniform
    const uint size = p.inner / p.norm_groups;
    const ulong yrow = ulong(row) * p.inner + ulong(g) * size;
    const ulong prow = ulong(row) * p.proj_stride + p.proj_gate_off + ulong(g) * size;
    float ss = 0.0f;
    for (uint i = lane; i < size; i += MAMBA_NORM_WG) {
        const float v = y[yrow + i] * r_silu(projected[prow + i]);
        ss += v * v;
    }
    ss = r_tg_sum<MAMBA_NORM_WG>(ss, red, lane);
    const float inv = 1.0f / sqrt(ss / float(size) + p.eps);
    for (uint i = lane; i < size; i += MAMBA_NORM_WG) {
        const float v = y[yrow + i] * r_silu(projected[prow + i]);
        y[yrow + i] = v * inv * norm_w[g * size + i];
    }
}

struct mamba_bc_norm_params {
    uint rows;
    uint rank;
    uint state;
    uint split_stride;
    float eps;
};

kernel void mamba_bc_norm(
    constant mamba_bc_norm_params &p [[buffer(0)]],
    device float *split_bc [[buffer(1)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float red[MAMBA_BC_WG];
    const uint part = tgp.x;
    const uint row = tgp.y;
    if (row >= p.rows || part >= 3u) return;  // uniform
    const uint width = part == 0u ? p.rank : p.state;
    const uint off = part == 0u ? 0u : p.rank + (part - 1u) * p.state;
    device float *v = split_bc + ulong(row) * p.split_stride + off;
    float ss = 0.0f;
    for (uint i = lane; i < width; i += MAMBA_BC_WG) ss += v[i] * v[i];
    ss = r_tg_sum<MAMBA_BC_WG>(ss, red, lane);
    const float inv = 1.0f / sqrt(ss / float(width) + p.eps);
    for (uint i = lane; i < width; i += MAMBA_BC_WG) v[i] *= inv;
}
