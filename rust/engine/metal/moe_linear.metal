// FLLM-KERNEL: moe_linear_<fmt> | constants: rows,cols,slots,x_div,n_experts,flags,nb | buffers: 1 table(r), 2 bias_table(r), 3 ids(r), 4 x(r), 5 out(w), 6 arena(r), 7 bias_arena(r) | threads: 64,1,1
// (the constants struct is buffer 0; grid: [rows, slots, 1] threadgroups)
// Expert-indexed matrix-vector product (Stream::moe_linear):  out[slot][r] = dot(W[ids[slot]][r], x[slot / x_div]) (+ bias[ids[slot]][r]),
// slot < slots, r < rows, W[e] a rows x cols weight in the raw GGUF block layout of its format (see common_rec.metal; one kernel per format, named
// moe_linear_<q4_0|q4_1|q5_0|q5_1|q8_0|iq4_nl|mxfp4|q4_k|q5_k|q6_k>). cols is a multiple of the block size QK of the format (and of 8). An expert id
// outside 0..n_experts gives zeros (the CPU backend skips such experts). One threadgroup of 64 threads per (row, slot): the threads stride over the
// (block, part) units of the row and reduce in threadgroup memory.
//
// How the weights of an expert are reached (flags bit 1 selects, one pipeline for both):
//   flags bit 1 = 0 "arena":   all experts of the projection live in ONE buffer (`arena`, buffer 6); table[e] = BYTE offset of expert e in it (ulong),
//                              the biases (if any) live in ONE buffer (`bias_arena`, buffer 7); bias_table[e] = BYTE offset of expert e's float bias vector.
//   flags bit 1 = 1 "address": table[e] / bias_table[e] = the absolute 64-bit GPU address of the expert's weight / bias buffer (AddressTable of
//                              buffers.rs, `upload_pointer_table`); `arena` and `bias_arena` are not read (bind any small buffer, e.g. the table).
// flags bit 0: bias_table present (bias vectors of `rows` floats, 4-byte aligned). nb = cols / QK (blocks per row). x is read as plain floats
// (4-byte alignment is enough). The expert weights are never written (bind them untracked/read-only).
// A row of an expert starts at byte (row * nb) * BYTES of the expert (blocks back to back, no padding).

#define MOE_LINEAR_WG 64u

struct moe_linear_params {
    uint rows;
    uint cols;
    uint slots;
    uint x_div;
    uint n_experts;
    uint flags;
    uint nb;
};

template <typename F>
[[kernel]] void moe_linear_t(
    constant moe_linear_params &p [[buffer(0)]],
    device const ulong *table [[buffer(1)]],
    device const ulong *bias_table [[buffer(2)]],
    device const int *ids [[buffer(3)]],
    device const float *x [[buffer(4)]],
    device float *out [[buffer(5)]],
    device const uchar *arena [[buffer(6)]],
    device const uchar *bias_arena [[buffer(7)]],
    uint2 tgp [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_threadgroup]]) {
    threadgroup float red[MOE_LINEAR_WG];
    const uint row = tgp.x;
    const uint slot = tgp.y;
    if (row >= p.rows || slot >= p.slots) return;  // uniform
    const int expert = ids[slot];
    const bool ok = expert >= 0 && uint(expert) < p.n_experts;
    const bool by_address = (p.flags & 2u) != 0u;
    float acc = 0.0f;
    if (ok) {
        const ulong e = ulong(expert);
        device const uchar *w = by_address ? (device const uchar *)table[e] : arena + table[e];
        w += ulong(row) * p.nb * F::BYTES;
        device const float *xs = x + ulong(slot / p.x_div) * p.cols;
        const uint units = p.nb * F::PARTS;
        for (uint u = lane; u < units; u += MOE_LINEAR_WG) {
            const uint blk = u / F::PARTS;
            const uint part = u - blk * F::PARTS;
            acc += F::dot_part(w + ulong(blk) * F::BYTES, part, xs + blk * F::QK + part * F::PART);
        }
    }
    acc = r_tg_sum<MOE_LINEAR_WG>(acc, red, lane);
    if (lane == 0u) {
        float result = 0.0f;
        if (ok) {
            result = acc;
            if ((p.flags & 1u) != 0u) {
                const ulong e = ulong(expert);
                device const uchar *bp = by_address ? (device const uchar *)bias_table[e] : bias_arena + bias_table[e];
                result += ((device const float *)bp)[row];
            }
        }
        out[ulong(slot) * p.rows + row] = result;
    }
}

template [[host_name("moe_linear_q4_0")]] [[kernel]] decltype(moe_linear_t<r_fmt_q4_0>) moe_linear_t<r_fmt_q4_0>;
template [[host_name("moe_linear_q4_1")]] [[kernel]] decltype(moe_linear_t<r_fmt_q4_1>) moe_linear_t<r_fmt_q4_1>;
template [[host_name("moe_linear_q5_0")]] [[kernel]] decltype(moe_linear_t<r_fmt_q5_0>) moe_linear_t<r_fmt_q5_0>;
template [[host_name("moe_linear_q5_1")]] [[kernel]] decltype(moe_linear_t<r_fmt_q5_1>) moe_linear_t<r_fmt_q5_1>;
template [[host_name("moe_linear_q8_0")]] [[kernel]] decltype(moe_linear_t<r_fmt_q8_0>) moe_linear_t<r_fmt_q8_0>;
template [[host_name("moe_linear_iq4_nl")]] [[kernel]] decltype(moe_linear_t<r_fmt_iq4_nl>) moe_linear_t<r_fmt_iq4_nl>;
template [[host_name("moe_linear_mxfp4")]] [[kernel]] decltype(moe_linear_t<r_fmt_mxfp4>) moe_linear_t<r_fmt_mxfp4>;
template [[host_name("moe_linear_q4_k")]] [[kernel]] decltype(moe_linear_t<r_fmt_q4_k>) moe_linear_t<r_fmt_q4_k>;
template [[host_name("moe_linear_q5_k")]] [[kernel]] decltype(moe_linear_t<r_fmt_q5_k>) moe_linear_t<r_fmt_q5_k>;
template [[host_name("moe_linear_q6_k")]] [[kernel]] decltype(moe_linear_t<r_fmt_q6_k>) moe_linear_t<r_fmt_q6_k>;
