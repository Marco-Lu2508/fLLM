// FLLM-KERNEL: fllm_attn_decode_{f16,q8}_e{1,2,3,4,5,6,8,12,16,18}, fllm_attn_merge_f32
// decode: buffers: q, kpool, vpool, meta, bias, out, part | params: o_attn_params | grid: [nsplit, heads, rows in this dispatch] x [256, 1, 1]
// merge:  buffers: bias, out, part | params: o_merge_params | grid: [heads, rows in this dispatch, 1] x [128, 1, 1]
//
// Paged attention for decode and for short batches (Stream::attention_paged), split over the key range (split-KV / "flash decoding"), one KV
// format per variant (f16 / q8_0 cache of `DeviceKvCache`), E = channels per lane = ceil(max(dk, dv) / 32) (so head widths up to 32 * E; key and
// value widths may differ). One threadgroup of 8 simdgroups (256 threads) per (row, query head, split):
//   * simdgroup g takes the keys lo + g, lo + g + 8, ... of its split: the 32 lanes hold `ceil(dk / 32)` consecutive channels of q each, a key is
//     read by the same lanes, the partial dot products are added with simd_sum (score replicated in all lanes), the score gets ALiBi and soft cap,
//     and an online softmax (running maximum m, denominator l, accumulator of `ceil(dv / 32)` consecutive value channels per lane) is updated;
//   * the eight simdgroups are combined through threadgroup memory (weights exp(m_g - M)); nsplit == 1: the result is final (the sink is folded
//     in: it adds exp(sink) to the denominator) and written to `out`; nsplit > 1: the split writes M, L and the unnormalised accumulator to
//     `part`, and `fllm_attn_merge_f32` combines the splits (the sink is added there).
// Row r (global row index, row0 + grid z) attends to positions start .. lengths[r] - 1 (start: chunk, window or 0, see AttentionParams) through its
// block table row; q is `rows x heads x dk` (f32), out `rows x heads x dv`. `meta` = int32: lengths[rows_total], then the block tables
// (rows_total x stride). `bias` = float: sinks[heads] then ALiBi slopes[heads] (always bound; flags bit 0 = sinks used, bit 1 = ALiBi used).
// `part`: (M, L) pairs for every (row, head, split) record, then the accumulators (dv floats per record); record = (z * heads + head) * nsplit
// + split (z = grid z). Empty splits write L = 0 and are skipped by the merge. nsplit <= 256. Uses simd_sum: the SIMD width must be 32.
//
// Structure (one simdgroup per key, simd_sum for the score, online softmax per simdgroup, combine through threadgroup memory) after the MLX kernel
// sdpa_vector.h / sdpa_vector_2pass; paged KV, ALiBi, soft cap, sinks, windows, chunks and q8 caches are additions of this project.
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

#define O_DEC_NSG 8u

template <bool Q8, int E>
[[kernel]] void o_attn_decode(
    device const float *q [[buffer(1)]],
    device const char *kpool [[buffer(2)]],
    device const char *vpool [[buffer(3)]],
    device const int *meta [[buffer(4)]],
    device const float *bias [[buffer(5)]],
    device float *outv [[buffer(6)]],
    device float *part [[buffer(7)]],
    constant o_attn_params &p [[buffer(0)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint NSG = O_DEC_NSG;
    constexpr uint DMAX = 32u * uint(E);
    threadgroup float tg_m[NSG];
    threadgroup float tg_l[NSG];
    threadgroup float tg_acc[NSG * DMAX];

    const uint split = tgid.x;
    const uint h = tgid.y;
    const uint r = p.row0 + tgid.z;
    const uint kh = h / (p.heads / p.kv_heads);
    const uint n = uint(meta[r]);
    const uint start = o_row_start(n, p.window, p.chunk);
    uint lo = start;
    uint hi = n;
    if (p.nsplit > 1u) {
        const uint per = (n - start + p.nsplit - 1u) / p.nsplit;
        lo = min(n, start + split * per);
        hi = min(n, lo + per);
    }
    const ulong rec = ulong(tgid.z * p.heads + h) * p.nsplit + split;
    const ulong nrec = ulong(tpg.z) * p.heads * p.nsplit;
    if (lo >= hi) {  // empty split (uniform over the threadgroup)
        if (tid == 0u) {
            part[rec * 2ul] = o_neg;
            part[rec * 2ul + 1ul] = 0.0f;
        }
        return;
    }

    // channels of this lane: [ck, ck + nk) of the key / query, [cv, cv + nv) of the value
    const uint ek = (p.dk + 31u) / 32u;
    const uint ev = (p.dv + 31u) / 32u;
    const uint ck = lane * ek;
    const uint cv = lane * ev;
    const uint nk = ck < p.dk ? min(ek, p.dk - ck) : 0u;
    const uint nv = cv < p.dv ? min(ev, p.dv - cv) : 0u;

    float qv[E];
    device const float *qrow = q + (ulong(r) * p.heads + h) * p.dk;
    O_UNROLL
    for (int i = 0; i < E; i++) qv[i] = uint(i) < nk ? qrow[ck + uint(i)] * p.scale : 0.0f;

    const bool use_alibi = (p.flags & 2u) != 0u;
    const float slope = use_alibi ? bias[p.heads + h] : 0.0f;
    const ulong tab = ulong(p.rows_total) + ulong(r) * p.stride;
    const uint kvk = p.kv_heads * p.dk;
    const uint kvv = p.kv_heads * p.dv;
    const uint gps_k = p.k_gshift != 0u ? (kvk >> p.k_gshift) : p.kv_heads;
    const uint gps_v = p.v_gshift != 0u ? (kvv >> p.v_gshift) : p.kv_heads;

    float m = o_neg;
    float l = 0.0f;
    float acc[E];
    O_UNROLL
    for (int i = 0; i < E; i++) acc[i] = 0.0f;

    for (uint pos = lo + sg; pos < hi; pos += NSG) {
        const ulong slot = ulong(uint(meta[tab + pos / p.block_size])) * p.block_size + (pos % p.block_size);
        float kk[E];
        o_kv_load<Q8, E>(kpool, p.k_base, p.k_scale, gps_k, p.k_gshift, kvk, slot, kh * p.dk + ck, kh, nk, kk);
        float s = 0.0f;
        O_UNROLL
        for (int i = 0; i < E; i++) s += qv[i] * kk[i];
        s = simd_sum(s);
        if (use_alibi) s += slope * (float(pos) - float(n - 1u));
        if (p.softcap != 0.0f) s = o_softcap(s, p.softcap);

        const float new_m = max(m, s);
        const float factor = exp(m - new_m);
        const float pe = exp(s - new_m);
        l = l * factor + pe;
        float vv[E];
        o_kv_load<Q8, E>(vpool, p.v_base, p.v_scale, gps_v, p.v_gshift, kvv, slot, kh * p.dv + cv, kh, nv, vv);
        O_UNROLL
        for (int i = 0; i < E; i++) acc[i] = acc[i] * factor + pe * vv[i];
        m = new_m;
    }

    // combine the simdgroups
    if (lane == 0u) {
        tg_m[sg] = m;
        tg_l[sg] = l;
    }
    O_UNROLL
    for (int i = 0; i < E; i++) {
        if (uint(i) < nv) tg_acc[sg * DMAX + cv + uint(i)] = acc[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float big_m = tg_m[0];
    for (uint g = 1u; g < NSG; g++) big_m = max(big_m, tg_m[g]);
    float w[NSG];
    float big_l = 0.0f;
    for (uint g = 0u; g < NSG; g++) {
        w[g] = exp(tg_m[g] - big_m);
        big_l += w[g] * tg_l[g];
    }

    if (p.nsplit == 1u) {
        float corr = 1.0f;
        if ((p.flags & 1u) != 0u) {  // attention sink: adds exp(sink) to the denominator, no value
            const float sink = bias[h];
            const float m2 = max(big_m, sink);
            corr = exp(big_m - m2);
            big_l = big_l * corr + exp(sink - m2);
        }
        const float inv = corr / big_l;
        device float *orow = outv + (ulong(r) * p.heads + h) * p.dv;
        for (uint d = tid; d < p.dv; d += 256u) {
            float a = 0.0f;
            for (uint g = 0u; g < NSG; g++) a += w[g] * tg_acc[g * DMAX + d];
            orow[d] = a * inv;
        }
    } else {
        if (tid == 0u) {
            part[rec * 2ul] = big_m;
            part[rec * 2ul + 1ul] = big_l;
        }
        device float *prow = part + nrec * 2ul + rec * p.dv;
        for (uint d = tid; d < p.dv; d += 256u) {
            float a = 0.0f;
            for (uint g = 0u; g < NSG; g++) a += w[g] * tg_acc[g * DMAX + d];
            prow[d] = a;
        }
    }
}

template [[host_name("fllm_attn_decode_f16_e1")]] [[kernel]] decltype(o_attn_decode<false, 1>) o_attn_decode<false, 1>;
template [[host_name("fllm_attn_decode_f16_e2")]] [[kernel]] decltype(o_attn_decode<false, 2>) o_attn_decode<false, 2>;
template [[host_name("fllm_attn_decode_f16_e3")]] [[kernel]] decltype(o_attn_decode<false, 3>) o_attn_decode<false, 3>;
template [[host_name("fllm_attn_decode_f16_e4")]] [[kernel]] decltype(o_attn_decode<false, 4>) o_attn_decode<false, 4>;
template [[host_name("fllm_attn_decode_f16_e5")]] [[kernel]] decltype(o_attn_decode<false, 5>) o_attn_decode<false, 5>;
template [[host_name("fllm_attn_decode_f16_e6")]] [[kernel]] decltype(o_attn_decode<false, 6>) o_attn_decode<false, 6>;
template [[host_name("fllm_attn_decode_f16_e8")]] [[kernel]] decltype(o_attn_decode<false, 8>) o_attn_decode<false, 8>;
template [[host_name("fllm_attn_decode_f16_e12")]] [[kernel]] decltype(o_attn_decode<false, 12>) o_attn_decode<false, 12>;
template [[host_name("fllm_attn_decode_f16_e16")]] [[kernel]] decltype(o_attn_decode<false, 16>) o_attn_decode<false, 16>;
template [[host_name("fllm_attn_decode_f16_e18")]] [[kernel]] decltype(o_attn_decode<false, 18>) o_attn_decode<false, 18>;
template [[host_name("fllm_attn_decode_q8_e1")]] [[kernel]] decltype(o_attn_decode<true, 1>) o_attn_decode<true, 1>;
template [[host_name("fllm_attn_decode_q8_e2")]] [[kernel]] decltype(o_attn_decode<true, 2>) o_attn_decode<true, 2>;
template [[host_name("fllm_attn_decode_q8_e3")]] [[kernel]] decltype(o_attn_decode<true, 3>) o_attn_decode<true, 3>;
template [[host_name("fllm_attn_decode_q8_e4")]] [[kernel]] decltype(o_attn_decode<true, 4>) o_attn_decode<true, 4>;
template [[host_name("fllm_attn_decode_q8_e5")]] [[kernel]] decltype(o_attn_decode<true, 5>) o_attn_decode<true, 5>;
template [[host_name("fllm_attn_decode_q8_e6")]] [[kernel]] decltype(o_attn_decode<true, 6>) o_attn_decode<true, 6>;
template [[host_name("fllm_attn_decode_q8_e8")]] [[kernel]] decltype(o_attn_decode<true, 8>) o_attn_decode<true, 8>;
template [[host_name("fllm_attn_decode_q8_e12")]] [[kernel]] decltype(o_attn_decode<true, 12>) o_attn_decode<true, 12>;
template [[host_name("fllm_attn_decode_q8_e16")]] [[kernel]] decltype(o_attn_decode<true, 16>) o_attn_decode<true, 16>;
template [[host_name("fllm_attn_decode_q8_e18")]] [[kernel]] decltype(o_attn_decode<true, 18>) o_attn_decode<true, 18>;

// Second stage of the split-KV attention (decode with nsplit > 1): combines the partial results of the splits of one (row, head):
//   M = max(M_s of the non-empty splits, sink), L = sum_s L_s exp(M_s - M) (+ exp(sink - M)), out[d] = sum_s exp(M_s - M) acc_s[d] / L.
// Splits with L_s == 0 are empty and skipped. `bias` = sinks[heads] then ALiBi slopes[heads] (flags bit 0: sinks used). `part` as written by the
// decode kernel (record = (z * heads + head) * nsplit + split, z = grid y); `out` rows are the global rows row0 + z. nsplit <= 256.
kernel void fllm_attn_merge_f32(
    device const float *bias [[buffer(1)]],
    device float *outv [[buffer(2)]],
    device const float *part [[buffer(3)]],
    constant o_merge_params &p [[buffer(0)]],
    uint3 tgid [[threadgroup_position_in_grid]],
    uint3 tpg [[threadgroups_per_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 tg_size3 [[threads_per_threadgroup]]) {  // same vector type as the other grid attributes (MSL rejects a mix of uint and uint3)
    const uint tg_size = tg_size3.x;               // one-dimensional threadgroups: x is the thread count
    threadgroup float wgt[256];
    threadgroup float s_total;
    const uint h = tgid.x;
    const uint z = tgid.y;
    const uint r = p.row0 + z;
    const ulong rec0 = ulong(z * p.heads + h) * p.nsplit;
    const ulong nrec = ulong(tpg.y) * p.heads * p.nsplit;
    const bool use_sink = (p.flags & 1u) != 0u;
    if (tid == 0u) {
        float mx = o_neg;
        for (uint s = 0u; s < p.nsplit; s++) {
            if (part[(rec0 + s) * 2ul + 1ul] > 0.0f) mx = max(mx, part[(rec0 + s) * 2ul]);
        }
        float sink = o_neg;
        if (use_sink) {
            sink = bias[h];
            mx = max(mx, sink);
        }
        float total = 0.0f;
        for (uint s = 0u; s < p.nsplit; s++) {
            const float ls = part[(rec0 + s) * 2ul + 1ul];
            const float wv = ls > 0.0f ? exp(part[(rec0 + s) * 2ul] - mx) : 0.0f;
            wgt[s] = wv;
            total += wv * ls;
        }
        if (use_sink) total += exp(sink - mx);
        s_total = total;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = 1.0f / s_total;
    device float *orow = outv + (ulong(r) * p.heads + h) * p.dv;
    for (uint d = tid; d < p.dv; d += tg_size) {
        float a = 0.0f;
        for (uint s = 0u; s < p.nsplit; s++) {
            if (wgt[s] != 0.0f) a += wgt[s] * part[nrec * 2ul + (rec0 + s) * p.dv + d];
        }
        orow[d] = a * inv;
    }
}
