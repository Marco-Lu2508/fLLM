// FLLM-KERNEL: fllm_attn_prefill_{f16,q8}_k{DKF}v{DVF} for (DKF, DVF) in (8,8) (10,10) (12,12) (14,14) (16,16) (24,16) (24,24) (32,32)
// buffers: q, kpool, vpool, meta, bias, out | params: o_attn_params | grid: [ceil((rows_end - row0) / 32), heads, 1] x [128, 1, 1]
//
// Paged attention for prompt processing (Stream::attention_paged with one row per token), tiled with simdgroup_matrix (8x8 f32 fragments): one
// threadgroup of four simdgroups handles one query head of a TILE of 32 consecutive rows (simdgroup s owns the rows 8 s .. 8 s + 7) and walks the
// keys in tiles of 32, so every key / value row is decoded once per 32 query rows. DKF / DVF = key / value width in fragments (dk <= 8 DKF,
// dv <= 8 DVF; dims that are no multiple of 8 are zero padded). Per key tile and simdgroup:
//   1. S = Q K^T: the Q fragments stay in registers (loaded once through threadgroup memory, rows past the end of the batch are zero); the keys
//      arrive in threadgroup memory in chunks of 64 channels (decoded from the f16 / q8 pool, zero past the end of the row) and are loaded
//      transposed into the B operand. S is stored to threadgroup memory (private rows of the simdgroup, so only simdgroup barriers are needed).
//   2. Softmax per row on the lanes: lane = 4 * row + quarter, so 4 lanes share a row (reduced with simd_shuffle_xor); scale, ALiBi, soft cap
//      and the mask (chunk / window / causal length of that ROW) are applied; the probabilities P replace S in threadgroup memory; the running
//      maximum m and denominator l live in the lanes.
//   3. O = diag(corr) O + P V: the rescale of the accumulators by the per-row correction is a multiplication with a diagonal 8x8 matrix (only
//      documented simdgroup_matrix API, no assumptions about the element layout of a fragment), skipped when no row changed its maximum.
//      V arrives in chunks of 64 channels like K.
// At the end the sink (exp(sink) in the denominator, no value) is folded in, the rows are scaled by 1 / l through the same diagonal trick and the
// fragments are written out through a small threadgroup staging area (rows past rows_end are not written).
// REQUIREMENT (the planner checks it): all rows of a tile read their keys through ONE block table: for every block a row needs, its table agrees
// with the table of the longest row of the tile (which the kernel uses). Row r attends to start(r) .. lengths[r] - 1 like fllm_attn_decode.
// Registers: DKF + DVF fragments (2 registers each per lane) plus S; threadgroup memory 15.5 KiB. The SIMD width must be 32. Needs simdgroup_matrix
// (FLLM_HAS_SIMDGROUP_MATRIX); without it the file has no kernels and the planner does not select them.
#include <metal_stdlib>
using namespace metal;

#if FLLM_HAS_SIMDGROUP_MATRIX

template <bool Q8, int DKF, int DVF>
[[kernel]] void o_attn_prefill(
    device const float *q [[buffer(1)]],
    device const char *kpool [[buffer(2)]],
    device const char *vpool [[buffer(3)]],
    device const int *meta [[buffer(4)]],
    device const float *bias [[buffer(5)]],
    device float *outv [[buffer(6)]],
    constant o_attn_params &p [[buffer(0)]],
    uint2 tgid [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint sg [[simdgroup_index_in_threadgroup]],
    uint lane [[thread_index_in_simdgroup]]) {
    constexpr uint BQ = 32u;   // query rows per threadgroup
    constexpr uint BK = 32u;   // keys per tile
    constexpr uint DC = 64u;   // channels per staged chunk
    constexpr uint LD = 68u;   // row stride of the staging area (DC + 4)
    constexpr uint SLD = 36u;  // row stride of the score tile (BK + 4)
    constexpr uint NT = 128u;  // threads
    threadgroup float stage[BK * LD];   // K chunk, V chunk; at the start the Q staging (4 simdgroups x 8 rows x LD)
    threadgroup float s_tile[BQ * SLD];  // scores, then probabilities
    threadgroup float diag_t[4 * 64];    // per simdgroup: diagonal 8x8 matrix
    threadgroup float o_t[4 * 64];       // per simdgroup: output staging
    threadgroup uint slot_s[BK];

    const uint h = tgid.y;
    const uint kh = h / (p.heads / p.kv_heads);
    const uint rb0 = p.row0 + tgid.x * BQ;
    const uint rend = min(rb0 + BQ, p.rows_end);
    const uint row_l = lane >> 2;  // row inside the 8 rows of this simdgroup
    const uint cq = lane & 3u;     // quarter of the 32 keys of a tile
    const uint r_i = rb0 + sg * 8u + row_l;
    const bool rvalid = r_i < rend;
    const uint n_i = rvalid ? uint(meta[r_i]) : 1u;
    const uint start_i = o_row_start(n_i, p.window, p.chunk);
    const uint kvk = p.kv_heads * p.dk;
    const uint kvv = p.kv_heads * p.dv;
    const uint gps_k = p.k_gshift != 0u ? (kvk >> p.k_gshift) : p.kv_heads;
    const uint gps_v = p.v_gshift != 0u ? (kvv >> p.v_gshift) : p.kv_heads;

    // key range of the tile and the row whose block table serves it (the longest row)
    uint n_max = 0u;
    uint r_tab = rb0;
    uint kstart = 0xFFFFFFFFu;
    for (uint ii = 0u; ii < BQ; ii++) {
        const uint rr = rb0 + ii;
        if (rr < rend) {
            const uint nn = uint(meta[rr]);
            if (nn > n_max) {
                n_max = nn;
                r_tab = rr;
            }
            kstart = min(kstart, o_row_start(nn, p.window, p.chunk));
        }
    }
    const ulong tab = ulong(p.rows_total) + ulong(r_tab) * p.stride;
    const bool use_alibi = (p.flags & 2u) != 0u;
    const float slope = use_alibi ? bias[p.heads + h] : 0.0f;

    threadgroup float *srow = s_tile + (sg * 8u) * SLD;
    threadgroup float *sdiag = diag_t + sg * 64u;
    threadgroup float *sout = o_t + sg * 64u;

    // Q fragments of this simdgroup's 8 rows, staged through threadgroup memory (rows past the batch and channels past dk are zero)
    simdgroup_matrix<float, 8, 8> qf[DKF];
    {
        threadgroup float *qs = stage + sg * (8u * LD);
        O_UNROLL
        for (uint c0 = 0u; c0 < uint(DKF) * 8u; c0 += DC) {
            for (uint e = lane; e < 8u * DC; e += 32u) {
                const uint rr = e / DC;
                const uint dd = e % DC;
                const uint d = c0 + dd;
                const uint row = rb0 + sg * 8u + rr;
                qs[rr * LD + dd] = (row < rend && d < p.dk) ? q[(ulong(row) * p.heads + h) * p.dk + d] : 0.0f;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            O_UNROLL
            for (uint j = 0u; j < DC / 8u; j++) {
                if (c0 / 8u + j < uint(DKF)) simdgroup_load(qf[c0 / 8u + j], qs + j * 8u, LD);
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    simdgroup_matrix<float, 8, 8> of[DVF];
    O_UNROLL
    for (int i = 0; i < DVF; i++) of[i] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    const simdgroup_matrix<float, 8, 8> zero_frag = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
    float m_i = o_neg;
    float l_i = 0.0f;

    for (uint t0 = kstart; t0 < n_max; t0 += BK) {
        if (tid < BK) {
            const uint pos = t0 + tid;
            slot_s[tid] = pos < n_max ? uint(meta[tab + pos / p.block_size]) * p.block_size + (pos % p.block_size) : 0u;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 1. S = Q K^T
        simdgroup_matrix<float, 8, 8> sf[BK / 8u];
        O_UNROLL
        for (uint kb = 0u; kb < BK / 8u; kb++) sf[kb] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
        O_UNROLL
        for (uint c0 = 0u; c0 < uint(DKF) * 8u; c0 += DC) {
            for (uint e = tid; e < BK * DC; e += NT) {
                const uint kk = e / DC;
                const uint dd = e % DC;
                const uint d = c0 + dd;
                float val = 0.0f;
                if (d < p.dk && t0 + kk < n_max) val = o_kv_at<Q8>(kpool, p.k_base, p.k_scale, gps_k, p.k_gshift, kvk, slot_s[kk], kh * p.dk + d, kh);
                stage[kk * LD + dd] = val;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            O_UNROLL
            for (uint j = 0u; j < DC / 8u; j++) {
                if (c0 / 8u + j < uint(DKF)) {
                    O_UNROLL
                    for (uint kb = 0u; kb < BK / 8u; kb++) {
                        simdgroup_matrix<float, 8, 8> kf;
                        simdgroup_load(kf, stage + kb * 8u * LD + j * 8u, LD, ulong2(0, 0), true);
                        simdgroup_multiply_accumulate(sf[kb], qf[c0 / 8u + j], kf, sf[kb]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        O_UNROLL
        for (uint kb = 0u; kb < BK / 8u; kb++) simdgroup_store(sf[kb], srow + kb * 8u, SLD);
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // 2. mask, scale, ALiBi, soft cap, online softmax of row `row_l` (this lane's 8 keys cq * 8 .. cq * 8 + 7 of the tile)
        float sc[8];
        bool ok[8];
        float tmax = o_neg;
        O_UNROLL
        for (uint jj = 0u; jj < 8u; jj++) {
            const uint kidx = cq * 8u + jj;
            const uint pos = t0 + kidx;
            const bool valid = rvalid && pos >= start_i && pos < n_i;
            float s = o_neg;
            if (valid) {
                s = srow[row_l * SLD + kidx] * p.scale;
                if (use_alibi) s += slope * (float(pos) - float(n_i - 1u));
                if (p.softcap != 0.0f) s = o_softcap(s, p.softcap);
            }
            sc[jj] = s;
            ok[jj] = valid;
            tmax = max(tmax, s);
        }
        tmax = max(tmax, simd_shuffle_xor(tmax, 1));
        tmax = max(tmax, simd_shuffle_xor(tmax, 2));
        const float m_new = max(m_i, tmax);
        const float corr = exp(m_i - m_new);
        float psum = 0.0f;
        O_UNROLL
        for (uint jj = 0u; jj < 8u; jj++) {
            const float e = ok[jj] ? exp(sc[jj] - m_new) : 0.0f;
            srow[row_l * SLD + cq * 8u + jj] = e;
            psum += e;
        }
        psum += simd_shuffle_xor(psum, 1);
        psum += simd_shuffle_xor(psum, 2);
        l_i = l_i * corr + psum;
        m_i = m_new;

        // 3a. rescale the accumulators: O = diag(corr) O (skipped when no row changed its maximum)
        if (simd_any(corr != 1.0f)) {
            for (uint idx = lane; idx < 64u; idx += 32u) {
                const uint rr = idx >> 3;
                const uint cc = idx & 7u;
                const float cv = simd_shuffle(corr, ushort(rr * 4u));
                sdiag[idx] = rr == cc ? cv : 0.0f;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
            simdgroup_matrix<float, 8, 8> dm;
            simdgroup_load(dm, sdiag, 8);
            O_UNROLL
            for (int v = 0; v < DVF; v++) {
                simdgroup_matrix<float, 8, 8> tmp;
                simdgroup_multiply_accumulate(tmp, dm, of[v], zero_frag);
                of[v] = tmp;
            }
            simdgroup_barrier(mem_flags::mem_threadgroup);
        } else {
            simdgroup_barrier(mem_flags::mem_threadgroup);
        }

        // 3b. O += P V (V in chunks of 64 channels)
        simdgroup_matrix<float, 8, 8> pf[BK / 8u];
        O_UNROLL
        for (uint kb = 0u; kb < BK / 8u; kb++) simdgroup_load(pf[kb], srow + kb * 8u, SLD);
        O_UNROLL
        for (uint c0 = 0u; c0 < uint(DVF) * 8u; c0 += DC) {
            for (uint e = tid; e < BK * DC; e += NT) {
                const uint kk = e / DC;
                const uint dd = e % DC;
                const uint d = c0 + dd;
                float val = 0.0f;
                if (d < p.dv && t0 + kk < n_max) val = o_kv_at<Q8>(vpool, p.v_base, p.v_scale, gps_v, p.v_gshift, kvv, slot_s[kk], kh * p.dv + d, kh);
                stage[kk * LD + dd] = val;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            O_UNROLL
            for (uint j = 0u; j < DC / 8u; j++) {
                if (c0 / 8u + j < uint(DVF)) {
                    O_UNROLL
                    for (uint kb = 0u; kb < BK / 8u; kb++) {
                        simdgroup_matrix<float, 8, 8> vf;
                        simdgroup_load(vf, stage + kb * 8u * LD + j * 8u, LD);
                        simdgroup_multiply_accumulate(of[c0 / 8u + j], pf[kb], vf, of[c0 / 8u + j]);
                    }
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }

    // sink (exp(sink) in the denominator, no value) and normalisation: O = diag(corr2 / l) O
    float corr2 = 1.0f;
    if ((p.flags & 1u) != 0u) {
        const float sink = bias[h];
        const float m2 = max(m_i, sink);
        corr2 = exp(m_i - m2);
        l_i = l_i * corr2 + exp(sink - m2);
    }
    const float fac = rvalid ? corr2 / l_i : 0.0f;
    for (uint idx = lane; idx < 64u; idx += 32u) {
        const uint rr = idx >> 3;
        const uint cc = idx & 7u;
        const float cv = simd_shuffle(fac, ushort(rr * 4u));
        sdiag[idx] = rr == cc ? cv : 0.0f;
    }
    simdgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_matrix<float, 8, 8> dm;
    simdgroup_load(dm, sdiag, 8);
    O_UNROLL
    for (int v = 0; v < DVF; v++) {
        simdgroup_matrix<float, 8, 8> tmp;
        simdgroup_multiply_accumulate(tmp, dm, of[v], zero_frag);
        simdgroup_store(tmp, sout, 8);
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint idx = lane; idx < 64u; idx += 32u) {
            const uint row = rb0 + sg * 8u + (idx >> 3);
            const uint d = uint(v) * 8u + (idx & 7u);
            if (row < rend && d < p.dv) outv[(ulong(row) * p.heads + h) * p.dv + d] = sout[idx];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
    }
}

template [[host_name("fllm_attn_prefill_f16_k8v8")]] [[kernel]] decltype(o_attn_prefill<false, 8, 8>) o_attn_prefill<false, 8, 8>;
template [[host_name("fllm_attn_prefill_f16_k10v10")]] [[kernel]] decltype(o_attn_prefill<false, 10, 10>) o_attn_prefill<false, 10, 10>;
template [[host_name("fllm_attn_prefill_f16_k12v12")]] [[kernel]] decltype(o_attn_prefill<false, 12, 12>) o_attn_prefill<false, 12, 12>;
template [[host_name("fllm_attn_prefill_f16_k14v14")]] [[kernel]] decltype(o_attn_prefill<false, 14, 14>) o_attn_prefill<false, 14, 14>;
template [[host_name("fllm_attn_prefill_f16_k16v16")]] [[kernel]] decltype(o_attn_prefill<false, 16, 16>) o_attn_prefill<false, 16, 16>;
template [[host_name("fllm_attn_prefill_f16_k24v16")]] [[kernel]] decltype(o_attn_prefill<false, 24, 16>) o_attn_prefill<false, 24, 16>;
template [[host_name("fllm_attn_prefill_f16_k24v24")]] [[kernel]] decltype(o_attn_prefill<false, 24, 24>) o_attn_prefill<false, 24, 24>;
template [[host_name("fllm_attn_prefill_f16_k32v32")]] [[kernel]] decltype(o_attn_prefill<false, 32, 32>) o_attn_prefill<false, 32, 32>;
template [[host_name("fllm_attn_prefill_q8_k8v8")]] [[kernel]] decltype(o_attn_prefill<true, 8, 8>) o_attn_prefill<true, 8, 8>;
template [[host_name("fllm_attn_prefill_q8_k10v10")]] [[kernel]] decltype(o_attn_prefill<true, 10, 10>) o_attn_prefill<true, 10, 10>;
template [[host_name("fllm_attn_prefill_q8_k12v12")]] [[kernel]] decltype(o_attn_prefill<true, 12, 12>) o_attn_prefill<true, 12, 12>;
template [[host_name("fllm_attn_prefill_q8_k14v14")]] [[kernel]] decltype(o_attn_prefill<true, 14, 14>) o_attn_prefill<true, 14, 14>;
template [[host_name("fllm_attn_prefill_q8_k16v16")]] [[kernel]] decltype(o_attn_prefill<true, 16, 16>) o_attn_prefill<true, 16, 16>;
template [[host_name("fllm_attn_prefill_q8_k24v16")]] [[kernel]] decltype(o_attn_prefill<true, 24, 16>) o_attn_prefill<true, 24, 16>;
template [[host_name("fllm_attn_prefill_q8_k24v24")]] [[kernel]] decltype(o_attn_prefill<true, 24, 24>) o_attn_prefill<true, 24, 24>;
template [[host_name("fllm_attn_prefill_q8_k32v32")]] [[kernel]] decltype(o_attn_prefill<true, 32, 32>) o_attn_prefill<true, 32, 32>;

#endif  // FLLM_HAS_SIMDGROUP_MATRIX
