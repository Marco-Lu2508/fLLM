// Shared code of the quantized linear kernels (package p06a): parameter block, function constants and the matrix-vector kernel
// body for batch 1..8. Only shared code lives here (no `kernel` keyword is spelled out, see Q_KERNEL), so that the backend
// prepends this file to every compile unit. The kernels themselves are instantiated in q_mv_*.metal.
//
// out[tok][row_base + r] = scale * row_scale(r) * dot(x[tok], w[r]) + bias[row_base + r]  (+ out[...] if flags & 2)
//
// Verification level: L0 (clang-shim syntax check and CPU emulation only).

// Function constant: number of tokens per threadgroup column of the matrix-vector kernels (1..8). Indices 500..509 are reserved
// for package p06a.
constant uint q_fc_nb [[function_constant(500)]];

#define Q_MV_NSG 2  // simdgroups per threadgroup of the matrix-vector kernels

// NR weight rows per simdgroup, NB (= nb) tokens at once. Lane l of a simdgroup handles the chunks l, l + width, ... of a row.
// Grid: x = ceil(rows / (Q_MV_NSG * NR)) threadgroups, y = ceil(batch / nb); threadgroup = Q_MV_NSG * simd width threads.
template <typename F, int NR>
Q_FUNC void q_mv_body(q_bytes w, device const float *x, device float *out, device const float *bias, constant q_params &p,
                      uint nb, uint3 tg, uint sg, uint lane, uint sgw) {
    const uint row0 = (tg.x * Q_MV_NSG + sg) * NR;
    const uint tok0 = tg.y * nb;
    const uint nchunks = p.cols >> 3;

    float acc[NR][8];
    #pragma clang loop unroll(full)
    for (int r = 0; r < NR; ++r)
        #pragma clang loop unroll(full)
        for (int b = 0; b < 8; ++b) acc[r][b] = 0.0f;

    uint rowc[NR];
    #pragma clang loop unroll(full)
    for (int r = 0; r < NR; ++r) rowc[r] = min(row0 + uint(r), p.rows - 1u);

    for (uint ci = lane; ci < nchunks; ci += sgw) {
        const uint blk = ci / F::CPB;
        const uint c = ci % F::CPB;
        float4 xa[8];
        float4 xb[8];
        #pragma clang loop unroll(full)
        for (uint b = 0; b < 8; ++b) {
            xa[b] = float4(0.0f);
            xb[b] = float4(0.0f);
            if (b < nb && tok0 + b < p.batch) {
                device const float4 *xp = (device const float4 *)(x + ulong(tok0 + b) * p.x_stride + ulong(ci) * 8u);
                xa[b] = xp[0];
                xb[b] = xp[1];
            }
        }
        #pragma clang loop unroll(full)
        for (int r = 0; r < NR; ++r) {
            float wv[8];
            F::dec8(w, ulong(rowc[r]) * p.nb_blocks + blk, c, p.plane1, wv);
            const float4 w0 = float4(wv[0], wv[1], wv[2], wv[3]);
            const float4 w1 = float4(wv[4], wv[5], wv[6], wv[7]);
            #pragma clang loop unroll(full)
            for (uint b = 0; b < 8; ++b) {
                if (b < nb) acc[r][b] += dot(w0, xa[b]) + dot(w1, xb[b]);
            }
        }
    }

    #pragma clang loop unroll(full)
    for (int r = 0; r < NR; ++r) {
        const uint row = row0 + uint(r);
        const float rs = row < p.rows ? F::row_scale(w, p.plane1, row) : 1.0f;
        #pragma clang loop unroll(full)
        for (uint b = 0; b < 8; ++b) {
            if (b < nb) {
                const float s = simd_sum(acc[r][b]);
                if (lane == 0u && row < p.rows && tok0 + b < p.batch) {
                    float v = q_finish(s, rs, p, bias, row);
                    const ulong idx = ulong(tok0 + b) * p.out_stride + p.row_base + row;
                    if ((p.flags & 2u) != 0u) v += out[idx];
                    out[idx] = v;
                }
            }
        }
    }
}

// Kernel template, instantiated per format in q_mv_*.metal. Buffers: 0 weights, 1 x (f32), 2 out (f32), 3 bias (f32), 4 q_params.
#define Q_MV_KERNEL_ARGS                                                                                                      \
    q_bytes w [[buffer(0)]], device const float *x [[buffer(1)]], device float *out [[buffer(2)]],                          \
        device const float *bias [[buffer(3)]], constant q_params &p [[buffer(4)]],                                          \
        uint3 tg [[threadgroup_position_in_grid]], uint sg [[simdgroup_index_in_threadgroup]],                                 \
        uint lane [[thread_index_in_simdgroup]], uint sgw [[threads_per_simdgroup]]

template <typename F>
Q_KERNEL void q_mv_r4_t(Q_MV_KERNEL_ARGS) {
    q_mv_body<F, 4>(w, x, out, bias, p, q_fc_nb, tg, sg, lane, sgw);
}

template <typename F>
Q_KERNEL void q_mv_r2_t(Q_MV_KERNEL_ARGS) {
    q_mv_body<F, 2>(w, x, out, bias, p, q_fc_nb, tg, sg, lane, sgw);
}
