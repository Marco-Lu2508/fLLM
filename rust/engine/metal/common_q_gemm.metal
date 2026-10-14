// Shared code of the quantized matrix-matrix kernels (package p06a): simdgroup_matrix tiles for batch > 8 (prompt processing).
// Shared code only (see common_q_gemv.metal for the Q_KERNEL convention); the kernels are instantiated in q_mm_*.metal.
//
// C[tok][row] = sum_k X[tok][k] * W[row][k] with 64 weight rows x 32 tokens per threadgroup, K in slices of 32. The weights of a
// slice are dequantized into threadgroup memory as T (half or float), so are the activations; 4 simdgroups of 32 threads
// (2 x 2, each 32 rows x 16 tokens) multiply 8x8 tiles with float accumulation and stage the result through threadgroup memory,
// where scale, bias and the optional residual are applied. Needs simdgroup_matrix, a simd width of 32 and 128 threads per group.
//
// Verification level: L0 (clang-shim syntax check and CPU emulation only).

#define Q_BM 64u
#define Q_BN 32u
#define Q_BK 32u
#define Q_MM_THREADS 128u

// Dynamic threadgroup memory of one kernel, in bytes (buffer slot [[threadgroup(0)]]).
#define Q_MM_SHMEM_HALF ((Q_BM + Q_BN) * Q_BK * 2u + Q_BM * Q_BN * 4u)
#define Q_MM_SHMEM_FLOAT ((Q_BM + Q_BN) * Q_BK * 4u + Q_BM * Q_BN * 4u)

// Grid: x = ceil(rows / 64), y = ceil(batch / 32); threadgroup = 128 threads.
template <typename F, typename T>
Q_FUNC void q_mm_body(q_bytes w, device const float *x, device float *out, device const float *bias, constant q_params &p,
                      threadgroup T *sa, threadgroup T *sb, threadgroup float *so, uint3 tg, uint tid, uint sg) {
    const uint row_tile = tg.x * Q_BM;
    const uint tok_tile = tg.y * Q_BN;
    const uint nchunks = p.cols >> 3;
    const uint lr = tid >> 2;   // loader row / token 0..31
    const uint lc = tid & 3u;   // loader chunk inside the K slice
    const uint sg_r = sg & 1u;  // 32 weight rows of the tile
    const uint sg_t = sg >> 1;  // 16 tokens of the tile

    simdgroup_matrix<float, 8, 8> mc[2][4];
    #pragma clang loop unroll(full)
    for (int i = 0; i < 2; ++i)
        #pragma clang loop unroll(full)
        for (int j = 0; j < 4; ++j) mc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    for (uint k0 = 0; k0 < p.cols; k0 += Q_BK) {
        const uint ci = (k0 >> 3) + lc;
        #pragma clang loop unroll(full)
        for (uint h = 0; h < 2; ++h) {
            const uint r = lr + h * 32u;
            const uint grow = row_tile + r;
            float v[8];
            if (grow < p.rows && ci < nchunks) {
                F::dec8(w, ulong(grow) * p.nb_blocks + (ci / F::CPB), ci % F::CPB, p.plane1, v);
            } else {
                #pragma clang loop unroll(full)
                for (int i = 0; i < 8; ++i) v[i] = 0.0f;
            }
            threadgroup T *d = sa + r * Q_BK + lc * 8u;
            #pragma clang loop unroll(full)
            for (int i = 0; i < 8; ++i) d[i] = T(v[i]);
        }
        {
            const uint tok = tok_tile + lr;
            threadgroup T *d = sb + lr * Q_BK + lc * 8u;
            float4 a = float4(0.0f);
            float4 b = float4(0.0f);
            if (tok < p.batch && ci < nchunks) {
                device const float4 *xp = (device const float4 *)(x + ulong(tok) * p.x_stride + ulong(ci) * 8u);
                a = xp[0];
                b = xp[1];
            }
            d[0] = T(a.x);
            d[1] = T(a.y);
            d[2] = T(a.z);
            d[3] = T(a.w);
            d[4] = T(b.x);
            d[5] = T(b.y);
            d[6] = T(b.z);
            d[7] = T(b.w);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        #pragma clang loop unroll(full)
        for (uint kk = 0; kk < Q_BK / 8u; ++kk) {
            simdgroup_matrix<T, 8, 8> ma[2];
            simdgroup_matrix<T, 8, 8> mb[4];
            #pragma clang loop unroll(full)
            for (uint i = 0; i < 2; ++i) simdgroup_load(ma[i], sb + (sg_t * 16u + i * 8u) * Q_BK + kk * 8u, Q_BK);
            // B(k, n) = W[n][k]: the tile of W is stored row-major, load it transposed
            #pragma clang loop unroll(full)
            for (uint j = 0; j < 4; ++j) simdgroup_load(mb[j], sa + (sg_r * 32u + j * 8u) * Q_BK + kk * 8u, Q_BK, ulong2(0, 0), true);
            #pragma clang loop unroll(full)
            for (int i = 0; i < 2; ++i)
                #pragma clang loop unroll(full)
                for (int j = 0; j < 4; ++j) simdgroup_multiply_accumulate(mc[i][j], ma[i], mb[j], mc[i][j]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    #pragma clang loop unroll(full)
    for (uint i = 0; i < 2; ++i)
        #pragma clang loop unroll(full)
        for (uint j = 0; j < 4; ++j) simdgroup_store(mc[i][j], so + (sg_t * 16u + i * 8u) * Q_BM + sg_r * 32u + j * 8u, Q_BM);
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint idx = tid; idx < Q_BM * Q_BN; idx += Q_MM_THREADS) {
        const uint t = idx / Q_BM;
        const uint r = idx % Q_BM;
        const uint row = row_tile + r;
        const uint tok = tok_tile + t;
        if (row < p.rows && tok < p.batch) {
            float v = q_finish(so[idx], F::row_scale(w, p.plane1, row), p, bias, row);
            const ulong oi = ulong(tok) * p.out_stride + p.row_base + row;
            if ((p.flags & 2u) != 0u) v += out[oi];
            out[oi] = v;
        }
    }
}

// Kernel template (T = half or float tiles). Buffers: 0 weights, 1 x (f32), 2 out (f32), 3 bias (f32), 4 q_params,
// threadgroup(0) = Q_MM_SHMEM_HALF / Q_MM_SHMEM_FLOAT bytes.
#define Q_MM_KERNEL_ARGS                                                                                                      \
    q_bytes w [[buffer(0)]], device const float *x [[buffer(1)]], device float *out [[buffer(2)]],                          \
        device const float *bias [[buffer(3)]], constant q_params &p [[buffer(4)]],                                          \
        threadgroup char *shmem [[threadgroup(0)]], uint3 tg [[threadgroup_position_in_grid]],                                \
        uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]]

template <typename F, typename T>
Q_KERNEL void q_mm_t(Q_MM_KERNEL_ARGS) {
    threadgroup T *sa = (threadgroup T *)shmem;
    threadgroup T *sb = sa + Q_BM * Q_BK;
    threadgroup float *so = (threadgroup float *)(sb + Q_BN * Q_BK);
    q_mm_body<F, T>(w, x, out, bias, p, sa, sb, so, tg, tid, sg);
}
