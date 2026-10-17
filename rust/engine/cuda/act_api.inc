// fLLM CUDA backend: C API of the narrow-activation path (FLLM_ACT_DTYPE=f16|bf16, see act_narrow.cu and docs/activation-dtype.md).
//
// Included at the end of the extern "C" block of api.cu: it needs FllmStream, PREP, stage and CHECK from there. Every entry flushes the pending fusions of
// the FP32 path (PREP) first, so FP32 and narrow operations can follow each other on one stream (the logits head, the embeddings, a layer without a narrow
// kernel).
//
// dt: 0 FP32, 1 FP16, 2 BF16 (see fllm_cuda_internal.h).

// Persistent scratch of the stream (index 0..2): grows when a larger call comes, never shrinks. While a CUDA graph is recorded it cannot grow (a
// graph is only recorded after an ordinary pass of the same shape sized every buffer).
static void *act_scratch(FllmStream *st, int idx, size_t bytes) {
    if (bytes <= st->act_tmp_bytes[idx]) return st->act_tmp[idx];
    if (st->capture) {
        fllm_set_error("narrow activations: scratch %d would grow to %zu bytes while a CUDA graph is recorded", idx, bytes);
        return nullptr;
    }
    if (cudaStreamSynchronize(st->s) != cudaSuccess) {
        fllm_set_error("narrow activations: cudaStreamSynchronize before growing the scratch failed");
        return nullptr;
    }
    cudaFree(st->act_tmp[idx]);
    st->epoch++;
    st->act_tmp[idx] = nullptr;
    st->act_tmp_bytes[idx] = 0;
    const size_t want = (bytes * 3 / 2 + 255) & ~size_t(255);
    if (cudaMalloc(&st->act_tmp[idx], want) != cudaSuccess) {
        cudaGetLastError();
        fllm_set_error("narrow activations: cannot allocate %zu bytes of scratch", want);
        return nullptr;
    }
    st->act_tmp_bytes[idx] = want;
    return st->act_tmp[idx];
}

// FLLM_WORKSPACE_ROWS: size the persistent scratch of the stream (Marlin activations / partials / locks, the FP16 conversion scratch, the attention workspace
// and the narrow-activation scratch) once for batches of up to `rows` rows whose layers are up to `cols` wide at the input and `out_cols` at the output, so that no
// allocation, free or stream synchronisation happens later in a run of that size (a larger batch still grows the buffers as before). Must run outside a CUDA graph capture.
int fllm_cuda_reserve_workspace(FllmStream *st, int rows, int cols, int out_cols) {
    PREP(st);
    if (st->capture) return fllm_set_error("reserve_workspace: a CUDA graph is being recorded");
    if (rows <= 0 || cols <= 0 || out_cols <= 0) return 0;
    if (st->sms == 0) {
        cudaDeviceProp prop;
        CHECK(cudaGetDeviceProperties(&prop, st->dev), "cudaGetDeviceProperties");
        st->sms = prop.multiProcessorCount;
    }
    CHECK(cudaStreamSynchronize(st->s), "cudaStreamSynchronize");
    auto grow = [&](void **ptr, size_t *have, size_t need, const char *what) -> int {
        if (need <= *have) return 0;
        cudaFree(*ptr);
        st->epoch++;
        *ptr = nullptr;
        *have = 0;
        if (cudaMalloc(ptr, need) != cudaSuccess) {
            cudaGetLastError();
            return fllm_set_error("reserve_workspace: cannot allocate %zu bytes for %s", need, what);
        }
        *have = need;
        return 0;
    };
    const size_t cmax = (size_t)(cols > out_cols ? cols : out_cols);
    if (fllm_q4z_supported(out_cols, cols, 128) || fllm_q4z_supported(out_cols, cols, 32)) {
        if (int e = grow(&st->q4z_io, &st->q4z_io_bytes, fllm_q4z_io_bytes(out_cols, cols, rows) * 2, "the Marlin activations")) return e;
        st->q4z_ready = false;
        if (int e = grow(&st->q4z_ctmp, &st->q4z_ctmp_bytes, fllm_q4z_ctmp_bytes(rows, st->sms), "the Marlin partials")) return e;
        const size_t ws = fllm_q4z_workspace_bytes((int)cmax) * 2;
        if (ws > st->q4z_ws_bytes) {
            if (int e = grow(&st->q4z_ws, &st->q4z_ws_bytes, ws, "the Marlin locks")) return e;
            CHECK(cudaMemset(st->q4z_ws, 0, st->q4z_ws_bytes), "cudaMemset(Marlin locks)");
        }
    }
    const size_t x16 = (((size_t)rows * (((size_t)cols + 255) & ~size_t(255)) * 2 + 255) & ~size_t(255)) + (rows <= 64 ? (size_t)4 << 20 : (size_t)16 << 20);
    if (int e = grow(reinterpret_cast<void **>(&st->x16), &st->x16_bytes, x16 * 2, "the activation conversion scratch")) return e;
    st->x16_ready = false;
    st->x8_ready = false;
    if (!st->ws) {
        st->ws_bytes = 16u << 20;
        CHECK(cudaMalloc(reinterpret_cast<void **>(&st->ws), st->ws_bytes), "cudaMalloc(attention workspace)");
    }
    for (int i = 0; i < 3; i++)
        if (!act_scratch(st, i, (size_t)rows * cmax * 4)) return 1;
    return 0;
}

int fllm_cuda_na_convert(FllmStream *st, const void *src, int src_dt, void *dst, int dst_dt, size_t n) {
    PREP(st);
    return fllm_na_convert_launch(st->s, src, src_dt, dst, dst_dt, n);
}

// x (rows x cols, dt) += scale * branch when branch != null, then out = rms_norm(x) * w
int fllm_cuda_na_rms_norm(FllmStream *st, int dt, void *x, const void *branch, float scale, const float *w, void *out, int rows, int cols, float eps) {
    PREP(st);
    return fllm_na_add_rms_norm_launch(st->s, dt, x, branch, scale, w, out, rows, cols, eps);
}

int fllm_cuda_na_residual(FllmStream *st, int dt, void *x, const void *res, size_t count, float scale) {
    PREP(st);
    return fllm_na_residual_launch(st->s, dt, x, res, count, scale);
}

int fllm_cuda_na_glu(FllmStream *st, int dt, const void *gate, const void *up, void *out, size_t count, int mode) {
    PREP(st);
    return fllm_na_glu_launch(st->s, dt, gate, up, out, count, mode);
}

int fllm_cuda_na_rope(FllmStream *st, int dt, void *q, void *k, const int32_t *positions, int rows, const float *cs, const float *sn, int heads, int kv_heads,
                       int dim, int half_, int neox) {
    PREP(st);
    void *d = nullptr;
    if (int e = stage(st, 0, positions, (size_t)rows * sizeof(int32_t), &d)) return e;
    return fllm_na_rope_launch(st->s, dt, q, k, static_cast<const int32_t *>(d), cs, sn, rows, heads, kv_heads, dim, half_, neox);
}

int fllm_cuda_na_kv_write(FllmStream *st, int dt, const void *src, uint16_t *pool_layer, const int32_t *slots, int rows, int kv) {
    PREP(st);
    void *d = nullptr;
    if (int e = stage(st, 1, slots, (size_t)rows * sizeof(int32_t), &d)) return e;
    return fllm_na_kv_write_launch(st->s, dt, src, pool_layer, static_cast<const int32_t *>(d), rows, kv);
}

// Rotation of q (in place), key rotation + cache write, value cache write in one kernel, optionally with the per-head RMS norm of q and k in front
// (qw / kw: norm weights of width dk, nullptr = none). dt 0 is the FP32 variant. Returns 3 (nothing launched) for head shapes the kernel does not cover.
int fllm_cuda_na_rope_kv(FllmStream *st, int dt, void *q, const void *k, const void *v, uint16_t *pool_k_layer, uint16_t *pool_v_layer,
                          const int32_t *positions, const int32_t *slots, int rows, const float *cs, const float *sn, const float *qw, const float *kw,
                          float eps, int heads, int kv_heads, int dk, int dv, int half_, int neox) {
    PREP(st);
    void *dp = nullptr, *ds = nullptr;
    if (int e = stage(st, 0, positions, (size_t)rows * sizeof(int32_t), &dp)) return e;
    if (int e = stage(st, 1, slots, (size_t)rows * sizeof(int32_t), &ds)) return e;
    return fllm_na_rope_kv_launch(st->s, dt, q, k, v, pool_k_layer, pool_v_layer, static_cast<const int32_t *>(dp), static_cast<const int32_t *>(ds), cs, sn,
                                   qw, kw, eps, rows, heads, kv_heads, dk, dv, half_, neox);
}

// The same for q, k and v as the three column blocks (n0 = heads * dk, n1 = kv_heads * dk, n2 = kv_heads * dv) of one merged projection result `qkv` of the activation
// type (FLLM_MERGE_PROJ): q is rotated (and normed) into the dense q_dst, k and v go to the cache. bias: FP32 bias of the merged projection (n0 + n1 + n2 values) or null.
int fllm_cuda_na_rope_kv_qkv(FllmStream *st, int dt, void *q_dst, const void *qkv, uint16_t *pool_k_layer, uint16_t *pool_v_layer, const int32_t *positions,
                             const int32_t *slots, int rows, const float *cs, const float *sn, const float *qw, const float *kw, const float *bias, float eps,
                             int heads, int kv_heads, int dk, int dv, int half_, int neox) {
    PREP(st);
    void *dp = nullptr, *ds = nullptr;
    if (int e = stage(st, 0, positions, (size_t)rows * sizeof(int32_t), &dp)) return e;
    if (int e = stage(st, 1, slots, (size_t)rows * sizeof(int32_t), &ds)) return e;
    const int n0 = heads * dk, n1 = kv_heads * dk, n2 = kv_heads * dv, stride = n0 + n1 + n2;
    const size_t es = dt == 0 ? 4 : 2;
    const char *base = static_cast<const char *>(qkv);
    return fllm_na_rope_kv_strided_launch(st->s, dt, q_dst, qkv, stride, base + n0 * es, stride, base + (size_t)(n0 + n1) * es, stride, pool_k_layer, pool_v_layer,
                                          static_cast<const int32_t *>(dp), static_cast<const int32_t *>(ds), cs, sn, qw, kw, bias, bias ? bias + n0 : nullptr,
                                          bias ? bias + n0 + n1 : nullptr, eps, rows, heads, kv_heads, dk, dv, half_, neox);
}

int fllm_cuda_na_split3(FllmStream *st, int dt, const void *src, void *d0, void *d1, void *d2, int rows, int n0, int n1, int n2) {
    PREP(st);
    return fllm_na_split3_launch(st->s, dt, src, d0, d1, d2, rows, n0, n1, n2);
}

int fllm_cuda_na_glu_merged(FllmStream *st, int dt, const void *gu, void *out, int rows, int ff, int mode) {
    PREP(st);
    return fllm_na_glu_merged_launch(st->s, dt, gu, out, rows, ff, mode);
}

// Marlin 4-bit GEMM on narrow activations: x [batch, cols] and y [batch, rows] are read and written as they are (no conversion kernels). bias (FP32, may be
// null) is added by a small kernel. res != null: y is not written; res += res_scale * (W x + bias) instead, and with norm_w != null that add is done by the RMS norm
// that follows (norm_out = rms_norm(res) * norm_w, eps). Weights built with the scales of this activation
// type (FP16 scales for dt 1, BF16 scales for dt 2, see CudaBackend::upload_q4z). Returns 3 when the shape or the alignment is not covered (nothing launched).
int fllm_cuda_na_linear_q4z(FllmStream *st, int dt, const void *w, int group, const void *x, void *y, void *res, float res_scale, const float *bias, int rows,
                             int cols, int batch, const float *norm_w, void *norm_out, float eps) {
    PREP(st);
    if (!fllm_q4z_supported(rows, cols, group) || (reinterpret_cast<uintptr_t>(x) & 15) != 0 || (reinterpret_cast<uintptr_t>(w) & 15) != 0) return 3;
    if (st->sms == 0) {
        cudaDeviceProp prop;
        CHECK(cudaGetDeviceProperties(&prop, st->dev), "cudaGetDeviceProperties");
        st->sms = prop.multiProcessorCount;
    }
    const size_t ctmp = fllm_q4z_ctmp_bytes(batch, st->sms), ws = fllm_q4z_workspace_bytes(rows);
    if (ctmp > st->q4z_ctmp_bytes || ws > st->q4z_ws_bytes) {
        if (st->capture) return fllm_set_error("narrow Marlin GEMM: scratch would grow while a CUDA graph is recorded");
        CHECK(cudaStreamSynchronize(st->s), "cudaStreamSynchronize");
        if (ctmp > st->q4z_ctmp_bytes) {
            cudaFree(st->q4z_ctmp);
            st->epoch++;
            st->q4z_ctmp = nullptr;
            st->q4z_ctmp_bytes = ctmp;
            CHECK(cudaMalloc(&st->q4z_ctmp, ctmp), "cudaMalloc(Marlin partials)");
        }
        if (ws > st->q4z_ws_bytes) {
            cudaFree(st->q4z_ws);
            st->epoch++;
            st->q4z_ws = nullptr;
            st->q4z_ws_bytes = ws * 2;
            CHECK(cudaMalloc(&st->q4z_ws, st->q4z_ws_bytes), "cudaMalloc(Marlin locks)");
            CHECK(cudaMemsetAsync(st->q4z_ws, 0, st->q4z_ws_bytes, st->s), "cudaMemset(Marlin locks)");
        }
    }
    // the result goes straight into y, or into scratch when it is added to the residual stream
    void *out = y;
    if (res != nullptr) {
        out = act_scratch(st, 0, (size_t)batch * rows * 2);
        if (!out) return 1;
    }
    const int mm = batch;
    const uint8_t *base = static_cast<const uint8_t *>(w);
    const size_t qbytes = (size_t)cols * rows / 2;
    const size_t scale_at = (qbytes + 255) & ~size_t(255);
    const size_t zero_at = scale_at + (((size_t)(cols / group) * rows * 2 + 255) & ~size_t(255));
    const int rc = fllm_vllm_marlin_gemm_t(x, w, out, st->q4z_ctmp, base + scale_at, base + zero_at, nullptr, nullptr, mm, rows, cols, cols, st->q4z_ws, 0,
                                           group, st->dev, st->sms, st->s, dt == 2 ? 1 : 0);
    if (rc != 0) return rc == -1 ? 3 : fllm_set_error("Marlin 4-bit GEMM (narrow activations) failed for %d x %d, batch %d, group %d", rows, cols, batch, group);
    if (int e = fllm_check_launch("act q4z")) return e;
    if (bias != nullptr && res == nullptr) {
        if (int e = fllm_na_bias_launch(st->s, dt, out, bias, rows, batch)) return e;
    }
    if (res != nullptr) {
        if (bias != nullptr) {
            if (int e = fllm_na_bias_launch(st->s, dt, out, bias, rows, batch)) return e;
        }
        // with a norm weight the residual add happens inside the RMS norm of the next block (what vLLM's fused_add_rms_norm does): res = round(res + scale * out),
        // norm_out = rms_norm(res) * norm_w
        if (norm_w != nullptr) return fllm_na_add_rms_norm_launch(st->s, dt, res, out, res_scale, norm_w, norm_out, batch, rows, eps);
        return fllm_na_residual_launch(st->s, dt, res, out, (size_t)batch * rows, res_scale);
    }
    return 0;
}

// K-quant (Q4_K, Q5_K, Q6_K) GEMV on narrow activations: the activations are quantized to int8 straight from the narrow array into the scratch the integer dot-product
// kernel reads (fllm_mmvq_try then finds them prepared), the result is FP32. Returns 3 (nothing launched) when that kernel would not take the product.
int fllm_cuda_na_linear_kquant(FllmStream *st, int dt, const void *w, int kind, const void *x, float *y, const float *bias, int rows, int cols, int batch) {
    PREP(st);
    static const bool no_reuse = getenv("FLLM_CUDA_NO_X16_REUSE") != nullptr;
    if (no_reuse || !fllm_mmvq_prep_ok(kind, cols, batch) || (reinterpret_cast<uintptr_t>(x) & 15) != 0 || (reinterpret_cast<uintptr_t>(w) & 15) != 0) return 3;
    // the scratch exactly as fllm_cuda_linear sizes it, so that it does not reallocate (and drop the prepared activations) behind our back
    const size_t need = (((size_t)batch * pad256(cols) * 2 + 255) & ~size_t(255)) + (batch <= 64 ? (size_t)4 << 20 : (size_t)16 << 20);
    if (need > st->x16_bytes) {
        if (st->capture) return fllm_set_error("narrow K-quant GEMV: scratch would grow while a CUDA graph is recorded");
        CHECK(cudaStreamSynchronize(st->s), "cudaStreamSynchronize");
        cudaFree(st->x16);
        st->epoch++;
        st->x16 = nullptr;
        st->x16_bytes = (need * 2 + 255) & ~size_t(255);
        CHECK(cudaMalloc(&st->x16, st->x16_bytes), "cudaMalloc(activation scratch)");
    }
    const size_t xq_bytes = ((size_t)batch * cols + 255) & ~size_t(255);
    const size_t groups = (size_t)batch * cols / 32;
    int8_t *xq = static_cast<int8_t *>(st->x16);
    float2 *xs = reinterpret_cast<float2 *>(static_cast<char *>(st->x16) + xq_bytes);
    if (int e = fllm_na_quantize_x8_launch(st->s, dt, x, xq, xs, groups)) return e;
    st->x8_x = static_cast<const float *>(x);
    st->x8_batch = batch;
    st->x8_cols = cols;
    st->x8_ready = true;
    return fllm_cuda_linear(st, w, kind, static_cast<const float *>(x), y, bias, rows, cols, batch);
}

// Attention on narrow q (and output): the attention kernels read and write FP32, so q is widened and the result narrowed around the call (the
// FlashInfer prompt kernel converts to half itself anyway). KV cache kinds as in fllm_cuda_attention.
int fllm_cuda_na_attention(FllmStream *st, int dt, const void *q, const uint16_t *kpool_layer, const uint16_t *vpool_layer, void *out, const int64_t *tables_host,
                            size_t tables_len, const int64_t *lengths_host, int rows, const float *sinks, const float *alibi, int heads, int kv_heads, int dk,
                            int dv, int stride, int block_size, int kind, int kgroup, int vgroup, int kscale_base, int vscale_base, float scale, float softcap,
                            int window, int chunk) {
    PREP(st);
    const size_t qn = (size_t)rows * heads * dk, on = (size_t)rows * heads * dv;
    // prompt rows of one sequence: FlashInfer reads and writes FP16 directly (FP16 activations as they are, BF16 through two FP16<->BF16 conversions)
    {
        static const bool old_attention = getenv("FLLM_CUDA_OLD_ATTN") != nullptr, no_flash = getenv("FLLM_CUDA_NO_FLASH") != nullptr;
        if (!old_attention && !no_flash && dk == dv && kind == 1 && sinks == nullptr && !alibi && softcap == 0.f && window == 0 && chunk == 0 && rows >= 16 &&
            !st->capture && !fllm_exact_math() && step_cached(st)) {
            step_runs(st, tables_host, lengths_host, rows, stride);
            const auto &runs = st->step.runs;
            if (runs.size() == 1 && runs[0].first == 0 && runs[0].second == rows) {
                void *dc = nullptr, *dl = nullptr;
                if (int e = stage_step(st, st->step.lengths, 3, lengths_host, (size_t)rows * sizeof(int64_t), &dl)) return e;
                auto &compact = st->step.compact_tables;
                if (int e = stage_step(st, st->step.compact, 2, compact.data(), compact.size() * sizeof(int64_t), &dc)) return e;
                const void *qh = q;
                void *oh = out;
                if (dt == 2) {
                    qh = act_scratch(st, 1, qn * 2);
                    oh = act_scratch(st, 2, on * 2);
                    if (!qh || !oh) return 1;
                    if (int e = fllm_na_convert_launch(st->s, q, 2, const_cast<void *>(qh), 1, qn)) return e;
                }
                fllm_attn_half_io = true;
                const int rc = fllm_attention3_try(st->s, static_cast<const float *>(qh), kpool_layer, vpool_layer, static_cast<float *>(oh), static_cast<const int64_t *>(dc),
                                                   static_cast<const int64_t *>(dl), sinks, rows, heads, kv_heads, dk, dv, stride, block_size, kind, scale, kscale_base,
                                                   vscale_base, (int)lengths_host[rows - 1]);
                fllm_attn_half_io = false;
                if (rc > 0) return rc;
                if (rc == 0) return dt == 2 ? fllm_na_convert_launch(st->s, oh, 1, out, 2, on) : 0;
            }
        }
    }
    // decode kernel (attention4.cu) on the narrow arrays as they are
    {
        static const bool old_attention = getenv("FLLM_CUDA_OLD_ATTN") != nullptr;
        if (!old_attention && dk == dv && (kind == 1 || kind == 2) && alibi == nullptr) {
            void *dt_dev = nullptr, *dl = nullptr;
            const bool cached = step_cached(st);
            if (cached) {
                if (int e = stage_step(st, st->step.lengths, 3, lengths_host, (size_t)rows * sizeof(int64_t), &dl)) return e;
                if (int e = stage_step(st, st->step.tables, 2, tables_host, tables_len * sizeof(int64_t), &dt_dev)) return e;
            } else {
                if (int e = stage(st, 2, tables_host, tables_len * sizeof(int64_t), &dt_dev)) return e;
                if (int e = stage(st, 3, lengths_host, (size_t)rows * sizeof(int64_t), &dl)) return e;
            }
            if (!st->ws) {
                st->ws_bytes = 16u << 20;
                CHECK(cudaMalloc(reinterpret_cast<void **>(&st->ws), st->ws_bytes), "cudaMalloc(attention workspace)");
            }
            int max_len = 1;
            for (int r = 0; r < rows; r++) max_len = lengths_host[r] > max_len ? (int)lengths_host[r] : max_len;
            if (st->capture) max_len = st->capture->max_len;
            fllm_attn_act_dt = dt;
            const int rc4 = fllm_attention4_try(st->s, static_cast<const float *>(q), kpool_layer, vpool_layer, static_cast<float *>(out), static_cast<const int64_t *>(dt_dev),
                                                static_cast<const int64_t *>(dl), max_len, sinks, alibi, rows, heads, kv_heads, dk, dv, stride, block_size, kind, scale,
                                                softcap, window, chunk, st->ws, st->ws_bytes, kscale_base, vscale_base);
            fllm_attn_act_dt = 0;
            if (rc4 >= 0) return rc4;
        }
    }
    float *qf = static_cast<float *>(act_scratch(st, 1, qn * 4));
    if (!qf) return 1;
    float *of = static_cast<float *>(act_scratch(st, 2, on * 4));
    if (!of) return 1;
    if (int e = fllm_na_convert_launch(st->s, q, dt, qf, 0, qn)) return e;
    if (int e = fllm_cuda_attention(st, qf, kpool_layer, vpool_layer, of, tables_host, tables_len, lengths_host, rows, sinks, alibi, heads, kv_heads, dk, dv,
                                    stride, block_size, kind, kgroup, vgroup, kscale_base, vscale_base, scale, softcap, window, chunk))
        return e;
    PREP(st);
    return fllm_na_convert_launch(st->s, of, 0, out, dt, on);
}

// Largest finite |x| and the number of NaN / Inf values of `count` narrow (or FP32) values: out2[0] = max (float), out2[1] = bad count (as float bits of an int).
// Synchronizes the stream.
int fllm_cuda_na_check(FllmStream *st, int dt, const void *x, size_t count, float *max_abs, int *bad) {
    PREP(st);
    if (!st->act_check) CHECK(cudaMalloc(reinterpret_cast<void **>(&st->act_check), 16), "cudaMalloc(activation check)");
    CHECK(cudaMemsetAsync(st->act_check, 0, 16, st->s), "cudaMemset(activation check)");
    if (int e = fllm_na_check_launch(st->s, dt, x, count, st->act_check)) return e;
    int host[2] = {0, 0};
    CHECK(cudaMemcpyAsync(host, st->act_check, sizeof host, cudaMemcpyDeviceToHost, st->s), "activation check download");
    CHECK(cudaStreamSynchronize(st->s), "activation check sync");
    memcpy(max_abs, &host[0], sizeof(float));
    *bad = host[1];
    return 0;
}
