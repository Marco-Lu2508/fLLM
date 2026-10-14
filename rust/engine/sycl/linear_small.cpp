// SPDX-License-Identifier: Apache-2.0
// fLLM Intel GPU backend (SYCL / oneAPI): launch code of the small-batch linear kernels (decode), the mixture-of-experts
// projections and the embedding lookup. Verification level: L0.
#include "common.hpp"
#include "linear_kernels.hpp"

namespace fllm {

namespace {

template <class F, int B, bool MOE>
void run_small(Stream& s, const uint8_t* w, const void* const* tab, const float* const* btab, const int32_t* ids, const float* x,
               float* y, const float* bias, int rows, int cols, int x_div, int slots) {
    constexpr int R = MOE ? rows_per_sg<F, 1>() : rows_per_sg<F, B>();
    size_t nbp = 0, row_bytes = 0;
    weight_layout(F::KIND, static_cast<size_t>(cols), &nbp, &row_bytes);
    const size_t groups = ceil_div(static_cast<size_t>(rows), SGPW * R);
    const LinSmall<F, B, R, MOE> kernel{w, tab, btab, ids, x, y, bias, rows, cols, static_cast<int>(nbp), x_div, row_bytes};
    s.q.parallel_for(sycl::nd_range<2>(sycl::range<2>(MOE ? static_cast<size_t>(slots) : 1, groups * SGPW * SGSZ),
                                       sycl::range<2>(1, SGPW * SGSZ)),
                     kernel);
}

template <class F>
void small_batches(Stream& s, const void* w, const float* x, float* y, const float* bias, int rows, int cols, int batch) {
    int done = 0;
    while (done < batch) {
        const int left = batch - done;
        const float* xc = x + static_cast<size_t>(done) * cols;
        float* yc = y + static_cast<size_t>(done) * rows;
        const uint8_t* wb = static_cast<const uint8_t*>(w);
        if (left >= 8) {
            run_small<F, 8, false>(s, wb, nullptr, nullptr, nullptr, xc, yc, bias, rows, cols, 1, 1);
            done += 8;
        } else if (left >= 4) {
            run_small<F, 4, false>(s, wb, nullptr, nullptr, nullptr, xc, yc, bias, rows, cols, 1, 1);
            done += 4;
        } else if (left >= 2) {
            run_small<F, 2, false>(s, wb, nullptr, nullptr, nullptr, xc, yc, bias, rows, cols, 1, 1);
            done += 2;
        } else {
            run_small<F, 1, false>(s, wb, nullptr, nullptr, nullptr, xc, yc, bias, rows, cols, 1, 1);
            done += 1;
        }
    }
}

// One work-group per token; the work-items walk the units of the row. Rows past `cols` of the float formats are skipped.
template <class F>
void run_embedding(Stream& s, const uint8_t* w, const int64_t* ids, int n, float* out, int cols) {
    size_t nbp = 0, row_bytes = 0;
    weight_layout(F::KIND, static_cast<size_t>(cols), &nbp, &row_bytes);
    constexpr int WG = 128;
    const int nblk = F::TAIL ? static_cast<int>(nbp) : cols / F::BLK;
    const int nbp_i = static_cast<int>(nbp);
    s.q.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(n) * WG), sycl::range<1>(WG)), [=](sycl::nd_item<1> it) {
        const int token = static_cast<int>(it.get_group(0));
        const uint8_t* row = w + static_cast<size_t>(ids[token]) * row_bytes;
        float* dst = out + static_cast<size_t>(token) * cols;
        const int total = nblk * F::UPB;
        for (int t = static_cast<int>(it.get_local_id(0)); t < total; t += WG) {
            const int blk = t / F::UPB, u = t % F::UPB;
            float wv[F::V];
            F::decode(row, nbp_i, blk, u, wv);
#pragma unroll
            for (int i = 0; i < F::V; i++) {
                const int k = blk * F::BLK + F::pos(u, i);
                if (!F::TAIL || k < cols) dst[k] = wv[i];
            }
        }
    });
}

}  // namespace

bool launch_linear_small(Stream& s, const void* w, int kind, int rows, int cols, const float* x, float* y, const float* bias,
                         int batch) {
    switch (kind) {
#define X(K, F) \
    case K:     \
        small_batches<F>(s, w, x, y, bias, rows, cols, batch); \
        return true;
        FLLM_SYCL_FORMATS(X)
#undef X
        default:
            return false;
    }
}

bool launch_moe_linear(Stream& s, const void* const* tab, const float* const* btab, int kind, const int32_t* ids, const float* x,
                       float* y, int rows, int cols, int slots, int x_div) {
    switch (kind) {
#define X(K, F) \
    case K:     \
        run_small<F, 1, true>(s, nullptr, tab, btab, ids, x, y, nullptr, rows, cols, x_div, slots); \
        return true;
        FLLM_SYCL_FORMATS(X)
#undef X
        default:
            return false;
    }
}

bool launch_embedding(Stream& s, const void* w, int kind, const int64_t* ids_dev, int n, float* out, int cols) {
    switch (kind) {
#define X(K, F) \
    case K:     \
        run_embedding<F>(s, static_cast<const uint8_t*>(w), ids_dev, n, out, cols); \
        return true;
        FLLM_SYCL_FORMATS(X)
#undef X
        default:
            return false;
    }
}

}  // namespace fllm
