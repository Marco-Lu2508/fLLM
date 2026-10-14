// SPDX-License-Identifier: Apache-2.0
// fLLM Intel GPU backend (SYCL / oneAPI): launch code of the tiled FP32 linear kernel (prompt processing without XMX).
// Verification level: L0.
#include "common.hpp"
#include "linear_kernels.hpp"

namespace fllm {

namespace {

template <class F>
void run_tiled(Stream& s, const uint8_t* w, const float* x, float* y, const float* bias, int rows, int cols, int batch) {
    size_t nbp = 0, row_bytes = 0;
    weight_layout(F::KIND, static_cast<size_t>(cols), &nbp, &row_bytes);
    using K = LinTiled<F>;
    s.q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> wt(sycl::range<1>(K::TR * (KT + 1)), h);
        sycl::local_accessor<float, 1> xt(sycl::range<1>(K::TB * (K::KS + 1)), h);
        const K kernel{w, x, y, bias, rows, cols, batch, static_cast<int>(nbp), row_bytes, wt, xt};
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(ceil_div(static_cast<size_t>(batch), K::TB),
                                                         ceil_div(static_cast<size_t>(rows), K::TR) * K::WG),
                                         sycl::range<2>(1, K::WG)),
                       kernel);
    });
}

}  // namespace

bool launch_linear_tiled(Stream& s, const void* w, int kind, int rows, int cols, const float* x, float* y, const float* bias,
                         int batch) {
    switch (kind) {
#define X(K, F) \
    case K:     \
        run_tiled<F>(s, static_cast<const uint8_t*>(w), x, y, bias, rows, cols, batch); \
        return true;
        FLLM_SYCL_FORMATS(X)
#undef X
        default:
            return false;
    }
}

}  // namespace fllm