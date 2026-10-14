// SPDX-License-Identifier: Apache-2.0
// fLLM Intel GPU backend (SYCL / oneAPI): launch code of the matrix-engine (XMX, joint_matrix) linear kernel.
// Verification level: L0 (the device code compiles to SPIR-V; the matrix shapes are untested on hardware).
#include "common.hpp"
#include "linear_kernels.hpp"

namespace fllm {

namespace {

template <class F, int N>
void run_xmx(Stream& s, const uint8_t* w, const float* x, float* y, const float* bias, int rows, int cols, int batch) {
    size_t nbp = 0, row_bytes = 0;
    weight_layout(F::KIND, static_cast<size_t>(cols), &nbp, &row_bytes);
    using K = LinXmx<F, N>;
    s.q.submit([&](sycl::handler& h) {
        sycl::local_accessor<sycl::half, 1> wt(sycl::range<1>(K::TR * KT), h);
        sycl::local_accessor<sycl::half, 1> xt(sycl::range<1>(K::TB * KT), h);
        sycl::local_accessor<float, 1> cs(sycl::range<1>(K::SGS * 8 * N), h);
        const K kernel{w, x, y, bias, rows, cols, batch, static_cast<int>(nbp), row_bytes, wt, xt, cs};
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(ceil_div(static_cast<size_t>(batch), K::TB),
                                                         ceil_div(static_cast<size_t>(rows), K::TR) * K::WG),
                                         sycl::range<2>(1, K::WG)),
                       kernel);
    });
}

}  // namespace

bool launch_linear_xmx(Stream& s, const void* w, int kind, int rows, int cols, const float* x, float* y, const float* bias,
                       int batch) {
    const uint8_t* wb = static_cast<const uint8_t*>(w);
    if (s.dev->xmx_n == 16) {
        switch (kind) {
#define X(K, F) \
    case K:     \
        run_xmx<F, 16>(s, wb, x, y, bias, rows, cols, batch); \
        return true;
            FLLM_SYCL_FORMATS(X)
#undef X
            default:
                return false;
        }
    }
    if (s.dev->xmx_n == 8) {
        switch (kind) {
#define X(K, F) \
    case K:     \
        run_xmx<F, 8>(s, wb, x, y, bias, rows, cols, batch); \
        return true;
            FLLM_SYCL_FORMATS(X)
#undef X
            default:
                return false;
        }
    }
    return false;
}

}  // namespace fllm