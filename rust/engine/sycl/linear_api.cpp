// SPDX-License-Identifier: Apache-2.0
// fLLM Intel GPU backend (SYCL / oneAPI): C entry points of the linear projections, the mixture-of-experts projections and the
// embedding lookup; picks the kernel family by batch size and device. Verification level: L0.
#include "common.hpp"

using fllm::guarded;
using fllm::Stream;

extern "C" {

// y[b][r] = dot(x[b], w[r]) + bias[r]
//   batch <= 8        LinSmall (exact FP32, decode)
//   batch >= 16, XMX  LinXmx   (fp16 inputs, FP32 accumulation; off with fllm_sycl_set_exact(1) or FLLM_SYCL_EXACT)
//   otherwise         LinTiled (exact FP32, shared-memory tiles)
int fllm_sycl_linear(FllmSyclStream* stp, const void* w, int kind, int rows, int cols, const float* x, float* y,
                     const float* bias, int batch) {
    return guarded("linear", [&] {
        if (rows <= 0 || batch <= 0) return;
        Stream& s = *reinterpret_cast<Stream*>(stp);
        const int k = fllm::device_kind(kind);
        bool ok;
        if (batch <= 8) ok = fllm::launch_linear_small(s, w, k, rows, cols, x, y, bias, batch);
        else if (s.dev->xmx_n != 0 && batch >= 16 && !fllm::exact_math()) ok = fllm::launch_linear_xmx(s, w, k, rows, cols, x, y, bias, batch);
        else ok = fllm::launch_linear_tiled(s, w, k, rows, cols, x, y, bias, batch);
        if (!ok) throw std::runtime_error("unsupported weight kind " + std::to_string(kind));
    });
}

int fllm_sycl_moe_linear(FllmSyclStream* stp, const void* const* tab, const float* const* btab, int kind, const int32_t* ids,
                         const float* x, float* y, int rows, int cols, int slots, int x_div) {
    return guarded("moe_linear", [&] {
        if (rows <= 0 || slots <= 0) return;
        if (x_div <= 0) throw std::runtime_error("x_div must be positive");
        Stream& s = *reinterpret_cast<Stream*>(stp);
        if (!fllm::launch_moe_linear(s, tab, btab, fllm::device_kind(kind), ids, x, y, rows, cols, slots, x_div))
            throw std::runtime_error("unsupported weight kind " + std::to_string(kind));
    });
}

int fllm_sycl_embedding(FllmSyclStream* stp, const void* w, int kind, const int64_t* ids_host, int n, float* out, int cols) {
    return guarded("embedding", [&] {
        if (n <= 0) return;
        Stream& s = *reinterpret_cast<Stream*>(stp);
        const int64_t* ids = static_cast<const int64_t*>(s.stage(ids_host, static_cast<size_t>(n) * sizeof(int64_t)));
        if (!fllm::launch_embedding(s, w, fllm::device_kind(kind), ids, n, out, cols))
            throw std::runtime_error("unsupported weight kind " + std::to_string(kind));
    });
}

}  // extern "C"