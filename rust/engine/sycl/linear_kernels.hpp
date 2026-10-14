// SPDX-License-Identifier: Apache-2.0
// fLLM Intel GPU backend (SYCL / oneAPI): kernel templates of the linear projections y[b][r] = dot(x[b], w[r]) + bias[r].
//
//   LinSmall  decode and small batches (1..8 rows of x): one sub-group of 16 work-items per R weight rows, every work-item
//             decodes units of the row and multiplies them with the activations in FP32; sub-group reduction at the end.
//             The same kernel serves the mixture-of-experts projections (MOE: the weight is picked by an expert index).
//   LinTiled  prompt processing without matrix engines: a work-group decodes a 32 x 256 weight tile into local memory and
//             multiplies it with a 32 x 64 activation tile, FP32 throughout.
//   LinXmx    prompt processing on Intel matrix engines (XMX): weight tile decoded to fp16 in local memory in the packed (VNNI)
//             layout, activation tile in the row-major layout, sycl::ext::oneapi::experimental::matrix::joint_matrix_mad,
//             FP32 accumulation. N = 16 (sub-group 16: Ponte Vecchio, Xe2) or N = 8 (sub-group 8: Arc Alchemist).
// Verification level: L0 (compiled with the DPC++ front end only).
#pragma once

#include <sycl/sycl.hpp>

#include "formats.hpp"

namespace fllm {

namespace xmx = sycl::ext::oneapi::experimental::matrix;

constexpr int SGSZ = 16;  // sub-group size of all kernels except LinXmx<_, 8>
constexpr int SGPW = 4;   // sub-groups per work-group of LinSmall

// Rows a sub-group handles at once: bounded by the registers the decoded values need.
template <class F, int B>
constexpr int rows_per_sg() {
    return B == 1 ? (F::V >= 32 ? 2 : 4) : (B == 2 ? (F::V >= 32 ? 1 : 2) : 1);
}

template <class F, int B, int R, bool MOE>
struct LinSmall {
    const uint8_t* w;
    const void* const* tab;
    const float* const* btab;
    const int32_t* ids;
    const float* x;
    float* y;
    const float* bias;
    int rows, cols, nbp, x_div;
    size_t row_bytes;

    [[sycl::reqd_sub_group_size(SGSZ)]] void operator()(sycl::nd_item<2> it) const {
        auto sg = it.get_sub_group();
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const int sgid = static_cast<int>(sg.get_group_linear_id());
        const int r0 = static_cast<int>(it.get_group(1)) * (SGPW * R) + sgid * R;
        if (r0 >= rows) return;  // uniform over the sub-group
        const uint8_t* wb = w;
        const float* xr = x;
        float* yr = y;
        const float* br = bias;
        if constexpr (MOE) {
            const int slot = static_cast<int>(it.get_group(0));
            const int e = ids[slot];
            wb = static_cast<const uint8_t*>(tab[e]);
            xr = x + static_cast<size_t>(slot / x_div) * cols;
            yr = y + static_cast<size_t>(slot) * rows;
            br = btab != nullptr ? btab[e] : nullptr;
        }
        float acc[R][B];
#pragma unroll
        for (int r = 0; r < R; r++) {
#pragma unroll
            for (int b = 0; b < B; b++) acc[r][b] = 0.f;
        }
        const uint8_t* rowp[R];
#pragma unroll
        for (int r = 0; r < R; r++) {
            const int rr = r0 + r < rows ? r0 + r : rows - 1;
            rowp[r] = wb + static_cast<size_t>(rr) * row_bytes;
        }
        const int nblk = F::TAIL ? nbp : cols / F::BLK;
        const int total = nblk * F::UPB;
        for (int t = lane; t < total; t += SGSZ) {
            const int blk = t / F::UPB;
            const int u = t % F::UPB;
            float wv[R][F::V];
#pragma unroll
            for (int r = 0; r < R; r++) F::decode(rowp[r], nbp, blk, u, wv[r]);
#pragma unroll
            for (int i = 0; i < F::V; i++) {
                const int k = blk * F::BLK + F::pos(u, i);
#pragma unroll
                for (int b = 0; b < B; b++) {
                    const float xv = (F::TAIL && k >= cols) ? 0.f : xr[static_cast<size_t>(b) * cols + k];
#pragma unroll
                    for (int r = 0; r < R; r++) acc[r][b] += wv[r][i] * xv;
                }
            }
        }
#pragma unroll
        for (int r = 0; r < R; r++) {
#pragma unroll
            for (int b = 0; b < B; b++) {
                const float v = sycl::reduce_over_group(sg, acc[r][b], sycl::plus<float>());
                if (lane == 0 && r0 + r < rows) yr[static_cast<size_t>(b) * rows + r0 + r] = v + (br != nullptr ? br[r0 + r] : 0.f);
            }
        }
    }
};

// ---- shared tile decoding -----------------------------------------------------------------------------------------------
constexpr int KT = 256;  // values per k step of the tiled kernels

template <class F>
constexpr int blocks_per_step() {
    return KT / F::BLK;
}

// ---- LinTiled -----------------------------------------------------------------------------------------------------------------
template <class F>
struct LinTiled {
    static constexpr int TR = 32, TB = 32, KS = 64, WG = 256;
    const uint8_t* w;
    const float* x;
    float* y;
    const float* bias;
    int rows, cols, batch, nbp;
    size_t row_bytes;
    sycl::local_accessor<float, 1> wt;  // TR * (KT + 1)
    sycl::local_accessor<float, 1> xt;  // TB * (KS + 1)

    [[sycl::reqd_sub_group_size(SGSZ)]] void operator()(sycl::nd_item<2> it) const {
        const int lid = static_cast<int>(it.get_local_id(1));
        const int tr = lid % 16, tb = lid / 16;
        const int b0 = static_cast<int>(it.get_group(0)) * TB;
        const int r0 = static_cast<int>(it.get_group(1)) * TR;
        const int nblk = F::TAIL ? nbp : cols / F::BLK;
        const int ksteps = (nblk * F::BLK + KT - 1) / KT;
        constexpr int BPS = blocks_per_step<F>();
        float acc[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
        float* wts = &wt[0];
        float* xts = &xt[0];
        for (int ks = 0; ks < ksteps; ks++) {
            sycl::group_barrier(it.get_group());
            for (int idx = lid; idx < TR * BPS * F::UPB; idx += WG) {
                const int r = idx / (BPS * F::UPB);
                const int rem = idx % (BPS * F::UPB);
                const int bl = rem / F::UPB;
                const int u = rem % F::UPB;
                const int blk = ks * BPS + bl;
                float wv[F::V];
                if (r0 + r < rows && blk < nblk) {
                    F::decode(w + static_cast<size_t>(r0 + r) * row_bytes, nbp, blk, u, wv);
                } else {
#pragma unroll
                    for (int i = 0; i < F::V; i++) wv[i] = 0.f;
                }
#pragma unroll
                for (int i = 0; i < F::V; i++) wts[r * (KT + 1) + bl * F::BLK + F::pos(u, i)] = wv[i];
            }
            for (int kk0 = 0; kk0 < KT; kk0 += KS) {
                sycl::group_barrier(it.get_group());
                for (int idx = lid; idx < TB * KS; idx += WG) {
                    const int b = idx / KS, k = idx % KS;
                    const int gb = b0 + b, gk = ks * KT + kk0 + k;
                    xts[b * (KS + 1) + k] = (gb < batch && gk < cols) ? x[static_cast<size_t>(gb) * cols + gk] : 0.f;
                }
                sycl::group_barrier(it.get_group());
                for (int k = 0; k < KS; k++) {
                    const float w0 = wts[tr * (KT + 1) + kk0 + k], w1 = wts[(tr + 16) * (KT + 1) + kk0 + k];
                    const float x0 = xts[tb * (KS + 1) + k], x1 = xts[(tb + 16) * (KS + 1) + k];
                    acc[0][0] += w0 * x0;
                    acc[0][1] += w1 * x0;
                    acc[1][0] += w0 * x1;
                    acc[1][1] += w1 * x1;
                }
            }
        }
#pragma unroll
        for (int j = 0; j < 2; j++) {
#pragma unroll
            for (int i = 0; i < 2; i++) {
                const int gb = b0 + tb + 16 * j, gr = r0 + tr + 16 * i;
                if (gb < batch && gr < rows) y[static_cast<size_t>(gb) * rows + gr] = acc[j][i] + (bias != nullptr ? bias[gr] : 0.f);
            }
        }
    }
};

// ---- LinXmx -------------------------------------------------------------------------------------------------------------------
template <class F, int N>
struct LinXmx {
    static constexpr int SGZ = (N == 8) ? 8 : 16;  // sub-group size the matrix engine shape needs
    static constexpr int SGS = 8;                  // sub-groups per work-group: 2 (batch) x 4 (weight rows)
    static constexpr int WG = SGS * SGZ;
    static constexpr int TB = 32, TR = 64;
    static constexpr int NB = 16 / N;  // n-blocks of N weight rows per sub-group (16 rows)
    static constexpr int MB = 2;       // m-blocks of 8 activation rows per sub-group (16 rows)
    const uint8_t* w;
    const float* x;
    float* y;
    const float* bias;
    int rows, cols, batch, nbp;
    size_t row_bytes;
    sycl::local_accessor<sycl::half, 1> wt;  // TR * KT, packed (VNNI) B fragments
    sycl::local_accessor<sycl::half, 1> xt;  // TB * KT, row-major A fragments
    sycl::local_accessor<float, 1> cs;       // SGS * 8 * N, accumulator staging

    template <class T>
    static auto local_ptr(T* p) {
        return sycl::address_space_cast<sycl::access::address_space::local_space, sycl::access::decorated::no>(p);
    }

    [[sycl::reqd_sub_group_size(SGZ)]] void operator()(sycl::nd_item<2> it) const {
        auto sg = it.get_sub_group();
        const int lid = static_cast<int>(it.get_local_id(1));
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const int sgid = static_cast<int>(sg.get_group_linear_id());
        const int sb = sgid / 4, sr = sgid % 4;
        const int b0 = static_cast<int>(it.get_group(0)) * TB;
        const int r0 = static_cast<int>(it.get_group(1)) * TR;
        const int nblk = F::TAIL ? nbp : cols / F::BLK;
        const int ksteps = (nblk * F::BLK + KT - 1) / KT;
        constexpr int BPS = blocks_per_step<F>();
        sycl::half* wts = &wt[0];
        sycl::half* xts = &xt[0];
        float* css = &cs[0] + sgid * 8 * N;

        xmx::joint_matrix<sycl::sub_group, float, xmx::use::accumulator, 8, N> c[MB][NB];
#pragma unroll
        for (int m = 0; m < MB; m++) {
#pragma unroll
            for (int n = 0; n < NB; n++) xmx::joint_matrix_fill(sg, c[m][n], 0.0f);
        }

        for (int ks = 0; ks < ksteps; ks++) {
            sycl::group_barrier(it.get_group());
            // weight tile -> packed B fragments: element (row r, k) at (((k/16) * (TR/N) + r/N) * 8 + (k%16)/2) * 2N + (r%N) * 2 + k%2
            for (int idx = lid; idx < TR * BPS * F::UPB; idx += WG) {
                const int r = idx / (BPS * F::UPB);
                const int rem = idx % (BPS * F::UPB);
                const int bl = rem / F::UPB;
                const int u = rem % F::UPB;
                const int blk = ks * BPS + bl;
                float wv[F::V];
                if (r0 + r < rows && blk < nblk) {
                    F::decode(w + static_cast<size_t>(r0 + r) * row_bytes, nbp, blk, u, wv);
                } else {
#pragma unroll
                    for (int i = 0; i < F::V; i++) wv[i] = 0.f;
                }
#pragma unroll
                for (int i = 0; i < F::V; i++) {
                    const int k = bl * F::BLK + F::pos(u, i);
                    const int at = (((k >> 4) * (TR / N) + r / N) * 8 + ((k & 15) >> 1)) * (2 * N) + (r % N) * 2 + (k & 1);
                    wts[at] = sycl::half(wv[i]);
                }
            }
            // activation tile -> row-major A fragments of 8 x 16: element (b, k) at (((k/16) * (TB/8) + b/8) * 8 + b%8) * 16 + k%16
            for (int idx = lid; idx < TB * KT; idx += WG) {
                const int b = idx / KT, k = idx % KT;
                const int gb = b0 + b, gk = ks * KT + k;
                const float v = (gb < batch && gk < cols) ? x[static_cast<size_t>(gb) * cols + gk] : 0.f;
                xts[(((k >> 4) * (TB / 8) + (b >> 3)) * 8 + (b & 7)) * 16 + (k & 15)] = sycl::half(v);
            }
            sycl::group_barrier(it.get_group());
            for (int kc = 0; kc < KT / 16; kc++) {
                xmx::joint_matrix<sycl::sub_group, sycl::half, xmx::use::a, 8, 16, xmx::layout::row_major> ma[MB];
#pragma unroll
                for (int m = 0; m < MB; m++) {
                    xmx::joint_matrix_load(sg, ma[m], local_ptr(xts + (kc * (TB / 8) + sb * MB + m) * 128), 16);
                }
#pragma unroll
                for (int n = 0; n < NB; n++) {
                    xmx::joint_matrix<sycl::sub_group, sycl::half, xmx::use::b, 16, N, xmx::layout::ext_intel_packed> mb;
                    xmx::joint_matrix_load(sg, mb, local_ptr(wts + (kc * (TR / N) + sr * NB + n) * (16 * N)), 2 * N);
#pragma unroll
                    for (int m = 0; m < MB; m++) xmx::joint_matrix_mad(sg, c[m][n], ma[m], mb, c[m][n]);
                }
            }
        }
        // epilogue: through local memory so that the edges of the output can be guarded
#pragma unroll
        for (int m = 0; m < MB; m++) {
#pragma unroll
            for (int n = 0; n < NB; n++) {
                xmx::joint_matrix_store(sg, c[m][n], local_ptr(css), N, xmx::layout::row_major);
                sycl::group_barrier(sg);
                for (int e = lane; e < 8 * N; e += SGZ) {
                    const int mm = e / N, nn = e % N;
                    const int gb = b0 + sb * 16 + m * 8 + mm, gr = r0 + sr * 16 + n * N + nn;
                    if (gb < batch && gr < rows) y[static_cast<size_t>(gb) * rows + gr] = css[e] + (bias != nullptr ? bias[gr] : 0.f);
                }
                sycl::group_barrier(sg);
            }
        }
    }
};

}  // namespace fllm
