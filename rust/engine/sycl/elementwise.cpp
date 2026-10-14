// SPDX-License-Identifier: Apache-2.0
// fLLM Intel GPU backend (SYCL / oneAPI): normalisation, rotary embedding, gated activations, residual, KV-cache writes, greedy
// argmax, row gather/scatter and the mixture-of-experts helpers. Semantics follow the CUDA kernels (rust/engine/cuda/elementwise.cu)
// and the CPU backend. Verification level: L0.
#include <algorithm>
#include <limits>
#include <stdexcept>

#include "common.hpp"
#include "formats.hpp"

namespace fllm {

namespace {

constexpr int NWG = 256;  // work-group size of the row kernels
constexpr int NSG = NWG / 16;

// Sum over the work-group: sub-group reduction, then the sub-group totals through local memory (must be called by all items).
inline float wg_sum(sycl::nd_item<1> it, float v, float* scratch) {
    auto sg = it.get_sub_group();
    v = sycl::reduce_over_group(sg, v, sycl::plus<float>());
    sycl::group_barrier(it.get_group());
    if (sg.get_local_linear_id() == 0) scratch[sg.get_group_linear_id()] = v;
    sycl::group_barrier(it.get_group());
    float total = 0.f;
    for (int i = 0; i < NSG; i++) total += scratch[i];
    return total;
}

struct RmsNorm {
    const float* x;
    const float* w;
    float* out;
    int cols;
    float eps;
    sycl::local_accessor<float, 1> sc;
    [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> it) const {
        const size_t row = it.get_group(0);
        const float* xr = x + row * cols;
        float* orow = out + row * cols;
        float sum = 0.f;
        for (int c = static_cast<int>(it.get_local_id(0)); c < cols; c += NWG) sum = sycl::fma(xr[c], xr[c], sum);
        const float total = wg_sum(it, sum, &sc[0]);
        const float inv = 1.0f / sycl::sqrt(total / static_cast<float>(cols) + eps);
        for (int c = static_cast<int>(it.get_local_id(0)); c < cols; c += NWG) orow[c] = xr[c] * (inv * w[c]);
    }
};

// out = (x - mean) * rsqrt(var + eps) * weight + bias, biased variance (matches the CPU backend)
struct LayerNorm {
    const float* x;
    const float* w;
    const float* b;
    float* out;
    int cols;
    float eps;
    sycl::local_accessor<float, 1> sc;
    [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> it) const {
        const size_t row = it.get_group(0);
        const float* xr = x + row * cols;
        float* orow = out + row * cols;
        float sum = 0.f;
        for (int c = static_cast<int>(it.get_local_id(0)); c < cols; c += NWG) sum += xr[c];
        const float mean = wg_sum(it, sum, &sc[0]) / static_cast<float>(cols);
        float sq = 0.f;
        for (int c = static_cast<int>(it.get_local_id(0)); c < cols; c += NWG) sq += (xr[c] - mean) * (xr[c] - mean);
        const float var = wg_sum(it, sq, &sc[0]) / static_cast<float>(cols);
        const float inv = 1.0f / sycl::sqrt(var + eps);
        for (int c = static_cast<int>(it.get_local_id(0)); c < cols; c += NWG)
            orow[c] = (xr[c] - mean) * inv * w[c] + (b != nullptr ? b[c] : 0.f);
    }
};

// Greedy sampling: the first index of the largest value of a row wins, NaN never does.
struct ArgmaxRows {
    const float* logits;
    int vocab;
    int32_t* out;
    sycl::local_accessor<float, 1> sval;
    sycl::local_accessor<int32_t, 1> sidx;
    void operator()(sycl::nd_item<1> it) const {
        const int t = static_cast<int>(it.get_local_id(0));
        const float* row = logits + it.get_group(0) * static_cast<size_t>(vocab);
        float best = -std::numeric_limits<float>::infinity();
        int32_t besti = 0;
        for (int i = t; i < vocab; i += NWG) {
            const float v = row[i];
            if (v > best) {
                best = v;
                besti = i;
            }
        }
        sval[t] = best;
        sidx[t] = besti;
        sycl::group_barrier(it.get_group());
        for (int stride = NWG / 2; stride > 0; stride >>= 1) {
            if (t < stride) {
                const float ov = sval[t + stride];
                const int32_t oi = sidx[t + stride];
                if (ov > sval[t] || (ov == sval[t] && oi < sidx[t])) {
                    sval[t] = ov;
                    sidx[t] = oi;
                }
            }
            sycl::group_barrier(it.get_group());
        }
        if (t == 0) out[it.get_group(0)] = sidx[0];
    }
};

// Softmax gating over `experts` logits, k best by score (ties to the lower index), optional renormalisation. One sub-group per row.
struct MoeRoute {
    const float* logits;
    int32_t* ids;
    float* w;
    int experts, k, normalize;
    float scale;
    [[sycl::reqd_sub_group_size(16)]] void operator()(sycl::nd_item<1> it) const {
        auto sg = it.get_sub_group();
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const size_t row = it.get_group(0);
        const float* l = logits + row * experts;
        float m = -3.0e38f;
        for (int e = lane; e < experts; e += 16) m = sycl::fmax(m, l[e]);
        m = sycl::reduce_over_group(sg, m, sycl::maximum<float>());
        float sum = 0.f;
        for (int e = lane; e < experts; e += 16) sum += sycl::exp(l[e] - m);
        sum = sycl::reduce_over_group(sg, sum, sycl::plus<float>());
        int32_t chosen[16];
        float score[16];
        float total = 0.f;
        for (int j = 0; j < k; j++) {
            float best = -3.0e38f;
            int32_t bi = 0x7fffffff;
            for (int e = lane; e < experts; e += 16) {
                bool taken = false;
                for (int i = 0; i < j; i++) taken |= chosen[i] == e;
                if (!taken && (l[e] > best || (l[e] == best && e < bi))) {
                    best = l[e];
                    bi = e;
                }
            }
            const float gbest = sycl::reduce_over_group(sg, best, sycl::maximum<float>());
            const int32_t gi = sycl::reduce_over_group(sg, best == gbest ? bi : 0x7fffffff, sycl::minimum<int32_t>());
            chosen[j] = gi;
            score[j] = sycl::exp(gbest - m) / sum;
            total += score[j];
        }
        if (lane == 0) {
            for (int j = 0; j < k; j++) {
                ids[row * k + j] = chosen[j];
                w[row * k + j] = (normalize ? score[j] / total : score[j]) * scale;
            }
        }
    }
};

inline float glu_value(float g, float u, int mode) {
    const float a = mode == 0 ? g / (1.0f + sycl::exp(-g)) : 0.5f * g * (1.0f + sycl::tanh(0.7978846f * (g + 0.044715f * g * g * g)));
    return a * u;
}

}  // namespace

}  // namespace fllm

using fllm::guarded;
using fllm::Stream;

#define FLLM_STREAM Stream& s = *reinterpret_cast<Stream*>(stp)

extern "C" {

int fllm_sycl_rms_norm(FllmSyclStream* stp, const float* x, const float* w, float* out, int rows, int cols, float eps) {
    return guarded("rms_norm", [&] {
        if (rows <= 0) return;
        FLLM_STREAM;
        s.q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> sc(sycl::range<1>(fllm::NSG), h);
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(rows) * fllm::NWG), sycl::range<1>(fllm::NWG)),
                           fllm::RmsNorm{x, w, out, cols, eps, sc});
        });
    });
}

int fllm_sycl_layer_norm(FllmSyclStream* stp, const float* x, const float* w, const float* b, float* out, int rows, int cols,
                         float eps) {
    return guarded("layer_norm", [&] {
        if (rows <= 0) return;
        FLLM_STREAM;
        s.q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> sc(sycl::range<1>(fllm::NSG), h);
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(rows) * fllm::NWG), sycl::range<1>(fllm::NWG)),
                           fllm::LayerNorm{x, w, b, out, cols, eps, sc});
        });
    });
}

int fllm_sycl_glu(FllmSyclStream* stp, const float* gate, const float* up, float* out, size_t count, int mode) {
    return guarded("glu", [&] {
        if (count == 0) return;
        FLLM_STREAM;
        s.q.parallel_for(sycl::range<1>(count), [=](sycl::id<1> i) { out[i] = fllm::glu_value(gate[i], up[i], mode); });
    });
}

// gpt-oss: out = (min(g, limit) * sigmoid(alpha * min(g, limit))) * (clamp(u, -limit, limit) + 1)
int fllm_sycl_swiglu_oai(FllmSyclStream* stp, const float* gate, const float* up, float* out, size_t count, float alpha,
                         float limit) {
    return guarded("swiglu_oai", [&] {
        if (count == 0) return;
        FLLM_STREAM;
        s.q.parallel_for(sycl::range<1>(count), [=](sycl::id<1> i) {
            const float g = sycl::fmin(gate[i], limit);
            const float u = sycl::fmin(sycl::fmax(up[i], -limit), limit);
            out[i] = (u + 1.f) * g / (1.f + sycl::exp(-alpha * g));
        });
    });
}

int fllm_sycl_residual(FllmSyclStream* stp, float* x, const float* res, size_t count, float scale) {
    return guarded("residual", [&] {
        if (count == 0) return;
        FLLM_STREAM;
        s.q.parallel_for(sycl::range<1>(count), [=](sycl::id<1> i) { x[i] = x[i] + res[i] * scale; });
    });
}

// One work-item rotates one pair. Query heads first, then key heads.
int fllm_sycl_rope(FllmSyclStream* stp, float* q, float* k, const int32_t* pos_host, int rows, const float* cos_dev,
                   const float* sin_dev, int heads, int kv_heads, int dim, int half_rot, int neox) {
    return guarded("rope", [&] {
        if (rows <= 0) return;
        FLLM_STREAM;
        const int32_t* pos = static_cast<const int32_t*>(s.stage(pos_host, static_cast<size_t>(rows) * sizeof(int32_t)));
        const int per_row = (heads + kv_heads) * half_rot;
        const size_t total = static_cast<size_t>(rows) * per_row;
        s.q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
            const size_t i = id[0];
            const int r = static_cast<int>(i / per_row), rem = static_cast<int>(i % per_row);
            const int h = rem / half_rot, d = rem % half_rot;
            const size_t p = static_cast<size_t>(pos[r]);
            const float c = cos_dev[p * half_rot + d], sn = sin_dev[p * half_rot + d];
            const int ia = neox ? d : 2 * d, ib = neox ? d + half_rot : 2 * d + 1;
            float* base = h < heads ? q + (static_cast<size_t>(r) * heads + h) * dim
                                    : k + (static_cast<size_t>(r) * kv_heads + (h - heads)) * dim;
            const float a = base[ia], b = base[ib];
            base[ia] = a * c - b * sn;
            base[ib] = b * c + a * sn;
        });
    });
}

// KV cache write. f16: one work-item per value. q8: one work-item per group of `group` values: d = amax / 127, scale = f16(d),
// q = round-half-away(x / d) clamped to +-127 (the CPU encoder).
int fllm_sycl_kv_write(FllmSyclStream* stp, const float* src, uint16_t* pool, const int32_t* slots_host, int rows, int kv,
                       int kind, int group, int scale_base_words) {
    return guarded("kv_write", [&] {
        if (rows <= 0) return;
        FLLM_STREAM;
        const int32_t* slots = static_cast<const int32_t*>(s.stage(slots_host, static_cast<size_t>(rows) * sizeof(int32_t)));
        if (kind == 1) {
            const size_t total = static_cast<size_t>(rows) * kv;
            s.q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
                const size_t i = id[0];
                const size_t slot = static_cast<size_t>(slots[i / kv]);
                pool[slot * kv + i % kv] = fllm::f2h(src[i]);
            });
        } else {
            const int groups = kv / group;
            const size_t total = static_cast<size_t>(rows) * groups;
            int8_t* qbytes = reinterpret_cast<int8_t*>(pool);
            uint16_t* scales = pool + scale_base_words;
            s.q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
                const size_t i = id[0];
                const int r = static_cast<int>(i / groups), g = static_cast<int>(i % groups);
                const size_t slot = static_cast<size_t>(slots[r]);
                const float* sp = src + static_cast<size_t>(r) * kv + static_cast<size_t>(g) * group;
                float amax = 0.f;
                for (int j = 0; j < group; j++) amax = sycl::fmax(amax, sycl::fabs(sp[j]));
                const float d = amax / 127.0f;
                scales[slot * groups + g] = fllm::f2h(d);
                int8_t* dst = qbytes + slot * kv + static_cast<size_t>(g) * group;
                for (int j = 0; j < group; j++) {
                    float qv = 0.f;
                    if (d != 0.f) {
                        const float t = sp[j] / d;
                        qv = sycl::fmin(sycl::fmax(sycl::copysign(sycl::floor(sycl::fabs(t) + 0.5f), t), -127.f), 127.f);
                    }
                    dst[j] = static_cast<int8_t>(static_cast<int>(qv));
                }
            });
        }
    });
}

int fllm_sycl_argmax(FllmSyclStream* stp, const float* logits, int rows, int vocab, int32_t* host_out) {
    return guarded("argmax", [&] {
        if (rows <= 0 || vocab <= 0) return;
        FLLM_STREAM;
        int32_t* tmp = static_cast<int32_t*>(s.get_scratch(static_cast<size_t>(rows) * sizeof(int32_t)));
        s.q.submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> sval(sycl::range<1>(fllm::NWG), h);
            sycl::local_accessor<int32_t, 1> sidx(sycl::range<1>(fllm::NWG), h);
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(rows) * fllm::NWG), sycl::range<1>(fllm::NWG)),
                           fllm::ArgmaxRows{logits, vocab, tmp, sval, sidx});
        });
        s.q.memcpy(host_out, tmp, static_cast<size_t>(rows) * sizeof(int32_t));
        s.drain();
    });
}

// dst[i] = src[idx[i]] * w[i] (w optional, host arrays)
int fllm_sycl_gather_rows(FllmSyclStream* stp, float* dst, const float* src, const int32_t* idx_host, const float* w_host, int n,
                          int row_len) {
    return guarded("gather_rows", [&] {
        if (n <= 0) return;
        FLLM_STREAM;
        const int32_t* idx = static_cast<const int32_t*>(s.stage(idx_host, static_cast<size_t>(n) * sizeof(int32_t)));
        const float* w = w_host != nullptr ? static_cast<const float*>(s.stage(w_host, static_cast<size_t>(n) * sizeof(float))) : nullptr;
        s.q.parallel_for(sycl::range<1>(static_cast<size_t>(n) * row_len), [=](sycl::id<1> id) {
            const size_t i = id[0];
            const size_t r = i / row_len, c = i % row_len;
            const float v = src[static_cast<size_t>(idx[r]) * row_len + c];
            dst[i] = w != nullptr ? v * w[r] : v;
        });
    });
}

// dst[idx[i]] += src[i] * w[i]; the destinations of one call are distinct, so a plain read-modify-write is enough
int fllm_sycl_scatter_add_rows(FllmSyclStream* stp, float* dst, const float* src, const int32_t* idx_host, const float* w_host,
                               int n, int row_len) {
    return guarded("scatter_add_rows", [&] {
        if (n <= 0) return;
        FLLM_STREAM;
        const int32_t* idx = static_cast<const int32_t*>(s.stage(idx_host, static_cast<size_t>(n) * sizeof(int32_t)));
        const float* w = w_host != nullptr ? static_cast<const float*>(s.stage(w_host, static_cast<size_t>(n) * sizeof(float))) : nullptr;
        s.q.parallel_for(sycl::range<1>(static_cast<size_t>(n) * row_len), [=](sycl::id<1> id) {
            const size_t i = id[0];
            const size_t r = i / row_len, c = i % row_len;
            dst[static_cast<size_t>(idx[r]) * row_len + c] += (w != nullptr ? w[r] : 1.f) * src[i];
        });
    });
}

int fllm_sycl_moe_route(FllmSyclStream* stp, const float* logits, int32_t* ids, float* w, int rows, int experts, int k,
                        int normalize, float scale) {
    return guarded("moe_route", [&] {
        if (k < 1 || k > 16 || experts < k) throw std::runtime_error("unsupported expert counts");
        if (rows <= 0) return;
        FLLM_STREAM;
        s.q.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(rows) * 16), sycl::range<1>(16)),
                         fllm::MoeRoute{logits, ids, w, experts, k, normalize, scale});
    });
}

int fllm_sycl_moe_combine(FllmSyclStream* stp, float* out, const float* y, const float* w, int tokens, int k, int hidden) {
    return guarded("moe_combine", [&] {
        if (tokens <= 0) return;
        FLLM_STREAM;
        s.q.parallel_for(sycl::range<1>(static_cast<size_t>(tokens) * hidden), [=](sycl::id<1> id) {
            const size_t i = id[0];
            const size_t t = i / hidden, h = i % hidden;
            float acc = 0.f;
            for (int j = 0; j < k; j++) acc += w[t * k + j] * y[(t * k + j) * hidden + h];
            out[i] = acc;
        });
    });
}

// Gated attention: the query projection holds query and gate of every head side by side per head, or all queries then all gates.
int fllm_sycl_split_q_gate(FllmSyclStream* stp, const float* qg, float* q, float* gate, int rows, int heads, int dk, int per_head) {
    return guarded("split_q_gate", [&] {
        const size_t total = static_cast<size_t>(rows) * heads * dk;
        if (total == 0) return;
        FLLM_STREAM;
        s.q.parallel_for(sycl::range<1>(total), [=](sycl::id<1> id) {
            const size_t i = id[0];
            const size_t width = static_cast<size_t>(heads) * dk;
            const size_t r = i / width, rem = i % width;
            const size_t h = rem / dk, d = rem % dk;
            const size_t base = r * 2 * width;
            const size_t sq = per_head ? base + h * 2 * dk + d : base + h * dk + d;
            const size_t sg = per_head ? sq + dk : base + width + h * dk + d;
            q[i] = qg[sq];
            gate[i] = qg[sg];
        });
    });
}

int fllm_sycl_sigmoid_mul(FllmSyclStream* stp, float* x, const float* g, size_t n) {
    return guarded("sigmoid_mul", [&] {
        if (n == 0) return;
        FLLM_STREAM;
        s.q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) { x[id] *= 1.f / (1.f + sycl::exp(-g[id])); });
    });
}

}  // extern "C"
