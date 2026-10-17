// SPDX-License-Identifier: Apache-2.0
// fLLM Intel GPU backend (SYCL / oneAPI): paged attention over the device KV cache (decode and prompt rows).
//
// Row r of a batch attends to positions start..lengths[r]-1 of its block table (start from the chunk / window, see AttentionParams in
// rust/engine/src/backend/mod.rs); a score is dot(q, k) * scale (+ ALiBi slope * (p - (n - 1))), soft-capped as cap * tanh(s / cap);
// a sink adds exp(sink) to the softmax denominator. Handles f16 and q8_0 caches, GQA, head widths up to 512.
//
// Two stages (flash-decoding): AttnPartial runs one work-group of 8 sub-groups per (row, head, split); every sub-group walks the
// positions p0 + sgid, p0 + sgid + 8 ... with an online softmax, a lane owns CH consecutive elements of the head; the sub-groups
// are merged through local memory and the (max, denominator, accumulators) of the split are written to scratch. AttnMerge combines
// the splits (and the sink) into the output. The number of splits grows when there are few (row, head) pairs for the device
// (decode of a single sequence on a long context). Verification level: L0.
#include "common.hpp"
#include "formats.hpp"

namespace fllm {

namespace {

constexpr int ASG = 16;   // sub-group size
constexpr int ANSG = 8;   // sub-groups per work-group
constexpr int AWG = ANSG * ASG;
constexpr float kFloor = -3.0e38f;

template <int CH, int KIND>
struct AttnPartial {
    const float* q;
    const uint16_t* kpool;
    const uint16_t* vpool;
    float* part;
    const int64_t* tables;
    const int64_t* lengths;
    const float* alibi;
    int heads, kv_heads, dk, dv, stride, block_size;
    int kgroup, vgroup, kscale_base, vscale_base;
    int window, chunk, splits;
    float scale, softcap;
    sycl::local_accessor<float, 1> wacc;  // ANSG * CH * ASG
    sycl::local_accessor<float, 1> wml;   // ANSG * 2

    [[sycl::reqd_sub_group_size(ASG)]] void operator()(sycl::nd_item<2> it) const {
        auto sg = it.get_sub_group();
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const int sgid = static_cast<int>(sg.get_group_linear_id());
        // dimension 1 is the fastest one and holds the work-items of the sub-groups (SYCL: the local range of the last dimension must be a
        // multiple of the sub-group size; the OpenCL CPU runtime crashes when it is 1)
        const int lid = static_cast<int>(it.get_local_id(1));
        const int item = static_cast<int>(it.get_group(1));
        const int split = static_cast<int>(it.get_group(0));
        const int r = item / heads, h = item % heads;
        const int kh = h / (heads / kv_heads);
        const int n = static_cast<int>(lengths[r]);
        int start = 0;
        if (chunk != 0) start = (n - 1) / chunk * chunk;
        else if (window != 0 && n > window) start = n - window;
        const int span = n - start;
        const int per = (span + splits - 1) / splits;
        const int p0 = start + split * per;
        const int p1 = sycl::min(n, p0 + per);
        const int64_t* table = tables + static_cast<size_t>(r) * stride;

        const int e0 = lane * CH;
        const int nvk = sycl::clamp(dk - e0, 0, CH);
        const int nvv = sycl::clamp(dv - e0, 0, CH);
        float qr[CH];
#pragma unroll
        for (int c = 0; c < CH; c++) qr[c] = c < nvk ? q[static_cast<size_t>(item) * dk + e0 + c] : 0.f;
        float acc[CH];
#pragma unroll
        for (int c = 0; c < CH; c++) acc[c] = 0.f;
        float m = kFloor, l = 0.f;
        const size_t kvk = static_cast<size_t>(kv_heads) * dk, kvv = static_cast<size_t>(kv_heads) * dv;
        // q8: the chunk of a lane lies inside one scale group (CH divides 32; a group of a whole head contains every chunk)
        const int kidx = KIND == 2 ? (kh * dk + e0) / kgroup : 0;
        const int vidx = KIND == 2 ? (kh * dv + e0) / vgroup : 0;

        for (int p = p0 + sgid; p < p1; p += ANSG) {
            const size_t slot = static_cast<size_t>(table[p / block_size]) * block_size + p % block_size;
            float dot = 0.f;
            if constexpr (KIND == 1) {
                const uint16_t* kp = kpool + slot * kvk + static_cast<size_t>(kh) * dk + e0;
#pragma unroll
                for (int c = 0; c < CH; c++) {
                    if (c < nvk) dot += qr[c] * h2f(kp[c]);
                }
            } else {
                const int8_t* kb = reinterpret_cast<const int8_t*>(kpool) + slot * kvk + static_cast<size_t>(kh) * dk + e0;
                const float ks = nvk > 0 ? h2f(kpool[kscale_base + slot * (kvk / kgroup) + kidx]) : 0.f;
                float sum = 0.f;
#pragma unroll
                for (int c = 0; c < CH; c++) {
                    if (c < nvk) sum += qr[c] * static_cast<float>(kb[c]);
                }
                dot = sum * ks;
            }
            float s = sycl::reduce_over_group(sg, dot, sycl::plus<float>()) * scale;
            if (alibi != nullptr) s += alibi[h] * (static_cast<float>(p) - static_cast<float>(n - 1));
            if (softcap != 0.f) s = softcap * sycl::tanh(sycl::fmin(sycl::fmax(s / softcap, -15.f), 15.f));
            const float m_new = sycl::fmax(m, s);
            const float corr = sycl::exp(m - m_new), e = sycl::exp(s - m_new);
            l = l * corr + e;
            if constexpr (KIND == 1) {
                const uint16_t* vp = vpool + slot * kvv + static_cast<size_t>(kh) * dv + e0;
#pragma unroll
                for (int c = 0; c < CH; c++) acc[c] = c < nvv ? acc[c] * corr + e * h2f(vp[c]) : 0.f;
            } else {
                const int8_t* vb = reinterpret_cast<const int8_t*>(vpool) + slot * kvv + static_cast<size_t>(kh) * dv + e0;
                const float vs = nvv > 0 ? h2f(vpool[vscale_base + slot * (kvv / vgroup) + vidx]) : 0.f;
#pragma unroll
                for (int c = 0; c < CH; c++) acc[c] = c < nvv ? acc[c] * corr + e * (static_cast<float>(vb[c]) * vs) : 0.f;
            }
            m = m_new;
        }

        float* wa = &wacc[0];
        float* ml = &wml[0];
#pragma unroll
        for (int c = 0; c < CH; c++) wa[sgid * (CH * ASG) + e0 + c] = acc[c];
        if (lane == 0) {
            ml[sgid * 2] = m;
            ml[sgid * 2 + 1] = l;
        }
        sycl::group_barrier(it.get_group());
        float M = kFloor;
        for (int s = 0; s < ANSG; s++) M = sycl::fmax(M, ml[s * 2]);
        float* dst = part + (static_cast<size_t>(item) * splits + split) * (dv + 2);
        for (int e = lid; e < dv; e += AWG) {
            float v = 0.f;
            for (int s = 0; s < ANSG; s++) v += wa[s * (CH * ASG) + e] * sycl::exp(ml[s * 2] - M);
            dst[2 + e] = v;
        }
        if (lid == 0) {
            float L = 0.f;
            for (int s = 0; s < ANSG; s++) L += ml[s * 2 + 1] * sycl::exp(ml[s * 2] - M);
            dst[0] = M;
            dst[1] = L;
        }
    }
};

struct AttnMerge {
    const float* part;
    float* out;
    const float* sinks;
    int heads, dv, splits;

    void operator()(sycl::nd_item<1> it) const {
        const int item = static_cast<int>(it.get_group(0));
        const int h = item % heads;
        const size_t stride = static_cast<size_t>(dv) + 2;
        const float* base = part + static_cast<size_t>(item) * splits * stride;
        float M = sinks != nullptr ? sinks[h] : kFloor;
        for (int s = 0; s < splits; s++) M = sycl::fmax(M, base[s * stride]);
        float L = sinks != nullptr ? sycl::exp(sinks[h] - M) : 0.f;
        for (int s = 0; s < splits; s++) L += base[s * stride + 1] * sycl::exp(base[s * stride] - M);
        const float inv = 1.0f / L;
        for (int e = static_cast<int>(it.get_local_id(0)); e < dv; e += static_cast<int>(it.get_local_range(0))) {
            float v = 0.f;
            for (int s = 0; s < splits; s++) v += base[s * stride + 2 + e] * sycl::exp(base[s * stride] - M);
            out[static_cast<size_t>(item) * dv + e] = v * inv;
        }
    }
};

template <int CH, int KIND>
void run_attention(Stream& s, const float* q, const uint16_t* kpool, const uint16_t* vpool, float* out, const int64_t* tables,
                   const int64_t* lengths, const float* sinks, const float* alibi, int rows, int heads, int kv_heads, int dk,
                   int dv, int stride, int block_size, int kgroup, int vgroup, int kscale_base, int vscale_base, float scale,
                   float softcap, int window, int chunk, int splits) {
    const size_t items = static_cast<size_t>(rows) * heads;
    float* part = static_cast<float*>(s.get_scratch(items * splits * (static_cast<size_t>(dv) + 2) * sizeof(float)));
    using K = AttnPartial<CH, KIND>;
    s.q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> wacc(sycl::range<1>(ANSG * CH * ASG), h);
        sycl::local_accessor<float, 1> wml(sycl::range<1>(ANSG * 2), h);
        const K kernel{q,        kpool,   vpool,        part,        tables, lengths,    alibi,  heads, kv_heads, dk,
                       dv,       stride,  block_size,   kgroup,      vgroup, kscale_base, vscale_base, window, chunk,
                       splits,   scale,   softcap,      wacc,        wml};
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>(static_cast<size_t>(splits), items * AWG), sycl::range<2>(1, AWG)), kernel);
    });
    constexpr int MWG = 64;
    s.q.parallel_for(sycl::nd_range<1>(sycl::range<1>(items * MWG), sycl::range<1>(MWG)),
                     AttnMerge{part, out, sinks, heads, dv, splits});
}

template <int KIND>
bool dispatch_attention(int ch, Stream& s, const float* q, const uint16_t* kpool, const uint16_t* vpool, float* out,
                        const int64_t* tables, const int64_t* lengths, const float* sinks, const float* alibi, int rows,
                        int heads, int kv_heads, int dk, int dv, int stride, int block_size, int kgroup, int vgroup,
                        int kscale_base, int vscale_base, float scale, float softcap, int window, int chunk, int splits) {
#define ATT_CASE(C)                                                                                                       \
    case C:                                                                                                               \
        run_attention<C, KIND>(s, q, kpool, vpool, out, tables, lengths, sinks, alibi, rows, heads, kv_heads, dk, dv,     \
                               stride, block_size, kgroup, vgroup, kscale_base, vscale_base, scale, softcap, window, chunk, \
                               splits);                                                                                   \
        return true;
    switch (ch) {
        ATT_CASE(1)
        ATT_CASE(2)
        ATT_CASE(4)
        ATT_CASE(8)
        ATT_CASE(16)
        ATT_CASE(32)
        default:
            return false;
    }
#undef ATT_CASE
}

}  // namespace

}  // namespace fllm

extern "C" int fllm_sycl_attention(FllmSyclStream* stp, const float* q, const uint16_t* kpool, const uint16_t* vpool, float* out,
                                   const int64_t* tables_host, size_t table_entries, const int64_t* lengths_host, int rows,
                                   const float* sinks_dev, const float* alibi_dev, int heads, int kv_heads, int dk, int dv,
                                   int stride, int block_size, int kind, int kgroup, int vgroup, int kscale_base,
                                   int vscale_base, float scale, float softcap, int window, int chunk) {
    using namespace fllm;
    return guarded("attention", [&] {
        if (rows <= 0) return;
        if (dk > 512 || dv > 512 || dk <= 0 || dv <= 0) throw std::runtime_error("head width exceeds 512");
        if (heads <= 0 || kv_heads <= 0 || heads % kv_heads != 0) throw std::runtime_error("heads not a multiple of kv heads");
        if (kind != 1 && kind != 2) throw std::runtime_error("unknown KV cache kind");
        Stream& s = *reinterpret_cast<Stream*>(stp);
        int64_t max_len = 1;
        for (int r = 0; r < rows; r++) max_len = std::max<int64_t>(max_len, lengths_host[r]);
        // enough work-groups to fill the device: 2 per compute unit (an EU runs one work-group of 8 sub-groups at a time)
        const int64_t items = static_cast<int64_t>(rows) * heads;
        const int64_t target = 2 * std::max<int64_t>(1, s.dev->info.compute_units);
        int splits = 1;
        if (items < target) splits = static_cast<int>(std::min<int64_t>(std::min<int64_t>(32, (target + items - 1) / items), (max_len + 63) / 64));
        splits = std::max(splits, 1);
        const int need = (std::max(dk, dv) + ASG - 1) / ASG;
        int ch = 1;
        while (ch < need) ch *= 2;
        const int64_t* tables = static_cast<const int64_t*>(s.stage(tables_host, table_entries * sizeof(int64_t)));
        const int64_t* lengths = static_cast<const int64_t*>(s.stage(lengths_host, static_cast<size_t>(rows) * sizeof(int64_t)));
        const bool ok = kind == 1 ? dispatch_attention<1>(ch, s, q, kpool, vpool, out, tables, lengths, sinks_dev, alibi_dev, rows,
                                                           heads, kv_heads, dk, dv, stride, block_size, kgroup, vgroup,
                                                           kscale_base, vscale_base, scale, softcap, window, chunk, splits)
                                   : dispatch_attention<2>(ch, s, q, kpool, vpool, out, tables, lengths, sinks_dev, alibi_dev, rows,
                                                           heads, kv_heads, dk, dv, stride, block_size, kgroup, vgroup,
                                                           kscale_base, vscale_base, scale, softcap, window, chunk, splits);
        if (!ok) throw std::runtime_error("unsupported head width");
    });
}
