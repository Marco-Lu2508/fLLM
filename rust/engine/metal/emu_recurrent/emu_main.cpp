// CPU run of the REAL kernel text of package 6c (see msl_emu/emu.h and run_emulation.sh). Every test feeds random data to the kernels, calls them like
// the Metal dispatch would (threadgroup grid, threads per threadgroup, bound buffers, constants struct) and compares with straightforward references:
//   * moe_topk, moe_combine, gather / scatter, gate kernels, Mamba, DeltaNet: double-precision re-implementations of the CPU decoder code
//     (rust_decoder.rs `mamba_forward` / `mamba1_forward`, cpu_delta.rs, `moe_route`) written independently of the kernels;
//   * moe_linear: the dequantised weights come from the Rust CPU dequantiser (`fllm_engine::compute::dequantize`, vectors.bin by gen_vectors.rs).
// Not tested here: anything specific to the Metal compiler or GPU (see emu.h).
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "emu.h"  // keep after the C++ headers: it defines `device`, `thread`, ... as macros

#include "common_rec.inc"
#include "delta_conv.inc"
#include "delta_rec.inc"
#include "gate_attention.inc"
#include "mamba_conv.inc"
#include "mamba_norm.inc"
#include "mamba_scan1.inc"
#include "mamba_scan2.inc"
#include "moe_combine.inc"
#include "moe_linear.inc"
#include "moe_route.inc"
#include "moe_rows.inc"

#undef device
#undef constant
#undef thread
#undef threadgroup
#undef kernel

static int g_fail = 0;
static int g_checks = 0;

struct Rng {
    std::mt19937_64 g;
    explicit Rng(uint64_t seed) : g(seed) {}
    float f(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(g); }
    float n(float sd) { return std::normal_distribution<float>(0.0f, sd)(g); }
    uint u(uint n) { return uint(g() % n); }
};

static void report(bool ok, const std::string &name, const std::string &detail = "") {
    g_checks++;
    if (!ok) {
        g_fail++;
        std::printf("[FAIL] %s %s\n", name.c_str(), detail.c_str());
    } else {
        std::printf("[ ok ] %s\n", name.c_str());
    }
}

// compares got with want: |a - b| <= tol * max(1, |b|); the message names the first offender
static void expect_close(const std::string &name, const std::vector<float> &got, const std::vector<double> &want, double tol) {
    if (got.size() != want.size()) {
        report(false, name, "size mismatch");
        return;
    }
    double worst = 0.0;
    size_t at = 0;
    for (size_t i = 0; i < got.size(); i++) {
        const double e = std::fabs(double(got[i]) - want[i]) / std::max(1.0, std::fabs(want[i]));
        if (!(e <= worst)) {  // also catches NaN
            worst = std::isnan(e) ? 1e30 : e;
            at = i;
            if (std::isnan(e)) break;
        }
    }
    char buf[200];
    std::snprintf(buf, sizeof buf, "(worst %.3g at %zu: got %.7g want %.7g, tol %.1g)", worst, at, got.empty() ? 0.0 : double(got[at]), want.empty() ? 0.0 : want[at], tol);
    report(worst <= tol, name, buf);
}

// grid-stride kernels (no barriers): run every thread of every group in a shuffled group order
template <class K>
static void run_strided(Rng &rng, uint groups, uint threads, K &&kernel) {
    std::vector<uint> order(groups);
    std::iota(order.begin(), order.end(), 0u);
    std::shuffle(order.begin(), order.end(), rng.g);
    for (uint g : order)
        for (uint lane = 0; lane < threads; lane++) kernel(g * threads + lane, groups * threads);
}

// kernels with barriers: one emulated threadgroup (real threads) per group, groups in shuffled order
template <class K>
static void run_groups(Rng &rng, std::vector<std::pair<uint, uint>> grid, uint threads, K &&kernel) {
    std::shuffle(grid.begin(), grid.end(), rng.g);
    for (auto [x, y] : grid) emu_threadgroup(threads, [&](uint lane) { kernel(uint2{x, y}, lane); });
}

static std::vector<std::pair<uint, uint>> grid2(uint x, uint y) {
    std::vector<std::pair<uint, uint>> v;
    for (uint j = 0; j < y; j++)
        for (uint i = 0; i < x; i++) v.push_back({i, j});
    return v;
}

static double silu_d(double x) { return x / (1.0 + std::exp(-x)); }
static double softplus_d(double x) { return x > 20.0 ? x : std::log1p(std::exp(x)); }

// ---------------------------------------------------------------------------------------------------------------------------------------
// moe_topk
// ---------------------------------------------------------------------------------------------------------------------------------------
static void test_topk(Rng &rng) {
    struct C {
        uint rows, experts, k;
        bool norm;
        float scale;
        int ties;  // 0: normal random, 1: logits quantised to halves (many ties), 2: all logits equal
    };
    const C cases[] = {{5, 8, 2, true, 1.0f, 0},      {4, 64, 8, true, 2.5f, 0},  {3, 100, 6, false, 1.0f, 0}, {2, 300, 8, true, 1.0f, 1},   {2, 1024, 32, true, 1.0f, 0},
                       {1, 2048, 4, false, 0.5f, 0}, {3, 16, 16, true, 1.0f, 1}, {2, 1, 1, true, 1.0f, 0},    {2, 300, 8, true, 1.0f, 2},   {1, 16, 16, true, 1.0f, 2},
                       {1, 1000, 32, false, 1.0f, 2}};
    for (const C &c : cases) {
        std::vector<float> logits(size_t(c.rows) * c.experts);
        for (float &v : logits) v = c.ties == 2 ? 0.25f : c.ties == 1 ? std::round(rng.f(-3.0f, 3.0f) * 2.0f) / 2.0f : rng.n(2.0f);
        std::vector<int> ids(size_t(c.rows) * c.k, -7);
        std::vector<float> weights(size_t(c.rows) * c.k, -7.0f);
        moe_topk_params p{c.rows, c.experts, c.k, c.norm ? 1u : 0u, c.scale};
        run_groups(rng, grid2(c.rows, 1), 128, [&](uint2 g, uint lane) { moe_topk(p, logits.data(), ids.data(), weights.data(), g.x, lane); });
        bool ids_ok = true;
        std::vector<float> got;
        std::vector<double> want;
        for (uint r = 0; r < c.rows; r++) {
            const float *l = &logits[size_t(r) * c.experts];
            std::vector<uint> order(c.experts);
            std::iota(order.begin(), order.end(), 0u);
            std::stable_sort(order.begin(), order.end(), [&](uint a, uint b) { return l[a] > l[b]; });  // ties: lower index first
            double m = -1e300, sum = 0.0;
            for (uint e = 0; e < c.experts; e++) m = std::max(m, double(l[e]));
            for (uint e = 0; e < c.experts; e++) sum += std::exp(double(l[e]) - m);
            double total = 0.0;
            std::vector<double> sc(c.k);
            for (uint j = 0; j < c.k; j++) {
                sc[j] = std::exp(double(l[order[j]]) - m) / sum;
                total += sc[j];
            }
            const double denom = std::max(total, 1e-8);
            for (uint j = 0; j < c.k; j++) {
                ids_ok &= ids[size_t(r) * c.k + j] == int(order[j]);
                got.push_back(weights[size_t(r) * c.k + j]);
                want.push_back((c.norm ? sc[j] / denom : sc[j]) * c.scale);
            }
        }
        char name[100];
        std::snprintf(name, sizeof name, "moe_topk rows=%u experts=%u k=%u norm=%d ties=%d", c.rows, c.experts, c.k, c.norm, c.ties);
        report(ids_ok, std::string(name) + " ids");
        expect_close(std::string(name) + " weights", got, want, 1e-4);
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// moe_linear against the Rust dequantiser
// ---------------------------------------------------------------------------------------------------------------------------------------
struct Fmt {
    uint kind = 0, qk = 0, bytes = 0, rows = 0, cols = 0, ne = 0;
    std::vector<uint8_t> raw;  // ne experts back to back
    std::vector<float> deq;    // ne x rows x cols
};

static bool load_vectors(const std::string &path, std::vector<Fmt> &out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    auto rd = [&](void *dst, size_t n) { f.read(reinterpret_cast<char *>(dst), std::streamsize(n)); };
    uint32_t count = 0;
    rd(&count, 4);
    for (uint32_t i = 0; i < count && f; i++) {
        Fmt t;
        rd(&t.kind, 4);
        rd(&t.qk, 4);
        rd(&t.bytes, 4);
        rd(&t.rows, 4);
        rd(&t.cols, 4);
        rd(&t.ne, 4);
        t.raw.resize(size_t(t.ne) * t.rows * (t.cols / t.qk) * t.bytes);
        t.deq.resize(size_t(t.ne) * t.rows * t.cols);
        rd(t.raw.data(), t.raw.size());
        rd(t.deq.data(), t.deq.size() * 4);
        out.push_back(std::move(t));
    }
    return bool(f);
}

template <class F>
static void run_moe_linear(Rng &rng, const Fmt &t, const char *label, bool by_address, bool with_bias, uint x_div, uint slots) {
    const size_t expert_bytes = size_t(t.rows) * (t.cols / t.qk) * t.bytes;
    // arena: experts at 256-byte aligned offsets with 0xFF padding (an over-read decodes to NaN / garbage and fails the comparison)
    std::vector<ulong> off(t.ne);
    size_t at = 0;
    for (uint e = 0; e < t.ne; e++) {
        at = (at + 255) / 256 * 256 + 256 * (e % 2);
        off[e] = at;
        at += expert_bytes;
    }
    std::vector<uint8_t> arena(at + 512, 0xFF);
    for (uint e = 0; e < t.ne; e++) std::memcpy(&arena[off[e]], &t.raw[size_t(e) * expert_bytes], expert_bytes);
    std::vector<float> bias(size_t(t.ne) * t.rows);
    for (float &b : bias) b = rng.f(-1.0f, 1.0f);
    std::vector<ulong> boff(t.ne);
    std::vector<uint8_t> bias_arena(size_t(t.ne) * (t.rows * 4 + 8) + 16, 0xFF);
    for (uint e = 0; e < t.ne; e++) {
        boff[e] = size_t(e) * (t.rows * 4 + 8) + 4;
        std::memcpy(&bias_arena[boff[e]], &bias[size_t(e) * t.rows], size_t(t.rows) * 4);
    }
    std::vector<ulong> table(t.ne), bias_table(t.ne);
    for (uint e = 0; e < t.ne; e++) {
        table[e] = by_address ? ulong(reinterpret_cast<uintptr_t>(arena.data() + off[e])) : off[e];
        bias_table[e] = by_address ? ulong(reinterpret_cast<uintptr_t>(bias_arena.data() + boff[e])) : boff[e];
    }
    std::vector<int> ids(slots);
    for (uint s = 0; s < slots; s++) ids[s] = int(rng.u(t.ne));
    if (slots >= 4) {
        ids[1] = -1;
        ids[slots - 1] = int(t.ne);  // exactly one past the last expert
        if (slots >= 6) ids[2] = int(t.ne) + 3;
    }
    const uint xrows = (slots + x_div - 1) / x_div;
    std::vector<float> x(size_t(xrows) * t.cols);
    for (float &v : x) v = rng.n(1.0f);
    std::vector<float> out(size_t(slots) * t.rows, -9.0f);
    moe_linear_params p{t.rows, t.cols, slots, x_div, t.ne, (with_bias ? 1u : 0u) | (by_address ? 2u : 0u), t.cols / t.qk};
    run_groups(rng, grid2(t.rows, slots), 64, [&](uint2 g, uint lane) {
        moe_linear_t<F>(p, table.data(), bias_table.data(), ids.data(), x.data(), out.data(), arena.data(), bias_arena.data(), g, lane);
    });
    double worst = 0.0;
    size_t where = 0;
    for (uint s = 0; s < slots; s++) {
        for (uint r = 0; r < t.rows; r++) {
            double want = 0.0, mag = 1e-3;
            if (ids[s] >= 0 && uint(ids[s]) < t.ne) {
                const float *w = &t.deq[(size_t(ids[s]) * t.rows + r) * t.cols];
                const float *xs = &x[size_t(s / x_div) * t.cols];
                for (uint c = 0; c < t.cols; c++) {
                    want += double(w[c]) * xs[c];
                    mag += std::fabs(double(w[c]) * xs[c]);
                }
                if (with_bias) {
                    want += bias[size_t(ids[s]) * t.rows + r];
                    mag += 1.0;
                }
            }
            const double e = std::fabs(double(out[size_t(s) * t.rows + r]) - want) / mag;
            if (!(e <= worst)) {
                worst = std::isnan(e) ? 1e30 : e;
                where = size_t(s) * t.rows + r;
            }
        }
    }
    char name[160], detail[120];
    std::snprintf(name, sizeof name, "moe_linear_%s %s bias=%d x_div=%u slots=%u rows=%u cols=%u", label, by_address ? "address" : "arena", with_bias, x_div, slots, t.rows, t.cols);
    std::snprintf(detail, sizeof detail, "(worst rel %.3g at %zu)", worst, where);
    report(worst <= 3e-4, name, detail);
}

static void test_moe_linear(Rng &rng, const std::vector<Fmt> &fmts) {
    for (const Fmt &t : fmts) {
#define CASE(KIND, F, LABEL)                                                 \
    case KIND:                                                               \
        if (F::QK != t.qk || F::BYTES != t.bytes) {                          \
            report(false, std::string("block size of ") + LABEL);            \
            break;                                                           \
        }                                                                    \
        for (bool addr : {false, true}) {                                    \
            run_moe_linear<F>(rng, t, LABEL, addr, false, 1, 5);             \
            run_moe_linear<F>(rng, t, LABEL, addr, true, 2, 6);              \
        }                                                                    \
        break;
        switch (t.kind) {
            CASE(2, r_fmt_q4_0, "q4_0")
            CASE(3, r_fmt_q4_1, "q4_1")
            CASE(6, r_fmt_q5_0, "q5_0")
            CASE(7, r_fmt_q5_1, "q5_1")
            CASE(8, r_fmt_q8_0, "q8_0")
            CASE(20, r_fmt_iq4_nl, "iq4_nl")
            CASE(39, r_fmt_mxfp4, "mxfp4")
            CASE(12, r_fmt_q4_k, "q4_k")
            CASE(13, r_fmt_q5_k, "q5_k")
            CASE(14, r_fmt_q6_k, "q6_k")
            default:
                report(false, "unknown format kind " + std::to_string(t.kind));
        }
#undef CASE
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// moe_combine, gather, scatter, gates
// ---------------------------------------------------------------------------------------------------------------------------------------
static void test_rows_and_gates(Rng &rng) {
    {
        const uint tokens = 7, k = 4, hidden = 37;
        std::vector<float> y(size_t(tokens) * k * hidden), w(size_t(tokens) * k), out(size_t(tokens) * hidden, -1.0f);
        for (float &v : y) v = rng.n(1.0f);
        for (float &v : w) v = rng.f(0.0f, 1.0f);
        moe_combine_params p{tokens, k, hidden};
        run_strided(rng, 3, 256, [&](uint gid, uint grid) { moe_combine(p, out.data(), y.data(), w.data(), gid, grid); });
        std::vector<double> want(out.size());
        for (uint t = 0; t < tokens; t++)
            for (uint h = 0; h < hidden; h++) {
                double a = 0;
                for (uint j = 0; j < k; j++) a += double(w[t * k + j]) * y[(t * k + j) * hidden + h];
                want[t * hidden + h] = a;
            }
        expect_close("moe_combine (grid smaller than the work)", out, want, 1e-5);
    }
    for (bool weights : {false, true}) {
        const uint src_rows = 9, n = 6, row_len = 21;
        std::vector<float> src(size_t(src_rows) * row_len), w(n);
        for (float &v : src) v = rng.n(1.0f);
        for (float &v : w) v = rng.f(0.5f, 2.0f);
        std::vector<int> idx = {5, 0, 8, 3, 1, 7};
        moe_rows_params p{n, row_len, weights ? 1u : 0u};
        std::vector<float> dst(size_t(n) * row_len, -1.0f);
        run_strided(rng, 2, 256, [&](uint gid, uint grid) { moe_gather_rows(p, dst.data(), src.data(), idx.data(), w.data(), gid, grid); });
        std::vector<double> want(dst.size());
        for (uint i = 0; i < n; i++)
            for (uint c = 0; c < row_len; c++) want[i * row_len + c] = double(src[idx[i] * row_len + c]) * (weights ? w[i] : 1.0f);
        expect_close(std::string("moe_gather_rows weights=") + (weights ? "1" : "0"), dst, want, 1e-6);
        // scatter: dst[idx[i]] += src[i] * w[i] on top of existing content
        std::vector<float> acc(size_t(src_rows) * row_len);
        for (float &v : acc) v = rng.n(1.0f);
        std::vector<double> want_acc(acc.begin(), acc.end());
        for (uint i = 0; i < n; i++)
            for (uint c = 0; c < row_len; c++) want_acc[idx[i] * row_len + c] += double(dst[i * row_len + c]) * (weights ? w[i] : 1.0f);
        run_strided(rng, 2, 256, [&](uint gid, uint grid) { moe_scatter_add_rows(p, acc.data(), dst.data(), idx.data(), w.data(), gid, grid); });
        expect_close(std::string("moe_scatter_add_rows weights=") + (weights ? "1" : "0"), acc, want_acc, 1e-6);
    }
    for (bool per_head : {false, true}) {
        const uint rows = 3, heads = 5, dk = 11;
        std::vector<float> proj(size_t(rows) * 2 * heads * dk);
        for (float &v : proj) v = rng.n(1.0f);
        std::vector<float> q(size_t(rows) * heads * dk, -1.0f), gate(q.size(), -1.0f);
        gate_split_q_params p{rows * heads * dk, heads, dk, per_head ? 1u : 0u};
        run_strided(rng, 2, 256, [&](uint gid, uint grid) { gate_split_q(p, proj.data(), q.data(), gate.data(), gid, grid); });
        std::vector<double> wq(q.size()), wg(q.size());
        for (uint r = 0; r < rows; r++)
            for (uint h = 0; h < heads; h++)
                for (uint d = 0; d < dk; d++) {
                    const size_t row = size_t(r) * 2 * heads * dk;
                    const size_t qi = per_head ? row + size_t(h) * 2 * dk + d : row + size_t(h) * dk + d;
                    const size_t gi = per_head ? qi + dk : row + size_t(heads) * dk + size_t(h) * dk + d;
                    wq[(size_t(r) * heads + h) * dk + d] = proj[qi];
                    wg[(size_t(r) * heads + h) * dk + d] = proj[gi];
                }
        expect_close(std::string("gate_split_q q per_head=") + (per_head ? "1" : "0"), q, wq, 0.0);
        expect_close(std::string("gate_split_q gate per_head=") + (per_head ? "1" : "0"), gate, wg, 0.0);
    }
    {
        const uint count = 1000;
        std::vector<float> x(count), g(count);
        for (float &v : x) v = rng.n(1.0f);
        for (float &v : g) v = rng.f(-30.0f, 30.0f);
        std::vector<double> want(count);
        for (uint i = 0; i < count; i++) want[i] = double(x[i]) / (1.0 + std::exp(-double(g[i])));
        gate_sigmoid_mul_params p{count};
        run_strided(rng, 3, 256, [&](uint gid, uint grid) { gate_sigmoid_mul(p, x.data(), g.data(), gid, grid); });
        expect_close("gate_sigmoid_mul", x, want, 1e-5);
    }
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// Gated DeltaNet
// ---------------------------------------------------------------------------------------------------------------------------------------
struct DeltaCfg {
    uint kh, vh, hk, hv, ksz;
    bool tiled;
    uint n;
};

// token-at-a-time double reference of the whole sequence (cpu_delta.rs semantics; state starts at zero)
static std::vector<double> delta_ref(const DeltaCfg &c, const std::vector<float> &qkv, const std::vector<float> &z, const std::vector<float> &beta, const std::vector<float> &alpha,
                                     const std::vector<float> &convw, const std::vector<float> &dt_bias, const std::vector<float> &decay, const std::vector<float> &norm,
                                     float beta_scale, float eps) {
    const uint key_dim = c.kh * c.hk, value_dim = c.vh * c.hv, conv_dim = 2 * key_dim + value_dim;
    std::vector<double> out(size_t(c.n) * value_dim);
    std::vector<double> S(size_t(c.vh) * c.hk * c.hv, 0.0);
    for (uint t = 0; t < c.n; t++) {
        std::vector<double> mixed(conv_dim);
        for (uint ch = 0; ch < conv_dim; ch++) {
            double acc = 0.0;
            for (uint k = 0; k < c.ksz; k++) {
                const int src = int(t) - int(c.ksz - 1) + int(k);
                if (src >= 0) acc += double(convw[size_t(ch) * c.ksz + k]) * qkv[size_t(src) * conv_dim + ch];
            }
            mixed[ch] = silu_d(acc);
        }
        std::vector<double> qn(key_dim), kn(key_dim);
        for (uint h = 0; h < c.kh; h++) {
            double sq = 0, sk = 0;
            for (uint i = 0; i < c.hk; i++) {
                sq += mixed[h * c.hk + i] * mixed[h * c.hk + i];
                sk += mixed[key_dim + h * c.hk + i] * mixed[key_dim + h * c.hk + i];
            }
            const double nq = 1.0 / std::sqrt(sq + 1e-6), nk = 1.0 / std::sqrt(sk + 1e-6), qs = std::pow(double(c.hk), -0.5);
            for (uint i = 0; i < c.hk; i++) {
                qn[h * c.hk + i] = mixed[h * c.hk + i] * nq * qs;
                kn[h * c.hk + i] = mixed[key_dim + h * c.hk + i] * nk;
            }
        }
        for (uint j = 0; j < c.vh; j++) {
            const uint src = c.tiled ? j % c.kh : j / (c.vh / c.kh);
            const double forget = std::exp(double(decay[j]) * softplus_d(double(alpha[size_t(t) * c.vh + j]) + dt_bias[j]));
            const double sigma = beta_scale / (1.0 + std::exp(-double(beta[size_t(t) * c.vh + j])));
            std::vector<double> o(c.hv, 0.0);
            for (uint col = 0; col < c.hv; col++) {
                double memory = 0.0;
                for (uint i = 0; i < c.hk; i++) {
                    double &s = S[(size_t(j) * c.hk + i) * c.hv + col];
                    s *= forget;
                    memory += s * kn[src * c.hk + i];
                }
                const double delta = (mixed[2 * key_dim + j * c.hv + col] - memory) * sigma;
                for (uint i = 0; i < c.hk; i++) {
                    double &s = S[(size_t(j) * c.hk + i) * c.hv + col];
                    s += kn[src * c.hk + i] * delta;
                    o[col] += s * qn[src * c.hk + i];
                }
            }
            double ss = 0;
            for (double v : o) ss += v * v;
            const double inv = 1.0 / std::sqrt(ss / c.hv + eps);
            for (uint col = 0; col < c.hv; col++) out[size_t(t) * value_dim + j * c.hv + col] = o[col] * inv * norm[col] * silu_d(z[size_t(t) * value_dim + j * c.hv + col]);
        }
    }
    return out;
}

template <uint HK>
static void delta_run(Rng &rng, const DeltaCfg &c, uint row0, uint n, bool reset, uint state_floats_off, std::vector<float> &state, const std::vector<float> &qkv,
                      const std::vector<float> &z, const std::vector<float> &beta, const std::vector<float> &alpha, const std::vector<float> &convw,
                      const std::vector<float> &dt_bias, const std::vector<float> &decay, const std::vector<float> &norm, float beta_scale, float eps, std::vector<float> &mixed,
                      std::vector<float> &out) {
    const uint conv_dim = 2 * c.kh * c.hk + c.vh * c.hv;
    const uint rec_floats = c.vh * c.hk * c.hv;
    delta_conv_params cp{conv_dim, c.kh, c.hk, c.ksz, row0, n, reset ? 1u : 0u, state_floats_off + rec_floats, float(std::pow(double(c.hk), -0.5))};
    const uint value_blocks = (c.vh * c.hv + 127) / 128;
    run_groups(rng, grid2(2 * c.kh + value_blocks, (n + 15) / 16), 128,
               [&](uint2 g, uint lane) { delta_conv(cp, qkv.data(), convw.data(), state.data(), mixed.data(), g, lane); });
    delta_rec_params rp{c.kh, c.vh, c.hv, conv_dim, c.tiled ? 1u : 0u, row0, n, reset ? 1u : 0u, state_floats_off, beta_scale, eps};
    run_groups(rng, grid2(c.vh, 1), 256, [&](uint2 g, uint lane) {
        delta_rec_t<HK>(rp, mixed.data(), z.data(), beta.data(), alpha.data(), dt_bias.data(), decay.data(), norm.data(), state.data(), out.data(), g.x, lane);
    });
}

template <uint HK>
static void test_delta_one(Rng &rng, DeltaCfg c) {
    c.hk = HK;
    const uint key_dim = c.kh * c.hk, value_dim = c.vh * c.hv, conv_dim = 2 * key_dim + value_dim;
    std::vector<float> qkv(size_t(c.n) * conv_dim), z(size_t(c.n) * value_dim), beta(size_t(c.n) * c.vh), alpha(size_t(c.n) * c.vh);
    std::vector<float> convw(size_t(conv_dim) * c.ksz), dt_bias(c.vh), decay(c.vh), norm(c.hv);
    for (float &v : qkv) v = rng.n(1.0f);
    for (float &v : z) v = rng.n(1.0f);
    for (float &v : beta) v = rng.n(1.0f);
    for (float &v : alpha) v = rng.n(1.0f);
    for (float &v : convw) v = rng.f(-0.6f, 0.6f);
    for (float &v : dt_bias) v = rng.f(-1.0f, 1.0f);
    for (float &v : decay) v = -std::exp(rng.f(-1.0f, 1.0f));
    for (float &v : norm) v = rng.f(0.5f, 1.5f);
    const float beta_scale = 0.9f, eps = 1e-6f;
    const std::vector<double> want = delta_ref(c, qkv, z, beta, alpha, convw, dt_bias, decay, norm, beta_scale, eps);
    const uint layer_off = 24;  // floats before the layer's state in the state buffer (state_offset / 4)
    const size_t state_total = layer_off + size_t(c.vh) * c.hk * c.hv + size_t(conv_dim) * (c.ksz - 1) + 8;
    // splits of the sequence into runs: one run; two runs; one token at a time (small cases)
    std::vector<std::vector<uint>> plans = {{c.n}};
    if (c.n >= 2) plans.push_back({1, c.n - 1});
    if (c.n >= 5) plans.push_back({2, 1, c.n - 3});
    if (c.n >= 20) plans.push_back({17, c.n - 17});
    if (c.n <= 8) plans.push_back(std::vector<uint>(c.n, 1));
    for (const auto &runs : plans) {
        // the NaN fill proves that `reset` ignores the old state
        std::vector<float> state(state_total, std::nanf(""));
        std::vector<float> mixed(size_t(c.n) * conv_dim, -5.0f), out(size_t(c.n) * value_dim, -5.0f);
        uint row = 0;
        bool first = true;
        for (uint len : runs) {
            delta_run<HK>(rng, c, row, len, first, layer_off, state, qkv, z, beta, alpha, convw, dt_bias, decay, norm, beta_scale, eps, mixed, out);
            row += len;
            first = false;
        }
        std::string split;
        for (uint r : runs) split += std::to_string(r) + ",";
        char name[200];
        std::snprintf(name, sizeof name, "delta hk=%u hv=%u kh=%u vh=%u ksz=%u tiled=%d n=%u runs=%s", c.hk, c.hv, c.kh, c.vh, c.ksz, c.tiled, c.n, split.size() > 30 ? "(many)" : split.c_str());
        expect_close(name, out, want, 2e-4);
        // the words around the layer's state are untouched
        bool guard = true;
        for (uint i = 0; i < layer_off; i++) guard &= std::isnan(state[i]);
        report(guard, std::string(name) + " (state offset respected)");
    }
}

static void test_delta(Rng &rng) {
    test_delta_one<32>(rng, {2, 4, 0, 24, 4, false, 7});
    test_delta_one<32>(rng, {2, 4, 0, 5, 2, true, 20});
    test_delta_one<64>(rng, {1, 2, 0, 33, 3, false, 40});
    test_delta_one<96>(rng, {2, 2, 0, 16, 8, false, 9});
    test_delta_one<128>(rng, {1, 2, 0, 128, 4, false, 6});
    test_delta_one<128>(rng, {2, 4, 0, 20, 5, true, 3});
}

// ---------------------------------------------------------------------------------------------------------------------------------------
// Mamba
// ---------------------------------------------------------------------------------------------------------------------------------------
struct M2Cfg {
    uint inner, ns, heads, groups, ksz, norm_groups, n;
    float step_min, eps;
};

// rust_decoder.rs `mamba_forward`, a whole sequence from a zero state, per token
static std::vector<double> mamba2_ref(const M2Cfg &c, const std::vector<float> &projected, const std::vector<float> &convw, const std::vector<float> &convb, const std::vector<float> &dt_bias,
                                      const std::vector<float> &decay, const std::vector<float> &skip, const std::vector<float> &norm) {
    const uint conv_dim = c.inner + 2 * c.groups * c.ns, width = c.inner + conv_dim + c.heads, head_dim = c.inner / c.heads, per_group = c.heads / c.groups;
    std::vector<double> out(size_t(c.n) * c.inner);
    std::vector<double> rec(size_t(c.heads) * head_dim * c.ns, 0.0);
    for (uint t = 0; t < c.n; t++) {
        const float *row = &projected[size_t(t) * width];
        const float *gate = row;
        std::vector<double> mixed(conv_dim);
        for (uint ch = 0; ch < conv_dim; ch++) {
            double acc = convb[ch];
            for (uint k = 0; k < c.ksz; k++) {
                const int src = int(t) - int(c.ksz - 1) + int(k);
                if (src >= 0) acc += double(convw[size_t(ch) * c.ksz + k]) * projected[size_t(src) * width + c.inner + ch];
            }
            mixed[ch] = silu_d(acc);
        }
        const float *dt_raw = row + c.inner + conv_dim;
        std::vector<double> y(c.inner);
        for (uint h = 0; h < c.heads; h++) {
            const double dt = std::max(softplus_d(double(dt_raw[h]) + dt_bias[h]), double(c.step_min));
            const double da = std::exp(dt * decay[h]);
            const uint g = h / std::max(per_group, 1u);
            const double *bvec = &mixed[c.inner + g * c.ns];
            const double *cvec = &mixed[c.inner + c.groups * c.ns + g * c.ns];
            for (uint p = 0; p < head_dim; p++) {
                const double xp = mixed[h * head_dim + p];
                double acc = 0.0;
                for (uint k = 0; k < c.ns; k++) {
                    double &cell = rec[(size_t(h) * head_dim + p) * c.ns + k];
                    cell = cell * da + dt * xp * bvec[k];
                    acc += cell * cvec[k];
                }
                y[h * head_dim + p] = acc + double(skip[h]) * xp;
            }
        }
        for (uint i = 0; i < c.inner; i++) y[i] *= silu_d(gate[i]);
        const uint size = c.inner / c.norm_groups;
        for (uint g = 0; g < c.norm_groups; g++) {
            double ss = 0;
            for (uint i = 0; i < size; i++) ss += y[g * size + i] * y[g * size + i];
            const double inv = 1.0 / std::sqrt(ss / size + c.eps);
            for (uint i = 0; i < size; i++) out[size_t(t) * c.inner + g * size + i] = y[g * size + i] * inv * norm[g * size + i];
        }
    }
    return out;
}

template <uint NS>
static void test_mamba2_one(Rng &rng, M2Cfg c) {
    const uint conv_dim = c.inner + 2 * c.groups * c.ns, width = c.inner + conv_dim + c.heads, head_dim = c.inner / c.heads;
    std::vector<float> projected(size_t(c.n) * width), convw(size_t(conv_dim) * c.ksz), convb(conv_dim), dt_bias(c.heads), decay(c.heads), skip(c.heads), norm(c.inner);
    for (float &v : projected) v = rng.n(1.0f);
    for (float &v : convw) v = rng.f(-0.6f, 0.6f);
    for (float &v : convb) v = rng.f(-0.3f, 0.3f);
    for (float &v : dt_bias) v = rng.f(-1.0f, 1.0f);
    for (float &v : decay) v = -std::exp(rng.f(-1.0f, 1.0f));
    for (float &v : skip) v = rng.f(-1.0f, 1.0f);
    for (float &v : norm) v = rng.f(0.5f, 1.5f);
    const std::vector<double> want = mamba2_ref(c, projected, convw, convb, dt_bias, decay, skip, norm);
    const uint layer_off = 8;
    const uint hist_floats = conv_dim * (c.ksz - 1), rec_floats = c.heads * head_dim * c.ns;
    std::vector<std::vector<uint>> plans = {{c.n}};
    if (c.n >= 2) plans.push_back({1, c.n - 1});
    if (c.n >= 20) plans.push_back({17, c.n - 17});
    if (c.n <= 8) plans.push_back(std::vector<uint>(c.n, 1));
    for (const auto &runs : plans) {
        std::vector<float> state(layer_off + hist_floats + rec_floats + 8, std::nanf(""));
        std::vector<float> mixed(size_t(c.n) * conv_dim, -5.0f), out(size_t(c.n) * c.inner, -5.0f);
        uint row = 0;
        bool first = true;
        for (uint len : runs) {
            mamba_conv_params cp{conv_dim, c.ksz, width, c.inner, row, len, first ? 1u : 0u, layer_off, 1u};
            run_groups(rng, grid2((conv_dim + 63) / 64, (len + 15) / 16), 64, [&](uint2 g, uint lane) { mamba_conv(cp, projected.data(), convw.data(), convb.data(), state.data(), mixed.data(), g, lane); });
            mamba_scan2_params sp{c.inner, c.ns, c.heads, c.groups, head_dim, conv_dim, width, c.inner + conv_dim, row, len, first ? 1u : 0u, layer_off + hist_floats, c.step_min};
            run_groups(rng, grid2(c.heads, (head_dim + 63) / 64), 64, [&](uint2 g, uint lane) {
                mamba_scan2_t<NS>(sp, mixed.data(), projected.data(), dt_bias.data(), decay.data(), skip.data(), state.data(), out.data(), g, lane);
            });
            row += len;
            first = false;
        }
        // gated group norm for all rows (in place on out)
        mamba_gate_norm_params np{c.inner, c.norm_groups, c.n, width, 0, c.eps};
        run_groups(rng, grid2(c.norm_groups, c.n), 256, [&](uint2 g, uint lane) { mamba_gate_norm(np, out.data(), projected.data(), norm.data(), g, lane); });
        char name[200];
        std::snprintf(name, sizeof name, "mamba2 kernel_ns=%u ns=%u inner=%u heads=%u groups=%u ksz=%u n=%u runs=%zu", NS, c.ns, c.inner, c.heads, c.groups, c.ksz, c.n, runs.size());
        expect_close(name, out, want, 3e-4);
    }
}

struct M1Cfg {
    uint inner, ns, rank, ksz, n;
    float bc_eps;  // 0: no norm
};

// rust_decoder.rs `mamba1_forward` with random split / dt inputs (x_proj and dt_proj are matrix products outside the kernels)
static std::vector<double> mamba1_ref(const M1Cfg &c, const std::vector<float> &projected, const std::vector<float> &convw, const std::vector<float> &convb, const std::vector<float> &split,
                                      const std::vector<float> &dt, const std::vector<float> &decay, const std::vector<float> &skip) {
    std::vector<double> out(size_t(c.n) * c.inner);
    std::vector<double> rec(size_t(c.inner) * c.ns, 0.0);
    const uint sw = c.rank + 2 * c.ns;
    for (uint t = 0; t < c.n; t++) {
        const float *row = &projected[size_t(t) * 2 * c.inner];
        std::vector<double> mixed(c.inner);
        for (uint ch = 0; ch < c.inner; ch++) {
            double acc = convb[ch];
            for (uint k = 0; k < c.ksz; k++) {
                const int src = int(t) - int(c.ksz - 1) + int(k);
                if (src >= 0) acc += double(convw[size_t(ch) * c.ksz + k]) * projected[size_t(src) * 2 * c.inner + ch];
            }
            mixed[ch] = silu_d(acc);
        }
        std::vector<double> sp(split.begin() + size_t(t) * sw, split.begin() + size_t(t + 1) * sw);
        if (c.bc_eps > 0.0f) {
            const std::pair<uint, uint> parts[3] = {{0, c.rank}, {c.rank, c.ns}, {c.rank + c.ns, c.ns}};
            for (auto [off, len] : parts) {
                double ss = 0;
                for (uint i = 0; i < len; i++) ss += sp[off + i] * sp[off + i];
                const double inv = 1.0 / std::sqrt(ss / len + c.bc_eps);
                for (uint i = 0; i < len; i++) sp[off + i] *= inv;
            }
        }
        for (uint ch = 0; ch < c.inner; ch++) {
            const double step = softplus_d(dt[size_t(t) * c.inner + ch]);
            double acc = 0;
            for (uint k = 0; k < c.ns; k++) {
                double &s = rec[size_t(ch) * c.ns + k];
                s = s * std::exp(step * decay[size_t(ch) * c.ns + k]) + step * sp[c.rank + k] * mixed[ch];
                acc += s * sp[c.rank + c.ns + k];
            }
            out[size_t(t) * c.inner + ch] = (acc + double(skip[ch]) * mixed[ch]) * silu_d(row[c.inner + ch]);
        }
    }
    return out;
}

template <uint NS>
static void test_mamba1_one(Rng &rng, M1Cfg c) {
    const uint sw = c.rank + 2 * c.ns;
    std::vector<float> projected(size_t(c.n) * 2 * c.inner), convw(size_t(c.inner) * c.ksz), convb(c.inner), split(size_t(c.n) * sw), dt(size_t(c.n) * c.inner), decay(size_t(c.inner) * c.ns),
        skip(c.inner);
    for (float &v : projected) v = rng.n(1.0f);
    for (float &v : convw) v = rng.f(-0.6f, 0.6f);
    for (float &v : convb) v = rng.f(-0.3f, 0.3f);
    for (float &v : split) v = rng.n(1.0f);
    for (float &v : dt) v = rng.n(1.0f);
    for (float &v : decay) v = -std::exp(rng.f(-1.0f, 1.0f));
    for (float &v : skip) v = rng.f(-1.0f, 1.0f);
    const std::vector<double> want = mamba1_ref(c, projected, convw, convb, split, dt, decay, skip);
    const uint layer_off = 12, hist_floats = c.inner * (c.ksz - 1), rec_floats = c.inner * c.ns;
    std::vector<std::vector<uint>> plans = {{c.n}};
    if (c.n >= 2) plans.push_back({1, c.n - 1});
    if (c.n >= 20) plans.push_back({17, c.n - 17});
    for (const auto &runs : plans) {
        std::vector<float> state(layer_off + hist_floats + rec_floats + 8, std::nanf(""));
        std::vector<float> mixed(size_t(c.n) * c.inner, -5.0f), out(size_t(c.n) * c.inner, -5.0f);
        std::vector<float> sp = split;
        if (c.bc_eps > 0.0f) {
            mamba_bc_norm_params bp{c.n, c.rank, c.ns, sw, c.bc_eps};
            run_groups(rng, grid2(3, c.n), 128, [&](uint2 g, uint lane) { mamba_bc_norm(bp, sp.data(), g, lane); });
        }
        uint row = 0;
        bool first = true;
        for (uint len : runs) {
            mamba_conv_params cp{c.inner, c.ksz, 2 * c.inner, 0, row, len, first ? 1u : 0u, layer_off, 1u};
            run_groups(rng, grid2((c.inner + 63) / 64, (len + 15) / 16), 64, [&](uint2 g, uint lane) { mamba_conv(cp, projected.data(), convw.data(), convb.data(), state.data(), mixed.data(), g, lane); });
            mamba_scan1_params sc{c.inner, c.ns, c.rank, sw, 2 * c.inner, c.inner, row, len, first ? 1u : 0u, layer_off + hist_floats};
            run_groups(rng, grid2((c.inner + 63) / 64, 1), 64, [&](uint2 g, uint lane) {
                mamba_scan1_t<NS>(sc, mixed.data(), dt.data(), sp.data(), projected.data(), decay.data(), skip.data(), state.data(), out.data(), g.x, lane);
            });
            row += len;
            first = false;
        }
        char name[200];
        std::snprintf(name, sizeof name, "mamba1 kernel_ns=%u ns=%u inner=%u rank=%u ksz=%u n=%u bc_norm=%d runs=%zu", NS, c.ns, c.inner, c.rank, c.ksz, c.n, c.bc_eps > 0.0f, runs.size());
        expect_close(name, out, want, 3e-4);
    }
}

static void test_mamba(Rng &rng) {
    test_mamba2_one<16>(rng, {64, 16, 4, 2, 4, 2, 9, 0.0f, 1e-5f});
    test_mamba2_one<0>(rng, {96, 24, 6, 3, 3, 3, 21, 0.01f, 1e-5f});
    test_mamba2_one<32>(rng, {80, 32, 2, 1, 5, 2, 6, 0.0f, 1e-5f});
    test_mamba2_one<64>(rng, {192, 64, 3, 3, 2, 1, 4, 0.0f, 1e-5f});
    test_mamba2_one<0>(rng, {70, 130, 1, 1, 7, 1, 3, 0.0f, 1e-5f});
    test_mamba1_one<8>(rng, {100, 8, 6, 4, 11, 1e-5f});
    test_mamba1_one<16>(rng, {64, 16, 4, 3, 21, 0.0f});
    test_mamba1_one<0>(rng, {130, 20, 5, 4, 5, 1e-5f});
    test_mamba1_one<32>(rng, {70, 32, 8, 2, 3, 0.0f});
}

int main(int argc, char **argv) {
    const std::string vectors = argc > 1 ? argv[1] : "";
    Rng rng(argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 20260101ull);
    std::printf("== moe_topk\n");
    test_topk(rng);
    std::printf("== rows and gates\n");
    test_rows_and_gates(rng);
    std::printf("== delta\n");
    test_delta(rng);
    std::printf("== mamba\n");
    test_mamba(rng);
    std::printf("== moe_linear (vectors: %s)\n", vectors.empty() ? "none" : vectors.c_str());
    std::vector<Fmt> fmts;
    if (vectors.empty() || !load_vectors(vectors, fmts)) {
        report(false, "moe_linear vectors could not be loaded");
    } else {
        test_moe_linear(rng, fmts);
    }
    std::printf("\n%d checks, %d failed\n", g_checks, g_fail);
    return g_fail == 0 ? 0 : 1;
}
