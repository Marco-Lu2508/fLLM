// Runs the real text of act_glu.metal and arch_ops.metal on the CPU and compares every kernel with an independent reference in double
// (see msl_emu/emu.h for what that does and does not prove). Exit code 0 only if every check passes.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "msl_emu/emu.h"

// the kernel sources, transformed to C++ by run_emulation.sh
#include "common_ops.inc"
#include "act_glu.inc"
#include "arch_ops.inc"

static int failures = 0;
static int checks = 0;

static void report(const std::string &name, double worst, double tolerance, size_t at, double got, double want) {
    checks++;
    if (!(worst <= tolerance)) {
        failures++;
        std::printf("[FAIL] %-48s worst error %.3e > %.1e at element %zu (kernel %.9g, reference %.9g)\n", name.c_str(), worst, tolerance, at, got, want);
    } else {
        std::printf("[ ok ] %-48s worst error %.3e (tolerance %.1e)\n", name.c_str(), worst, tolerance);
    }
}

// error of `got` against `want`: absolute for |want| < 1, relative above
static void compare(const std::string &name, const std::vector<float> &got, const std::vector<double> &want, double tolerance) {
    double worst = 0;
    size_t at = 0;
    for (size_t i = 0; i < want.size(); i++) {
        double err = std::fabs(double(got[i]) - want[i]) / std::max(1.0, std::fabs(want[i]));
        if (!(err <= worst)) {  // also catches NaN
            worst = std::isnan(err) ? INFINITY : err;
            at = i;
        }
    }
    report(name, worst, tolerance, at, at < got.size() ? got[at] : 0, at < want.size() ? want[at] : 0);
}

static std::mt19937 rng(12345);
static float uniform(float lo, float hi) { return std::uniform_real_distribution<float>(lo, hi)(rng); }
static std::vector<float> random_vec(size_t n, float lo, float hi) {
    std::vector<float> v(n);
    for (auto &x : v) x = uniform(lo, hi);
    return v;
}

// ---- activations --------------------------------------------------------------------------------------------------------------------------

static double ref_act(int mode, double x) {
    switch (mode) {
        case 0: return x / (1.0 + std::exp(-x));
        case 1: return 0.5 * x * (1.0 + std::tanh(0.7978845608028654 * (x + 0.044715 * x * x * x)));
        case 2: return 0.5 * x * (1.0 + std::erf(x * 0.7071067811865476));
        case 3: return std::max(x, 0.0);
        default: return std::max(x, 0.0) * std::max(x, 0.0);
    }
}

// the planner's grid: groups of `tx` threads along x (rows along y)
template <int MODE>
static void run_act(const std::vector<float> &gate, const std::vector<float> &up, std::vector<float> &out, const o_act_params &p) {
    const uint tx = std::min<uint>((p.width + 31) / 32 * 32, 256);
    const uint groups = (p.width + tx - 1) / tx;
    for (uint r = 0; r < p.n_rows; r++)
        for (uint gx = 0; gx < groups * tx; gx++) o_act_glu<MODE>(gate.data(), up.data(), out.data(), p, uint2{gx, r});
}

static void test_activations() {
    // plain: Stream::activation modes 0..4 -> kernels MODE 3 (silu), 4 (gelu tanh), 7 (gelu erf), 6 (relu), 5 (relu2)
    // gated: Stream::activation_glu modes 0..4 -> MODE 0 (swiglu), 1 (geglu), 8, 9, 10
    for (int shape = 0; shape < 3; shape++) {
        const uint rows = shape == 0 ? 1 : 3;
        const uint width = shape == 2 ? 333 : (shape == 1 ? 64 : 1000);
        const uint stride = width + (shape == 2 ? 7 : 0);  // strided rows exercise gate_stride / out_stride
        o_act_params p{rows, width, stride, 0, stride, 0, stride, 0.0f, 0.0f};
        std::vector<float> gate = random_vec(size_t(rows) * stride, -8.0f, 8.0f);
        // special values in the first row: exact zeros and tiny values
        gate[0] = 0.0f;
        if (width > 3) {
            gate[1] = -0.0f;
            gate[2] = 1e-7f;
            gate[3] = -1e-7f;
        }
        std::vector<float> up = random_vec(size_t(rows) * stride, -2.0f, 2.0f);
        const std::string tag = " [" + std::to_string(rows) + "x" + std::to_string(width) + " stride " + std::to_string(stride) + "]";
        auto check = [&](const std::string &name, int mode, bool gated, auto runner) {
            std::vector<float> out(size_t(rows) * stride, -777.0f);
            runner(out);
            std::vector<double> want(out.size());
            for (size_t i = 0; i < out.size(); i++) {
                const uint c = uint(i % stride);
                if (c >= width) {
                    want[i] = -777.0;  // padding must stay untouched
                    continue;
                }
                want[i] = ref_act(mode, gate[i]) * (gated ? up[i] : 1.0);
            }
            compare(name + tag, out, want, 2e-6);
        };
        check("act silu", 0, false, [&](auto &o) { run_act<3>(gate, up, o, p); });
        check("act gelu_tanh", 1, false, [&](auto &o) { run_act<4>(gate, up, o, p); });
        check("act gelu_erf", 2, false, [&](auto &o) { run_act<7>(gate, up, o, p); });
        check("act relu", 3, false, [&](auto &o) { run_act<6>(gate, up, o, p); });
        check("act relu2", 4, false, [&](auto &o) { run_act<5>(gate, up, o, p); });
        check("glu swiglu", 0, true, [&](auto &o) { run_act<0>(gate, up, o, p); });
        check("glu geglu_tanh", 1, true, [&](auto &o) { run_act<1>(gate, up, o, p); });
        check("glu geglu_erf", 2, true, [&](auto &o) { run_act<8>(gate, up, o, p); });
        check("glu reglu", 3, true, [&](auto &o) { run_act<9>(gate, up, o, p); });
        check("glu relu2glu", 4, true, [&](auto &o) { run_act<10>(gate, up, o, p); });
    }
    // GELU (erf) over a dense sweep, including the tails where 1 + erf cancels
    {
        const uint n = 4001;
        std::vector<float> x(n);
        for (uint i = 0; i < n; i++) x[i] = -20.0f + 40.0f * float(i) / float(n - 1);
        std::vector<float> out(n), dummy(n);
        o_act_params p{1, n, n, 0, n, 0, n, 0.0f, 0.0f};
        run_act<7>(x, dummy, out, p);
        std::vector<double> want(n);
        for (uint i = 0; i < n; i++) want[i] = ref_act(2, x[i]);
        compare("gelu_erf sweep -20..20", out, want, 3e-7);
    }
}

// ---- clamp and softplus ---------------------------------------------------------------------------------------------------------------------

static void test_clamp_softplus() {
    for (uint n : {1u, 255u, 256u, 257u, 5000u}) {
        // the stride loop with fewer threads than elements
        for (uint threads : {n, std::max(1u, n / 3), 7u}) {
            std::vector<float> x = random_vec(n, -10.0f, 10.0f);
            std::vector<double> want(n);
            const float limit = 3.5f;
            for (uint i = 0; i < n; i++) want[i] = std::min(std::max(double(x[i]), -3.5), 3.5);
            arch_clamp_params p{n, limit};
            for (uint t = 0; t < threads; t++) arch_clamp(p, x.data(), t, threads);
            compare("clamp n=" + std::to_string(n) + " threads=" + std::to_string(threads), x, want, 0.0);
        }
    }
    for (float beta : {1.0f, 2.0f, 0.5f}) {
        const uint n = 3000;
        std::vector<float> x = random_vec(n, -3.0f, 3.0f), gate = random_vec(n, -30.0f, 30.0f);
        std::vector<double> want(n);
        for (uint i = 0; i < n; i++) {
            const double s = double(beta) * gate[i];
            want[i] = double(x[i]) * std::log1p(std::exp(s)) / beta;
        }
        arch_softplus_mul_params p{n, beta};
        for (uint t = 0; t < n; t++) arch_softplus_mul(p, x.data(), gate.data(), t, n);
        // the kernel passes the gate through above scaled > 20 (log1p(exp(s)) == s to 2e-9 there) and loses exp(s) below about -16
        compare("softplus_mul beta=" + std::to_string(beta), x, want, 2e-6);
    }
}

// ---- MLA data movement ----------------------------------------------------------------------------------------------------------------------

static void test_mla() {
    struct Case {
        uint rows, heads, nope, rope, value, kv_lora;
    };
    for (Case c : {Case{1, 1, 4, 2, 4, 8}, Case{5, 3, 16, 8, 12, 24}, Case{4, 16, 128, 64, 128, 512}}) {
        const std::string tag = " [rows " + std::to_string(c.rows) + " heads " + std::to_string(c.heads) + " nope " + std::to_string(c.nope) + " rope " +
                                std::to_string(c.rope) + " value " + std::to_string(c.value) + "]";
        const uint dk = c.nope + c.rope, width = c.kv_lora + c.rope;
        // distinct values everywhere so that any index mistake shows
        auto iota = [](size_t n, float base) {
            std::vector<float> v(n);
            for (size_t i = 0; i < n; i++) v[i] = base + float(i);
            return v;
        };
        std::vector<float> kv_a = iota(size_t(c.rows) * width, 0.0f);
        std::vector<float> q_full = iota(size_t(c.rows) * c.heads * dk, 1e6f);
        std::vector<float> kv = iota(size_t(c.rows) * c.heads * (c.nope + c.value), 2e6f);

        // latent
        {
            const uint total = c.rows * c.kv_lora;
            std::vector<float> latent(total, -1.0f);
            arch_mla_latent_params p{total, c.kv_lora, width};
            const uint threads = std::max(1u, total / 2 + 1);
            for (uint t = 0; t < threads; t++) arch_mla_latent(p, kv_a.data(), latent.data(), t, threads);
            std::vector<double> want(total);
            for (uint r = 0; r < c.rows; r++)
                for (uint j = 0; j < c.kv_lora; j++) want[r * c.kv_lora + j] = kv_a[r * width + j];
            compare("mla_latent" + tag, latent, want, 0.0);
        }
        // assemble
        {
            const uint pairs = c.rows * c.heads;
            const uint total_qk = pairs * dk, total_v = pairs * c.value;
            std::vector<float> q(total_qk, -1.0f), k(total_qk, -1.0f), v(total_v, -1.0f);
            arch_mla_assemble_params p{c.rows, c.heads, c.nope, c.rope, c.value, c.kv_lora, total_qk, total_v};
            const uint total = std::max(total_qk, total_v);
            const uint threads = std::max(1u, total / 3 + 1);
            for (uint t = 0; t < threads; t++) arch_mla_assemble(p, q_full.data(), kv_a.data(), kv.data(), q.data(), k.data(), v.data(), t, threads);
            std::vector<double> wq(total_qk), wk(total_qk), wv(total_v);
            for (uint r = 0; r < c.rows; r++)
                for (uint h = 0; h < c.heads; h++) {
                    const uint at = r * c.heads + h;
                    // query: [rotary | content] from the projection's [content | rotary]; key: [shared rotary | content]
                    for (uint d = 0; d < c.rope; d++) {
                        wq[at * dk + d] = q_full[at * dk + c.nope + d];
                        wk[at * dk + d] = kv_a[r * width + c.kv_lora + d];
                    }
                    for (uint d = 0; d < c.nope; d++) {
                        wq[at * dk + c.rope + d] = q_full[at * dk + d];
                        wk[at * dk + c.rope + d] = kv[at * (c.nope + c.value) + d];
                    }
                    for (uint e = 0; e < c.value; e++) wv[at * c.value + e] = kv[at * (c.nope + c.value) + c.nope + e];
                }
            compare("mla_assemble q" + tag, q, wq, 0.0);
            compare("mla_assemble k" + tag, k, wk, 0.0);
            compare("mla_assemble v" + tag, v, wv, 0.0);
        }
    }
}

int main() {
    test_activations();
    test_clamp_softplus();
    test_mla();
    std::printf("%d checks, %d failed\n", checks, failures);
    return failures == 0 ? 0 : 1;
}
