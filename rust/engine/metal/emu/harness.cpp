// CPU emulation of the quantized linear kernels of package p06a (see README.md). Not a GPU test.
//
// The MSL text (unit.cpp = common.metal, common_q_gemv.metal, common_q_gemm.metal, q_prepass.metal after a mechanical sed) is compiled
// as C++ together with this driver. For every format the driver reads a dump of the Rust side (device image + reference
// dequantization, see weights/tests.rs `dump_vectors_for_the_msl_emulation`), runs the real kernel templates on emulated
// threadgroups (emu_runtime.h) and compares the outputs with an f64 reference. The weight image sits right in front of a PROT_NONE page
// so that an out-of-bounds read is a crash, and the unit is built with -fsanitize=alignment (trap), which checks the alignment
// promises of common.metal.
#include "emu_runtime.h"

#include <sys/mman.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "unit.cpp"

namespace {

struct Guarded {
    uint8_t *data = nullptr;
    size_t size = 0;
    explicit Guarded(size_t n) : size(n) {
        const size_t page = 4096;
        const size_t body = (n + page - 1) / page * page;
        uint8_t *base = static_cast<uint8_t *>(mmap(nullptr, body + page, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0));
        if (base == MAP_FAILED) { perror("mmap"); exit(2); }
        mprotect(base + body, page, PROT_NONE);
        data = base + body - ((n + 15) & ~size_t(15));  // ends at the guard page (up to 15 bytes of slack behind n so that the start is 16-byte aligned)
    }
};

struct Case {
    uint32_t rows, cols, qk, nbb, p1lo, p1hi, devlen, perm, type;
    std::vector<uint8_t> dev;
    std::vector<float> ref;
    std::vector<uint32_t> permutation;
};

bool load(const std::string &path, Case &c) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    uint32_t h[10];
    if (fread(h, 4, 10, f) != 10 || h[0] != 0x4d544c51u) { fprintf(stderr, "bad header %s\n", path.c_str()); exit(2); }
    c.rows = h[1]; c.cols = h[2]; c.qk = h[3]; c.nbb = h[4]; c.p1lo = h[5]; c.p1hi = h[6]; c.devlen = h[7]; c.perm = h[8]; c.type = h[9];
    c.dev.resize(c.devlen);
    c.ref.resize((size_t)c.rows * c.cols);
    bool ok = fread(c.dev.data(), 1, c.devlen, f) == c.devlen && fread(c.ref.data(), 4, c.ref.size(), f) == c.ref.size();
    if (c.perm) {
        c.permutation.resize(c.cols);
        ok = ok && fread(c.permutation.data(), 4, c.cols, f) == c.cols;
    }
    fclose(f);
    if (!ok) { fprintf(stderr, "short file %s\n", path.c_str()); exit(2); }
    return true;
}

uint64_t lcg_state = 12345;
float unit() {
    lcg_state = lcg_state * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<float>((lcg_state >> 33) % 20001) / 10000.0f - 1.0f;
}

float *aligned_floats(size_t n) {
    void *p = nullptr;
    if (posix_memalign(&p, 64, n * 4 + 64) != 0) exit(2);
    memset(p, 0, n * 4 + 64);
    return static_cast<float *>(p);
}

unsigned ceil_div(unsigned a, unsigned b) { return (a + b - 1) / b; }

enum Kind { MV_R4, MV_R2, MM_H, MM_F };

struct Totals {
    int checks = 0, failures = 0;
};

template <class F>
void dispatch(Kind kind, const Case &c, const uint8_t *w, const float *x, float *out, const float *bias, const q_params &p, unsigned nb) {
    if (kind == MV_R4 || kind == MV_R2) {
        const unsigned nr = kind == MV_R4 ? 4 : 2;
        q_fc_nb = nb;
        const unsigned gx = ceil_div(p.rows, 2 * nr), gy = ceil_div(p.batch, nb);
        for (unsigned y = 0; y < gy; ++y)
            for (unsigned xg = 0; xg < gx; ++xg)
                emu::run_group(64, 32, [&](unsigned, unsigned sg, unsigned lane) {
                    if (kind == MV_R4) q_mv_r4_t<F>(w, x, out, bias, p, uint3(xg, y, 0), sg, lane, 32);
                    else q_mv_r2_t<F>(w, x, out, bias, p, uint3(xg, y, 0), sg, lane, 32);
                });
    } else {
        const bool half_tiles = kind == MM_H;
        const unsigned bytes = half_tiles ? Q_MM_SHMEM_HALF : Q_MM_SHMEM_FLOAT;
        const unsigned gx = ceil_div(p.rows, 64), gy = ceil_div(p.batch, 32);
        for (unsigned y = 0; y < gy; ++y)
            for (unsigned xg = 0; xg < gx; ++xg) {
                std::vector<float> shmem(bytes / 4 + 8);  // 4-byte aligned is enough for the emulation; sanitize=alignment checks use
                emu::run_group(128, 32, [&](unsigned tid, unsigned sg, unsigned) {
                    char *sh = reinterpret_cast<char *>(shmem.data());
                    if (half_tiles) q_mm_t<F, half>(w, x, out, bias, p, sh, uint3(xg, y, 0), tid, sg);
                    else q_mm_t<F, float>(w, x, out, bias, p, sh, uint3(xg, y, 0), tid, sg);
                });
            }
    }
}

template <class F>
void run_case(const std::string &name, const Case &c, Totals &tot) {
    Guarded wbuf(c.devlen);
    memcpy(wbuf.data, c.dev.data(), c.devlen);
    const unsigned maxb = 40;
    float *x = aligned_floats((size_t)maxb * c.cols);
    for (size_t i = 0; i < (size_t)maxb * c.cols; ++i) x[i] = unit() * 1.5f;
    float *bias = aligned_floats(c.rows);
    for (unsigned r = 0; r < c.rows; ++r) bias[r] = unit();
    float *y = aligned_floats((size_t)maxb * c.cols);  // permuted activations
    // the same activations with 8 padding floats (poisoned) behind every row, to test x_stride
    const unsigned wide_stride = c.cols + 8;
    float *x2 = aligned_floats((size_t)maxb * wide_stride);
    for (unsigned b = 0; b < maxb; ++b)
        for (unsigned k = 0; k < wide_stride; ++k) x2[(size_t)b * wide_stride + k] = k < c.cols ? x[(size_t)b * c.cols + k] : 12345.0f;

    struct Run { Kind kind; unsigned batch; unsigned nb; const char *label; };
    std::vector<Run> runs;
    for (unsigned b : {1u, 2u, 3u, 4u, 5u, 8u}) {
        runs.push_back({MV_R4, b, b, "mv_r4"});
        runs.push_back({MV_R2, b, b, "mv_r2"});
    }
    runs.push_back({MV_R2, 11, 8, "mv_r2(y-grid)"});
    runs.push_back({MM_F, 9, 0, "mm_f"});
    runs.push_back({MM_F, 40, 0, "mm_f"});
    runs.push_back({MM_H, 9, 0, "mm_h"});
    runs.push_back({MM_H, 33, 0, "mm_h"});

    for (const Run &run : runs) {
        for (int mode = 0; mode < 3; ++mode) {  // 0 plain, 1 bias + scale + row_base, 2 residual
            if (mode != 0 && !(run.batch == 3 || run.batch == 9 || run.batch == 33 || run.batch == 11)) continue;
            q_params p{};
            p.plane1 = (uint64_t)c.p1lo | ((uint64_t)c.p1hi << 32);
            p.rows = c.rows; p.cols = c.cols; p.batch = run.batch;
            p.scale = mode == 1 ? 0.7f : 1.0f;
            p.flags = mode == 1 ? 1u : mode == 2 ? 2u : 0u;
            const unsigned row_base = mode == 1 ? 5 : 0;
            p.out_stride = c.rows + row_base + 3;
            p.row_base = row_base;
            p.x_stride = c.cols;
            p.nb_blocks = c.nbb;
            const bool wide = mode == 1;  // padded activation rows
            const float *xsrc = wide ? x2 : x;
            const unsigned stride_src = wide ? wide_stride : c.cols;
            const float *xin = xsrc;
            p.x_stride = stride_src;
            if (c.perm) {  // fllm_q_permute_x: reads x with its stride, writes the dense copy
                q_permute_params pp{run.batch, c.cols, stride_src, 0};
                for (unsigned gid = 0; gid < run.batch * c.cols; ++gid) fllm_q_permute_x(xsrc, y, c.permutation.data(), pp, gid);
                xin = y;
                p.x_stride = c.cols;
            }
            std::vector<float> out((size_t)run.batch * p.out_stride + 16, 0.0f);
            std::vector<float> before;
            if (mode == 2) {
                for (auto &v : out) v = unit();
                before = out;
            }
            const std::vector<float> sentinel = out;
            // the kernels get bias pointing at bias[0]; row_base shifts the index inside the kernel
            std::vector<float> bias_full(c.rows + row_base, 0.0f);
            for (unsigned r = 0; r < c.rows; ++r) bias_full[row_base + r] = bias[r];
            Guarded bg((bias_full.size() * 4 + 15) / 16 * 16);
            memcpy(bg.data, bias_full.data(), bias_full.size() * 4);
            const float *bias_ptr = reinterpret_cast<const float *>(bg.data);
            dispatch<F>(run.kind, c, wbuf.data, xin, out.data(), bias_ptr, p, run.nb);

            double worst = 0;
            int bad = 0;
            const double tol_rel = run.kind == MM_H ? 4e-3 : 1e-4;
            for (unsigned b = 0; b < run.batch; ++b) {
                for (unsigned r = 0; r < c.rows; ++r) {
                    double acc = 0, mag = 0;
                    for (unsigned k = 0; k < c.cols; ++k) {
                        const double t = (double)c.ref[(size_t)r * c.cols + k] * x[(size_t)b * c.cols + k];
                        acc += t;
                        mag += fabs(t);
                    }
                    double want = acc * p.scale + (mode == 1 ? bias[r] : 0.0) + (mode == 2 ? before[(size_t)b * p.out_stride + row_base + r] : 0.0);
                    double got = out[(size_t)b * p.out_stride + row_base + r];
                    double err = fabs(got - want);
                    double allow = tol_rel * mag * fabs(p.scale) + 1e-5 + (mode == 2 ? 1e-6 : 0.0);
                    worst = fmax(worst, err / (mag * fabs(p.scale) + 1e-9));
                    if (!(err <= allow)) {
                        if (bad < 3) printf("    MISMATCH %s %s batch %u mode %d token %u row %u: got %.7g want %.7g (mag %.3g)\n", name.c_str(), run.label, run.batch, mode, b, r, got, want, mag);
                        ++bad;
                    }
                }
            }
            // untouched elements (outside the written rows) must keep their previous value
            for (size_t i = 0; i < out.size(); ++i) {
                const unsigned b = i / p.out_stride, col = i % p.out_stride;
                const bool written = b < run.batch && col >= row_base && col < row_base + c.rows;
                if (!written && out[i] != sentinel[i]) { if (bad < 3) printf("    STRAY WRITE %s %s at %zu\n", name.c_str(), run.label, i); ++bad; }
            }
            ++tot.checks;
            if (bad) ++tot.failures;
            printf("  %-9s %-14s batch %2u mode %d  worst rel err %.2e  %s\n", name.c_str(), run.label, run.batch, mode, worst, bad ? "FAIL" : "ok");
        }
    }
}

struct Entry { const char *tag; void (*run)(const std::string &, const Case &, Totals &); };

#define ENTRY(tag, F) {tag, run_case<F>}
const Entry entries[] = {
    ENTRY("q4_0", q_fmt_q4_0), ENTRY("q4_1", q_fmt_q4_1), ENTRY("q5_0", q_fmt_q5_0), ENTRY("q5_1", q_fmt_q5_1),
    ENTRY("q8_0", q_fmt_q8_0), ENTRY("iq4_nl", q_fmt_iq4_nl), ENTRY("q2_k", q_fmt_q2_k), ENTRY("q3_k", q_fmt_q3_k),
    ENTRY("q4_k", q_fmt_q4_k), ENTRY("q5_k", q_fmt_q5_k), ENTRY("q6_k", q_fmt_q6_k), ENTRY("iq4_xs", q_fmt_iq4_xs),
    ENTRY("mxfp4", q_fmt_mxfp4), ENTRY("nvfp4", q_fmt_nvfp4), ENTRY("q4z32", q_fmt_q4z32), ENTRY("q4z64", q_fmt_q4z64),
    ENTRY("q4z128", q_fmt_q4z128), ENTRY("q4z32p", q_fmt_q4z32), ENTRY("q4z64p", q_fmt_q4z64), ENTRY("q4z128p", q_fmt_q4z128),
    ENTRY("f32", q_fmt_f32), ENTRY("f16", q_fmt_f16), ENTRY("bf16", q_fmt_bf16), ENTRY("fp8e4m3", q_fmt_fp8e4m3),
    ENTRY("fp8e5m2", q_fmt_fp8e5m2), ENTRY("fp8m", q_fmt_fp8m),
    // MLX affine (MLXA): the `b` tags are the bfloat16 variants (the dump carries Q_MLX_BF16 in plane 1)
    ENTRY("mlx2g32", q_fmt_mlx2g32), ENTRY("mlx2g32b", q_fmt_mlx2g32), ENTRY("mlx2g64", q_fmt_mlx2g64), ENTRY("mlx2g64b", q_fmt_mlx2g64), ENTRY("mlx2g128", q_fmt_mlx2g128), ENTRY("mlx2g128b", q_fmt_mlx2g128),
    ENTRY("mlx3g32", q_fmt_mlx3g32), ENTRY("mlx3g32b", q_fmt_mlx3g32), ENTRY("mlx3g64", q_fmt_mlx3g64), ENTRY("mlx3g64b", q_fmt_mlx3g64), ENTRY("mlx3g128", q_fmt_mlx3g128), ENTRY("mlx3g128b", q_fmt_mlx3g128),
    ENTRY("mlx4g32", q_fmt_mlx4g32), ENTRY("mlx4g32b", q_fmt_mlx4g32), ENTRY("mlx4g64", q_fmt_mlx4g64), ENTRY("mlx4g64b", q_fmt_mlx4g64), ENTRY("mlx4g128", q_fmt_mlx4g128), ENTRY("mlx4g128b", q_fmt_mlx4g128),
    ENTRY("mlx5g32", q_fmt_mlx5g32), ENTRY("mlx5g32b", q_fmt_mlx5g32), ENTRY("mlx5g64", q_fmt_mlx5g64), ENTRY("mlx5g64b", q_fmt_mlx5g64), ENTRY("mlx5g128", q_fmt_mlx5g128), ENTRY("mlx5g128b", q_fmt_mlx5g128),
    ENTRY("mlx6g32", q_fmt_mlx6g32), ENTRY("mlx6g32b", q_fmt_mlx6g32), ENTRY("mlx6g64", q_fmt_mlx6g64), ENTRY("mlx6g64b", q_fmt_mlx6g64), ENTRY("mlx6g128", q_fmt_mlx6g128), ENTRY("mlx6g128b", q_fmt_mlx6g128),
    ENTRY("mlx8g32", q_fmt_mlx8g32), ENTRY("mlx8g32b", q_fmt_mlx8g32), ENTRY("mlx8g64", q_fmt_mlx8g64), ENTRY("mlx8g64b", q_fmt_mlx8g64), ENTRY("mlx8g128", q_fmt_mlx8g128), ENTRY("mlx8g128b", q_fmt_mlx8g128),
};

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <dump dir> <tag>...\n", argv[0]); return 2; }
    const std::string dir = argv[1];
    int failures = 0, checks = 0;
    for (int a = 2; a < argc; ++a) {
        const Entry *entry = nullptr;
        for (const Entry &e : entries) if (std::string(e.tag) == argv[a]) entry = &e;
        if (!entry) { fprintf(stderr, "unknown tag %s\n", argv[a]); return 2; }
        for (const char *suffix : {"", "_big", "_tail", "_odd"}) {
            Case c;
            if (!load(dir + "/" + entry->tag + suffix + ".bin", c)) continue;
            Totals tot;
            printf("%s%s: %u x %u (qk %u, device image %u bytes)\n", entry->tag, suffix, c.rows, c.cols, c.qk, c.devlen);
            entry->run(std::string(entry->tag) + suffix, c, tot);
            printf("%s%s: %d checks, %d failures\n", entry->tag, suffix, tot.checks, tot.failures);
            checks += tot.checks;
            failures += tot.failures;
        }
    }
    printf("TOTAL %d checks, %d failures\n", checks, failures);
    return failures ? 1 : 0;
}
