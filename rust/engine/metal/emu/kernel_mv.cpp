// Kernel-level check of the matrix-vector kernels `q_mv_r4_t<F>` / `q_mv_r2_t<F>` of common_q_gemv.metal for the formats added with the row-aware
// decoder selector (FP8B in several tile shapes, bitsandbytes NF4 / FP4 / INT8R) and, as a control, MLX 4 bit (a format that is not row aware).
// Build: run_dec8.sh kernel_mv (common.metal + common_q_gemv.metal after the mechanical sed, mini_stdlib).
//
// The threads of a threadgroup are std::threads (emu_runtime.h), the lanes of a simdgroup meet in `simd_sum` through a barrier, threadgroups run one
// after the other, the grid is walked as the planner states it (x = ceil(rows / (2 * NR)), y = ceil(batch / nb)). The reference matrix is the output
// of the chunk decoder selector q_dec8 for every chunk (checked bit for bit against independent formulas by dec8_fp8b.cpp / dec8_bnb.cpp /
// dec8_mlx.cpp), so this test is about the wiring of the kernel body: the row and block index handed to the decoder, the partial last threadgroup,
// the batch columns, the bias / scale epilogue, the padded activation rows. Rows are not a multiple of the rows per threadgroup. No GPU involved.
// The matrix-matrix kernel needs simdgroup_matrix and is covered by run_emu.sh with the full shim only.
#include "emu_runtime.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "unit.cpp"
#include "gemv.cpp"

float simd_sum(float v) {
    auto &c = emu::ctx;
    c.group->slots[c.tid] = v;
    c.group->sg[c.sg]->arrive_and_wait();
    float s = 0;
    for (unsigned i = 0; i < c.group->simd_width; ++i) s += c.group->slots[c.sg * c.group->simd_width + i];
    c.group->sg[c.sg]->arrive_and_wait();
    return s;
}

namespace {

uint64_t state = 3;
uint32_t next32() {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<uint32_t>(state >> 33);
}

struct Image {
    std::vector<uint8_t> bytes;
    q_params p{};
};

void fill(Image &im, size_t n) {
    im.bytes.resize((n + 15) / 16 * 16 + 16);
    for (auto &b : im.bytes) b = static_cast<uint8_t>(next32());
}

void put_f32(Image &im, size_t at, float v) { memcpy(&im.bytes[at], &v, 4); }

float small() { return 0.001f + static_cast<float>(next32() % 100) / 4000.0f; }

// kind: 0 fp8b (tile tr x tc), 1 nf4b64, 2 fp4b128, 3 int8r, 4 mlx4g32
Image make(int kind, unsigned rows, unsigned cols, unsigned tr, unsigned tc) {
    Image im;
    im.p.rows = rows;
    im.p.cols = cols;
    if (kind == 0) {
        const size_t n = static_cast<size_t>(rows) * cols;
        const size_t ntiles = static_cast<size_t>((rows + tr - 1) / tr) * ((cols + tc - 1) / tc);
        fill(im, n + 4 * ntiles + 16);
        for (size_t i = 0; i < n; ++i)
            if ((im.bytes[i] & 0x7f) == 0x7f) im.bytes[i] ^= 1;
        for (size_t t = 0; t < ntiles; ++t) put_f32(im, n + 4 * t, small());
        unsigned lr = 0, lc = 0;
        while ((1u << lr) < tr) ++lr;
        while ((1u << lc) < tc) ++lc;
        im.p.plane1 = n;
        im.p.pad = lr | (lc << 8);
        im.p.nb_blocks = cols / 8;
    } else if (kind >= 1 && kind <= 3) {
        const unsigned block = kind == 2 ? 128 : 64, bytes = kind == 3 ? 68 : 4 + block / 2;
        const size_t blocks = static_cast<size_t>(rows) * (cols / block);
        fill(im, blocks * bytes);
        for (size_t g = 0; g < blocks; ++g) put_f32(im, g * bytes, small());
        im.p.nb_blocks = cols / block;
    } else {
        // MLX 4 bit, group 32, f16 scales: plane 0 = 16 code bytes per group, plane 1 = one word per group (scale | bias << 16)
        const size_t groups = static_cast<size_t>(rows) * (cols / 32);
        const size_t plane1 = (groups * 16 + 15) / 16 * 16;
        fill(im, plane1 + groups * 4);
        for (size_t g = 0; g < groups; ++g) {
            const _Float16 s = static_cast<_Float16>(small()), z = static_cast<_Float16>(0.01f);
            uint16_t sb, zb;
            memcpy(&sb, &s, 2);
            memcpy(&zb, &z, 2);
            const uint32_t word = sb | (static_cast<uint32_t>(zb) << 16);
            memcpy(&im.bytes[plane1 + 4 * g], &word, 4);
        }
        im.p.plane1 = plane1;
        im.p.nb_blocks = cols / 32;
    }
    return im;
}

template <class F>
int run(const char *name, int kind, unsigned rows, unsigned cols, unsigned tr, unsigned tc, int &checks) {
    Image im = make(kind, rows, cols, tr, tc);
    const unsigned cpb = F::CPB;
    // reference matrix from the decoder selector
    std::vector<double> ref(static_cast<size_t>(rows) * cols);
    for (unsigned r = 0; r < rows; ++r)
        for (unsigned ci = 0; ci < cols / 8; ++ci) {
            float o[8];
            q_dec8<F>(im.bytes.data(), r, ci / cpb, ci % cpb, im.p, o);
            for (unsigned k = 0; k < 8; ++k) ref[static_cast<size_t>(r) * cols + ci * 8 + k] = o[k];
        }
    int bad = 0;
    for (unsigned variant = 0; variant < 2; ++variant) {  // r4, r2
        const unsigned nr = variant == 0 ? 4 : 2;
        for (unsigned batch : {1u, 3u, 4u, 5u, 8u, 11u}) {
            const unsigned nb = batch > 8 ? 8 : batch;
            if ((variant == 0) != (nb <= 4)) continue;
            q_params p = im.p;
            p.batch = batch;
            p.x_stride = cols + 8;
            p.out_stride = rows + 3;
            p.row_base = 2;
            p.flags = 1;  // bias
            p.scale = 0.75f;
            std::vector<float> x(static_cast<size_t>(batch) * p.x_stride + 8, 777.0f), bias(rows + 2, 0.0f);
            for (unsigned b = 0; b < batch; ++b)
                for (unsigned k = 0; k < cols; ++k) x[static_cast<size_t>(b) * p.x_stride + k] = static_cast<float>(next32() % 2001) / 1000.0f - 1.0f;
            for (unsigned r = 0; r < rows; ++r) bias[2 + r] = static_cast<float>(next32() % 100) / 100.0f;
            std::vector<float> out(static_cast<size_t>(batch) * p.out_stride + 4, -5.0f);
            q_fc_nb = nb;
            const unsigned gx = (rows + 2 * nr - 1) / (2 * nr), gy = (batch + nb - 1) / nb;
            for (unsigned y = 0; y < gy; ++y)
                for (unsigned xg = 0; xg < gx; ++xg)
                    emu::run_group(64, 32, [&](unsigned, unsigned sg, unsigned lane) {
                        if (variant == 0) q_mv_r4_t<F>(im.bytes.data(), x.data(), out.data(), bias.data(), p, uint3(xg, y, 0), sg, lane, 32);
                        else q_mv_r2_t<F>(im.bytes.data(), x.data(), out.data(), bias.data(), p, uint3(xg, y, 0), sg, lane, 32);
                    });
            int local = 0;
            for (unsigned b = 0; b < batch; ++b)
                for (unsigned r = 0; r < rows; ++r) {
                    double acc = 0, mag = 0;
                    for (unsigned k = 0; k < cols; ++k) {
                        const double t = ref[static_cast<size_t>(r) * cols + k] * x[static_cast<size_t>(b) * p.x_stride + k];
                        acc += t;
                        mag += std::fabs(t);
                    }
                    const double want = acc * 0.75 + bias[2 + r];
                    const double got = out[static_cast<size_t>(b) * p.out_stride + 2 + r];
                    if (!(std::fabs(got - want) <= 1e-4 * mag * 0.75 + 1e-5)) {
                        if (local < 3) printf("  MISMATCH %s r%u batch %u token %u row %u: got %.7g want %.7g\n", name, nr, batch, b, r, got, want);
                        ++local;
                    }
                }
            // elements outside the written range keep their sentinel
            for (size_t i = 0; i < out.size(); ++i) {
                const size_t b = i / p.out_stride, col = i % p.out_stride;
                const bool written = b < batch && col >= 2 && col < 2 + rows;
                if (!written && out[i] != -5.0f) {
                    if (local < 3) printf("  STRAY WRITE %s at %zu\n", name, i);
                    ++local;
                }
            }
            ++checks;
            bad += local ? 1 : 0;
        }
    }
    return bad;
}

}  // namespace

int main() {
    int checks = 0, bad = 0;
    const unsigned tiles[][2] = {{128, 128}, {16, 32}, {2, 4}, {1, 512}, {64, 1}};
    for (const auto &t : tiles) bad += run<q_fmt_fp8b>("fp8b", 0, 37, 264, t[0], t[1], checks);
    bad += run<q_fmt_nf4b64>("nf4b64", 1, 37, 192, 0, 0, checks);
    bad += run<q_fmt_fp4b128>("fp4b128", 2, 21, 256, 0, 0, checks);
    bad += run<q_fmt_int8r>("int8r", 3, 37, 192, 0, 0, checks);
    bad += run<q_fmt_mlx4g32>("mlx4g32 (control)", 4, 37, 96, 0, 0, checks);
    printf("kernel_mv: %d kernel runs, %d failures\n", checks, bad);
    return bad ? 1 : 0;
}
