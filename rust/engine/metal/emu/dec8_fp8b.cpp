// Standalone check of the block-wise FP8 chunk decoder of common.metal (`q_fmt_fp8b::dec8p` through the `q_dec8<F>` selector the kernel bodies
// call), run as C++ on the CPU. Same build as dec8_mlx.cpp (run_dec8.sh fp8b): common.metal after the mechanical sed, mini_stdlib/metal_stdlib.
//
// The data are made here, not by the Rust side: random E4M3 codes (all 256 patterns, NaN included) and random f32 tile scales are written into a
// device image laid out as the Rust `pack_weight` leaves it (codes row-major, then the scale table, then the 16-byte trailer), for many tile shapes
// (rows and columns that are no whole tiles, tiles narrower than a chunk of 8 codes, a tile wider than the matrix, 1 x N tiles). The parameter
// block is built as `plan_quantized_linear` does (plane1 = rows * cols, pad = log2(tile rows) | log2(tile cols) << 8, nb_blocks = cols / 8).
// Every chunk is decoded by the real selector and compared bit for bit with `e4m3(code) * scale` computed here by a formula that shares no code
// with the decoder (ldexp instead of exponent-field arithmetic), the arithmetic of cpu_fp8.rs.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "unit.cpp"

namespace {

uint64_t state = 7;
uint32_t next32() {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<uint32_t>(state >> 33);
}

// E4M3FN by definition: bias 7, subnormals m * 2^-9, S.1111.111 is NaN
float e4m3(unsigned code) {
    const unsigned e = (code >> 3) & 15, m = code & 7;
    float v;
    if (e == 15 && m == 7) return NAN;
    if (e == 0) v = std::ldexp(static_cast<float>(m), -9);
    else v = std::ldexp(1.0f + static_cast<float>(m) / 8.0f, static_cast<int>(e) - 7);
    return (code & 0x80) ? -v : v;
}

struct Shape { unsigned rows, cols, tr, tc; };

int check(const Shape &s, int &total) {
    const unsigned tiles_per_row = (s.cols + s.tc - 1) / s.tc, tile_rows = (s.rows + s.tr - 1) / s.tr;
    const size_t n = static_cast<size_t>(s.rows) * s.cols, ntiles = static_cast<size_t>(tile_rows) * tiles_per_row;
    const size_t bytes = n + ntiles * 4 + 16;
    // 16-byte aligned image that ends right at the allocation: a read behind the scale table or the codes is caught by the sanitizer build / valgrind
    uint8_t *image = static_cast<uint8_t *>(aligned_alloc(16, (bytes + 15) / 16 * 16));
    memset(image, 0, (bytes + 15) / 16 * 16);
    for (size_t i = 0; i < n; ++i) image[i] = static_cast<uint8_t>(next32());
    std::vector<float> scales(ntiles);
    for (size_t t = 0; t < ntiles; ++t) {
        scales[t] = 0.001f + static_cast<float>(next32() % 1000) / 500.0f;
        if (next32() % 5 == 0) scales[t] = std::ldexp(1.0f, static_cast<int>(next32() % 20) - 10);  // power-of-two (UE8M0) scales
        memcpy(image + n + 4 * t, &scales[t], 4);
    }
    q_params p{};
    p.plane1 = n;
    p.rows = s.rows;
    p.cols = s.cols;
    p.batch = 1;
    p.nb_blocks = s.cols / 8;
    unsigned lr = 0, lc = 0;
    while ((1u << lr) < s.tr) ++lr;
    while ((1u << lc) < s.tc) ++lc;
    p.pad = lr | (lc << 8);
    int bad = 0;
    for (unsigned row = 0; row < s.rows; ++row)
        for (unsigned blk = 0; blk < s.cols / 8; ++blk) {
            float o[8];
            q_dec8<q_fmt_fp8b>(image, row, blk, 0, p, o);
            for (unsigned k = 0; k < 8; ++k) {
                const unsigned col = blk * 8 + k;
                const float want = e4m3(image[static_cast<size_t>(row) * s.cols + col]) * scales[(row / s.tr) * tiles_per_row + col / s.tc];
                ++total;
                const bool same = std::isnan(want) ? std::isnan(o[k]) : memcmp(&o[k], &want, 4) == 0;
                if (!same) {
                    if (bad < 5) printf("  MISMATCH tile %ux%u matrix %ux%u row %u col %u: got %.9g want %.9g\n", s.tr, s.tc, s.rows, s.cols, row, col, o[k], want);
                    ++bad;
                }
            }
        }
    free(image);
    return bad;
}

}  // namespace

int main() {
    static_assert(q_fmt_fp8b::QK == 8 && q_fmt_fp8b::BYTES == 8 && q_fmt_fp8b::CPB == 1, "geometry");
    static_assert(q_fmt_fp8b::ROW_AWARE, "row aware");
    static_assert(!q_fmt_q4_0::ROW_AWARE && !q_fmt_mlx4g32::ROW_AWARE && !q_fmt_fp8m::ROW_AWARE && !q_fmt_nf4b64::ROW_AWARE, "the other formats take the block index");
    int total = 0, bad = 0;
    const unsigned tiles[][2] = {{128, 128}, {16, 32}, {1, 8}, {2, 4}, {64, 1}, {1, 512}, {32, 16}, {4, 4}, {1, 1}, {4096, 2}, {2, 32768}, {1, 128}, {128, 1}, {8, 8}};
    const unsigned shapes[][2] = {{37, 72}, {9, 24}, {130, 264}, {3, 8}, {64, 256}};
    for (const auto &t : tiles)
        for (const auto &m : shapes) bad += check({m[0], m[1], t[0], t[1]}, total);
    printf("dec8_fp8b: %d values decoded, %d mismatches\n", total, bad);
    return bad ? 1 : 0;
}
