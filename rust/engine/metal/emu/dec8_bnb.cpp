// Standalone check of the bitsandbytes chunk decoders of common.metal (`q_fmt_bnb4<BLOCK, FP4>`: NF4 / FP4, `q_fmt_int8r`: LLM.int8()), run as C++ on
// the CPU through the `q_dec8<F>` selector the kernel bodies call. Same build as dec8_mlx.cpp (run_dec8.sh bnb).
//
// Random code bytes and f32 scales are written into a device image in the file layout of gguf-probe `bnb` (blocks of `[f32 scale][codes]`; code byte
// j holds the elements 2j in the high and 2j + 1 in the low nibble) and every chunk of every block is decoded and compared bit for bit with the
// value the definition gives: `book[nibble] * scale` with the books written here from their definition (the NF4 table of bitsandbytes
// `create_normal_map` as doubles, the FP4 grid as 0, 1/16, 8, 12, 4, 6, 2, 3 divided by 12 in f32), `int8 * scale` for INT8R.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "unit.cpp"

namespace {

uint64_t state = 11;
uint32_t next32() {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<uint32_t>(state >> 33);
}

const double nf4_double[16] = {-1.0, -0.6961928009986877, -0.5250730514526367, -0.39491748809814453, -0.28444138169288635, -0.18477343022823334,
                               -0.09105003625154495, 0.0, 0.07958029955625534, 0.16093020141124725, 0.24611230194568634, 0.33791524171829224,
                               0.44070982933044434, 0.5626170039176941, 0.7229568362236023, 1.0};

float book(bool fp4, unsigned i) {
    if (!fp4) return static_cast<float>(nf4_double[i]);
    const float mag[8] = {0.0f, 0.0625f / 12.0f, 8.0f / 12.0f, 12.0f / 12.0f, 4.0f / 12.0f, 6.0f / 12.0f, 2.0f / 12.0f, 3.0f / 12.0f};
    return (i & 8) ? -mag[i & 7] : mag[i & 7];
}

template <class F>
int check(const char *name, unsigned block, int kind, int &total) {  // kind 0 NF4, 1 FP4, 2 INT8R
    constexpr unsigned rows = 6;
    const unsigned blocks_per_row = 5, cols = blocks_per_row * block;
    const unsigned bytes = kind == 2 ? 4 + block : 4 + block / 2;
    static_assert(F::BYTES % 4 == 0, "4-byte blocks");
    if (F::QK != block || F::BYTES != bytes || F::CPB != block / 8) { printf("  GEOMETRY mismatch %s\n", name); return 1; }
    const size_t n = static_cast<size_t>(rows) * blocks_per_row * bytes;
    uint8_t *image = static_cast<uint8_t *>(aligned_alloc(16, (n + 15) / 16 * 16));
    std::vector<float> scale(rows * blocks_per_row);
    for (unsigned g = 0; g < rows * blocks_per_row; ++g) {
        scale[g] = 0.001f + static_cast<float>(next32() % 100) / 4000.0f;
        if (next32() % 7 == 0) scale[g] = -scale[g];
        memcpy(image + g * bytes, &scale[g], 4);
        for (unsigned i = 4; i < bytes; ++i) image[g * bytes + i] = static_cast<uint8_t>(next32());
    }
    q_params p{};
    p.rows = rows;
    p.cols = cols;
    p.batch = 1;
    p.nb_blocks = blocks_per_row;
    int bad = 0;
    for (unsigned row = 0; row < rows; ++row)
        for (unsigned ci = 0; ci < cols / 8; ++ci) {
            const unsigned blk = ci / F::CPB, c = ci % F::CPB;
            float o[8];
            q_dec8<F>(image, row, blk, c, p, o);
            const uint8_t *codes = image + (row * blocks_per_row + blk) * bytes + 4;
            const float d = scale[row * blocks_per_row + blk];
            for (unsigned k = 0; k < 8; ++k) {
                const unsigned e = c * 8 + k;  // element of the block
                float want;
                if (kind == 2) want = static_cast<float>(static_cast<int8_t>(codes[e])) * d;
                else {
                    const uint8_t byte = codes[e / 2];
                    want = book(kind == 1, (e % 2 == 0) ? byte >> 4 : byte & 15) * d;
                }
                ++total;
                if (memcmp(&o[k], &want, 4) != 0) {
                    if (bad < 5) printf("  MISMATCH %s row %u chunk %u value %u: got %.9g want %.9g\n", name, row, ci, k, o[k], want);
                    ++bad;
                }
            }
        }
    free(image);
    return bad;
}

}  // namespace

int main() {
    int total = 0;
    const int bad = check<q_fmt_nf4b64>("nf4b64", 64, 0, total) + check<q_fmt_nf4b128>("nf4b128", 128, 0, total) + check<q_fmt_fp4b64>("fp4b64", 64, 1, total) +
                    check<q_fmt_fp4b128>("fp4b128", 128, 1, total) + check<q_fmt_int8r>("int8r", 64, 2, total);
    // the code book tables are the definition's, entry by entry
    int book_bad = 0;
    for (unsigned i = 0; i < 16; ++i) {
        float a = as_type<float>(q_kv_nf4[i]), b = as_type<float>(q_kv_bnbfp4[i]);
        const float wa = book(false, i), wb = book(true, i);
        if (memcmp(&a, &wa, 4) != 0 || memcmp(&b, &wb, 4) != 0) { printf("  BOOK entry %u differs\n", i); ++book_bad; }
    }
    printf("dec8_bnb: %d values decoded, %d mismatches, %d code book entries differ\n", total, bad, book_bad);
    return (bad || book_bad) ? 1 : 0;
}
