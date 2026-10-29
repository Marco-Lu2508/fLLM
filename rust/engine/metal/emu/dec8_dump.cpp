// Chunk-decoder check against the vectors of the Rust side, for the formats added with the row-aware selector (block-wise FP8 in every tile shape the
// dump carries, bitsandbytes NF4 / FP4 / INT8R, EXL2) and a few older formats as controls.
//
// Reads the files written by the Rust test `backend::metal::weights::tests::dump_vectors_for_the_msl_emulation` (device image exactly as `pack_weight`
// makes it, parameter block fields `plane1` / `pad`, the reference weights of `compute::dequantize`, and the permutation of EXL2 / act-order weights)
// and decodes EVERY chunk of every row through the real `q_dec8<F>` selector of common.metal, compared bit for bit with the reference. For the
// permuted formats the chunk holds the weights of the stored inputs 8 c .. 8 c + 8, which are the reference columns perm[k]. No GPU, no shim: this runs with
// the mini stdlib (run_dec8.sh dump). The matrix-vector body wiring is checked by kernel_mv.cpp, the full kernels by harness.cpp (run_emu.sh).
//   usage: dec8_dump <dump dir> [tag...]      (default: the tags below)
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "unit.cpp"

namespace {

struct Case {
    uint32_t rows, cols, qk, nbb, p1lo, p1hi, devlen, perm, type, pad;
    std::vector<uint8_t> dev;
    std::vector<float> ref;
    std::vector<uint32_t> permutation;
};

bool load(const std::string &path, Case &c) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    uint32_t h[11];
    if (fread(h, 4, 11, f) != 11 || h[0] != 0x4d544c51u) { fprintf(stderr, "bad header %s\n", path.c_str()); exit(2); }
    c.rows = h[1]; c.cols = h[2]; c.qk = h[3]; c.nbb = h[4]; c.p1lo = h[5]; c.p1hi = h[6]; c.devlen = h[7]; c.perm = h[8]; c.type = h[9]; c.pad = h[10];
    c.dev.resize(c.devlen + 16);  // the image sits in an aligned allocation
    c.ref.resize(static_cast<size_t>(c.rows) * c.cols);
    bool ok = fread(c.dev.data(), 1, c.devlen, f) == c.devlen && fread(c.ref.data(), 4, c.ref.size(), f) == c.ref.size();
    if (c.perm) {
        c.permutation.resize(c.cols);
        ok = ok && fread(c.permutation.data(), 4, c.cols, f) == c.cols;
    }
    fclose(f);
    if (!ok) { fprintf(stderr, "short file %s\n", path.c_str()); exit(2); }
    return true;
}

template <class F>
int run(const std::string &name, const Case &c, long &total) {
    uint8_t *image = static_cast<uint8_t *>(aligned_alloc(16, (c.devlen + 15) / 16 * 16));
    memcpy(image, c.dev.data(), c.devlen);
    q_params p{};
    p.plane1 = static_cast<uint64_t>(c.p1lo) | (static_cast<uint64_t>(c.p1hi) << 32);
    p.rows = c.rows;
    p.cols = c.cols;
    p.batch = 1;
    p.nb_blocks = c.nbb;
    p.pad = c.pad;
    int bad = 0;
    for (unsigned row = 0; row < c.rows; ++row)
        for (unsigned ci = 0; ci < c.cols / 8; ++ci) {
            float o[8];
            q_dec8<F>(image, row, ci / F::CPB, ci % F::CPB, p, o);
            for (unsigned k = 0; k < 8; ++k) {
                const unsigned col = c.perm ? c.permutation[ci * 8 + k] : ci * 8 + k;
                const float want = c.ref[static_cast<size_t>(row) * c.cols + col];
                ++total;
                const bool same = std::isnan(want) ? std::isnan(o[k]) : memcmp(&o[k], &want, 4) == 0;
                if (!same) {
                    if (bad < 5) printf("  MISMATCH %s row %u chunk %u value %u: got %.9g want %.9g\n", name.c_str(), row, ci, k, o[k], want);
                    ++bad;
                }
            }
        }
    free(image);
    return bad;
}

struct Entry { const char *tag; int (*run)(const std::string &, const Case &, long &); };

#define ENTRY(tag, F) {tag, run<F>}
const Entry entries[] = {
    ENTRY("fp8b", q_fmt_fp8b), ENTRY("fp8bt16x32", q_fmt_fp8b), ENTRY("fp8bt1x8", q_fmt_fp8b), ENTRY("fp8bt2x4", q_fmt_fp8b),
    ENTRY("fp8bt64x1", q_fmt_fp8b), ENTRY("fp8bt1x512", q_fmt_fp8b), ENTRY("fp8bt32x16", q_fmt_fp8b),
    ENTRY("nf4b64", q_fmt_nf4b64), ENTRY("nf4b128", q_fmt_nf4b128), ENTRY("fp4b64", q_fmt_fp4b64), ENTRY("fp4b128", q_fmt_fp4b128), ENTRY("int8r", q_fmt_int8r),
    ENTRY("exl2p", q_fmt_exl2),
    // controls: formats that are not row aware go through the same selector (FP8M is left out: its row scale is applied after the dot product)
    ENTRY("q4_0", q_fmt_q4_0), ENTRY("q6_k", q_fmt_q6_k), ENTRY("mxfp4", q_fmt_mxfp4), ENTRY("q4z64p", q_fmt_q4z64), ENTRY("mlx4g32b", q_fmt_mlx4g32),
    ENTRY("f16", q_fmt_f16),
};

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <dump dir> [tag...]\n", argv[0]); return 2; }
    const std::string dir = argv[1];
    std::vector<std::string> tags;
    for (int a = 2; a < argc; ++a) tags.push_back(argv[a]);
    if (tags.empty()) for (const Entry &e : entries) tags.push_back(e.tag);
    long total = 0;
    int bad = 0, files = 0;
    for (const std::string &tag : tags) {
        const Entry *entry = nullptr;
        for (const Entry &e : entries) if (tag == e.tag) entry = &e;
        if (!entry) { fprintf(stderr, "unknown tag %s\n", tag.c_str()); return 2; }
        for (const char *suffix : {"", "_big", "_tail", "_odd"}) {
            Case c;
            if (!load(dir + "/" + tag + suffix + ".bin", c)) continue;
            const long before = total;
            const int b = entry->run(tag + suffix, c, total);
            printf("%-18s %3u x %-5u %8ld values  %s\n", (tag + suffix).c_str(), c.rows, c.cols, total - before, b ? "FAIL" : "ok");
            bad += b;
            ++files;
        }
    }
    printf("dec8_dump: %d files, %ld values decoded, %d mismatches\n", files, total, bad);
    return (bad || files == 0) ? 1 : 0;
}
