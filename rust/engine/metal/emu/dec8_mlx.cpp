// Standalone check of the MLX affine chunk decoder of common.metal (`q_fmt_mlx<BITS, GROUP>::dec8`, the MLXA formats), run as C++ on the CPU.
//
// No GPU, no Metal compiler and none of the shims of run_emu.sh are needed: common.metal is turned into C++ by the same mechanical sed as in
// run_emu.sh and compiled against mini_stdlib/metal_stdlib (the scalar types, the address-space words, as_type). The data are made here, not
// by the Rust side: random codes of every width are written into the device image the Rust `pack_weight` produces for the formats
// (plane 0 = the MLX words of every group, plane 1 = one `scale | bias << 16` word per group, bit 63 of the plane-1 offset for bfloat16),
// with a bit-by-bit writer that shares no code with the decoder; every chunk of 8 values of every group is decoded by the real `dec8`
// and compared with `code * scale + bias`, bit for bit. Run: run_dec8_mlx.sh
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "unit.cpp"

namespace {

uint64_t state = 99;
uint32_t next32() {
    state = state * 6364136223846793005ull + 1442695040888963407ull;
    return static_cast<uint32_t>(state >> 33);
}

// IEEE binary16 / bfloat16 of a float (values here are small and exactly representable after the rounding below)
uint16_t to_bf16(float v) {
    uint32_t u;
    memcpy(&u, &v, 4);
    return static_cast<uint16_t>(u >> 16);  // truncation: the value is then what the decoder must reproduce
}
uint16_t to_f16(float v) {
    _Float16 h = static_cast<_Float16>(v);
    uint16_t u;
    memcpy(&u, &h, 2);
    return u;
}
float from_bf16(uint16_t u) {
    uint32_t w = static_cast<uint32_t>(u) << 16;
    float v;
    memcpy(&v, &w, 4);
    return v;
}
float from_f16(uint16_t u) {
    _Float16 h;
    memcpy(&h, &u, 2);
    return static_cast<float>(h);
}

template <unsigned BITS, unsigned GROUP>
int check(bool bf16, int &total) {
    const unsigned groups = 40;  // groups, as if 5 rows of 8 groups
    const unsigned group_bytes = GROUP * BITS / 8;
    std::vector<uint32_t> codes(groups * GROUP);
    std::vector<uint8_t> image;
    // plane 0: the codes of every group as a little-endian bit stream, written bit by bit
    image.assign(groups * group_bytes, 0);
    for (unsigned g = 0; g < groups; ++g)
        for (unsigned i = 0; i < GROUP; ++i) {
            const uint32_t c = next32() & ((1u << BITS) - 1);
            codes[g * GROUP + i] = c;
            for (unsigned b = 0; b < BITS; ++b)
                if ((c >> b) & 1u) {
                    const unsigned at = i * BITS + b;
                    image[g * group_bytes + at / 8] |= static_cast<uint8_t>(1u << (at % 8));
                }
        }
    // plane 1 (16-byte aligned offset): scale | bias << 16
    const size_t plane1 = (image.size() + 15) / 16 * 16;
    image.resize(plane1 + groups * 4 + 16, 0);
    std::vector<float> scale(groups), bias(groups);
    for (unsigned g = 0; g < groups; ++g) {
        const float s = 0.001f + (next32() % 100) / 4000.0f, z = (next32() % 100) / 1000.0f - 0.05f;
        const uint16_t sr = bf16 ? to_bf16(s) : to_f16(s), br = bf16 ? to_bf16(z) : to_f16(z);
        scale[g] = bf16 ? from_bf16(sr) : from_f16(sr);
        bias[g] = bf16 ? from_bf16(br) : from_f16(br);
        const uint32_t word = sr | (static_cast<uint32_t>(br) << 16);
        memcpy(&image[plane1 + 4 * g], &word, 4);
    }
    const ulong p1 = static_cast<ulong>(plane1) | (bf16 ? Q_MLX_BF16 : 0ul);
    int bad = 0;
    for (unsigned g = 0; g < groups; ++g)
        for (unsigned c = 0; c < GROUP / 8; ++c) {
            float o[8];
            q_fmt_mlx<BITS, GROUP>::dec8(image.data(), g, c, p1, o);
            for (unsigned k = 0; k < 8; ++k) {
                const float want = static_cast<float>(codes[g * GROUP + c * 8 + k]) * scale[g] + bias[g];
                ++total;
                if (memcmp(&o[k], &want, 4) != 0) {
                    if (bad < 5) printf("  MISMATCH %u bits group %u bf16 %d: g %u chunk %u value %u: got %.9g want %.9g\n", BITS, GROUP, bf16, g, c, k, o[k], want);
                    ++bad;
                }
            }
        }
    static_assert(q_fmt_mlx<BITS, GROUP>::QK == GROUP, "QK");
    static_assert(q_fmt_mlx<BITS, GROUP>::BYTES == GROUP * BITS / 8, "BYTES");
    static_assert(q_fmt_mlx<BITS, GROUP>::CPB == GROUP / 8, "CPB");
    return bad;
}

template <unsigned BITS>
int widths(int &total) {
    int bad = 0;
    for (bool bf16 : {false, true}) {
        bad += check<BITS, 32>(bf16, total);
        bad += check<BITS, 64>(bf16, total);
        bad += check<BITS, 128>(bf16, total);
    }
    return bad;
}

}  // namespace

int main() {
    int total = 0;
    const int bad = widths<2>(total) + widths<3>(total) + widths<4>(total) + widths<5>(total) + widths<6>(total) + widths<8>(total);
    printf("dec8_mlx: %d values decoded, %d mismatches\n", total, bad);
    return bad ? 1 : 0;
}
