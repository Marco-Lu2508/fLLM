// SPDX-License-Identifier: Apache-2.0
// fLLM Intel GPU backend (SYCL / oneAPI): host-side check of the weight packer and of the device decoders (formats.hpp).
//
// No device is used. For every format the program makes random GGUF blocks, packs them with fllm_sycl_pack_weight (weights.cpp) and
// decodes the packed rows with the same F::decode / F::pos the kernels use, then compares the values with a reference written from
// the CPU decoders of the engine (rust/engine/src/compute.rs `dequantize`, quant_extra.rs, quant_iq.rs). A mismatch means that a
// layout offset, a unit mapping or a bit extraction of a kernel is wrong. Build and run (see tests/run_decode_check.sh):
//   icpx -fsycl -std=c++17 -I.. decode_check.cpp ../weights.cpp ../device.cpp -o decode_check && ./decode_check
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

#include "common.hpp"
#include "formats.hpp"

using namespace fllm;

namespace {

std::mt19937 g_rng(12345);

float rhalf(const uint8_t* p) { return h2f(static_cast<uint16_t>(p[0] | (p[1] << 8))); }
void put_half(uint8_t* p, float v) {
    const uint16_t h = f2h(v);
    p[0] = static_cast<uint8_t>(h & 0xff);
    p[1] = static_cast<uint8_t>(h >> 8);
}

struct Spec {
    int kind;
    size_t block;       // values per block
    size_t bytes;       // bytes per block
    std::vector<int> halves;  // offsets of f16 scale fields to normalise
};

Spec spec_of(int kind) {
    switch (kind) {
        case 2: return {2, 32, 18, {0}};
        case 3: return {3, 32, 20, {0, 2}};
        case 6: return {6, 32, 22, {0}};
        case 7: return {7, 32, 24, {0, 2}};
        case 8: return {8, 32, 34, {0}};
        case 10: return {10, 256, 84, {80, 82}};
        case 11: return {11, 256, 110, {108}};
        case 12: return {12, 256, 144, {0, 2}};
        case 13: return {13, 256, 176, {0, 2}};
        case 14: return {14, 256, 210, {208}};
        case 20: return {20, 32, 18, {0}};
        case 23: return {23, 256, 136, {0}};
        case 39: return {39, 32, 17, {}};
        default: return {kind, 1, 0, {}};
    }
}

std::vector<uint8_t> random_raw(int kind, size_t rows, size_t cols) {
    std::uniform_int_distribution<int> byte(0, 255);
    std::uniform_real_distribution<float> scale(0.01f, 0.5f), val(-1.f, 1.f);
    std::vector<uint8_t> raw;
    if (kind == 0 || kind == 1 || kind == 30) {
        const size_t w = kind == 0 ? 4 : 2;
        raw.resize(rows * cols * w);
        for (size_t i = 0; i < rows * cols; i++) {
            const float v = val(g_rng);
            if (kind == 0) std::memcpy(&raw[i * 4], &v, 4);
            else if (kind == 1) put_half(&raw[i * 2], v);
            else {
                uint32_t bits;
                std::memcpy(&bits, &v, 4);
                const uint16_t b = static_cast<uint16_t>(bits >> 16);
                raw[i * 2] = static_cast<uint8_t>(b & 0xff);
                raw[i * 2 + 1] = static_cast<uint8_t>(b >> 8);
            }
        }
        return raw;
    }
    const Spec s = spec_of(kind);
    const size_t blocks = rows * cols / s.block;
    raw.resize(blocks * s.bytes);
    for (uint8_t& b : raw) b = static_cast<uint8_t>(byte(g_rng));
    for (size_t b = 0; b < blocks; b++) {
        for (int off : s.halves) put_half(&raw[b * s.bytes + off], scale(g_rng));
        if (kind == 39) raw[b * s.bytes] = static_cast<uint8_t>(118 + byte(g_rng) % 10);
    }
    return raw;
}

// ---- reference decoders, in the loop structure of the Rust code --------------------------------------------------------------
float classic_value(const uint8_t* b, int kind, int j) {
    const float d = rhalf(b);
    const bool affine = kind == 3 || kind == 7, five = kind == 6 || kind == 7;
    const int offset = affine ? 4 : 2;
    const int qoffset = offset + (five ? 4 : 0);
    const int byte = b[qoffset + j % 16];
    int q = j < 16 ? (byte & 15) : (byte >> 4);
    if (five) {
        uint32_t high;
        std::memcpy(&high, b + offset, 4);
        q |= static_cast<int>((high >> j) & 1u) << 4;
    }
    if (affine) return d * static_cast<float>(q) + rhalf(b + 2);
    return d * (static_cast<float>(q) - (five ? 16.0f : 8.0f));
}

void scale_min(const uint8_t* s, int group, int* sc, int* mn) {
    if (group < 4) {
        *sc = s[group] & 63;
        *mn = s[group + 4] & 63;
    } else {
        *sc = (s[group + 4] & 15) | ((s[group - 4] >> 6) << 4);
        *mn = (s[group + 4] >> 4) | ((s[group] >> 6) << 4);
    }
}

std::vector<float> reference(int kind, const std::vector<uint8_t>& raw, size_t count) {
    std::vector<float> out;
    out.reserve(count);
    const Spec s = spec_of(kind);
    if (kind == 0) {
        for (size_t i = 0; i < count; i++) {
            float v;
            std::memcpy(&v, &raw[i * 4], 4);
            out.push_back(v);
        }
        return out;
    }
    if (kind == 1 || kind == 30) {
        for (size_t i = 0; i < count; i++) {
            const uint16_t b = static_cast<uint16_t>(raw[i * 2] | (raw[i * 2 + 1] << 8));
            if (kind == 1) out.push_back(h2f(b));
            else out.push_back(bf2f(b));
        }
        return out;
    }
    for (size_t blk = 0; blk < count / s.block; blk++) {
        const uint8_t* b = &raw[blk * s.bytes];
        switch (kind) {
            case 8: {
                const float d = rhalf(b);
                for (int j = 0; j < 32; j++) out.push_back(d * static_cast<float>(static_cast<int8_t>(b[2 + j])));
                break;
            }
            case 2: case 3: case 6: case 7:
                for (int j = 0; j < 32; j++) out.push_back(classic_value(b, kind, j));
                break;
            case 12: case 13: {
                const float d = rhalf(b), mn_ = rhalf(b + 2);
                const uint8_t* scales = b + 4;
                const uint8_t* qs = b + (kind == 13 ? 48 : 16);
                for (int group = 0; group < 8; group++) {
                    int sc, mn;
                    scale_min(scales, group, &sc, &mn);
                    const uint8_t* chunk = qs + group / 2 * 32;
                    for (int j = 0; j < 32; j++) {
                        int q = group % 2 == 0 ? (chunk[j] & 15) : (chunk[j] >> 4);
                        if (kind == 13) q |= ((b[16 + j] >> group) & 1) << 4;
                        out.push_back(d * static_cast<float>(sc) * static_cast<float>(q) - mn_ * static_cast<float>(mn));
                    }
                }
                break;
            }
            case 14: {
                const float d = rhalf(b + 208);
                float block[256];
                for (int n = 0; n < 2; n++) {
                    const uint8_t* ql = b + n * 64;
                    const uint8_t* qh = b + 128 + n * 32;
                    const int8_t* scales = reinterpret_cast<const int8_t*>(b + 192 + n * 8);
                    for (int j = 0; j < 32; j++) {
                        const int scale = j / 16;
                        const int values[4] = {(ql[j] & 15) | ((qh[j] & 3) << 4), (ql[j + 32] & 15) | (((qh[j] >> 2) & 3) << 4),
                                               (ql[j] >> 4) | (((qh[j] >> 4) & 3) << 4), (ql[j + 32] >> 4) | (((qh[j] >> 6) & 3) << 4)};
                        for (int group = 0; group < 4; group++)
                            block[n * 128 + group * 32 + j] = d * static_cast<float>(scales[scale + group * 2]) * (static_cast<float>(values[group]) - 32.0f);
                    }
                }
                out.insert(out.end(), block, block + 256);
                break;
            }
            case 10: {
                const float d = rhalf(b + 80), dmin = rhalf(b + 82);
                int si = 0;
                for (int n = 0; n < 2; n++) {
                    const uint8_t* q = b + 16 + n * 32;
                    int shift = 0;
                    for (int step = 0; step < 4; step++) {
                        for (int hg = 0; hg < 2; hg++) {
                            const int scale = b[si++];
                            const float ds = d * static_cast<float>(scale & 15), dm = dmin * static_cast<float>(scale >> 4);
                            for (int l = 0; l < 16; l++) out.push_back(ds * static_cast<float>((q[hg * 16 + l] >> shift) & 3) - dm);
                        }
                        shift += 2;
                    }
                }
                break;
            }
            case 11: {
                const float d = rhalf(b + 108);
                int scales[16];
                for (int i = 0; i < 4; i++) {
                    const int hi = b[96 + i], lo = b[96 + 4 + i], packed = b[96 + 8 + i];
                    scales[i] = ((hi & 15) | ((packed & 3) << 4)) - 32;
                    scales[i + 4] = ((lo & 15) | (((packed >> 2) & 3) << 4)) - 32;
                    scales[i + 8] = ((hi >> 4) | (((packed >> 4) & 3) << 4)) - 32;
                    scales[i + 12] = ((lo >> 4) | (((packed >> 6) & 3) << 4)) - 32;
                }
                const uint8_t* q = b + 32;
                const uint8_t* hm = b;
                int scale_idx = 0;
                int mask = 1;
                for (int outer = 0; outer < 2; outer++) {
                    for (int sub = 0; sub < 4; sub++) {
                        const int shift = sub * 2;
                        for (int side = 0; side < 2; side++) {
                            const float ds = d * static_cast<float>(scales[scale_idx++]);
                            for (int l = 0; l < 16; l++) {
                                const int ix = side * 16 + l;
                                const int low = (q[outer * 32 + ix] >> shift) & 3;
                                const int high = (hm[ix] & mask) ? 0 : 4;
                                out.push_back(ds * static_cast<float>(low - high));
                            }
                        }
                        mask <<= 1;
                    }
                }
                break;
            }
            case 20: {
                const float d = rhalf(b);
                for (int j = 0; j < 16; j++) out.push_back(d * static_cast<float>(kIq4Grid[b[2 + j] & 15]));
                for (int j = 0; j < 16; j++) out.push_back(d * static_cast<float>(kIq4Grid[b[2 + j] >> 4]));
                break;
            }
            case 23: {
                const float d = rhalf(b);
                const unsigned scales_h = static_cast<unsigned>(b[2] | (b[3] << 8));
                for (int ib = 0; ib < 8; ib++) {
                    const int low = (b[4 + ib / 2] >> (4 * (ib % 2))) & 0xf;
                    const int ls = low | static_cast<int>(((scales_h >> (2 * ib)) & 3) << 4);
                    const float dl = d * static_cast<float>(ls - 32);
                    for (int j = 0; j < 16; j++) out.push_back(dl * static_cast<float>(kIq4Grid[b[8 + ib * 16 + j] & 15]));
                    for (int j = 0; j < 16; j++) out.push_back(dl * static_cast<float>(kIq4Grid[b[8 + ib * 16 + j] >> 4]));
                }
                break;
            }
            case 39: {
                const int x = b[0];
                const uint32_t bits = x < 2 ? (0x00200000u << x) : (static_cast<uint32_t>(x) - 1) << 23;
                float d;
                std::memcpy(&d, &bits, 4);
                static const float fp4[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
                for (int j = 0; j < 16; j++) out.push_back(fp4[b[1 + j] & 15] * d);
                for (int j = 0; j < 16; j++) out.push_back(fp4[b[1 + j] >> 4] * d);
                break;
            }
            default:
                break;
        }
    }
    return out;
}

// ---- device model: pack, then decode unit by unit --------------------------------------------------------------------------------
template <class F>
bool check_format(int kind, size_t rows, size_t cols, double tol) {
    const std::vector<uint8_t> raw = random_raw(kind, rows, cols);
    const size_t bytes = fllm_sycl_weight_bytes(kind, rows, cols);
    if (bytes == 0) {
        std::printf("FAIL kind %d %zux%zu: weight_bytes is 0\n", kind, rows, cols);
        return false;
    }
    std::vector<uint8_t> packed(bytes + 64, 0xAB);  // guard bytes behind the weight
    if (fllm_sycl_pack_weight(kind, rows, cols, raw.data(), raw.size(), packed.data()) != 0) {
        std::printf("FAIL kind %d %zux%zu: pack: %s\n", kind, rows, cols, fllm_sycl_last_error());
        return false;
    }
    for (size_t i = bytes; i < bytes + 64; i++) {
        if (packed[i] != 0xAB) {
            std::printf("FAIL kind %d %zux%zu: pack wrote past the end\n", kind, rows, cols);
            return false;
        }
    }
    size_t nbp = 0, rb = 0;
    weight_layout(kind, cols, &nbp, &rb);
    if (rb * rows != bytes) {
        std::printf("FAIL kind %d: row bytes inconsistent\n", kind);
        return false;
    }
    const std::vector<float> want = reference(kind, raw, rows * cols);
    if (want.size() != rows * cols) {
        std::printf("FAIL kind %d %zux%zu: reference produced %zu values\n", kind, rows, cols, want.size());
        return false;
    }
    const int nblk = F::TAIL ? static_cast<int>(nbp) : static_cast<int>(cols / F::BLK);
    double worst = 0;
    for (size_t r = 0; r < rows; r++) {
        std::vector<char> seen(cols, 0);
        for (int blk = 0; blk < nblk; blk++) {
            for (int u = 0; u < F::UPB; u++) {
                float w[F::V];
                F::decode(packed.data() + r * rb, static_cast<int>(nbp), blk, u, w);
                for (int i = 0; i < F::V; i++) {
                    const size_t k = static_cast<size_t>(blk) * F::BLK + F::pos(u, i);
                    if (k >= cols) {
                        if (!F::TAIL || w[i] != 0.f) {
                            std::printf("FAIL kind %d: value past the end of the row (k=%zu)\n", kind, k);
                            return false;
                        }
                        continue;
                    }
                    if (seen[k]) {
                        std::printf("FAIL kind %d: position %zu decoded twice (row %zu)\n", kind, k, r);
                        return false;
                    }
                    seen[k] = 1;
                    const double ref = want[r * cols + k];
                    const double err = std::fabs(static_cast<double>(w[i]) - ref) / std::max(1.0, std::fabs(ref));
                    worst = std::max(worst, err);
                    if (err > tol) {
                        std::printf("FAIL kind %d %zux%zu: row %zu col %zu: got %g want %g\n", kind, rows, cols, r, k, static_cast<double>(w[i]), ref);
                        return false;
                    }
                }
            }
        }
        for (size_t k = 0; k < cols; k++) {
            if (!seen[k]) {
                std::printf("FAIL kind %d %zux%zu: row %zu column %zu never decoded\n", kind, rows, cols, r, k);
                return false;
            }
        }
    }
    std::printf("ok   kind %2d %zux%-5zu worst relative error %.2e\n", kind, rows, cols, worst);
    return true;
}

}  // namespace

int main() {
    bool ok = true;
    // exact formats: only floating-point rounding; the converted scales of IQ4_XS and MXFP4 pass through f16
    const double exact = 1e-5, via_half = 3e-3;
    for (size_t rows : {5u, 13u}) {
        ok &= check_format<FF32>(0, rows, 77, exact);
        ok &= check_format<FF32>(0, rows, 4096, exact);
        ok &= check_format<FF16>(1, rows, 77, exact);
        ok &= check_format<FF16>(1, rows, 96, exact);
        ok &= check_format<FBF16>(30, rows, 77, exact);
        for (size_t cols : {96u, 800u}) {
            ok &= check_format<FQ4_0>(2, rows, cols, exact);
            ok &= check_format<FQ4_1>(3, rows, cols, exact);
            ok &= check_format<FQ5_0>(6, rows, cols, exact);
            ok &= check_format<FQ5_1>(7, rows, cols, exact);
            ok &= check_format<FQ8_0>(8, rows, cols, exact);
            ok &= check_format<FIQ4_NL>(20, rows, cols, exact);
            ok &= check_format<FMXFP4>(39, rows, cols, via_half);
        }
        for (size_t cols : {256u, 768u}) {
            ok &= check_format<FQ2_K>(10, rows, cols, exact);
            ok &= check_format<FQ3_K>(11, rows, cols, exact);
            ok &= check_format<FQ4_K>(12, rows, cols, exact);
            ok &= check_format<FQ5_K>(13, rows, cols, exact);
            ok &= check_format<FQ6_K>(14, rows, cols, exact);
            ok &= check_format<FIQ4_NL>(23, rows, cols, via_half);  // IQ4_XS packs as IQ4_NL blocks
        }
    }
    std::printf(ok ? "ALL OK\n" : "FAILED\n");
    return ok ? 0 : 1;
}
