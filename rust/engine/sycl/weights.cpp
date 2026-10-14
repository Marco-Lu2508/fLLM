// SPDX-License-Identifier: Apache-2.0
// fLLM Intel GPU backend (SYCL / oneAPI): the device layouts of the weight formats and the host-side repacking from the GGUF bytes.
//
// The kernels (formats.hpp) read 8- and 16-byte units, so a row keeps its arrays apart and 16-byte aligned:
//   F32 / F16 / BF16   values, the row padded with zeros to a multiple of 8
//   Q8_0               [d: f16 x nbp][quants: i8 x 32 x nbp]                       nbp = blocks rounded up to a multiple of 8
//   Q4_0 / IQ4_NL / MXFP4  [d: f16 x nbp][quants: 16 bytes x nbp]                  (MXFP4: d = 2^(e-128), IQ4_XS -> 8 IQ4_NL blocks)
//   Q4_1               [d][m][quants]      Q5_0 [d][qh: u32][quants]      Q5_1 [d][m][qh][quants]
//   Q4_K 144, Q5_K 176, Q6_K 210 -> 224, Q2_K 84 -> 96, Q3_K 110 -> 112 bytes per super-block (zero padded)
// Verification level: L0 (host code, compiled; the layout is checked against the decoders in formats.hpp by reading only).
#include <cmath>
#include <cstring>
#include <thread>

#include "common.hpp"
#include "formats.hpp"

namespace fllm {

int device_kind(int kind) { return kind == 23 ? 20 : kind; }

bool weight_layout(int kind, size_t cols, size_t* nbp, size_t* row_bytes) {
    if (kind == 23 && cols % 256 != 0) return false;  // IQ4_XS super-blocks
    kind = device_kind(kind);
    size_t block = 0, bytes = 0;
    bool padded8 = false;
    switch (kind) {
        case 0: *nbp = ceil_div(cols, 8); *row_bytes = *nbp * 32; return cols > 0;
        case 1:
        case 30: *nbp = ceil_div(cols, 8); *row_bytes = *nbp * 16; return cols > 0;
        case 2: block = 32; bytes = 18; padded8 = true; break;
        case 3: block = 32; bytes = 20; padded8 = true; break;
        case 6: block = 32; bytes = 22; padded8 = true; break;
        case 7: block = 32; bytes = 24; padded8 = true; break;
        case 8: block = 32; bytes = 34; padded8 = true; break;
        case 20: block = 32; bytes = 18; padded8 = true; break;
        case 39: block = 32; bytes = 18; padded8 = true; break;
        case 10: block = 256; bytes = 96; break;
        case 11: block = 256; bytes = 112; break;
        case 12: block = 256; bytes = 144; break;
        case 13: block = 256; bytes = 176; break;
        case 14: block = 256; bytes = 224; break;
        default: return false;
    }
    if (cols == 0 || cols % block != 0) return false;
    const size_t nb = cols / block;
    *nbp = padded8 ? round_up(nb, 8) : nb;
    *row_bytes = *nbp * bytes;
    return true;
}

size_t raw_row_bytes(int kind, size_t cols) {
    size_t block = 0, bytes = 0;
    switch (kind) {
        case 0: return cols * 4;
        case 1:
        case 30: return cols * 2;
        case 2: block = 32; bytes = 18; break;
        case 3: block = 32; bytes = 20; break;
        case 6: block = 32; bytes = 22; break;
        case 7: block = 32; bytes = 24; break;
        case 8: block = 32; bytes = 34; break;
        case 10: block = 256; bytes = 84; break;
        case 11: block = 256; bytes = 110; break;
        case 12: block = 256; bytes = 144; break;
        case 13: block = 256; bytes = 176; break;
        case 14: block = 256; bytes = 210; break;
        case 20: block = 32; bytes = 18; break;
        case 23: block = 256; bytes = 136; break;
        case 39: block = 32; bytes = 17; break;
        default: return 0;
    }
    return cols % block == 0 ? cols / block * bytes : 0;
}

namespace {

float half_to_float(uint16_t b) { return static_cast<float>(sycl::bit_cast<sycl::half>(b)); }
uint16_t float_to_half(float v) { return sycl::bit_cast<uint16_t>(sycl::half(v)); }

// Converts one row of the GGUF encoding `src` (raw_row_bytes) to the device layout `dst` (row_bytes, zero filled).
void pack_row(int kind, size_t cols, size_t nbp, const uint8_t* src, uint8_t* dst) {
    switch (kind) {
        case 0:
        case 1:
        case 30:
            std::memcpy(dst, src, raw_row_bytes(kind, cols));
            return;
        case 2:
        case 3:
        case 6:
        case 7: {
            const bool affine = kind == 3 || kind == 7, five = kind == 6 || kind == 7;
            const size_t block_bytes = 2 + (affine ? 2 : 0) + (five ? 4 : 0) + 16;
            const size_t m_off = nbp * 2, qh_off = nbp * 2 * (affine ? 2 : 1), qs_off = qh_off + (five ? nbp * 4 : 0);
            for (size_t b = 0; b < cols / 32; b++) {
                const uint8_t* blk = src + b * block_bytes;
                std::memcpy(dst + b * 2, blk, 2);
                size_t at = 2;
                if (affine) {
                    std::memcpy(dst + m_off + b * 2, blk + at, 2);
                    at += 2;
                }
                if (five) {
                    std::memcpy(dst + qh_off + b * 4, blk + at, 4);
                    at += 4;
                }
                std::memcpy(dst + qs_off + b * 16, blk + at, 16);
            }
            return;
        }
        case 8:
            for (size_t b = 0; b < cols / 32; b++) {
                std::memcpy(dst + b * 2, src + b * 34, 2);
                std::memcpy(dst + nbp * 2 + b * 32, src + b * 34 + 2, 32);
            }
            return;
        case 20:
            for (size_t b = 0; b < cols / 32; b++) {
                std::memcpy(dst + b * 2, src + b * 18, 2);
                std::memcpy(dst + nbp * 2 + b * 16, src + b * 18 + 2, 16);
            }
            return;
        case 23:  // IQ4_XS super-block {d: f16, scales_h: u16, scales_l: [u8; 4], qs: [u8; 128]} as eight IQ4_NL blocks
            for (size_t sb = 0; sb < cols / 256; sb++) {
                const uint8_t* s = src + sb * 136;
                const float d = half_to_float(static_cast<uint16_t>(s[0] | (s[1] << 8)));
                const unsigned scales_h = static_cast<unsigned>(s[2] | (s[3] << 8));
                for (size_t ib = 0; ib < 8; ib++) {
                    const unsigned lo = (s[4 + ib / 2] >> (4 * (ib % 2))) & 0xf;
                    const unsigned hi = (scales_h >> (2 * ib)) & 3;
                    const int ls = static_cast<int>(lo | (hi << 4)) - 32;
                    const size_t b = sb * 8 + ib;
                    const uint16_t h = float_to_half(d * static_cast<float>(ls));
                    std::memcpy(dst + b * 2, &h, 2);
                    std::memcpy(dst + nbp * 2 + b * 16, s + 8 + 16 * ib, 16);
                }
            }
            return;
        case 39:  // MXFP4: [e8m0][16 bytes]; the 16-entry grid holds twice the E2M1 values, so d = 2^(e - 128)
            for (size_t b = 0; b < cols / 32; b++) {
                const uint16_t h = float_to_half(std::ldexp(1.0f, static_cast<int>(src[b * 17]) - 128));
                std::memcpy(dst + b * 2, &h, 2);
                std::memcpy(dst + nbp * 2 + b * 16, src + b * 17 + 1, 16);
            }
            return;
        case 10:
        case 11:
        case 12:
        case 13:
        case 14: {
            const size_t raw = kind == 10 ? 84 : kind == 11 ? 110 : kind == 12 ? 144 : kind == 13 ? 176 : 210;
            const size_t padded = kind == 10 ? 96 : kind == 11 ? 112 : kind == 12 ? 144 : kind == 13 ? 176 : 224;
            for (size_t sb = 0; sb < cols / 256; sb++) std::memcpy(dst + sb * padded, src + sb * raw, raw);
            return;
        }
        default:
            return;
    }
}

}  // namespace

}  // namespace fllm

extern "C" {

size_t fllm_sycl_weight_bytes(int kind, size_t rows, size_t cols) {
    size_t nbp = 0, rb = 0;
    if (!fllm::weight_layout(kind, cols, &nbp, &rb)) return 0;
    return rows * rb;
}

size_t fllm_sycl_row_bytes(int kind, size_t cols) {
    size_t nbp = 0, rb = 0;
    return fllm::weight_layout(kind, cols, &nbp, &rb) ? rb : 0;
}

size_t fllm_sycl_raw_bytes(int kind, size_t rows, size_t cols) { return rows * fllm::raw_row_bytes(kind, cols); }

int fllm_sycl_pack_weight(int kind, size_t rows, size_t cols, const void* raw, size_t raw_bytes, void* out) {
    return fllm::guarded("pack_weight", [&] {
        size_t nbp = 0, rb = 0;
        if (!fllm::weight_layout(kind, cols, &nbp, &rb)) throw std::runtime_error("unsupported weight kind " + std::to_string(kind) + " or width " + std::to_string(cols));
        const size_t rrb = fllm::raw_row_bytes(kind, cols);
        if (rrb == 0 || raw_bytes != rows * rrb) throw std::runtime_error("weight has " + std::to_string(raw_bytes) + " bytes, expected " + std::to_string(rows * rrb));
        std::memset(out, 0, rows * rb);
        const uint8_t* src = static_cast<const uint8_t*>(raw);
        uint8_t* dst = static_cast<uint8_t*>(out);
        const size_t threads = std::max<size_t>(1, std::min<size_t>(rows / 64 + 1, std::min<size_t>(16, std::thread::hardware_concurrency())));
        if (threads == 1) {
            for (size_t r = 0; r < rows; r++) fllm::pack_row(kind, cols, nbp, src + r * rrb, dst + r * rb);
            return;
        }
        std::vector<std::thread> pool;
        for (size_t t = 0; t < threads; t++) {
            pool.emplace_back([=] {
                for (size_t r = t; r < rows; r += threads) fllm::pack_row(kind, cols, nbp, src + r * rrb, dst + r * rb);
            });
        }
        for (std::thread& th : pool) th.join();
    });
}

}  // extern "C"
