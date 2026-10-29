// bitsandbytes NF4 / FP4 (`fllm_gguf_probe::bnb`), device layout: plane 0 = the code bytes of a block as in the file (QK / 2 bytes; byte j holds element 2j in the
// high and element 2j + 1 in the low nibble), plane 1 = the f32 absmax of every block. weight = code_book[nibble] * absmax with the code book of the family
// (FMT_NF4 or FMT_FP4; the entries are the f32 bit patterns of `bnb::NF4_CODE` / `bnb::FP4_CODE`). Float chunk decoder: only the f32 shaders (`_s`) exist.
// BNB_QK = 64 or 128.
#define QK BNB_QK
#define FLOAT_CHUNK 1
#if defined(FMT_NF4)
const uint BOOK[16] = uint[16](0xbf800000u, 0xbf3239b1u, 0xbf066b30u, 0xbeca32a0u, 0xbe91a24du, 0xbe3d353fu, 0xbdba7871u, 0x00000000u,
                               0x3da2faffu, 0x3e24cae3u, 0x3e7c04ddu, 0x3ead033au, 0x3ee1a4b8u, 0x3f1007abu, 0x3f3913b3u, 0x3f800000u);
#else
const uint BOOK[16] = uint[16](0x00000000u, 0x3baaaaabu, 0x3f2aaaabu, 0x3f800000u, 0x3eaaaaabu, 0x3f000000u, 0x3e2aaaabu, 0x3e800000u,
                               0x80000000u, 0xbbaaaaabu, 0xbf2aaaabu, 0xbf800000u, 0xbeaaaaabu, 0xbf000000u, 0xbe2aaaabu, 0xbe800000u);
#endif

void wchunk_f(uint g, uint cin, out float v[8]) {
    const uint codes = w[g * (QK / 8u) + cin];
    const float s = uintBitsToFloat(w[p.off1 + g]);
    [[unroll]] for (uint k = 0; k < 4u; ++k) {
        const uint byte = (codes >> (8u * k)) & 255u;
        v[2u * k] = uintBitsToFloat(BOOK[byte >> 4]) * s;
        v[2u * k + 1u] = uintBitsToFloat(BOOK[byte & 15u]) * s;
    }
}
