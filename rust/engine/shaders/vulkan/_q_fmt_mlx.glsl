// MLX affine group quantization (`fllm_gguf_probe::MlxAffine`, MLX_BITS = 2, 3, 4, 5, 6 or 8), device layout: blocks of 32 codes whatever the group size:
// plane 0 = MLX_BITS words per block of 32 codes (the little-endian bit stream of the file: code i at bit i * MLX_BITS of the block, a code of 3, 5 or 6 bits may
// straddle two words), plane 1 = one word per *group* with the scale in the low and the bias in the high 16 bits (binary16 or bfloat16 as the checkpoint has them).
// Push constants: off2 = (group / 32) (blocks per scale) | bf16 << 16. weight = code * scale + bias.
// Chunk model: codes below 128 are the int8 lanes themselves (ws = scale, wm = bias); 8-bit codes become code - 128 (flip bit 7) with wm = bias + 128 * scale.
#define QK 32
#define HAS_WM 1
#define MLX_MASK ((1u << MLX_BITS) - 1u)
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    const uint base = g * MLX_BITS;
    uint c[8];
    [[unroll]] for (uint i = 0; i < 8u; ++i) {
        const uint pos = (cin * 8u + i) * MLX_BITS;
        const uint wi = pos >> 5;
        const uint sh = pos & 31u;
        uint v = w[base + wi] >> sh;
        if (sh + MLX_BITS > 32u) v |= w[base + wi + 1u] << (32u - sh);
        c[i] = v & MLX_MASK;
    }
    p0 = c[0] | (c[1] << 8) | (c[2] << 16) | (c[3] << 24);
    p1 = c[4] | (c[5] << 8) | (c[6] << 16) | (c[7] << 24);
    const uint sb = w[p.off1 + g / (p.off2 & 0xffffu)];
    float scale, bias;
    if ((p.off2 >> 16) != 0u) {
        scale = uintBitsToFloat(sb << 16);
        bias = uintBitsToFloat(sb & 0xffff0000u);
    } else {
        const vec2 sbv = unpackHalf2x16(sb);
        scale = sbv.x;
        bias = sbv.y;
    }
    ws = scale;
    wm = bias;
#if MLX_BITS == 8
    p0 ^= 0x80808080u;
    p1 ^= 0x80808080u;
    wm = bias + 128.0 * scale;
#endif
}
