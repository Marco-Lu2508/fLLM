// ExLlamaV2 (`fllm_gguf_probe::exl`), device layout (weights/layout.rs, `pack_exl2`): plane 0 = `rows` rows of `row_words` words, each row the scale nibbles of
// its groups (four bits per group, eight per word) followed by the bitstreams of the groups; plane 1 = two words per block of 32 inputs (stored order): the
// word offset of the block's codes inside a row, and `bits | group << 8`; plane 2 = one f32 per group: `q_scale_max / 256`.
// A group is a multiple of 32 inputs, so a block of 32 lies inside one group and has `bits` words of codes (LSB-first bit stream, a code may straddle two
// words). weight = (code - 2^(bits-1)) * (s + 1)^2 * max / 256 with s the scale nibble of the row and group, as `Exl2View::scale` of the CPU kernel.
// The activations are gathered through the permutation of the blob before the product (`x'[k] = x[perm[k]]`, the prepass of the act-order GPTQ weights
// does it): the stored input k of a row is column k of the gathered activations.
// QK = 8: a block is one chunk, `g = row * nb + chunk` with nb = cols / 8; push constant off3 = row_words.
#define QK 8
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    const uint row = g / p.nb;
    const uint c8 = g - row * p.nb;
    const uint blk = c8 >> 2;
    const uint t0 = w[p.off1 + blk * 2u];
    const uint t1 = w[p.off1 + blk * 2u + 1u];
    const uint bits = t1 & 255u;
    const uint gid = t1 >> 8;
    const uint rowbase = row * p.off3;
    const uint base = rowbase + t0;
    const uint mask = (1u << bits) - 1u;
    const int zero = 1 << (bits - 1u);
    uint c[8];
    [[unroll]] for (uint i = 0; i < 8u; ++i) {
        const uint pos = ((c8 & 3u) * 8u + i) * bits;
        const uint wi = pos >> 5;
        const uint sh = pos & 31u;
        uint v = w[base + wi] >> sh;
        if (sh + bits > 32u) v |= w[base + wi + 1u] << (32u - sh);
        c[i] = uint(int(v & mask) - zero) & 255u;
    }
    p0 = c[0] | (c[1] << 8) | (c[2] << 16) | (c[3] << 24);
    p1 = c[4] | (c[5] << 8) | (c[6] << 16) | (c[7] << 24);
    const uint s = (w[rowbase + (gid >> 3)] >> ((gid & 7u) * 4u)) & 15u;
    ws = float((s + 1u) * (s + 1u)) * uintBitsToFloat(w[p.off2 + gid]);
    wm = 0.0;
}
