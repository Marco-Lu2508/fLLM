// FP8B (block-wise FP8, `fllm_gguf_probe::Fp8bLayout`), device layout: plane 0 = the E4M3FN codes, row-major (one byte per weight; a chunk of 8 weights is
// two words), plane 1 = one f32 scale per (row, tile column): the scale of the tile of the file, repeated for the rows of the tile so that a row never needs the
// tile rows (and a slice of rows is self-contained). weight = e4m3(code) * scale, exactly the CPU kernel (cpu_fp8.rs).
// The decoder is a *float* chunk decoder (FLOAT_CHUNK): E4M3 values are not small integers, so only the f32 shaders (`_s`) exist for this format.
// QK = 8: a block is one chunk, `g = row * nb + chunk`; push constant off2 = block_cols of the tile (a multiple of 8: the upload checks it).
#define QK 8
#define FLOAT_CHUNK 1

// value of an E4M3FN code, as `decode_code` of cpu_fp8.rs: exponent and mantissa moved into an f32 (bias 127 instead of 7), subnormals are mantissa * 2^-9,
// 0x7f / 0xff are NaN
float e4m3_value(uint code) {
    uint mag = code & 0x7fu;
    float normal = uintBitsToFloat((mag << 20) + (120u << 23));
    float subnormal = float(mag & 7u) * (1.0 / 512.0);
    float value = (mag >> 3) == 0u ? subnormal : normal;
    if (mag == 0x7fu) value = uintBitsToFloat(0x7fc00000u);
    return uintBitsToFloat(floatBitsToUint(value) | ((code & 0x80u) << 24));
}

void wchunk_f(uint g, uint cin, out float v[8]) {
    const uint row = g / p.nb;
    const uint col = (g - row * p.nb) * 8u;
    const uint tile_cols = (p.cols + p.off2 - 1u) / p.off2;
    const float s = uintBitsToFloat(w[p.off1 + row * tile_cols + col / p.off2]);
    const uint a = w[g * 2u];
    const uint b = w[g * 2u + 1u];
    [[unroll]] for (uint k = 0; k < 4u; ++k) {
        v[k] = e4m3_value((a >> (8u * k)) & 255u) * s;
        v[4u + k] = e4m3_value((b >> (8u * k)) & 255u) * s;
    }
}
