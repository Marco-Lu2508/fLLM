// Decoder of the generic KV formats for one pool: include after defining KV_GEN_POOL (the storage array) and KV_GEN_NAME (the function), then
// the file undefines both. Layout and formulas: _ops_kv.glsl, kv_format.rs (`dequant_code`).
float KV_GEN_NAME(uint fmt, uint base, uint hi, uint sc, uint mn, uint group, uint n) {
    if (fmt == 1u) return KV_HALF(KV_GEN_POOL, base + n);
    if (fmt == 3u) return uintBitsToFloat(KV_U16(KV_GEN_POOL, base + n) << 16);
    if (fmt == 13u) return uintBitsToFloat(KV_U16(KV_GEN_POOL, base + 2u * n) | (KV_U16(KV_GEN_POOL, base + 2u * n + 1u) << 16));
    if (fmt == 2u || fmt == 6u) {
        return KV_I8(KV_GEN_POOL, base * 2u + n) * KV_HALF(KV_GEN_POOL, base + sc + n / group);
    }
    if (fmt == 4u) return kv_fp8(KV_U8(KV_GEN_POOL, base * 2u + n), 3u, 7) * KV_HALF(KV_GEN_POOL, base + sc + n / group);
    if (fmt == 5u) return kv_fp8(KV_U8(KV_GEN_POOL, base * 2u + n), 2u, 15);
    // 4-bit codes (q4_0, q4_1, iq4_nl, nvfp4) and the low four bits of the 5-bit ones: value i in bits 4 * (i % 4) of word i / 4
    uint code = (KV_U16(KV_GEN_POOL, base + (n >> 2u)) >> ((n & 3u) << 2u)) & 15u;
    if (fmt == 12u) {
        const float d = kv_fp8(KV_U8(KV_GEN_POOL, (base + sc) * 2u + (n >> 4u)), 3u, 7);
        const float v = KV_FP4[code & 7u] * d;
        return (code & 8u) != 0u ? -v : v;
    }
    const uint g = n / group;
    if (fmt == 7u || fmt == 8u) code |= ((KV_U16(KV_GEN_POOL, base + hi + (n >> 4u)) >> (n & 15u)) & 1u) << 4u;
    const float s = KV_HALF(KV_GEN_POOL, base + sc + g);
    if (fmt == 9u) return (float(code) - 8.0) * s;
    if (fmt == 7u) return (float(code) - 16.0) * s;
    if (fmt == 11u) return KV_IQ4NL[code] * s;
    return float(code) * s + KV_HALF(KV_GEN_POOL, base + mn + g);  // q4_1, q5_1
}
#undef KV_GEN_POOL
#undef KV_GEN_NAME
