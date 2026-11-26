// FLLM Vulkan shaders, package 3: element access to the paged KV pools (DeviceKvCache, backend/mod.rs) from `uint[]` storage buffers.
// The pools are arrays of 16-bit words; a layer may start at an ODD 16-bit word, so every access goes through 32-bit words and a lane select.
//   f16 (kv_kind 1):  value (slot, c) = binary16 at 16-bit position  base + slot * kv + c.
//   q8  (kv_kind 2):  value (slot, c) = int8 at byte (2 * base + slot * kv + c)  *  binary16 scale at 16-bit position
//                     base + scale_base + slot * (kv / group) + c / group.
// `kv` = kv_heads * dim of that pool; base / scale_base / group are push constants (base = layer start in 16-bit words relative to the bound window).
#ifndef FLLM_OPS_KV
#define FLLM_OPS_KV

// binary16 number at 16-bit position i of the pool
#define KV_HALF(pool, i) (unpackHalf2x16((pool)[(i) >> 1u])[(i) & 1u])
// int8 number at byte b of the pool (little endian inside the words), sign extended
#define KV_I8(pool, b) float(int((pool)[(b) >> 2u] << (24u - (((b) & 3u) << 3))) >> 24)
// the four int8 lanes of a word
vec4 kv_s8x4(uint w) {
    return vec4(float(int(w << 24) >> 24), float(int(w << 16) >> 24), float(int(w << 8) >> 24), float(int(w) >> 24));
}

// ---- generic formats (kv_format.rs: bf16, f32, fp8, int8, q5_0 / q5_1, q4_0 / q4_1, iq4_nl, nvfp4; keys and values may differ) ----
// `fmt` = KvFormat code 1..13. Regions of a layer in 16-bit words from `base`: codes at 0, `hi` (5-bit formats: the fifth bit, 16 values per word),
// `sc` (one binary16 scale per `group` values; nvfp4: one E4M3 byte per 16 values), `mn` (binary16 minima of q4_1 / q5_1). n = slot * kv + channel.
#define KV_U16(pool, i) ((((pool)[(i) >> 1u]) >> (((i) & 1u) << 4u)) & 0xFFFFu)
#define KV_U8(pool, b) ((((pool)[(b) >> 2u]) >> (((b) & 3u) << 3u)) & 0xFFu)

const float KV_IQ4NL[16] = float[16](-127.0, -104.0, -83.0, -65.0, -49.0, -35.0, -22.0, -10.0, 1.0, 13.0, 25.0, 38.0, 53.0, 69.0, 89.0, 113.0);
const float KV_FP4[8] = float[8](0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0);
const float KV_FP4_MID[7] = float[7](0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0);

// FP8 code to float (sign bit 7): e4m3 = (3 mantissa bits, bias 7), e5m2 = (2, 15)
float kv_fp8(uint b, uint mbits, int bias) {
    const uint e = (b & 0x7Fu) >> mbits;
    const uint m = b & ((1u << mbits) - 1u);
    float v;
    if (e == 0u) {
        v = float(m) * uintBitsToFloat(uint(1 - bias - int(mbits) + 127) << 23);
    } else {
        v = uintBitsToFloat((uint(int(e) - bias + 127) << 23) | (m << (23u - mbits)));
    }
    return (b & 0x80u) != 0u ? -v : v;
}

// float to FP8 code, round to nearest even, saturating (kv_format.rs `f32_to_fp8`)
uint kv_to_fp8(float x, uint mbits, int bias, uint max_code) {
    const uint sign = (floatBitsToUint(x) >> 31) != 0u ? 0x80u : 0u;
    const float a = abs(x);
    if (isnan(a)) return 0u;
    const float min_normal = uintBitsToFloat(uint(1 - bias + 127) << 23);
    uint code;
    if (a < min_normal) {
        const float scaled = a * uintBitsToFloat(uint(bias - 1 + int(mbits) + 127) << 23);
        code = uint(roundEven(scaled));
    } else {
        const uint bits = floatBitsToUint(a);
        const uint shift = 23u - mbits;
        const uint r = (bits + ((1u << (shift - 1u)) - 1u) + ((bits >> shift) & 1u)) >> shift;
        code = r - (uint(127 - bias) << mbits);
    }
    return sign | min(code, max_code);
}
#endif
