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

#endif
