// FLLM Vulkan shaders, package 3: decoders of the planar device layout of quantized weights, for the embedding lookup and the expert matrix-vector
// product (moe_linear). The layout is the one of the quantized-linear shaders (package 2, INTERFACE_QUANT.md: the GGUF blocks are split into
// planes, each plane starting at a word offset; plane 0 at word 0). The decoders below repeat that layout; they are NOT shared with the
// `_q_fmt_*.glsl` files of package 2 so that this package builds on its own. If a layout there changes, change it here too.
//
// The includer provides, before including this file:
//   WQ_W(i)     the 32-bit word i of the weight buffer (a macro, so the same decoders serve a bound buffer and a buffer reference)
//   WQ_OFF1, WQ_OFF2, WQ_OFF3   word offsets of planes 1..3 (push constants)
//   exactly one of FMT_Q4_0 FMT_Q4_1 FMT_Q5_0 FMT_Q5_1 FMT_Q8_0 FMT_Q2_K FMT_Q3_K FMT_Q4_K FMT_Q5_K FMT_Q6_K FMT_IQ4_NL FMT_MXFP4
// This file defines QK (weights per block), HAS_WM and
//   void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm)
// g = global block index (row * blocks_per_row + block), cin = chunk (8 consecutive weights) inside the block: eight small signed integers q[0..7] as
// int8 lanes (lane i of p0 for i < 4, lane i - 4 of p1 otherwise) and two floats with  weight[i] = ws * q[i] + wm.
#ifndef FLLM_OPS_WQ
#define FLLM_OPS_WQ

// binary16 number i of a plane of packed halves starting at word `base`
float wq_f16at(uint base, uint i) {
    return unpackHalf2x16(WQ_W(base + (i >> 1)))[i & 1u];
}

// four int8 lanes -> vec4
vec4 wq_s8x4(uint v) {
    return vec4((ivec4(int(v)) << ivec4(24, 16, 8, 0)) >> 24);
}

// lane-wise (x - c) for four bytes x < 128 and c <= 128, as two's complement int8 lanes
uint wq_sub8(uint x, uint c) {
    return ((x | 0x80808080u) - c * 0x01010101u) ^ 0x80808080u;
}

#if defined(FMT_Q4_0)
// Q4_0: plane 0 = qs (16 bytes per block, as in the file), plane 1 = d (binary16 per block). value j (0..15) = low nibble of byte j, j+16 = high nibble; d * (q - 8).
#define QK 32
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    uint b = g * 4u + ((cin & 1u) << 1);
    uint sh = (cin >> 1) << 2;
    p0 = wq_sub8((WQ_W(b) >> sh) & 0x0F0F0F0Fu, 8u);
    p1 = wq_sub8((WQ_W(b + 1u) >> sh) & 0x0F0F0F0Fu, 8u);
    ws = wq_f16at(WQ_OFF1, g);
    wm = 0.0;
}
#elif defined(FMT_Q4_1)
// Q4_1: plane 0 = qs, plane 1 = one word per block (d | m << 16). weight = d * q + m.
#define QK 32
#define HAS_WM 1
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    uint b = g * 4u + ((cin & 1u) << 1);
    uint sh = (cin >> 1) << 2;
    p0 = (WQ_W(b) >> sh) & 0x0F0F0F0Fu;
    p1 = (WQ_W(b + 1u) >> sh) & 0x0F0F0F0Fu;
    vec2 dm = unpackHalf2x16(WQ_W(WQ_OFF1 + g));
    ws = dm.x;
    wm = dm.y;
}
#elif defined(FMT_Q5_0)
// Q5_0: plane 0 = qs, plane 1 = qh (one word per block), plane 2 = d (binary16). q = nibble | bit j of qh << 4; weight = d * (q - 16).
#define QK 32
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    uint b = g * 4u + ((cin & 1u) << 1);
    uint sh = (cin >> 1) << 2;
    uint qh = WQ_W(WQ_OFF1 + g) >> (cin << 3);
    uint h0 = ((qh & 15u) * 0x00204081u) & 0x01010101u;
    uint h1 = (((qh >> 4) & 15u) * 0x00204081u) & 0x01010101u;
    p0 = wq_sub8(((WQ_W(b) >> sh) & 0x0F0F0F0Fu) | (h0 << 4), 16u);
    p1 = wq_sub8(((WQ_W(b + 1u) >> sh) & 0x0F0F0F0Fu) | (h1 << 4), 16u);
    ws = wq_f16at(WQ_OFF2, g);
    wm = 0.0;
}
#elif defined(FMT_Q5_1)
// Q5_1: plane 0 = qs, plane 1 = qh (one word per block), plane 2 = one word per block (d | m << 16). weight = d * q + m with q = nibble | bit j of qh << 4.
#define QK 32
#define HAS_WM 1
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    uint b = g * 4u + ((cin & 1u) << 1);
    uint sh = (cin >> 1) << 2;
    uint qh = WQ_W(WQ_OFF1 + g) >> (cin << 3);
    uint h0 = ((qh & 15u) * 0x00204081u) & 0x01010101u;
    uint h1 = (((qh >> 4) & 15u) * 0x00204081u) & 0x01010101u;
    p0 = ((WQ_W(b) >> sh) & 0x0F0F0F0Fu) | (h0 << 4);
    p1 = ((WQ_W(b + 1u) >> sh) & 0x0F0F0F0Fu) | (h1 << 4);
    vec2 dm = unpackHalf2x16(WQ_W(WQ_OFF2 + g));
    ws = dm.x;
    wm = dm.y;
}
#elif defined(FMT_Q8_0)
// Q8_0: plane 0 = qs (32 int8 per block), plane 1 = d (binary16). weight = d * q.
#define QK 32
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    uint b = g * 8u + (cin << 1);
    p0 = WQ_W(b);
    p1 = WQ_W(b + 1u);
    ws = wq_f16at(WQ_OFF1, g);
    wm = 0.0;
}
#elif defined(FMT_Q2_K)
// Q2_K: plane 0 = qs (64 bytes per block), plane 1 = scales (16 bytes: low nibble scale, high nibble min), plane 2 = one word per block (d | dmin << 16).
// 16 sub-blocks of 16 values: sub-block si = n*8 + sub*2 + hg; q = (qs[n*32 + hg*16 + l] >> (2*sub)) & 3; weight = d * (scale & 15) * q - dmin * (scale >> 4).
#define QK 256
#define HAS_WM 1
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    uint n = cin >> 4;
    uint r = cin & 15u;
    uint sub = r >> 2;
    uint j4 = r & 3u;
    uint b = g * 16u + (n << 3) + (j4 << 1);
    uint sh = sub << 1;
    p0 = (WQ_W(b) >> sh) & 0x03030303u;
    p1 = (WQ_W(b + 1u) >> sh) & 0x03030303u;
    uint si = (n << 3) + (sub << 1) + (j4 >> 1);
    uint sc = (WQ_W(WQ_OFF1 + g * 4u + (si >> 2)) >> ((si & 3u) << 3)) & 255u;
    vec2 dm = unpackHalf2x16(WQ_W(WQ_OFF2 + g));
    ws = dm.x * float(sc & 15u);
    wm = -dm.y * float(sc >> 4);
}
#elif defined(FMT_Q3_K)
// Q3_K: plane 0 = qs (64 bytes), plane 1 = hmask (32 bytes), plane 2 = the 16 sub-block scales decoded to int8 (scale - 32), plane 3 = d as float32.
// q = ((qs[outer*32 + ix] >> 2*sub) & 3) + 4 * bit(outer*4 + sub) of hmask[ix] - 4; sub-block scale index outer*8 + sub*2 + side; weight = d * scale * q.
#define QK 256
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    uint outer = cin >> 4;
    uint r = cin & 15u;
    uint sub = r >> 2;
    uint j4 = r & 3u;
    uint b = g * 16u + (outer << 3) + (j4 << 1);
    uint sh = sub << 1;
    uint hb = (outer << 2) + sub;
    uint hm = g * 8u + (j4 << 1);
    p0 = wq_sub8(((WQ_W(b) >> sh) & 0x03030303u) | (((WQ_W(WQ_OFF1 + hm) >> hb) & 0x01010101u) << 2), 4u);
    p1 = wq_sub8(((WQ_W(b + 1u) >> sh) & 0x03030303u) | (((WQ_W(WQ_OFF1 + hm + 1u) >> hb) & 0x01010101u) << 2), 4u);
    uint si = (outer << 3) + (sub << 1) + (j4 >> 1);
    int sc = bitfieldExtract(int(WQ_W(WQ_OFF2 + g * 4u + (si >> 2))), int((si & 3u) << 3), 8);
    ws = uintBitsToFloat(WQ_W(WQ_OFF3 + g)) * float(sc);
    wm = 0.0;
}
#elif defined(FMT_Q4_K)
// Q4_K: plane 0 = qs (128 bytes), plane 1 = 16 bytes: eight decoded 6-bit scales then eight decoded 6-bit mins, plane 2 = one word per block (d | dmin << 16).
// Group g8 (32 values) uses bytes (g8/2)*32 .. +31 of qs, low nibbles for even g8, high nibbles for odd g8; weight = d * sc[g8] * q - dmin * mn[g8].
#define QK 256
#define HAS_WM 1
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    uint g8 = cin >> 2;
    uint j4 = cin & 3u;
    uint b = g * 32u + ((g8 >> 1) << 3) + (j4 << 1);
    uint sh = (g8 & 1u) << 2;
    p0 = (WQ_W(b) >> sh) & 0x0F0F0F0Fu;
    p1 = (WQ_W(b + 1u) >> sh) & 0x0F0F0F0Fu;
    uint q = WQ_OFF1 + g * 4u + (g8 >> 2);
    uint lane = (g8 & 3u) << 3;
    float sc = float((WQ_W(q) >> lane) & 255u);
    float mn = float((WQ_W(q + 2u) >> lane) & 255u);
    vec2 dm = unpackHalf2x16(WQ_W(WQ_OFF2 + g));
    ws = dm.x * sc;
    wm = -dm.y * mn;
}
#elif defined(FMT_Q5_K)
// Q5_K: plane 0 = qs (128 bytes), plane 1 = qh (32 bytes), plane 2 = scales/mins as Q4_K, plane 3 = one word per block (d | dmin << 16).
// As Q4_K, with the fifth bit of value (g8, j) taken from bit g8 of qh[j].
#define QK 256
#define HAS_WM 1
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    uint g8 = cin >> 2;
    uint j4 = cin & 3u;
    uint b = g * 32u + ((g8 >> 1) << 3) + (j4 << 1);
    uint sh = (g8 & 1u) << 2;
    uint h = g * 8u + (j4 << 1);
    p0 = ((WQ_W(b) >> sh) & 0x0F0F0F0Fu) | (((WQ_W(WQ_OFF1 + h) >> g8) & 0x01010101u) << 4);
    p1 = ((WQ_W(b + 1u) >> sh) & 0x0F0F0F0Fu) | (((WQ_W(WQ_OFF1 + h + 1u) >> g8) & 0x01010101u) << 4);
    uint q = WQ_OFF2 + g * 4u + (g8 >> 2);
    uint lane = (g8 & 3u) << 3;
    float sc = float((WQ_W(q) >> lane) & 255u);
    float mn = float((WQ_W(q + 2u) >> lane) & 255u);
    vec2 dm = unpackHalf2x16(WQ_W(WQ_OFF3 + g));
    ws = dm.x * sc;
    wm = -dm.y * mn;
}
#elif defined(FMT_Q6_K)
// Q6_K: plane 0 = ql (128 bytes), plane 1 = qh (64 bytes), plane 2 = the 16 int8 scales, plane 3 = d as float32.
// Half n (128 values), group grp (32 values), j in 0..31: low 4 bits = low (grp < 2) or high (grp >= 2) nibble of ql[n*64 + (grp&1)*32 + j], high 2 bits =
// bits 2*grp.. of qh[n*32 + j]; q = value - 32; scale index n*8 + j/16 + grp*2; weight = d * scale * q.
#define QK 256
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    uint n = cin >> 4;
    uint r = cin & 15u;
    uint grp = r >> 2;
    uint j4 = r & 3u;
    uint b = g * 32u + (n << 4) + ((grp & 1u) << 3) + (j4 << 1);
    uint sh = (grp >> 1) << 2;
    uint h = g * 16u + (n << 3) + (j4 << 1);
    uint hs = grp << 1;
    p0 = wq_sub8(((WQ_W(b) >> sh) & 0x0F0F0F0Fu) | (((WQ_W(WQ_OFF1 + h) >> hs) & 0x03030303u) << 4), 32u);
    p1 = wq_sub8(((WQ_W(b + 1u) >> sh) & 0x0F0F0F0Fu) | (((WQ_W(WQ_OFF1 + h + 1u) >> hs) & 0x03030303u) << 4), 32u);
    uint si = (n << 3) + (j4 >> 1) + (grp << 1);
    int sc = bitfieldExtract(int(WQ_W(WQ_OFF2 + g * 4u + (si >> 2))), int((si & 3u) << 3), 8);
    ws = uintBitsToFloat(WQ_W(WQ_OFF3 + g)) * float(sc);
    wm = 0.0;
}
#elif defined(FMT_IQ4_NL)
// IQ4_NL: layout as Q4_0 (plane 0 = qs, plane 1 = d binary16); weight = d * KVALUES[nibble].
#define QK 32
const int WQ_KV_IQ4NL[16] = int[16](-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113);
uint wq_lut_iq4nl(uint n) {
    return (uint(WQ_KV_IQ4NL[n & 15u]) & 255u)
        | ((uint(WQ_KV_IQ4NL[(n >> 8) & 15u]) & 255u) << 8)
        | ((uint(WQ_KV_IQ4NL[(n >> 16) & 15u]) & 255u) << 16)
        | (uint(WQ_KV_IQ4NL[(n >> 24) & 15u]) << 24);
}
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    uint b = g * 4u + ((cin & 1u) << 1);
    uint sh = (cin >> 1) << 2;
    p0 = wq_lut_iq4nl((WQ_W(b) >> sh) & 0x0F0F0F0Fu);
    p1 = wq_lut_iq4nl((WQ_W(b + 1u) >> sh) & 0x0F0F0F0Fu);
    ws = wq_f16at(WQ_OFF1, g);
    wm = 0.0;
}
#elif defined(FMT_MXFP4)
// MXFP4: plane 0 = qs (16 bytes per block, as in the file), plane 1 = the E8M0 scale byte of every block (4 per word).
// value j (0..15) = low nibble of byte j, value j+16 = high nibble; q = 2 * E2M1 value (an integer), weight = q * 2^(e - 128).
#define QK 32
const int WQ_KV_FP4[16] = int[16](0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
uint wq_lut_fp4(uint n) {
    return (uint(WQ_KV_FP4[n & 15u]) & 255u)
        | ((uint(WQ_KV_FP4[(n >> 8) & 15u]) & 255u) << 8)
        | ((uint(WQ_KV_FP4[(n >> 16) & 15u]) & 255u) << 16)
        | (uint(WQ_KV_FP4[(n >> 24) & 15u]) << 24);
}
// half of 2^(x - 127) (the E8M0 scale), as the CPU decoder: denormal bit patterns for x < 2
float wq_e8m0_half(uint x) {
    return uintBitsToFloat(x < 2u ? (0x00200000u << x) : ((x - 1u) << 23));
}
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    uint b = g * 4u + ((cin & 1u) << 1);
    uint sh = (cin >> 1) << 2;
    p0 = wq_lut_fp4((WQ_W(b) >> sh) & 0x0F0F0F0Fu);
    p1 = wq_lut_fp4((WQ_W(b + 1u) >> sh) & 0x0F0F0F0Fu);
    ws = wq_e8m0_half((WQ_W(WQ_OFF1 + (g >> 2)) >> ((g & 3u) << 3)) & 255u);
    wm = 0.0;
}
#else
#error "define one FMT_* macro (see variants_ops.list)"
#endif

#endif
