// NVFP4, device layout: plane 0 = qs (32 bytes per block of 64 values), plane 1 = the four UE4M3 scales of the block (one word).
// Sub-block s (16 values) = bytes 8s..8s+7 of qs: values 0..7 of it are the low nibbles, 8..15 the high nibbles.
// q = 2 * E2M1 value, weight = q * ue4m3(scale) / 2. The per-tensor factor comes in through the push constant `scale`.
#define QK 64
const int KV_FP4[16] = int[16](0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
uint lut_fp4(uint n) {
    return (uint(KV_FP4[n & 15u]) & 255u)
        | ((uint(KV_FP4[(n >> 8) & 15u]) & 255u) << 8)
        | ((uint(KV_FP4[(n >> 16) & 15u]) & 255u) << 16)
        | (uint(KV_FP4[(n >> 24) & 15u]) << 24);
}
// half of the unsigned E4M3 value (0 and 0x7f give 0), as the CPU decoder
float ue4m3_half(uint x) {
    if (x == 0u || x == 0x7fu) return 0.0;
    uint e = (x >> 3) & 15u;
    uint m = x & 7u;
    if (e == 0u) return float(m) * (1.0 / 1024.0);
    return 0.5 * uintBitsToFloat(((e + 120u) << 23) | (m << 20));
}
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    uint s = cin >> 1;
    uint b = g * 8u + (s << 1);
    uint sh = (cin & 1u) << 2;
    p0 = lut_fp4((w[b] >> sh) & 0x0F0F0F0Fu);
    p1 = lut_fp4((w[b + 1u] >> sh) & 0x0F0F0F0Fu);
    ws = ue4m3_half((w[p.off1 + g] >> (s << 3)) & 255u);
    wm = 0.0;
}