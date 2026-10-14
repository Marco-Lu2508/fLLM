// MXFP4, device layout: plane 0 = qs (16 bytes per block, as in the file), plane 1 = the E8M0 scale byte of every block (4 per word).
// value j (0..15) = low nibble of byte j, value j+16 = high nibble; q = 2 * E2M1 value (an integer), weight = q * 2^(e - 128).
#define QK 32
const int KV_FP4[16] = int[16](0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12);
uint lut_fp4(uint n) {
    return (uint(KV_FP4[n & 15u]) & 255u)
        | ((uint(KV_FP4[(n >> 8) & 15u]) & 255u) << 8)
        | ((uint(KV_FP4[(n >> 16) & 15u]) & 255u) << 16)
        | (uint(KV_FP4[(n >> 24) & 15u]) << 24);
}
// half of 2^(x - 127) (the E8M0 scale), as the CPU decoder: denormal bit patterns for x < 2
float e8m0_half(uint x) {
    return uintBitsToFloat(x < 2u ? (0x00200000u << x) : ((x - 1u) << 23));
}
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    uint b = g * 4u + ((cin & 1u) << 1);
    uint sh = (cin >> 1) << 2;
    p0 = lut_fp4((w[b] >> sh) & 0x0F0F0F0Fu);
    p1 = lut_fp4((w[b + 1u] >> sh) & 0x0F0F0F0Fu);
    ws = e8m0_half((w[p.off1 + (g >> 2)] >> ((g & 3u) << 3)) & 255u);
    wm = 0.0;
}