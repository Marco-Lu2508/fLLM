// IQ4_NL, device layout as Q4_0 (plane 0 = qs, plane 1 = d binary16); weight = d * KVALUES[nibble].
#define QK 32
const int KV_IQ4NL[16] = int[16](-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113);
// four nibbles (one per byte lane) -> four int8 code book values
uint lut_iq4nl(uint n) {
    return (uint(KV_IQ4NL[n & 15u]) & 255u)
        | ((uint(KV_IQ4NL[(n >> 8) & 15u]) & 255u) << 8)
        | ((uint(KV_IQ4NL[(n >> 16) & 15u]) & 255u) << 16)
        | (uint(KV_IQ4NL[(n >> 24) & 15u]) << 24);
}
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    uint b = g * 4u + ((cin & 1u) << 1);
    uint sh = (cin >> 1) << 2;
    p0 = lut_iq4nl((w[b] >> sh) & 0x0F0F0F0Fu);
    p1 = lut_iq4nl((w[b + 1u] >> sh) & 0x0F0F0F0Fu);
    ws = f16at(p.off1, g);
    wm = 0.0;
}