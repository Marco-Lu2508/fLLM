// Q5_K, device layout: plane 0 = qs (128 bytes per block), plane 1 = qh (32 bytes), plane 2 = 16 bytes: eight decoded 6-bit scales then eight
// decoded 6-bit mins, plane 3 = one word per block (d | dmin << 16).
// As Q4_K, with the fifth bit of value (g8, j) taken from bit g8 of qh[j]; weight = d * sc[g8] * q - dmin * mn[g8].
#define QK 256
#define HAS_WM 1
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    uint g8 = cin >> 2;
    uint j4 = cin & 3u;
    uint b = g * 32u + ((g8 >> 1) << 3) + (j4 << 1);
    uint sh = (g8 & 1u) << 2;
    uint h = g * 8u + (j4 << 1);
    p0 = ((w[b] >> sh) & 0x0F0F0F0Fu) | (((w[p.off1 + h] >> g8) & 0x01010101u) << 4);
    p1 = ((w[b + 1u] >> sh) & 0x0F0F0F0Fu) | (((w[p.off1 + h + 1u] >> g8) & 0x01010101u) << 4);
    uint q = p.off2 + g * 4u + (g8 >> 2);
    uint lane = (g8 & 3u) << 3;
    float sc = float((w[q] >> lane) & 255u);
    float mn = float((w[q + 2u] >> lane) & 255u);
    vec2 dm = unpackHalf2x16(w[p.off3 + g]);
    ws = dm.x * sc;
    wm = -dm.y * mn;
}