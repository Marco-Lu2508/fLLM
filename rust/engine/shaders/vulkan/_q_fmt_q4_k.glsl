// Q4_K, device layout: plane 0 = qs (128 bytes per block, as in the file), plane 1 = 16 bytes: the eight 6-bit scales then the eight 6-bit
// mins, already decoded (get_scale_min_k4), plane 2 = one word per block (d | dmin << 16, binary16 each).
// Group g8 (32 values) uses bytes (g8/2)*32 .. +31 of qs, low nibbles for even g8, high nibbles for odd g8;
// weight = d * sc[g8] * q - dmin * mn[g8].
#define QK 256
#define HAS_WM 1
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    uint g8 = cin >> 2;
    uint j4 = cin & 3u;
    uint b = g * 32u + ((g8 >> 1) << 3) + (j4 << 1);
    uint sh = (g8 & 1u) << 2;
    p0 = (w[b] >> sh) & 0x0F0F0F0Fu;
    p1 = (w[b + 1u] >> sh) & 0x0F0F0F0Fu;
    uint q = p.off1 + g * 4u + (g8 >> 2);
    uint lane = (g8 & 3u) << 3;
    float sc = float((w[q] >> lane) & 255u);
    float mn = float((w[q + 2u] >> lane) & 255u);
    vec2 dm = unpackHalf2x16(w[p.off2 + g]);
    ws = dm.x * sc;
    wm = -dm.y * mn;
}