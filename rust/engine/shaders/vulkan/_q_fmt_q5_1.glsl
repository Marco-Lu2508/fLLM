// Q5_1, device layout: plane 0 = qs (16 bytes per block), plane 1 = qh (one word per block), plane 2 = one word per block (d | m << 16).
// q = nibble | bit j of qh << 4; weight = d * q + m.
#define QK 32
#define HAS_WM 1
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    uint b = g * 4u + ((cin & 1u) << 1);
    uint sh = (cin >> 1) << 2;
    uint qh = w[p.off1 + g] >> (cin << 3);
    uint h0 = ((qh & 15u) * 0x00204081u) & 0x01010101u;
    uint h1 = (((qh >> 4) & 15u) * 0x00204081u) & 0x01010101u;
    p0 = ((w[b] >> sh) & 0x0F0F0F0Fu) | (h0 << 4);
    p1 = ((w[b + 1u] >> sh) & 0x0F0F0F0Fu) | (h1 << 4);
    vec2 dm = unpackHalf2x16(w[p.off2 + g]);
    ws = dm.x;
    wm = dm.y;
}