// Q3_K, device layout: plane 0 = qs (64 bytes per block), plane 1 = hmask (32 bytes), plane 2 = the 16 sub-block scales already
// decoded to int8 (scale - 32; 16 bytes), plane 3 = d as float32 (one word per block).
// Value (outer, sub, ix) with ix = side*16 + l: q = ((qs[outer*32 + ix] >> 2*sub) & 3) + 4 * bit(outer*4 + sub) of hmask[ix] - 4;
// sub-block scale index outer*8 + sub*2 + side; weight = d * scale * q.
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
    p0 = sub8(((w[b] >> sh) & 0x03030303u) | (((w[p.off1 + hm] >> hb) & 0x01010101u) << 2), 4u);
    p1 = sub8(((w[b + 1u] >> sh) & 0x03030303u) | (((w[p.off1 + hm + 1u] >> hb) & 0x01010101u) << 2), 4u);
    uint si = (outer << 3) + (sub << 1) + (j4 >> 1);
    int sc = bitfieldExtract(int(w[p.off2 + g * 4u + (si >> 2)]), int((si & 3u) << 3), 8);
    ws = uintBitsToFloat(w[p.off3 + g]) * float(sc);
    wm = 0.0;
}