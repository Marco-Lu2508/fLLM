// Q4_1, device layout: plane 0 = qs (16 bytes per block), plane 1 = one word per block (d | m << 16, binary16 each).
// weight = d * q + m, q = low nibble (values 0..15) or high nibble (16..31) of byte j.
#define QK 32
#define HAS_WM 1
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    uint b = g * 4u + ((cin & 1u) << 1);
    uint sh = (cin >> 1) << 2;
    p0 = (w[b] >> sh) & 0x0F0F0F0Fu;
    p1 = (w[b + 1u] >> sh) & 0x0F0F0F0Fu;
    vec2 dm = unpackHalf2x16(w[p.off1 + g]);
    ws = dm.x;
    wm = dm.y;
}