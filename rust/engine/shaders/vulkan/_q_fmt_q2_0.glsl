// Q2_0, device layout (INTERFACE_QUANT.md): plane 0 = qs (16 bytes per block of 64 values, as in the file), plane 1 = d (binary16 per block).
// value j (0..63) = bits 2 * (j % 4) .. +1 of byte j / 4; weight = d * (q - 1) with q in 0..3, i.e. -1, 0, 1 or 2 times d.
#define QK 64
// four 2-bit codes (the low 8 bits of x) -> one code per byte lane
uint spread2(uint x) {
    return (x | (x << 6) | (x << 12) | (x << 18)) & 0x03030303u;
}
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    uint h = (w[g * 4u + (cin >> 1)] >> ((cin & 1u) << 4)) & 0xFFFFu;   // eight codes: values 8 * cin .. 8 * cin + 7
    p0 = sub8(spread2(h & 0xFFu), 1u);
    p1 = sub8(spread2(h >> 8), 1u);
    ws = f16at(p.off1, g);
    wm = 0.0;
}
