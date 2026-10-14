// Q4_0, device layout (INTERFACE_QUANT.md): plane 0 = qs (16 bytes per block, as in the file), plane 1 = d (binary16 per block).
// value j (0..15) = low nibble of byte j, value j+16 = high nibble of byte j; weight = d * (q - 8).
#define QK 32
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    uint b = g * 4u + ((cin & 1u) << 1);
    uint sh = (cin >> 1) << 2;
    p0 = sub8((w[b] >> sh) & 0x0F0F0F0Fu, 8u);
    p1 = sub8((w[b + 1u] >> sh) & 0x0F0F0F0Fu, 8u);
    ws = f16at(p.off1, g);
    wm = 0.0;
}