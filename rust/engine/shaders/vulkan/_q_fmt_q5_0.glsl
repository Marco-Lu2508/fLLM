// Q5_0, device layout: plane 0 = qs (16 bytes per block), plane 1 = qh (one word per block), plane 2 = d (binary16 per block).
// q = nibble | bit j of qh << 4 (j = value index 0..31); weight = d * (q - 16).
#define QK 32
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    uint b = g * 4u + ((cin & 1u) << 1);
    uint sh = (cin >> 1) << 2;
    uint qh = w[p.off1 + g] >> (cin << 3);           // bits of values 8*cin .. 8*cin+7 in the low byte
    uint h0 = ((qh & 15u) * 0x00204081u) & 0x01010101u;          // spreads 4 bits to 4 byte lanes
    uint h1 = (((qh >> 4) & 15u) * 0x00204081u) & 0x01010101u;
    p0 = sub8(((w[b] >> sh) & 0x0F0F0F0Fu) | (h0 << 4), 16u);
    p1 = sub8(((w[b + 1u] >> sh) & 0x0F0F0F0Fu) | (h1 << 4), 16u);
    ws = f16at(p.off2, g);
    wm = 0.0;
}