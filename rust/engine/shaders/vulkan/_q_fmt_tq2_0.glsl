// TQ2_0, device layout: plane 0 = qs (64 bytes per block of 256 values, as in the file), plane 1 = d (binary16 per block).
// Two chunks of 32 bytes; value (128 * c + 32 * l + m) = bits 2 * l .. +1 of byte 32 * c + m; weight = d * (q - 1), q in 0..3 (-1, 0 or 1 times d, 2 never occurs in files).
// chunk cin (8 values): c = cin / 16, l = (cin % 16) / 4, bytes 32 * c + 8 * (cin % 4) .. +7 = words 8 * c + 2 * (cin % 4) and the next one.
#define QK 256
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    uint at = g * 16u + (cin >> 4) * 8u + ((cin & 3u) << 1);
    uint shift = ((cin & 15u) >> 2) << 1;
    p0 = sub8((w[at] >> shift) & 0x03030303u, 1u);
    p1 = sub8((w[at + 1u] >> shift) & 0x03030303u, 1u);
    ws = f16at(p.off1, g);
    wm = 0.0;
}
