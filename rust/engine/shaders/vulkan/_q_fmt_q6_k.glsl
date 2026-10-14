// Q6_K, device layout: plane 0 = ql (128 bytes per block), plane 1 = qh (64 bytes), plane 2 = the 16 int8 scales (16 bytes),
// plane 3 = d as float32 (one word per block).
// Half n (128 values), group grp (32 values), j in 0..31: low 4 bits = low (grp < 2) or high (grp >= 2) nibble of ql[n*64 + (grp&1)*32 + j],
// high 2 bits = bits 2*grp.. of qh[n*32 + j]; q = value - 32; scale index n*8 + j/16 + grp*2; weight = d * scale * q.
#define QK 256
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    uint n = cin >> 4;
    uint r = cin & 15u;
    uint grp = r >> 2;
    uint j4 = r & 3u;
    uint b = g * 32u + (n << 4) + ((grp & 1u) << 3) + (j4 << 1);
    uint sh = (grp >> 1) << 2;
    uint h = g * 16u + (n << 3) + (j4 << 1);
    uint hs = grp << 1;
    p0 = sub8(((w[b] >> sh) & 0x0F0F0F0Fu) | (((w[p.off1 + h] >> hs) & 0x03030303u) << 4), 32u);
    p1 = sub8(((w[b + 1u] >> sh) & 0x0F0F0F0Fu) | (((w[p.off1 + h + 1u] >> hs) & 0x03030303u) << 4), 32u);
    uint si = (n << 3) + (j4 >> 1) + (grp << 1);
    int sc = bitfieldExtract(int(w[p.off2 + g * 4u + (si >> 2)]), int((si & 3u) << 3), 8);
    ws = uintBitsToFloat(w[p.off3 + g]) * float(sc);
    wm = 0.0;
}