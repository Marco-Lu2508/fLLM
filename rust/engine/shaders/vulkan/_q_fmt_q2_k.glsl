// Q2_K, device layout: plane 0 = qs (64 bytes per block, as in the file), plane 1 = scales (16 bytes: low nibble scale, high nibble min),
// plane 2 = one word per block (d | dmin << 16, binary16 each).
// 16 sub-blocks of 16 values: sub-block si = n*8 + sub*2 + hg (n = 128-value half, sub = shift step, hg = 16-value half of the 32 bytes);
// q = (qs[n*32 + hg*16 + l] >> (2*sub)) & 3; weight = d * (scale & 15) * q - dmin * (scale >> 4).
#define QK 256
#define HAS_WM 1
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    uint n = cin >> 4;
    uint r = cin & 15u;
    uint sub = r >> 2;
    uint j4 = r & 3u;
    uint b = g * 16u + (n << 3) + (j4 << 1);
    uint sh = sub << 1;
    p0 = (w[b] >> sh) & 0x03030303u;
    p1 = (w[b + 1u] >> sh) & 0x03030303u;
    uint si = (n << 3) + (sub << 1) + (j4 >> 1);
    uint sc = (w[p.off1 + g * 4u + (si >> 2)] >> ((si & 3u) << 3)) & 255u;
    vec2 dm = unpackHalf2x16(w[p.off2 + g]);
    ws = dm.x * float(sc & 15u);
    wm = -dm.y * float(sc >> 4);
}