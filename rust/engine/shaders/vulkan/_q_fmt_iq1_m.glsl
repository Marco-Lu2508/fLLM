// IQ1_M, device layout: plane 0 = qs (32 bytes per block, as in the file), plane 1 = qh (16 bytes: 2 bytes per group of 32 values, a nibble per entry: 3 high
// index bits and the sign of the shift), plane 2 = the 8 scale bytes (four u16 with 3-bit scales, one per 16 values), plane 3 = d (binary16, assembled on the
// host from the top nibbles of the four scale words).
// chunk cin = 4 * group + l: entry l of the group (values -1, 0, 1); weight = d * (2 * scale + 1) * (value + delta), delta = +-0.125, i.e. ws * q + wm.
#define QK 256
#define HAS_WM 1
#include "_iq_common.glsl"
#include "_iq_grid_iq1s.glsl"
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    uint grp = cin >> 2;
    uint l = cin & 3u;
    uint hb = (grp << 1) + (l >> 1);
    uint qh = (w[p.off1 + g * 4u + (hb >> 2)] >> ((hb & 3u) << 3)) >> ((l & 1u) << 2);
    uint idx = ((w[g * 8u + grp] >> (l << 3)) & 255u) | ((qh & 7u) << 8);
    uint e = iq_entry(IQ1S_GRID2[idx >> 1], idx);
    uint sc = (w[p.off2 + g * 2u + (grp >> 2)] >> (((grp >> 1) & 1u) << 4)) >> (6u * (grp & 1u) + (l >> 1) * 3u);
    p0 = iq_dec1(iq_spread2(e & 255u));
    p1 = iq_dec1(iq_spread2(e >> 8));
    ws = f16at(p.off3, g) * float(2u * (sc & 7u) + 1u);
    wm = ws * (((qh & 8u) != 0u) ? -0.125 : 0.125);
}
