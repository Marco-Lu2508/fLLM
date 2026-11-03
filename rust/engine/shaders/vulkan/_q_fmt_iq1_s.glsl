// IQ1_S, device layout: plane 0 = qs (32 bytes per block, as in the file: grid index low bytes, 4 per group of 32 values), plane 1 = qh (16 bytes: one u16 per
// group: 3 high index bits for each of the 4 entries, a 3-bit scale at bits 12..14, the sign of the shift at bit 15), plane 2 = d (binary16 per block).
// chunk cin = 4 * group + l: entry l of the group (values -1, 0, 1); weight = d * (2 * scale + 1) * (value + delta), delta = +-0.125, i.e. ws * q + wm.
#define QK 256
#define HAS_WM 1
#include "_iq_common.glsl"
#include "_iq_grid_iq1s.glsl"
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    uint grp = cin >> 2;
    uint l = cin & 3u;
    uint qh = (w[p.off1 + g * 4u + (grp >> 1)] >> ((grp & 1u) << 4)) & 0xFFFFu;
    uint idx = ((w[g * 8u + grp] >> (l << 3)) & 255u) | (((qh >> (3u * l)) & 7u) << 8);
    uint e = iq_entry(IQ1S_GRID2[idx >> 1], idx);
    p0 = iq_dec1(iq_spread2(e & 255u));
    p1 = iq_dec1(iq_spread2(e >> 8));
    ws = f16at(p.off2, g) * float(2u * ((qh >> 12) & 7u) + 1u);
    wm = ws * (((qh & 0x8000u) != 0u) ? -0.125 : 0.125);
}
