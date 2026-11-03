// IQ2_S, device layout: plane 0 = qs (64 bytes per block, as in the file: 32 grid index bytes, then 32 sign bytes), plane 1 = 16 bytes (qh: 8 bytes with
// two high index bits per entry, then the 8 scale bytes with two 4-bit scales), plane 2 = d (binary16 per block).
// chunk cin = 4 * group + l: entry l of the group, index = qs | 2 bits of qh << 8; weight = d * (0.5 + scale[l / 2]) * 0.25 * magnitude * sign.
#define QK 256
#include "_iq_common.glsl"
#include "_iq_grid_iq2s.glsl"
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    uint grp = cin >> 2;
    uint l = cin & 3u;
    uint sh = (grp & 3u) << 3;
    uint qh = (w[p.off1 + g * 4u + (grp >> 2)] >> sh) >> (l << 1);
    uint idx = ((w[g * 16u + grp] >> (l << 3)) & 255u) | ((qh & 3u) << 8);
    uint e = iq_entry(IQ2S_GRID2[idx >> 1], idx);
    uint s8 = (w[g * 16u + 8u + grp] >> (l << 3)) & 255u;
    uint sc = (w[p.off1 + g * 4u + 2u + (grp >> 2)] >> sh) >> ((l >> 1) << 2);
    p0 = iq_apply(iq_mag2(iq_spread2(e & 255u)), s8);
    p1 = iq_apply(iq_mag2(iq_spread2(e >> 8)), s8 >> 4);
    ws = f16at(p.off2, g) * ((0.5 + float(sc & 15u)) * 0.25);
    wm = 0.0;
}
