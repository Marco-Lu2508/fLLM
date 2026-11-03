// IQ2_XS, device layout: plane 0 = qs (64 bytes per block, as in the file: 32 x u16, 9 bits grid index + 7 bits sign field), plane 1 = scales (8 bytes,
// two 4-bit scales per group of 32 values), plane 2 = d (binary16 per block).
// chunk cin = 4 * group + l: u16 l of the group; weight = d * (0.5 + scale[l / 2]) * 0.25 * magnitude * sign.
#define QK 256
#include "_iq_common.glsl"
#include "_iq_grid_iq2xs.glsl"
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    uint grp = cin >> 2;
    uint l = cin & 3u;
    uint q = (w[g * 16u + (grp << 1) + (l >> 1)] >> ((l & 1u) << 4)) & 0xFFFFu;
    uint idx = q & 511u;
    uint e = iq_entry(IQ2XS_GRID2[idx >> 1], idx);
    uint s8 = iq_signs8(q >> 9);
    uint sc = (w[p.off1 + g * 2u + (grp >> 2)] >> ((grp & 3u) << 3)) >> ((l >> 1) << 2);
    p0 = iq_apply(iq_mag2(iq_spread2(e & 255u)), s8);
    p1 = iq_apply(iq_mag2(iq_spread2(e >> 8)), s8 >> 4);
    ws = f16at(p.off2, g) * ((0.5 + float(sc & 15u)) * 0.25);
    wm = 0.0;
}
