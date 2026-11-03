// IQ3_S, device layout: plane 0 = qs (64 bytes per block, as in the file: grid index low bytes, 8 per group of 32 values), plane 1 = signs (32 bytes, one
// byte per 8 values), plane 2 = 16 bytes (qh: 8 bytes with one high index bit per entry; the 4 scale bytes with two 4-bit scales; d as binary16; 2 unused).
// chunk cin = 4 * group + l: two grid entries (4 magnitudes each), sign byte l; weight = d * (1 + 2 * scale) * magnitude * sign, one scale nibble per group.
#define QK 256
#include "_iq_common.glsl"
#include "_iq_grid_iq3s.glsl"
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    uint grp = cin >> 2;
    uint l = cin & 3u;
    uint h = (w[g * 16u + (grp << 1) + (l >> 1)] >> ((l & 1u) << 4)) & 0xFFFFu;
    uint qh = (w[p.off2 + g * 4u + (grp >> 2)] >> ((grp & 3u) << 3)) >> (l << 1);
    uint s8 = (w[p.off1 + g * 8u + grp] >> (l << 3)) & 255u;
    uint sc = (w[p.off2 + g * 4u + 2u] >> ((grp >> 1) << 3)) >> ((grp & 1u) << 2);
    p0 = iq_apply(IQ3S_GRID[(h & 255u) | ((qh & 1u) << 8)], s8);
    p1 = iq_apply(IQ3S_GRID[(h >> 8) | ((qh & 2u) << 7)], s8 >> 4);
    ws = unpackHalf2x16(w[p.off2 + g * 4u + 3u]).x * float(1u + 2u * (sc & 15u));
    wm = 0.0;
}
