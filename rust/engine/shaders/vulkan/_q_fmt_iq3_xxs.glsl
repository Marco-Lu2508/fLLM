// IQ3_XXS, device layout: plane 0 = qs (64 bytes per block, as in the file: grid indices, 8 per group of 32 values), plane 1 = 32 bytes (one word per group:
// four 7-bit sign fields with the 4-bit scale on top), plane 2 = d (binary16 per block).
// chunk cin = 4 * group + l: two grid entries (4 magnitudes each) and sign field l; weight = d * (0.5 + scale) * 0.5 * magnitude * sign.
#define QK 256
#include "_iq_common.glsl"
#include "_iq_grid_iq3xxs.glsl"
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    uint grp = cin >> 2;
    uint l = cin & 3u;
    uint h = (w[g * 16u + (grp << 1) + (l >> 1)] >> ((l & 1u) << 4)) & 0xFFFFu;
    uint ss = w[p.off1 + g * 8u + grp];
    uint s8 = iq_signs8((ss >> (7u * l)) & 127u);
    p0 = iq_apply(IQ3XXS_GRID[h & 255u], s8);
    p1 = iq_apply(IQ3XXS_GRID[h >> 8], s8 >> 4);
    ws = f16at(p.off2, g) * ((0.5 + float(ss >> 28)) * 0.5);
    wm = 0.0;
}
