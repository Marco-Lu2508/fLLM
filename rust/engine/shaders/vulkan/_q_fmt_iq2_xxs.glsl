// IQ2_XXS, device layout (INTERFACE_QUANT.md): plane 0 = qs (64 bytes per block, as in the file: per group of 32 values one word of four grid indices and
// one word of four 7-bit sign fields with the 4-bit scale on top), plane 1 = d (binary16 per block).
// chunk cin = 4 * group + l: grid entry l of the group (magnitudes 8 / 25 / 43), sign field l; weight = d * (0.5 + scale) * 0.25 * magnitude * sign.
#define QK 256
#include "_iq_common.glsl"
#include "_iq_grid_iq2xxs.glsl"
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    uint b = g * 16u + ((cin >> 2) << 1);
    uint l = cin & 3u;
    uint sw = w[b + 1u];
    uint idx = (w[b] >> (l << 3)) & 255u;
    uint e = iq_entry(IQ2XXS_GRID2[idx >> 1], idx);
    uint s8 = iq_signs8((sw >> (7u * l)) & 127u);
    p0 = iq_apply(iq_mag2(iq_spread2(e & 255u)), s8);
    p1 = iq_apply(iq_mag2(iq_spread2(e >> 8)), s8 >> 4);
    ws = f16at(p.off1, g) * ((0.5 + float(sw >> 28)) * 0.25);
    wm = 0.0;
}
