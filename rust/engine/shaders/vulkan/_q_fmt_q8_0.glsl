// Q8_0, device layout: plane 0 = qs (32 int8 per block), plane 1 = d (binary16 per block). weight = d * q.
#define QK 32
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    uint b = g * 8u + (cin << 1);
    p0 = w[b];
    p1 = w[b + 1u];
    ws = f16at(p.off1, g);
    wm = 0.0;
}