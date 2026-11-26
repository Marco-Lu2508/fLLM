// TQ1_0, device layout: plane 0 = qs (48 bytes per block of 256 values, as in the file), plane 1 = qh (4 bytes), plane 2 = d (binary16 per block).
// Base-3 digits, five per byte: values 0..159 = digit n = v / 32 of qs[v % 32], values 160..239 = digit (v - 160) / 16 of qs[32 + (v - 160) % 16], values 240..255 =
// digit (v - 240) / 4 of qh[(v - 240) % 4]; digit n of a byte b = ((b * 3^n mod 256) * 3 >> 8), weight = d * (digit - 1).
// chunk cin (8 values): 0..19 and 20..29 read two words of qs, 30 and 31 the word of qh.
#define QK 256
uint tq1_pow3(uint n) {
    uint r = 1u;
    for (uint i = 0u; i < n; i++) r *= 3u;
    return r;
}
// the four bytes of `word` as ternary digits number `n`, one signed byte (-1, 0, 1) per lane
uint tq1_dec4(uint word, uint pw) {
    uint r = 0u;
    for (uint k = 0u; k < 4u; k++) {
        uint b = (word >> (8u * k)) & 255u;
        uint xi = (((b * pw) & 255u) * 3u) >> 8u;
        r |= ((xi - 1u) & 255u) << (8u * k);
    }
    return r;
}
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    if (cin < 30u) {
        uint n;
        uint at;
        if (cin < 20u) {
            n = cin >> 2;
            at = g * 12u + ((cin & 3u) << 1);
        } else {
            uint j = cin - 20u;
            n = j >> 1;
            at = g * 12u + 8u + ((j & 1u) << 1);
        }
        uint pw = tq1_pow3(n);
        p0 = tq1_dec4(w[at], pw);
        p1 = tq1_dec4(w[at + 1u], pw);
    } else {
        uint qh = w[p.off1 + g];
        uint n = (cin - 30u) << 1;
        p0 = tq1_dec4(qh, tq1_pow3(n));
        p1 = tq1_dec4(qh, tq1_pow3(n + 1u));
    }
    ws = f16at(p.off2, g);
    wm = 0.0;
}
