// Q4Z (AWQ / GPTQ 4-bit, group Q4Z_GROUP = 32, 64 or 128), device layout: plane 0 = codes repacked per chunk of 8 values into one word
// (byte k: low nibble = value k, high nibble = value k + 4; the file stores value 2j in the low nibble of byte j),
// plane 1 = one word per group (binary16 scale | zero point << 16). weight = (q - zero) * scale.
// Q4ZP weights (act-order GPTQ) use the same layout: the prepass permutes the activations (x'[c] = x[perm[c]]).
#ifndef Q4Z_GROUP
#error "Q4Z_GROUP must be 32, 64 or 128"
#endif
#define QK Q4Z_GROUP
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    uint v = w[g * (Q4Z_GROUP / 8) + cin];
    uint sz = w[p.off1 + g];
    uint zero = (sz >> 16) & 15u;
    p0 = sub8(v & 0x0F0F0F0Fu, zero);
    p1 = sub8((v >> 4) & 0x0F0F0F0Fu, zero);
    ws = unpackHalf2x16(sz).x;
    wm = 0.0;
}