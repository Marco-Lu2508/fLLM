// bitsandbytes LLM.int8() (`bnb::INT8_ROW`), device layout: plane 0 = the 64 int8 codes of a block (16 words), plane 1 = the f32 scale of every block
// (the scale of the output row, repeated in each block of the row, as in the file). weight = scale * code.
#define QK 64
void wchunk(uint g, uint cin, out uint p0, out uint p1, out float ws, out float wm) {
    uint b = g * 16u + (cin << 1);
    p0 = w[b];
    p1 = w[b + 1u];
    ws = uintBitsToFloat(w[p.off1 + g]);
    wm = 0.0;
}
