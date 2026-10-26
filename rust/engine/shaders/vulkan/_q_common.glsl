// FLLM quantized linear (package 2): declarations shared by mv_scalar / mv_idot / mm_scalar / mm_idot / mm_coopmat.
// Layout of the weight buffer, the push constants and the chunk decoders are documented in INTERFACE_QUANT.md.
//
// A "chunk" is 8 consecutive weights of one row along K. Every format decodes a chunk into
//   p0, p1 : 8 small signed integers (int8 each, 4 per uint, byte i = value i, value 0 = lowest byte)
//   ws, wm : float scale and float offset, so that  weight[i] = ws * q[i] + wm
// (wm is only non-zero for formats with an additive term: Q4_1, Q5_1, Q2_K, Q4_K, Q5_K; they define HAS_WM).

layout(push_constant) uniform QPush {
    uint rows;        // weight rows handled by this dispatch (= output features)
    uint cols;        // K, input features; multiple of QK
    uint batch;       // activation rows (tokens)
    uint flags;       // bit 0: add bias
    float scale;      // per-tensor factor applied to every dot product (DeviceWeight::scale)
    uint nb;          // blocks per weight row = cols / QK
    uint off1;        // word offset of plane 1 in the weight buffer
    uint off2;        // word offset of plane 2
    uint off3;        // word offset of plane 3
    uint out_stride;  // floats between two output rows (tokens); normally the total number of weight rows
    uint row_base;    // first output feature of this dispatch (added to the row index of out and bias)
    uint xq_d;        // int-dot shaders: word offset of the block scales inside the quantized activations
} p;

layout(std430, set = 0, binding = 0) readonly buffer WeightBuf { uint w[]; };

// binary16 number i of a plane of packed halves starting at word `base`
float f16at(uint base, uint i) {
    return unpackHalf2x16(w[base + (i >> 1)])[i & 1u];
}

// four int8 lanes -> vec4
vec4 s8x4(uint v) {
    return vec4((ivec4(int(v)) << ivec4(24, 16, 8, 0)) >> 24);
}

// lane-wise (x - c) for four bytes x < 128 and c <= 128, the result as two's complement int8 lanes
uint sub8(uint x, uint c) {
    return ((x | 0x80808080u) - c * 0x01010101u) ^ 0x80808080u;
}

// signed 4 x int8 dot product of two packed words
int idot4(uint a, uint b) {
#if FLLM_HW_DOT == 1
    return dotPacked4x8EXT(int(a), int(b));
#elif FLLM_HW_DOT == 2
    return fllm_sdot(int(a), int(b), 0);
#else
    ivec4 x = (ivec4(int(a)) << ivec4(24, 16, 8, 0)) >> 24;
    ivec4 y = (ivec4(int(b)) << ivec4(24, 16, 8, 0)) >> 24;
    return x.x * y.x + x.y * y.y + x.z * y.z + x.w * y.w;
#endif
}

// acc + signed 4 x int8 dot product (one instruction with the hardware dot product; saturation cannot happen for the sums of one 32-value block)
int idot4acc(uint a, uint b, int acc) {
#if FLLM_HW_DOT == 1
    return dotPacked4x8AccSatEXT(int(a), int(b), acc);
#elif FLLM_HW_DOT == 2
    return fllm_sdot_acc(int(a), int(b), acc, 0);
#else
    return acc + idot4(a, b);
#endif
}

// 8 values of a chunk as floats, ws * q + wm
void chunk_floats(uint p0, uint p1, float ws, float wm, out float v[8]) {
    vec4 a = s8x4(p0) * ws + wm;
    vec4 b = s8x4(p1) * ws + wm;
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w;
    v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}
