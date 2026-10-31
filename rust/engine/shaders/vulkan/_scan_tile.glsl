// FLLM Vulkan shaders: 64 x 64 tile products on shared memory for the scan_tm / scan_wide / scan_pm shaders (256 invocations; port of the tile_* helpers of
// cuda/scan_core.cuh). Included after `sc` (the scratch buffer, `float sc[]`) is declared and `_scan_common.glsl` is included.
//
// tY and tX are [k][64] tiles. tile_gemm adds  sum_k tY[k][ty * 4 + i] * tX[k][tx * 4 + j]  to acc[i * 4 + j] with (ty, tx) = (tid >> 4, tid & 15).
#ifndef FLLM_SCAN_TILE
#define FLLM_SCAN_TILE

shared float tX[4096];
shared float tY[4096];

// tile[k][n] = sc[src + k * ld + n] for k < rows, n < cols (else 0), into tY (toY) or tX; lanes along n.
void tile_rows(bool toY, uint src, uint ld, uint rows, uint cols) {
    for (uint e = gl_LocalInvocationID.x; e < 4096u; e += 256u) {
        const uint k = e >> 6u;
        const uint n = e & 63u;
        const float v = (k < rows && n < cols) ? sc[src + k * ld + n] : 0.0;
        if (toY) tY[e] = v; else tX[e] = v;
    }
}

// tile[k][m] = sc[src + m * ld + k] for m < rows, k < cols (else 0): the transpose of a row-major tile; lanes along m.
void tile_trans(bool toY, uint src, uint ld, uint rows, uint cols) {
    for (uint e = gl_LocalInvocationID.x; e < 4096u; e += 256u) {
        const uint k = e >> 6u;
        const uint m = e & 63u;
        const float v = (m < rows && k < cols) ? sc[src + m * ld + k] : 0.0;
        if (toY) tY[e] = v; else tX[e] = v;
    }
}

void tile_gemm(inout float acc[16]) {
    const uint ty = gl_LocalInvocationID.x >> 4u;
    const uint tx = gl_LocalInvocationID.x & 15u;
    for (uint k = 0u; k < 64u; k++) {
        float a[4];
        float b[4];
        [[unroll]] for (uint i = 0u; i < 4u; i++) {
            a[i] = tY[k * 64u + ty * 4u + i];
            b[i] = tX[k * 64u + tx * 4u + i];
        }
        [[unroll]] for (uint i = 0u; i < 4u; i++) {
            [[unroll]] for (uint j = 0u; j < 4u; j++) acc[i * 4u + j] += a[i] * b[j];
        }
    }
}

// acc[i * 4 + j] += sum_k A[ty * 4 + i][k] * tX[k][tx * 4 + j] with the left operand a row-major matrix in the scratch buffer (`a_base`, row stride lda) ...
void tile_gemm_ga(uint a_base, uint lda, inout float acc[16]) {
    const uint ty = gl_LocalInvocationID.x >> 4u;
    const uint tx = gl_LocalInvocationID.x & 15u;
    const uint a0 = a_base + ty * 4u * lda;
    for (uint k = 0u; k < 64u; k++) {
        float a[4];
        float b[4];
        [[unroll]] for (uint i = 0u; i < 4u; i++) {
            a[i] = sc[a0 + i * lda + k];
            b[i] = tX[k * 64u + tx * 4u + i];
        }
        [[unroll]] for (uint i = 0u; i < 4u; i++) {
            [[unroll]] for (uint j = 0u; j < 4u; j++) acc[i * 4u + j] += a[i] * b[j];
        }
    }
}

// ... or a row-major [64][64] tile in shared memory (tY).
void tile_gemm_sa(inout float acc[16]) {
    const uint ty = gl_LocalInvocationID.x >> 4u;
    const uint tx = gl_LocalInvocationID.x & 15u;
    for (uint k = 0u; k < 64u; k++) {
        float a[4];
        float b[4];
        [[unroll]] for (uint i = 0u; i < 4u; i++) {
            a[i] = tY[(ty * 4u + i) * 64u + k];
            b[i] = tX[k * 64u + tx * 4u + i];
        }
        [[unroll]] for (uint i = 0u; i < 4u; i++) {
            [[unroll]] for (uint j = 0u; j < 4u; j++) acc[i * 4u + j] += a[i] * b[j];
        }
    }
}

#endif
