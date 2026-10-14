// fLLM CUDA backend: integer dot-product kernel for narrow batches (1..4 tokens) of the formats decoded by iq_decode.cuh
// (Q2_K, Q3_K, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S). Included at the end of mmvq.cu, which owns the int8
// activation layout (`quantize_x8_kernel`).
//
// A lane takes one run of 16 weights: `x_run` produces them as four words of signed 8-bit integers plus two floats such that
//   weight = scale * q - off
// and the dot product with the int8 activations xq (scale dx per 32 values) is
//   dx * (scale * dp4a(q, xq) - off * sum(xq)).
// The decoding rules are those of decode16_iq (iq_decode.cuh); selftest-gpu compares the result with the CPU reference.
#pragma once

// per-byte (a - b) without borrows between the bytes
__device__ __forceinline__ uint32_t x_vsub4(uint32_t a, uint32_t b) {
    return ((a | 0x80808080u) - (b & 0x7f7f7f7fu)) ^ ((a ^ ~b) & 0x80808080u);
}
// bits 0..3 of `bits` as bytes 0x00 / 0xFF
__device__ __forceinline__ uint32_t x_sign_mask(uint32_t bits) { return ((bits * 0x00204081u) & 0x01010101u) * 0xffu; }
// negate the bytes of g (values < 128) where the mask byte is 0xFF
__device__ __forceinline__ uint32_t x_negate(uint32_t g, uint32_t mask) { return x_vsub4(g ^ mask, mask); }
// eight grid bytes with the signs of `signs` (bit j = byte j) as two words
__device__ __forceinline__ void x_entry8(uint32_t *q, uint64_t grid, uint32_t signs) {
    q[0] = x_negate((uint32_t)grid, x_sign_mask(signs & 15u));
    q[1] = x_negate((uint32_t)(grid >> 32), x_sign_mask((signs >> 4) & 15u));
}
__device__ __forceinline__ void x_entry8_3(uint32_t *q, uint32_t g0, uint32_t g1, uint32_t signs) {
    q[0] = x_negate(g0, x_sign_mask(signs & 15u));
    q[1] = x_negate(g1, x_sign_mask((signs >> 4) & 15u));
}

template <int KIND>
__device__ __forceinline__ void x_run(const uint8_t *rp, int run, uint32_t *q, float &scale, float &off) {
    const uint8_t *b = rp + (size_t)(run >> 4) * kind_block_bytes(KIND);
    const int local = (run & 15) * 16;
    off = 0.f;
    if (KIND == KIND_Q2_K) {
        const float d = h2f_bits(ld16(b + 80)), dmin = h2f_bits(ld16(b + 82));
        const int n = local >> 7, s = (local & 127) >> 5, hg = (local >> 4) & 1;
        const uint32_t sc = b[n * 8 + s * 2 + hg];
        scale = d * (float)(sc & 15u);
        off = dmin * (float)(sc >> 4);
        const uint8_t *p = b + 16 + n * 32 + hg * 16;
#pragma unroll
        for (int k = 0; k < 4; k++) q[k] = (iq_ld32(p + 4 * k) >> (2 * s)) & 0x03030303u;
        return;
    }
    if (KIND == KIND_Q3_K) {
        const float d = h2f_bits(ld16(b + 108));
        const int n = local >> 7, sub = (local & 127) >> 5, side = (local >> 4) & 1;
        const int idx = n * 8 + sub * 2 + side, i = idx & 3, quad = idx >> 2;
        const uint32_t hi = b[96 + i], lo = b[100 + i], packed = b[104 + i];
        const uint32_t nib = (quad & 1 ? lo : hi) >> (quad >= 2 ? 4 : 0);
        scale = d * (float)((int)((nib & 15u) | (((packed >> (2 * quad)) & 3u) << 4)) - 32);
        off = 4.f * scale;  // value = scale * (low + 4 * highbit - 4)
        const uint8_t *p = b + 32 + n * 32 + side * 16, *hm = b + side * 16;
        const int bit = n * 4 + sub;
#pragma unroll
        for (int k = 0; k < 4; k++)
            q[k] = ((iq_ld32(p + 4 * k) >> (2 * sub)) & 0x03030303u) | (((iq_ld32(hm + 4 * k) >> bit) & 0x01010101u) << 2);
        return;
    }
    const float d = h2f_bits(ld16(b));
    const int ib32 = local >> 5, l0 = (local >> 3) & 3;
    if (KIND == KIND_IQ2_XXS) {
        const uint32_t lo = iq_ld32(b + 2 + ib32 * 8), hi = iq_ld32(b + 2 + ib32 * 8 + 4);
        scale = d * (0.5f + (float)(hi >> 28)) * 0.25f;
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const int l = l0 + e;
            x_entry8(q + 2 * e, kIq2xxsGrid[(lo >> (8 * l)) & 0xffu], kIqSigns[(hi >> (7 * l)) & 0x7fu]);
        }
        return;
    }
    if (KIND == KIND_IQ2_XS) {
        const uint32_t sc = b[66 + ib32];
        scale = d * (0.5f + (float)(l0 >= 2 ? sc >> 4 : sc & 15u)) * 0.25f;
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const uint32_t v = ld16(b + 2 + ib32 * 8 + (l0 + e) * 2);
            x_entry8(q + 2 * e, kIq2xsGrid[v & 0x1ffu], kIqSigns[v >> 9]);
        }
        return;
    }
    if (KIND == KIND_IQ2_S) {
        const uint32_t sc = b[74 + ib32], qh = b[66 + ib32];
        scale = d * (0.5f + (float)(l0 >= 2 ? sc >> 4 : sc & 15u)) * 0.25f;
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const int l = l0 + e;
            x_entry8(q + 2 * e, kIq2sGrid[b[2 + ib32 * 4 + l] | (((qh >> (2 * l)) & 3u) << 8)], b[34 + ib32 * 4 + l]);
        }
        return;
    }
    if (KIND == KIND_IQ3_XXS) {
        const uint32_t ss = iq_ld32(b + 66 + ib32 * 4);
        scale = d * (0.5f + (float)(ss >> 28)) * 0.5f;
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const int l = l0 + e;
            x_entry8_3(q + 2 * e, kIq3xxsGrid[b[2 + ib32 * 8 + l * 2]], kIq3xxsGrid[b[2 + ib32 * 8 + l * 2 + 1]],
                       kIqSigns[(ss >> (7 * l)) & 0x7fu]);
        }
        return;
    }
    if (KIND == KIND_IQ3_S) {
        const uint32_t s4 = b[106 + (ib32 >> 1)], qh = b[66 + ib32];
        scale = d * (float)(1 + 2 * (int)((ib32 & 1) ? (s4 >> 4) : (s4 & 15u)));
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const int l = l0 + e;
            const uint32_t q0 = b[2 + ib32 * 8 + l * 2] | (((qh >> (2 * l)) & 1u) << 8);
            const uint32_t q1 = b[2 + ib32 * 8 + l * 2 + 1] | (((qh >> (2 * l + 1)) & 1u) << 8);
            x_entry8_3(q + 2 * e, kIq3sGrid[q0], kIq3sGrid[q1], b[74 + ib32 * 4 + l]);
        }
        return;
    }
    if (KIND == KIND_IQ1_S) {
        const uint32_t qh = ld16(b + 34 + ib32 * 2);
        scale = d * (float)(2 * ((qh >> 12) & 7u) + 1);
        const float delta = (qh & 0x8000u) ? -0.125f : 0.125f;
        off = -scale * delta;  // value = scale * (grid + delta)
#pragma unroll
        for (int e = 0; e < 2; e++) {
            const int l = l0 + e;
            const uint64_t g = kIq1sGrid[b[2 + ib32 * 4 + l] | (((qh >> (3 * l)) & 7u) << 8)];
            q[2 * e] = (uint32_t)g;
            q[2 * e + 1] = (uint32_t)(g >> 32);
        }
        return;
    }
}

template <int KIND, int NB>
__global__ void __launch_bounds__(128)
mmvq_x_kernel(const uint8_t *__restrict__ W, const int8_t *__restrict__ Xq, const float2 *__restrict__ Xs,
              float *__restrict__ Y, const float *__restrict__ bias, int rows, int cols, int batch) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int row = blockIdx.x * 4 + warp;
    if (row >= rows) return;
    const uint8_t *rp = W + (size_t)row * kind_row_bytes(KIND, cols);
    const int runs = cols >> 4, sg = cols >> 5;
    float acc[NB];
#pragma unroll
    for (int b = 0; b < NB; b++) acc[b] = 0.f;
    for (int run = lane; run < runs; run += 32) {
        uint32_t q[4];
        float scale, off;
        x_run<KIND>(rp, run, q, scale, off);
#pragma unroll
        for (int b = 0; b < NB; b++) {
            if (b >= batch) break;
            const uint4 xv = __ldg(reinterpret_cast<const uint4 *>(Xq + (size_t)b * cols + (size_t)run * 16));
            const float dx = __ldg(&Xs[(size_t)b * sg + (run >> 1)].x);
            int dot = __dp4a((int)q[0], (int)xv.x, 0);
            dot = __dp4a((int)q[1], (int)xv.y, dot);
            dot = __dp4a((int)q[2], (int)xv.z, dot);
            dot = __dp4a((int)q[3], (int)xv.w, dot);
            float v = scale * (float)dot;
            if (off != 0.f) {
                int xsum = __dp4a(0x01010101, (int)xv.x, 0);
                xsum = __dp4a(0x01010101, (int)xv.y, xsum);
                xsum = __dp4a(0x01010101, (int)xv.z, xsum);
                xsum = __dp4a(0x01010101, (int)xv.w, xsum);
                v -= off * (float)xsum;
            }
            acc[b] += dx * v;
        }
    }
#pragma unroll
    for (int b = 0; b < NB; b++) acc[b] = warp_sum(acc[b]);
    if (lane == 0) {
        const float bv = bias ? bias[row] : 0.f;
#pragma unroll
        for (int b = 0; b < NB; b++)
            if (b < batch) Y[(size_t)b * rows + row] = acc[b] + bv;
    }
}

template <int KIND>
static int mmvq_x_run(cudaStream_t s, const uint8_t *W, const int8_t *xq, const float2 *xs, float *Y, const float *bias,
                      int rows, int cols, int batch) {
    const dim3 block(128), grid((rows + 3) / 4);
    switch (batch) {
        case 1: mmvq_x_kernel<KIND, 1><<<grid, block, 0, s>>>(W, xq, xs, Y, bias, rows, cols, batch); break;
        case 2: mmvq_x_kernel<KIND, 2><<<grid, block, 0, s>>>(W, xq, xs, Y, bias, rows, cols, batch); break;
        case 3: mmvq_x_kernel<KIND, 3><<<grid, block, 0, s>>>(W, xq, xs, Y, bias, rows, cols, batch); break;
        default: mmvq_x_kernel<KIND, 4><<<grid, block, 0, s>>>(W, xq, xs, Y, bias, rows, cols, batch); break;
    }
    return 1;
}

int fllm_mmvq_x_try(cudaStream_t s, const void *W, int kind, const float *X, float *Y, const float *bias, int rows, int cols,
                    int batch, void *scratch, size_t scratch_bytes) {
    if (batch < 1 || batch > 4 || (cols & 255) != 0 || !kind_is_iq_or_kextra(kind)) return 0;
    const size_t xq_bytes = ((size_t)batch * cols + 255) & ~size_t(255);
    const size_t need = xq_bytes + (size_t)batch * (cols / 32) * sizeof(float2);
    if ((reinterpret_cast<uintptr_t>(X) & 15) != 0 || (reinterpret_cast<uintptr_t>(W) & 3) != 0 || need > scratch_bytes) return 0;
    fllm_x16_filled = false;
    int8_t *xq = static_cast<int8_t *>(scratch);
    float2 *xs = reinterpret_cast<float2 *>(static_cast<char *>(scratch) + xq_bytes);
    fllm_x8_filled = true;
    if (!fllm_x8_reuse) {
        const size_t groups = (size_t)batch * cols / 32;
        quantize_x8_kernel<<<(unsigned)((groups * 32 + 255) / 256), 256, 0, s>>>(X, xq, xs, groups);
    }
    const uint8_t *w = static_cast<const uint8_t *>(W);
    switch (kind) {
        case KIND_Q2_K: return mmvq_x_run<KIND_Q2_K>(s, w, xq, xs, Y, bias, rows, cols, batch);
        case KIND_Q3_K: return mmvq_x_run<KIND_Q3_K>(s, w, xq, xs, Y, bias, rows, cols, batch);
        case KIND_IQ2_XXS: return mmvq_x_run<KIND_IQ2_XXS>(s, w, xq, xs, Y, bias, rows, cols, batch);
        case KIND_IQ2_XS: return mmvq_x_run<KIND_IQ2_XS>(s, w, xq, xs, Y, bias, rows, cols, batch);
        case KIND_IQ2_S: return mmvq_x_run<KIND_IQ2_S>(s, w, xq, xs, Y, bias, rows, cols, batch);
        case KIND_IQ3_XXS: return mmvq_x_run<KIND_IQ3_XXS>(s, w, xq, xs, Y, bias, rows, cols, batch);
        case KIND_IQ3_S: return mmvq_x_run<KIND_IQ3_S>(s, w, xq, xs, Y, bias, rows, cols, batch);
        case KIND_IQ1_S: return mmvq_x_run<KIND_IQ1_S>(s, w, xq, xs, Y, bias, rows, cols, batch);
        default: return 0;  // IQ1_M: its two 8-value entries of a run carry different deltas, left to the generic kernels
    }
}
