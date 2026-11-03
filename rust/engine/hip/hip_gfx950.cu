// fLLM HIP backend: the CDNA4 (gfx950, MI350 / MI355X) matrix instructions that earlier AMD GPUs do not have.
//
//   v_mfma_scale_f32_16x16x128_f8f6f4   16x16x128 matrix product of FP8 / FP6 / FP4 operands (any mix for A and B) with a block scale per
//                                       32 k-values and operand row (E8M0, the OCP microscaling formats), FP32 accumulation
//   v_mfma_f32_16x16x32_f16 / _bf16     twice the k of the 16x16x16 instruction the other CDNA GPUs have
//
// Contents
//   * fllm_gfx950_quant_act_mxfp8   FP32 activations -> MXFP8 (E4M3 + one E8M0 scale per 32 values), software conversion (hip_mx.cuh)
//   * fllm_gfx950_gemm_mxfp4        Y[token][row] = sum_k X[token][k] * W[row][k], W as MXFP4 (E2M1 nibbles + E8M0 per 32), X as MXFP8,
//                                   on the scaled MFMA (gfx950 only; other GPUs get the scalar reference kernel from the same entry point)
//   * fllm_gfx950_selftest          run both against the scalar reference on the device and report the error (the check a person with an
//                                   MI355X runs: `fllm-rs selftest-gpu --device hip:N`)
//
// Packed layouts (also produced by rust/engine/src/mx_ref.rs):
//   Wq   rows x (cols / 2) bytes     element k of a row is the low nibble (k even) or high nibble (k odd) of byte k / 2 (E2M1)
//   Ws   rows x (cols / 32) bytes    E8M0 scale of the 32 values k = 32 j .. 32 j + 31 of a row
//   Xq   tokens x cols bytes         E4M3 (OCP)
//   Xs   tokens x (cols / 32) bytes  E8M0
// `cols` must be a multiple of 128.
//
// Operand layout of the 16x16x128 instruction as used here (AMD matrix instruction calculator for gfx950): lane l holds, for A, the row
// l % 16 and the 32 consecutive k values 32 * (l / 16) .. +31 (FP8: 8 VGPRs of 4 bytes; FP4: 4 VGPRs, first element in the low nibble of
// the lowest byte); for B the column l % 16 and the same k range; the accumulator element i of lane l is D[4 * (l / 16) + i][l % 16].
// The scale operands are one E8M0 byte per lane for the lane's 32 values.
//
// Level of verification: L0. Compiles for gfx942, gfx950 and gfx1100 (the MFMA body only exists for gfx950); the lane layouts above come
// from the AMD documentation and have not been run on hardware, which is exactly what fllm_gfx950_selftest tests.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../cuda/common.cuh"
#include "fllm_hip.h"
#include "hip_mx.cuh"

using namespace fllm_mx;

namespace {

__global__ void quant_act_kernel(const float *X, uint8_t *Xq, uint8_t *Xs, int tokens, int cols) {
    const int groups = cols / 32;
    const size_t total = (size_t)tokens * groups;
    for (size_t g = (size_t)blockIdx.x * blockDim.x + threadIdx.x; g < total; g += (size_t)gridDim.x * blockDim.x) {
        const size_t t = g / groups;
        const int j = (int)(g % groups);
        const float *src = X + t * cols + (size_t)j * 32;
        float amax = 0.0f;
        for (int i = 0; i < 32; i++) amax = fmaxf(amax, fabsf(src[i]));
        const uint32_t e = mx_scale_for_e4m3(amax);
        const float inv = 1.0f / e8m0_to_float(e);
        uint8_t *dst = Xq + t * cols + (size_t)j * 32;
        for (int i = 0; i < 32; i++) dst[i] = (uint8_t)float_to_e4m3_ocp(src[i] * inv);
        Xs[g] = (uint8_t)e;
    }
}

// scalar reference: one thread per output element, dequantizes both operands (any GPU)
__global__ void gemm_ref_kernel(const uint8_t *Wq, const uint8_t *Ws, const uint8_t *Xq, const uint8_t *Xs, float *Y, int rows, int cols, int tokens) {
    const size_t total = (size_t)rows * tokens;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < total; idx += (size_t)gridDim.x * blockDim.x) {
        const int t = (int)(idx / rows), r = (int)(idx % rows);
        float acc = 0.0f;
        for (int j = 0; j < cols / 32; j++) {
            const float ws = e8m0_to_float(Ws[(size_t)r * (cols / 32) + j]), xs = e8m0_to_float(Xs[(size_t)t * (cols / 32) + j]);
            float part = 0.0f;
            for (int i = 0; i < 32; i++) {
                const int k = j * 32 + i;
                const uint32_t byte = Wq[(size_t)r * (cols / 2) + k / 2];
                const float w = e2m1_to_float((k & 1) ? (byte >> 4) : (byte & 15u));
                part += w * e4m3_ocp_to_float(Xq[(size_t)t * cols + k]);
            }
            acc += part * ws * xs;
        }
        Y[(size_t)t * rows + r] = acc;
    }
}

typedef int i32x8 __attribute__((ext_vector_type(8)));
typedef float f32x4 __attribute__((ext_vector_type(4)));

// One wavefront computes a 16 (weight rows) x 16 (tokens) tile of Y over the whole k range. Rows and tokens beyond the matrix read row 0 /
// token 0 and are not stored. The body exists only in the gfx950 device pass (empty elsewhere, as the kernels of hip_gemm.cuh).
__global__ void gemm_mfma_kernel(const uint8_t *Wq, const uint8_t *Ws, const uint8_t *Xq, const uint8_t *Xs, float *Y, int rows, int cols, int tokens) {
#if defined(__gfx950__)
    const int lane = threadIdx.x & 63;
    const int wave = threadIdx.x >> 6;
    const int tile_row = (blockIdx.x * (blockDim.x >> 6) + wave) * 16;
    const int tile_tok = blockIdx.y * 16;
    if (tile_row >= rows) return;
    const int row = tile_row + (lane & 15);
    const int tok = tile_tok + (lane & 15);
    const int lrow = row < rows ? row : 0;
    const int ltok = tok < tokens ? tok : 0;
    const int kgroup = lane >> 4;  // which 32 of the 128 k values of a step this lane holds
    f32x4 acc = {0.f, 0.f, 0.f, 0.f};
    for (int k0 = 0; k0 < cols; k0 += 128) {
        const int k = k0 + kgroup * 32;
        // A: 16 bytes of packed E2M1 -> 4 VGPRs (the other 4 are not read for an FP4 operand)
        const uint4 wv = *reinterpret_cast<const uint4 *>(Wq + (size_t)lrow * (cols / 2) + k / 2);
        i32x8 a = {(int)wv.x, (int)wv.y, (int)wv.z, (int)wv.w, 0, 0, 0, 0};
        // B: 32 bytes of E4M3 -> 8 VGPRs
        const uint4 x0 = *reinterpret_cast<const uint4 *>(Xq + (size_t)ltok * cols + k);
        const uint4 x1 = *reinterpret_cast<const uint4 *>(Xq + (size_t)ltok * cols + k + 16);
        i32x8 b = {(int)x0.x, (int)x0.y, (int)x0.z, (int)x0.w, (int)x1.x, (int)x1.y, (int)x1.z, (int)x1.w};
        const int sa = Ws[(size_t)lrow * (cols / 32) + k / 32];
        const int sb = Xs[(size_t)ltok * (cols / 32) + k / 32];
        // cbsz / blgp select the format of A / B: 0 = E4M3, 1 = E5M2, 2 = E2M3, 3 = E3M2, 4 = E2M1
        acc = __builtin_amdgcn_mfma_scale_f32_16x16x128_f8f6f4(a, b, acc, /*cbsz A*/ 4, /*blgp B*/ 0, /*opsel_a*/ 0, sa, /*opsel_b*/ 0, sb);
    }
    const int out_tok = tile_tok + (lane & 15);
    if (out_tok < tokens) {
        for (int i = 0; i < 4; i++) {
            const int out_row = tile_row + 4 * (lane >> 4) + i;
            if (out_row < rows) Y[(size_t)out_tok * rows + out_row] = acc[i];
        }
    }
#endif
}

bool is_gfx950() { return strncmp(fllm_hip_device().gcn, "gfx950", 6) == 0; }

}  // namespace

extern "C" {

int fllm_gfx950_quant_act_mxfp8(cudaStream_t s, const float *X, uint8_t *Xq, uint8_t *Xs, int tokens, int cols) {
    if (tokens <= 0 || cols <= 0 || cols % 32 != 0) return 0;
    const size_t groups = (size_t)tokens * (cols / 32);
    const unsigned grid = (unsigned)((groups + 255) / 256 < 4096 ? (groups + 255) / 256 : 4096);
    quant_act_kernel<<<grid, 256, 0, s>>>(X, Xq, Xs, tokens, cols);
    return 1;
}

// reference = 1: always the scalar kernel. Otherwise the scaled MFMA on gfx950 and the scalar kernel elsewhere. Returns 0 when the shapes
// are not supported (cols % 128).
int fllm_gfx950_gemm_mxfp4(cudaStream_t s, const uint8_t *Wq, const uint8_t *Ws, const uint8_t *Xq, const uint8_t *Xs, float *Y, int rows, int cols,
                           int tokens, int reference) {
    if (rows <= 0 || tokens <= 0 || cols <= 0 || cols % 128 != 0) return 0;
    if (!reference && is_gfx950()) {
        const int waves = 4;
        dim3 grid((unsigned)((rows + 16 * waves - 1) / (16 * waves)), (unsigned)((tokens + 15) / 16));
        gemm_mfma_kernel<<<grid, waves * 64, 0, s>>>(Wq, Ws, Xq, Xs, Y, rows, cols, tokens);
        return 1;
    }
    const size_t total = (size_t)rows * tokens;
    const unsigned grid = (unsigned)((total + 255) / 256 < 65535 ? (total + 255) / 256 : 65535);
    gemm_ref_kernel<<<grid, 256, 0, s>>>(Wq, Ws, Xq, Xs, Y, rows, cols, tokens);
    return 1;
}

// Self test for `selftest-gpu`: random MXFP4 weights and activations, the scaled MFMA against the scalar reference kernel on the device.
// Returns -1 when the current device is not a gfx950 (nothing to test), 0 when both agree, 1 on a mismatch or error (text in `msg`).
int fllm_gfx950_selftest(char *msg, size_t msg_len) {
    if (!is_gfx950()) {
        snprintf(msg, msg_len, "not a gfx950 device");
        return -1;
    }
    const int rows = 192, cols = 512, tokens = 40;
    std::vector<uint8_t> wq((size_t)rows * cols / 2), ws((size_t)rows * cols / 32), xq((size_t)tokens * cols), xs((size_t)tokens * cols / 32);
    std::vector<float> x((size_t)tokens * cols);
    uint32_t seed = 12345u;
    auto rnd = [&seed]() {
        seed = seed * 1664525u + 1013904223u;
        return (seed >> 8) & 0xffffu;
    };
    for (auto &b : wq) b = (uint8_t)rnd();
    for (auto &b : ws) b = (uint8_t)(120 + rnd() % 14);  // scales 2^-7 .. 2^6
    for (auto &v : x) v = ((float)rnd() / 32768.0f - 1.0f) * 3.0f;
    uint8_t *dwq, *dws, *dxq, *dxs;
    float *dx, *dy, *dr;
    const size_t ysize = (size_t)rows * tokens * sizeof(float);
    if (hipMalloc(&dwq, wq.size()) || hipMalloc(&dws, ws.size()) || hipMalloc(&dxq, xq.size()) || hipMalloc(&dxs, xs.size()) ||
        hipMalloc(&dx, x.size() * 4) || hipMalloc(&dy, ysize) || hipMalloc(&dr, ysize)) {
        snprintf(msg, msg_len, "hipMalloc failed");
        return 1;
    }
    hipMemcpy(dwq, wq.data(), wq.size(), hipMemcpyHostToDevice);
    hipMemcpy(dws, ws.data(), ws.size(), hipMemcpyHostToDevice);
    hipMemcpy(dx, x.data(), x.size() * 4, hipMemcpyHostToDevice);
    fllm_gfx950_quant_act_mxfp8(0, dx, dxq, dxs, tokens, cols);
    fllm_gfx950_gemm_mxfp4(0, dwq, dws, dxq, dxs, dy, rows, cols, tokens, 0);
    fllm_gfx950_gemm_mxfp4(0, dwq, dws, dxq, dxs, dr, rows, cols, tokens, 1);
    std::vector<float> y((size_t)rows * tokens), r((size_t)rows * tokens);
    hipError_t e = hipMemcpy(y.data(), dy, ysize, hipMemcpyDeviceToHost);
    hipMemcpy(r.data(), dr, ysize, hipMemcpyDeviceToHost);
    hipFree(dwq); hipFree(dws); hipFree(dxq); hipFree(dxs); hipFree(dx); hipFree(dy); hipFree(dr);
    if (e != hipSuccess) {
        snprintf(msg, msg_len, "device error: %s", hipGetErrorString(e));
        return 1;
    }
    double worst = 0.0, scale = 0.0;
    size_t at = 0;
    for (size_t i = 0; i < y.size(); i++) {
        scale = fmax(scale, fabs((double)r[i]));
        const double d = fabs((double)y[i] - (double)r[i]);
        if (d > worst) { worst = d; at = i; }
    }
    // both sums are exact products of small floats accumulated in FP32 in a different order: a relative 1e-4 of the largest value is generous
    if (!(worst <= 1e-4 * scale + 1e-6)) {
        snprintf(msg, msg_len, "scaled MFMA differs from the reference: max abs diff %g at %zu (largest value %g)", worst, at, scale);
        return 1;
    }
    snprintf(msg, msg_len, "scaled MFMA (MXFP4 x MXFP8, %dx%dx%d) matches the reference: max abs diff %g (largest value %g)", rows, cols, tokens, worst, scale);
    return 0;
}

}  // extern "C"
