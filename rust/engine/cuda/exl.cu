// fLLM CUDA backend: native EXL2 / EXL3 projections (ExLlamaV2 / ExLlamaV3 quantized weights read in their packed form).
//
// The device buffer of a weight is the blob defined in rust/gguf-probe/src/exl.rs (that file is the layout reference; the CPU kernels in
// rust/engine/src/cpu_exl.rs are the arithmetic reference). Nothing is expanded to 16 bit: the codes are decoded in registers while
// multiplying. These are portable CUDA-core kernels, not tensor-core kernels; they are written from the format definitions in
// docs/exl-formats.md, no ExLlama kernel code is copied.
//
// EXL2: one warp per output row and tile of up to 16 tokens. The lanes split the inputs of every group, decode their code from the
//   row's LSB-first bitstream (the code may straddle two words), gather the activation x[perm[k]] and accumulate; the scale of the group
//   ((s + 1)^2 * max / 256) is applied once per group and the lanes are reduced at the end.
// EXL3: y = diag(svh) * H128 * (R^T * (H128 * diag(suh) * x)). Three kernels: the input transform, the trellis product (one block per
//   16 outputs; the 256 threads each decode one symbol of the 16x16 tail-biting ring tile, the tile is shared, then multiplied against
//   up to 16 tokens per pass) and the output transform with the scatter through the output map. The scratch holds the transformed
//   input and the product for up to 128 tokens per pass.
#include "common.cuh"
#include "fllm_cuda_internal.h"

namespace {

struct ExlHeader {
    uint32_t data_off, row_bytes, groups, param, off_a, off_b, off_c;
};

// Header words (see exl.rs): 0 magic, 1 version, 2 rows, 3 cols, 4-5 total length, 6 data_off, 7 row_bytes, 8 groups, 9 param, 10 off_a,
// 11 off_b, 12 off_c.
__device__ __forceinline__ ExlHeader exl_header(const uint8_t *blob) {
    const uint32_t *h = reinterpret_cast<const uint32_t *>(blob);
    return ExlHeader{h[6], h[7], h[8], h[9], h[10], h[11], h[12]};
}

// ------------------------------------------------------------------------------------------------------------------------- EXL2

template <int TOK>
__global__ void __launch_bounds__(256) exl2_kernel(const uint8_t *__restrict__ blob, const float *__restrict__ x, float *__restrict__ y,
                                                   const float *__restrict__ bias, int rows, int cols, int batch) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int r = blockIdx.x * 8 + warp;
    if (r >= rows) return;  // uniform over the warp: all its lanes share the row
    const int t0 = blockIdx.y * TOK;
    const ExlHeader h = exl_header(blob);
    const uint32_t *row = reinterpret_cast<const uint32_t *>(blob + h.data_off + (size_t)r * h.row_bytes);
    const uint32_t *perm = reinterpret_cast<const uint32_t *>(blob + h.off_b);
    // the 4-bit scale indices follow the bitstreams of all groups, eight per word
    const int scale_word0 = (int)(h.row_bytes >> 2) - (int)((h.groups + 7) >> 3);
    float acc[TOK];
#pragma unroll
    for (int t = 0; t < TOK; t++) acc[t] = 0.f;
    for (uint32_t g = 0; g < h.groups; g++) {
        const uint32_t *e = reinterpret_cast<const uint32_t *>(blob + h.off_a + (size_t)g * 16);
        const int n = (int)e[0];
        const uint32_t bits_max = e[1];
        const int bits = (int)(bits_max & 0xffffu);
        const float max_scaled = __half2float(__ushort_as_half((unsigned short)(bits_max >> 16))) * (1.f / 256.f);
        const uint32_t *words = row + e[2];
        const int start = (int)e[3];
        const uint32_t s = (row[scale_word0 + (g >> 3)] >> (4 * (g & 7))) & 15u;
        const float scale = (float)((s + 1) * (s + 1)) * max_scaled;
        const int zero = 1 << (bits - 1);
        const uint32_t mask = (1u << bits) - 1u;
        float part[TOK];
#pragma unroll
        for (int t = 0; t < TOK; t++) part[t] = 0.f;
        for (int i = lane; i < n; i += 32) {
            const int bit = i * bits;
            const int w = bit >> 5, sh = bit & 31;
            uint32_t v = words[w] >> sh;
            if (sh + bits > 32) v |= words[w + 1] << (32 - sh);  // the code straddles two words
            const float wv = (float)((int)(v & mask) - zero);
            const int src = (int)perm[start + i];
#pragma unroll
            for (int t = 0; t < TOK; t++) {
                if (t0 + t < batch) part[t] += wv * __ldg(x + (size_t)(t0 + t) * cols + src);
            }
        }
#pragma unroll
        for (int t = 0; t < TOK; t++) acc[t] += scale * part[t];
    }
#pragma unroll
    for (int t = 0; t < TOK; t++) acc[t] = warp_sum(acc[t]);
    if (lane == 0) {
#pragma unroll
        for (int t = 0; t < TOK; t++) {
            if (t0 + t < batch) y[(size_t)(t0 + t) * rows + r] = acc[t] + (bias ? bias[r] : 0.f);
        }
    }
}

template <int TOK>
int launch_exl2(cudaStream_t s, const void *w, const float *x, float *y, const float *bias, int rows, int cols, int batch) {
    const dim3 grid((rows + 7) / 8, (batch + TOK - 1) / TOK);
    exl2_kernel<TOK><<<grid, 256, 0, s>>>(static_cast<const uint8_t *>(w), x, y, bias, rows, cols, batch);
    return fllm_check_launch("exl2");
}

// ------------------------------------------------------------------------------------------------------------------------- EXL3

// Codebook value of a 16-bit trellis state (F16 arithmetic of the upstream decoder).
__device__ __forceinline__ float exl3_codebook(uint32_t state, int codebook) {
    if (codebook == 2) {
        const uint32_t x = state * 0x83dcd12du;
        const uint32_t sum = (x & 0xffu) + ((x >> 8) & 0xffu) + ((x >> 16) & 0xffu) + (x >> 24) + 0x6400u;
        const __half r = __hfma(__ushort_as_half((unsigned short)sum), __ushort_as_half((unsigned short)0x1eee), __ushort_as_half((unsigned short)0xc931));
        return __half2float(r);
    }
    uint32_t x = codebook == 1 ? state * 0xcbac1fedu : state * 89226354u + 64248484u;
    x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
    const __half lo = __ushort_as_half((unsigned short)(x & 0xffffu)), hi = __ushort_as_half((unsigned short)(x >> 16));
    return __half2float(__hadd(lo, hi));
}

// The 16-bit window ending at symbol `index` of a tail-biting ring of nw words (256 symbols of `bits` bits, MSB-first).
__device__ __forceinline__ uint32_t exl3_state(const uint32_t *words, int nw, int bits, int index) {
    const int ring = 256 * bits;
    const int start = ((index + 1) * bits + ring - 16) % ring;
    const int w = start >> 5, off = start & 31;
    const uint64_t hi = words[w], lo = words[(w + 1 == nw) ? 0 : w + 1];
    return (uint32_t)((((hi << 32) | lo) >> (48 - off)) & 0xffffu);
}

// Normalized Hadamard transform of the 128 values held by the 128 threads of a block (one value each).
__device__ __forceinline__ float exl3_hadamard128(float *s, int tid, float v) {
    s[tid] = v;
    for (int stride = 1; stride < 128; stride <<= 1) {
        __syncthreads();
        const float other = s[tid ^ stride];
        __syncthreads();
        v = (tid & stride) ? other - v : v + other;
        s[tid] = v;
    }
    __syncthreads();
    return v * 0.08838834764831845f;  // 1 / sqrt(128)
}

// grid (cols / 128, tokens), block 128: x2 = H (suh * x) on every 128-block
__global__ void __launch_bounds__(128) exl3_pre_kernel(const uint8_t *__restrict__ blob, const float *__restrict__ x, float *__restrict__ x2, int cols) {
    __shared__ float s[128];
    const ExlHeader h = exl_header(blob);
    const int tid = threadIdx.x;
    const int i = blockIdx.x * 128 + tid;
    const size_t at = (size_t)blockIdx.y * cols + i;
    const float suh = __half2float(reinterpret_cast<const __half *>(blob + h.off_a)[i]);
    x2[at] = exl3_hadamard128(s, tid, x[at] * suh);
}

// grid (rows / 16), block 256: z[t][16 * no + c] = sum_i x2[t][i] * R[i][16 * no + c]. Thread `tid` decodes symbol `tid` of every tile and
// owns the output column tid & 15 of the token tid >> 4 of each group of 16 tokens.
template <int BT>
__global__ void __launch_bounds__(256) exl3_mm_kernel(const uint8_t *__restrict__ blob, const float *__restrict__ x2, float *__restrict__ z,
                                                      int rows, int cols, int batch) {
    __shared__ float tile[256];
    const ExlHeader h = exl_header(blob);
    const int bits = (int)(h.param & 0xffu), codebook = (int)(h.param >> 8);
    const int nw = 8 * bits;
    const int no = blockIdx.x, tid = threadIdx.x;
    const int c = tid & 15, tl = tid >> 4;
    // where the symbol this thread decodes lands inside the 16x16 tile (tensor-core fragment order of the upstream packer)
    const int lane_i = tid >> 3, j = tid & 7;
    const int pr = 2 * (lane_i & 3) + (j & 1) + ((j & 2) ? 8 : 0);
    const int pc = (lane_i >> 2) + ((j & 4) ? 8 : 0);
    const uint32_t *col_words = reinterpret_cast<const uint32_t *>(blob + h.data_off + (size_t)no * h.row_bytes);
    float acc[BT];
#pragma unroll
    for (int b = 0; b < BT; b++) acc[b] = 0.f;
    for (int ki = 0; ki < cols / 16; ki++) {
        const uint32_t *w = col_words + (size_t)ki * nw;
        tile[pr * 16 + pc] = exl3_codebook(exl3_state(w, nw, bits, tid), codebook);
        __syncthreads();
#pragma unroll
        for (int b = 0; b < BT; b++) {
            const int t = b * 16 + tl;
            if (t < batch) {
                const float *xr = x2 + (size_t)t * cols + ki * 16;
                float sum = 0.f;
#pragma unroll
                for (int r = 0; r < 16; r++) sum += xr[r] * tile[r * 16 + c];
                acc[b] += sum;
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int b = 0; b < BT; b++) {
        const int t = b * 16 + tl;
        if (t < batch) z[(size_t)t * rows + no * 16 + c] = acc[b];
    }
}

// grid (rows / 128, tokens), block 128: y[t][out_map[o]] = svh[o] * (H z)[o] + bias
__global__ void __launch_bounds__(128) exl3_post_kernel(const uint8_t *__restrict__ blob, const float *__restrict__ z, float *__restrict__ y,
                                                        const float *__restrict__ bias, int rows) {
    __shared__ float s[128];
    const ExlHeader h = exl_header(blob);
    const int tid = threadIdx.x;
    const int o = blockIdx.x * 128 + tid;
    float v = exl3_hadamard128(s, tid, z[(size_t)blockIdx.y * rows + o]);
    v *= __half2float(reinterpret_cast<const __half *>(blob + h.off_b)[o]);
    const int row = (int)reinterpret_cast<const uint32_t *>(blob + h.off_c)[o];
    y[(size_t)blockIdx.y * rows + row] = v + (bias ? bias[row] : 0.f);
}

constexpr int EXL3_CHUNK = 128;  // tokens per pass

template <int BT>
int launch_exl3_mm(cudaStream_t s, const void *w, const float *x2, float *z, int rows, int cols, int chunk) {
    exl3_mm_kernel<BT><<<rows / 16, 256, 0, s>>>(static_cast<const uint8_t *>(w), x2, z, rows, cols, chunk);
    return fllm_check_launch("exl3_mm");
}

}  // namespace

size_t fllm_exl3_scratch_bytes(int rows, int cols, int batch) {
    const size_t chunk = batch < EXL3_CHUNK ? (size_t)batch : (size_t)EXL3_CHUNK;
    return chunk * ((size_t)rows + (size_t)cols) * sizeof(float);
}

int fllm_exl_launch(cudaStream_t s, int kind, const void *w, const float *x, float *y, const float *bias, int rows, int cols, int batch,
                    void *scratch, size_t scratch_bytes) {
    if (rows <= 0 || cols <= 0 || batch <= 0) return 0;
    if ((reinterpret_cast<uintptr_t>(w) & 15) != 0) return fllm_set_error("EXL weight is not 16-byte aligned");
    if (kind == 0x80000030) {
        if (batch <= 1) return launch_exl2<1>(s, w, x, y, bias, rows, cols, batch);
        if (batch <= 2) return launch_exl2<2>(s, w, x, y, bias, rows, cols, batch);
        if (batch <= 4) return launch_exl2<4>(s, w, x, y, bias, rows, cols, batch);
        if (batch <= 8) return launch_exl2<8>(s, w, x, y, bias, rows, cols, batch);
        return launch_exl2<16>(s, w, x, y, bias, rows, cols, batch);
    }
    if (kind != 0x80000031) return fllm_set_error("not an EXL weight type: %d", kind);
    if ((rows & 127) != 0 || (cols & 127) != 0) return fllm_set_error("EXL3 needs rows and columns divisible by 128 (%d x %d)", rows, cols);
    if (scratch == nullptr || scratch_bytes < fllm_exl3_scratch_bytes(rows, cols, batch))
        return fllm_set_error("EXL3 scratch too small: %zu bytes", scratch_bytes);
    float *x2 = static_cast<float *>(scratch);
    for (int t0 = 0; t0 < batch; t0 += EXL3_CHUNK) {
        const int chunk = batch - t0 < EXL3_CHUNK ? batch - t0 : EXL3_CHUNK;
        float *z = x2 + (size_t)chunk * cols;
        const float *xin = x + (size_t)t0 * cols;
        float *yout = y + (size_t)t0 * rows;
        exl3_pre_kernel<<<dim3(cols / 128, chunk), 128, 0, s>>>(static_cast<const uint8_t *>(w), xin, x2, cols);
        if (int e = fllm_check_launch("exl3_pre")) return e;
        int rc;
        if (chunk <= 16) rc = launch_exl3_mm<1>(s, w, x2, z, rows, cols, chunk);
        else if (chunk <= 32) rc = launch_exl3_mm<2>(s, w, x2, z, rows, cols, chunk);
        else if (chunk <= 64) rc = launch_exl3_mm<4>(s, w, x2, z, rows, cols, chunk);
        else rc = launch_exl3_mm<8>(s, w, x2, z, rows, cols, chunk);
        if (rc) return rc;
        exl3_post_kernel<<<dim3(rows / 128, chunk), 128, 0, s>>>(static_cast<const uint8_t *>(w), z, yout, bias, rows);
        if (int e = fllm_check_launch("exl3_post")) return e;
    }
    return 0;
}
