// fLLM CUDA backend: 4-bit weights with group scales and zero points (the Q4Z format, i.e. AWQ and symmetric / asymmetric GPTQ)
// run through the Marlin GEMM of vLLM (marlin_vllm/). The weight lives on the device in the layout that kernel reads:
//   [Marlin-tiled int4 codes, K * N / 2 bytes][fp16 scales, permuted, G * N][int4 zero points, permuted and packed, G * N / 2]
// with every part starting at a multiple of 256 bytes. NVFP4 weights (group 16) carry [Marlin-tiled fp4 codes][fp8 scales in the
// S0E5M3 form of vLLM, G * N][one float: the per-tensor factor] instead. fLLM's activations and results are FP32; they are converted around the call.
#include "fllm_cuda_internal.h"
#include "common.cuh"
#include <cuda_fp16.h>

extern "C" {
size_t fllm_vllm_marlin_workspace_bytes(int n);
size_t fllm_vllm_marlin_ctmp_bytes(int m, int sms);
int fllm_vllm_marlin_gemm(const void *A, const void *B, void *C, void *C_tmp, const void *scales, const void *zeros, const void *bias,
                          const void *global_scale, int m, int n, int k, int lda, void *workspace, int kind, int group_size, int dev,
                          int sms, cudaStream_t stream);
int fllm_vllm_awq_repack(const void *in, void *out, int size_k, int size_n, int num_bits, cudaStream_t stream);
int fllm_vllm_gptq_repack(const void *in, void *out, int size_k, int size_n, int num_bits, cudaStream_t stream);
}

static __global__ void f32_to_f16_kernel(const float4 *__restrict__ x, __half2 *__restrict__ y, size_t n4) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    const float4 v = x[i];
    y[2 * i] = __floats2half2_rn(v.x, v.y);
    y[2 * i + 1] = __floats2half2_rn(v.z, v.w);
}
// gate and up of a gated activation to FP16 in one pass: out = glu(gate, up)
static __global__ void glu_to_f16_kernel(const float4 *__restrict__ gate, const float4 *__restrict__ up, int mode,
                                         __half2 *__restrict__ y, size_t n4) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    const float4 g = gate[i], u = up[i];
    y[2 * i] = __floats2half2_rn(fllm_glu(g.x, u.x, mode), fllm_glu(g.y, u.y, mode));
    y[2 * i + 1] = __floats2half2_rn(fllm_glu(g.z, u.z, mode), fllm_glu(g.w, u.w, mode));
}
// the same for FP16 gate and up (the results of two Marlin GEMMs run with fllm_q4z_f16out), eight values per thread
static __global__ void glu_f16_to_f16_kernel(const uint4 *__restrict__ gate, const uint4 *__restrict__ up, int mode,
                                             uint4 *__restrict__ y, size_t n8) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n8) return;
    const uint4 gv = gate[i], uv = up[i];
    const __half2 *gh = reinterpret_cast<const __half2 *>(&gv), *uh = reinterpret_cast<const __half2 *>(&uv);
    uint4 r;
    __half2 *rh = reinterpret_cast<__half2 *>(&r);
#pragma unroll
    for (int j = 0; j < 4; j++) {
        const float2 g = __half22float2(gh[j]), u = __half22float2(uh[j]);
        rh[j] = __floats2half2_rn(fllm_glu(g.x, u.x, mode), fllm_glu(g.y, u.y, mode));
    }
    y[i] = r;
}
// four results per thread; `rows` is a multiple of 4, so the four share one row of the bias
static __global__ void f16_to_f32_bias_kernel(const __half2 *__restrict__ y, const float *__restrict__ bias, float4 *__restrict__ out,
                                              size_t n4, int rows) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    const float2 a = __half22float2(y[2 * i]), b = __half22float2(y[2 * i + 1]);
    float4 r = make_float4(a.x, a.y, b.x, b.y);
    if (bias) {
        const int base = (int)((i * 4) % (size_t)rows);
        r.x += bias[base];
        r.y += bias[base + 1];
        r.z += bias[base + 2];
        r.w += bias[base + 3];
    }
    out[i] = r;
}

// y += scale * (result + bias): the output projection of a layer added straight into the residual stream
static __global__ void f16_to_f32_acc_kernel(const __half2 *__restrict__ y, const float *__restrict__ bias, float4 *__restrict__ out,
                                             size_t n4, int rows, float scale) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    const float2 a = __half22float2(y[2 * i]), b = __half22float2(y[2 * i + 1]);
    float4 r = make_float4(a.x, a.y, b.x, b.y);
    if (bias) {
        const int base = (int)((i * 4) % (size_t)rows);
        r.x += bias[base];
        r.y += bias[base + 1];
        r.z += bias[base + 2];
        r.w += bias[base + 3];
    }
    float4 o = out[i];
    o.x += scale * r.x;
    o.y += scale * r.y;
    o.z += scale * r.z;
    o.w += scale * r.w;
    out[i] = o;
}

static size_t align256(size_t v) { return (v + 255) & ~size_t(255); }

size_t fllm_q4z_io_bytes(int rows, int cols, int batch) { return align256((size_t)batch * cols * 2) + align256((size_t)batch * rows * 2); }
size_t fllm_q4z_ctmp_bytes(int batch, int sms) { return fllm_vllm_marlin_ctmp_bytes(batch, sms); }
size_t fllm_q4z_workspace_bytes(int rows) { return fllm_vllm_marlin_workspace_bytes(rows); }

bool fllm_q4z_supported(int rows, int cols, int group) {
    return fllm_cc() >= 75 && (group == 16 || group == 32 || group == 64 || group == 128) && cols % 128 == 0 && rows % 128 == 0 &&
           cols / group > 1;
}

template <bool GLU>
static __global__ void convert_to_f16_perm_kernel(const float *__restrict__ x,
                                                 const float *__restrict__ glu_gate,
                                                 const float *__restrict__ glu_up,
                                                 int glu_mode,
                                                 const int *__restrict__ perm,
                                                 __half2 *__restrict__ x16,
                                                 int cols) {
    const int col4 = (int)(blockIdx.x * blockDim.x + threadIdx.x) * 4;
    const int row = (int)blockIdx.y;
    if (col4 >= cols) return;
    const size_t row_base = (size_t)row * cols;
    const int4 p = *reinterpret_cast<const int4 *>(perm + col4);
    float v0, v1, v2, v3;
    if constexpr (GLU) {
        v0 = fllm_glu(glu_gate[row_base + p.x], glu_up[row_base + p.x], glu_mode);
        v1 = fllm_glu(glu_gate[row_base + p.y], glu_up[row_base + p.y], glu_mode);
        v2 = fllm_glu(glu_gate[row_base + p.z], glu_up[row_base + p.z], glu_mode);
        v3 = fllm_glu(glu_gate[row_base + p.w], glu_up[row_base + p.w], glu_mode);
    } else {
        v0 = x[row_base + p.x];
        v1 = x[row_base + p.y];
        v2 = x[row_base + p.z];
        v3 = x[row_base + p.w];
    }
    const size_t out_idx = (row_base + col4) >> 1;
    x16[out_idx] = __floats2half2_rn(v0, v1);
    x16[out_idx + 1] = __floats2half2_rn(v2, v3);
}

int fllm_q4z_launch(cudaStream_t s, int dev, int sms, const void *w, int group, const float *x, float *y, const float *bias, int rows,
                    int cols, int batch, void *io, void *ctmp, void *workspace, const float *glu_gate, const float *glu_up, int glu_mode,
                    bool reuse_x16, const int *perm) {
    if (!fllm_q4z_supported(rows, cols, group) || (reinterpret_cast<uintptr_t>(glu_gate ? glu_gate : x) & 15) != 0) return -1;
    const uint8_t *base = static_cast<const uint8_t *>(w);
    const size_t qbytes = (size_t)cols * rows / 2;
    const bool fp4 = group == 16;
    const void *scales = base + align256(qbytes);
    const void *zeros = fp4 ? nullptr : base + align256(qbytes) + align256((size_t)(cols / group) * rows * 2);
    const void *global = fp4 ? base + align256(qbytes) + align256((size_t)(cols / group) * rows) : nullptr;
    __half *x16 = static_cast<__half *>(io);
    __half *y16 = reinterpret_cast<__half *>(static_cast<uint8_t *>(io) + align256((size_t)batch * cols * 2));
    const size_t n4 = (size_t)batch * cols / 4;
    if (!reuse_x16) {
        if (perm) {
            dim3 grid((unsigned)((cols / 4 + 255) / 256), (unsigned)batch);
            if (glu_gate)
                convert_to_f16_perm_kernel<true><<<grid, 256, 0, s>>>(nullptr, glu_gate, glu_up, glu_mode, perm,
                                                                      reinterpret_cast<__half2 *>(x16), cols);
            else
                convert_to_f16_perm_kernel<false><<<grid, 256, 0, s>>>(x, nullptr, nullptr, 0, perm,
                                                                       reinterpret_cast<__half2 *>(x16), cols);
        } else {
            if (glu_gate && fllm_glu_half)
                glu_f16_to_f16_kernel<<<(unsigned)(((n4 / 2) + 255) / 256), 256, 0, s>>>(reinterpret_cast<const uint4 *>(glu_gate),
                                                                                       reinterpret_cast<const uint4 *>(glu_up), glu_mode,
                                                                                       reinterpret_cast<uint4 *>(x16), n4 / 2);
            else if (glu_gate)
                glu_to_f16_kernel<<<(unsigned)((n4 + 255) / 256), 256, 0, s>>>(reinterpret_cast<const float4 *>(glu_gate),
                                                                              reinterpret_cast<const float4 *>(glu_up), glu_mode,
                                                                              reinterpret_cast<__half2 *>(x16), n4);
            else
                f32_to_f16_kernel<<<(unsigned)((n4 + 255) / 256), 256, 0, s>>>(reinterpret_cast<const float4 *>(x),
                                                                              reinterpret_cast<__half2 *>(x16), n4);
        }
        if (int e = fllm_check_launch("q4z convert")) return e;
    }
    // fllm_q4z_f16out: the caller wants the FP16 result as it is (y points to a buffer of batch x rows halves)
    const int rc = fllm_vllm_marlin_gemm(x16, w, fllm_q4z_f16out ? static_cast<void *>(y) : static_cast<void *>(y16), ctmp, scales, zeros,
                                         nullptr, global, batch, rows, cols, cols, workspace, fp4 ? 3 : 0, group, dev, sms, s);
    if (rc != 0) return -2;
    if (fllm_q4z_f16out) return fllm_check_launch("q4z");
    const size_t count4 = (size_t)batch * rows / 4;
    if (fllm_q4z_acc)
        f16_to_f32_acc_kernel<<<(unsigned)((count4 + 255) / 256), 256, 0, s>>>(reinterpret_cast<const __half2 *>(y16), bias,
                                                                              reinterpret_cast<float4 *>(y), count4, rows,
                                                                              fllm_q4z_acc_scale);
    else
        f16_to_f32_bias_kernel<<<(unsigned)((count4 + 255) / 256), 256, 0, s>>>(reinterpret_cast<const __half2 *>(y16), bias,
                                                                               reinterpret_cast<float4 *>(y), count4, rows);
    return fllm_check_launch("q4z result");
}

// Weight upload: `awq_words` is the int4 matrix [k, n / 8] in the AutoAWQ order; `out` receives the Marlin tiles.
int fllm_q4z_repack(int dev, const void *awq_words, void *out, int k, int n) {
    if (cudaSetDevice(dev) != cudaSuccess) return fllm_set_error("cudaSetDevice(%d) failed", dev);
    if (fllm_vllm_awq_repack(awq_words, out, k, n, 4, 0) != 0) return fllm_set_error("Marlin repack does not support %d x %d", k, n);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return fllm_set_error("Marlin repack failed: %s", cudaGetErrorString(e));
    return 0;
}

// GPTQ-ordered int4 matrix [k / 8, n] (8 consecutive k per word, also how NVFP4 nibbles are stored) to Marlin tiles
int fllm_q4z_gptq_repack(int dev, const void *words, void *out, int k, int n) {
    if (cudaSetDevice(dev) != cudaSuccess) return fllm_set_error("cudaSetDevice(%d) failed", dev);
    if (fllm_vllm_gptq_repack(words, out, k, n, 4, 0) != 0) return fllm_set_error("Marlin repack does not support %d x %d", k, n);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return fllm_set_error("Marlin repack failed: %s", cudaGetErrorString(e));
    return 0;
}

bool fllm_fp8_marlin_supported(int rows, int cols) {
    return fllm_cc() >= 75 && cols % 128 == 0 && rows % 128 == 0;
}

int fllm_fp8_marlin_launch(cudaStream_t s, int dev, int sms, const void *w, const float *x, float *y, const float *bias, int rows,
                           int cols, int batch, void *io, void *ctmp, void *workspace, const float *glu_gate, const float *glu_up,
                           int glu_mode, bool reuse_x16) {
    if (!fllm_fp8_marlin_supported(rows, cols) || (reinterpret_cast<uintptr_t>(glu_gate ? glu_gate : x) & 15) != 0) return -1;
    const uint8_t *base = static_cast<const uint8_t *>(w);
    const size_t qbytes = (size_t)cols * rows;
    const void *scales = base + align256(qbytes);
    __half *x16 = static_cast<__half *>(io);
    __half *y16 = reinterpret_cast<__half *>(static_cast<uint8_t *>(io) + align256((size_t)batch * cols * 2));
    const size_t n4 = (size_t)batch * cols / 4;
    if (!reuse_x16) {
        if (glu_gate)
            glu_to_f16_kernel<<<(unsigned)((n4 + 255) / 256), 256, 0, s>>>(reinterpret_cast<const float4 *>(glu_gate),
                                                                          reinterpret_cast<const float4 *>(glu_up), glu_mode,
                                                                          reinterpret_cast<__half2 *>(x16), n4);
        else
            f32_to_f16_kernel<<<(unsigned)((n4 + 255) / 256), 256, 0, s>>>(reinterpret_cast<const float4 *>(x),
                                                                          reinterpret_cast<__half2 *>(x16), n4);
        if (int e = fllm_check_launch("fp8 marlin convert")) return e;
    }
    const int rc = fllm_vllm_marlin_gemm(x16, w, y16, ctmp, scales, nullptr, nullptr, nullptr, batch, rows, cols, cols, workspace,
                                         4, -1, dev, sms, s);
    if (rc != 0) return -2;
    const size_t count4 = (size_t)batch * rows / 4;
    f16_to_f32_bias_kernel<<<(unsigned)((count4 + 255) / 256), 256, 0, s>>>(reinterpret_cast<const __half2 *>(y16), bias,
                                                                           reinterpret_cast<float4 *>(y), count4, rows);
    return fllm_check_launch("fp8 marlin result");
}

int fllm_fp8_marlin_repack(int dev, const void *words, void *out, int k, int n) {
    if (cudaSetDevice(dev) != cudaSuccess) return fllm_set_error("cudaSetDevice(%d) failed", dev);
    if (fllm_vllm_gptq_repack(words, out, k, n, 8, 0) != 0) return fllm_set_error("Marlin FP8 repack does not support %d x %d", k, n);
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) return fllm_set_error("Marlin FP8 repack failed: %s", cudaGetErrorString(e));
    return 0;
}
