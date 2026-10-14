// fLLM CUDA backend: fused decode kernels.
//
// A decode step of one layer is a chain of small kernels (about 500 to 700 launches per generated token). The kernels here replace
// the chains that vLLM, SGLang and FlashInfer also fuse (fused_add_rms_norm, rotary_embedding + reshape_and_cache, silu_and_mul
// + quantization) by one launch each. Every fused kernel repeats the arithmetic of the kernels it replaces expression by expression
// (the residual update of residual_kernel, the sum of squares and the scaling of rms_norm_vec_kernel, the int8 / NVFP4 quantizers of
// mmvq.cu / nvfp4.cu, the rotation of rope_kernel), so its results are bit-identical to the unfused sequence; selftest-gpu compares
// the two. Each fusion has a switch: FLLM_FUSE_ADD_NORM, FLLM_FUSE_NORM_QUANT, FLLM_FUSE_ROPE_KV and FLLM_FUSE_GLU_Q8 =0|1; a switch that is
// not set follows FLLM_FUSE (0 turns every fusion off, default 1), so FLLM_FUSE=0 FLLM_FUSE_ROPE_KV=1 runs that one alone.
#include <atomic>
#include <cstdlib>
#include <cstring>

#include "common.cuh"
#include "fllm_cuda_internal.h"

// ---- switches -------------------------------------------------------------------------------------------------------------------
// FLLM_FUSE_<NAME> decides when it is set; otherwise FLLM_FUSE (default on)
static bool env_flag(const char *name) {
    const char *v = getenv(name);
    if (!v || !*v) {
        const char *all = getenv("FLLM_FUSE");
        if (!all || !*all) return true;
        v = all;
    }
    return !(strcmp(v, "0") == 0 || strcmp(v, "off") == 0 || strcmp(v, "false") == 0 || strcmp(v, "no") == 0);
}
static std::atomic<int> g_fuse_override[FLLM_FUSE_COUNT];  // 0: environment, 1: forced off, 2: forced on
bool fllm_fuse_enabled(int which) {
    static const bool from_env[FLLM_FUSE_COUNT] = {
        env_flag("FLLM_FUSE_ADD_NORM"),
        env_flag("FLLM_FUSE_NORM_QUANT"),
        env_flag("FLLM_FUSE_ROPE_KV"),
        env_flag("FLLM_FUSE_GLU_Q8"),
    };
    if (which < 0 || which >= FLLM_FUSE_COUNT) return false;
    const int o = g_fuse_override[which].load(std::memory_order_relaxed);
    return o == 0 ? from_env[which] : o == 2;
}
// selftest-gpu runs every fused path against the unfused one: on > 0 forces a fusion on, 0 forces it off, < 0 returns to the environment
void fllm_fuse_force(int which, int on) {
    if (which >= 0 && which < FLLM_FUSE_COUNT) g_fuse_override[which].store(on < 0 ? 0 : (on ? 2 : 1), std::memory_order_relaxed);
}

#if !defined(__HIPCC__)
#include "nvfp4_quant.cuh"

// Same reduction as block_sum in elementwise.cu (the sums must come out in the same order).
__device__ __forceinline__ float norm_block_sum(float v, float *scratch) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float total = 0.f;
    for (int i = 0; i < warps; i++) total += scratch[i];
    return total;
}

// quant: what the GEMM that follows reads (the norm output itself is always written as FP32 as well)
//   0 nothing, 1 FP16 activations of the Marlin GEMMs (row-major halves in q0), 2 NVFP4 (nibbles in q0, UE4M3 scales in q1, the layout
//   of nvfp4_quantize_kernel), 3 int8 with one float2 {d, d * sum(q)} per 32 values (q0, q1; the layout of quantize_x8_kernel)
// RES: 1 FP32 branch (x += scale * branch, the residual_kernel), 2 FP16 branch plus optional bias (x += scale * (branch + bias), the
//   f16_to_f32_acc_kernel of the Marlin GEMMs). The updated x is written back. One block per row, rows of up to 8192 values.
template <int PER, int RES, int QUANT>
__global__ void __launch_bounds__(256)
add_rms_norm_kernel(float *__restrict__ x, const void *__restrict__ branch, const float *__restrict__ rbias, float scale,
                    const float *__restrict__ w, float *__restrict__ out, void *__restrict__ q0, void *__restrict__ q1, int cols,
                    float eps, float inv_cols) {
    __shared__ float scratch[32];
    const int cols4 = cols >> 2;
    const int lane = threadIdx.x & 31;
    float4 *xr = reinterpret_cast<float4 *>(x) + (size_t)blockIdx.x * cols4;
    float4 *orow = reinterpret_cast<float4 *>(out) + (size_t)blockIdx.x * cols4;
    const float4 *w4 = reinterpret_cast<const float4 *>(w);
    float4 v[PER];
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < PER; i++) {
        const int c = threadIdx.x + i * 256;
        float4 t = make_float4(0.f, 0.f, 0.f, 0.f);
        if (c < cols4) {
            t = xr[c];
            if (RES == 1) {
                const float4 r = reinterpret_cast<const float4 *>(branch)[(size_t)blockIdx.x * cols4 + c];
                t.x = t.x + r.x * scale;
                t.y = t.y + r.y * scale;
                t.z = t.z + r.z * scale;
                t.w = t.w + r.w * scale;
            } else {
                const __half2 *h = reinterpret_cast<const __half2 *>(branch) + (((size_t)blockIdx.x * cols + (size_t)c * 4) >> 1);
                const float2 a = __half22float2(h[0]), b = __half22float2(h[1]);
                float4 r = make_float4(a.x, a.y, b.x, b.y);
                if (rbias) {
                    const float4 bb = reinterpret_cast<const float4 *>(rbias)[c];
                    r.x += bb.x;
                    r.y += bb.y;
                    r.z += bb.z;
                    r.w += bb.w;
                }
                t.x += scale * r.x;
                t.y += scale * r.y;
                t.z += scale * r.z;
                t.w += scale * r.w;
            }
            xr[c] = t;
        }
        v[i] = t;
        sum = fmaf(v[i].x, v[i].x, fmaf(v[i].y, v[i].y, fmaf(v[i].z, v[i].z, fmaf(v[i].w, v[i].w, sum))));
    }
    const float total = norm_block_sum(sum, scratch);
    const float inv = 1.0f / sqrtf(total * inv_cols + eps);
#pragma unroll
    for (int i = 0; i < PER; i++) {
        const int c = threadIdx.x + i * 256;
        const bool live = c < cols4;
        const float4 g = live ? __ldg(w4 + c) : make_float4(0.f, 0.f, 0.f, 0.f);
        const float4 o = make_float4(v[i].x * (inv * g.x), v[i].y * (inv * g.y), v[i].z * (inv * g.z), v[i].w * (inv * g.w));
        if (live) orow[c] = o;
        if (QUANT == 1) {
            if (live) {
                __half2 *y = reinterpret_cast<__half2 *>(q0) + (((size_t)blockIdx.x * cols + (size_t)c * 4) >> 1);
                y[0] = __floats2half2_rn(o.x, o.y);
                y[1] = __floats2half2_rn(o.z, o.w);
            }
        } else if (QUANT == 3) {
            // eight lanes hold the 32 values of one group (c is the float4 index, 256 is a multiple of 8)
            float amax = fmaxf(fmaxf(fabsf(o.x), fabsf(o.y)), fmaxf(fabsf(o.z), fabsf(o.w)));
#pragma unroll
            for (int off = 4; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
            const float dx = amax / 127.f;
            int qv[4];
            const float ov[4] = {o.x, o.y, o.z, o.w};
#pragma unroll
            for (int k = 0; k < 4; k++) qv[k] = amax > 0.f ? __float2int_rn(ov[k] * (127.f / amax)) : 0;
            int s = qv[0] + qv[1] + qv[2] + qv[3];
#pragma unroll
            for (int off = 4; off > 0; off >>= 1) s += __shfl_xor_sync(0xffffffffu, s, off);
            if (live) {
                const uint32_t word = (uint32_t)(qv[0] & 0xff) | ((uint32_t)(qv[1] & 0xff) << 8) | ((uint32_t)(qv[2] & 0xff) << 16) |
                                      ((uint32_t)(qv[3] & 0xff) << 24);
                reinterpret_cast<uint32_t *>(q0)[(size_t)blockIdx.x * cols4 + c] = word;
                if ((lane & 7) == 0)
                    reinterpret_cast<float2 *>(q1)[(size_t)blockIdx.x * (cols >> 5) + (c >> 3)] = make_float2(dx, dx * (float)s);
            }
        } else if (QUANT == 2) {
            // four lanes hold the 16 values of one NVFP4 block
            float amax = fmaxf(fmaxf(fabsf(o.x), fabsf(o.y)), fmaxf(fabsf(o.z), fabsf(o.w)));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
            const uint32_t sb = f_to_ue4m3(amax * (1.f / 6.f));
            const float sd = ue4m3_to_f(sb);
            const float inv4 = sd > 0.f ? 1.f / sd : 0.f;
            const float ov[4] = {o.x, o.y, o.z, o.w};
            uint32_t h16 = 0;
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const float t = ov[k] * inv4;
                const uint32_t nib = e2m1_idx(fabsf(t)) | (t < 0.f ? 8u : 0u);
                h16 |= nib << (4 * k);
            }
            const uint32_t other = __shfl_down_sync(0xffffffffu, h16, 1);
            if (live) {
                const size_t bid = (size_t)blockIdx.x * (cols >> 4) + (c >> 2);
                if ((lane & 1) == 0) {
                    // word h of block bid: elements 8h .. 8h + 7 = the nibbles of lanes 2h and 2h + 1
                    const size_t h = (lane >> 1) & 1;
                    const size_t j = (bid * 2 + h) & 7, base = (bid * 2 + h) - j;
                    reinterpret_cast<uint32_t *>(q0)[base + (j < 4 ? 2 * j : 2 * (j - 4) + 1)] = h16 | (other << 16);
                }
                if ((lane & 3) == 0) reinterpret_cast<uint8_t *>(q1)[bid] = (uint8_t)sb;
            }
        }
    }
}

template <int PER, int RES>
static void add_rms_norm_dispatch(cudaStream_t s, int quant, int rows, float *x, const void *branch, const float *rbias, float scale,
                                  const float *w, float *out, void *q0, void *q1, int cols, float eps) {
    const float inv_cols = 1.0f / (float)cols;
#define FLLM_ARN(Q) add_rms_norm_kernel<PER, RES, Q><<<rows, 256, 0, s>>>(x, branch, rbias, scale, w, out, q0, q1, cols, eps, inv_cols)
    switch (quant) {
        case 1: FLLM_ARN(1); break;
        case 2: FLLM_ARN(2); break;
        case 3: FLLM_ARN(3); break;
        default: FLLM_ARN(0); break;
    }
#undef FLLM_ARN
}

// Returns 1 when the shape or the alignment is not supported (nothing was launched), 0 after the launch.
int fllm_add_rms_norm_launch(cudaStream_t s, float *x, const void *branch, int res_kind, const float *rbias, float scale, const float *w,
                             float *out, int rows, int cols, float eps, int quant, void *q0, void *q1) {
    if (rows <= 0 || cols <= 0 || (cols & 3) != 0 || cols > 8192 || (res_kind != 1 && res_kind != 2)) return 1;
    if ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(branch) | reinterpret_cast<uintptr_t>(w) |
         reinterpret_cast<uintptr_t>(out) | reinterpret_cast<uintptr_t>(rbias)) % 16 != 0)
        return 1;
    if (quant == 1 && (reinterpret_cast<uintptr_t>(q0) & 15) != 0) return 1;
    if (quant == 2 && ((cols & 255) != 0 || (reinterpret_cast<uintptr_t>(q0) & 15) != 0)) return 1;
    if (quant == 3 && ((cols & 31) != 0 || (reinterpret_cast<uintptr_t>(q0) & 15) != 0 || (reinterpret_cast<uintptr_t>(q1) & 15) != 0)) return 1;
    const bool wide = cols / 4 > 1024;
    if (res_kind == 1) {
        if (wide) add_rms_norm_dispatch<8, 1>(s, quant, rows, x, branch, rbias, scale, w, out, q0, q1, cols, eps);
        else add_rms_norm_dispatch<4, 1>(s, quant, rows, x, branch, rbias, scale, w, out, q0, q1, cols, eps);
    } else {
        if (wide) add_rms_norm_dispatch<8, 2>(s, quant, rows, x, branch, rbias, scale, w, out, q0, q1, cols, eps);
        else add_rms_norm_dispatch<4, 2>(s, quant, rows, x, branch, rbias, scale, w, out, q0, q1, cols, eps);
    }
    return 0;
}

// ---- rotary embedding + KV cache write for FP32 q, k, v and an f16 cache -------------------------------------------------------------
// rope_pair (common.cuh) is the arithmetic of rope_kernel: one definition for both, so the compiler contracts it the same way.
// q is rotated in place; k is rotated and written (as half) into the pool together with v. Work item = one rotation pair, one
// unrotated tail element or one value element (the layout of rope_kv_f16_kernel).
__global__ void rope_kv_f32_kernel(float *__restrict__ q, const float *__restrict__ k, const float *__restrict__ v,
                                   uint16_t *__restrict__ pool_k, uint16_t *__restrict__ pool_v, const int32_t *__restrict__ pos,
                                   const int32_t *__restrict__ slots, const float *__restrict__ cs, const float *__restrict__ sn,
                                   int rows, int heads, int kv_heads, int dk, int dv, int half_, int neox) {
    const int pairs = (heads + kv_heads) * half_;
    const int tail = dk - 2 * half_;
    const int tails = (heads + kv_heads) * tail;
    const int vcount = kv_heads * dv;
    const int per_row = pairs + tails + vcount;
    const size_t total = (size_t)rows * per_row;
    const int kv_k = kv_heads * dk, kv_v = kv_heads * dv;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const int r = (int)(i / per_row);
        int rem = (int)(i % per_row);
        const size_t slot = (size_t)slots[r];
        if (rem < pairs) {
            const int h = rem / half_, d = rem % half_;
            const int p = pos[r];
            const float c = cs[(size_t)p * half_ + d], s = sn[(size_t)p * half_ + d];
            const int ia = neox ? d : 2 * d, ib = neox ? d + half_ : 2 * d + 1;
            if (h < heads) {
                float *base = q + ((size_t)r * heads + h) * dk;
                const float a = base[ia], b = base[ib];
                float ra, rb;
                rope_pair(a, b, c, s, ra, rb);
                base[ia] = ra;
                base[ib] = rb;
            } else {
                const int hk = h - heads;
                const float *base = k + ((size_t)r * kv_heads + hk) * dk;
                const float a = base[ia], b = base[ib];
                float ra, rb;
                rope_pair(a, b, c, s, ra, rb);
                pool_k[slot * kv_k + (size_t)hk * dk + ia] = f2h_bits(ra);
                pool_k[slot * kv_k + (size_t)hk * dk + ib] = f2h_bits(rb);
            }
            continue;
        }
        rem -= pairs;
        if (rem < tails) {
            // the unrotated tail of k goes to the cache as it is; the tail of q stays where it is
            const int h = rem / tail, d = 2 * half_ + rem % tail;
            if (h >= heads) {
                const int hk = h - heads;
                pool_k[slot * kv_k + (size_t)hk * dk + d] = f2h_bits(k[((size_t)r * kv_heads + hk) * dk + d]);
            }
            continue;
        }
        rem -= tails;
        pool_v[slot * kv_v + rem] = f2h_bits(v[(size_t)r * kv_v + rem]);
    }
}

int fllm_rope_kv_f32_launch(cudaStream_t s, float *q, const float *k, const float *v, uint16_t *pool_k, uint16_t *pool_v,
                            const int32_t *pos, const int32_t *slots, const float *cs, const float *sn, int rows, int heads,
                            int kv_heads, int dk, int dv, int half_, int neox) {
    const size_t total = (size_t)rows * ((size_t)(heads + kv_heads) * dk + (size_t)kv_heads * dv - (size_t)(heads + kv_heads) * half_);
    if (total == 0) return 0;
    const size_t blocks = (total + 255) / 256;
    rope_kv_f32_kernel<<<(unsigned)(blocks > 65535 ? 65535 : blocks), 256, 0, s>>>(q, k, v, pool_k, pool_v, pos, slots, cs, sn, rows,
                                                                                   heads, kv_heads, dk, dv, half_, neox);
    return fllm_check_launch("rope_kv_f32");
}

// ---- GLU + int8 quantization -----------------------------------------------------------------------------------------------------------
// quantize_x8_kernel of mmvq.cu with x = glu(gate, up): one warp per group of 32 values.
__global__ void quantize_x8_glu_kernel(const float *__restrict__ gate, const float *__restrict__ up, int mode, int8_t *__restrict__ xq,
                                       float2 *__restrict__ xs, size_t groups) {
    const size_t gid = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (gid >= groups) return;
    const float v = fllm_glu(gate[gid * 32 + lane], up[gid * 32 + lane], mode);
    float amax = fabsf(v);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    const float dx = amax / 127.f;
    const int q = amax > 0.f ? __float2int_rn(v * (127.f / amax)) : 0;
    int s = q;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    xq[gid * 32 + lane] = (int8_t)q;
    if (lane == 0) xs[gid] = make_float2(dx, dx * (float)s);
}
void fllm_quantize_x8_glu_launch(cudaStream_t s, const float *gate, const float *up, int mode, int8_t *xq, float2 *xs, size_t groups) {
    quantize_x8_glu_kernel<<<(unsigned)((groups * 32 + 255) / 256), 256, 0, s>>>(gate, up, mode, xq, xs, groups);
}
#endif  // !__HIPCC__
