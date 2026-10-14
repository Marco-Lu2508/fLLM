// fLLM CUDA backend: normalisation, rotary embedding, gated activations, residual, KV writes.
#include "common.cuh"
#include "fllm_cuda_internal.h"

__device__ __forceinline__ float block_sum(float v, float *scratch) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
    v = warp_sum(v);
    __syncthreads();
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    float total = 0.f;
    for (int i = 0; i < warps; i++) total += scratch[i];
    return total;
}

// One block per row. out = x * rsqrt(mean(x^2) + eps) * weight
// Rows of up to 8192 columns (multiple of 4) stay in registers: one read of x, one write.
__global__ void rms_norm_kernel(const float *x, const float *w, float *out, int cols, float eps) {
    __shared__ float scratch[32];
    const float *xr = x + (size_t)blockIdx.x * cols;
    float *orow = out + (size_t)blockIdx.x * cols;
    float sum = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) sum = fmaf(xr[c], xr[c], sum);
    const float total = block_sum(sum, scratch);
    const float inv = 1.0f / sqrtf(total / (float)cols + eps);
    for (int c = threadIdx.x; c < cols; c += blockDim.x) orow[c] = xr[c] * (inv * w[c]);
}

template <int PER>
__global__ void __launch_bounds__(256) rms_norm_vec_kernel(const float *x, const float *w, float *out, int cols4,
                                                           float eps, float inv_cols) {
    __shared__ float scratch[32];
    const float4 *xr = reinterpret_cast<const float4 *>(x) + (size_t)blockIdx.x * cols4;
    float4 *orow = reinterpret_cast<float4 *>(out) + (size_t)blockIdx.x * cols4;
    const float4 *w4 = reinterpret_cast<const float4 *>(w);
    float4 v[PER];
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < PER; i++) {
        const int c = threadIdx.x + i * 256;
        v[i] = c < cols4 ? xr[c] : make_float4(0.f, 0.f, 0.f, 0.f);
        sum = fmaf(v[i].x, v[i].x, fmaf(v[i].y, v[i].y, fmaf(v[i].z, v[i].z, fmaf(v[i].w, v[i].w, sum))));
    }
    const float total = block_sum(sum, scratch);
    const float inv = 1.0f / sqrtf(total * inv_cols + eps);
#pragma unroll
    for (int i = 0; i < PER; i++) {
        const int c = threadIdx.x + i * 256;
        if (c < cols4) {
            const float4 g = __ldg(w4 + c);
            orow[c] = make_float4(v[i].x * (inv * g.x), v[i].y * (inv * g.y), v[i].z * (inv * g.z), v[i].w * (inv * g.w));
        }
    }
}

// out = (x - mean) * rsqrt(var + eps) * weight + bias, biased variance (matches the CPU backend).
__global__ void layer_norm_kernel(const float *x, const float *w, const float *b, float *out, int cols, float eps) {
    __shared__ float scratch[32];
    const float *xr = x + (size_t)blockIdx.x * cols;
    float *orow = out + (size_t)blockIdx.x * cols;
    float sum = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) sum += xr[c];
    const float mean = block_sum(sum, scratch) / (float)cols;
    float sq = 0.f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) sq += (xr[c] - mean) * (xr[c] - mean);
    const float var = block_sum(sq, scratch) / (float)cols;
    const float inv = 1.0f / sqrtf(var + eps);
    for (int c = threadIdx.x; c < cols; c += blockDim.x)
        orow[c] = (xr[c] - mean) * inv * w[c] + (b ? b[c] : 0.f);
}

__global__ void glu_kernel(const float *gate, const float *up, float *out, size_t count, int mode) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x) {
        out[i] = fllm_glu(gate[i], up[i], mode);
    }
}

// out = glu(gate, up) of FP16 gate and up, two values per thread (count is even)
__global__ void glu_f16_kernel(const __half2 *gate, const __half2 *up, float2 *out, size_t pairs, int mode) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < pairs; i += (size_t)gridDim.x * blockDim.x) {
        const float2 g = __half22float2(gate[i]), u = __half22float2(up[i]);
        out[i] = make_float2(fllm_glu(g.x, u.x, mode), fllm_glu(g.y, u.y, mode));
    }
}

__global__ void residual_kernel(float *x, const float *res, size_t count, float scale) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x)
        x[i] = x[i] + res[i] * scale;
}

// One thread rotates one pair. Query heads first, then key heads.
__global__ void rope_kernel(float *q, float *k, const int32_t *pos, const float *cs, const float *sn, int rows,
                            int heads, int kv_heads, int dim, int half_, int neox) {
    const int per_row = (heads + kv_heads) * half_;
    const size_t total = (size_t)rows * per_row;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const int r = (int)(i / per_row), rem = (int)(i % per_row);
        const int h = rem / half_, d = rem % half_;
        const int p = pos[r];
        const float c = cs[(size_t)p * half_ + d], s = sn[(size_t)p * half_ + d];
        const int ia = neox ? d : 2 * d, ib = neox ? d + half_ : 2 * d + 1;
        if (h < heads) {
            float *base = q + ((size_t)r * heads + h) * dim;
            const float a = base[ia], b = base[ib];
            float ra, rb;
            rope_pair(a, b, c, s, ra, rb);
            base[ia] = ra;
            base[ib] = rb;
        } else {
            float *base = k + ((size_t)r * kv_heads + (h - heads)) * dim;
            const float a = base[ia], b = base[ib];
            float ra, rb;
            rope_pair(a, b, c, s, ra, rb);
            base[ia] = ra;
            base[ib] = rb;
        }
    }
}

// KV cache write. f16: one value per thread. q8: one thread per group of `group` values:
// d = amax / 127, scale = f16(d), q = round-half-away(x / d) clamped to +-127 (CPU encoder).
__global__ void kv_write_f16_kernel(const float *src, uint16_t *pool, const int32_t *slots, int rows, int kv) {
    const size_t total = (size_t)rows * kv;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const size_t slot = (size_t)slots[i / kv];
        pool[slot * kv + i % kv] = f2h_bits(src[i]);
    }
}

__global__ void kv_write_q8_kernel(const float *src, uint16_t *pool, const int32_t *slots, int rows, int kv,
                                   int group, int scale_base_words) {
    const int groups = kv / group;
    const size_t total = (size_t)rows * groups;
    int8_t *qbytes = reinterpret_cast<int8_t *>(pool);
    uint16_t *scales = pool + scale_base_words;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const int r = (int)(i / groups), g = (int)(i % groups);
        const size_t slot = (size_t)slots[r];
        const float *s = src + (size_t)r * kv + (size_t)g * group;
        float amax = 0.f;
        for (int j = 0; j < group; j++) amax = fmaxf(amax, fabsf(s[j]));
        const float d = amax / 127.0f;
        scales[slot * groups + g] = f2h_bits(d);
        int8_t *dst = qbytes + slot * kv + (size_t)g * group;
        for (int j = 0; j < group; j++) {
            float q = 0.f;
            if (d != 0.f) {
                const float t = s[j] / d;
                q = fminf(fmaxf(copysignf(floorf(fabsf(t) + 0.5f), t), -127.f), 127.f);
            }
            dst[j] = (int8_t)(int)q;
        }
    }
}

static unsigned grid_for(size_t n, int block) {
    size_t g = (n + block - 1) / block;
    return (unsigned)(g > 65535 ? 65535 : (g == 0 ? 1 : g));
}

int fllm_rms_norm_launch(cudaStream_t s, const float *x, const float *w, float *out, int rows, int cols, float eps) {
    if (rows <= 0) return 0;
    const bool aligned = (reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(w) |
                          reinterpret_cast<uintptr_t>(out)) % 16 == 0;
    if (aligned && cols % 4 == 0 && cols <= 8192) {
        const int c4 = cols / 4;
        if (c4 <= 1024) rms_norm_vec_kernel<4><<<rows, 256, 0, s>>>(x, w, out, c4, eps, 1.0f / (float)cols);
        else rms_norm_vec_kernel<8><<<rows, 256, 0, s>>>(x, w, out, c4, eps, 1.0f / (float)cols);
    } else {
        rms_norm_kernel<<<rows, 256, 0, s>>>(x, w, out, cols, eps);
    }
    return fllm_check_launch("rms_norm");
}
int fllm_layer_norm_launch(cudaStream_t s, const float *x, const float *w, const float *b, float *out, int rows,
                           int cols, float eps) {
    if (rows <= 0) return 0;
    layer_norm_kernel<<<rows, 256, 0, s>>>(x, w, b, out, cols, eps);
    return fllm_check_launch("layer_norm");
}
// Greedy sampling on the device (vLLM: Sampler.greedy_sample = logits.argmax(dim=-1)): one block per row, the first index of the
// largest value wins, NaN never does.
__global__ void argmax_rows_kernel(const float *__restrict__ logits, int vocab, int *__restrict__ out) {
    __shared__ float sval[1024];
    __shared__ int sidx[1024];
    const float *row = logits + (size_t)blockIdx.x * vocab;
    float best = -INFINITY;
    int besti = 0;
    if ((vocab & 3) == 0 && (reinterpret_cast<uintptr_t>(row) & 15) == 0) {
        // 16-byte loads: a thread takes four neighbours at a time (the first of equal values wins within a thread as well)
        const float4 *row4 = reinterpret_cast<const float4 *>(row);
        for (int i = threadIdx.x; i < (vocab >> 2); i += blockDim.x) {
            const float4 v = row4[i];
            const int base = i << 2;
            if (v.x > best) { best = v.x; besti = base; }
            if (v.y > best) { best = v.y; besti = base + 1; }
            if (v.z > best) { best = v.z; besti = base + 2; }
            if (v.w > best) { best = v.w; besti = base + 3; }
        }
    } else {
        for (int i = threadIdx.x; i < vocab; i += blockDim.x) {
            const float v = row[i];
            if (v > best) {
                best = v;
                besti = i;
            }
        }
    }
    sval[threadIdx.x] = best;
    sidx[threadIdx.x] = besti;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const float ov = sval[threadIdx.x + stride];
            const int oi = sidx[threadIdx.x + stride];
            if (ov > sval[threadIdx.x] || (ov == sval[threadIdx.x] && oi < sidx[threadIdx.x])) {
                sval[threadIdx.x] = ov;
                sidx[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) out[blockIdx.x] = sidx[0];
}
int fllm_argmax_launch(cudaStream_t s, const float *logits, int rows, int vocab, int *out) {
    argmax_rows_kernel<<<rows, 1024, 0, s>>>(logits, vocab, out);
    return fllm_check_launch("argmax");
}

int fllm_glu_launch(cudaStream_t s, const float *gate, const float *up, float *out, size_t count, int mode) {
    if (count == 0) return 0;
    glu_kernel<<<grid_for(count, 256), 256, 0, s>>>(gate, up, out, count, mode);
    return fllm_check_launch("glu");
}
// Gated attention (Qwen3.5): the query projection holds query and gate of every head, side by side per head (llama.cpp files) or as all
// queries followed by all gates (fLLM's own conversion); attention runs on the queries and its output is multiplied by sigmoid(gate).
__global__ void split_q_gate_kernel(const float *__restrict__ qg, float *__restrict__ q, float *__restrict__ gate, size_t total, int heads,
                                    int dk, int per_head) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const size_t width = (size_t)heads * dk;
        const size_t r = i / width;
        const size_t rem = i % width;
        const size_t h = rem / dk, d = rem % dk;
        const size_t base = r * 2 * width;
        const size_t sq = per_head ? base + h * 2 * dk + d : base + h * dk + d;
        const size_t sg = per_head ? sq + dk : base + width + h * dk + d;
        q[i] = qg[sq];
        gate[i] = qg[sg];
    }
}
__global__ void sigmoid_mul_kernel(float *__restrict__ x, const float *__restrict__ g, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        x[i] *= 1.f / (1.f + expf(-g[i]));
}
int fllm_split_q_gate_launch(cudaStream_t s, const float *qg, float *q, float *gate, int rows, int heads, int dk, int per_head) {
    const size_t total = (size_t)rows * heads * dk;
    if (total == 0) return 0;
    split_q_gate_kernel<<<grid_for(total, 256), 256, 0, s>>>(qg, q, gate, total, heads, dk, per_head);
    return fllm_check_launch("split_q_gate");
}
int fllm_sigmoid_mul_launch(cudaStream_t s, float *x, const float *g, size_t n) {
    if (n == 0) return 0;
    sigmoid_mul_kernel<<<grid_for(n, 256), 256, 0, s>>>(x, g, n);
    return fllm_check_launch("sigmoid_mul");
}
int fllm_glu_launch_f16(cudaStream_t s, const void *gate16, const void *up16, float *out, size_t count, int mode) {
    if (count == 0) return 0;
    glu_f16_kernel<<<grid_for(count / 2, 256), 256, 0, s>>>(static_cast<const __half2 *>(gate16), static_cast<const __half2 *>(up16),
                                                           reinterpret_cast<float2 *>(out), count / 2, mode);
    return fllm_check_launch("glu_f16");
}
int fllm_residual_launch(cudaStream_t s, float *x, const float *res, size_t count, float scale) {
    if (count == 0) return 0;
    residual_kernel<<<grid_for(count, 256), 256, 0, s>>>(x, res, count, scale);
    return fllm_check_launch("residual");
}
// q, k and v as the Marlin GEMMs leave them (FP16, no bias) to what attention and the cache need, in one pass (vLLM runs
// rotary_embedding and reshape_and_cache as two kernels after separate conversions): bias added in FP32, q rotated into `q_out`
// (FP32, read by attention), k rotated and v copied into the f16 cache pools of the layer. Work item = one rotation pair, one
// unrotated tail element or one value element.
__global__ void rope_kv_f16_kernel(const uint16_t *q16, const uint16_t *k16, const uint16_t *v16, const float *qb, const float *kb,
                                   const float *vb, float *q_out, uint16_t *pool_k, uint16_t *pool_v, const int32_t *pos,
                                   const int32_t *slots, const float *cs, const float *sn, int rows, int heads, int kv_heads, int dk,
                                   int dv, int half_, int neox) {
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
                const size_t base = ((size_t)r * heads + h) * dk;
                const float a = h2f_bits(q16[base + ia]) + (qb ? qb[(size_t)h * dk + ia] : 0.f);
                const float b = h2f_bits(q16[base + ib]) + (qb ? qb[(size_t)h * dk + ib] : 0.f);
                q_out[base + ia] = a * c - b * s;
                q_out[base + ib] = b * c + a * s;
            } else {
                const int hk = h - heads;
                const size_t base = ((size_t)r * kv_heads + hk) * dk;
                const float a = h2f_bits(k16[base + ia]) + (kb ? kb[(size_t)hk * dk + ia] : 0.f);
                const float b = h2f_bits(k16[base + ib]) + (kb ? kb[(size_t)hk * dk + ib] : 0.f);
                pool_k[slot * kv_k + (size_t)hk * dk + ia] = f2h_bits(a * c - b * s);
                pool_k[slot * kv_k + (size_t)hk * dk + ib] = f2h_bits(b * c + a * s);
            }
            continue;
        }
        rem -= pairs;
        if (rem < tails) {
            const int h = rem / tail, d = 2 * half_ + rem % tail;
            if (h < heads) {
                const size_t at = ((size_t)r * heads + h) * dk + d;
                q_out[at] = h2f_bits(q16[at]) + (qb ? qb[(size_t)h * dk + d] : 0.f);
            } else {
                const int hk = h - heads;
                const float v = h2f_bits(k16[((size_t)r * kv_heads + hk) * dk + d]) + (kb ? kb[(size_t)hk * dk + d] : 0.f);
                pool_k[slot * kv_k + (size_t)hk * dk + d] = f2h_bits(v);
            }
            continue;
        }
        rem -= tails;
        const float v = h2f_bits(v16[(size_t)r * kv_v + rem]) + (vb ? vb[rem] : 0.f);
        pool_v[slot * kv_v + rem] = f2h_bits(v);
    }
}
int fllm_rope_kv_f16_launch(cudaStream_t s, const uint16_t *q16, const uint16_t *k16, const uint16_t *v16, const float *qb,
                            const float *kb, const float *vb, float *q_out, uint16_t *pool_k, uint16_t *pool_v, const int32_t *pos,
                            const int32_t *slots, const float *cs, const float *sn, int rows, int heads, int kv_heads, int dk, int dv,
                            int half_, int neox) {
    const size_t total = (size_t)rows * ((size_t)(heads + kv_heads) * dk + (size_t)kv_heads * dv - (size_t)(heads + kv_heads) * half_);
    if (total == 0) return 0;
    rope_kv_f16_kernel<<<grid_for(total, 256), 256, 0, s>>>(q16, k16, v16, qb, kb, vb, q_out, pool_k, pool_v, pos, slots, cs, sn, rows,
                                                           heads, kv_heads, dk, dv, half_, neox);
    return fllm_check_launch("rope_kv");
}
int fllm_rope_launch(cudaStream_t s, float *q, float *k, const int32_t *pos, const float *cs, const float *sn, int rows,
                     int heads, int kv_heads, int dim, int half_, int neox) {
    const size_t total = (size_t)rows * (heads + kv_heads) * half_;
    if (total == 0) return 0;
    rope_kernel<<<grid_for(total, 256), 256, 0, s>>>(q, k, pos, cs, sn, rows, heads, kv_heads, dim, half_, neox);
    return fllm_check_launch("rope");
}
int fllm_kv_write_launch(cudaStream_t s, const float *src, uint16_t *pool, const int32_t *slots, int rows, int kv,
                         int kind, int group, int scale_base_words) {
    if (rows <= 0) return 0;
    if (kind == 1) {
        kv_write_f16_kernel<<<grid_for((size_t)rows * kv, 256), 256, 0, s>>>(src, pool, slots, rows, kv);
    } else {
        kv_write_q8_kernel<<<grid_for((size_t)rows * (kv / group), 128), 128, 0, s>>>(src, pool, slots, rows, kv,
                                                                                    group, scale_base_words);
    }
    return fllm_check_launch("kv_write");
}

// ---------------------------------------------------------------------------------------------
// Weights streamed from host memory arrive in the GGUF layout; two formats need their device layout (backend/cuda.rs):
// Q6_K blocks are padded from 210 to 224 bytes, Q8_0 rows become [fp16 scales][int8 quants].
// ---------------------------------------------------------------------------------------------
__global__ void repack_q6k_kernel(uint16_t *dst, const uint16_t *src, size_t words) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < words; i += (size_t)gridDim.x * blockDim.x) {
        const size_t b = i / 112;
        const int w = (int)(i % 112);
        dst[i] = w < 105 ? src[b * 105 + w] : (uint16_t)0;
    }
}

// Q8_0 rows become [fp16 scales][int8 quants] over a width padded to 256; the padding blocks are zero.
__global__ void repack_q8_kernel(uint8_t *dst, const uint8_t *src, size_t rows, size_t nb, size_t nbp) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < rows * nbp; i += (size_t)gridDim.x * blockDim.x) {
        const size_t r = i / nbp, b = i % nbp;
        uint8_t *row = dst + r * nbp * 34;
        uint16_t *dscale = reinterpret_cast<uint16_t *>(row + b * 2);
        uint16_t *q = reinterpret_cast<uint16_t *>(row + nbp * 2 + b * 32);
        if (b < nb) {
            const uint16_t *s = reinterpret_cast<const uint16_t *>(src + (r * nb + b) * 34);
            *dscale = s[0];
#pragma unroll
            for (int k = 0; k < 16; k++) q[k] = s[1 + k];
        } else {
            *dscale = 0;
#pragma unroll
            for (int k = 0; k < 16; k++) q[k] = 0;
        }
    }
}

// MXFP4 blocks [e8m0][16 bytes] become IQ4_NL-style arrays [fp16 d = 2^(e - 128)][quants] over a padded width.
__global__ void repack_mxfp4_kernel(uint8_t *dst, const uint8_t *src, size_t rows, size_t nb, size_t nbp) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < rows * nbp; i += (size_t)gridDim.x * blockDim.x) {
        const size_t r = i / nbp, b = i % nbp;
        uint8_t *row = dst + r * nbp * 18;
        uint8_t *q = row + nbp * 2 + b * 16;
        if (b < nb) {
            const uint8_t *s = src + (r * nb + b) * 17;
            *reinterpret_cast<__half *>(row + b * 2) = __float2half_rn(ldexpf(1.f, (int)s[0] - 128));
#pragma unroll
            for (int k = 0; k < 16; k++) q[k] = s[1 + k];
        } else {
            *reinterpret_cast<uint16_t *>(row + b * 2) = 0;
#pragma unroll
            for (int k = 0; k < 16; k++) q[k] = 0;
        }
    }
}

int fllm_repack_launch(cudaStream_t s, void *dst, const void *src, int kind, size_t rows, size_t cols) {
    if (kind == KIND_Q6_K) {
        const size_t blocks = rows * (cols / 256), words = blocks * 112;
        repack_q6k_kernel<<<grid_for(words, 256), 256, 0, s>>>(static_cast<uint16_t *>(dst), static_cast<const uint16_t *>(src), words);
    } else if (kind == KIND_Q8_0 || kind == KIND_MXFP4) {
        const size_t nb = cols / 32, nbp = (size_t)pad256((int)cols) / 32;
        if (kind == KIND_Q8_0)
            repack_q8_kernel<<<grid_for(rows * nbp, 256), 256, 0, s>>>(static_cast<uint8_t *>(dst), static_cast<const uint8_t *>(src), rows, nb, nbp);
        else
            repack_mxfp4_kernel<<<grid_for(rows * nbp, 256), 256, 0, s>>>(static_cast<uint8_t *>(dst), static_cast<const uint8_t *>(src), rows, nb, nbp);
    } else {
        return fllm_set_error("repack: unsupported weight type %d", kind);
    }
    return fllm_check_launch("repack");
}

// ---------------------------------------------------------------------------------------------
// Row gather / weighted scatter-add for mixture-of-experts layers whose expert weights are streamed from the host.
// ---------------------------------------------------------------------------------------------
__global__ void gather_rows_kernel(float *dst, const float *src, const int32_t *idx, const float *w, int n, int row_len) {
    const size_t total = (size_t)n * row_len;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const int r = (int)(i / row_len), c = (int)(i % row_len);
        const float v = src[(size_t)idx[r] * row_len + c];
        dst[i] = w ? v * w[r] : v;
    }
}

// The rows of one call have distinct destinations, so plain read-modify-write is enough.
__global__ void scatter_add_rows_kernel(float *dst, const float *src, const int32_t *idx, const float *w, int n, int row_len) {
    const size_t total = (size_t)n * row_len;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const int r = (int)(i / row_len), c = (int)(i % row_len);
        dst[(size_t)idx[r] * row_len + c] += (w ? w[r] : 1.f) * src[i];
    }
}

int fllm_gather_rows_launch(cudaStream_t s, float *dst, const float *src, const int32_t *idx, const float *w, int n, int row_len) {
    const size_t total = (size_t)n * row_len;
    gather_rows_kernel<<<grid_for(total, 256), 256, 0, s>>>(dst, src, idx, w, n, row_len);
    return fllm_check_launch("gather_rows");
}
int fllm_scatter_add_rows_launch(cudaStream_t s, float *dst, const float *src, const int32_t *idx, const float *w, int n, int row_len) {
    const size_t total = (size_t)n * row_len;
    scatter_add_rows_kernel<<<grid_for(total, 256), 256, 0, s>>>(dst, src, idx, w, n, row_len);
    return fllm_check_launch("scatter_add_rows");
}

// ---------------------------------------------------------------------------------------------
// Mixture-of-experts routing on the device (softmax gating, top-k by score with ties to the lower index, optional
// renormalisation of the selected scores) and the weighted sum of the experts' outputs.
// ---------------------------------------------------------------------------------------------
__global__ void moe_route_kernel(const float *logits, int32_t *ids, float *w, int experts, int k, int normalize, float scale) {
    const int row = blockIdx.x, lane = threadIdx.x;
    const float *l = logits + (size_t)row * experts;
    float m = -3.0e38f;
    for (int e = lane; e < experts; e += 32) m = fmaxf(m, l[e]);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffffu, m, o));
    float sum = 0.f;
    for (int e = lane; e < experts; e += 32) sum += expf(l[e] - m);
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    int32_t chosen[16];
    float score[16];
    float total = 0.f;
    for (int j = 0; j < k; j++) {
        float best = -3.0e38f;
        int bi = 0x7fffffff;
        for (int e = lane; e < experts; e += 32) {
            bool taken = false;
            for (int i = 0; i < j; i++) taken |= chosen[i] == e;
            if (!taken && (l[e] > best || (l[e] == best && e < bi))) {
                best = l[e];
                bi = e;
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, o);
            if (ob > best || (ob == best && oi < bi)) {
                best = ob;
                bi = oi;
            }
        }
        chosen[j] = bi;
        score[j] = expf(best - m) / sum;
        total += score[j];
    }
    if (lane < k) {
        // every lane holds the full list; lane j writes entry j
        int32_t id = 0;
        float sc = 0.f;
#pragma unroll
        for (int j = 0; j < 16; j++)
            if (j == lane) {
                id = chosen[j];
                sc = score[j];
            }
        ids[(size_t)row * k + lane] = id;
        w[(size_t)row * k + lane] = (normalize ? sc / total : sc) * scale;
    }
}

__global__ void moe_combine_kernel(float *out, const float *y, const float *w, int tokens, int k, int hidden) {
    const size_t total = (size_t)tokens * hidden;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const int t = (int)(i / hidden), h = (int)(i % hidden);
        float acc = 0.f;
        for (int j = 0; j < k; j++) acc += w[(size_t)t * k + j] * y[((size_t)t * k + j) * hidden + h];
        out[i] = acc;
    }
}

int fllm_moe_route_launch(cudaStream_t s, const float *logits, int32_t *ids, float *w, int rows, int experts, int k,
                          int normalize, float scale) {
    if (k < 1 || k > 16 || experts < k) return fllm_set_error("moe_route: unsupported expert counts");
    moe_route_kernel<<<rows, 32, 0, s>>>(logits, ids, w, experts, k, normalize, scale);
    return fllm_check_launch("moe_route");
}
int fllm_moe_combine_launch(cudaStream_t s, float *out, const float *y, const float *w, int tokens, int k, int hidden) {
    const size_t total = (size_t)tokens * hidden;
    moe_combine_kernel<<<grid_for(total, 256), 256, 0, s>>>(out, y, w, tokens, k, hidden);
    return fllm_check_launch("moe_combine");
}

// gpt-oss: out = (min(g, limit) * sigmoid(alpha * min(g, limit))) * (clamp(u, -limit, limit) + 1)
__global__ void swiglu_oai_kernel(const float *gate, const float *up, float *out, size_t count, float alpha, float limit) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x) {
        const float g = fminf(gate[i], limit);
        const float u = fminf(fmaxf(up[i], -limit), limit);
        out[i] = (u + 1.f) * g / (1.f + expf(-alpha * g));
    }
}
int fllm_swiglu_oai_launch(cudaStream_t s, const float *gate, const float *up, float *out, size_t count, float alpha, float limit) {
    swiglu_oai_kernel<<<grid_for(count, 256), 256, 0, s>>>(gate, up, out, count, alpha, limit);
    return fllm_check_launch("swiglu_oai");
}
