// fLLM CUDA backend: the attention of DeepSeek-V4 (the reference is `v4_attention` of rust_decoder/deepseek4.rs).
//
//   v4_rope      rotary embedding of the trailing `rope` values of groups of `dim` floats, rows x groups x dim, in place; cs holds (cos, sin) per row and pair
//   v4_compress  one block per run: writes the projections of every row into the buffer of the last `ratio` tokens (slot position % ratio) and pools every
//                complete window into an entry (softmax over gate + position bias weighting kv, the first series of the previous window in front for a
//                two-series compressor, RMS norm, rotation at the window's first position); keeps that first series as the overlap of the next window
//   v4_attn      one block per (row, head): the last `window` key vectors (batch rows or the ring of the state), the compressed entries (all that the row
//                sees, or those listed in `sel`), a per-head sink, online softmax; K = V; the rotation of the trailing values is undone and the output is
//                written group-major
//   v4_ring      the last `window` key vectors of a run go to the ring (slot position % window)
//
// FP32 CUDA-core code with block-level primitives only (no warp intrinsics), so the same file builds for AMD GPUs.
#include "common.cuh"
#include "fllm_cuda_internal.h"

#define V4_THREADS 128
#define V4_CHUNK 128
#define V4_MAX_HD 512

__global__ void v4_rope_kernel(float *__restrict__ x, const float *__restrict__ cs, size_t pairs, int groups, int dim, int rope, float sign) {
    const int half = rope / 2;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < pairs; i += (size_t)gridDim.x * blockDim.x) {
        const size_t g = i / half;
        const int j = (int)(i % half);
        const size_t r = g / groups;
        const float c = cs[(r * half + j) * 2], s = cs[(r * half + j) * 2 + 1] * sign;
        float *v = x + g * dim + (dim - rope) + 2 * j;
        const float a = v[0], b = v[1];
        v[0] = a * c - b * s;
        v[1] = b * c + a * s;
    }
}

__device__ __forceinline__ float v4_block_max(float v, float *red) {
    const int t = threadIdx.x;
    red[t] = v;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (t < off) red[t] = fmaxf(red[t], red[t + off]);
        __syncthreads();
    }
    const float out = red[0];
    __syncthreads();
    return out;
}
__device__ __forceinline__ float v4_block_sum(float v, float *red) {
    const int t = threadIdx.x;
    red[t] = v;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (t < off) red[t] += red[t + off];
        __syncthreads();
    }
    const float out = red[0];
    __syncthreads();
    return out;
}

// dynamic shared memory: blockDim.x floats
__global__ void v4_compress_kernel(float *__restrict__ state, const float *__restrict__ kvp, const float *__restrict__ gatep, const float *__restrict__ pos,
                                   const float *__restrict__ norm, const float *__restrict__ cs, int row0, int n, int position0, int dim, int coff, int ratio,
                                   int rope, float eps, size_t off_buf_kv, size_t off_buf_gate, size_t off_ov_kv, size_t off_ov_gate, size_t off_entries) {
    extern __shared__ float v4_red[];
    const int t = threadIdx.x, w = coff * dim, half = rope / 2;
    float *buf_kv = state + off_buf_kv, *buf_gate = state + off_buf_gate, *ov_kv = state + off_ov_kv, *ov_gate = state + off_ov_gate;
    float *entries = state + off_entries;
    for (int i = 0; i < n; i++) {
        const int p = position0 + i, row = row0 + i, slot = p % ratio;
        for (int c = t; c < w; c += blockDim.x) {
            buf_kv[(size_t)slot * w + c] = kvp[(size_t)row * w + c];
            buf_gate[(size_t)slot * w + c] = gatep[(size_t)row * w + c];
        }
        __syncthreads();
        if ((p + 1) % ratio != 0) continue;
        const int e = (p + 1) / ratio - 1;
        float *entry = entries + (size_t)e * dim;
        const int off = coff == 2 ? dim : 0;
        float ss = 0.f;
        for (int c = t; c < dim; c += blockDim.x) {
            float mx = -3.0e38f;
            if (coff == 2 && e > 0) {
                for (int j = 0; j < ratio; j++) mx = fmaxf(mx, ov_gate[(size_t)j * dim + c]);
            }
            for (int j = 0; j < ratio; j++) mx = fmaxf(mx, buf_gate[(size_t)j * w + off + c] + pos[(size_t)j * w + off + c]);
            float sum = 0.f, acc = 0.f;
            if (coff == 2 && e > 0) {
                for (int j = 0; j < ratio; j++) {
                    const float x = expf(ov_gate[(size_t)j * dim + c] - mx);
                    sum += x;
                }
            }
            for (int j = 0; j < ratio; j++) sum += expf(buf_gate[(size_t)j * w + off + c] + pos[(size_t)j * w + off + c] - mx);
            if (coff == 2 && e > 0) {
                for (int j = 0; j < ratio; j++) acc += expf(ov_gate[(size_t)j * dim + c] - mx) / sum * ov_kv[(size_t)j * dim + c];
            }
            for (int j = 0; j < ratio; j++) acc += expf(buf_gate[(size_t)j * w + off + c] + pos[(size_t)j * w + off + c] - mx) / sum * buf_kv[(size_t)j * w + off + c];
            entry[c] = acc;
            ss += acc * acc;
        }
        const float total = v4_block_sum(ss, v4_red);
        const float inv = rsqrtf(total / (float)dim + eps);
        for (int c = t; c < dim; c += blockDim.x) entry[c] = entry[c] * inv * norm[c];
        __syncthreads();
        for (int j = t; j < half; j += blockDim.x) {
            const float cc = cs[((size_t)row * half + j) * 2], sn = cs[((size_t)row * half + j) * 2 + 1];
            float *v = entry + (dim - rope) + 2 * j;
            const float a = v[0], b = v[1];
            v[0] = a * cc - b * sn;
            v[1] = b * cc + a * sn;
        }
        if (coff == 2) {
            for (int idx = t; idx < ratio * dim; idx += blockDim.x) {
                const int j = idx / dim, c = idx % dim;
                ov_kv[idx] = buf_kv[(size_t)j * w + c];
                ov_gate[idx] = buf_gate[(size_t)j * w + c] + pos[(size_t)j * w + c];
            }
        }
        __syncthreads();
    }
}

// grid (heads, rows of the run piece starting at row0 / position0 of the run that starts at run_row0 / run_position0); dynamic shared memory: hd query values, V4_CHUNK scores, blockDim.x reduction values
__global__ void v4_attn_kernel(float *__restrict__ out, const float *__restrict__ q, const float *__restrict__ kv, const float *__restrict__ state,
                               const float *__restrict__ sinks, const int32_t *__restrict__ sel, const float *__restrict__ cs, int row0, int position0, int run_row0,
                               int run_position0, int heads, int hd, int rope, int window, int ratio, int topk, int hpg, int rows_total, size_t off_ring,
                               size_t off_entries, float scale) {
    extern __shared__ float v4_sm[];
    float *qs = v4_sm;
    float *sc = qs + hd;
    float *red = sc + V4_CHUNK;
    __shared__ const float *kp[V4_CHUNK];
    const int h = blockIdx.x, t = threadIdx.x;
    const int row = row0 + blockIdx.y;
    const int p = position0 + blockIdx.y;
    const int nwin = min(p + 1, window), wfirst = p + 1 - nwin;
    const int cnt = ratio > 0 ? (p + 1) / ratio : 0;
    const float *ring = state + off_ring, *entries = state + off_entries;
    for (int d = t; d < hd; d += blockDim.x) qs[d] = q[((size_t)row * heads + h) * hd + d];
    int nent = cnt;
    if (sel) {
        int local = 0;
        for (int j = t; j < topk; j += blockDim.x) {
            const int s = sel[(size_t)row * topk + j];
            local += (s >= 0 && s < cnt);
        }
        nent = (int)v4_block_sum((float)local, red);
    }
    const int nk = nwin + nent;
    float m = sinks[h], l = 1.f;
    float acc[V4_MAX_HD / V4_THREADS];
    for (int i = 0; i < V4_MAX_HD / V4_THREADS; i++) acc[i] = 0.f;
    __syncthreads();
    for (int base = 0; base < nk; base += V4_CHUNK) {
        const int cn = min(V4_CHUNK, nk - base);
        if (t < cn) {
            const int j = base + t;
            if (j < nwin) {
                const int qpos = wfirst + j;
                kp[t] = qpos >= run_position0 ? kv + (size_t)(run_row0 + qpos - run_position0) * hd : ring + (size_t)(qpos % window) * hd;
            } else {
                const int e = sel ? sel[(size_t)row * topk + (j - nwin)] : (j - nwin);
                kp[t] = entries + (size_t)e * hd;
            }
        }
        __syncthreads();
        float s = -3.0e38f;
        if (t < cn) {
            float dot = 0.f;
            for (int d = 0; d < hd; d++) dot += qs[d] * kp[t][d];
            s = dot * scale;
        }
        const float cm = v4_block_max(s, red);
        const float mn = fmaxf(m, cm);
        const float a = expf(m - mn);
        const float pr = t < cn ? expf(s - mn) : 0.f;
        sc[t] = pr;
        const float lsum = v4_block_sum(pr, red);
        l = l * a + lsum;
        for (int i = 0; i < V4_MAX_HD / V4_THREADS; i++) {
            const int d = t + i * V4_THREADS;
            if (d < hd) {
                float v = acc[i] * a;
                for (int j = 0; j < cn; j++) v += sc[j] * kp[j][d];
                acc[i] = v;
            }
        }
        m = mn;
        __syncthreads();
    }
    // the result, then the inverse rotation of the trailing values; qs is free again
    for (int i = 0; i < V4_MAX_HD / V4_THREADS; i++) {
        const int d = t + i * V4_THREADS;
        if (d < hd) qs[d] = acc[i] / l;
    }
    __syncthreads();
    const int g = h / hpg, lh = h % hpg, half = rope / 2;
    float *dst = out + ((size_t)g * rows_total + row) * ((size_t)hpg * hd) + (size_t)lh * hd;
    for (int d = t; d < hd - rope; d += blockDim.x) dst[d] = qs[d];
    for (int j = t; j < half; j += blockDim.x) {
        const float c = cs[((size_t)row * half + j) * 2], s = -cs[((size_t)row * half + j) * 2 + 1];
        const float a = qs[hd - rope + 2 * j], b = qs[hd - rope + 2 * j + 1];
        dst[hd - rope + 2 * j] = a * c - b * s;
        dst[hd - rope + 2 * j + 1] = b * c + a * s;
    }
}

__global__ void v4_ring_kernel(float *__restrict__ ring, const float *__restrict__ kv, int row0, int n, int position0, int hd, int window) {
    const int m = min(n, window), first = n - m;
    for (size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x; idx < (size_t)m * hd; idx += (size_t)gridDim.x * blockDim.x) {
        const int i = first + (int)(idx / hd), d = (int)(idx % hd);
        ring[(size_t)((position0 + i) % window) * hd + d] = kv[(size_t)(row0 + i) * hd + d];
    }
}

static unsigned v4_grid(size_t n) {
    size_t g = (n + 255) / 256;
    return (unsigned)(g > 65535 ? 65535 : (g == 0 ? 1 : g));
}

int fllm_v4_supported(int head_dim, int rope, int window, int heads, int o_groups) {
    return head_dim > 0 && head_dim <= V4_MAX_HD && rope > 0 && rope <= head_dim && (rope & 1) == 0 && window > 0 && heads > 0 && o_groups > 0 && heads % o_groups == 0;
}

int fllm_v4_rope_launch(cudaStream_t s, float *x, const float *cs, int rows, int groups, int dim, int rope, float sign) {
    if (rows <= 0 || groups <= 0) return 0;
    const size_t pairs = (size_t)rows * groups * (rope / 2);
    v4_rope_kernel<<<v4_grid(pairs), 256, 0, s>>>(x, cs, pairs, groups, dim, rope, sign);
    return fllm_check_launch("v4_rope");
}

int fllm_v4_compress_launch(cudaStream_t s, float *state, const float *kv, const float *gate, const float *pos, const float *norm, const float *cs, int row0, int n,
                            int position0, int dim, int coff, int ratio, int rope, float eps, size_t off_buf_kv, size_t off_buf_gate, size_t off_ov_kv,
                            size_t off_ov_gate, size_t off_entries) {
    if (n <= 0) return 0;
    v4_compress_kernel<<<1, V4_THREADS, V4_THREADS * sizeof(float), s>>>(state, kv, gate, pos, norm, cs, row0, n, position0, dim, coff, ratio, rope, eps, off_buf_kv,
                                                                       off_buf_gate, off_ov_kv, off_ov_gate, off_entries);
    return fllm_check_launch("v4_compress");
}

int fllm_v4_attention_launch(cudaStream_t s, float *out, const float *q, const float *kv, const float *state, const float *sinks, const int32_t *sel,
                             const float *cs, int row0, int n, int position0, int heads, int hd, int rope, int window, int ratio, int topk, int o_groups,
                             int rows_total, size_t off_ring, size_t off_entries, float scale) {
    if (n <= 0) return 0;
    if (!fllm_v4_supported(hd, rope, window, heads, o_groups)) return fllm_set_error("v4_attention: unsupported shape (%d heads of %d, %d groups)", heads, hd, o_groups);
    const size_t smem = (size_t)(hd + V4_CHUNK + V4_THREADS) * sizeof(float);
    for (int done = 0; done < n; done += 65535) {
        const int count = n - done < 65535 ? n - done : 65535;
        dim3 grid(heads, count);
        v4_attn_kernel<<<grid, V4_THREADS, smem, s>>>(out, q, kv, state, sinks, sel, cs, row0 + done, position0 + done, row0, position0, heads, hd, rope, window,
                                                      ratio, topk, heads / o_groups, rows_total, off_ring, off_entries, scale);
        if (int e = fllm_check_launch("v4_attention")) return e;
    }
    return 0;
}

int fllm_v4_ring_launch(cudaStream_t s, float *state, const float *kv, int row0, int n, int position0, int hd, int window, size_t off_ring) {
    if (n <= 0) return 0;
    const size_t total = (size_t)(n < window ? n : window) * hd;
    v4_ring_kernel<<<v4_grid(total), 256, 0, s>>>(state + off_ring, kv, row0, n, position0, hd, window);
    return fllm_check_launch("v4_ring");
}
