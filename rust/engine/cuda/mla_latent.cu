// fLLM CUDA backend: latent-cache multi-head latent attention, the projections around the attention (query absorption, cache row, output).
//
// Method (DeepSeek-V2 "matrix absorption", as used by the MLA backends of vLLM and by FlashMLA): per head h the key is K_nope[h] = W_UK[h] c and the
// value V[h] = W_UV[h] c for the latent c of a token, so
//     q_nope[h] . K_nope[h] = (W_UK[h]^T q_nope[h]) . c        and        sum_t p_t V[h]_t = W_UV[h] (sum_t p_t c_t).
// The cache then stores only c (kv_lora values) and the rotated shared rope key (rope values) per token, all heads attend over the same rows
// (mla_decode.cu), the query is absorbed here and the output is expanded here. W_UK / W_UV are rows of the kv_b matrix [heads * (nope + value)][kv_lora]
// (c contiguous): row h * (nope + value) + j is row j of W_UK[h] (j < nope), row h * (nope + value) + nope + v is row v of W_UV[h].
//
// The formulation follows vLLM (vllm/v1/attention/backends/mla, Apache-2.0) and FlashMLA (https://github.com/deepseek-ai/FlashMLA, MIT); the kernels
// themselves are written for fLLM (mma.sync m16n8k16, ldmatrix, cp.async) and not copied. See THIRD_PARTY (docs/changes/hopper-mla.md).
#include <cstdlib>

#include "mla_common.cuh"
#include "fllm_cuda_internal.h"
#include "mla_latent.h"

// ---- cache row ----
__global__ void mla_kv_kernel(const float *__restrict__ kv_a, const float *__restrict__ latent, const int32_t *__restrict__ pos,
                              const float *__restrict__ cs, const float *__restrict__ sn, float *__restrict__ k_row, int rows, int kv_lora,
                              int rope, int neox) {
    const int D = kv_lora + rope, half_ = rope / 2;
    const size_t total = (size_t)rows * D;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < total; i += (size_t)gridDim.x * blockDim.x) {
        const size_t r = i / D;
        const int c = (int)(i % D);
        if (c < kv_lora) {
            k_row[i] = latent[r * kv_lora + c];
            continue;
        }
        const int e = c - kv_lora;
        const float *pe = kv_a + r * D + kv_lora;
        const int p = neox ? (e < half_ ? e : e - half_) : e >> 1;
        const float a = pe[neox ? p : 2 * p], b = pe[neox ? p + half_ : 2 * p + 1];
        const float cc = cs[(size_t)pos[r] * half_ + p], ss = sn[(size_t)pos[r] * half_ + p];
        float ra, rb;
        rope_pair(a, b, cc, ss, ra, rb);
        const bool first = neox ? e < half_ : (e & 1) == 0;
        k_row[i] = first ? ra : rb;
    }
}

// ---- query absorption: q_lat[r][h][0..kv_lora) = q_nope[r][h] (1 x nope) times W_UK[h] (nope x kv_lora), q_lat[r][h][kv_lora..) = rope(q_pe) ----
// CTA = 16 rows x one head x one slab of 4 * NTW * 8 columns; 4 warps, each NTW n-tiles of 8 columns; K = nope in steps of 16.
template <int NTW>
__global__ void __launch_bounds__(128) mla_query_kernel(const float *__restrict__ q_full, const uint16_t *__restrict__ w,
                                                       const int32_t *__restrict__ pos, const float *__restrict__ cs,
                                                       const float *__restrict__ sn, uint16_t *__restrict__ q_lat, int rows, int heads,
                                                       int nope, int rope, int value, int kv_lora, int neox) {
#if FLLM_TC
    constexpr int SLAB = 4 * NTW * 8;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    const int AS = nope + 8, WS = SLAB + 8;
    uint16_t *As = reinterpret_cast<uint16_t *>(smem_raw);
    uint16_t *Ws = As + 16 * AS;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int r0 = blockIdx.x * 16, h = blockIdx.y, c0 = blockIdx.z * SLAB;
    const int dq = nope + rope, per_head = nope + value, D = kv_lora + rope;
    for (int idx = tid; idx < nope * (SLAB / 8); idx += 128) {
        const int j = idx / (SLAB / 8), ch = idx % (SLAB / 8);
        cp_async16(Ws + j * WS + ch * 8, w + (size_t)(h * per_head + j) * kv_lora + c0 + ch * 8, true);
    }
    cp_async_commit();
    for (int idx = tid; idx < 16 * (nope / 2); idx += 128) {
        const int r = idx / (nope / 2), k2 = (idx % (nope / 2)) * 2, row = r0 + r;
        float a = 0.f, b = 0.f;
        if (row < rows) {
            const float2 v = *reinterpret_cast<const float2 *>(q_full + ((size_t)row * heads + h) * dq + k2);
            a = v.x;
            b = v.y;
        }
        *reinterpret_cast<uint32_t *>(As + r * AS + k2) = mla_pack_half2(a, b);
    }
    cp_async_wait<0>();
    __syncthreads();
    float acc[NTW][4];
#pragma unroll
    for (int i = 0; i < NTW; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
    for (int k0 = 0; k0 < nope; k0 += 16) {
        uint32_t a[4];
        ldmatrix4(a[0], a[1], a[2], a[3], As + ((lane & 7) + ((lane >> 3) & 1) * 8) * AS + k0 + (lane >> 4) * 8);
#pragma unroll
        for (int nt = 0; nt < NTW; nt += 2) {
            uint32_t b0, b1, b2, b3;
            const int nbase = warp * NTW * 8 + nt * 8;
            ldsm_x4_t(b0, b1, b2, b3, Ws + (k0 + (lane & 7) + ((lane >> 3) & 1) * 8) * WS + nbase + (lane >> 4) * 8);
            mma16816(acc[nt], a, b0, b1);
            mma16816(acc[nt + 1], a, b2, b3);
        }
    }
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int nt = 0; nt < NTW; nt++) {
        const int col = c0 + warp * NTW * 8 + nt * 8 + 2 * t;
        const int ra = r0 + g, rb = r0 + g + 8;
        if (ra < rows) *reinterpret_cast<uint32_t *>(q_lat + ((size_t)ra * heads + h) * D + col) = mla_pack_half2(acc[nt][0], acc[nt][1]);
        if (rb < rows) *reinterpret_cast<uint32_t *>(q_lat + ((size_t)rb * heads + h) * D + col) = mla_pack_half2(acc[nt][2], acc[nt][3]);
    }
    if (blockIdx.z == 0) {
        const int half_ = rope / 2;
        for (int idx = tid; idx < 16 * half_; idx += 128) {
            const int r = idx / half_, p = idx % half_, row = r0 + r;
            if (row >= rows) continue;
            const float *qp = q_full + ((size_t)row * heads + h) * dq + nope;
            const int ia = neox ? p : 2 * p, ib = neox ? p + half_ : 2 * p + 1;
            const float cc = cs[(size_t)pos[row] * half_ + p], ss = sn[(size_t)pos[row] * half_ + p];
            float ra, rb;
            rope_pair(qp[ia], qp[ib], cc, ss, ra, rb);
            uint16_t *o = q_lat + ((size_t)row * heads + h) * D + kv_lora;
            o[ia] = f2h_bits(ra);
            o[ib] = f2h_bits(rb);
        }
    }
#endif
}

// ---- output expansion: out[r][h][v] = sum_c o_lat[r][h][c] * W_UV[h][v][c]; CTA = 16 rows x one head, 4 warps x NTW n-tiles (value = 32 NTW) ----
template <int NTW>
__global__ void __launch_bounds__(128) mla_output_kernel(const uint16_t *__restrict__ o_lat, const uint16_t *__restrict__ w, float *__restrict__ out,
                                                        int rows, int heads, int nope, int value, int kv_lora) {
#if FLLM_TC
    constexpr int KS = 72;  // 64 values of K per chunk plus padding (conflict-free ldmatrix)
    extern __shared__ __align__(16) unsigned char smem_raw[];
    uint16_t *Wsm = reinterpret_cast<uint16_t *>(smem_raw);
    uint16_t *Asm = Wsm + value * KS;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int r0 = blockIdx.x * 16, h = blockIdx.y;
    const int per_head = nope + value;
    float acc[NTW][4];
#pragma unroll
    for (int i = 0; i < NTW; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
    for (int kc = 0; kc < kv_lora; kc += 64) {
        for (int idx = tid; idx < value * 8; idx += 128) {
            const int n = idx >> 3, ch = idx & 7;
            cp_async16(Wsm + n * KS + ch * 8, w + (size_t)(h * per_head + nope + n) * kv_lora + kc + ch * 8, true);
        }
        {
            const int r = tid >> 3, ch = tid & 7, row = r0 + r;
            const bool valid = row < rows;
            cp_async16(Asm + r * KS + ch * 8, valid ? o_lat + ((size_t)row * heads + h) * kv_lora + kc + ch * 8 : o_lat, valid);
        }
        cp_async_commit();
        cp_async_wait<0>();
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < 4; ks++) {
            uint32_t a[4];
            ldmatrix4(a[0], a[1], a[2], a[3], Asm + ((lane & 7) + ((lane >> 3) & 1) * 8) * KS + ks * 16 + (lane >> 4) * 8);
#pragma unroll
            for (int nt = 0; nt < NTW; nt++) {
                uint32_t b0, b1;
                ldsm_x2(b0, b1, Wsm + (warp * NTW * 8 + nt * 8 + (lane & 7)) * KS + ks * 16 + ((lane >> 3) & 1) * 8);
                mma16816(acc[nt], a, b0, b1);
            }
        }
        __syncthreads();
    }
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int nt = 0; nt < NTW; nt++) {
        const int col = warp * NTW * 8 + nt * 8 + 2 * t;
        const int ra = r0 + g, rb = r0 + g + 8;
        if (ra < rows) *reinterpret_cast<float2 *>(out + (size_t)ra * heads * value + (size_t)h * value + col) = make_float2(acc[nt][0], acc[nt][1]);
        if (rb < rows) *reinterpret_cast<float2 *>(out + (size_t)rb * heads * value + (size_t)h * value + col) = make_float2(acc[nt][2], acc[nt][3]);
    }
#endif
}

static unsigned mla_grid(size_t n) {
    size_t g = (n + 255) / 256;
    return (unsigned)(g > 65535 ? 65535 : (g == 0 ? 1 : g));
}

// shared memory the query / output kernels need
static size_t query_smem(int nope, int ntw) { return (size_t)(16 * (nope + 8) + nope * (4 * ntw * 8 + 8)) * 2; }
static size_t output_smem(int value) { return (size_t)(value * 72 + 16 * 72) * 2; }

bool fllm_mla_attention_dims_ok(int kv_lora, int rope, int fmt);  // mla_decode.cu

int fllm_mla_latent_supported(int kv_lora, int rope, int nope, int value, int kv_fmt) {
    if (fllm_cc() < 75 || fllm_exact_math()) return 0;
    if (nope <= 0 || nope % 16 != 0 || nope > 256 || value <= 0 || value % 32 != 0 || value > 256) return 0;
    if (kv_lora % 64 != 0 || rope % 2 != 0) return 0;
    if (!fllm_mla_attention_dims_ok(kv_lora, rope, kv_fmt)) return 0;
    const int ntw = kv_lora < 128 ? 2 : 4;
    return query_smem(nope, ntw) <= (size_t)fllm_smem_limit() && output_smem(value) <= (size_t)fllm_smem_limit();
}

int fllm_mla_kv_launch(cudaStream_t s, const float *kv_a, const float *latent, const int32_t *pos, const float *cs, const float *sn,
                       float *k_row, int rows, int kv_lora, int rope, int neox) {
    if (rows <= 0) return 0;
    mla_kv_kernel<<<mla_grid((size_t)rows * (kv_lora + rope)), 256, 0, s>>>(kv_a, latent, pos, cs, sn, k_row, rows, kv_lora, rope, neox);
    return fllm_check_launch("mla_kv");
}

template <int NTW>
static int query_launch(cudaStream_t s, const float *q_full, const uint16_t *w, const int32_t *pos, const float *cs, const float *sn,
                        uint16_t *q_lat, int rows, int heads, int nope, int rope, int value, int kv_lora, int neox) {
    constexpr int SLAB = 4 * NTW * 8;
    static size_t configured = 0;
    const size_t smem = query_smem(nope, NTW);
    if (smem > configured) {
        cudaFuncSetAttribute(mla_query_kernel<NTW>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        configured = smem;
    }
    if (kv_lora % SLAB != 0) return fllm_set_error("mla_query: kv_lora %d is not a multiple of %d", kv_lora, SLAB);
    const dim3 grid((rows + 15) / 16, heads, kv_lora / SLAB);
    mla_query_kernel<NTW><<<grid, 128, smem, s>>>(q_full, w, pos, cs, sn, q_lat, rows, heads, nope, rope, value, kv_lora, neox);
    return fllm_check_launch("mla_query");
}

int fllm_mla_query_launch(cudaStream_t s, const float *q_full, const uint16_t *w_kvb, const int32_t *pos, const float *cs, const float *sn,
                          uint16_t *q_lat, int rows, int heads, int nope, int rope, int value, int kv_lora, int neox) {
    if (rows <= 0) return 0;
    if (kv_lora < 128) return query_launch<2>(s, q_full, w_kvb, pos, cs, sn, q_lat, rows, heads, nope, rope, value, kv_lora, neox);
    return query_launch<4>(s, q_full, w_kvb, pos, cs, sn, q_lat, rows, heads, nope, rope, value, kv_lora, neox);
}

template <int NTW>
static int output_launch(cudaStream_t s, const uint16_t *o_lat, const uint16_t *w, float *out, int rows, int heads, int nope, int value,
                         int kv_lora) {
    static size_t configured = 0;
    const size_t smem = output_smem(value);
    if (smem > configured) {
        cudaFuncSetAttribute(mla_output_kernel<NTW>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        configured = smem;
    }
    mla_output_kernel<NTW><<<dim3((rows + 15) / 16, heads), 128, smem, s>>>(o_lat, w, out, rows, heads, nope, value, kv_lora);
    return fllm_check_launch("mla_output");
}

int fllm_mla_output_launch(cudaStream_t s, const uint16_t *o_lat, const uint16_t *w_kvb, float *out, int rows, int heads, int nope, int value,
                           int kv_lora) {
    if (rows <= 0) return 0;
    switch (value / 32) {
        case 1: return output_launch<1>(s, o_lat, w_kvb, out, rows, heads, nope, value, kv_lora);
        case 2: return output_launch<2>(s, o_lat, w_kvb, out, rows, heads, nope, value, kv_lora);
        case 3: return output_launch<3>(s, o_lat, w_kvb, out, rows, heads, nope, value, kv_lora);
        case 4: return output_launch<4>(s, o_lat, w_kvb, out, rows, heads, nope, value, kv_lora);
        case 6: return output_launch<6>(s, o_lat, w_kvb, out, rows, heads, nope, value, kv_lora);
        case 8: return output_launch<8>(s, o_lat, w_kvb, out, rows, heads, nope, value, kv_lora);
        default: return fllm_set_error("mla_output: value width %d is not covered", value);
    }
}
