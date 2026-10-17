// fLLM CUDA backend: attention over the latent MLA cache (decode and prompt rows), tensor cores (mma.sync m16n8k16).
//
// All heads of a layer attend over the same cache rows, [latent | rope key] of kv_lora + rope values per token, with the keys equal to the full
// row and the values equal to its first kv_lora values (see mla_latent.cu for the absorbed queries and the output expansion). A CTA owns 16 "slots"
// that form the M dimension of the MMAs:
//   * decode rows (independent sequences, or a prompt tail of fewer than 16 rows): the 16 slots are 16 heads of one row;
//   * prompt rows (a run of 16 or more rows of one sequence with the same block table): the 16 slots are 16 consecutive rows of one head, so a key
//     tile is read once for 16 rows; every slot has its own causal length.
// Per tile of 32 keys: S = Q K^T (the four warps take 8 keys each, K = kv_lora + rope), an online softmax over the 16 slots, O += P V (each warp owns
// a quarter of the kv_lora columns of the accumulator, 64 registers per thread for kv_lora = 512). K tiles are staged in shared memory with cp.async
// (f16 cache) or decoded from 8-bit codes with an f16 scale per group (q8_0, int8, fp8_e4m3), double buffered when shared memory allows. Long
// contexts with few CTAs are split over the keys (flash-decoding) and merged by mla_merge_kernel.
//
// The design (absorbed latent decode, one shared key/value row per token, head-parallel M dimension, split over keys) follows FlashMLA
// (https://github.com/deepseek-ai/FlashMLA, MIT) and the MLA decode paths of vLLM (Apache-2.0); the code is written for fLLM.
#include <cstdlib>
#include <cstring>
#include <vector>

#include "mla_common.cuh"
#include "fllm_cuda_internal.h"
#include "mla_latent.h"

struct MlaAttnArgs {
    const uint16_t *q, *pool;
    uint16_t *o;
    float *ws_o, *ws_ml;
    const int64_t *tables, *lengths;
    int row0, n_rows, heads, stride, block_size, splits, per_split, rowpack;
    float scale_log2;
    int fmt, gshift, gps;
    long long scale_off;
};

constexpr int MLA_T = 32;   // keys per tile
constexpr int MLA_PS = 40;  // row stride of the P tile (halves)
constexpr int MLA_SS = 36;  // row stride of the S tile (floats)

__host__ __device__ constexpr size_t mla_tc_smem(int kvl, int rope, int stages) {
    return (size_t)((16 + stages * MLA_T) * (kvl + rope + 8) + 16 * MLA_PS) * 2 + (size_t)16 * MLA_SS * 4;
}

#define MLA_NEG_INF (-INFINITY)

template <int KVL, int ROPE, int FM, int STAGES>
__global__ void __launch_bounds__(128) mla_attn_kernel(MlaAttnArgs a) {
#if FLLM_TC
    constexpr int D = KVL + ROPE, DP = D + 8, T = MLA_T, PS = MLA_PS, SS = MLA_SS, NT = KVL / 32;
    extern __shared__ __align__(16) unsigned char smem_raw[];
    uint16_t *Qsm = reinterpret_cast<uint16_t *>(smem_raw);
    uint16_t *Ksm = Qsm + 16 * DP;
    uint16_t *Psm = Ksm + STAGES * T * DP;
    float *Ssm = reinterpret_cast<float *>(Psm + 16 * PS);
    __shared__ int s_row[16], s_head[16], s_len[16];
    __shared__ float s_corr[16], s_l[16], s_m[16];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, g = lane >> 2, t = lane & 3;
    const int sp = blockIdx.z;
    if (tid < 16) {
        int row, head;
        bool valid;
        if (!a.rowpack) {
            row = a.row0 + blockIdx.x;
            head = blockIdx.y * 16 + tid;
            valid = head < a.heads;
        } else {
            const int rr = blockIdx.x * 16 + tid;
            row = a.row0 + rr;
            head = blockIdx.y;
            valid = rr < a.n_rows;
        }
        s_row[tid] = row;
        s_head[tid] = head;
        s_len[tid] = valid ? (int)a.lengths[row] : 0;
    }
    __syncthreads();
    const int trow = a.rowpack ? a.row0 + blockIdx.x * 16 : a.row0 + blockIdx.x;
    int maxlen = 0;
#pragma unroll
    for (int i = 0; i < 16; i++) maxlen = max(maxlen, s_len[i]);
    for (int idx = tid; idx < 16 * (D / 8); idx += 128) {
        const int i = idx / (D / 8), ch = idx % (D / 8);
        const bool v = s_len[i] > 0;
        cp_async16(Qsm + i * DP + ch * 8, v ? a.q + ((size_t)s_row[i] * a.heads + s_head[i]) * D + ch * 8 : a.q, v);
    }
    cp_async_commit();
    const int kbeg = sp * a.per_split;
    const int kend = min(maxlen, kbeg + a.per_split);
    const int ntile = kend > kbeg ? (kend - kbeg + T - 1) / T : 0;
    const int64_t *table = a.tables + (size_t)trow * a.stride;

    auto load_tile = [&](int tl, int st) {
        uint16_t *dst = Ksm + st * T * DP;
        if (FM == 0) {
            for (int idx = tid; idx < T * (D / 8); idx += 128) {
                const int kk = idx / (D / 8), ch = idx % (D / 8);
                const int key = kbeg + tl * T + kk;
                const bool v = key < kend;
                const uint16_t *src = a.pool;
                if (v) src = a.pool + ((size_t)table[key / a.block_size] * a.block_size + key % a.block_size) * D + ch * 8;
                cp_async16(dst + kk * DP + ch * 8, src, v);
            }
        } else {
            const uint8_t *codes = reinterpret_cast<const uint8_t *>(a.pool);
            for (int idx = tid; idx < T * (D / 4); idx += 128) {
                const int kk = idx / (D / 4), q4 = idx % (D / 4);
                const int key = kbeg + tl * T + kk;
                uint32_t lo = 0, hi = 0;
                if (key < kend) {
                    const size_t slot = (size_t)table[key / a.block_size] * a.block_size + key % a.block_size;
                    const uint32_t w4 = *reinterpret_cast<const uint32_t *>(codes + slot * D + (size_t)q4 * 4);
                    const float sc = h2f_bits(a.pool[a.scale_off + slot * a.gps + (a.gshift < 0 ? 0 : (q4 * 4) >> a.gshift)]);
                    float f[4];
#pragma unroll
                    for (int i = 0; i < 4; i++) {
                        const uint32_t b = (w4 >> (8 * i)) & 0xffu;
                        f[i] = (a.fmt == MLA_E4M3 ? mla_e4m3(b) : (float)(int8_t)b) * sc;
                    }
                    lo = mla_pack_half2(f[0], f[1]);
                    hi = mla_pack_half2(f[2], f[3]);
                }
                *reinterpret_cast<uint2 *>(dst + kk * DP + q4 * 4) = make_uint2(lo, hi);
            }
        }
    };

    float acc[NT][4];
#pragma unroll
    for (int i = 0; i < NT; i++) acc[i][0] = acc[i][1] = acc[i][2] = acc[i][3] = 0.f;
    float m_run = MLA_NEG_INF, l_run = 0.f;
    if (STAGES == 2 && ntile > 0) {
        load_tile(0, 0);
        cp_async_commit();
    }
    const int kk0 = warp * 8, dbase = warp * (KVL / 4);
    for (int tl = 0; tl < ntile; tl++) {
        int st = 0;
        if (STAGES == 2) {
            st = tl & 1;
            if (tl + 1 < ntile) {
                load_tile(tl + 1, (tl + 1) & 1);
                cp_async_commit();
                cp_async_wait<1>();
            } else {
                cp_async_wait<0>();
            }
        } else {
            load_tile(tl, 0);
            cp_async_commit();
            cp_async_wait<0>();
        }
        __syncthreads();
        const uint16_t *Kst = Ksm + st * T * DP;
        // S = Q K^T for the 8 keys of this warp
        float sc[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
        for (int k0 = 0; k0 < D; k0 += 16) {
            uint32_t qa[4], b0, b1;
            ldmatrix4(qa[0], qa[1], qa[2], qa[3], Qsm + ((lane & 7) + ((lane >> 3) & 1) * 8) * DP + k0 + (lane >> 4) * 8);
            ldsm_x2(b0, b1, Kst + (kk0 + (lane & 7)) * DP + k0 + ((lane >> 3) & 1) * 8);
            mma16816(sc, qa, b0, b1);
        }
        const int keyb = kbeg + tl * T + kk0 + 2 * t;
#pragma unroll
        for (int j = 0; j < 2; j++) {
            const int key = keyb + j;
            Ssm[g * SS + kk0 + 2 * t + j] = key < s_len[g] ? sc[j] * a.scale_log2 : MLA_NEG_INF;
            Ssm[(g + 8) * SS + kk0 + 2 * t + j] = key < s_len[g + 8] ? sc[2 + j] * a.scale_log2 : MLA_NEG_INF;
        }
        __syncthreads();
        // online softmax: 8 threads per slot, 4 keys each
        {
            const int sl = tid >> 3, j8 = tid & 7;
            float v[4];
            float tm = MLA_NEG_INF;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                v[i] = Ssm[sl * SS + j8 * 4 + i];
                tm = fmaxf(tm, v[i]);
            }
            tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, 1));
            tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, 2));
            tm = fmaxf(tm, __shfl_xor_sync(0xffffffffu, tm, 4));
            const float mn = fmaxf(m_run, tm);
            const float ms = mn == MLA_NEG_INF ? 0.f : mn;
            const float corr = m_run == MLA_NEG_INF ? 0.f : exp2f(m_run - ms);
            float ps = 0.f;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const float p = exp2f(v[i] - ms);
                ps += p;
                Psm[sl * PS + j8 * 4 + i] = f2h_bits(p);
            }
            l_run = l_run * corr + ps;
            m_run = mn;
            if (j8 == 0) s_corr[sl] = corr;
        }
        __syncthreads();
        // O = O * corr + P V
        const float ca = s_corr[g], cb = s_corr[g + 8];
#pragma unroll
        for (int i = 0; i < NT; i++) {
            acc[i][0] *= ca;
            acc[i][1] *= ca;
            acc[i][2] *= cb;
            acc[i][3] *= cb;
        }
#pragma unroll
        for (int kk16 = 0; kk16 < 2; kk16++) {
            uint32_t pa[4];
            ldmatrix4(pa[0], pa[1], pa[2], pa[3], Psm + ((lane & 7) + ((lane >> 3) & 1) * 8) * PS + kk16 * 16 + (lane >> 4) * 8);
#pragma unroll
            for (int np = 0; np < NT; np += 2) {
                uint32_t b0, b1, b2, b3;
                ldsm_x4_t(b0, b1, b2, b3, Kst + (kk16 * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * DP + dbase + np * 8 + (lane >> 4) * 8);
                mma16816(acc[np], pa, b0, b1);
                mma16816(acc[np + 1], pa, b2, b3);
            }
        }
        __syncthreads();
    }
    cp_async_wait<0>();
    float l = l_run;
    l += __shfl_xor_sync(0xffffffffu, l, 1);
    l += __shfl_xor_sync(0xffffffffu, l, 2);
    l += __shfl_xor_sync(0xffffffffu, l, 4);
    if ((tid & 7) == 0) {
        s_l[tid >> 3] = l;
        s_m[tid >> 3] = m_run;
    }
    __syncthreads();
#pragma unroll
    for (int hf = 0; hf < 2; hf++) {
        const int sl = g + hf * 8;
        if (s_len[sl] == 0) continue;
        const int row = s_row[sl], head = s_head[sl];
        if (a.splits == 1) {
            const float inv = s_l[sl] > 0.f ? 1.f / s_l[sl] : 0.f;
            uint16_t *op = a.o + ((size_t)row * a.heads + head) * KVL;
#pragma unroll
            for (int nt = 0; nt < NT; nt++)
                *reinterpret_cast<uint32_t *>(op + dbase + nt * 8 + 2 * t) = mla_pack_half2(acc[nt][hf * 2] * inv, acc[nt][hf * 2 + 1] * inv);
        } else {
            const size_t gi = (size_t)(row - a.row0) * a.heads + head;
            float *wo = a.ws_o + (gi * a.splits + sp) * KVL;
#pragma unroll
            for (int nt = 0; nt < NT; nt++)
                *reinterpret_cast<float2 *>(wo + dbase + nt * 8 + 2 * t) = make_float2(acc[nt][hf * 2], acc[nt][hf * 2 + 1]);
        }
    }
    if (a.splits > 1 && tid < 16 && s_len[tid] > 0) {
        const size_t gi = (size_t)(s_row[tid] - a.row0) * a.heads + s_head[tid];
        a.ws_ml[(gi * a.splits + sp) * 2] = s_m[tid];
        a.ws_ml[(gi * a.splits + sp) * 2 + 1] = s_l[tid];
    }
#endif
}

// Merges the partial results of the key splits of one (row, head).
__global__ void mla_merge_kernel(const float *__restrict__ ws_o, const float *__restrict__ ws_ml, uint16_t *__restrict__ o, int row0, int heads,
                                 int kvl, int splits) {
    const size_t gi = blockIdx.x;
    const float *ml = ws_ml + gi * splits * 2;
    float mx = MLA_NEG_INF;
    for (int s = 0; s < splits; s++) mx = fmaxf(mx, ml[2 * s]);
    float den = 0.f;
    for (int s = 0; s < splits; s++)
        if (ml[2 * s] != MLA_NEG_INF) den += exp2f(ml[2 * s] - mx) * ml[2 * s + 1];
    const size_t row = row0 + gi / heads, head = gi % heads;
    for (int c = threadIdx.x; c < kvl; c += blockDim.x) {
        float num = 0.f;
        for (int s = 0; s < splits; s++)
            if (ml[2 * s] != MLA_NEG_INF) num += exp2f(ml[2 * s] - mx) * ws_o[(gi * splits + s) * kvl + c];
        o[(row * heads + head) * kvl + c] = f2h_bits(den > 0.f ? num / den : 0.f);
    }
}

// CUDA-core reference: one CTA per (row, head), keys in chunks of 256, generic cache decoding. Used by the self test to tell a defect of the
// tensor-core kernel from a defect of the surrounding data movement (FLLM_MLA_KERNEL=ref or mode 1).
__global__ void __launch_bounds__(256) mla_attn_ref_kernel(const uint16_t *__restrict__ q, const uint16_t *__restrict__ pool, uint16_t *__restrict__ o,
                                                          const int64_t *__restrict__ tables, const int64_t *__restrict__ lengths, int row0,
                                                          int heads, int kvl, int rope, int stride, int block_size, FllmKvDesc kd, float scale_log2) {
    extern __shared__ float rsm[];
    const int D = kvl + rope, tid = threadIdx.x;
    const int row = row0 + blockIdx.x, head = blockIdx.y;
    float *qs = rsm, *ps = rsm + D;
    for (int c = tid; c < D; c += 256) qs[c] = h2f_bits(q[((size_t)row * heads + head) * D + c]);
    __syncthreads();
    const int len = (int)lengths[row];
    const int64_t *table = tables + (size_t)row * stride;
    float m = MLA_NEG_INF, l = 0.f, acc[2] = {0.f, 0.f};
    for (int base = 0; base < len; base += 256) {
        const int key = base + tid;
        float s = MLA_NEG_INF;
        if (key < len) {
            const size_t slot = (size_t)table[key / block_size] * block_size + key % block_size;
            float d = 0.f;
            for (int c = 0; c < D; c++) d += qs[c] * mla_cache_at(pool, kd, slot, c);
            s = d * scale_log2;
        }
        ps[tid] = s;
        __syncthreads();
        float tm = MLA_NEG_INF;
        for (int i = 0; i < 256; i++) tm = fmaxf(tm, ps[i]);
        const float mn = fmaxf(m, tm), ms = mn == MLA_NEG_INF ? 0.f : mn;
        const float corr = m == MLA_NEG_INF ? 0.f : exp2f(m - ms);
        acc[0] *= corr;
        acc[1] *= corr;
        float lsum = 0.f;
        for (int i = 0; i < 256 && base + i < len; i++) {
            const float p = exp2f(ps[i] - ms);
            lsum += p;
            const int ki = base + i;
            const size_t slot = (size_t)table[ki / block_size] * block_size + ki % block_size;
            for (int c2 = 0; c2 < 2; c2++) {
                const int c = tid + c2 * 256;
                if (c < kvl) acc[c2] += p * mla_cache_at(pool, kd, slot, c);
            }
        }
        l = l * corr + lsum;
        m = mn;
        __syncthreads();
    }
    for (int c2 = 0; c2 < 2; c2++) {
        const int c = tid + c2 * 256;
        if (c < kvl) o[((size_t)row * heads + head) * kvl + c] = f2h_bits(l > 0.f ? acc[c2] / l : 0.f);
    }
}

// ---- host side ----
static bool dims_listed(int kvl, int rope) {
    return (kvl == 512 && rope == 64) || (kvl == 256 && rope == 64) || (kvl == 256 && rope == 32) || (kvl == 128 && rope == 64) ||
           (kvl == 128 && rope == 32) || (kvl == 64 && rope == 32);
}

bool fllm_mla_attention_dims_ok(int kv_lora, int rope, int fmt) {
    return dims_listed(kv_lora, rope) && (fmt == MLA_F16 || fmt == MLA_Q8_0 || fmt == MLA_E4M3 || fmt == MLA_INT8) &&
           mla_tc_smem(kv_lora, rope, 1) <= (size_t)fllm_smem_limit();
}

namespace {
struct Workspace {
    void *ptr = nullptr;
    size_t bytes = 0;
    bool reserve(size_t need) {
        if (need <= bytes) return true;
        cudaFree(ptr);
        ptr = nullptr;
        bytes = 0;
        if (cudaMalloc(&ptr, need) != cudaSuccess) {
            cudaGetLastError();
            return false;
        }
        bytes = need;
        return true;
    }
};
thread_local Workspace g_ws;
constexpr size_t kWorkspaceCap = (size_t)64 << 20;

template <int KVL, int ROPE, int FM, int ST>
int run_tc(cudaStream_t s, const MlaAttnArgs &a, dim3 grid) {
    constexpr size_t smem = mla_tc_smem(KVL, ROPE, ST);
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(mla_attn_kernel<KVL, ROPE, FM, ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        configured = true;
    }
    mla_attn_kernel<KVL, ROPE, FM, ST><<<grid, 128, smem, s>>>(a);
    return fllm_check_launch("mla_attention");
}
template <int KVL, int ROPE, int FM>
int run_stages(cudaStream_t s, const MlaAttnArgs &a, dim3 grid) {
    if (mla_tc_smem(KVL, ROPE, 2) <= (size_t)fllm_smem_limit()) return run_tc<KVL, ROPE, FM, 2>(s, a, grid);
    return run_tc<KVL, ROPE, FM, 1>(s, a, grid);
}
template <int FM>
int run_dims(cudaStream_t s, int kvl, int rope, const MlaAttnArgs &a, dim3 grid) {
    if (kvl == 512 && rope == 64) return run_stages<512, 64, FM>(s, a, grid);
    if (kvl == 256 && rope == 64) return run_stages<256, 64, FM>(s, a, grid);
    if (kvl == 256 && rope == 32) return run_stages<256, 32, FM>(s, a, grid);
    if (kvl == 128 && rope == 64) return run_stages<128, 64, FM>(s, a, grid);
    if (kvl == 128 && rope == 32) return run_stages<128, 32, FM>(s, a, grid);
    if (kvl == 64 && rope == 32) return run_stages<64, 32, FM>(s, a, grid);
    return -1;
}

int sm_count() {
    static int cache[16] = {0};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) return 80;
    if (cache[dev] == 0) {
        int n = 0;
        cache[dev] = cudaDeviceGetAttribute(&n, cudaDevAttrMultiProcessorCount, dev) == cudaSuccess && n > 0 ? n : 80;
    }
    return cache[dev];
}

struct Group {
    int r0, n;
    bool pack;
};
}  // namespace

int fllm_mla_attention_launch(cudaStream_t s, const uint16_t *q_lat, const uint16_t *pool, uint16_t *o_lat, const int64_t *dtables,
                              const int64_t *dlengths, const int64_t *htables, const int64_t *hlengths, int rows, int heads, int kv_lora,
                              int rope, int stride, int block_size, const FllmKvDesc *kd, float scale, int mode) {
    if (rows <= 0) return 0;
    if (!fllm_mla_attention_dims_ok(kv_lora, rope, kd->fmt) || fllm_cc() < 75) return -1;
    static const bool ref_env = [] {
        const char *e = getenv("FLLM_MLA_KERNEL");
        return e != nullptr && strcmp(e, "ref") == 0;
    }();
    const bool ref = mode == 1 || ref_env;
    const int D = kv_lora + rope;
    MlaAttnArgs a{};
    a.q = q_lat;
    a.pool = pool;
    a.o = o_lat;
    a.tables = dtables;
    a.lengths = dlengths;
    a.heads = heads;
    a.stride = stride;
    a.block_size = block_size;
    a.scale_log2 = scale * 1.4426950408889634f;
    a.fmt = kd->fmt;
    a.gshift = kd->gshift;
    a.gps = kd->gps;
    a.scale_off = kd->scale_off;
    // groups of rows: runs of one prompt (same table, one more key per row) of 16 or more rows, and the rest
    std::vector<Group> groups;
    for (int r = 0; r < rows;) {
        int r1 = r + 1;
        while (r1 < rows && hlengths[r1] == hlengths[r1 - 1] + 1 &&
               memcmp(htables + (size_t)r1 * stride, htables + (size_t)r * stride, (size_t)stride * sizeof(int64_t)) == 0)
            r1++;
        if (r1 - r >= 16) {
            groups.push_back({r, r1 - r, true});
        } else if (!groups.empty() && !groups.back().pack) {
            groups.back().n += r1 - r;
        } else {
            groups.push_back({r, r1 - r, false});
        }
        r = r1;
    }
    for (const Group &gr : groups) {
        a.row0 = gr.r0;
        a.n_rows = gr.n;
        a.rowpack = gr.pack ? 1 : 0;
        int maxlen = 1;
        for (int r = gr.r0; r < gr.r0 + gr.n; r++) maxlen = hlengths[r] > maxlen ? (int)hlengths[r] : maxlen;
        if (ref) {
            const size_t smem = (size_t)(D + 256) * 4;
            mla_attn_ref_kernel<<<dim3(gr.n, heads), 256, smem, s>>>(q_lat, pool, o_lat, dtables, dlengths, gr.r0, heads, kv_lora, rope, stride,
                                                                    block_size, *kd, a.scale_log2);
            if (int e = fllm_check_launch("mla_attention_ref")) return e;
            continue;
        }
        const int row_tiles = gr.pack ? (gr.n + 15) / 16 : gr.n, head_tiles = gr.pack ? heads : (heads + 15) / 16;
        const long base = (long)row_tiles * head_tiles, target = 2L * sm_count();
        int splits = 1;
        if (base < target) {
            const int max_split = (maxlen + 127) / 128;
            splits = (int)((target + base - 1) / base);
            if (splits > max_split) splits = max_split;
            if (splits > 32) splits = 32;
            if (splits < 1) splits = 1;
        }
        while (splits > 1 && (size_t)gr.n * heads * splits * (kv_lora + 2) * 4 > kWorkspaceCap) splits >>= 1;
        if (splits > 1 && !g_ws.reserve((size_t)gr.n * heads * splits * (kv_lora + 2) * 4)) splits = 1;
        int per_split = (maxlen + 31) / 32 * 32;
        if (splits > 1) {
            per_split = ((maxlen + splits - 1) / splits + 31) / 32 * 32;
            splits = (maxlen + per_split - 1) / per_split;
        }
        a.splits = splits;
        a.per_split = per_split;
        if (splits > 1) {
            a.ws_o = static_cast<float *>(g_ws.ptr);
            a.ws_ml = a.ws_o + (size_t)gr.n * heads * splits * kv_lora;
        }
        const dim3 grid(row_tiles, head_tiles, splits);
        const int rc = kd->fmt == MLA_F16 ? run_dims<0>(s, kv_lora, rope, a, grid) : run_dims<1>(s, kv_lora, rope, a, grid);
        if (rc != 0) return rc;
        if (splits > 1) {
            mla_merge_kernel<<<gr.n * heads, 128, 0, s>>>(a.ws_o, a.ws_ml, o_lat, gr.r0, heads, kv_lora, splits);
            if (int e = fllm_check_launch("mla_merge")) return e;
        }
    }
    return 0;
}
