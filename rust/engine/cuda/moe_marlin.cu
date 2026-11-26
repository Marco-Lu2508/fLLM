// fLLM CUDA backend: Marlin-tiled mixture-of-experts kernels for Q4Z32 / Q4Z64 / Q4Z128 weights.
//
// At load time `moe_q4z_to_marlin` repacks the experts of a projection into contiguous Marlin [B_marlin][scales][zeros] arenas
// matching fllm_marlin_gemv.cu / vLLM Marlin ordering (16 x 64 tiles, uint4 coalesced across a warp).
//   * moe_marlin_tc_kernel<BN, GLU, GROUP>: grouped Tensor Core GEMM for prefill and multi-user decode; only activations live in
//     shared memory (4.5 KiB for BN=16, 18 KiB for BN=64 double-buffered), while weights stream via 128-bit coalesced loads and
//     dequantize through LOP3 directly into mma.sync registers.
//   * moe_marlin_gemv_kernel<GLU, U>: single-pass per-slot decode GEMV (fusing gate + up + GLU when gate is present).
#include <float.h>
#include <limits.h>
#include <string.h>

#include "common.cuh"
#include "fllm_cuda_internal.h"
#include "mma_util.cuh"

#define KIND_Q4ZM32 0x80000110u
#define KIND_Q4ZM64 0x80000111u
#define KIND_Q4ZM128 0x80000112u

static __host__ __device__ __forceinline__ size_t mm_align256(size_t v) { return (v + 255) & ~size_t(255); }

static __host__ __device__ __forceinline__ void mm_offsets(int rows, int cols, int group, size_t &s_off, size_t &z_off, size_t &total) {
    const size_t b_bytes = (size_t)cols * rows / 2;
    const size_t groups = (size_t)(cols / group);
    s_off = b_bytes;
    z_off = s_off + mm_align256(groups * (size_t)rows * 2);
    total = z_off + mm_align256(groups * (size_t)(rows / 8) * 4);
}

size_t fllm_moe_marlin_expert_bytes(int rows, int cols, int group) {
    size_t s_off, z_off, total;
    mm_offsets(rows, cols, group, s_off, z_off, total);
    return total;
}

template <int GROUP>
__device__ __forceinline__ uint32_t mm_q4z_nib(const uint8_t *raw, int row, int k, size_t row_bytes) {
    constexpr int BLOCK = GROUP / 2 + 3;
    const int blk = k / GROUP, off = k % GROUP;
    const uint8_t b = raw[(size_t)row * row_bytes + (size_t)blk * BLOCK + (off >> 1)];
    return (off & 1) ? (uint32_t)(b >> 4) : (uint32_t)(b & 15u);
}

template <int GROUP>
__global__ void moe_q4z_to_marlin_b_kernel(const uint8_t *__restrict__ raw_arena, size_t raw_stride, uint8_t *__restrict__ dst_arena,
                                           size_t dst_stride, int rows, int cols) {
    const int nt = (int)blockIdx.x, kt = (int)blockIdx.y, e = (int)blockIdx.z;
    const int lane = (int)threadIdx.x;
    const uint8_t *raw = raw_arena + (size_t)e * raw_stride;
    uint4 *dst_b = reinterpret_cast<uint4 *>(dst_arena + (size_t)e * dst_stride);
    const size_t row_bytes = (size_t)(cols / GROUP) * (GROUP / 2 + 3);
    const int k0 = kt * 16 + (lane & 3) * 2;
    const int n_tiles = rows >> 6;
    uint32_t words[4];
#pragma unroll
    for (int w = 0; w < 4; w++) {
        const int c = nt * 64 + w * 16 + (lane >> 2);
        words[w] = mm_q4z_nib<GROUP>(raw, c, k0, row_bytes)
                 | (mm_q4z_nib<GROUP>(raw, c, k0 + 8, row_bytes) << 4)
                 | (mm_q4z_nib<GROUP>(raw, c + 8, k0, row_bytes) << 8)
                 | (mm_q4z_nib<GROUP>(raw, c + 8, k0 + 8, row_bytes) << 12)
                 | (mm_q4z_nib<GROUP>(raw, c, k0 + 1, row_bytes) << 16)
                 | (mm_q4z_nib<GROUP>(raw, c, k0 + 9, row_bytes) << 20)
                 | (mm_q4z_nib<GROUP>(raw, c + 8, k0 + 1, row_bytes) << 24)
                 | (mm_q4z_nib<GROUP>(raw, c + 8, k0 + 9, row_bytes) << 28);
    }
    dst_b[((size_t)kt * n_tiles + nt) * 32 + lane] = make_uint4(words[0], words[1], words[2], words[3]);
}

template <int GROUP>
__global__ void moe_q4z_to_marlin_sz_kernel(const uint8_t *__restrict__ raw_arena, size_t raw_stride, uint8_t *__restrict__ dst_arena,
                                            size_t dst_stride, size_t s_off, size_t z_off, int rows, int cols) {
    constexpr int BLOCK = GROUP / 2 + 3;
    const int nt = (int)blockIdx.x, g = (int)blockIdx.y, e = (int)blockIdx.z;
    const int idx = (int)threadIdx.x;
    const uint8_t *raw = raw_arena + (size_t)e * raw_stride;
    uint8_t *dst = dst_arena + (size_t)e * dst_stride;
    uint16_t *scales = reinterpret_cast<uint16_t *>(dst + s_off);
    uint32_t *zeros = reinterpret_cast<uint32_t *>(dst + z_off);
    const size_t row_bytes = (size_t)(cols / GROUP) * BLOCK;
    const int col_s = nt * 64 + (idx >> 3) + 8 * (idx & 7);
    scales[(size_t)g * rows + nt * 64 + idx] = ld16_a1(raw + (size_t)col_s * row_bytes + (size_t)g * BLOCK + GROUP / 2);
    if (idx < 8) {
        constexpr int INTERLEAVE[8] = {0, 2, 4, 6, 1, 3, 5, 7};
        uint32_t zw = 0;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int col_z = nt * 64 + idx + 8 * INTERLEAVE[i];
            const uint32_t z = (uint32_t)(raw[(size_t)col_z * row_bytes + (size_t)g * BLOCK + GROUP / 2 + 2] & 15u);
            zw |= z << (4 * i);
        }
        zeros[(size_t)g * (rows >> 3) + nt * 8 + idx] = zw;
    }
}

int fllm_moe_marlin_pack_launch(cudaStream_t s, const void *raw_arena, size_t raw_stride, void *dst_arena, size_t dst_stride, int experts,
                                int rows, int cols, int group) {
    if (experts <= 0 || rows <= 0 || cols <= 0) return 0;
    if ((rows & 63) != 0 || (cols & 127) != 0 || (cols % group) != 0 || (group != 32 && group != 64 && group != 128))
        return fllm_set_error("moe_marlin_pack: unsupported shape (%d x %d, g=%d)", rows, cols, group);
    size_t s_off, z_off, total;
    mm_offsets(rows, cols, group, s_off, z_off, total);
    if (dst_stride < total) return fllm_set_error("moe_marlin_pack: dst_stride too small");
    const uint8_t *raw = static_cast<const uint8_t *>(raw_arena);
    uint8_t *dst = static_cast<uint8_t *>(dst_arena);
    const dim3 gb(rows / 64, cols / 16, experts);
    const dim3 gsz(rows / 64, cols / group, experts);
    if (group == 32) {
        moe_q4z_to_marlin_b_kernel<32><<<gb, 32, 0, s>>>(raw, raw_stride, dst, dst_stride, rows, cols);
        moe_q4z_to_marlin_sz_kernel<32><<<gsz, 64, 0, s>>>(raw, raw_stride, dst, dst_stride, s_off, z_off, rows, cols);
    } else if (group == 64) {
        moe_q4z_to_marlin_b_kernel<64><<<gb, 32, 0, s>>>(raw, raw_stride, dst, dst_stride, rows, cols);
        moe_q4z_to_marlin_sz_kernel<64><<<gsz, 64, 0, s>>>(raw, raw_stride, dst, dst_stride, s_off, z_off, rows, cols);
    } else {
        moe_q4z_to_marlin_b_kernel<128><<<gb, 32, 0, s>>>(raw, raw_stride, dst, dst_stride, rows, cols);
        moe_q4z_to_marlin_sz_kernel<128><<<gsz, 64, 0, s>>>(raw, raw_stride, dst, dst_stride, s_off, z_off, rows, cols);
    }
    return fllm_check_launch("moe_marlin_pack");
}

#if !defined(__HIPCC__)

__device__ __forceinline__ float mm_glu(float g, float u, int mode, float alpha, float limit) {
    if (mode == 2) {
        const float gc = fminf(g, limit);
        const float uc = fminf(fmaxf(u, -limit), limit);
        return (uc + 1.f) * gc / (1.f + expf(-alpha * gc));
    }
    return fllm_glu(g, u, mode);
}

template <int BN>
struct MoeMarlinTile {
    static constexpr int BK = 64, LD = BK + 8;
    static constexpr int WM = 1;
    static constexpr int WN = 4;
    static constexpr int NT = WM * WN * 32;  // 128 for both BN=16 and BN=64
    static constexpr int MI = BN / WM / 16;  // 1 for BN=16, 4 for BN=64
    static constexpr int RPK = BK / 16;      // 4
    static constexpr int BRUNS = BN * RPK;   // 64 for BN=16, 256 for BN=64
    static constexpr size_t SMEM = (size_t)BN * LD * sizeof(__half);
};

template <bool F16_IO>
__device__ __forceinline__ void mm_load16_x_h2(const float *p, int e0, float scale, bool has_pw, __half2 *out) {
    if constexpr (F16_IO) {
        const uint4 *q = reinterpret_cast<const uint4 *>(reinterpret_cast<const __half *>(p) + e0);
        const uint4 v0 = __ldg(q);
        const uint4 v1 = __ldg(q + 1);
        const __half2 *s0 = reinterpret_cast<const __half2 *>(&v0);
        const __half2 *s1 = reinterpret_cast<const __half2 *>(&v1);
        if (has_pw) {
            const __half2 hs = __float2half2_rn(scale);
#pragma unroll
            for (int i = 0; i < 4; i++) {
                out[i] = __hmul2(s0[i], hs);
                out[4 + i] = __hmul2(s1[i], hs);
            }
        } else {
#pragma unroll
            for (int i = 0; i < 4; i++) {
                out[i] = s0[i];
                out[4 + i] = s1[i];
            }
        }
    } else {
        const float4 *q = reinterpret_cast<const float4 *>(p + e0);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            float4 v = __ldg(q + i);
            if (has_pw) {
                v.x *= scale;
                v.y *= scale;
                v.z *= scale;
                v.w *= scale;
            }
            out[2 * i] = __floats2half2_rn(v.x, v.y);
            out[2 * i + 1] = __floats2half2_rn(v.z, v.w);
        }
    }
}

__device__ __forceinline__ void mm_store16_h2(__half *dst, const __half2 *v) {
    const uint4 *src = reinterpret_cast<const uint4 *>(v);
    uint4 *d = reinterpret_cast<uint4 *>(dst);
    d[0] = src[0];
    d[1] = src[1];
}

__device__ __forceinline__ void mm_load_sz_raw(const uint16_t *S, const uint32_t *Z, int g, int N, int nt, int wn, int c8, uint32_t &sp,
                                               uint32_t &zw) {
    sp = __ldg(reinterpret_cast<const uint32_t *>(S + (size_t)g * N + nt * 64 + 8 * c8 + 2 * wn));
    zw = __ldg(Z + (size_t)g * (N >> 3) + nt * 8 + c8);
}

__device__ __forceinline__ void mm_dequant_pair_sz(uint32_t q, uint32_t sp, uint32_t zw, int wn, uint32_t (&bf0)[2], uint32_t (&bf1)[2]) {
    constexpr uint32_t LO = 0x000f000fu, HI = 0x00f000f0u, EX = 0x64006400u, MUL16 = 0x2c002c00u;
    __half2 h_mul16, hsp;
    memcpy(&h_mul16, &MUL16, 4);
    memcpy(&hsp, &sp, 4);

    const __half2 hs0 = __half2half2(__low2half(hsp));
    const uint32_t z0 = (zw >> (4 * wn)) & 15u;
    const __half2 hn0 = __half2half2(__ushort_as_half((uint16_t)(0xE400u | z0)));
    const __half2 hm0 = __half2half2(__ushort_as_half((uint16_t)(0xD400u | (z0 << 4))));
    const uint32_t lo0 = lop3_eax(q, LO, EX), hi0 = lop3_eax(q, HI, EX);
    __half2 f_lo0, f_hi0;
    memcpy(&f_lo0, &lo0, 4);
    memcpy(&f_hi0, &hi0, 4);
    const __half2 d0_0 = __hmul2(__hadd2(f_lo0, hn0), hs0);
    const __half2 d0_1 = __hmul2(__hfma2(f_hi0, h_mul16, hm0), hs0);

    const __half2 hs1 = __half2half2(__high2half(hsp));
    const uint32_t z1 = (zw >> (4 * (wn + 4))) & 15u;
    const __half2 hn1 = __half2half2(__ushort_as_half((uint16_t)(0xE400u | z1)));
    const __half2 hm1 = __half2half2(__ushort_as_half((uint16_t)(0xD400u | (z1 << 4))));
    const uint32_t q2 = q >> 8;
    const uint32_t lo1 = lop3_eax(q2, LO, EX), hi1 = lop3_eax(q2, HI, EX);
    __half2 f_lo1, f_hi1;
    memcpy(&f_lo1, &lo1, 4);
    memcpy(&f_hi1, &hi1, 4);
    const __half2 d1_0 = __hmul2(__hadd2(f_lo1, hn1), hs1);
    const __half2 d1_1 = __hmul2(__hfma2(f_hi1, h_mul16, hm1), hs1);

    memcpy(&bf0[0], &d0_0, 4);
    memcpy(&bf0[1], &d0_1, 4);
    memcpy(&bf1[0], &d1_0, 4);
    memcpy(&bf1[1], &d1_1, 4);
}

template <int BN, bool GLU, int GROUP, bool F16_IO>
__global__ void __launch_bounds__(MoeMarlinTile<BN>::NT)
moe_marlin_tc_kernel(const uint8_t *const *__restrict__ wtab, const uint8_t *const *__restrict__ wtab2,
                     const float *const *__restrict__ btab, const float *const *__restrict__ btab2, const float *__restrict__ X,
                     const float *__restrict__ pw, float *__restrict__ Y, const int32_t *__restrict__ sorted,
                     const int32_t *__restrict__ block_expert, const int32_t *__restrict__ num_post, int rows, int cols, int pairs,
                     int xdiv, int glu_mode, float alpha, float limit) {
#if FLLM_TC
    using T = MoeMarlinTile<BN>;
    if ((int)blockIdx.y * BN >= num_post[0]) return;
    const int e = block_expert[blockIdx.y];
    if (e < 0) return;

    extern __shared__ __align__(16) unsigned char mm_smem[];
    __half *Bs = reinterpret_cast<__half *>(mm_smem);
    const int tid = (int)threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp % T::WM, wn = warp / T::WM;
    const int nt = (int)blockIdx.x, n_tiles = rows >> 6;
    const int32_t *my_sorted = sorted + (size_t)blockIdx.y * BN;
    const int ktiles = cols >> 6;  // cols / 64

    size_t s_off, z_off, dummy_tot;
    mm_offsets(rows, cols, GROUP, s_off, z_off, dummy_tot);
    const uint4 *Bg = reinterpret_cast<const uint4 *>(wtab[e]);
    const uint16_t *Sg = reinterpret_cast<const uint16_t *>(wtab[e] + s_off);
    const uint32_t *Zg = reinterpret_cast<const uint32_t *>(wtab[e] + z_off);
    const uint4 *Bu = GLU ? reinterpret_cast<const uint4 *>(wtab2[e]) : nullptr;
    const uint16_t *Su = GLU ? reinterpret_cast<const uint16_t *>(wtab2[e] + s_off) : nullptr;
    const uint32_t *Zu = GLU ? reinterpret_cast<const uint32_t *>(wtab2[e] + z_off) : nullptr;

    constexpr int XLOADS = (T::BRUNS + T::NT - 1) / T::NT;
    const float *xrow[XLOADS];
    float xsc[XLOADS];
#pragma unroll
    for (int b = 0; b < XLOADS; b++) {
        const int eidx = tid + b * T::NT;
        if (eidx < T::BRUNS) {
            const int p = my_sorted[eidx / T::RPK];
            if constexpr (F16_IO) {
                xrow[b] = p < pairs ? reinterpret_cast<const float *>(reinterpret_cast<const __half *>(X) + (size_t)(p / xdiv) * cols)
                                    : nullptr;
            } else {
                xrow[b] = p < pairs ? X + (size_t)(p / xdiv) * cols : nullptr;
            }
            xsc[b] = (pw && p < pairs) ? pw[p] : 1.f;
        } else {
            xrow[b] = nullptr;
            xsc[b] = 1.f;
        }
    }

    float acc_g[T::MI][2][4];
    float acc_u[GLU ? T::MI : 1][2][4];
#pragma unroll
    for (int mi = 0; mi < T::MI; mi++)
#pragma unroll
        for (int ni = 0; ni < 2; ni++)
#pragma unroll
            for (int c = 0; c < 4; c++) {
                acc_g[mi][ni][c] = 0.f;
                if constexpr (GLU) acc_u[mi][ni][c] = 0.f;
            }

    const int c8 = lane >> 2;
    uint32_t sp_g = 0, zw_g = 0, sp_u = 0, zw_u = 0;
    int cached_g = -1;

    for (int kt64 = 0; kt64 < ktiles; kt64++) {
#pragma unroll
        for (int b = 0; b < XLOADS; b++) {
            const int eidx = tid + b * T::NT;
            if (eidx < T::BRUNS) {
                alignas(16) __half2 xv[8];
                if (xrow[b]) {
                    mm_load16_x_h2<F16_IO>(xrow[b], kt64 * T::BK + (eidx % T::RPK) * 16, xsc[b], pw != nullptr, xv);
                } else {
                    const __half2 zero = __halves2half2(__ushort_as_half(0), __ushort_as_half(0));
#pragma unroll
                    for (int j = 0; j < 8; j++) xv[j] = zero;
                }
                mm_store16_h2(Bs + (size_t)(eidx / T::RPK) * T::LD + (eidx % T::RPK) * 16, xv);
            }
        }
        __syncthreads();

#pragma unroll
        for (int sub = 0; sub < 4; sub++) {
            const int kt16 = kt64 * 4 + sub;
            const int g = kt16 / (GROUP >> 4);
            if (g != cached_g) {
                cached_g = g;
                mm_load_sz_raw(Sg, Zg, g, rows, nt, wn, c8, sp_g, zw_g);
                if constexpr (GLU) mm_load_sz_raw(Su, Zu, g, rows, nt, wn, c8, sp_u, zw_u);
            }
            const size_t tile_off = ((size_t)kt16 * n_tiles + nt) * 32 + lane;
            const uint32_t qg = __ldg(reinterpret_cast<const uint32_t *>(Bg + tile_off) + wn);
            const uint32_t qu = GLU ? __ldg(reinterpret_cast<const uint32_t *>(Bu + tile_off) + wn) : 0u;
            uint32_t bg0[2], bg1[2];
            mm_dequant_pair_sz(qg, sp_g, zw_g, wn, bg0, bg1);
            uint32_t bu0[2], bu1[2];
            if constexpr (GLU) mm_dequant_pair_sz(qu, sp_u, zw_u, wn, bu0, bu1);
#pragma unroll
            for (int mi = 0; mi < T::MI; mi++) {
                uint32_t af[4];
                ldmatrix4(af[0], af[1], af[2], af[3],
                          Bs + (size_t)(wm * (BN / T::WM) + mi * 16 + (lane & 15)) * T::LD + sub * 16 + (lane >> 4) * 8);
                mma16816(acc_g[mi][0], af, bg0[0], bg0[1]);
                mma16816(acc_g[mi][1], af, bg1[0], bg1[1]);
                if constexpr (GLU) {
                    mma16816(acc_u[mi][0], af, bu0[0], bu0[1]);
                    mma16816(acc_u[mi][1], af, bu1[0], bu1[1]);
                }
            }
        }
        if (kt64 + 1 < ktiles) __syncthreads();
    }

    const int grp = lane >> 2, t = lane & 3;
    const float *bg = btab ? btab[e] : nullptr;
    const float *bu = (GLU && btab2) ? btab2[e] : nullptr;
#pragma unroll
    for (int mi = 0; mi < T::MI; mi++)
#pragma unroll
        for (int ni = 0; ni < 2; ni++)
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int tok_slot = wm * (BN / T::WM) + mi * 16 + grp + (c >> 1) * 8;
                const int p = my_sorted[tok_slot];
                if (p < pairs) {
                    const int col = nt * 64 + wn * 16 + ni * 8 + 2 * t + (c & 1);
                    const float gv = acc_g[mi][ni][c] + (bg ? bg[col] : 0.f);
                    float out_v = gv;
                    if constexpr (GLU) {
                        const float uv = acc_u[mi][ni][c] + (bu ? bu[col] : 0.f);
                        out_v = mm_glu(gv, uv, glu_mode, alpha, limit);
                    }
                    if constexpr (F16_IO) {
                        reinterpret_cast<__half *>(Y)[(size_t)p * rows + col] = __float2half(out_v);
                    } else {
                        Y[(size_t)p * rows + col] = out_v;
                    }
                }
            }
#endif
}

template <int BN, bool GLU, int GROUP>
static int mm_tc_run(cudaStream_t s, const void *const *wtab, const void *const *wtab2, const void *const *btab, const void *const *btab2,
                     const float *X, const float *pw, float *Y, const int32_t *sorted, const int32_t *block_expert, const int32_t *num_post,
                     int rows, int cols, int pairs, int xdiv, int max_blocks, int glu_mode, float alpha, float limit, int act_dt) {
    using T = MoeMarlinTile<BN>;
    const dim3 grid(rows / 64, max_blocks), block(T::NT);
    if (act_dt == 1) {
        moe_marlin_tc_kernel<BN, GLU, GROUP, true><<<grid, block, T::SMEM, s>>>(
            reinterpret_cast<const uint8_t *const *>(wtab), reinterpret_cast<const uint8_t *const *>(wtab2),
            reinterpret_cast<const float *const *>(btab), reinterpret_cast<const float *const *>(btab2), X, pw, Y, sorted, block_expert,
            num_post, rows, cols, pairs, xdiv, glu_mode, alpha, limit);
    } else {
        moe_marlin_tc_kernel<BN, GLU, GROUP, false><<<grid, block, T::SMEM, s>>>(
            reinterpret_cast<const uint8_t *const *>(wtab), reinterpret_cast<const uint8_t *const *>(wtab2),
            reinterpret_cast<const float *const *>(btab), reinterpret_cast<const float *const *>(btab2), X, pw, Y, sorted, block_expert,
            num_post, rows, cols, pairs, xdiv, glu_mode, alpha, limit);
    }
    return fllm_check_launch("moe_marlin_grouped");
}

int fllm_moe_marlin_grouped_launch(cudaStream_t s, uint32_t kind, int glu, const void *const *wtab, const void *const *wtab2,
                                   const void *const *btab, const void *const *btab2, const float *X, const float *pw, float *Y,
                                   const int32_t *sorted, const int32_t *block_expert, const int32_t *num_post, int rows, int cols,
                                   int pairs, int xdiv, int block, int max_blocks, int glu_mode, float alpha, float limit, int act_dt) {
    if (pairs <= 0 || rows <= 0) return 0;
    const int group = kind == KIND_Q4ZM32 ? 32 : kind == KIND_Q4ZM64 ? 64 : kind == KIND_Q4ZM128 ? 128 : 0;
    if (group == 0 || (rows & 63) != 0 || (cols & 127) != 0 || (cols % group) != 0 || (block != 16 && block != 64) || xdiv < 1 ||
        max_blocks < 1 || (glu && !wtab2) || (act_dt != 0 && act_dt != 1))
        return fllm_set_error("moe_marlin_grouped: unsupported shape or kind");
    if ((reinterpret_cast<uintptr_t>(X) & 15) != 0) return fllm_set_error("moe_marlin_grouped: unaligned activations");
    if (fllm_cc() < 75) return fllm_set_error("moe_marlin_grouped: requires sm_75+");
    if (glu) {
        if (block == 16) {
            return group == 32 ? mm_tc_run<16, true, 32>(s, wtab, wtab2, btab, btab2, X, pw, Y, sorted, block_expert, num_post, rows, cols,
                                                         pairs, xdiv, max_blocks, glu_mode, alpha, limit, act_dt)
                 : group == 64 ? mm_tc_run<16, true, 64>(s, wtab, wtab2, btab, btab2, X, pw, Y, sorted, block_expert, num_post, rows, cols,
                                                         pairs, xdiv, max_blocks, glu_mode, alpha, limit, act_dt)
                               : mm_tc_run<16, true, 128>(s, wtab, wtab2, btab, btab2, X, pw, Y, sorted, block_expert, num_post, rows, cols,
                                                          pairs, xdiv, max_blocks, glu_mode, alpha, limit, act_dt);
        }
        return group == 32 ? mm_tc_run<64, true, 32>(s, wtab, wtab2, btab, btab2, X, pw, Y, sorted, block_expert, num_post, rows, cols,
                                                     pairs, xdiv, max_blocks, glu_mode, alpha, limit, act_dt)
             : group == 64 ? mm_tc_run<64, true, 64>(s, wtab, wtab2, btab, btab2, X, pw, Y, sorted, block_expert, num_post, rows, cols,
                                                     pairs, xdiv, max_blocks, glu_mode, alpha, limit, act_dt)
                           : mm_tc_run<64, true, 128>(s, wtab, wtab2, btab, btab2, X, pw, Y, sorted, block_expert, num_post, rows, cols,
                                                      pairs, xdiv, max_blocks, glu_mode, alpha, limit, act_dt);
    }
    if (block == 16) {
        return group == 32 ? mm_tc_run<16, false, 32>(s, wtab, wtab2, btab, btab2, X, pw, Y, sorted, block_expert, num_post, rows, cols,
                                                      pairs, xdiv, max_blocks, glu_mode, alpha, limit, act_dt)
             : group == 64 ? mm_tc_run<16, false, 64>(s, wtab, wtab2, btab, btab2, X, pw, Y, sorted, block_expert, num_post, rows, cols,
                                                      pairs, xdiv, max_blocks, glu_mode, alpha, limit, act_dt)
                           : mm_tc_run<16, false, 128>(s, wtab, wtab2, btab, btab2, X, pw, Y, sorted, block_expert, num_post, rows, cols,
                                                       pairs, xdiv, max_blocks, glu_mode, alpha, limit, act_dt);
    }
    return group == 32 ? mm_tc_run<64, false, 32>(s, wtab, wtab2, btab, btab2, X, pw, Y, sorted, block_expert, num_post, rows, cols, pairs,
                                                  xdiv, max_blocks, glu_mode, alpha, limit, act_dt)
         : group == 64 ? mm_tc_run<64, false, 64>(s, wtab, wtab2, btab, btab2, X, pw, Y, sorted, block_expert, num_post, rows, cols, pairs,
                                                  xdiv, max_blocks, glu_mode, alpha, limit, act_dt)
                       : mm_tc_run<64, false, 128>(s, wtab, wtab2, btab, btab2, X, pw, Y, sorted, block_expert, num_post, rows, cols, pairs,
                                                   xdiv, max_blocks, glu_mode, alpha, limit, act_dt);
}

// Per-slot GEMV for Marlin MoE weights (`slots` tokens/experts pairs; fuses gate + up + GLU when GLU=true).
template <bool GLU, int U, bool F16_IO>
__global__ void __launch_bounds__(128)
moe_marlin_gemv_kernel(const uint8_t *const *__restrict__ wtab, const uint8_t *const *__restrict__ wtab2,
                       const float *const *__restrict__ btab, const float *const *__restrict__ btab2, const int32_t *__restrict__ ids,
                       const float *__restrict__ X, float *__restrict__ Y, int rows, int cols, int xdiv, int gtiles, int glu_mode,
                       float alpha, float limit, size_t s_off, size_t z_off) {
    constexpr int kWarps = 4;
    __shared__ float red[kWarps][GLU ? 2 : 1][64];
    const int nt = (int)blockIdx.x, n_tiles = (int)gridDim.x, slot = (int)blockIdx.y;
    const int e = ids[slot];
    const int warp = (int)threadIdx.x >> 5, lane = (int)threadIdx.x & 31;
    const int r = (lane & 3) * 2, c8 = lane >> 2;
    const int units = (cols >> 4) / U;

    const uint4 *Bg = reinterpret_cast<const uint4 *>(wtab[e]);
    const uint16_t *Sg = reinterpret_cast<const uint16_t *>(wtab[e] + s_off);
    const uint32_t *Zg = reinterpret_cast<const uint32_t *>(wtab[e] + z_off);
    const uint4 *Bu = GLU ? reinterpret_cast<const uint4 *>(wtab2[e]) : nullptr;
    const uint16_t *Su = GLU ? reinterpret_cast<const uint16_t *>(wtab2[e] + s_off) : nullptr;
    const uint32_t *Zu = GLU ? reinterpret_cast<const uint32_t *>(wtab2[e] + z_off) : nullptr;
    const float *xp = F16_IO ? nullptr : (X + (size_t)(slot / xdiv) * cols);
    const __half *xh = F16_IO ? (reinterpret_cast<const __half *>(X) + (size_t)(slot / xdiv) * cols) : nullptr;

    float total_g[4][2];
    float total_u[GLU ? 4 : 1][2];
#pragma unroll
    for (int w = 0; w < 4; w++) {
        total_g[w][0] = total_g[w][1] = 0.f;
        if constexpr (GLU) total_u[w][0] = total_u[w][1] = 0.f;
    }

    for (int u = warp; u < units; u += kWarps) {
        const int kt0 = u * U, g = kt0 / gtiles;
        uint4 wqg[U], wqu[GLU ? U : 1];
#pragma unroll
        for (int i = 0; i < U; i++) {
            const size_t off = ((size_t)(kt0 + i) * n_tiles + nt) * 32 + lane;
            wqg[i] = __ldcs(Bg + off);
            if constexpr (GLU) wqu[i] = __ldcs(Bu + off);
        }
        const uint4 svg = *reinterpret_cast<const uint4 *>(Sg + (size_t)g * rows + nt * 64 + 8 * c8);
        const uint32_t zwg = Zg[(size_t)g * (rows >> 3) + nt * 8 + c8];
        uint4 svu{};
        uint32_t zwu = 0;
        if constexpr (GLU) {
            svu = *reinterpret_cast<const uint4 *>(Su + (size_t)g * rows + nt * 64 + 8 * c8);
            zwu = Zu[(size_t)g * (rows >> 3) + nt * 8 + c8];
        }

        float accg[4][2], accu[GLU ? 4 : 1][2];
        float xs = 0.f;
#pragma unroll
        for (int w = 0; w < 4; w++) {
            accg[w][0] = accg[w][1] = 0.f;
            if constexpr (GLU) accu[w][0] = accu[w][1] = 0.f;
        }
#pragma unroll
        for (int i = 0; i < U; i++) {
            const int kk = (kt0 + i) * 16 + r;
            float2 xa, xb;
            if constexpr (F16_IO) {
                xa = __half22float2(*reinterpret_cast<const __half2 *>(xh + kk));
                xb = __half22float2(*reinterpret_cast<const __half2 *>(xh + kk + 8));
            } else {
                xa = *reinterpret_cast<const float2 *>(xp + kk);
                xb = *reinterpret_cast<const float2 *>(xp + kk + 8);
            }
            xs += (xa.x + xa.y) + (xb.x + xb.y);
            const uint32_t words_g[4] = {wqg[i].x, wqg[i].y, wqg[i].z, wqg[i].w};
#pragma unroll
            for (int w = 0; w < 4; w++) {
                const uint32_t v = words_g[w];
                const float q0 = (float)(v & 15u), q1 = (float)((v >> 4) & 15u), q2 = (float)((v >> 8) & 15u), q3 = (float)((v >> 12) & 15u);
                const float q4 = (float)((v >> 16) & 15u), q5 = (float)((v >> 20) & 15u), q6 = (float)((v >> 24) & 15u), q7 = (float)(v >> 28);
                accg[w][0] = fmaf(q0, xa.x, fmaf(q1, xb.x, fmaf(q4, xa.y, fmaf(q5, xb.y, accg[w][0]))));
                accg[w][1] = fmaf(q2, xa.x, fmaf(q3, xb.x, fmaf(q6, xa.y, fmaf(q7, xb.y, accg[w][1]))));
            }
            if constexpr (GLU) {
                const uint32_t words_u[4] = {wqu[i].x, wqu[i].y, wqu[i].z, wqu[i].w};
#pragma unroll
                for (int w = 0; w < 4; w++) {
                    const uint32_t v = words_u[w];
                    const float q0 = (float)(v & 15u), q1 = (float)((v >> 4) & 15u), q2 = (float)((v >> 8) & 15u), q3 = (float)((v >> 12) & 15u);
                    const float q4 = (float)((v >> 16) & 15u), q5 = (float)((v >> 20) & 15u), q6 = (float)((v >> 24) & 15u), q7 = (float)(v >> 28);
                    accu[w][0] = fmaf(q0, xa.x, fmaf(q1, xb.x, fmaf(q4, xa.y, fmaf(q5, xb.y, accu[w][0]))));
                    accu[w][1] = fmaf(q2, xa.x, fmaf(q3, xb.x, fmaf(q6, xa.y, fmaf(q7, xb.y, accu[w][1]))));
                }
            }
        }
        const uint32_t swg[4] = {svg.x, svg.y, svg.z, svg.w};
#pragma unroll
        for (int w = 0; w < 4; w++) {
            __half2 sh;
            memcpy(&sh, &swg[w], 4);
            const float2 s = __half22float2(sh);
            const float z0 = (float)((zwg >> (4 * w)) & 15u);
            const float z1 = (float)((zwg >> (4 * (w + 4))) & 15u);
            total_g[w][0] = fmaf(s.x, accg[w][0] - z0 * xs, total_g[w][0]);
            total_g[w][1] = fmaf(s.y, accg[w][1] - z1 * xs, total_g[w][1]);
        }
        if constexpr (GLU) {
            const uint32_t swu[4] = {svu.x, svu.y, svu.z, svu.w};
#pragma unroll
            for (int w = 0; w < 4; w++) {
                __half2 sh;
                memcpy(&sh, &swu[w], 4);
                const float2 s = __half22float2(sh);
                const float z0 = (float)((zwu >> (4 * w)) & 15u);
                const float z1 = (float)((zwu >> (4 * (w + 4))) & 15u);
                total_u[w][0] = fmaf(s.x, accu[w][0] - z0 * xs, total_u[w][0]);
                total_u[w][1] = fmaf(s.y, accu[w][1] - z1 * xs, total_u[w][1]);
            }
        }
    }

#pragma unroll
    for (int w = 0; w < 4; w++) {
#pragma unroll
        for (int p = 0; p < 2; p++) {
            float v = total_g[w][p];
            v += __shfl_xor_sync(0xffffffffu, v, 1);
            v += __shfl_xor_sync(0xffffffffu, v, 2);
            total_g[w][p] = v;
            if constexpr (GLU) {
                float u = total_u[w][p];
                u += __shfl_xor_sync(0xffffffffu, u, 1);
                u += __shfl_xor_sync(0xffffffffu, u, 2);
                total_u[w][p] = u;
            }
        }
    }
    if ((lane & 3) == 0) {
#pragma unroll
        for (int w = 0; w < 4; w++) {
            red[warp][0][w * 16 + c8] = total_g[w][0];
            red[warp][0][w * 16 + c8 + 8] = total_g[w][1];
            if constexpr (GLU) {
                red[warp][1][w * 16 + c8] = total_u[w][0];
                red[warp][1][w * 16 + c8 + 8] = total_u[w][1];
            }
        }
    }
    __syncthreads();
    if (threadIdx.x < 64) {
        const int c = (int)threadIdx.x;
        const int col = nt * 64 + c;
        float gv = 0.f;
#pragma unroll
        for (int w = 0; w < kWarps; w++) gv += red[w][0][c];
        if (btab) gv += btab[e][col];
        float out_v = gv;
        if constexpr (GLU) {
            float uv = 0.f;
#pragma unroll
            for (int w = 0; w < kWarps; w++) uv += red[w][1][c];
            if (btab2) uv += btab2[e][col];
            out_v = mm_glu(gv, uv, glu_mode, alpha, limit);
        }
        if constexpr (F16_IO) {
            reinterpret_cast<__half *>(Y)[(size_t)slot * rows + col] = __float2half(out_v);
        } else {
            Y[(size_t)slot * rows + col] = out_v;
        }
    }
}

int fllm_moe_marlin_linear_launch(cudaStream_t s, const void *const *tab, const void *const *btab, int kind, const int32_t *ids,
                                  const float *X, float *Y, int rows, int cols, int slots, int xdiv, int act_dt) {
    const uint32_t k = (uint32_t)kind;
    const int group = k == KIND_Q4ZM32 ? 32 : k == KIND_Q4ZM64 ? 64 : k == KIND_Q4ZM128 ? 128 : 0;
    if (group == 0 || (rows & 63) != 0 || (cols & 127) != 0 || (cols % group) != 0 || slots <= 0 || xdiv <= 0 ||
        (act_dt != 0 && act_dt != 1))
        return fllm_set_error("moe_marlin_linear: unsupported shape");
    if ((reinterpret_cast<uintptr_t>(X) & (act_dt == 1 ? 3 : 7)) != 0)
        return fllm_set_error("moe_marlin_linear: activations unaligned");
    size_t s_off, z_off, tot;
    mm_offsets(rows, cols, group, s_off, z_off, tot);
    const int u = group == 32 ? 2 : 4;
    const dim3 grid(rows / 64, slots), block(128);
    const uint8_t *const *wt = reinterpret_cast<const uint8_t *const *>(tab);
    const float *const *bt = reinterpret_cast<const float *const *>(btab);
    if (act_dt == 1) {
        if (u == 2) moe_marlin_gemv_kernel<false, 2, true><<<grid, block, 0, s>>>(wt, nullptr, bt, nullptr, ids, X, Y, rows, cols, xdiv, group >> 4, 0, 0.f, 0.f, s_off, z_off);
        else moe_marlin_gemv_kernel<false, 4, true><<<grid, block, 0, s>>>(wt, nullptr, bt, nullptr, ids, X, Y, rows, cols, xdiv, group >> 4, 0, 0.f, 0.f, s_off, z_off);
    } else {
        if (u == 2) moe_marlin_gemv_kernel<false, 2, false><<<grid, block, 0, s>>>(wt, nullptr, bt, nullptr, ids, X, Y, rows, cols, xdiv, group >> 4, 0, 0.f, 0.f, s_off, z_off);
        else moe_marlin_gemv_kernel<false, 4, false><<<grid, block, 0, s>>>(wt, nullptr, bt, nullptr, ids, X, Y, rows, cols, xdiv, group >> 4, 0, 0.f, 0.f, s_off, z_off);
    }
    return fllm_check_launch("moe_marlin_linear");
}

int fllm_moe_marlin_linear_glu_launch(cudaStream_t s, const void *const *wtab_gate, const void *const *wtab_up,
                                      const void *const *btab_gate, const void *const *btab_up, int kind, const int32_t *ids,
                                      const float *X, float *Y, int rows, int cols, int slots, int xdiv, int glu_mode, float alpha,
                                      float limit, int act_dt) {
    const uint32_t k = (uint32_t)kind;
    const int group = k == KIND_Q4ZM32 ? 32 : k == KIND_Q4ZM64 ? 64 : k == KIND_Q4ZM128 ? 128 : 0;
    if (group == 0 || (rows & 63) != 0 || (cols & 127) != 0 || (cols % group) != 0 || slots <= 0 || xdiv <= 0 ||
        (act_dt != 0 && act_dt != 1))
        return fllm_set_error("moe_marlin_linear_glu: unsupported shape");
    if ((reinterpret_cast<uintptr_t>(X) & (act_dt == 1 ? 3 : 7)) != 0)
        return fllm_set_error("moe_marlin_linear_glu: activations unaligned");
    size_t s_off, z_off, tot;
    mm_offsets(rows, cols, group, s_off, z_off, tot);
    const int u = group == 32 ? 2 : 4;
    const dim3 grid(rows / 64, slots), block(128);
    const uint8_t *const *wg = reinterpret_cast<const uint8_t *const *>(wtab_gate);
    const uint8_t *const *wu = reinterpret_cast<const uint8_t *const *>(wtab_up);
    const float *const *bg = reinterpret_cast<const float *const *>(btab_gate);
    const float *const *bu = reinterpret_cast<const float *const *>(btab_up);
    if (act_dt == 1) {
        if (u == 2) moe_marlin_gemv_kernel<true, 2, true><<<grid, block, 0, s>>>(wg, wu, bg, bu, ids, X, Y, rows, cols, xdiv, group >> 4, glu_mode, alpha, limit, s_off, z_off);
        else moe_marlin_gemv_kernel<true, 4, true><<<grid, block, 0, s>>>(wg, wu, bg, bu, ids, X, Y, rows, cols, xdiv, group >> 4, glu_mode, alpha, limit, s_off, z_off);
    } else {
        if (u == 2) moe_marlin_gemv_kernel<true, 2, false><<<grid, block, 0, s>>>(wg, wu, bg, bu, ids, X, Y, rows, cols, xdiv, group >> 4, glu_mode, alpha, limit, s_off, z_off);
        else moe_marlin_gemv_kernel<true, 4, false><<<grid, block, 0, s>>>(wg, wu, bg, bu, ids, X, Y, rows, cols, xdiv, group >> 4, glu_mode, alpha, limit, s_off, z_off);
    }
    return fllm_check_launch("moe_marlin_linear_glu");
}

#endif  // !__HIPCC__
