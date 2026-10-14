// fLLM CUDA backend: NVFP4 GEMM on the FP4 tensor cores of Blackwell (sm_120a).
//
//   Y[token][row] = gscale * sum_k X[token][k] * W[row][k]  (+ bias[row])
//
// Weights use the GGUF NVFP4 block (64 values = 4 UE4M3 scales, one per 16 values, + 32 bytes of E2M1
// nibbles; inside a 16-value sub-block byte j holds element j in the low and element j + 8 in the high
// nibble). `gscale` is the per-tensor weight scale of the ModelOpt checkpoint.
//
// The activations are quantized on the fly to the same format (one UE4M3 scale per 16 values, no tensor
// scale needed because the dynamic range of E4M3 covers activations). `mma.m16n8k64.kind::mxf4nvf4.block_scale`
// then multiplies 64 values per instruction with the scales applied inside the tensor core.
//
// The weights are stored pre-arranged for the tensor-core registers (backend/cuda.rs), in tiles of 16 rows: for every
// 256 values (one pipeline stage) a tile holds 4 steps of 32 lanes x 16 bytes, and one 16-byte shared-memory load per
// lane fills the four A registers {row g, row g + 8} x {low, high half of k} in the order the mma wants them, so no
// register shuffling is needed (that cost four MOVs per mma before). The 16 x 4 x 4 scale bytes of the tile follow the
// data. The activations use the same natural order per token.
#include <cstdlib>

#include "common.cuh"
#include "fllm_cuda_internal.h"
#include "mma_util.cuh"

#if defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_SPECIFIC__) || defined(__CUDA_ARCH_FAMILY_SPECIFIC__)
#define FLLM_FP4_MMA 1
#endif

#include "nvfp4_quant.cuh"

// xq[token][cols / 2 bytes], xs[token][cols / 16]; one thread per 16-value block.
template <bool GLU>
__global__ void nvfp4_quantize_kernel(const float *__restrict__ x, const float *__restrict__ up, int mode,
                                      uint32_t *__restrict__ xq, uint8_t *__restrict__ xs, size_t blocks) {
    const size_t bid = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (bid >= blocks) return;
    const float4 *p = reinterpret_cast<const float4 *>(x + bid * 16);
    float v[16];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const float4 t = __ldg(p + i);
        v[4 * i] = t.x; v[4 * i + 1] = t.y; v[4 * i + 2] = t.z; v[4 * i + 3] = t.w;
    }
    if (GLU) {
        const float4 *pu = reinterpret_cast<const float4 *>(up + bid * 16);
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const float4 t = __ldg(pu + i);
            v[4 * i] = fllm_glu(v[4 * i], t.x, mode);
            v[4 * i + 1] = fllm_glu(v[4 * i + 1], t.y, mode);
            v[4 * i + 2] = fllm_glu(v[4 * i + 2], t.z, mode);
            v[4 * i + 3] = fllm_glu(v[4 * i + 3], t.w, mode);
        }
    }
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; i++) amax = fmaxf(amax, fabsf(v[i]));
    const uint32_t sb = f_to_ue4m3(amax * (1.f / 6.f));
    const float sd = ue4m3_to_f(sb);
    const float inv = sd > 0.f ? 1.f / sd : 0.f;
    uint32_t words[2];
#pragma unroll
    for (int h = 0; h < 2; h++) {
        uint32_t w = 0;
#pragma unroll
        for (int q = 0; q < 8; q++) {
            const float t = v[8 * h + q] * inv;
            const uint32_t nib = e2m1_idx(fabsf(t)) | (t < 0.f ? 8u : 0u);
            w |= nib << (4 * q);
        }
        words[h] = w;
    }
    // inside a step of 64 values (8 words) the operand registers of one lane are adjacent: word j goes to
    // 2 * j (j < 4, register b0) or 2 * (j - 4) + 1 (register b1)
#pragma unroll
    for (int h = 0; h < 2; h++) {
        const size_t j = (bid * 2 + h) & 7, base = (bid * 2 + h) - j;
        xq[base + (j < 4 ? 2 * j : 2 * (j - 4) + 1)] = words[h];
    }
    xs[bid] = (uint8_t)sb;
}

#ifndef FP4_BM
#define FP4_BM 128
#define FP4_BN 128
#define FP4_WM 4
#define FP4_WN 2
#define FP4_ST 2
#define FP4_SBK 4
#endif

template <int BM, int BN, int WM, int WN, int STAGES, int SBK>
__global__ void __launch_bounds__(WM * WN * 32)
nvfp4_gemm_kernel(const uint8_t *__restrict__ W, const uint8_t *__restrict__ Xq, const uint8_t *__restrict__ Xs,
                  float *__restrict__ Y, const float *__restrict__ bias, float gscale, int rows, int cols, int batch) {
#ifdef FLLM_FP4_MMA
    constexpr int NT = WM * WN * 32;
    constexpr int MI = BM / WM / 16;
    constexpr int NI = BN / WN / 8;
    constexpr int SA = SBK * 36;           // bytes per row / token and stage (SBK GGUF blocks of 64 values)
    constexpr int CG = SA % 16 == 0 ? 16 : 8;  // cp.async granularity
    constexpr int NCH = SA / CG;           // copies per row / token
    constexpr int NIB = SBK * 32;          // nibble bytes of a token row inside the stage
    constexpr int SCP = SBK * 4 / CG;      // copies holding the weight scales of a stage
    constexpr int SP = SA + 16;  // activation row stride; 160 bytes = 40 words spreads the lanes over all banks
    // Weights of a stage: BM rows of SBK * 32 nibble bytes (the row stride is 4 mod 32 words, so the 32-bit fragment loads
    // of the eight rows of a warp hit distinct banks), followed by the SBK scale atoms (512 bytes each: 32 slots of 16 bytes,
    // slot a holds rows a, a + 32, a + 64, a + 96 with four scales each) copied as they are.
    static_assert(BM == 64 || BM == 128, "scale atoms are copied per 64 or 128 rows");
    constexpr int SPA = SBK * 32 + 16;
    constexpr int SCB = BM == 128 ? 16 : 8;  // bytes of a slot this block needs: all rows of the slot or two of four
    constexpr int A_STAGE = BM * SPA + SBK * 512;
    static_assert(SBK == 4 || SBK == 8, "stages of 256 or 512 values");
    extern __shared__ __align__(16) unsigned char smem[];
    unsigned char *As = smem;
    unsigned char *Bs = smem + (size_t)STAGES * A_STAGE;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int wm = warp % WM, wn = warp / WM;
    const int g = lane >> 2, t4 = lane & 3;
    const int m0 = blockIdx.x * BM, n0 = blockIdx.y * BN;
    const int nsb_all = cols / (64 * SBK);
    // weights: [rows][cols / 2] packed E2M1 in natural order, then the block scales in the swizzled layout of CUTLASS
    // block-scaled GEMMs (see nvfp4_device_layout in backend/cuda.rs; sf_index() below)
    const uint8_t *Wsf = W + (((size_t)rows * (cols >> 1) + 255) & ~size_t(255));
    const size_t sf_row_tile = (size_t)(cols >> 6) * 512;
    const int splits = gridDim.z;
    const int per = (nsb_all + splits - 1) / splits;
    const int sb0 = blockIdx.z * per;
    const int nsb = max(0, min(nsb_all, sb0 + per) - sb0);

    auto copy = [](void *dst, const void *src, bool ok) {
        if (CG == 16) cp_async16(dst, src, ok);
        else cp_async8(dst, src, ok);
    };
    auto issue = [&](int sb, int stage) {
        constexpr int NP = SBK * 2;  // 16-byte copies of the nibbles per row
        unsigned char *stage_base = As + (size_t)stage * A_STAGE;
        for (int c = tid; c < BM * NP; c += NT) {
            const int r = c / NP, part = c % NP;
            const int row = m0 + r;
            const bool ok = row < rows;
            cp_async16(stage_base + (size_t)r * SPA + part * 16, W + (size_t)(ok ? row : 0) * (cols >> 1) + (size_t)sb * (SBK * 32) + part * 16, ok);
        }
        for (int c = tid; c < SBK * 32; c += NT) {  // slot `a` of atom `kt` (= 64-value step) of the 128-row group of this block
            const int kt = c / 32, a = c % 32;
            const size_t off = (size_t)(m0 >> 7) * sf_row_tile + (size_t)(sb * SBK + kt) * 512 + a * 16 + (SCB == 16 ? 0 : ((m0 >> 5) & 3) * 4);
            unsigned char *dst = stage_base + BM * SPA + kt * 512 + a * 16 + (SCB == 16 ? 0 : ((m0 >> 5) & 3) * 4);
            if (SCB == 16) cp_async16(dst, Wsf + off, true);
            else cp_async8(dst, Wsf + off, true);
        }
        for (int c = tid; c < BN * NCH; c += NT) {
            const int n = c / NCH, part = c % NCH;
            const int tok = n0 + n;
            const bool ok = tok < batch;
            const uint8_t *src = part * CG < NIB ? Xq + (size_t)(ok ? tok : 0) * (cols >> 1) + (size_t)sb * NIB + part * CG
                                                 : Xs + (size_t)(ok ? tok : 0) * (cols >> 4) + (size_t)sb * (SBK * 4) + (part * CG - NIB);
            copy(Bs + ((size_t)stage * BN + n) * SP + part * CG, src, ok);
        }
    };

    float acc[MI][NI][4];
#pragma unroll
    for (int a = 0; a < MI; a++)
#pragma unroll
        for (int b = 0; b < NI; b++)
#pragma unroll
            for (int c = 0; c < 4; c++) acc[a][b][c] = 0.f;

#pragma unroll
    for (int s = 0; s < STAGES - 1; s++) {
        if (s < nsb) issue(sb0 + s, s);
        cp_async_commit();
    }

    for (int sbi = 0; sbi < nsb; sbi++) {
        cp_async_wait<STAGES - 2>();
        __syncthreads();
        if (sbi + STAGES - 1 < nsb) issue(sb0 + sbi + STAGES - 1, (sbi + STAGES - 1) % STAGES);
        cp_async_commit();

        const int stage = sbi % STAGES;
        const unsigned char *a_stage = As + (size_t)stage * A_STAGE;
        const unsigned char *b_stage = Bs + (size_t)stage * BN * SP;
    
#pragma unroll
        for (int st = 0; st < SBK; st++) {  // 64 values per step
            uint32_t af[MI][4], sfa[MI];
#pragma unroll
            for (int mi = 0; mi < MI; mi++) {
                const int row0 = wm * (BM / WM) + mi * 16 + g;  // row of this lane inside the block; row0 + 8 is the second one
                const unsigned char *ap = a_stage + (size_t)row0 * SPA;
                const unsigned char *aq = ap + 8 * SPA;
                const unsigned char *atom = a_stage + BM * SPA + st * 512;
                const int n0 = m0 + row0, n1 = n0 + 8;
                const uint32_t sc0 = *reinterpret_cast<const uint32_t *>(atom + (n0 & 31) * 16 + ((n0 >> 5) & 3) * 4);
                const uint32_t sc1 = *reinterpret_cast<const uint32_t *>(atom + (n1 & 31) * 16 + ((n1 >> 5) & 3) * 4);
                sfa[mi] = t4 == 1 ? sc1 : sc0;
                af[mi][0] = *reinterpret_cast<const uint32_t *>(ap + st * 32 + 4 * t4);
                af[mi][1] = *reinterpret_cast<const uint32_t *>(aq + st * 32 + 4 * t4);
                af[mi][2] = *reinterpret_cast<const uint32_t *>(ap + st * 32 + 16 + 4 * t4);
                af[mi][3] = *reinterpret_cast<const uint32_t *>(aq + st * 32 + 16 + 4 * t4);
            }
#pragma unroll
            for (int ni = 0; ni < NI; ni++) {
                const int n = wn * (BN / WN) + ni * 8 + g;
                const unsigned char *bp = b_stage + (size_t)n * SP;
                const uint2 bb = *reinterpret_cast<const uint2 *>(bp + st * 32 + 8 * t4);
                const uint32_t b0 = bb.x, b1 = bb.y;
                const uint32_t sfb = *reinterpret_cast<const uint32_t *>(bp + NIB + st * 4);
#pragma unroll
                for (int mi = 0; mi < MI; mi++) {
                    asm volatile(
                        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
                        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3},{%10},{%11,%12},{%13},{%14,%15};\n"
                        : "+f"(acc[mi][ni][0]), "+f"(acc[mi][ni][1]), "+f"(acc[mi][ni][2]), "+f"(acc[mi][ni][3])
                        : "r"(af[mi][0]), "r"(af[mi][1]), "r"(af[mi][2]), "r"(af[mi][3]), "r"(b0), "r"(b1),
                          "r"(sfa[mi]), "h"((uint16_t)0), "h"((uint16_t)0), "r"(sfb), "h"((uint16_t)0), "h"((uint16_t)0));
                }
            }
        }
    }
    cp_async_wait<0>();

#pragma unroll
    for (int mi = 0; mi < MI; mi++)
#pragma unroll
        for (int ni = 0; ni < NI; ni++)
#pragma unroll
            for (int c = 0; c < 4; c++) {
                const int m = m0 + wm * (BM / WM) + mi * 16 + g + (c >> 1) * 8;
                const int n = n0 + wn * (BN / WN) + ni * 8 + 2 * t4 + (c & 1);
                if (m < rows && n < batch) {
                    if (splits > 1) Y[(size_t)blockIdx.z * batch * rows + (size_t)n * rows + m] = acc[mi][ni][c];
                    else Y[(size_t)n * rows + m] = acc[mi][ni][c] * gscale + (bias ? bias[m] : 0.f);
                }
            }
#endif
}

__global__ void nvfp4_reduce_kernel(const float *__restrict__ part, float *__restrict__ Y, const float *__restrict__ bias,
                                    float gscale, int rows, size_t count, int splits) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < count; i += (size_t)gridDim.x * blockDim.x) {
        float s = 0.f;
        for (int z = 0; z < splits; z++) s += part[(size_t)z * count + i];
        Y[i] = s * gscale + (bias ? bias[i % rows] : 0.f);
    }
}

template <int BM, int BN, int WM, int WN, int STAGES, int SBK = 4>
static int fp4_launch(cudaStream_t s, const uint8_t *W, const uint8_t *xq, const uint8_t *xs, float *Y, const float *bias,
                      float gscale, int rows, int cols, int batch, float *part, size_t part_bytes) {
    constexpr size_t smem = (size_t)STAGES * ((size_t)(BM * (SBK * 32 + 16) + SBK * 512 + BN * (SBK * 36 + 16)));
    static_assert(smem <= 99 * 1024, "shared memory budget");
    static bool configured = false;
    if (!configured) {
        cudaFuncSetAttribute(nvfp4_gemm_kernel<BM, BN, WM, WN, STAGES, SBK>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
        configured = true;
    }
    const int row_blocks = (rows + BM - 1) / BM;
    int splits = 1;
    const int grid_blocks = row_blocks * ((batch + BN - 1) / BN);
    if (grid_blocks < 92) {
        splits = (92 + grid_blocks - 1) / grid_blocks;
        if (splits > 8) splits = 8;
        while (splits > 1 && ((cols / (64 * SBK)) / splits < 2 || (size_t)splits * batch * rows * 4 > part_bytes)) splits--;
    }
    const dim3 grid(row_blocks, (batch + BN - 1) / BN, splits), block(WM * WN * 32);
    nvfp4_gemm_kernel<BM, BN, WM, WN, STAGES, SBK><<<grid, block, smem, s>>>(W, xq, xs, splits > 1 ? part : Y, bias, gscale, rows, cols, batch);
    if (splits > 1) {
        const size_t count = (size_t)batch * rows;
        nvfp4_reduce_kernel<<<(unsigned)((count + 255) / 256), 256, 0, s>>>(part, Y, bias, gscale, rows, count, splits);
    }
    return 1;
}

#ifdef FLLM_HAVE_CUTLASS
int fllm_nvfp4_cutlass_try(cudaStream_t s, const void *W, const float *X, void *Y, const float *bias, float gscale, int rows,
                           int cols, int batch, int quantize, float acc_scale, int out_half);
int fllm_nvfp4_cutlass_check(const float *X, const float *Y, const float *bias, int rows, int cols, int batch, int mode);
void fllm_nvfp4_cutlass_invalidate();
#endif
// true while the scratch buffer holds activations quantized by the built-in kernels
static thread_local bool g_scratch_valid = false;

// Scratch layout: [xq: batch * cols / 2 bytes][xs: batch * cols / 16 bytes][split-K partials].
// Returns 1 when handled, 0 when the device or shape is not supported.
#ifdef FLLM_NO_FP4
int fllm_nvfp4_try(cudaStream_t, const void *, const float *, float *, const float *, float, int, int, int, void *, size_t, int) {
    return 0;
}
#else
int fllm_nvfp4_try(cudaStream_t s, const void *W, const float *X, float *Y, const float *bias, float gscale, int rows,
                   int cols, int batch, void *scratch, size_t scratch_bytes, int quantize) {
    static const bool supported = [] {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceProp p;
        if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return false;
        return p.major == 12 && fllm_cc() >= 120;
    }();
    if (!supported || batch < 1 || (cols & 255) != 0 || (reinterpret_cast<uintptr_t>(X) & 15) != 0 ||
        (reinterpret_cast<uintptr_t>(W) & 15) != 0)
        return 0;
#ifdef FLLM_HAVE_CUTLASS
    // prompt-sized batches: CUTLASS block-scaled GEMM (tile 128x128 / 256x128, TMA, warp specialization)
    if (batch > 64 && fllm_nvfp4_cutlass_try(s, W, X, Y, bias, gscale, rows, cols, batch, quantize, fllm_fp4_acc ? fllm_fp4_acc_scale : 0.f, fllm_fp4_f16out ? 1 : 0)) {
        g_scratch_valid = false;
        return 1;
    }
#endif
    // the built-in kernels overwrite Y with FP32 results of FP32 activations: a product that has to be added to Y, leave FP16 or read FP16
    // activations is declined (the caller checked fllm_nvfp4_cutlass_ok)
    if (fllm_fp4_acc || fllm_fp4_f16out || (fllm_glu_up && fllm_glu_half)) return 0;
    const size_t xq_bytes = ((size_t)batch * (cols >> 1) + 255) & ~size_t(255);
    const size_t xs_bytes = ((size_t)batch * (cols >> 4) + 255) & ~size_t(255);
    if (xq_bytes + xs_bytes > scratch_bytes) return 0;
    uint8_t *xq = static_cast<uint8_t *>(scratch);
    uint8_t *xs = xq + xq_bytes;
    float *part = reinterpret_cast<float *>(xs + xs_bytes);
    const size_t part_bytes = scratch_bytes - xq_bytes - xs_bytes;
    const size_t blocks = (size_t)batch * (cols >> 4);
    if (!quantize && !g_scratch_valid) quantize = 1;  // the last activations went through the CUTLASS path
    if (quantize) {
        g_scratch_valid = true;
#ifdef FLLM_HAVE_CUTLASS
        fllm_nvfp4_cutlass_invalidate();
#endif
        if (fllm_glu_up) nvfp4_quantize_kernel<true><<<(unsigned)((blocks + 127) / 128), 128, 0, s>>>(X, fllm_glu_up, fllm_glu_mode, reinterpret_cast<uint32_t *>(xq), xs, blocks);
        else nvfp4_quantize_kernel<false><<<(unsigned)((blocks + 127) / 128), 128, 0, s>>>(X, nullptr, 0, reinterpret_cast<uint32_t *>(xq), xs, blocks);
    }
    const uint8_t *w = static_cast<const uint8_t *>(W);
    if (batch <= 16) return fp4_launch<64, 16, 4, 1, 3>(s, w, xq, xs, Y, bias, gscale, rows, cols, batch, part, part_bytes);
    if (batch <= 32) return fp4_launch<64, 32, 4, 1, 2>(s, w, xq, xs, Y, bias, gscale, rows, cols, batch, part, part_bytes);
    if (batch <= 64) return fp4_launch<64, 64, 4, 1, 2>(s, w, xq, xs, Y, bias, gscale, rows, cols, batch, part, part_bytes);
    return fp4_launch<FP4_BM, FP4_BN, FP4_WM, FP4_WN, FP4_ST, FP4_SBK>(s, w, xq, xs, Y, bias, gscale, rows, cols, batch, part, part_bytes);
}
#endif

// Will fllm_nvfp4_try compute the product for this batch with the CUTLASS GEMM (mode: bit 0 = and add it to Y, bit 1 = leaving FP16)?
// Always 0 without CUTLASS or off sm_12x.
int fllm_nvfp4_cutlass_ok(const float *X, const float *Y, const float *bias, int rows, int cols, int batch, int mode) {
#if defined(FLLM_HAVE_CUTLASS) && !defined(FLLM_NO_FP4)
    static const bool supported = [] {
        int dev = 0;
        cudaDeviceProp p;
        return cudaGetDevice(&dev) == cudaSuccess && cudaGetDeviceProperties(&p, dev) == cudaSuccess && p.major == 12 && fllm_cc() >= 120;
    }();
    if (!supported || (cols & 255) != 0) return 0;
    return fllm_nvfp4_cutlass_check(X, Y, bias, rows, cols, batch, mode);
#else
    return 0;
#endif
}

// The fused norm kernels (fused.cu) quantize the activations of the next NVFP4 GEMM straight into the scratch that fllm_nvfp4_try reads
// (batches of up to 64 tokens; larger ones go to CUTLASS, which keeps its own copy).
bool fllm_nvfp4_prep_ok(int cols, int batch) {
#ifdef FLLM_NO_FP4
    (void)cols; (void)batch;
    return false;
#else
    static const bool supported = [] {
        int dev = 0;
        cudaGetDevice(&dev);
        cudaDeviceProp p;
        if (cudaGetDeviceProperties(&p, dev) != cudaSuccess) return false;
        return p.major == 12 && fllm_cc() >= 120;
    }();
    return supported && batch >= 1 && batch <= 64 && (cols & 255) == 0;
#endif
}
void fllm_nvfp4_scratch_filled() {
#ifndef FLLM_NO_FP4
    g_scratch_valid = true;
#ifdef FLLM_HAVE_CUTLASS
    fllm_nvfp4_cutlass_invalidate();
#endif
#endif
}
