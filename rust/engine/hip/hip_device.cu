// fLLM HIP backend: properties of the current AMD device, the device-side probe of the loaded code object, and run-time switches for the
// kernels in this directory.
//
// A binary carries one code object per target of FLLM_HIP_ARCHS (build.rs) and the HIP runtime picks the one that fits the GPU: the exact gfx
// target, else the generic target of its family (gfx10-3-generic ...), else (HSA_OVERRIDE_GFX_VERSION set by the user or by fLLM, see
// src/backend/hip_arch.rs) the code of the chip it was told to be. Which instructions the loaded code may use is therefore decided by the
// device pass, not by the name of the GPU. This file asks the device: `fllm_hip_selfcheck` launches small kernels that
//   * report which kernel set the code object has (FLLM_HIP_CDNA / _RDNA3 / _RDNA4 ... of cuda/hip_compat.h) and its wave size,
//   * run a small FP32 matrix product, the integer dot product (`fllm_dp4a`, signed bytes) and, where the code object has them, one 16x16x16
//     matrix-core step, and compare with the host.
// A device whose code cannot run at all is reported "present but not usable" (Rust side); one whose matrix-core step gives a wrong result
// keeps working with the plain kernels (`arch` OTHER). The name table (hip_arch.h) only decides what to distrust when the loaded code does
// not match the chip. FLLM_HIP_ARCH_OVERRIDE=gfxNNNN makes the host treat the device as that chip for the kernel selection.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>

#include "../cuda/common.cuh"
#include "fllm_hip.h"
#include "hip_arch.h"
#include "hip_mma.cuh"

namespace {

bool starts_with(const std::string &s, const char *prefix) { return s.rfind(prefix, 0) == 0; }

constexpr unsigned PROBE_MAGIC = 0x464c4c4du;  // "FLLM"
constexpr int PROBE_WORDS = 8;
constexpr int DP_TESTS = 16;

// ---- the probe kernels ---------------------------------------------------------------------------------------------------------

// out[0] = magic, [1] = kernel set | dot4 kind << 4 | dot2 << 6 | gfx950 << 7, [2] = warpSize, [3] = sum of the lane ids (shuffle reduction)
__global__ void probe_kernel(unsigned *out) {
    unsigned v = threadIdx.x & (warpSize - 1);
    for (int o = warpSize / 2; o > 0; o >>= 1) v += __shfl_xor((int)v, o);
    if (threadIdx.x == 0) {
        unsigned set = 0, dot4 = 0, dot2 = 0, big_lds = 0;
#if defined(FLLM_HIP_CDNA)
        set = FLLM_HIP_ARCH_CDNA;
#elif defined(FLLM_HIP_RDNA3)
        set = FLLM_HIP_ARCH_RDNA3;
#elif defined(FLLM_HIP_RDNA4)
        set = FLLM_HIP_ARCH_RDNA4;
#endif
#if defined(FLLM_HIP_SUDOT4)
        dot4 = 2;
#elif defined(FLLM_HIP_SDOT4)
        dot4 = 1;
#endif
#if defined(FLLM_HIP_DOT2)
        dot2 = 1;
#endif
#if defined(__gfx950__)
        big_lds = 1;
#endif
        out[0] = PROBE_MAGIC;
        out[1] = set | (dot4 << 4) | (dot2 << 6) | (big_lds << 7);
        out[2] = (unsigned)warpSize;
        out[3] = v;
    }
}

// plain FP32 product C[m][n] = sum_k A[m][k] * B[n][k] (16 x 16 x 16) and the integer dot product on the host's test vectors
__global__ void plain_kernel(const float *a, const float *b, float *c, const int *dp_in, int *dp_out) {
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        const int m = i / 16, n = i % 16;
        float acc = 0.f;
        for (int k = 0; k < 16; k++) acc += a[m * 16 + k] * b[n * 16 + k];
        c[i] = acc;
    }
    if (threadIdx.x < DP_TESTS) dp_out[threadIdx.x] = fllm_dp4a(dp_in[3 * threadIdx.x], dp_in[3 * threadIdx.x + 1], dp_in[3 * threadIdx.x + 2]);
}

// one matrix-core step of the code object's kind: D = A * B^T on 16 x 16 x 16, FP16 inputs, FP32 accumulation
__global__ void mma_kernel(const float *a, const float *b, float *d) {
#if defined(FLLM_HIP_MMA)
    __shared__ __align__(32) __half As[256];
    __shared__ __align__(32) __half Bs[256];
    for (int i = threadIdx.x; i < 256; i += blockDim.x) {
        As[i] = __float2half(a[i]);
        Bs[i] = __float2half(b[i]);
    }
    __syncthreads();
    if (threadIdx.x < fllm_hip::MMA_WAVE) {
        const int lane = threadIdx.x, fr = lane & 15, k0 = fllm_hip::mma_k0(lane);
        const fllm_hip::MmaFrag af = fllm_hip::mma_load(As + fr * 16 + k0), bf = fllm_hip::mma_load(Bs + fr * 16 + k0);
        fllm_hip::MmaAcc acc = {};
        acc = fllm_hip::mma16(af, bf, acc);
        for (int i = 0; i < fllm_hip::MMA_ACC; i++) d[fllm_hip::mma_row(lane, i) * 16 + fr] = acc[i];
    }
#else
    (void)a; (void)b; (void)d;
#endif
}

// ---- host side of the check ----------------------------------------------------------------------------------------------------

struct Check {
    bool usable = false;     // the code object loads and runs the plain kernel correctly
    char reason[240] = {0};  // why not
    int set = 0;             // kernel set of the loaded code (FllmHipArch)
    int wave = 0;
    int dot4 = 0;            // 0 none, 1 v_dot4_i32_i8, 2 v_dot4_i32_iu8
    bool dot2 = false;
    bool big_lds = false;
    int mma = -1;            // -1: the code has no matrix-core kernels, 0: step gave a wrong result / failed, 1: right
};

int ref_dp4a(int a, int b, int c) {
    int r = c;
    for (int i = 0; i < 4; i++) r += (int)(int8_t)(a >> (8 * i)) * (int)(int8_t)(b >> (8 * i));
    return r;
}

void set_reason(Check &c, const char *fmt, const char *detail) { snprintf(c.reason, sizeof c.reason, fmt, detail); }

Check run_check(int dev) {
    Check c;
    int prev = 0;
    hipGetDevice(&prev);
    hipError_t e = hipSetDevice(dev);
    if (e != hipSuccess) {
        set_reason(c, "cannot select the device: %s", hipGetErrorString(e));
        return c;
    }
    unsigned *d_probe = nullptr;
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr, *d_d = nullptr;
    int *d_dpi = nullptr, *d_dpo = nullptr;
    const auto done = [&]() {
        hipFree(d_probe); hipFree(d_a); hipFree(d_b); hipFree(d_c); hipFree(d_d); hipFree(d_dpi); hipFree(d_dpo);
        hipGetLastError();
        hipSetDevice(prev);
    };
    if (hipMalloc(&d_probe, 64) != hipSuccess || hipMalloc(&d_a, 1024) != hipSuccess || hipMalloc(&d_b, 1024) != hipSuccess ||
        hipMalloc(&d_c, 1024) != hipSuccess || hipMalloc(&d_d, 1024) != hipSuccess || hipMalloc(&d_dpi, 3 * DP_TESTS * 4) != hipSuccess ||
        hipMalloc(&d_dpo, DP_TESTS * 4) != hipSuccess) {
        set_reason(c, "cannot allocate a few KiB on the device: %s", hipGetErrorString(hipGetLastError()));
        done();
        return c;
    }
    // 1. which code object was loaded, and does a kernel run at all
    unsigned probe[PROBE_WORDS] = {0};
    hipMemset(d_probe, 0, 64);
    probe_kernel<<<1, 64>>>(d_probe);
    e = hipGetLastError();
    if (e == hipSuccess) e = hipDeviceSynchronize();
    if (e == hipSuccess) e = hipMemcpy(probe, d_probe, sizeof probe, hipMemcpyDeviceToHost);
    if (e != hipSuccess) {
        set_reason(c, "no code object of this binary runs on the device (%s)", hipGetErrorString(e));
        done();
        return c;
    }
    if (probe[0] != PROBE_MAGIC || probe[2] == 0 || probe[3] != probe[2] * (probe[2] - 1) / 2) {
        snprintf(c.reason, sizeof c.reason, "the first kernel returned wrong data (wave size %u, lane sum %u): the code object does not match the device", probe[2], probe[3]);
        done();
        return c;
    }
    c.set = (int)(probe[1] & 15);
    c.dot4 = (int)((probe[1] >> 4) & 3);
    c.dot2 = ((probe[1] >> 6) & 1) != 0;
    c.big_lds = ((probe[1] >> 7) & 1) != 0;
    c.wave = (int)probe[2];
    // 2. FP32 matrix product and the integer dot product against the host
    float ha[256], hb[256], hc[256], hd[256];
    for (int m = 0; m < 16; m++)
        for (int k = 0; k < 16; k++) {
            ha[m * 16 + k] = (float)(((m * 3 + k * 5) % 7) - 3);
            hb[m * 16 + k] = (float)(((m * 5 + k * 3) % 9) - 4);
        }
    int dpi[3 * DP_TESTS], dpo[DP_TESTS];
    unsigned seed = 0x9e3779b9u;
    for (int i = 0; i < 3 * DP_TESTS; i++) {
        seed = seed * 1664525u + 1013904223u;
        dpi[i] = (int)seed;
        if (i % 3 == 2) dpi[i] = (int)(seed >> 20) - 2048;  // the accumulator: small, either sign
    }
    hipMemcpy(d_a, ha, sizeof ha, hipMemcpyHostToDevice);
    hipMemcpy(d_b, hb, sizeof hb, hipMemcpyHostToDevice);
    hipMemcpy(d_dpi, dpi, sizeof dpi, hipMemcpyHostToDevice);
    plain_kernel<<<1, 64>>>(d_a, d_b, d_c, d_dpi, d_dpo);
    e = hipGetLastError();
    if (e == hipSuccess) e = hipDeviceSynchronize();
    if (e != hipSuccess) {
        set_reason(c, "the plain FP32 / integer-dot kernel fails on the device (%s): the code object uses an instruction this chip lacks", hipGetErrorString(e));
        done();
        return c;
    }
    hipMemcpy(hc, d_c, sizeof hc, hipMemcpyDeviceToHost);
    hipMemcpy(dpo, d_dpo, sizeof dpo, hipMemcpyDeviceToHost);
    for (int m = 0; m < 16; m++)
        for (int n = 0; n < 16; n++) {
            float want = 0.f;
            for (int k = 0; k < 16; k++) want += ha[m * 16 + k] * hb[n * 16 + k];
            hd[m * 16 + n] = want;
            if (hc[m * 16 + n] != want) {
                snprintf(c.reason, sizeof c.reason, "a 16x16x16 FP32 product is wrong at [%d][%d]: %g instead of %g", m, n, hc[m * 16 + n], want);
                done();
                return c;
            }
        }
    for (int i = 0; i < DP_TESTS; i++)
        if (dpo[i] != ref_dp4a(dpi[3 * i], dpi[3 * i + 1], dpi[3 * i + 2])) {
            snprintf(c.reason, sizeof c.reason, "the 8-bit integer dot product is wrong (test %d: %d instead of %d)", i, dpo[i],
                     ref_dp4a(dpi[3 * i], dpi[3 * i + 1], dpi[3 * i + 2]));
            done();
            return c;
        }
    c.usable = true;
    // 3. the matrix-core step of this code object (a wrong layout or a missing unit must not produce wrong text: the plain kernels run instead)
    if (c.set != FLLM_HIP_ARCH_OTHER) {
        hipMemset(d_d, 0, 1024);
        mma_kernel<<<1, 64>>>(d_a, d_b, d_d);
        e = hipGetLastError();
        if (e == hipSuccess) e = hipDeviceSynchronize();
        float got[256];
        c.mma = 0;
        if (e == hipSuccess && hipMemcpy(got, d_d, sizeof got, hipMemcpyDeviceToHost) == hipSuccess) {
            c.mma = 1;
            for (int i = 0; i < 256; i++)
                if (got[i] != hd[i]) c.mma = 0;
        }
        if (e != hipSuccess) {
            // an illegal instruction may leave the context unusable: say so instead of guessing
            c.usable = false;
            set_reason(c, "the matrix-core step fails on the device (%s): the code object does not fit this chip", hipGetErrorString(e));
        }
    }
    done();
    return c;
}

Check &check_of(int dev) {
    static std::mutex mu;
    static Check cache[16];
    static bool have[16] = {false};
    static Check bad;
    if (dev < 0 || dev >= 16) {
        snprintf(bad.reason, sizeof bad.reason, "device ordinal %d is out of range", dev);
        return bad;
    }
    std::lock_guard<std::mutex> lock(mu);
    if (!have[dev]) {
        cache[dev] = run_check(dev);
        have[dev] = true;
    }
    return cache[dev];
}

FllmHipDevice query(int dev) {
    FllmHipDevice d{};
    d.arch = FLLM_HIP_ARCH_OTHER;
    d.wave = 64;
    d.lds_bytes = 64 * 1024;
    hipDeviceProp_t p;
    if (hipGetDeviceProperties(&p, dev) != hipSuccess) return d;
    snprintf(d.gcn, sizeof d.gcn, "%s", p.gcnArchName);
    d.cus = p.multiProcessorCount;
    if (p.sharedMemPerBlock > 0) d.lds_bytes = (int)p.sharedMemPerBlock;
    // the chip the host believes it drives: its name, or what FLLM_HIP_ARCH_OVERRIDE says
    const char *ov = getenv("FLLM_HIP_ARCH_OVERRIDE");
    const std::string seen = (ov && *ov) ? std::string(ov) : std::string(p.gcnArchName);
    const fllm_hip_arch::Info name = fllm_hip_arch::classify(seen);
    d.wave = name.wave > 0 ? name.wave : 64;
    d.gfx1151 = starts_with(fllm_hip_arch::base_name(seen), "gfx1151");
    // the code object that is loaded decides what can run (see the top of the file)
    const Check &c = check_of(dev);
    d.usable = c.usable;
    if (!c.usable) return d;
    d.wave = c.wave;
    d.dot2 = c.dot2;
    d.dot4 = c.dot4;
    d.big_lds = c.big_lds;
    int use = c.set;
    if (use != FLLM_HIP_ARCH_OTHER) {
        if (name.known && (int)name.use != use) {
            fprintf(stderr, "[fllm-rs] hip: %s runs code built for another chip (%s expected matrix-core set %d, the loaded code has %d): matrix cores off, plain kernels used\n",
                    p.gcnArchName, seen.c_str(), (int)name.use, use);
            use = FLLM_HIP_ARCH_OTHER;
        } else if (c.mma != 1) {
            fprintf(stderr, "[fllm-rs] hip: the matrix-core self-check gave a wrong result on %s: matrix cores off, plain kernels used\n", p.gcnArchName);
            use = FLLM_HIP_ARCH_OTHER;
        }
    }
    d.arch = use;
    return d;
}

}  // namespace

const FllmHipDevice &fllm_hip_device() {
    static std::mutex mu;
    static FllmHipDevice cache[16];
    static bool have[16] = {false};
    static FllmHipDevice fallback{};
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 16) return fallback;
    std::lock_guard<std::mutex> lock(mu);
    if (!have[dev]) {
        cache[dev] = query(dev);
        have[dev] = true;
    }
    return cache[dev];
}

bool fllm_hip_enabled(const char *name) {
    char var[96];
    snprintf(var, sizeof var, "FLLM_HIP_NO_%s", name);
    const char *v = getenv(var);
    return !(v && *v && strcmp(v, "0") != 0);
}

// Rust side (src/backend/cuda.rs, list_devices): 0 = the device runs this binary's code; otherwise `msg` says why not. `info` (6 ints): kernel
// set of the loaded code, wave size, dot kind, dot2, matrix-core step (-1 none, 0 wrong, 1 right), kernel set the host will use.
extern "C" int fllm_hip_selfcheck(int dev, char *msg, size_t msg_len, int *info) {
    const Check &c = check_of(dev);
    if (msg && msg_len) snprintf(msg, msg_len, "%s", c.usable ? "" : c.reason);
    if (info) {
        info[0] = c.set; info[1] = c.wave; info[2] = c.dot4; info[3] = c.dot2 ? 1 : 0; info[4] = c.mma;
        int prev = 0;
        hipGetDevice(&prev);
        info[5] = FLLM_HIP_ARCH_OTHER;
        if (c.usable && hipSetDevice(dev) == hipSuccess) {
            info[5] = fllm_hip_device().arch;
            hipSetDevice(prev);
        }
    }
    return c.usable ? 0 : 1;
}
