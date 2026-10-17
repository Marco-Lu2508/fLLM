// fLLM CUDA backend: dense FP16 / BF16 GEMM through cuBLASLt, loaded at run time.
//
// Hopper (wgmma + TMA) and datacenter Blackwell (tcgen05) kernels for the weights of non-quantized models come with the CUDA libraries; fLLM does
// not link them. libcublasLt is opened with dlopen (LoadLibrary on Windows) the first time a prompt-sized product asks for it; when the library is
// missing, a symbol is absent or any call fails, the function declines (-1) and the caller keeps the built-in tensor-core GEMM (tc_gemm.cu).
// Only the declarations of <cublasLt.h> are used at build time, nothing is linked.
//
//   Y[token][row] = sum_k X[token][k] * W[row][k] + bias[row]     (W: FP16 or BF16 [rows][cols], X and Y: FP32 in the engine's buffers)
//
// The FP32 activations are converted to the weight's 16-bit format in a scratch buffer, the product accumulates in FP32 and is written as FP32.
// Switch FLLM_CUBLASLT = auto | 0 | 1 (fllm_cublaslt_set_mode overrides it): auto uses cuBLASLt from compute capability 9.0 on for batches of 64 tokens
// and more, 1 uses it on every device with a tensor-core GEMM (batches of 16 and more), 0 never.
//
// Test level: compiled only; the self test group `hgemm` checks it on a device.
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <tuple>

#include <cuda_bf16.h>
#include <cublasLt.h>

#if defined(_WIN32)
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "common.cuh"
#include "fllm_cuda_internal.h"
#include "hopper_util.h"

void fllm_cublaslt_set_mode(int mode);
int fllm_cublaslt_gemm_try(cudaStream_t s, const void *W, int kind, const float *X, float *Y, const float *bias, int rows, int cols, int batch);

namespace {

struct Api {
    decltype(&cublasLtCreate) create = nullptr;
    decltype(&cublasLtMatmulDescCreate) desc_create = nullptr;
    decltype(&cublasLtMatmulDescDestroy) desc_destroy = nullptr;
    decltype(&cublasLtMatmulDescSetAttribute) desc_set = nullptr;
    decltype(&cublasLtMatrixLayoutCreate) layout_create = nullptr;
    decltype(&cublasLtMatrixLayoutDestroy) layout_destroy = nullptr;
    decltype(&cublasLtMatmulPreferenceCreate) pref_create = nullptr;
    decltype(&cublasLtMatmulPreferenceDestroy) pref_destroy = nullptr;
    decltype(&cublasLtMatmulPreferenceSetAttribute) pref_set = nullptr;
    decltype(&cublasLtMatmulAlgoGetHeuristic) heuristic = nullptr;
    decltype(&cublasLtMatmul) matmul = nullptr;
    bool ok = false;
};

void *open_library() {
    const char *forced = getenv("FLLM_CUBLASLT_LIB");
#if defined(_WIN32)
    const char *names[] = {forced, "cublasLt64_13.dll", "cublasLt64_12.dll", "cublasLt64_11.dll"};
    for (const char *n : names)
        if (n && *n)
            if (void *h = (void *)LoadLibraryA(n)) return h;
#else
    const char *names[] = {forced, "libcublasLt.so.13", "libcublasLt.so.12", "libcublasLt.so.11", "libcublasLt.so"};
    for (const char *n : names)
        if (n && *n)
            if (void *h = dlopen(n, RTLD_NOW | RTLD_LOCAL)) return h;
#endif
    return nullptr;
}

template <class F>
bool load(void *lib, const char *name, F &out) {
#if defined(_WIN32)
    out = reinterpret_cast<F>(GetProcAddress((HMODULE)lib, name));
#else
    out = reinterpret_cast<F>(dlsym(lib, name));
#endif
    return out != nullptr;
}

const Api &api() {
    static const Api a = [] {
        Api r;
        void *lib = open_library();
        if (!lib) return r;
        r.ok = load(lib, "cublasLtCreate", r.create) && load(lib, "cublasLtMatmulDescCreate", r.desc_create) &&
               load(lib, "cublasLtMatmulDescDestroy", r.desc_destroy) && load(lib, "cublasLtMatmulDescSetAttribute", r.desc_set) &&
               load(lib, "cublasLtMatrixLayoutCreate", r.layout_create) && load(lib, "cublasLtMatrixLayoutDestroy", r.layout_destroy) &&
               load(lib, "cublasLtMatmulPreferenceCreate", r.pref_create) && load(lib, "cublasLtMatmulPreferenceDestroy", r.pref_destroy) &&
               load(lib, "cublasLtMatmulPreferenceSetAttribute", r.pref_set) && load(lib, "cublasLtMatmulAlgoGetHeuristic", r.heuristic) &&
               load(lib, "cublasLtMatmul", r.matmul);
        return r;
    }();
    return a;
}

constexpr size_t kWorkspace = (size_t)32 << 20;

struct Plan {
    cublasLtMatmulDesc_t desc = nullptr;
    cublasLtMatrixLayout_t a = nullptr, b = nullptr, c = nullptr;
    cublasLtMatmulAlgo_t algo;
    size_t workspace = 0;
};

// Per thread and device: the handle, the workspace, the converted activations, and the plan of every product shape.
struct Context {
    cublasLtHandle_t handle = nullptr;
    void *workspace = nullptr;
    void *x16 = nullptr;
    size_t x16_bytes = 0;
    std::map<std::tuple<int, int, int, int>, Plan> plans;  // (rows, cols, batch, kind)
};

__global__ void to_half_kernel(const float *__restrict__ x, __half *__restrict__ y, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) y[i] = __float2half_rn(x[i]);
}
__global__ void to_bf16_kernel(const float *__restrict__ x, __nv_bfloat16 *__restrict__ y, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) y[i] = __float2bfloat16_rn(x[i]);
}
__global__ void add_bias_kernel(float *__restrict__ y, const float *__restrict__ bias, size_t n, int rows) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) y[i] += bias[i % rows];
}

std::atomic_int g_mode{-2};
std::atomic_bool g_failed{false};  // a plan or call failed once: stay with the built-in GEMM

int mode() {
    const int forced = g_mode.load();
    if (forced >= 0) return forced;
    // off unless asked for (not run on a device yet); auto = from compute capability 9.0 for prompts
    static const int env = [] {
        const char *v = getenv("FLLM_CUBLASLT");
        return v == nullptr || *v == 0 ? 0 : fllm_env_tri("FLLM_CUBLASLT");
    }();
    return env;
}

bool make_plan(const Api &api, Context &ctx, Plan &p, int rows, int cols, int batch, cudaDataType in) {
    if (api.desc_create(&p.desc, CUBLAS_COMPUTE_32F, CUDA_R_32F) != CUBLAS_STATUS_SUCCESS) return false;
    const cublasOperation_t trans = CUBLAS_OP_T, none = CUBLAS_OP_N;
    if (api.desc_set(p.desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans, sizeof trans) != CUBLAS_STATUS_SUCCESS ||
        api.desc_set(p.desc, CUBLASLT_MATMUL_DESC_TRANSB, &none, sizeof none) != CUBLAS_STATUS_SUCCESS)
        return false;
    // column-major view of the row-major operands: A = W^T is cols x rows (ld cols), B = X^T is cols x batch (ld cols), D = Y^T is rows x batch (ld rows)
    if (api.layout_create(&p.a, in, cols, rows, cols) != CUBLAS_STATUS_SUCCESS || api.layout_create(&p.b, in, cols, batch, cols) != CUBLAS_STATUS_SUCCESS ||
        api.layout_create(&p.c, CUDA_R_32F, rows, batch, rows) != CUBLAS_STATUS_SUCCESS)
        return false;
    cublasLtMatmulPreference_t pref = nullptr;
    if (api.pref_create(&pref) != CUBLAS_STATUS_SUCCESS) return false;
    const uint64_t ws = kWorkspace;
    bool ok = api.pref_set(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws) == CUBLAS_STATUS_SUCCESS;
    cublasLtMatmulHeuristicResult_t result;
    int found = 0;
    ok = ok && api.heuristic(ctx.handle, p.desc, p.a, p.b, p.c, p.c, pref, 1, &result, &found) == CUBLAS_STATUS_SUCCESS && found > 0 &&
         result.state == CUBLAS_STATUS_SUCCESS;
    api.pref_destroy(pref);
    if (!ok) return false;
    p.algo = result.algo;
    p.workspace = result.workspaceSize;
    return true;
}

}  // namespace

void fllm_cublaslt_set_mode(int mode_value) { g_mode.store(mode_value < -1 || mode_value > 1 ? -1 : mode_value); }

// Returns 0 when the product was computed, -1 when the caller has to use another kernel, > 0 on a launch error.
int fllm_cublaslt_gemm_try(cudaStream_t s, const void *W, int kind, const float *X, float *Y, const float *bias, int rows, int cols, int batch) {
    const int m = mode();
    if (m == 0 || g_failed.load() || (kind != KIND_F16 && kind != KIND_BF16) || fllm_exact_math() || fllm_cc() < 80) return -1;
    if (m < 0 && (fllm_cc() < 90 || batch < 64)) return -1;
    if (batch < 16 || rows <= 0 || cols <= 0 || (cols & 7) != 0 || (rows & 3) != 0) return -1;
    if ((reinterpret_cast<uintptr_t>(W) & 15) != 0 || (reinterpret_cast<uintptr_t>(X) & 15) != 0 || (reinterpret_cast<uintptr_t>(Y) & 15) != 0) return -1;
    const Api &lt = api();
    if (!lt.ok) {
        g_failed = true;
        return -1;
    }
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) return -1;
    static thread_local std::map<int, Context> contexts;
    Context &ctx = contexts[dev];
    cudaStreamCaptureStatus cap = cudaStreamCaptureStatusNone;
    const bool capturing = cudaStreamIsCapturing(s, &cap) != cudaSuccess || cap != cudaStreamCaptureStatusNone;
    cudaGetLastError();
    const size_t x_bytes = (size_t)batch * cols * 2;
    if (!ctx.handle || !ctx.workspace || x_bytes > ctx.x16_bytes) {
        if (capturing) return -1;  // allocation and library start-up are not allowed inside a graph capture; the first uncaptured call does them
        if (!ctx.handle && lt.create(&ctx.handle) != CUBLAS_STATUS_SUCCESS) {
            g_failed = true;
            return -1;
        }
        if (!ctx.workspace && cudaMalloc(&ctx.workspace, kWorkspace) != cudaSuccess) {
            cudaGetLastError();
            return -1;
        }
        if (x_bytes > ctx.x16_bytes) {
            // not freed: a captured graph may hold the old pointer
            void *p = nullptr;
            const size_t want = x_bytes > 2 * ctx.x16_bytes ? x_bytes : 2 * ctx.x16_bytes;
            if (cudaMalloc(&p, want) != cudaSuccess) {
                cudaGetLastError();
                return -1;
            }
            ctx.x16 = p;
            ctx.x16_bytes = want;
        }
    }
    const auto key = std::make_tuple(rows, cols, batch, kind);
    auto it = ctx.plans.find(key);
    if (it == ctx.plans.end()) {
        if (capturing) return -1;
        Plan p;
        if (!make_plan(lt, ctx, p, rows, cols, batch, kind == KIND_F16 ? CUDA_R_16F : CUDA_R_16BF)) {
            // this shape has no algorithm: remember that with an empty plan
            p.desc = nullptr;
        }
        it = ctx.plans.emplace(key, p).first;
    }
    const Plan &p = it->second;
    if (!p.desc) return -1;
    const size_t n = (size_t)batch * cols;
    const unsigned grid = (unsigned)((n + 255) / 256 > 4096 ? 4096 : (n + 255) / 256);
    if (kind == KIND_F16) to_half_kernel<<<grid, 256, 0, s>>>(X, static_cast<__half *>(ctx.x16), n);
    else to_bf16_kernel<<<grid, 256, 0, s>>>(X, static_cast<__nv_bfloat16 *>(ctx.x16), n);
    const float alpha = 1.f, beta = 0.f;
    const cublasStatus_t st = lt.matmul(ctx.handle, p.desc, &alpha, W, p.a, ctx.x16, p.b, &beta, Y, p.c, Y, p.c, &p.algo,
                                        p.workspace ? ctx.workspace : nullptr, p.workspace, s);
    if (st != CUBLAS_STATUS_SUCCESS) {
        cudaGetLastError();
        g_failed = true;
        return -1;
    }
    if (bias) {
        const size_t total = (size_t)batch * rows;
        add_bias_kernel<<<(unsigned)((total + 255) / 256 > 4096 ? 4096 : (total + 255) / 256), 256, 0, s>>>(Y, bias, total, rows);
    }
    return fllm_check_launch("cublasLt gemm");
}
