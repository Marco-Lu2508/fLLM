// fLLM CUDA backend: programmatic dependent launch (FLLM_PDL=1, sm_90 and newer).
//
// A kernel launched with the attribute `cudaLaunchAttributeProgrammaticStreamSerialization` may start its blocks while the kernel before it in the stream is
// still running, as soon as every block of that kernel has called `griddepcontrol.launch_dependents` (or ended); it must call `griddepcontrol.wait` before it reads
// anything the earlier kernel writes. The wait returns when the earlier grid has completed and its writes are visible, so a chain of such kernels still computes
// exactly what the plain stream order computes; what is saved is the launch latency between two small kernels (a few microseconds each, hundreds per step).
// A kernel that calls the wait but was launched without the attribute is not affected (the instruction does nothing then), and the earlier kernel need not be a
// PDL kernel (Marlin, CUTLASS, FlashInfer never trigger: the dependent then starts when they finish, like a normal launch).
// Only kernels that call `fllm_pdl_wait()` at their start may be launched through `fllm_launch_pdl`.
#pragma once
#include <atomic>
#include <cstdlib>
#include <cstring>

#include <cuda_runtime.h>

int fllm_cc();
int fllm_set_error(const char *fmt, ...);

__device__ __forceinline__ void fllm_pdl_wait() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.wait;" ::: "memory");
#endif
}
__device__ __forceinline__ void fllm_pdl_trigger() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
    asm volatile("griddepcontrol.launch_dependents;" :::);
#endif
}

// FLLM_PDL=0|1 (default 0); selftest-gpu flips it with fllm_pdl_force (negative: back to the environment)
inline std::atomic<int> &fllm_pdl_override() {
    static std::atomic<int> v{-1};
    return v;
}
inline bool fllm_pdl_enabled() {
#if !defined(CUDART_VERSION) || CUDART_VERSION < 12000
    return false;
#endif
    const int o = fllm_pdl_override().load(std::memory_order_relaxed);
    if (o >= 0) return o != 0 && fllm_cc() >= 90;
    static const bool env = [] {
        const char *v = getenv("FLLM_PDL");
        return v && (!strcmp(v, "1") || !strcmp(v, "on") || !strcmp(v, "true"));
    }();
    return env && fllm_cc() >= 90;
}

// Launch `kernel` on `s`; with FLLM_PDL on (and sm_90+) as a programmatic dependent of the previous kernel of the stream. Returns 0 or sets the error.
template <typename... KArgs, typename... Args>
inline int fllm_launch_pdl(void (*kernel)(KArgs...), dim3 grid, dim3 block, size_t smem, cudaStream_t s, Args... args) {
#if defined(CUDART_VERSION) && CUDART_VERSION >= 12000
    if (fllm_pdl_enabled()) {
        cudaLaunchConfig_t cfg = {};
        cfg.gridDim = grid;
        cfg.blockDim = block;
        cfg.dynamicSmemBytes = smem;
        cfg.stream = s;
        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attr[0].val.programmaticStreamSerializationAllowed = 1;
        cfg.attrs = attr;
        cfg.numAttrs = 1;
        const cudaError_t e = cudaLaunchKernelEx(&cfg, kernel, args...);
        if (e != cudaSuccess) return fllm_set_error("programmatic launch: %s", cudaGetErrorString(e));
        return 0;
    }
#endif
    kernel<<<grid, block, smem, s>>>(args...);
    return 0;
}
