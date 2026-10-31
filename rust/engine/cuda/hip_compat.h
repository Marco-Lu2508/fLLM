// fLLM GPU backend: AMD (HIP / ROCm) build of the CUDA sources.
//
// The kernels in this directory are written against the CUDA runtime API. For AMD GPUs the same sources are compiled with
// hipcc (build.rs, FLLM_HIP=1); this header maps the runtime calls and intrinsics they use onto HIP. The NVIDIA-specific
// tensor-core code (PTX mma / ldmatrix / cp.async, FP4, CUTLASS, FlashInfer) is not compiled for AMD: FLLM_TC is 0 and the
// dispatcher (which asks fllm_cc()) uses the portable kernels (integer dot-product GEMV, FP32 GEMM, plain attention).
#pragma once
#if defined(__HIPCC__)
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#define FLLM_HIP 1

#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t
#define cudaError_t hipError_t
#define cudaSuccess hipSuccess
#define cudaDeviceProp hipDeviceProp_t
#define cudaSetDevice hipSetDevice
#define cudaGetDevice hipGetDevice
#define cudaGetDeviceCount hipGetDeviceCount
#define cudaGetDeviceProperties hipGetDeviceProperties
#define cudaDeviceGetAttribute hipDeviceGetAttribute
#define cudaDevAttrMaxSharedMemoryPerBlockOptin hipDeviceAttributeMaxSharedMemoryPerBlock
#define cudaMalloc hipMalloc
#define cudaMallocHost hipHostMalloc
#define cudaFree hipFree
#define cudaMemcpy hipMemcpy
#define cudaMemcpyAsync hipMemcpyAsync
#define cudaMemset hipMemset
#define cudaMemsetAsync hipMemsetAsync
#define cudaMemGetInfo hipMemGetInfo
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost
#define cudaMemcpyDeviceToDevice hipMemcpyDeviceToDevice
#define cudaHostRegister hipHostRegister
#define cudaHostUnregister hipHostUnregister
#define cudaHostRegisterPortable hipHostRegisterPortable
#define cudaHostRegisterReadOnly hipHostRegisterReadOnly
#define cudaStreamCreateWithFlags hipStreamCreateWithFlags
#define cudaStreamNonBlocking hipStreamNonBlocking
#define cudaStreamDestroy hipStreamDestroy
#define cudaStreamSynchronize hipStreamSynchronize
#define cudaDeviceSynchronize hipDeviceSynchronize
#define cudaGetLastError hipGetLastError
#define cudaGetErrorString hipGetErrorString
// dynamic shared memory up to the LDS size needs no opt-in on AMD
#define cudaFuncSetAttribute(...) ((void)0)
#define cudaFuncAttributeMaxDynamicSharedMemorySize hipFuncAttributeMaxDynamicSharedMemorySize
#define cudaEventCreate hipEventCreate
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaEventDestroy hipEventDestroy
#define cudaEventDisableTiming hipEventDisableTiming
#define cudaEventCreateWithFlags hipEventCreateWithFlags
#define cudaStreamWaitEvent hipStreamWaitEvent
// graph capture of a decode step (api.cu, fllm_cuda_graph_*)
#define cudaGraph_t hipGraph_t
#define cudaGraphExec_t hipGraphExec_t
#define cudaStreamBeginCapture hipStreamBeginCapture
#define cudaStreamEndCapture hipStreamEndCapture
#define cudaStreamCaptureModeThreadLocal hipStreamCaptureModeThreadLocal
#define cudaGraphInstantiate hipGraphInstantiateWithFlags
#define cudaGraphLaunch hipGraphLaunch
#define cudaGraphDestroy hipGraphDestroy
#define cudaGraphExecDestroy hipGraphExecDestroy
#define cudaHostAlloc(p, n, f) hipHostMalloc(p, n, f)
#define cudaHostAllocDefault hipHostMallocDefault
#define cudaFreeHost hipHostFree
// api.cu (fllm_cuda_download) asks what a failing pointer is; hipPointerAttribute_t has the same `type` and `device` fields
#define cudaPointerAttributes hipPointerAttribute_t
#define cudaPointerGetAttributes hipPointerGetAttributes

// ---- Architecture families, as the device pass of hipcc sees them (the host pass defines none of these macros) ----------------
//   FLLM_HIP_CDNA    gfx908 / gfx90a / gfx94x / gfx950: wave64, MFMA matrix cores, v_dot2c / sdot4
//   FLLM_HIP_RDNA3   gfx11xx: wave32 (the default), WMMA with 16-element fragments, v_dot2 / sudot4
//   FLLM_HIP_RDNA4   gfx12xx: wave32, WMMA with 8-element fragments (gfx12 encoding), v_dot2 / sudot4
// The kernels of hip/*.cu use them to pick instructions; on every other architecture (and in the host pass) their bodies are empty
// and the host-side dispatcher (fllm_hip_arch() in hip/hip_common.cuh) never launches them.
#if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx950__)
#define FLLM_HIP_CDNA 1
#endif
#if defined(__GFX11__)
#define FLLM_HIP_RDNA3 1
#endif
#if defined(__GFX12__)
#define FLLM_HIP_RDNA4 1
#endif

// HIP wants 64-bit lane masks for the _sync variants; the unmasked forms are equivalent for the full-warp uses here
template <class T>
__device__ __forceinline__ T __shfl_xor_sync(unsigned int, T var, int lane_mask, int width = 32) {
    return __shfl_xor(var, lane_mask, width);
}
template <class T>
__device__ __forceinline__ T __shfl_down_sync(unsigned int, T var, unsigned int delta, int width = 32) {
    return __shfl_down(var, delta, width);
}
template <class T>
__device__ __forceinline__ T __shfl_sync(unsigned int, T var, int src_lane, int width = 32) {
    return __shfl(var, src_lane, width);
}

// 8-bit integer dot product with 32-bit accumulation: v_dot4_i32_i8 on CDNA2+ and RDNA2 (sdot4), v_dot4_i32_iu8 on RDNA3/4
// (sudot4); a byte loop elsewhere.
__device__ __forceinline__ int fllm_dp4a(int a, int b, int c) {
#if defined(__GFX11__) || defined(__GFX12__)
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#elif defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx942__) || defined(__gfx950__) || defined(__gfx1030__) ||     defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1033__) || defined(__gfx1034__) || defined(__gfx1035__) ||     defined(__gfx1036__)
    return __builtin_amdgcn_sdot4(a, b, c, false);
#else
    int r = c;
    for (int i = 0; i < 4; i++) r += (int)(int8_t)(a >> (8 * i)) * (int)(int8_t)(b >> (8 * i));
    return r;
#endif
}
#define __dp4a fllm_dp4a

// NVIDIA-only encodings of the compute capability are meaningless here
#define FLLM_NO_CUDA_ARCH 1
#endif
