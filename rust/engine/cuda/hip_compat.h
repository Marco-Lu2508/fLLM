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
// the communication layer (comm_kernels.cu): peer access and peer copies between the GPUs of one node
#define cudaDeviceCanAccessPeer hipDeviceCanAccessPeer
#define cudaDeviceEnablePeerAccess hipDeviceEnablePeerAccess
#define cudaDeviceDisablePeerAccess hipDeviceDisablePeerAccess
#define cudaErrorPeerAccessAlreadyEnabled hipErrorPeerAccessAlreadyEnabled
#define cudaErrorPeerAccessNotEnabled hipErrorPeerAccessNotEnabled
#define cudaMemcpyPeerAsync hipMemcpyPeerAsync
#define cudaErrorInvalidValue hipErrorInvalidValue
#define cudaDeviceGetP2PAttribute hipDeviceGetP2PAttribute
#define cudaDevP2PAttrPerformanceRank hipDevP2PAttrPerformanceRank
#define cudaDevP2PAttrNativeAtomicSupported hipDevP2PAttrNativeAtomicSupported

// ---- Architecture families, as the device pass of hipcc sees them (the host pass defines none of these macros) ----------------
// One code object exists per target of FLLM_HIP_ARCHS, and the HIP runtime hands a GPU the exact one, else the generic one of its family
// (gfx9-generic, gfx9-4-generic, gfx10-1-generic, gfx10-3-generic, gfx11-generic, gfx11-7-generic, gfx12-generic, gfx12-5-generic). The
// names below therefore test the specific targets AND their generic target; the table is hip/arch_table.csv (scripts/hip_arch_check.sh
// compiles every row and compares).
//   FLLM_HIP_CDNA    gfx908 / gfx90a / gfx94x / gfx950 / gfx9-4-generic: wave64, MFMA 16x16x16 on FP16 (hip/hip_mma.cuh)
//   FLLM_HIP_RDNA3   gfx11xx except gfx117x: wave32, WMMA with 16-element fragments (wmma_f32_16x16x16_f16_w32)
//   FLLM_HIP_RDNA4   gfx1200 / gfx1201 / gfx12-generic: wave32, WMMA with 8-element fragments (the gfx12 encoding)
//   FLLM_HIP_DOT2    packed FP16 dot product into FP32 (v_dot2c_f32_f16 on gfx9, v_dot2_f32_f16 on gfx11 / gfx12): the skinny GEMM kernels
//   FLLM_HIP_SDOT4   v_dot4_i32_i8 (gfx906, gfx908, gfx90a, gfx94x, gfx950, gfx1011, gfx1012, gfx103x and their generic targets)
//   FLLM_HIP_SUDOT4  v_dot4_i32_iu8 (gfx11, gfx12)
// Every other target (gfx900 .. gfx90c, gfx1010, gfx117x, gfx125x, whatever comes later) gets none of them: the kernels of hip/*.cu and the
// dispatchers of cuda/*.cu then run the plain FP32 / byte-loop code, which uses no matrix-core, dot or DPP-restricted instruction.
// gfx117x has WMMA (gfx12 encoding) but the layouts of hip_mma.cuh were never run on it, and gfx125x has a different matrix unit.
// The kernels of hip/*.cu use these to pick instructions; in the host pass their bodies are empty and the host-side dispatcher
// (fllm_hip_device() in hip/hip_device.cu, which asks the device-side probe what the loaded code object is) never launches them.
#if defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx9_4_generic__) ||     defined(__gfx950__)
#define FLLM_HIP_CDNA 1
#endif
#if defined(__gfx1170__) || defined(__gfx1171__) || defined(__gfx1172__) || defined(__gfx11_7_generic__)
#define FLLM_HIP_GFX117X 1
#endif
#if defined(__GFX11__) && !defined(FLLM_HIP_GFX117X)
#define FLLM_HIP_RDNA3 1
#endif
#if defined(__gfx1200__) || defined(__gfx1201__) || defined(__gfx12_generic__)
#define FLLM_HIP_RDNA4 1
#endif
#if defined(__gfx12_5_generic__) || defined(__gfx1250__) || defined(__gfx1251__) || defined(__GFX13__)
#define FLLM_HIP_GFX125X 1
#endif
#if defined(__gfx906__) || defined(__gfx908__) || defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) ||     defined(__gfx9_4_generic__) || defined(__gfx950__) || defined(__gfx1011__) || defined(__gfx1012__) ||     defined(__gfx1030__) || defined(__gfx1031__) || defined(__gfx1032__) || defined(__gfx1033__) || defined(__gfx1034__) || defined(__gfx1035__) ||     defined(__gfx1036__) || defined(__gfx10_3_generic__)
#define FLLM_HIP_SDOT4 1
#endif
#if (defined(__GFX11__) || defined(__GFX12__)) && !defined(__GFX13__)
#define FLLM_HIP_SUDOT4 1
#endif
#if defined(__gfx90a__) || defined(__gfx940__) || defined(__gfx941__) || defined(__gfx942__) || defined(__gfx9_4_generic__) || defined(__gfx950__) ||     ((defined(__GFX11__) || defined(__GFX12__)) && !defined(FLLM_HIP_GFX125X))
#define FLLM_HIP_DOT2 1
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

// 8-bit integer dot product with 32-bit accumulation: v_dot4_i32_iu8 (sudot4) on RDNA3 and newer, v_dot4_i32_i8 (sdot4) where the target has
// it (see FLLM_HIP_SDOT4), a byte loop elsewhere (Vega 10, RDNA1 gfx1010, gfx125x ...). Same result in every branch.
__device__ __forceinline__ int fllm_dp4a(int a, int b, int c) {
#if defined(FLLM_HIP_SUDOT4)
    return __builtin_amdgcn_sudot4(true, a, true, b, c, false);
#elif defined(FLLM_HIP_SDOT4)
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
