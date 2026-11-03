// Device helpers of the multi-GPU communication layer (src/comm/): device selection, plain device memory, a stream per rank,
// synchronous host copies for the host-slice collectives, and peer-access queries for the topology report.
// NCCL (CUDA) and RCCL (HIP, same API) are not linked: they are loaded at run time (src/comm/nccl.rs). This file is built by nvcc and, with
// FLLM_HIP=1, by hipcc (hip_compat.h maps the runtime calls).
#if defined(__HIPCC__)
#include "hip_compat.h"
#else
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#endif
#include <stddef.h>
#include <stdint.h>

extern "C" {

static int fllm_comm_status(cudaError_t e) { return e == cudaSuccess ? 0 : (int)e; }

const char *fllm_comm_error_string(int code) { return cudaGetErrorString((cudaError_t)code); }

int fllm_comm_device_count(int *count) { return fllm_comm_status(cudaGetDeviceCount(count)); }

int fllm_comm_set_device(int dev) { return fllm_comm_status(cudaSetDevice(dev)); }

int fllm_comm_malloc(void **ptr, size_t bytes) { return fllm_comm_status(cudaMalloc(ptr, bytes)); }

int fllm_comm_free(void *ptr) { return fllm_comm_status(cudaFree(ptr)); }

int fllm_comm_h2d(void *dst, const void *src, size_t bytes) {
    return fllm_comm_status(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice));
}

int fllm_comm_d2h(void *dst, const void *src, size_t bytes) {
    return fllm_comm_status(cudaMemcpy(dst, src, bytes, cudaMemcpyDeviceToHost));
}

int fllm_comm_stream_create(void **stream) {
    cudaStream_t s = nullptr;
    int rc = fllm_comm_status(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
    *stream = (void *)s;
    return rc;
}

int fllm_comm_stream_destroy(void *stream) { return fllm_comm_status(cudaStreamDestroy((cudaStream_t)stream)); }

int fllm_comm_stream_sync(void *stream) { return fllm_comm_status(cudaStreamSynchronize((cudaStream_t)stream)); }

int fllm_comm_can_access_peer(int dev, int peer, int *can) {
    return fllm_comm_status(cudaDeviceCanAccessPeer(can, dev, peer));
}

// 0 = not supported, otherwise the performance rank reported by CUDA (higher is faster; NVLink pairs rank above PCIe pairs)
int fllm_comm_p2p_rank(int dev, int peer, int *rank) {
    return fllm_comm_status(cudaDeviceGetP2PAttribute(rank, cudaDevP2PAttrPerformanceRank, dev, peer));
}

int fllm_comm_p2p_native_atomics(int dev, int peer, int *supported) {
    return fllm_comm_status(cudaDeviceGetP2PAttribute(supported, cudaDevP2PAttrNativeAtomicSupported, dev, peer));
}


// --- P2P & Custom All-Reduce Extensions ---

int fllm_comm_enable_peer_access(int dev, int peer) {
    cudaError_t e = cudaSetDevice(dev);
    if (e != cudaSuccess) return (int)e;
    e = cudaDeviceEnablePeerAccess(peer, 0);
    // cudaErrorPeerAccessAlreadyEnabled is benign
    if (e == cudaErrorPeerAccessAlreadyEnabled) {
        cudaGetLastError();
        return 0;
    }
    return fllm_comm_status(e);
}

int fllm_comm_disable_peer_access(int dev, int peer) {
    cudaError_t e = cudaSetDevice(dev);
    if (e != cudaSuccess) return (int)e;
    e = cudaDeviceDisablePeerAccess(peer);
    if (e == cudaErrorPeerAccessNotEnabled) {
        cudaGetLastError();
        return 0;
    }
    return fllm_comm_status(e);
}

int fllm_comm_peer_copy_async(void *dst, int dst_dev, const void *src, int src_dev, size_t bytes, void *stream) {
    return fllm_comm_status(cudaMemcpyPeerAsync(dst, dst_dev, src, src_dev, bytes, (cudaStream_t)stream));
}

__global__ void fllm_reduce_add_f32_kernel(float *dst, const float *src, size_t n) {
    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    // Process in float4 if aligned
    size_t n4 = n / 4;
    float4 *dst4 = reinterpret_cast<float4*>(dst);
    const float4 *src4 = reinterpret_cast<const float4*>(src);
    if ((reinterpret_cast<uintptr_t>(dst) % 16 == 0) && (reinterpret_cast<uintptr_t>(src) % 16 == 0)) {
        for (size_t i = idx; i < n4; i += (size_t)gridDim.x * blockDim.x) {
            float4 d = dst4[i];
            float4 s = src4[i];
            d.x += s.x;
            d.y += s.y;
            d.z += s.z;
            d.w += s.w;
            dst4[i] = d;
        }
        size_t rem_start = n4 * 4;
        for (size_t i = rem_start + idx; i < n; i += (size_t)gridDim.x * blockDim.x) {
            dst[i] += src[i];
        }
    } else {
        for (size_t i = idx; i < n; i += (size_t)gridDim.x * blockDim.x) {
            dst[i] += src[i];
        }
    }
}

int fllm_comm_reduce_add_f32(float *dst, const float *src, size_t n, void *stream) {
    if (n == 0) return 0;
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 1024) blocks = 1024;
    fllm_reduce_add_f32_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(dst, src, n);
    return fllm_comm_status(cudaGetLastError());
}

// ---- one-shot all-reduce over peer-mapped memory (small messages: the decode step) ------------------------------------------------------
// Every rank stages its input in a buffer that all peers can read (peer access), tells the peers with a flag, waits for the flags of the
// peers and sums all staged inputs itself in rank order (so the result is bit-identical on every rank). One launch, no ring: for the few
// KiB to hundreds of KiB of a decode step this has far lower latency than a ring or tree all-reduce, as in the one-shot kernels of vLLM
// and TensorRT-LLM. Two staging halves alternate, so no closing barrier is needed: a peer that entered call k has finished call k-2.
// Flags are 64-bit and only grow (epoch = call number), the ranks must issue the same sequence of calls.
}  // extern "C" (templates follow)

#define FLLM_AR_MAX_BLOCKS 8
#define FLLM_AR_MAX_RANKS 16

struct FllmArParams {
    char *staging[FLLM_AR_MAX_RANKS];
    unsigned long long *flags[FLLM_AR_MAX_RANKS];
    int rank, world;
};

template <int DT>  // 0 = f32, 1 = f16, 2 = bf16
struct FllmArElem;
template <>
struct FllmArElem<0> {
    typedef uint32_t Raw;
    static __device__ __forceinline__ float load(Raw r) { return __uint_as_float(r); }
    static __device__ __forceinline__ Raw store(float v) { return __float_as_uint(v); }
};
template <>
struct FllmArElem<1> {
    typedef uint16_t Raw;
    static __device__ __forceinline__ float load(Raw r) { return __half2float(__ushort_as_half(r)); }
    static __device__ __forceinline__ Raw store(float v) { return __half_as_ushort(__float2half_rn(v)); }
};
template <>
struct FllmArElem<2> {
    typedef uint16_t Raw;
    static __device__ __forceinline__ float load(Raw r) { return __uint_as_float(((uint32_t)r) << 16); }
    static __device__ __forceinline__ Raw store(float v) {
        uint32_t u = __float_as_uint(v);
        if ((u & 0x7fffffffu) > 0x7f800000u) return (Raw)((u >> 16) | 0x40);  // NaN stays a NaN
        u += 0x7fffu + ((u >> 16) & 1u);  // round to nearest even
        return (Raw)(u >> 16);
    }
};

template <int DT>
__global__ void fllm_ar_oneshot_kernel(FllmArParams P, void *data, size_t n, unsigned long long epoch, size_t half_off) {
    typedef typename FllmArElem<DT>::Raw Raw;
    const int b = blockIdx.x;
    const size_t nb = gridDim.x;
    size_t per = (n + nb - 1) / nb;
    per = (per + 3) & ~(size_t)3;
    const size_t start = (size_t)b * per;
    const size_t end = start + per < n ? start + per : n;
    Raw *mine = reinterpret_cast<Raw *>(P.staging[P.rank] + half_off);
    const Raw *in = reinterpret_cast<const Raw *>(data);
    for (size_t i = start + threadIdx.x; i < end; i += blockDim.x) mine[i] = in[i];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x < P.world) {
        volatile unsigned long long *peer_flag = P.flags[threadIdx.x] + (size_t)b * P.world + P.rank;
        *peer_flag = epoch;
        __threadfence_system();
        volatile unsigned long long *my_flag = P.flags[P.rank] + (size_t)b * P.world + threadIdx.x;
        while (*my_flag < epoch) {
        }
        __threadfence_system();
    }
    __syncthreads();
    Raw *out = reinterpret_cast<Raw *>(data);
    for (size_t i = start + threadIdx.x; i < end; i += blockDim.x) {
        float acc = 0.0f;
        for (int p = 0; p < P.world; ++p) {
            const volatile Raw *src = reinterpret_cast<const volatile Raw *>(P.staging[p] + half_off);
            acc += FllmArElem<DT>::load(src[i]);
        }
        out[i] = FllmArElem<DT>::store(acc);
    }
}

extern "C" {

// `staging[r]` / `flags[r]` are the addresses of rank r's buffers (valid on this device through peer access). `data` is reduced in place.
// `half_off` selects the staging half of this call. Returns 0 or a runtime error code.
int fllm_comm_ar_oneshot(void *const *staging, unsigned long long *const *flags, int rank, int world, void *data, size_t n, int dtype,
                         unsigned long long epoch, size_t half_off, void *stream) {
    if (n == 0) return 0;
    if (world < 1 || world > FLLM_AR_MAX_RANKS || rank < 0 || rank >= world) return (int)cudaErrorInvalidValue;
    FllmArParams P;
    for (int r = 0; r < world; ++r) {
        P.staging[r] = (char *)staging[r];
        P.flags[r] = flags[r];
    }
    P.rank = rank;
    P.world = world;
    int blocks = (int)((n + 2047) / 2048);
    if (blocks > FLLM_AR_MAX_BLOCKS) blocks = FLLM_AR_MAX_BLOCKS;
    cudaStream_t st = (cudaStream_t)stream;
    switch (dtype) {
        case 0: fllm_ar_oneshot_kernel<0><<<blocks, 256, 0, st>>>(P, data, n, epoch, half_off); break;
        case 1: fllm_ar_oneshot_kernel<1><<<blocks, 256, 0, st>>>(P, data, n, epoch, half_off); break;
        case 2: fllm_ar_oneshot_kernel<2><<<blocks, 256, 0, st>>>(P, data, n, epoch, half_off); break;
        default: return (int)cudaErrorInvalidValue;
    }
    return fllm_comm_status(cudaGetLastError());
}

// Flag memory: on AMD the flags that peers poll must not sit in a write-back cache of the writing GPU, so they come from uncached device
// memory (hipDeviceMallocUncached, as the signal buffers of the vLLM ROCm all-reduce); on NVIDIA plain device memory with system fences works.
int fllm_comm_malloc_flags(void **ptr, size_t bytes) {
#if defined(__HIPCC__)
    int rc = fllm_comm_status(hipExtMallocWithFlags(ptr, bytes, hipDeviceMallocUncached));
    if (rc == 0) return 0;
#endif
    return fllm_comm_status(cudaMalloc(ptr, bytes));
}

int fllm_comm_memset(void *ptr, int value, size_t bytes) { return fllm_comm_status(cudaMemset(ptr, value, bytes)); }

int fllm_comm_device_sync() { return fllm_comm_status(cudaDeviceSynchronize()); }

}  // extern "C"
