// Device helpers of the multi-GPU communication layer (src/comm/): device selection, plain device memory, a stream per rank,
// synchronous host copies for the host-slice collectives, and peer-access queries for the topology report.
// NCCL itself is not linked: it is loaded at run time (src/comm/nccl.rs).
#include <cuda_runtime.h>
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

__global__ void fllm_custom_all_reduce_f32_kernel(float *out, const float * const *peer_ptrs, int num_peers, size_t n) {
    size_t idx = (size_t)blockDim.x * blockIdx.x + threadIdx.x;
    for (size_t i = idx; i < n; i += (size_t)gridDim.x * blockDim.x) {
        float sum = 0.0f;
        for (int p = 0; p < num_peers; ++p) {
            sum += peer_ptrs[p][i];
        }
        out[i] = sum;
    }
}

int fllm_comm_custom_all_reduce_f32(float *out, const float * const *peer_ptrs, int num_peers, size_t n, void *stream) {
    if (n == 0) return 0;
    int threads = 256;
    int blocks = (int)((n + threads - 1) / threads);
    if (blocks > 1024) blocks = 1024;
    fllm_custom_all_reduce_f32_kernel<<<blocks, threads, 0, (cudaStream_t)stream>>>(out, peer_ptrs, num_peers, n);
    return fllm_comm_status(cudaGetLastError());
}

}  // extern "C"
