// fLLM CUDA backend: prompt attention on Hopper (sm_90a) through FlashInfer's FA3 kernel (warp-specialized, wgmma, TMA for Q / O).
//
// The kernel and its work planner are FlashInfer's port of FlashAttention-3 (include/flashinfer/attention/hopper/*,
// https://github.com/flashinfer-ai/flashinfer, Apache License 2.0, Copyright (c) 2023 by FlashInfer team; the Hopper files carry
// "Copyright (c) 2024, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao. Licensed under the BSD 3-Clause", see
// THIRD_PARTY.md). This file only prepares their inputs, following csrc/batch_prefill_sm90.cu of FlashInfer for one request: the page table
// of the sequence (the KV pool is NHD with pages of `block_size` positions: the "sparse" paged mainloop of FA3 reads it with cp.async), FP32
// queries become halves, the halves of the result are expanded to FP32 (like flashinfer_prefill.cu does for the FA2 kernel).
//
// FLLM_ATTN_BACKEND=auto|fa2|fa3 (default fa2 until the self test passed on an H100): auto picks FA3 on compute capability 9.0 for the shapes below, fa2 keeps FlashInfer's
// FA2 kernel (flashinfer_prefill.cu) / the built-in kernel, fa3 insists on FA3 and falls back when a shape is not covered.
// Covered: f16 cache, key width = value width in {64, 128, 256}, causal, optional sliding window, no attention sinks, no soft cap, one
// request per call. Everything else returns -1 (the caller uses the next path). Compiled for sm_90a only (build.rs, Flavor::Hopper); in other
// builds (-DFLLM_NO_HOPPER) the entry points decline every call.
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <vector>

#include "common.cuh"
#include "fllm_cuda_internal.h"
#include "hopper_util.h"

#if defined(FLLM_HOPPER) && defined(FLLM_HAVE_FLASHINFER) && !defined(FLLM_NO_HOPPER)
#define FLLM_FA3_BUILT 1
#include <cutlass/numeric_types.h>
#include <flashinfer/attention/hopper/default_params.cuh>
#include <flashinfer/attention/hopper/prefill_sm90.cuh>
#include <flashinfer/attention/hopper/variants.cuh>
#include <flashinfer/attention/mask.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/utils.cuh>

namespace fllm_fa3 {

using DType = cutlass::half_t;
using Params = flashinfer::BatchPrefillPagedParams<DType, DType, DType, int32_t>;
using Variant = flashinfer::StandardAttention;

__global__ void table_to_int32_kernel(const int64_t *__restrict__ table, int32_t *__restrict__ dst, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = (int32_t)table[i];
}
__global__ void float_to_half_kernel(const float *__restrict__ src, __half *__restrict__ dst, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        dst[i] = __float2half_rn(src[i]);
}
__global__ void half_to_float_kernel(const __half *__restrict__ src, float *__restrict__ dst, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x)
        dst[i] = __half2float(src[i]);
}
__global__ void half_to_float_copy_kernel(const __half *__restrict__ src, float *__restrict__ dst, __half *__restrict__ dst16, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        dst[i] = __half2float(src[i]);
        dst16[i] = src[i];
    }
}

struct DeviceBuffer {
    void *ptr = nullptr;
    size_t bytes = 0;
    bool reserve(size_t need) {
        if (need <= bytes) return true;
        if (ptr) cudaFree(ptr);
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

constexpr size_t kIntWorkspace = (size_t)8 << 20;

struct State {
    DeviceBuffer q16, o16, indices, int_ws;
    void *pinned = nullptr;
    bool have_plan = false;
    uint64_t plan_sig = 0;
    flashinfer::PrefillPlanSM90Info plan;
};
thread_local State g_state;

template <typename Variant, typename Params, uint32_t D, bool WINDOW>
cudaError_t run_d(cudaStream_t s, const State &st, Params &p) {
    auto *ws = static_cast<int32_t *>(st.int_ws.ptr);
    using flashinfer::GetPtrFromBaseOffset;
    p.qo_tile_indices = GetPtrFromBaseOffset<int32_t>(ws, st.plan.qo_tile_indices_offset);
    p.qo_indptr = GetPtrFromBaseOffset<int32_t>(ws, st.plan.qo_indptr_offset);
    p.kv_indptr = GetPtrFromBaseOffset<int32_t>(ws, st.plan.kv_indptr_offset);
    p.qo_lens = GetPtrFromBaseOffset<int32_t>(ws, st.plan.qo_len_offset);
    p.kv_lens = GetPtrFromBaseOffset<int32_t>(ws, st.plan.kv_len_offset);
    p.head_indices = GetPtrFromBaseOffset<int32_t>(ws, st.plan.head_indices_offset);
    p.work_indptr = GetPtrFromBaseOffset<int32_t>(ws, st.plan.work_indptr_offset);
    p.batch_indices = GetPtrFromBaseOffset<int32_t>(ws, st.plan.batch_indices_offset);
    cudaError_t status = cudaSuccess;
    if (st.plan.same_schedule_for_all_heads) {
        status = flashinfer::BatchPrefillWithPagedKVCacheDispatched<D, D, flashinfer::MaskMode::kCausal, WINDOW, true, Variant, Params>(
            p, /*enable_pdl=*/false, s);
    } else {
        status = flashinfer::BatchPrefillWithPagedKVCacheDispatched<D, D, flashinfer::MaskMode::kCausal, WINDOW, false, Variant, Params>(
            p, /*enable_pdl=*/false, s);
    }
    return status;
}

template <typename Variant, typename Params, bool WINDOW>
cudaError_t run(cudaStream_t s, const State &st, Params &p, int dim) {
    if (dim == 64) return run_d<Variant, Params, 64, WINDOW>(s, st, p);
    if (dim == 128) return run_d<Variant, Params, 128, WINDOW>(s, st, p);
    return run_d<Variant, Params, 256, WINDOW>(s, st, p);
}

}  // namespace fllm_fa3
#endif

// FLLM_ATTN_BACKEND: 0 = auto, 1 = fa2, 2 = fa3 (run-time switch 3 of fi_attn.cu)
int fllm_attn_mode(int which);
bool fllm_fa3_built() {
#if defined(FLLM_FA3_BUILT)
    return true;
#else
    return false;
#endif
}

// Is FA3 the attention kernel of the prompt on this device (FLLM_ATTN_BACKEND, build, compute capability)?
bool fllm_fa3_enabled() {
#if defined(FLLM_FA3_BUILT)
    return fllm_attn_mode(3) != 1 && fllm_cc() == 90 && getenv("FLLM_CUDA_NO_FLASHINFER") == nullptr;
#else
    return false;
#endif
}

#if defined(FLLM_FA3_BUILT)
namespace fllm_fa3 {
// `nreq` requests: query rows qo[i]..qo[i+1] of q, kv_len[i] cached positions, pages pages[kvp[i]..kvp[i+1]) (int32, host) -- or, for one request,
// the block table on the device (int64, `table_dev`).
static int core(cudaStream_t s, const float *q, const uint16_t *kpool, const uint16_t *vpool, float *out, int nreq, const int32_t *qo, const int32_t *kvp,
                const int32_t *kv_len, const int32_t *pages_host, const int64_t *table_dev, int heads, int kv_heads, int dim, int block_size, float scale,
                int window, float softcap = 0.f, const float *sinks = nullptr, int kind = 1) {
    State &st = g_state;
    const int rows = qo[nreq], npages = kvp[nreq];
    const size_t row_halves = (size_t)kv_heads * dim;
    if (row_halves % 8 != 0 || (reinterpret_cast<uintptr_t>(kpool) & 15) != 0 || (reinterpret_cast<uintptr_t>(vpool) & 15) != 0) return -1;
    if (fllm_pending_attn.o16) {  // never overwrite an unexpanded result (api.cu flushes before every operation)
        const FllmPendingAttn p = fllm_pending_attn;
        fllm_pending_attn = FllmPendingAttn();
        fllm_flashinfer_expand_half(s, p.o16, p.out, p.count);
    }
    const size_t q_bytes = (size_t)rows * heads * dim * 2;
    if (!st.q16.reserve(q_bytes) || !st.o16.reserve(q_bytes) || !st.indices.reserve((size_t)npages * 4 + 256) || !st.int_ws.reserve(kIntWorkspace)) return -1;
    if (!st.pinned && cudaMallocHost(&st.pinned, kIntWorkspace) != cudaSuccess) {
        cudaGetLastError();
        st.pinned = nullptr;
        return -1;
    }
    try {
        uint64_t sig = 14695981039346656037ull;
        auto mix = [&](const void *p, size_t bytes) {
            const unsigned char *b = static_cast<const unsigned char *>(p);
            for (size_t i = 0; i < bytes; i++) sig = (sig ^ b[i]) * 1099511628211ull;
        };
        const int shape[5] = {heads, kv_heads, dim, block_size, nreq};
        mix(shape, sizeof shape);
        mix(qo, (size_t)(nreq + 1) * 4);
        mix(kvp, (size_t)(nreq + 1) * 4);
        mix(kv_len, (size_t)nreq * 4);
        if (!(st.have_plan && st.plan_sig == sig)) {
            // the staging buffer is read by an asynchronous copy: wait for the previous plan's copy before rewriting it
            cudaStreamSynchronize(s);
            std::vector<int32_t> qo_plan(qo, qo + nreq + 1), kv_plan(kvp, kvp + nreq + 1), len_plan(kv_len, kv_len + nreq);
            const cudaError_t e = flashinfer::PrefillSM90Plan<int32_t>(nullptr, 0, st.int_ws.ptr, st.pinned, kIntWorkspace, st.plan, qo_plan.data(), kv_plan.data(),
                                                                       len_plan.data(), (uint32_t)rows, (uint32_t)nreq, heads, kv_heads, dim, dim, block_size,
                                                                       /*causal=*/true, /*enable_cuda_graph=*/false, /*sizeof_dtype_o=*/2, s);
            if (e != cudaSuccess) {
                cudaGetLastError();
                st.have_plan = false;
                return -1;
            }
            cudaStreamSynchronize(s);
            st.have_plan = true;
            st.plan_sig = sig;
        }
        const size_t q_elems = (size_t)rows * heads * dim;
        auto grid = [](size_t n) { return (unsigned)((n + 255) / 256 > 4096 ? 4096 : (n + 255) / 256); };
        if (table_dev) {
            table_to_int32_kernel<<<(npages + 255) / 256, 256, 0, s>>>(table_dev, static_cast<int32_t *>(st.indices.ptr), npages);
        } else if (cudaMemcpyAsync(st.indices.ptr, pages_host, (size_t)npages * 4, cudaMemcpyHostToDevice, s) != cudaSuccess) {
            cudaGetLastError();
            return -1;
        }
        float_to_half_kernel<<<grid(q_elems), 256, 0, s>>>(q, static_cast<__half *>(st.q16.ptr), q_elems);

        Params p;
        memset(&p, 0, sizeof p);
        p.q_ptr = static_cast<DType *>(st.q16.ptr);
        p.k_ptr = const_cast<DType *>(reinterpret_cast<const DType *>(kpool));
        p.v_ptr = const_cast<DType *>(reinterpret_cast<const DType *>(vpool));
        p.o_ptr = static_cast<DType *>(st.o16.ptr);
        p.lse_ptr = nullptr;
        p.kv_indices = static_cast<int32_t *>(st.indices.ptr);
        p.additional_params.logits_soft_cap = softcap;
        p.additional_params.sm_scale = scale;
        p.q_stride_n = (int64_t)heads * dim;
        p.q_stride_h = dim;
        p.o_stride_n = (int64_t)heads * dim;
        p.o_stride_h = dim;
        p.k_stride_n = p.v_stride_n = (int64_t)kv_heads * dim;
        p.k_stride_h = p.v_stride_h = dim;
        p.k_page_stride = p.v_page_stride = (int64_t)block_size * kv_heads * dim;
        p.nnz_qo = rows;
        p.num_qo_heads = heads;
        p.num_kv_heads = kv_heads;
        p.group_size = heads / kv_heads;
        p.page_size = block_size;
        // FlashInfer masks the keys more than window_left positions before the last one; the window of the engine counts the current position
        p.window_left = window > 0 ? window - 1 : -1;
        p.causal = true;
        cudaError_t status = cudaSuccess;
        if (softcap > 0.f) {
            status = window > 0 ? run<flashinfer::LogitsSoftCap, Params, true>(s, st, p, dim)
                                : run<flashinfer::LogitsSoftCap, Params, false>(s, st, p, dim);
        } else {
            status = window > 0 ? run<flashinfer::StandardAttention, Params, true>(s, st, p, dim)
                                : run<flashinfer::StandardAttention, Params, false>(s, st, p, dim);
        }
        if (status != cudaSuccess) {
            cudaGetLastError();
            return -1;
        }
        if (fllm_attn_x16 != nullptr && (reinterpret_cast<uintptr_t>(fllm_attn_x16) & 15) == 0) {
            half_to_float_copy_kernel<<<grid(q_elems), 256, 0, s>>>(static_cast<const __half *>(st.o16.ptr), out, static_cast<__half *>(fllm_attn_x16), q_elems);
            fllm_attn_x16_written = true;
        } else if (fllm_fuse_nvfp4_attn_out() && (reinterpret_cast<uintptr_t>(out) & 15) == 0 && (reinterpret_cast<uintptr_t>(st.o16.ptr) & 15) == 0) {
            fllm_pending_attn.o16 = st.o16.ptr;
            fllm_pending_attn.out = out;
            fllm_pending_attn.count = q_elems;
        } else {
            half_to_float_kernel<<<grid(q_elems), 256, 0, s>>>(static_cast<const __half *>(st.o16.ptr), out, q_elems);
        }
    } catch (const std::exception &) {
        cudaGetLastError();
        g_state.have_plan = false;
        return -1;
    }
    fllm_attn_last_path = 1;
    return fllm_check_launch("fa3 prefill");
}
}  // namespace fllm_fa3
#endif

// `rows` consecutive query rows of one sequence whose last row sees `kv_len` positions (`table`: its block table on the device, one entry
// per page); window: sliding window in positions (0 = none). Returns 0 when the kernel ran, -1 when the shape or the build is not covered
// (the caller uses the next path), > 0 on a launch error.
int fllm_fa3_prefill_try(cudaStream_t s, const float *q, const uint16_t *kpool, const uint16_t *vpool, float *out, const int64_t *table, int rows,
                         int kv_len, int heads, int kv_heads, int dk, int dv, int block_size, float scale, int window,
                         float softcap, const float *sinks, int kind) {
#if defined(FLLM_FA3_BUILT)
    if (!fllm_fa3_enabled() || dk != dv || (dk != 64 && dk != 128 && dk != 256) || kv_len < rows || rows < 1 || heads % kv_heads != 0) return -1;
    const int32_t qo[2] = {0, rows}, kvp[2] = {0, (kv_len + block_size - 1) / block_size}, len[1] = {kv_len};
    return fllm_fa3::core(s, q, kpool, vpool, out, 1, qo, kvp, len, nullptr, table, heads, kv_heads, dk, block_size, scale, window, softcap, sinks, kind);
#else
    (void)s; (void)q; (void)kpool; (void)vpool; (void)out; (void)table; (void)rows; (void)kv_len; (void)heads;
    (void)kv_heads; (void)dk; (void)dv; (void)block_size; (void)scale; (void)window;
    return -1;
#endif
}

// Several requests (ragged prompts): request i has query rows qo[i]..qo[i+1], kv_len[i] cached positions and the pages pages[kvp[i]..kvp[i+1]) (host
// arrays, int32 page numbers). Same return values.
int fllm_fa3_prefill_runs(cudaStream_t s, const float *q, const uint16_t *kpool, const uint16_t *vpool, float *out, int nreq, const int32_t *qo,
                          const int32_t *kvp, const int32_t *kv_len, const int32_t *pages, int heads, int kv_heads, int dk, int dv, int block_size,
                          float scale, int window) {
#if defined(FLLM_FA3_BUILT)
    if (!fllm_fa3_enabled() || dk != dv || (dk != 64 && dk != 128 && dk != 256) || nreq < 1 || heads % kv_heads != 0) return -1;
    for (int i = 0; i < nreq; i++)
        if (kv_len[i] < qo[i + 1] - qo[i] || qo[i + 1] == qo[i]) return -1;
    return fllm_fa3::core(s, q, kpool, vpool, out, nreq, qo, kvp, kv_len, pages, nullptr, heads, kv_heads, dk, block_size, scale, window);
#else
    (void)s; (void)q; (void)kpool; (void)vpool; (void)out; (void)nreq; (void)qo; (void)kvp; (void)kv_len; (void)pages; (void)heads;
    (void)kv_heads; (void)dk; (void)dv; (void)block_size; (void)scale; (void)window;
    return -1;
#endif
}
