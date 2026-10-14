// fLLM CUDA backend: prompt attention through FlashInfer's batched prefill kernel on the paged KV cache.
//
// The attention kernel and its work planner are FlashInfer (https://github.com/flashinfer-ai/flashinfer, Apache License
// 2.0, Copyright (c) 2023 by FlashInfer team); this file only prepares their inputs, following csrc/batch_prefill.cu of
// that project for one request: the page table of the sequence (our KV pool is already FlashInfer's NHD layout with
// pages of `block_size` positions), FP32 queries become halves, the halves of the result are expanded to FP32.
// FlashInfer reaches 72-86 TFLOPS on an RTX PRO 3000 Blackwell where the built-in tensor-core attention (attention3.cu)
// reaches about 47. It is used for f16 caches without attention sinks (gpt-oss) when the headers are available at build
// time (third_party/flashinfer or FLLM_FLASHINFER_DIR, see THIRD_PARTY.md); everything else keeps the built-in kernels.
#include <cstdint>
#include <cstdlib>
#include <exception>

#include "common.cuh"

#include "fllm_cuda_internal.h"

#if defined(FLLM_HAVE_FLASHINFER)
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/utils.cuh>

namespace fllm_fi {

using Params = flashinfer::BatchPrefillPagedParams<__half, __half, __half, int32_t>;
using Variant = flashinfer::DefaultAttention</*custom_mask=*/false, /*sliding_window=*/false, /*logits_soft_cap=*/false,
                                             /*alibi=*/false>;

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
// The same, and the halves are also copied to `dst16`: the activations the output projection (a Marlin GEMM) would otherwise convert back again.
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

constexpr size_t kFloatWorkspace = (size_t)128 << 20, kIntWorkspace = (size_t)8 << 20;

struct State {
    DeviceBuffer q16, o16, indices, small, float_ws, int_ws;
    void *pinned = nullptr;  // page-locked staging of the plan
    // the plan sitting in the integer workspace: it depends on these numbers only, so it serves all layers of a chunk
    bool have_plan = false;
    int plan_rows = 0, plan_kv = 0, plan_heads = 0, plan_kv_heads = 0, plan_dim = 0, plan_block = 0;
    // the request description in `small` (qo_indptr, kv_indptr, last_page_len) depends on these three numbers only: it is copied once per chunk, not per layer
    const void *small_ptr = nullptr;
    cudaStream_t small_stream = nullptr;  // only the stream that wrote it may rely on it
    int small_rows = -1, small_pages = -1, small_last = -1;
    flashinfer::PrefillPlanInfo plan;
};
thread_local State g_state;

template <uint32_t D>
cudaError_t run(cudaStream_t s, State &st, Params &p, int32_t *int_ws_base, void *float_ws_base) {
    using namespace flashinfer;
    const PrefillPlanInfo &plan = st.plan;
    p.request_indices = GetPtrFromBaseOffset<int32_t>(int_ws_base, plan.request_indices_offset);
    p.qo_tile_indices = GetPtrFromBaseOffset<int32_t>(int_ws_base, plan.qo_tile_indices_offset);
    p.kv_tile_indices = GetPtrFromBaseOffset<int32_t>(int_ws_base, plan.kv_tile_indices_offset);
    p.o_indptr = GetPtrFromBaseOffset<int32_t>(int_ws_base, plan.o_indptr_offset);
    p.kv_chunk_size_ptr = GetPtrFromBaseOffset<int32_t>(int_ws_base, plan.kv_chunk_size_ptr_offset);
    __half *tmp_v = nullptr;
    float *tmp_s = nullptr;
    if (plan.split_kv) {
        p.merge_indptr = GetPtrFromBaseOffset<int32_t>(int_ws_base, plan.merge_indptr_offset);
        tmp_v = GetPtrFromBaseOffset<__half>(float_ws_base, plan.v_offset);
        tmp_s = GetPtrFromBaseOffset<float>(float_ws_base, plan.s_offset);
    }
    p.padded_batch_size = (uint32_t)plan.padded_batch_size;
    p.max_total_num_rows = (uint32_t)plan.total_num_rows;
    p.partition_kv = plan.split_kv;
    cudaError_t status = cudaSuccess;
    DISPATCH_CTA_TILE_Q(plan.cta_tile_q, CTA_TILE_Q, {
        status = flashinfer::BatchPrefillWithPagedKVCacheDispatched<CTA_TILE_Q, D, D, flashinfer::PosEncodingMode::kNone,
                                                                    /*use_fp16_qk_reduction=*/false,
                                                                    flashinfer::MaskMode::kCausal, Variant, Params>(
            p, tmp_v, tmp_s, /*enable_pdl=*/false, s);
    });
    return status;
}

}  // namespace fllm_fi
#endif  // FLLM_HAVE_FLASHINFER

// `rows` consecutive query rows of one sequence whose last row sees `kv_len` positions (`table`: its block table on the
// device, one entry per page). Returns 0 when the kernel ran, -1 when the shape or the build is not covered (the caller
// uses the built-in attention), > 0 on a launch error.
int fllm_flashinfer_prefill_try(cudaStream_t s, const float *q, const uint16_t *kpool, const uint16_t *vpool, float *out,
                                const int64_t *table, int rows, int kv_len, int heads, int kv_heads, int dk, int dv,
                                int block_size, float scale) {
#if defined(FLLM_HAVE_FLASHINFER)
    using namespace fllm_fi;
    static const bool disabled = getenv("FLLM_CUDA_NO_FLASHINFER") != nullptr;
    if (disabled || fllm_cc() < 75 || dk != dv || (dk != 64 && dk != 128) || kv_len < rows || heads % kv_heads != 0)
        return -1;
    const size_t row_halves = (size_t)kv_heads * dk;
    if (row_halves % 8 != 0 || (reinterpret_cast<uintptr_t>(kpool) & 15) != 0 || (reinterpret_cast<uintptr_t>(vpool) & 15) != 0)
        return -1;
    State &st = g_state;
    const int pages = (kv_len + block_size - 1) / block_size;
    const size_t q_bytes = (size_t)rows * heads * dk * 2;
    if (!st.q16.reserve(q_bytes) || !st.o16.reserve(q_bytes) || !st.indices.reserve((size_t)pages * 4 + 256) ||
        !st.small.reserve(256) || !st.float_ws.reserve(kFloatWorkspace) || !st.int_ws.reserve(kIntWorkspace))
        return -1;
    if (!st.pinned && cudaMallocHost(&st.pinned, kIntWorkspace) != cudaSuccess) {
        cudaGetLastError();
        st.pinned = nullptr;
        return -1;
    }
    // device-side request description: qo_indptr {0, rows}, kv_indptr {0, pages}, last_page_len {n}
    int32_t *small = static_cast<int32_t *>(st.small.ptr);
    const int32_t qo_h[2] = {0, rows}, kv_h[2] = {0, pages}, last_h[1] = {kv_len - (pages - 1) * block_size};
    if (!(fllm_opt(FLLM_OPT_STEP_CACHE) && st.small_ptr == st.small.ptr && st.small_stream == s && st.small_rows == rows && st.small_pages == pages &&
          st.small_last == last_h[0])) {
        st.small_ptr = nullptr;
        if (cudaMemcpyAsync(small, qo_h, sizeof qo_h, cudaMemcpyHostToDevice, s) != cudaSuccess ||
            cudaMemcpyAsync(small + 4, kv_h, sizeof kv_h, cudaMemcpyHostToDevice, s) != cudaSuccess ||
            cudaMemcpyAsync(small + 8, last_h, sizeof last_h, cudaMemcpyHostToDevice, s) != cudaSuccess) {
            cudaGetLastError();
            return -1;
        }
        st.small_ptr = st.small.ptr;
        st.small_stream = s;
        st.small_rows = rows;
        st.small_pages = pages;
        st.small_last = last_h[0];
    }
    table_to_int32_kernel<<<(pages + 255) / 256, 256, 0, s>>>(table, static_cast<int32_t *>(st.indices.ptr), pages);

    try {
        if (!(st.have_plan && st.plan_rows == rows && st.plan_kv == kv_len && st.plan_heads == heads &&
              st.plan_kv_heads == kv_heads && st.plan_dim == dk && st.plan_block == block_size)) {
            // the staging buffer is read by an asynchronous copy: wait for the previous plan's copy before rewriting it
            cudaStreamSynchronize(s);
            int32_t qo_plan[2] = {0, rows}, kv_plan[2] = {0, pages};
            const cudaError_t e = flashinfer::PrefillPlan<int32_t>(
                st.float_ws.ptr, kFloatWorkspace, st.int_ws.ptr, st.pinned, kIntWorkspace, st.plan, qo_plan, kv_plan,
                rows, /*batch_size=*/1, heads, kv_heads, dk, dk, block_size, /*enable_cuda_graph=*/false,
                /*sizeof_dtype_o=*/2, /*window_left=*/-1, /*fixed_split_size=*/-1, /*disable_split_kv=*/false,
                /*num_colocated_ctas=*/0, s);
            if (e != cudaSuccess) {
                cudaGetLastError();
                st.have_plan = false;
                return -1;
            }
            cudaStreamSynchronize(s);
            st.have_plan = true;
            st.plan_rows = rows;
            st.plan_kv = kv_len;
            st.plan_heads = heads;
            st.plan_kv_heads = kv_heads;
            st.plan_dim = dk;
            st.plan_block = block_size;
        }
        const size_t q_elems = (size_t)rows * heads * dk;
        auto grid = [](size_t n) { return (unsigned)((n + 255) / 256 > 4096 ? 4096 : (n + 255) / 256); };
        float_to_half_kernel<<<grid(q_elems), 256, 0, s>>>(q, static_cast<__half *>(st.q16.ptr), q_elems);

        Params p;
        p.q = static_cast<__half *>(st.q16.ptr);
        p.paged_kv = flashinfer::paged_kv_t<__half, int32_t>(
            kv_heads, block_size, dk, /*batch_size=*/1, flashinfer::QKVLayout::kNHD,
            const_cast<__half *>(reinterpret_cast<const __half *>(kpool)),
            const_cast<__half *>(reinterpret_cast<const __half *>(vpool)), static_cast<int32_t *>(st.indices.ptr),
            small + 4, small + 8);
        p.q_indptr = small;
        p.o = static_cast<__half *>(st.o16.ptr);
        p.lse = nullptr;
        p.num_qo_heads = heads;
        p.group_size = flashinfer::uint_fastdiv(heads / kv_heads);
        p.q_stride_n = heads * dk;
        p.q_stride_h = dk;
        p.window_left = -1;
        p.maybe_custom_mask = nullptr;
        p.maybe_mask_indptr = nullptr;
        p.maybe_alibi_slopes = nullptr;
        p.logits_soft_cap = 0.f;
        p.sm_scale = scale;
        p.rope_rcp_scale = 1.f;
        p.rope_rcp_theta = 1.f;
        p.request_indices = nullptr;
        p.qo_tile_indices = nullptr;
        p.kv_tile_indices = nullptr;
        p.merge_indptr = nullptr;
        p.o_indptr = nullptr;
        p.kv_chunk_size_ptr = nullptr;
        p.block_valid_mask = nullptr;
        p.total_num_rows = nullptr;
        int32_t *int_ws = static_cast<int32_t *>(st.int_ws.ptr);
        const cudaError_t status = dk == 64 ? run<64>(s, st, p, int_ws, st.float_ws.ptr) : run<128>(s, st, p, int_ws, st.float_ws.ptr);
        if (status != cudaSuccess) {
            cudaGetLastError();
            return -1;
        }
        if (fllm_attn_x16 != nullptr && (reinterpret_cast<uintptr_t>(fllm_attn_x16) & 15) == 0) {
            half_to_float_copy_kernel<<<grid(q_elems), 256, 0, s>>>(static_cast<const __half *>(st.o16.ptr), out,
                                                                    static_cast<__half *>(fllm_attn_x16), q_elems);
            fllm_attn_x16_written = true;
        } else {
            half_to_float_kernel<<<grid(q_elems), 256, 0, s>>>(static_cast<const __half *>(st.o16.ptr), out, q_elems);
        }
    } catch (const std::exception &) {
        cudaGetLastError();
        st.have_plan = false;
        return -1;
    }
    return fllm_check_launch("flashinfer prefill");
#else
    (void)s; (void)q; (void)kpool; (void)vpool; (void)out; (void)table; (void)rows; (void)kv_len; (void)heads;
    (void)kv_heads; (void)dk; (void)dv; (void)block_size; (void)scale;
    return -1;
#endif
}
