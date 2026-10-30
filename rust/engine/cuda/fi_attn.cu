// fLLM CUDA backend: attention on the paged f16 KV cache through FlashInfer's batched FA2 kernel with the variants the engine needs.
//
// flashinfer_prefill.cu runs FlashInfer for one sequence without sliding window, soft cap and attention sinks. This file covers what that one
// leaves to the slower built-in kernels (attention4.cu processes one query row per block with CUDA cores):
//   * sliding window (Mistral, Gemma 2/3, gpt-oss), logits soft cap (Gemma 2), attention sinks (gpt-oss), through one variant with run-time
//     flags (FlashInfer's variant interface: LogitsTransform / LogitsMask / update_m_d / OutputTransform, following the AttentionSink variant of
//     flashinfer/jit/attention/variants.py);
//   * several sequences in one launch (ragged prompts, prompt chunks next to decode rows): every run of consecutive rows of one sequence is one
//     FlashInfer request;
//   * decode (one query row per sequence) on tensor cores: FlashInfer's prefill kernel with a 16-row query tile packs the query heads of a KV
//     head (GQA group 4 .. 8) into the M dimension of the mma, where attention4.cu does the same products with CUDA cores;
//   * cascade attention: sequences that share a long prefix of cache pages read it once (the group's query rows are one non-causal request),
//     their own suffix in a second request, and the two states are merged with the log-sum-exp rule (vLLM's cascade_attention).
//
// The kernels and the planner are FlashInfer (https://github.com/flashinfer-ai/flashinfer, Apache License 2.0, Copyright (c) 2023 by
// FlashInfer team); the sink update follows their AttentionSink variant (Apache-2.0). The cascade idea is vLLM's (Apache-2.0).
//
// Switches (read once):
//   FLLM_ATTN_FI=auto|0|1   1: prompt runs with sliding window / soft cap / sinks and batches of several runs use this path; auto = 0 until the
//                           path is verified on hardware.
//   FLLM_ATTN_DECODE=builtin|flashinfer   decode rows (one row per sequence) on the tensor-core path (default builtin: attention4.cu).
//   FLLM_CASCADE_ATTN=0|1   cascade attention for decode rows that share a prefix (default 0; needs FLLM_ATTN_DECODE=flashinfer or FLLM_ATTN_FI=1
//                           to be reached, see fllm_fi_attention_try).
//   FLLM_CASCADE_MIN_PREFIX   shortest shared prefix in tokens (default 128).
// Covered: f16 cache, key width = value width in {64, 128, 256}, no ALiBi, no chunked attention. Returns -1 otherwise (the caller continues with
// the built-in kernels). Not used while a CUDA graph is captured (the plan needs host synchronization).
#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <mutex>
#include <vector>

#include "common.cuh"
#include "fllm_cuda_internal.h"
#include "hopper_util.h"

// ---- run-time switches of the Hopper / FlashInfer attention paths (the environment gives the start values; selftest-gpu flips them) ----
//   0 FLLM_ATTN_FI (-1 auto, 0, 1)   1 FLLM_ATTN_DECODE (0 builtin, 1 flashinfer)   2 FLLM_CASCADE_ATTN (0, 1)   3 FLLM_ATTN_BACKEND (0 auto, 1 fa2, 2 fa3)
// 4 (get only) = the path the last call on this thread took: 0 built-in kernels, 1 FA3, 2 FlashInfer FA2 with variants / batch / decode, 3 cascade
// 5 (get only) = what this build and device can run: bit 0 FA3 (sm_90a build on a 9.0 GPU), bit 1 FlashInfer (build with the headers, sm_75+ GPU)
static std::atomic<int> g_attn_mode[4];
static std::atomic<bool> g_attn_mode_init{false};
thread_local int fllm_attn_last_path = 0;
int fllm_attn_mode(int which) {
    if (!g_attn_mode_init.load(std::memory_order_acquire)) {
        static std::mutex mu;
        std::lock_guard<std::mutex> lock(mu);
        if (!g_attn_mode_init.load()) {
            static const char *const decode_names[] = {"builtin", "flashinfer"};
            static const char *const backend_names[] = {"auto", "fa2", "fa3"};
            g_attn_mode[0] = fllm_env_tri("FLLM_ATTN_FI");
            g_attn_mode[1] = fllm_env_choice("FLLM_ATTN_DECODE", decode_names, 2, 0);
            g_attn_mode[2] = fllm_env_tri("FLLM_CASCADE_ATTN") == 1 ? 1 : 0;
            g_attn_mode[3] = fllm_env_choice("FLLM_ATTN_BACKEND", backend_names, 3, 1);
            g_attn_mode_init.store(true, std::memory_order_release);
        }
    }
    return which >= 0 && which < 4 ? g_attn_mode[which].load(std::memory_order_relaxed) : 0;
}
bool fllm_fa3_built();
extern "C" int fllm_cuda_attn_get(int which) {
    if (which == 4) return fllm_attn_last_path;
    if (which == 5) {
        int caps = 0;
        if (fllm_fa3_built() && fllm_cc() == 90) caps |= 1;
#if defined(FLLM_HAVE_FLASHINFER)
        if (fllm_cc() >= 75) caps |= 2;
#endif
        return caps;
    }
    return fllm_attn_mode(which);
}
// Sets a switch (or, for 4, the last-path record) and returns the previous value; -1 for an unknown switch.
extern "C" int fllm_cuda_attn_set(int which, int value) {
    if (which == 4) {
        const int old = fllm_attn_last_path;
        fllm_attn_last_path = value;
        return old;
    }
    if (which < 0 || which > 3) return -1;
    fllm_attn_mode(0);  // initialize from the environment first
    return g_attn_mode[which].exchange(value);
}

#if defined(FLLM_HAVE_FLASHINFER)
#include <flashinfer/attention/default_prefill_params.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/prefill.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variant_helper.cuh>
#include <flashinfer/math.cuh>
#include <flashinfer/pos_enc.cuh>
#include <flashinfer/utils.cuh>

namespace fllm_fi2 {

template <typename DType>
using Params = flashinfer::BatchPrefillPagedParams<DType, DType, DType, int32_t>;

// One variant for all combinations, decided at run time (a handful of uniform branches per logit; a template per combination would multiply the
// instantiations of the heavy FlashInfer kernel by eight). `params.maybe_alibi_slopes` carries the per-head sink logits (ALiBi is not used).
struct Variant : flashinfer::AttentionVariantBase {
    static constexpr bool use_softmax = true;

    uint32_t qo_len, kv_len, window_left;
    float sm_scale_log2, soft_cap_pre_tanh_scale;
    bool cap, sink;

    template <typename P>
    __device__ __host__ Variant(const P &params, uint32_t batch_idx, uint8_t *) {
        qo_len = params.get_qo_len(batch_idx);
        kv_len = params.get_kv_len(batch_idx);
        window_left = (params.window_left >= 0) ? params.window_left : kv_len;
        cap = params.logits_soft_cap > 0.f;
        sink = params.maybe_alibi_slopes != nullptr;
        if (cap) {
            soft_cap_pre_tanh_scale = params.sm_scale / params.logits_soft_cap;
            sm_scale_log2 = flashinfer::math::log2e * params.logits_soft_cap;
        } else {
            soft_cap_pre_tanh_scale = 0.f;
            sm_scale_log2 = params.sm_scale * flashinfer::math::log2e;
        }
    }

    REGISTER_LOGITS_TRANSFORM(params, logits, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, {
        if (cap) logits = float(flashinfer::math::tanh(logits * soft_cap_pre_tanh_scale));
        return logits;
    })

    // causality is the kernel's mask mode; this only adds the sliding window (window_left is kv_len when there is none)
    REGISTER_LOGITS_MASK(params, batch_idx, qo_idx, kv_idx, qo_head_idx, kv_head_idx, { return kv_idx + qo_len + window_left >= kv_len + qo_idx; })

    // m and d are in the base-2 domain here (m already multiplied by sm_scale_log2); the sink logit joins the softmax denominator once, in the
    // first KV chunk of a request
    REGISTER_M_D_UPDATE(params, kv_tile_idx, qo_head_idx, m, d, scale, {
        if (!sink) {
            scale = 1.f;
            return;
        }
        const float log_sink = (kv_tile_idx == 0 && qo_head_idx < params.num_qo_heads) ? params.maybe_alibi_slopes[qo_head_idx] * flashinfer::math::log2e
                                                                                        : -flashinfer::math::inf;
        const float m_new = (log_sink > m) ? log_sink : m;
        scale = flashinfer::math::ptx_exp2(m - m_new);
        const float d_new = flashinfer::math::ptx_exp2(log_sink - m_new) + d * scale;
        m = m_new;
        d = d_new;
    })

    REGISTER_OUTPUT_TRANSFORM(params, output, batch_idx, qo_idx, qo_head_idx, m, d, scale, {
        const float d_rcp = (m != -flashinfer::math::inf) ? flashinfer::math::ptx_rcp(d) : 0.f;
        return output * scale * d_rcp;
    })
};

// internal row i of the batch reads original row perm[i] (perm null: identity)
template <typename DType>
__global__ void gather_f32_to_d(const float *__restrict__ src, DType *__restrict__ dst, const int32_t *__restrict__ perm, int rows, int row_elems) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)rows * row_elems; i += (size_t)gridDim.x * blockDim.x) {
        const int r = (int)(i / row_elems), c = (int)(i % row_elems);
        const int sr = perm ? perm[r] : r;
        if constexpr (std::is_same_v<DType, __nv_bfloat16>) {
            dst[i] = __float2bfloat16_rn(src[(size_t)sr * row_elems + c]);
        } else {
            dst[i] = __float2half_rn(src[(size_t)sr * row_elems + c]);
        }
    }
}
__global__ void gather_f32_to_f16(const float *__restrict__ src, __half *__restrict__ dst, const int32_t *__restrict__ perm, int rows, int row_elems) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < (size_t)rows * row_elems; i += (size_t)gridDim.x * blockDim.x) {
        const int r = (int)(i / row_elems), c = (int)(i % row_elems);
        const int sr = perm ? perm[r] : r;
        dst[i] = __float2half_rn(src[(size_t)sr * row_elems + c]);
    }
}
template <typename DType>
__global__ void d_to_float_k(const DType *__restrict__ src, float *__restrict__ dst, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        if constexpr (std::is_same_v<DType, __nv_bfloat16>) {
            dst[i] = __bfloat162float(src[i]);
        } else {
            dst[i] = __half2float(src[i]);
        }
    }
}
template <typename DType>
__global__ void d_to_float_copy_k(const DType *__restrict__ src, float *__restrict__ dst, DType *__restrict__ dst16, size_t n) {
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        if constexpr (std::is_same_v<DType, __nv_bfloat16>) {
            dst[i] = __bfloat162float(src[i]);
        } else {
            dst[i] = __half2float(src[i]);
        }
        dst16[i] = src[i];
    }
}
// Cascade merge and re-ordering: internal row i < shared_rows is the log-sum-exp merge of the prefix state (o0, s0) and the suffix state (o1, s1)
// (both from FlashInfer: base-2 logsumexp per row and head), the others are the suffix state alone; the result goes to original row perm[i].
// The suffix state of a request without any key has s = -inf and drops out.
__global__ void merge_scatter(const __half *__restrict__ o0, const float *__restrict__ s0, const __half *__restrict__ o1, const float *__restrict__ s1,
                              __half *__restrict__ dst, const int32_t *__restrict__ perm, int rows, int shared_rows, int heads, int dim) {
    const int r = blockIdx.x;
    if (r >= rows) return;
    const int dr = perm ? perm[r] : r;
    for (int h = threadIdx.y; h < heads; h += blockDim.y) {
        float a = 1.f, b = 0.f;
        const size_t at = (size_t)r * heads + h;
        if (r < shared_rows) {
            const float x = s0[at], y = s1[at];
            const float mx = fmaxf(x, y);
            const float ex = exp2f(x - mx), ey = exp2f(y - mx);
            a = ex / (ex + ey);
            b = ey / (ex + ey);
        }
        for (int c = threadIdx.x; c < dim; c += blockDim.x) {
            const float v1 = __half2float(o1[at * dim + c]);
            const float v = r < shared_rows ? a * __half2float(o0[at * dim + c]) + b * v1 : v1;
            dst[((size_t)dr * heads + h) * dim + c] = __float2half_rn(v);
        }
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

constexpr size_t kFloatWorkspace = (size_t)64 << 20, kIntWorkspace = (size_t)8 << 20;

// One FlashInfer request set: the host description, its device copy and the plan.
struct Level {
    DeviceBuffer ints, int_ws;
    void *pinned = nullptr;
    flashinfer::PrefillPlanInfo plan;
    bool planned = false;
    uint64_t sig = 0;
    int32_t *d_qo = nullptr, *d_kvp = nullptr, *d_last = nullptr, *d_idx = nullptr, *d_perm = nullptr;
};

struct State {
    DeviceBuffer q16, o16a, o16b, o16c, lse0, lse1, float_ws;
    Level level[2];
};
thread_local State g_state;

inline uint64_t fnv(uint64_t h, const void *data, size_t bytes) {
    const unsigned char *p = static_cast<const unsigned char *>(data);
    for (size_t i = 0; i < bytes; i++) h = (h ^ p[i]) * 1099511628211ull;
    return h;
}

// Request description on the host.
struct Requests {
    std::vector<int32_t> qo{0}, kvp{0}, last, idx, perm;
    int n = 0;
    int rows = 0;
    void add(int qo_len, int kv_len, int block, const int64_t *table, int first_page) {
        const int pages = (kv_len + block - 1) / block;
        for (int p = first_page; p < first_page + pages; p++) idx.push_back((int32_t)table[p]);
        last.push_back(kv_len - (pages - 1) * block);
        kvp.push_back((int32_t)idx.size());
        qo.push_back(qo.back() + qo_len);
        rows += qo_len;
        n++;
    }
};

// Copies the description to the device and plans it (skipped when it is the one already there).
bool prepare(cudaStream_t s, State &st, Level &lv, const Requests &rq, int heads, int kv_heads, int dim, int block, int window_left, bool want_perm) {
    uint64_t sig = 14695981039346656037ull;
    sig = fnv(sig, rq.qo.data(), rq.qo.size() * 4);
    sig = fnv(sig, rq.kvp.data(), rq.kvp.size() * 4);
    sig = fnv(sig, rq.last.data(), rq.last.size() * 4);
    sig = fnv(sig, rq.idx.data(), rq.idx.size() * 4);
    if (want_perm) sig = fnv(sig, rq.perm.data(), rq.perm.size() * 4);
    const int params[6] = {heads, kv_heads, dim, block, window_left, (int)(uintptr_t)s & 0x7fffffff};
    sig = fnv(sig, params, sizeof params);
    if (lv.planned && lv.sig == sig) return true;
    lv.planned = false;
    auto up4 = [](size_t n) { return (n + 3) & ~size_t(3); };
    const size_t o_kvp = up4(rq.qo.size()), o_last = o_kvp + up4(rq.kvp.size()), o_idx = o_last + up4(rq.last.size()),
                 o_perm = o_idx + up4(rq.idx.size()), total = o_perm + (want_perm ? up4(rq.perm.size()) : 0);
    if (!lv.ints.reserve(total * 4 + 64) || !lv.int_ws.reserve(kIntWorkspace) || !st.float_ws.reserve(kFloatWorkspace)) return false;
    if (!lv.pinned && cudaMallocHost(&lv.pinned, kIntWorkspace) != cudaSuccess) {
        cudaGetLastError();
        lv.pinned = nullptr;
        return false;
    }
    std::vector<int32_t> packed(total, 0);
    std::copy(rq.qo.begin(), rq.qo.end(), packed.begin());
    std::copy(rq.kvp.begin(), rq.kvp.end(), packed.begin() + o_kvp);
    std::copy(rq.last.begin(), rq.last.end(), packed.begin() + o_last);
    std::copy(rq.idx.begin(), rq.idx.end(), packed.begin() + o_idx);
    if (want_perm) std::copy(rq.perm.begin(), rq.perm.end(), packed.begin() + o_perm);
    // the pageable source is staged before the call returns
    if (cudaMemcpyAsync(lv.ints.ptr, packed.data(), total * 4, cudaMemcpyHostToDevice, s) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    int32_t *base = static_cast<int32_t *>(lv.ints.ptr);
    lv.d_qo = base;
    lv.d_kvp = base + o_kvp;
    lv.d_last = base + o_last;
    lv.d_idx = base + o_idx;
    lv.d_perm = want_perm ? base + o_perm : nullptr;
    // the staging buffer of the plan is read by an asynchronous copy: wait for the previous plan's copy before rewriting it
    cudaStreamSynchronize(s);
    std::vector<int32_t> qo_plan = rq.qo, kv_plan = rq.kvp;
    const cudaError_t e = flashinfer::PrefillPlan<int32_t>(st.float_ws.ptr, kFloatWorkspace, lv.int_ws.ptr, lv.pinned, kIntWorkspace, lv.plan, qo_plan.data(),
                                                           kv_plan.data(), (uint32_t)rq.rows, (uint32_t)rq.n, heads, kv_heads, dim, dim, block,
                                                           /*enable_cuda_graph=*/false, /*sizeof_dtype_o=*/2, window_left, /*fixed_split_size=*/-1,
                                                           /*disable_split_kv=*/false, /*num_colocated_ctas=*/0, s);
    if (e != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    cudaStreamSynchronize(s);
    lv.sig = sig;
    lv.planned = true;
    return true;
}

template <typename DType, uint32_t CTA, uint32_t D, flashinfer::MaskMode MASK>
cudaError_t run_tile(cudaStream_t s, Params<DType> &p, DType *tmp_v, float *tmp_s) {
    return flashinfer::BatchPrefillWithPagedKVCacheDispatched<CTA, D, D, flashinfer::PosEncodingMode::kNone, /*use_fp16_qk_reduction=*/false, MASK, Variant, Params<DType>>(
        p, tmp_v, tmp_s, /*enable_pdl=*/false, s);
}

template <typename DType, uint32_t D, flashinfer::MaskMode MASK>
cudaError_t run_dim(cudaStream_t s, State &st, Level &lv, Params<DType> &p) {
    using namespace flashinfer;
    const PrefillPlanInfo &plan = lv.plan;
    int32_t *ws = static_cast<int32_t *>(lv.int_ws.ptr);
    p.request_indices = GetPtrFromBaseOffset<int32_t>(ws, plan.request_indices_offset);
    p.qo_tile_indices = GetPtrFromBaseOffset<int32_t>(ws, plan.qo_tile_indices_offset);
    p.kv_tile_indices = GetPtrFromBaseOffset<int32_t>(ws, plan.kv_tile_indices_offset);
    p.o_indptr = GetPtrFromBaseOffset<int32_t>(ws, plan.o_indptr_offset);
    p.kv_chunk_size_ptr = GetPtrFromBaseOffset<int32_t>(ws, plan.kv_chunk_size_ptr_offset);
    DType *tmp_v = nullptr;
    float *tmp_s = nullptr;
    if (plan.split_kv) {
        p.merge_indptr = GetPtrFromBaseOffset<int32_t>(ws, plan.merge_indptr_offset);
        tmp_v = GetPtrFromBaseOffset<DType>(st.float_ws.ptr, plan.v_offset);
        tmp_s = GetPtrFromBaseOffset<float>(st.float_ws.ptr, plan.s_offset);
    }
    p.padded_batch_size = (uint32_t)plan.padded_batch_size;
    p.max_total_num_rows = (uint32_t)plan.total_num_rows;
    p.partition_kv = plan.split_kv;
    switch (plan.cta_tile_q) {
        case 16: return run_tile<DType, 16, D, MASK>(s, p, tmp_v, tmp_s);
        case 64: return run_tile<DType, 64, D, MASK>(s, p, tmp_v, tmp_s);
        case 128:
            if constexpr (D < 256) return run_tile<DType, 128, D, MASK>(s, p, tmp_v, tmp_s);
            return cudaErrorNotSupported;
        default: return cudaErrorNotSupported;
    }
}

template <typename DType, flashinfer::MaskMode MASK>
cudaError_t run_level(cudaStream_t s, State &st, Level &lv, Params<DType> &p, int dim) {
    if (dim == 64) return run_dim<DType, 64, MASK>(s, st, lv, p);
    if (dim == 128) return run_dim<DType, 128, MASK>(s, st, lv, p);
    if constexpr (MASK == flashinfer::MaskMode::kCausal) {
        if (dim == 256) return run_dim<DType, 256, MASK>(s, st, lv, p);
    }
    return cudaErrorNotSupported;
}

// Fills the parameters of one launch and runs it. q16 / o16: DType buffers of the rows of `rq` in the order of its requests.
template <typename DType>
cudaError_t launch(cudaStream_t s, State &st, Level &lv, const Requests &rq, bool causal, const DType *q16, DType *o16, float *lse, const float *sinks,
                   const uint16_t *kpool, const uint16_t *vpool, int heads, int kv_heads, int dim, int block, float scale, float softcap, int window_left) {
    Params<DType> p;
    p.q = const_cast<DType *>(q16);
    p.paged_kv = flashinfer::paged_kv_t<DType, int32_t>(kv_heads, block, dim, rq.n, flashinfer::QKVLayout::kNHD, const_cast<DType *>(reinterpret_cast<const DType *>(kpool)),
                                                        const_cast<DType *>(reinterpret_cast<const DType *>(vpool)), lv.d_idx, lv.d_kvp, lv.d_last);
    p.q_indptr = lv.d_qo;
    p.o = o16;
    p.lse = lse;
    p.num_qo_heads = heads;
    p.group_size = flashinfer::uint_fastdiv(heads / kv_heads);
    p.q_stride_n = heads * dim;
    p.q_stride_h = dim;
    p.window_left = window_left;
    p.maybe_custom_mask = nullptr;
    p.maybe_mask_indptr = nullptr;
    p.maybe_alibi_slopes = const_cast<float *>(sinks);
    p.logits_soft_cap = softcap;
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
    return causal ? run_level<DType, flashinfer::MaskMode::kCausal>(s, st, lv, p, dim) : run_level<DType, flashinfer::MaskMode::kNone>(s, st, lv, p, dim);
}

struct Run {
    int r0, n, kv;
    const int64_t *table;
};

}  // namespace fllm_fi2
#endif  // FLLM_HAVE_FLASHINFER

// Returns 0 when the attention was computed, -1 when this path does not take the call (the caller continues), > 0 on a launch error.
// tables_host / lengths_host: block table (int64, `stride` per row) and length of every row; rows of one sequence are consecutive with the
// length growing by one and one table. `sinks`: device array of one logit per head or null.
int fllm_fi_attention_try(cudaStream_t s, const float *q, const uint16_t *kpool, const uint16_t *vpool, float *out, const int64_t *tables_host,
                          const int64_t *lengths_host, int rows, const float *sinks, bool has_alibi, int heads, int kv_heads, int dk, int dv, int stride,
                          int block_size, int kind, float scale, float softcap, int window, int chunk) {
#if defined(FLLM_HAVE_FLASHINFER)
    using namespace fllm_fi2;
    const int fi_mode = fllm_attn_mode(0);  // 1 on, otherwise off (auto = off until verified on hardware)
    const bool decode_on = fllm_attn_mode(1) == 1;
    const bool cascade_on = fllm_attn_mode(2) == 1;
    static const int min_prefix = [] {
        const char *e = getenv("FLLM_CASCADE_MIN_PREFIX");
        return e && atoi(e) > 0 ? atoi(e) : 128;
    }();
    static const bool disabled = getenv("FLLM_CUDA_NO_FLASHINFER") != nullptr;
    const bool fi_on = fi_mode == 1;  // auto = off until verified on hardware
    if ((!fi_on && !decode_on && !cascade_on) || disabled) return -1;
    if ((kind != 1 && kind != 3) || has_alibi || chunk != 0 || dk != dv || (dk != 64 && dk != 128 && dk != 256) || fllm_cc() < 75 || rows <= 0 || kv_heads <= 0 ||
        heads % kv_heads != 0 || fllm_exact_math())
        return -1;
    if (((size_t)kv_heads * dk) % 8 != 0 || (reinterpret_cast<uintptr_t>(kpool) & 15) != 0 || (reinterpret_cast<uintptr_t>(vpool) & 15) != 0 ||
        (reinterpret_cast<uintptr_t>(sinks) & 3) != 0)
        return -1;
    cudaStreamCaptureStatus cs = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(s, &cs) != cudaSuccess || cs != cudaStreamCaptureStatusNone) {
        cudaGetLastError();
        return -1;
    }
    // runs of consecutive rows of one sequence
    std::vector<Run> runs;
    int max_qo = 0;
    for (int r0 = 0; r0 < rows;) {
        int r1 = r0 + 1;
        const int64_t *t0 = tables_host + (size_t)r0 * stride;
        while (r1 < rows && lengths_host[r1] == lengths_host[r1 - 1] + 1 && tables_host[(size_t)r1 * stride] == t0[0]) {
            const int pages = (int)((lengths_host[r1] + block_size - 1) / block_size);
            if (pages > stride || memcmp(tables_host + (size_t)r1 * stride, t0, (size_t)pages * sizeof(int64_t)) != 0) break;
            r1++;
        }
        const int kv = (int)lengths_host[r1 - 1];
        if (kv < r1 - r0 || (kv + block_size - 1) / block_size > stride) return -1;
        runs.push_back({r0, r1 - r0, kv, t0});
        max_qo = std::max(max_qo, r1 - r0);
        r0 = r1;
    }
    const bool decode_like = max_qo < 16;
    const bool variants = softcap > 0.f || window > 0 || sinks != nullptr;
    const bool take = decode_like ? (decode_on || cascade_on) : (fi_on && (variants || runs.size() > 1));
    if (!take) return -1;
    if (fllm_pending_attn.o16) {  // never overwrite an unexpanded result (api.cu flushes before every operation)
        const FllmPendingAttn p = fllm_pending_attn;
        fllm_pending_attn = FllmPendingAttn();
        fllm_flashinfer_expand_half(s, p.o16, p.out, p.count);
    }
    State &st = g_state;
    const size_t row_elems = (size_t)heads * dk, q_elems = (size_t)rows * row_elems;
    if (!st.q16.reserve(q_elems * 2) || !st.o16a.reserve(q_elems * 2) || !st.o16c.reserve(q_elems * 2)) return -1;
    auto grid = [](size_t n) { return (unsigned)((n + 255) / 256 > 4096 ? 4096 : (n + 255) / 256); };
    const int window_left = window > 0 ? window - 1 : -1;
    const __half *final16 = nullptr;
    bool cascaded = false;
    try {
        // ---- cascade: groups of single-row sequences sharing the leading pages of their tables
        struct Group {
            std::vector<int> members;
            int pages;
        };
        std::vector<Group> groups;
        std::vector<char> grouped(runs.size(), 0);
        if (cascade_on && window == 0 && dk < 256) {
            std::map<int64_t, std::vector<int>> by_first;
            for (size_t i = 0; i < runs.size(); i++)
                if (runs[i].n == 1) by_first[runs[i].table[0]].push_back((int)i);
            const int min_pages = (min_prefix + block_size - 1) / block_size;
            for (auto &kv : by_first) {
                if (kv.second.size() < 2) continue;
                // the longest run is the reference; a run that cannot share min_pages pages with it (too short, or another prefix) stays out of the
                // group and runs on its own, instead of keeping the others from sharing
                std::vector<int> &ids = kv.second;
                std::stable_sort(ids.begin(), ids.end(), [&](int x, int y) { return runs[x].kv > runs[y].kv; });
                const Run &a = runs[ids[0]];
                std::vector<int> members{ids[0]};
                int common = (a.kv - 1) / block_size;  // keep at least one suffix key in every member
                for (size_t j = 1; j < ids.size(); j++) {
                    const Run &b = runs[ids[j]];
                    const int lim = std::min(common, (b.kv - 1) / block_size);
                    int k = 0;
                    while (k < lim && a.table[k] == b.table[k]) k++;
                    if (k < min_pages) continue;
                    common = k;
                    members.push_back(ids[j]);
                }
                if (members.size() >= 2 && common * block_size >= min_prefix) {
                    groups.push_back({members, common});
                    for (int m : members) grouped[m] = 1;
                }
            }
        }
        if (decode_like && !decode_on && groups.empty()) return -1;  // cascade only: nothing shared, the built-in kernel keeps the call
        if (!groups.empty()) {
            // internal order: grouped runs group by group, then the rest
            Requests shared, rest;
            std::vector<int> order;
            for (const Group &g : groups) {
                const Run &a = runs[g.members[0]];
                shared.n++;
                for (int m : g.members) order.push_back(m);
                for (int p = 0; p < g.pages; p++) shared.idx.push_back((int32_t)a.table[p]);
                shared.last.push_back(block_size);
                shared.kvp.push_back((int32_t)shared.idx.size());
                shared.qo.push_back(shared.qo.back() + (int)g.members.size());
                shared.rows += (int)g.members.size();
            }
            const int shared_rows = (int)order.size();
            // the suffix request set: all runs in internal order; grouped ones skip the shared pages
            std::vector<int> group_pages(runs.size(), 0);
            for (const Group &g : groups)
                for (int m : g.members) group_pages[m] = g.pages;
            for (size_t i = 0; i < runs.size(); i++)
                if (!grouped[i]) order.push_back((int)i);
            for (int m : order) {
                const Run &r = runs[m];
                rest.add(r.n, r.kv - group_pages[m] * block_size, block_size, r.table, group_pages[m]);
                for (int j = 0; j < r.n; j++) rest.perm.push_back(r.r0 + j);
            }
            // The suffix request of a grouped run attends to the keys after the shared pages: its single query row sees all of them, which is what
            // the causal mask of a one-row request gives. The query rows of the shared level are the first shared_rows internal rows.
            if (!st.o16b.reserve(q_elems * 2) || !st.lse0.reserve((size_t)rows * heads * 4) || !st.lse1.reserve((size_t)rows * heads * 4)) return -1;
            if (!prepare(s, st, st.level[0], shared, heads, kv_heads, dk, block_size, -1, false) ||
                !prepare(s, st, st.level[1], rest, heads, kv_heads, dk, block_size, window_left, true))
                return -1;
            gather_f32_to_f16<<<grid(q_elems), 256, 0, s>>>(q, static_cast<__half *>(st.q16.ptr), st.level[1].d_perm, rows, (int)row_elems);
            // prefix part (no sinks: the sink joins the denominator once, in the suffix part)
            cudaError_t e = launch<__half>(s, st, st.level[0], shared, /*causal=*/false, static_cast<__half *>(st.q16.ptr), static_cast<__half *>(st.o16b.ptr),
                                           static_cast<float *>(st.lse0.ptr), nullptr, kpool, vpool, heads, kv_heads, dk, block_size, scale, softcap, -1);
            if (e == cudaSuccess)
                e = launch<__half>(s, st, st.level[1], rest, /*causal=*/true, static_cast<__half *>(st.q16.ptr), static_cast<__half *>(st.o16a.ptr),
                                   static_cast<float *>(st.lse1.ptr), sinks, kpool, vpool, heads, kv_heads, dk, block_size, scale, softcap, window_left);
            if (e != cudaSuccess) {
                cudaGetLastError();
                st.level[0].planned = st.level[1].planned = false;
                return -1;
            }
            const int by = std::min(heads, 1024 / 32);
            merge_scatter<<<rows, dim3(32, by), 0, s>>>(static_cast<const __half *>(st.o16b.ptr), static_cast<const float *>(st.lse0.ptr),
                                                        static_cast<const __half *>(st.o16a.ptr), static_cast<const float *>(st.lse1.ptr),
                                                        static_cast<__half *>(st.o16c.ptr), st.level[1].d_perm, rows, shared_rows, heads, dk);
            final16 = static_cast<const __half *>(st.o16c.ptr);
            // the merged FP16 rows go back to the FP32 output like those of the other paths (without this the call returned the old contents of `out`)
            if (fllm_attn_x16 != nullptr && (reinterpret_cast<uintptr_t>(fllm_attn_x16) & 15) == 0) {
                d_to_float_copy_k<<<grid(q_elems), 256, 0, s>>>(final16, out, static_cast<__half *>(fllm_attn_x16), q_elems);
                fllm_attn_x16_written = true;
            } else {
                d_to_float_k<<<grid(q_elems), 256, 0, s>>>(final16, out, q_elems);
            }
            cascaded = true;
        } else {
            Requests rq;
            for (const Run &r : runs) rq.add(r.n, r.kv, block_size, r.table, 0);
            if (!decode_like && softcap == 0.f && sinks == nullptr && fllm_fa3_enabled()) {
                // Hopper: the FA3 kernel takes ragged prompts and the sliding window as well
                std::vector<int32_t> lens;
                for (const Run &r : runs) lens.push_back(r.kv);
                const int rc3 = fllm_fa3_prefill_runs(s, q, kpool, vpool, out, rq.n, rq.qo.data(), rq.kvp.data(), lens.data(), rq.idx.data(), heads, kv_heads,
                                                      dk, dv, block_size, scale, window);
                if (rc3 >= 0) return rc3;
            }
            if (!prepare(s, st, st.level[1], rq, heads, kv_heads, dk, block_size, window_left, false)) return -1;
            if (kind == 3) {
                gather_f32_to_d<<<grid(q_elems), 256, 0, s>>>(q, static_cast<__nv_bfloat16 *>(st.q16.ptr), nullptr, rows, (int)row_elems);
                cudaError_t e = cudaErrorNotSupported;
                if (e != cudaSuccess) {
                    e = launch<__nv_bfloat16>(s, st, st.level[1], rq, /*causal=*/true, static_cast<__nv_bfloat16 *>(st.q16.ptr), static_cast<__nv_bfloat16 *>(st.o16a.ptr), nullptr,
                                              sinks, kpool, vpool, heads, kv_heads, dk, block_size, scale, softcap, window_left);
                }
                if (e != cudaSuccess) {
                    cudaGetLastError();
                    st.level[1].planned = false;
                    return -1;
                }
                if (fllm_attn_x16 != nullptr && (reinterpret_cast<uintptr_t>(fllm_attn_x16) & 15) == 0) {
                    d_to_float_copy_k<<<grid(q_elems), 256, 0, s>>>(static_cast<const __nv_bfloat16 *>(st.o16a.ptr), out, static_cast<__nv_bfloat16 *>(fllm_attn_x16), q_elems);
                    fllm_attn_x16_written = true;
                } else {
                    d_to_float_k<<<grid(q_elems), 256, 0, s>>>(static_cast<const __nv_bfloat16 *>(st.o16a.ptr), out, q_elems);
                }
            } else {
                gather_f32_to_d<<<grid(q_elems), 256, 0, s>>>(q, static_cast<__half *>(st.q16.ptr), nullptr, rows, (int)row_elems);
                cudaError_t e = cudaErrorNotSupported;
                if (e != cudaSuccess) {
                    e = launch<__half>(s, st, st.level[1], rq, /*causal=*/true, static_cast<__half *>(st.q16.ptr), static_cast<__half *>(st.o16a.ptr), nullptr,
                                       sinks, kpool, vpool, heads, kv_heads, dk, block_size, scale, softcap, window_left);
                }
                if (e != cudaSuccess) {
                    cudaGetLastError();
                    st.level[1].planned = false;
                    return -1;
                }
                final16 = static_cast<const __half *>(st.o16a.ptr);
                if (fllm_attn_x16 != nullptr && (reinterpret_cast<uintptr_t>(fllm_attn_x16) & 15) == 0) {
                    d_to_float_copy_k<<<grid(q_elems), 256, 0, s>>>(final16, out, static_cast<__half *>(fllm_attn_x16), q_elems);
                    fllm_attn_x16_written = true;
                } else {
                    d_to_float_k<<<grid(q_elems), 256, 0, s>>>(final16, out, q_elems);
                }
            }
        }
    } catch (const std::exception &) {
        cudaGetLastError();
        g_state.level[0].planned = g_state.level[1].planned = false;
        return -1;
    }
    fllm_attn_last_path = cascaded ? 3 : 2;
    return fllm_check_launch("flashinfer attention");
#else
    (void)s; (void)q; (void)kpool; (void)vpool; (void)out; (void)tables_host; (void)lengths_host; (void)rows; (void)sinks; (void)has_alibi;
    (void)heads; (void)kv_heads; (void)dk; (void)dv; (void)stride; (void)block_size; (void)kind; (void)scale; (void)softcap; (void)window; (void)chunk;
    return -1;
#endif
}
