// fLLM CUDA backend: C API used by the Rust side (rust/engine/src/backend/cuda.rs).
//
// All functions return 0 on success and a non-zero code on failure; `fllm_cuda_last_error`
// describes the last failure of the calling thread. The library links the CUDA runtime
// statically and loads the driver (libcuda) lazily, so a binary built with it still starts on a
// machine without NVIDIA drivers (device_count then reports an error and fLLM falls back).
#include <cstdarg>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <atomic>
#include <mutex>
#include <vector>

#include "common.cuh"
#include "fllm_cuda.h"
#include "fllm_cuda_internal.h"
#if !defined(__HIPCC__)
#include "fp8_gemm.h"
void fllm_cublaslt_set_mode(int mode);
#endif

#include "mla_latent.h"

static thread_local char g_error[512] = "";

// state of the NVFP4 / Marlin FP16 paths (declared in fllm_cuda_internal.h; defined outside of extern "C" like fllm_glu_up)
thread_local bool fllm_fp4_acc = false;
thread_local float fllm_fp4_acc_scale = 0.f;
thread_local bool fllm_fp4_f16out = false;
thread_local bool fllm_glu_half = false;
// FLLM_FUSE_NVFP4_ATTN_OUT / FLLM_FUSE_NVFP4_NORM (see fllm_cuda_internal.h): off unless set to something other than 0
thread_local FllmPendingAttn fllm_pending_attn;
thread_local const void *fllm_x_half = nullptr;
static bool env_switch(const char *name) {
    const char *v = getenv(name);
    return v && *v && strcmp(v, "0") != 0;
}
bool fllm_fuse_nvfp4_attn_out() {
    static const bool on = env_switch("FLLM_FUSE_NVFP4_ATTN_OUT");
    return on;
}
bool fllm_fuse_nvfp4_norm() {
    static const bool on = env_switch("FLLM_FUSE_NVFP4_NORM");
    return on;
}
// state of the Marlin 4-bit path (same rule: the header declares these with C++ linkage, so they must not be defined inside extern "C")
thread_local bool fllm_q4z_f16out = false;
thread_local bool fllm_q4z_skip_result = false;
thread_local bool fllm_q4z_acc = false;
thread_local float fllm_q4z_acc_scale = 0.f;
thread_local bool fllm_q4z_defer_acc = false;
thread_local bool fllm_q4z_deferred = false;
thread_local const void *fllm_q4z_y16 = nullptr;
// prompt attention: when fllm_attn_x16 is set the FlashInfer path also writes its result as FP16 there and reports it in fllm_attn_x16_written
thread_local void *fllm_attn_x16 = nullptr;
thread_local bool fllm_attn_x16_written = false;

int fllm_set_error(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(g_error, sizeof g_error, fmt, ap);
    va_end(ap);
    return 1;
}
int fllm_check_launch(const char *what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) return fllm_set_error("%s: %s", what, cudaGetErrorString(e));
    return 0;
}
static int fail(const char *what, cudaError_t e) { return fllm_set_error("%s: %s", what, cudaGetErrorString(e)); }
#define CHECK(call, what)                    \
    do {                                     \
        const cudaError_t e_ = (call);       \
        if (e_ != cudaSuccess) return fail(what, e_); \
    } while (0)

// Compute capability (major * 10 + minor) of the current device; FLLM_CUDA_FORCE_CC overrides it (testing the fallbacks).
int fllm_cc() {
    static int cache[16] = {0};
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) return 0;
    if (cache[dev] == 0) {
        if (const char *forced = getenv("FLLM_CUDA_FORCE_CC")) {
            cache[dev] = atoi(forced);
        } else {
#if defined(__HIPCC__)
            cache[dev] = 10;  // not an NVIDIA architecture: the tensor-core kernels (cc >= 75) stay off
#else
            cudaDeviceProp p;
            cache[dev] = cudaGetDeviceProperties(&p, dev) == cudaSuccess ? p.major * 10 + p.minor : 1;
#endif
        }
    }
    return cache[dev];
}

// Largest dynamic shared memory per block the current device allows (opt-in maximum); FLLM_CUDA_FORCE_SMEM overrides it
// (testing a smaller device).
static std::atomic<int> g_forced_smem{0};
int fllm_smem_limit() {
    static int cache[16] = {0};
    int dev = 0;
    // (the self test sets it at run time to check the paths of a device with less shared memory than this one, such as Turing's 64 KB)
    if (const int forced = g_forced_smem.load(std::memory_order_relaxed)) return forced;
    if (cudaGetDevice(&dev) != cudaSuccess || dev < 0 || dev >= 16) return 48 * 1024;
    if (cache[dev] == 0) {
        if (const char *forced = getenv("FLLM_CUDA_FORCE_SMEM")) {
            cache[dev] = atoi(forced);
        } else {
            int v = 0;
            cache[dev] = cudaDeviceGetAttribute(&v, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev) == cudaSuccess && v > 0 ? v : 48 * 1024;
        }
    }
    return cache[dev];
}

// Prefill switches (see FllmOpt): the environment gives the start value, fllm_cuda_set_opt overrides it (selftest-gpu compares variants).
static std::atomic<int> g_opt[FLLM_OPT_COUNT];
static std::atomic<bool> g_opt_init{false};
static void opt_init() {
    static std::mutex mu;
    std::lock_guard<std::mutex> lock(mu);
    if (g_opt_init.load()) return;
    auto flag = [](const char *name) {
        const char *v = getenv(name);
        return v == nullptr || (strcmp(v, "0") != 0 && strcmp(v, "off") != 0 && strcmp(v, "false") != 0);
    };
    g_opt[FLLM_OPT_STEP_CACHE] = flag("FLLM_PREFILL_STEP_CACHE");
    g_opt[FLLM_OPT_FUSE_ADD_NORM] = flag("FLLM_FUSE_PREFILL_ADD_NORM");
    g_opt[FLLM_OPT_FUSE_NORM_F16] = flag("FLLM_FUSE_PREFILL_NORM_F16");
    g_opt[FLLM_OPT_FUSE_ATTN_F16] = flag("FLLM_FUSE_PREFILL_ATTN_F16");
    const char *cfg = getenv("FLLM_MARLIN_PREFILL_CFG");
    g_opt[FLLM_OPT_MARLIN_CFG] = cfg != nullptr && strcmp(cfg, "fllm") == 0;
    g_opt_init.store(true);
}
int fllm_opt(int id) {
    if (id < 0 || id >= FLLM_OPT_COUNT) return 0;
    if (!g_opt_init.load(std::memory_order_acquire)) opt_init();
    return g_opt[id].load(std::memory_order_relaxed);
}

extern "C" {

const char *fllm_cuda_last_error(void) { return g_error; }

int fllm_cuda_get_opt(int id) { return fllm_opt(id); }
int fllm_cuda_set_opt(int id, int value) {
    if (id < 0 || id >= FLLM_OPT_COUNT) return fllm_set_error("unknown prefill switch %d", id);
    if (!g_opt_init.load()) opt_init();
    g_opt[id] = value;
    return 0;
}

int fllm_cuda_device_count(void) {
    int n = 0;
    const cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess) {
        cudaGetLastError();
        fllm_set_error("cudaGetDeviceCount: %s", cudaGetErrorString(e));
        return -1;
    }
    return n;
}

int fllm_cuda_device_info(int dev, char *name, size_t name_len, int *cc_major, int *cc_minor, size_t *free_bytes,
                          size_t *total_bytes, int *sm_count) {
    cudaDeviceProp p;
    CHECK(cudaGetDeviceProperties(&p, dev), "cudaGetDeviceProperties");
#if defined(__HIPCC__)
    snprintf(name, name_len, "%s (%s)", p.name, p.gcnArchName);
    *cc_major = 0;  // the Rust side prints "CUDA/ROCm" for 0
    *cc_minor = p.integrated ? 1 : 0;  // 1: an APU, its memory is the system's
#else
    snprintf(name, name_len, "%s", p.name);
    *cc_major = p.major;
    *cc_minor = p.minor;
#endif
    *sm_count = p.multiProcessorCount;
    int prev = 0;
    cudaGetDevice(&prev);
    CHECK(cudaSetDevice(dev), "cudaSetDevice");
    CHECK(cudaMemGetInfo(free_bytes, total_bytes), "cudaMemGetInfo");
    cudaSetDevice(prev);
#if defined(__HIPCC__)
    // An APU reports the whole shared memory as free, but the ROCm runtime hands out only about three quarters of it to one process (a Radeon 840M with
    // 7.7 GB reported failed with "HSA exception: MemoryRegion::BlockAllocator::alloc failed" at about 6 GB, and ran with 5.4 GB).
    // (FLLM_HIP_FULL_APU_MEMORY=1 keeps the figures of the runtime, for a driver that hands out what it reports, such as a large dedicated block on Windows)
    const char *full = getenv("FLLM_HIP_FULL_APU_MEMORY");
    if (p.integrated && !(full && *full && full[0] != '0')) {
        *free_bytes = *free_bytes / 4 * 3;
        *total_bytes = *total_bytes / 4 * 3;
    }
#endif
    return 0;
}

int fllm_cuda_malloc(int dev, void **ptr, size_t bytes) {
    CHECK(cudaSetDevice(dev), "cudaSetDevice");
    CHECK(cudaMalloc(ptr, bytes ? bytes : 1), "cudaMalloc");
    return 0;
}
int fllm_cuda_free(void *ptr) { CHECK(cudaFree(ptr), "cudaFree"); return 0; }
int fllm_cuda_h2d_sync(int dev, void *dst, const void *src, size_t bytes) {
    CHECK(cudaSetDevice(dev), "cudaSetDevice");
    CHECK(cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice), "cudaMemcpy");
    // A synchronous copy from pageable memory may return once the data is staged, with the DMA to the device still queued in the legacy default
    // stream, and the compute streams are non-blocking, so nothing orders their kernels after it: wait for the legacy stream so that a weight (or its
    // NVFP4 trailer, the last bytes of the buffer) is complete when the first GEMM reads it.
    CHECK(cudaStreamSynchronize(0), "cudaStreamSynchronize(legacy)");
    return 0;
}
int fllm_cuda_memset(int dev, void *ptr, int value, size_t bytes) {
    CHECK(cudaSetDevice(dev), "cudaSetDevice");
    CHECK(cudaMemset(ptr, value, bytes), "cudaMemset");
    CHECK(cudaStreamSynchronize(0), "cudaStreamSynchronize(legacy)");  // as above: the compute streams are not ordered after the legacy stream
    return 0;
}

// Per-step host data of a captured decode step (positions, cache slots, block tables, lengths): a device buffer the kernels read, and a
// pinned mirror that a copy node of the graph uploads at every launch.
struct GraphSlot {
    void *dev = nullptr, *pinned = nullptr;
    size_t cap = 0, bytes = 0;
    bool staged = false;
};
struct FllmGraph {
    void *exec = nullptr;  // cudaGraphExec_t
    GraphSlot slot[4];
    int max_len = 0;       // attention sizes its splits for this context length instead of the one in the batch
    unsigned epoch = 0;
};

static void graph_free(FllmGraph *g);

struct FllmStream {
    int dev;
    cudaStream_t s;
    void *arena;      // ring of per-call metadata (positions, slots, block tables ...)
    size_t cap, off;
    float *ws;        // scratch for split-KV attention partials
    size_t ws_bytes;
    void *x16;        // FP16 copy of the activations for the tensor-core GEMM
    size_t x16_bytes;
    // recently staged metadata: layers of one step upload identical positions, slots, tables and lengths
    struct Staged {
        std::vector<unsigned char> data;
        void *dev = nullptr;
    } staged[8];
    int staged_next = 0;
    // activations of the last NVFP4 linear are already quantized in `x16` (q, k and v share their input)
    const float *fp4_x = nullptr;
    int fp4_batch = 0, fp4_cols = 0;
    bool fp4_ready = false;
    // activations of the last linear are already converted in `x16`
    const float *x16_x = nullptr;
    int x16_batch = 0, x16_cols = 0;
    bool x16_ready = false;
    const float *x8_x = nullptr;
    int x8_batch = 0, x8_cols = 0;
    bool x8_ready = false;  // the scratch holds the int8 activations of x8_x (mmvq.cu)
    // a glu is only launched when something else needs its result: a following GEMM folds it into its activation
    // conversion (see fllm_cuda_glu)
    const float *glu_gate = nullptr, *glu_up = nullptr;
    float *glu_out = nullptr;
    size_t glu_count = 0;
    int glu_mode = 0;
    bool glu_pending = false;
    bool glu_half = false;  // the pending glu reads FP16 gate and up (cast to float pointers): only an NVFP4 CUTLASS GEMM folds it in
    // a prompt RMSNorm that has not run yet: the NVFP4 CUTLASS GEMM that reads its result runs it fused with the activation quantization (FLLM_FUSE_NVFP4_NORM)
    struct {
        bool pending = false;
        const float *x = nullptr, *w = nullptr;
        float *out = nullptr;
        int rows = 0, cols = 0;
        float eps = 0.f;
    } norm;
    // Marlin (Q4Z weights): FP16 activations and result, split-K partials and the lock workspace (zero between calls)
    void *q4z_io = nullptr, *q4z_ctmp = nullptr, *q4z_ws = nullptr;
    size_t q4z_io_bytes = 0, q4z_ctmp_bytes = 0, q4z_ws_bytes = 0;
    int sms = 0;
    int *argmax_out = nullptr;  // device ids of the last argmax (rows x int)
    size_t argmax_cap = 0;
    void *sample_scratch = nullptr;  // parameters, lists and results of the last sample call, and the work copy of the logits
    size_t sample_cap = 0;
    // CUDA graphs of whole decode steps (see fllm_cuda_graph_begin): the one being captured, and a counter that is bumped whenever a
    // scratch buffer a captured kernel may point to is freed (a graph captured before that is stale)
    std::vector<struct FllmGraph *> graphs;
    struct FllmGraph *capture = nullptr;
    unsigned epoch = 1;
    // the FP16 activations in q4z_io are those of q4z_x (q, k and v, and gate and up, share their input)
    const float *q4z_x = nullptr;
    int q4z_batch = 0, q4z_cols = 0;
    const int *q4z_perm = nullptr;
    bool q4z_ready = false;
    // Host data of one forward pass (fllm_cuda_step_begin): block tables and lengths are staged once and the layout of the prompt rows is
    // computed once, instead of once per layer. A slot is valid while the staging ring has not been wrapped or reallocated (arena_gen).
    struct StepSlot {
        const void *host = nullptr;
        size_t bytes = 0;
        unsigned gen = 0;
        void *dev = nullptr;
    };
    struct Step {
        bool open = false;
        StepSlot tables, lengths, compact;
        const int64_t *run_tables = nullptr, *run_lengths = nullptr;
        int run_rows = 0, run_stride = 0;
        std::vector<std::pair<int, int>> runs;  // rows [first, end) that share one block table and grow by one position per row
        std::vector<int64_t> compact_tables;    // the block table of the first row of every run, back to back
    } step;
    unsigned arena_gen = 1;
    // The output projection of a Marlin GEMM leaves `x += scale * (y16 + bias)` pending when the RMS norm that follows can do it in its
    // own pass (FLLM_FUSE_PREFILL_ADD_NORM); anything else that comes first flushes it (flush_add).
    struct PendingAdd {
        bool on = false;
        const void *y16 = nullptr;
        const float *bias = nullptr;
        float *x = nullptr;
        int width = 0, batch = 0, in_cols = 0;
        float scale = 0.f;
    } add;
    // FLLM_ACT_DTYPE (act_api.inc): scratch for the conversions around operations without a narrow kernel and for the narrow Marlin result, and the device
    // counters of FLLM_ACT_CHECK
    void *act_tmp[3] = {nullptr, nullptr, nullptr};
    size_t act_tmp_bytes[3] = {0, 0, 0};
    int *act_check = nullptr;
    // arenas replaced by a larger one: pointers into them may still be held by the call that grew the ring (an attention call stages its
    // lengths, then its block tables), so they are freed with the stream
    std::vector<void *> retired_arenas;
};

static int flush_add(FllmStream *st) {
    if (!st->add.on) return 0;
    st->add.on = false;
    return fllm_q4z_acc_launch(st->s, st->add.y16, st->add.bias, st->add.x, st->add.width, st->add.batch, st->add.scale);
}
static int flush_glu(FllmStream *st) {
    if (int e = flush_add(st)) return e;
    if (st->norm.pending) {  // a deferred RMSNorm (fllm_cuda_rms_norm): nobody fused it, run it now
        st->norm.pending = false;
        if (int e = fllm_rms_norm_launch(st->s, st->norm.x, st->norm.w, st->norm.out, st->norm.rows, st->norm.cols, st->norm.eps)) return e;
    }
    if (fllm_pending_attn.o16) {  // FP16 attention output nobody quantized directly
        const FllmPendingAttn p = fllm_pending_attn;
        fllm_pending_attn = FllmPendingAttn();
        if (int e = fllm_flashinfer_expand_half(st->s, p.o16, p.out, p.count)) return e;
    }
    if (!st->glu_pending) return 0;
    st->glu_pending = false;
    if (st->glu_half) return fllm_glu_launch_f16(st->s, st->glu_gate, st->glu_up, st->glu_out, st->glu_count, st->glu_mode);
    return fllm_glu_launch(st->s, st->glu_gate, st->glu_up, st->glu_out, st->glu_count, st->glu_mode);
}

FllmStream *fllm_cuda_stream_create(int dev) {
    if (cudaSetDevice(dev) != cudaSuccess) { fllm_set_error("cudaSetDevice(%d) failed", dev); return nullptr; }
    auto *st = new FllmStream{dev, nullptr, nullptr, 8u << 20, 0, nullptr, 0, nullptr, 0, {}, 0};
    if (cudaStreamCreateWithFlags(&st->s, cudaStreamNonBlocking) != cudaSuccess ||
        cudaMalloc(&st->arena, st->cap) != cudaSuccess) {
        fllm_set_error("cannot create CUDA stream: %s", cudaGetErrorString(cudaGetLastError()));
        delete st;
        return nullptr;
    }
    return st;
}
void fllm_cuda_stream_destroy(FllmStream *st) {
    if (!st) return;
    cudaSetDevice(st->dev);
    cudaStreamSynchronize(st->s);
    cudaFree(st->arena);
    for (void *p : st->retired_arenas) cudaFree(p);
    cudaFree(st->ws);
    cudaFree(st->x16);
        st->epoch++;
    cudaFree(st->q4z_io);
            st->epoch++;
    cudaFree(st->q4z_ctmp);
            st->epoch++;
    cudaFree(st->q4z_ws);
            st->epoch++;
    cudaFree(st->argmax_out);
    for (void *p : st->act_tmp) cudaFree(p);
    cudaFree(st->act_check);
    cudaFree(st->sample_scratch);
    for (auto *g : st->graphs) {
        if (g) {
            graph_free(g);
            delete g;
        }
    }
    cudaStreamDestroy(st->s);
    delete st;
}
// A forward pass of the decoder: the host arrays that its attention layers receive (block tables, lengths) do not change until fllm_cuda_step_end,
// so they are staged and analysed once instead of once per layer (FLLM_PREFILL_STEP_CACHE).
int fllm_cuda_step_begin(FllmStream *st) {
    st->step = FllmStream::Step{};
    st->step.open = true;
    return 0;
}
int fllm_cuda_step_end(FllmStream *st) {
    st->step = FllmStream::Step{};
    return 0;
}
int fllm_cuda_stream_sync(FllmStream *st) {
    CHECK(cudaSetDevice(st->dev), "cudaSetDevice");
    if (int e = flush_glu(st)) return e;
    CHECK(cudaStreamSynchronize(st->s), "cudaStreamSynchronize");
    return 0;
}
int fllm_cuda_stream_wait_stream(FllmStream *waiter, FllmStream *target) {
    if (!waiter || !target) return 0;
    if (waiter == target) return 0;
    CHECK(cudaSetDevice(waiter->dev), "cudaSetDevice");
    if (int e = flush_glu(waiter)) return e;
    if (int e = flush_glu(target)) return e;
    cudaEvent_t ev;
    if (cudaEventCreateWithFlags(&ev, cudaEventDisableTiming) != cudaSuccess) {
        return fllm_set_error("cudaEventCreate failed");
    }
    cudaEventRecord(ev, target->s);
    cudaStreamWaitEvent(waiter->s, ev, 0);
    cudaEventDestroy(ev);
    return fllm_check_launch("stream_wait_stream");
}

// Stage host data for a kernel. The arena is a ring: stream order guarantees that earlier kernels
// have finished before a later copy overwrites their metadata.
static int stage(FllmStream *st, int kind, const void *host, size_t bytes, void **dev) {
    if (st->capture) {
        // while a graph is captured the data goes through the fixed buffers of the graph (kind: 0 positions, 1 slots, 2 tables, 3 lengths)
        if (kind < 0 || kind > 3) return fllm_set_error("graph capture: this operation stages data that is not part of a decode step");
        GraphSlot &sl = st->capture->slot[kind];
        if (!sl.dev || bytes > sl.cap) return fllm_set_error("graph capture: staged data of kind %d does not fit its buffer", kind);
        if (!sl.staged) {
            memcpy(sl.pinned, host, bytes);
            sl.bytes = bytes;
            sl.staged = true;
            CHECK(cudaMemcpyAsync(sl.dev, sl.pinned, bytes, cudaMemcpyHostToDevice, st->s), "cudaMemcpyAsync(graph stage)");
        } else if (bytes != sl.bytes) {
            return fllm_set_error("graph capture: staged data of kind %d changes size", kind);
        }
        *dev = sl.dev;
        return 0;
    }
    const size_t need = (bytes + 255) & ~size_t(255);
    // a block larger than half the ring could be overwritten by the next stage of the same call (block tables of a
    // 512-row batch at 32k context are 8 MB; the lengths staged right after them would wrap onto them)
    if (need > st->cap / 2) {
        CHECK(cudaStreamSynchronize(st->s), "cudaStreamSynchronize");
        // not freed yet: the caller may hold a pointer staged into it a moment ago (the lengths of the attention call whose block tables
        // are staged now); freeing it made the next kernel read released memory ("illegal memory access" with many rows at long contexts)
        if (st->arena) st->retired_arenas.push_back(st->arena);
        st->epoch++;
        st->cap = need * 4;
        st->arena = nullptr;
        CHECK(cudaMalloc(&st->arena, st->cap), "cudaMalloc(arena)");
        st->off = 0;
        st->arena_gen++;
        for (auto &e : st->staged) e.dev = nullptr;
    }
    if (st->off + need > st->cap) {
        st->off = 0;
        st->arena_gen++;
        for (auto &e : st->staged) e.dev = nullptr;
    }
    for (auto &e : st->staged)
        if (e.dev && e.data.size() == bytes && memcmp(e.data.data(), host, bytes) == 0) {
            *dev = e.dev;
            return 0;
        }
    *dev = static_cast<char *>(st->arena) + st->off;
    st->off += need;
    CHECK(cudaMemcpyAsync(*dev, host, bytes, cudaMemcpyHostToDevice, st->s), "cudaMemcpyAsync(stage)");
    auto &e = st->staged[st->staged_next];
    st->staged_next = (st->staged_next + 1) % 8;
    e.data.assign(static_cast<const unsigned char *>(host), static_cast<const unsigned char *>(host) + bytes);
    e.dev = *dev;
    return 0;
}

// `stage` for host arrays that stay unchanged during one forward pass (the block tables and lengths its attention layers receive): the
// first layer stages them, the others take the device copy without comparing the (rows x stride) table with the staged one again.
static bool step_cached(const FllmStream *st) { return st->step.open && !st->capture && fllm_opt(FLLM_OPT_STEP_CACHE) != 0; }
static int stage_step(FllmStream *st, FllmStream::StepSlot &slot, int kind, const void *host, size_t bytes, void **dev) {
    if (step_cached(st)) {
        if (slot.dev && slot.host == host && slot.bytes == bytes && slot.gen == st->arena_gen) {
            *dev = slot.dev;
            return 0;
        }
        if (int e = stage(st, kind, host, bytes, dev)) return e;
        slot = FllmStream::StepSlot{host, bytes, st->arena_gen, *dev};
        return 0;
    }
    return stage(st, kind, host, bytes, dev);
}
// Rows [first, end) of a prompt that belong to one sequence: same block table, one more position per row. Computed once per forward pass.
static void step_runs(FllmStream *st, const int64_t *tables, const int64_t *lengths, int rows, int stride) {
    auto &s = st->step;
    if (s.run_tables == tables && s.run_lengths == lengths && s.run_rows == rows && s.run_stride == stride) return;
    s.runs.clear();
    s.compact_tables.clear();
    for (int r0 = 0; r0 < rows;) {
        int r1 = r0 + 1;
        while (r1 < rows && lengths[r1] == lengths[r1 - 1] + 1 &&
               memcmp(tables + (size_t)r1 * stride, tables + (size_t)r0 * stride, (size_t)stride * sizeof(int64_t)) == 0)
            r1++;
        s.runs.emplace_back(r0, r1);
        s.compact_tables.insert(s.compact_tables.end(), tables + (size_t)r0 * stride, tables + (size_t)(r0 + 1) * stride);
        r0 = r1;
    }
    s.run_tables = tables;
    s.run_lengths = lengths;
    s.run_rows = rows;
    s.run_stride = stride;
    s.compact = FllmStream::StepSlot{};
}
#define PREP(st) \
    do { \
        if (int e_ = flush_glu(st)) return e_; \
        (st)->fp4_ready = false; \
        (st)->q4z_ready = false; \
        (st)->x16_ready = false; \
        (st)->x8_ready = false; \
        CHECK(cudaSetDevice((st)->dev), "cudaSetDevice"); \
    } while (0)

int fllm_cuda_upload(FllmStream *st, void *dst, const void *src, size_t bytes) {
    PREP(st);
    const cudaError_t e = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, st->s);
    if (e != cudaSuccess) {
        // "invalid argument" seen on an AMD APU (HIP): say what the two pointers are
        cudaPointerAttributes a{}, b{};
        const cudaError_t pd = cudaPointerGetAttributes(&a, dst);
        cudaGetLastError();
        const cudaError_t ps = cudaPointerGetAttributes(&b, src);
        cudaGetLastError();
        return fllm_set_error("upload: %s (dst %p: %s type %d device %d, src %p: %s type %d, %zu bytes, stream device %d)", cudaGetErrorString(e), dst,
                              pd == cudaSuccess ? "known" : cudaGetErrorString(pd), pd == cudaSuccess ? (int)a.type : -1, pd == cudaSuccess ? a.device : -1,
                              src, ps == cudaSuccess ? "known" : cudaGetErrorString(ps), ps == cudaSuccess ? (int)b.type : -1, bytes, st->dev);
    }
    return 0;
}
int fllm_cuda_download(FllmStream *st, void *dst, const void *src, size_t bytes) {
    PREP(st);
    const cudaError_t e = cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, st->s);
    if (e != cudaSuccess) {
        // rare "invalid argument" seen on a Tesla T4 selftest (1 run in 3): say what the two pointers are
        cudaPointerAttributes a{};
        const cudaError_t pe = cudaPointerGetAttributes(&a, src);
        cudaGetLastError();
        return fllm_set_error("download: %s (src %p: %s type %d device %d, dst %p, %zu bytes, stream device %d)", cudaGetErrorString(e), src,
                              pe == cudaSuccess ? "known" : cudaGetErrorString(pe), pe == cudaSuccess ? (int)a.type : -1,
                              pe == cudaSuccess ? a.device : -1, dst, bytes, st->dev);
    }
    CHECK(cudaStreamSynchronize(st->s), "download sync");
    return 0;
}
int fllm_cuda_copy(FllmStream *st, void *dst, const void *src, size_t bytes) {
    PREP(st);
    CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, st->s), "copy");
    return 0;
}

// Can the GEMM of a (pending) glu's consumer read gate and up itself? Only the tensor-core paths that convert the
// activations as their first step qualify; everything else needs the glu result in memory.
static bool glu_fusable(FllmStream *st, int kind, const void *w, const float *x, const float *y, int cols, int batch) {
    if (!st->glu_pending || x != st->glu_out || (const void *)y == (const void *)x || (size_t)batch * cols != st->glu_count)
        return false;
    if (st->glu_half && kind != 40 && kind != 8) return false;
    if ((reinterpret_cast<uintptr_t>(st->glu_gate) & 15) != 0 || (reinterpret_cast<uintptr_t>(st->glu_up) & 15) != 0 ||
        (reinterpret_cast<uintptr_t>(w) & 15) != 0)
        return false;
    const bool narrow_ok = kind == 2 || kind == 3 || kind == 6 || kind == 7 || kind == 8 || kind == 20 || kind == 39;
    if ((cols & (narrow_ok ? 31 : 255)) != 0) return false;
    if (kind == 40) return true;
    static const bool no_qgemm = getenv("FLLM_CUDA_NO_QGEMM") != nullptr;
    // Turing (sm_75): the tile shapes of the quantized GEMMs do not all fit its shared memory, and a GEMM that gives up after the fused conversion hands the raw
    // gate to the next kernel (checked on a T4: Llama-3.2-3B Q4_K_M wrote nonsense, the same run was right with FLLM_CUDA_NO_GLU_FUSE=1)
    if (batch < 16 || no_qgemm || fllm_exact_math() || fllm_cc() < 80) return false;
    switch (kind) {
        case 2: case 3: case 6: case 7: case 8: case 12: case 13: case 14: case 20: return true;
        default: return false;
    }
}

// A pending gated activation feeding a K-quant GEMV: the int8 quantization of its input computes the activation itself (FLLM_FUSE_GLU_Q8)
static bool glu_q8_fusable(FllmStream *st, int kind, const void *w, const float *x, const float *y, int cols, int batch) {
#if defined(__HIPCC__)
    (void)st; (void)kind; (void)w; (void)x; (void)y; (void)cols; (void)batch;
    return false;
#else
    if (!st->glu_pending || x != st->glu_out || (const void *)y == (const void *)x || (size_t)batch * cols != st->glu_count)
        return false;
    if (!fllm_fuse_enabled(FLLM_FUSE_GLU_Q8)) return false;
    if ((reinterpret_cast<uintptr_t>(st->glu_gate) & 15) != 0 || (reinterpret_cast<uintptr_t>(st->glu_up) & 15) != 0 ||
        (reinterpret_cast<uintptr_t>(w) & 15) != 0)
        return false;
    return fllm_mmvq_prep_ok(kind, cols, batch);
#endif
}

int fllm_cuda_linear(FllmStream *st, const void *w, int kind, const float *x, float *y, const float *bias, int rows,
                     int cols, int batch) {
    CHECK(cudaSetDevice(st->dev), "cudaSetDevice");
    if (int e = flush_add(st)) return e;
    st->q4z_ready = false;
    st->fp4_ready = false;
    const bool fuse = glu_fusable(st, kind, w, x, y, cols, batch) || glu_q8_fusable(st, kind, w, x, y, cols, batch);
    if (!fuse) {
        if (int e = flush_glu(st)) return e;
    }
    // activations as FP16 (256-byte aligned) plus room for split-K partials of narrow batches
    const size_t need = (((size_t)batch * pad256(cols) * 2 + 255) & ~size_t(255)) + (batch <= 64 ? (size_t)4 << 20 : (size_t)16 << 20);
    if (need > st->x16_bytes && !fllm_exact_math()) {
        CHECK(cudaStreamSynchronize(st->s), "cudaStreamSynchronize");
        cudaFree(st->x16);
        st->epoch++;
        st->x16 = nullptr;
        st->x16_bytes = (need * 2 + 255) & ~size_t(255);
        CHECK(cudaMalloc(&st->x16, st->x16_bytes), "cudaMalloc(activation scratch)");
        st->x16_ready = false;
        st->x8_ready = false;
    }
    static const bool no_reuse = getenv("FLLM_CUDA_NO_X16_REUSE") != nullptr;
    fllm_x16_reuse = !fuse && !no_reuse && st->x16_ready && st->x16_x == x && st->x16_batch == batch && st->x16_cols == cols;
    fllm_x8_reuse = !fuse && !no_reuse && st->x8_ready && st->x8_x == x && st->x8_batch == batch && st->x8_cols == cols;
    fllm_x16_filled = false;
    fllm_x8_filled = false;
    if (fuse) {
        fllm_glu_up = st->glu_up;
        fllm_glu_dst = const_cast<float *>(x);
        fllm_glu_mode = st->glu_mode;
        st->glu_pending = false;
    }
    const int rc = fllm_linear_launch(st->s, w, kind, fuse ? st->glu_gate : x, y, bias, rows, cols, batch, st->x16, st->x16_bytes);
    fllm_glu_up = nullptr;
    fllm_glu_dst = nullptr;
    fllm_x16_reuse = false;
    fllm_x8_reuse = false;
    st->x16_x = x;
    st->x16_batch = batch;
    st->x16_cols = cols;
    st->x16_ready = fllm_x16_filled && (const void *)y != (const void *)x;
    st->x8_x = x;
    st->x8_batch = batch;
    st->x8_cols = cols;
    st->x8_ready = fllm_x8_filled && (const void *)y != (const void *)x;
    return rc;
}

// Packed EXL2 / EXL3 weights (exl.cu): CUDA-core kernels that decode the codes while multiplying. Any pending fused work is flushed first, the
// activation caches are invalidated (the scratch of EXL3 is the FP16 activation buffer).
int fllm_cuda_linear_exl(FllmStream *st, const void *w, int kind, const float *x, float *y, const float *bias, int rows, int cols, int batch) {
    CHECK(cudaSetDevice(st->dev), "cudaSetDevice");
    if (int e = flush_glu(st)) return e;
    st->q4z_ready = false;
    st->fp4_ready = false;
    st->x16_ready = false;
    st->x8_ready = false;
    const size_t need = kind == (int)0x80000031 ? fllm_exl3_scratch_bytes(rows, cols, batch) : 0;
    if (need > st->x16_bytes) {
        if (st->capture) {
            return fllm_set_error("EXL3 linear: scratch buffer must not reallocate during graph capture (needs %zu, has %zu)", need, st->x16_bytes);
        }
        CHECK(cudaStreamSynchronize(st->s), "cudaStreamSynchronize");
        cudaFree(st->x16);
        st->epoch++;
        st->x16 = nullptr;
        st->x16_bytes = (need * 2 + 255) & ~size_t(255);
        CHECK(cudaMalloc(&st->x16, st->x16_bytes), "cudaMalloc(activation scratch)");
    }
    return fllm_exl_launch(st->s, kind, w, x, y, bias, rows, cols, batch, st->x16, st->x16_bytes);
}
// Switch the accumulate mode of the Marlin 4-bit GEMM on or off (vLLM fuses the residual add into the norm that follows; here it is
// the conversion of the FP16 result that does it).
int fllm_cuda_q4z_accumulate(int on, float scale) {
    fllm_q4z_acc = on != 0;
    fllm_q4z_acc_scale = scale;
    return 0;
}
// 4-bit group-quantized weights (Q4Z) through the Marlin GEMM; `w` is the device layout built by fllm_q4z_repack and the host
int fllm_cuda_linear_q4z(FllmStream *st, const void *w, int group, const float *x, float *y, const float *bias, int rows, int cols,
                         int batch, const int *perm) {
    CHECK(cudaSetDevice(st->dev), "cudaSetDevice");
    if (int e = flush_add(st)) return e;
    // a pending gated activation feeding this GEMM is folded into the conversion of the activations to FP16
    static const bool no_glu_fuse = getenv("FLLM_CUDA_NO_GLU_FUSE") != nullptr;
    // (an FP16 gate and up are folded in only without an activation permutation: that conversion kernel reads FP32)
    const bool fuse = !no_glu_fuse && st->glu_pending && (!st->glu_half || perm == nullptr) && x == st->glu_out && (const void *)y != (const void *)x &&
                      (size_t)batch * cols == st->glu_count && (reinterpret_cast<uintptr_t>(st->glu_gate) & 15) == 0 &&
                      (reinterpret_cast<uintptr_t>(st->glu_up) & 15) == 0;
    if (!fuse) {
        if (int e = flush_glu(st)) return e;
    }
    st->fp4_ready = false;
    st->x16_ready = false;
    st->x8_ready = false;
    if (!fllm_q4z_supported(rows, cols, group))
        return fllm_set_error("Marlin 4-bit GEMM unsupported for %d x %d (group %d)", rows, cols, group);
    if (st->sms == 0) {
        cudaDeviceProp prop;
        CHECK(cudaGetDeviceProperties(&prop, st->dev), "cudaGetDeviceProperties");
        st->sms = prop.multiProcessorCount;
    }
    const size_t io = fllm_q4z_io_bytes(rows, cols, batch), ctmp = fllm_q4z_ctmp_bytes(batch, st->sms), ws = fllm_q4z_workspace_bytes(rows);
    if (io > st->q4z_io_bytes || ctmp > st->q4z_ctmp_bytes || ws > st->q4z_ws_bytes) {
        CHECK(cudaStreamSynchronize(st->s), "cudaStreamSynchronize");
        if (io > st->q4z_io_bytes) {
            cudaFree(st->q4z_io);
            st->epoch++;
            st->q4z_io = nullptr;
            st->q4z_io_bytes = io * 2;
            st->q4z_ready = false;
            CHECK(cudaMalloc(&st->q4z_io, st->q4z_io_bytes), "cudaMalloc(Marlin activations)");
        }
        if (ctmp > st->q4z_ctmp_bytes) {
            cudaFree(st->q4z_ctmp);
            st->epoch++;
            st->q4z_ctmp = nullptr;
            st->q4z_ctmp_bytes = ctmp;
            CHECK(cudaMalloc(&st->q4z_ctmp, ctmp), "cudaMalloc(Marlin partials)");
        }
        if (ws > st->q4z_ws_bytes) {
            cudaFree(st->q4z_ws);
            st->epoch++;
            st->q4z_ws = nullptr;
            st->q4z_ws_bytes = ws * 2;
            CHECK(cudaMalloc(&st->q4z_ws, st->q4z_ws_bytes), "cudaMalloc(Marlin locks)");
            CHECK(cudaMemsetAsync(st->q4z_ws, 0, st->q4z_ws_bytes, st->s), "cudaMemset(Marlin locks)");
        }
    }
    static const bool no_reuse = getenv("FLLM_CUDA_NO_X16_REUSE") != nullptr;
    const bool reuse = !fuse && !no_reuse && st->q4z_ready && st->q4z_x == x && st->q4z_batch == batch && st->q4z_cols == cols && st->q4z_perm == perm;
    fllm_glu_half = fuse && st->glu_half;
    // the residual add of an output projection waits for the RMS norm that follows (see PendingAdd)
    const bool defer = fllm_q4z_acc && !fllm_q4z_f16out && batch >= FLLM_PREFILL_MIN_ROWS && !st->capture && !st->add.on && fllm_opt(FLLM_OPT_FUSE_ADD_NORM) != 0;
    fllm_q4z_defer_acc = defer;
    fllm_q4z_deferred = false;
    const int rc = fllm_q4z_launch(st->s, st->dev, st->sms, w, group, x, y, bias, rows, cols, batch, st->q4z_io, st->q4z_ctmp, st->q4z_ws,
                                   fuse ? st->glu_gate : nullptr, fuse ? st->glu_up : nullptr, st->glu_mode, reuse, perm);
    fllm_glu_half = false;
    fllm_q4z_defer_acc = false;
    if (rc == 0 && fllm_q4z_deferred) {
        st->add = FllmStream::PendingAdd{true, fllm_q4z_y16, bias, y, rows, batch, cols, fllm_q4z_acc_scale};
        fllm_q4z_deferred = false;
    }
    if (fuse) st->glu_pending = false;
    st->q4z_x = x;
    st->q4z_batch = batch;
    st->q4z_cols = cols;
    st->q4z_perm = perm;
    st->q4z_ready = rc == 0 && (const void *)y != (const void *)x;
    if (rc == -1 || rc == -2)
        return fllm_set_error("Marlin 4-bit GEMM rejected %d x %d, batch %d, group %d (alignment or configuration)", rows, cols, batch, group);
    return rc;
}
// Marlin tiles from the AutoAWQ-ordered int4 matrix `awq_words` [k, n / 8] (both device pointers), synchronous
int fllm_cuda_q4z_repack(int dev, const void *awq_words, void *out, int k, int n) { return fllm_q4z_repack(dev, awq_words, out, k, n); }
int fllm_cuda_gptq_repack(int dev, const void *words, void *out, int k, int n) { return fllm_q4z_gptq_repack(dev, words, out, k, n); }
int fllm_cuda_q4z_supported(int rows, int cols, int group) { return fllm_q4z_supported(rows, cols, group) ? 1 : 0; }
int fllm_cuda_fp8_marlin_supported(int rows, int cols) { return fllm_fp8_marlin_supported(rows, cols) ? 1 : 0; }
int fllm_cuda_fp8_marlin_repack(int dev, const void *words, void *out, int k, int n) { return fllm_fp8_marlin_repack(dev, words, out, k, n); }
int fllm_cuda_linear_fp8_marlin(FllmStream *st, const void *w, const float *x, float *y, const float *bias, int rows, int cols, int batch) {
    CHECK(cudaSetDevice(st->dev), "cudaSetDevice");
    if (int e = flush_add(st)) return e;
    static const bool no_glu_fuse = getenv("FLLM_CUDA_NO_GLU_FUSE") != nullptr;
    const bool fuse = !no_glu_fuse && st->glu_pending && !st->glu_half && x == st->glu_out && (const void *)y != (const void *)x &&
                      (size_t)batch * cols == st->glu_count && (reinterpret_cast<uintptr_t>(st->glu_gate) & 15) == 0 &&
                      (reinterpret_cast<uintptr_t>(st->glu_up) & 15) == 0;
    if (!fuse) {
        if (int e = flush_glu(st)) return e;
    }
    st->fp4_ready = false;
    st->x16_ready = false;
    st->x8_ready = false;
    if (!fllm_fp8_marlin_supported(rows, cols))
        return fllm_set_error("Marlin FP8 GEMM unsupported for %d x %d", rows, cols);
    if (st->sms == 0) {
        cudaDeviceProp prop;
        CHECK(cudaGetDeviceProperties(&prop, st->dev), "cudaGetDeviceProperties");
        st->sms = prop.multiProcessorCount;
    }
    const size_t io = fllm_q4z_io_bytes(rows, cols, batch), ctmp = fllm_q4z_ctmp_bytes(batch, st->sms), ws = fllm_q4z_workspace_bytes(rows);
    if (io > st->q4z_io_bytes || ctmp > st->q4z_ctmp_bytes || ws > st->q4z_ws_bytes) {
        CHECK(cudaStreamSynchronize(st->s), "cudaStreamSynchronize");
        if (io > st->q4z_io_bytes) {
            cudaFree(st->q4z_io);
            st->epoch++;
            st->q4z_io = nullptr;
            st->q4z_io_bytes = io * 2;
            st->q4z_ready = false;
            CHECK(cudaMalloc(&st->q4z_io, st->q4z_io_bytes), "cudaMalloc(Marlin activations)");
        }
        if (ctmp > st->q4z_ctmp_bytes) {
            cudaFree(st->q4z_ctmp);
            st->epoch++;
            st->q4z_ctmp = nullptr;
            st->q4z_ctmp_bytes = ctmp;
            CHECK(cudaMalloc(&st->q4z_ctmp, ctmp), "cudaMalloc(Marlin partials)");
        }
        if (ws > st->q4z_ws_bytes) {
            cudaFree(st->q4z_ws);
            st->epoch++;
            st->q4z_ws = nullptr;
            st->q4z_ws_bytes = ws * 2;
            CHECK(cudaMalloc(&st->q4z_ws, st->q4z_ws_bytes), "cudaMalloc(Marlin locks)");
            CHECK(cudaMemsetAsync(st->q4z_ws, 0, st->q4z_ws_bytes, st->s), "cudaMemset(Marlin locks)");
        }
    }
    static const bool no_reuse = getenv("FLLM_CUDA_NO_X16_REUSE") != nullptr;
    const bool reuse = !fuse && !no_reuse && st->q4z_ready && st->q4z_x == x && st->q4z_batch == batch && st->q4z_cols == cols && st->q4z_perm == nullptr;
    const int rc = fllm_fp8_marlin_launch(st->s, st->dev, st->sms, w, x, y, bias, rows, cols, batch, st->q4z_io, st->q4z_ctmp, st->q4z_ws,
                                          fuse ? st->glu_gate : nullptr, fuse ? st->glu_up : nullptr, st->glu_mode, reuse);
    if (fuse) st->glu_pending = false;
    st->q4z_x = x;
    st->q4z_batch = batch;
    st->q4z_cols = cols;
    st->q4z_perm = nullptr;
    st->q4z_ready = rc == 0 && (const void *)y != (const void *)x;
    if (rc == -1 || rc == -2)
        return fllm_set_error("Marlin FP8 GEMM rejected %d x %d, batch %d (alignment or configuration)", rows, cols, batch);
    return rc;
}
// ---- FP8 / int8 W8A8 GEMMs (fp8_gemm.cu, int8_gemm.cu) ----
// 1 when an FP8M weight of rows x cols on device `dev` is to be uploaded in the W8A8 layout (row-major E4M3 bytes, FP32 scales)
int fllm_cuda_fp8_gemm_enabled(int dev, int rows, int cols) {
#if defined(__HIPCC__)
    (void)dev; (void)rows; (void)cols;
    return 0;
#else
    if (cudaSetDevice(dev) != cudaSuccess) return 0;
    return fllm_fp8_w8a8_enabled(rows, cols) ? 1 : 0;
#endif
}
// mode: -1 environment (FLLM_FP8_GEMM), 0 off, 1 mma.sync, 2 CUTLASS; the self test flips it before it uploads a weight
int fllm_cuda_fp8_gemm_set_mode(int mode) {
#if !defined(__HIPCC__)
    fllm_fp8_gemm_set_mode(mode);
#else
    (void)mode;
#endif
    return 0;
}
int fllm_cuda_int8_gemm_set_mode(int mode) {
#if !defined(__HIPCC__)
    fllm_int8_gemm_set_mode(mode);
#else
    (void)mode;
#endif
    return 0;
}
int fllm_cuda_cublaslt_set_mode(int mode) {
#if !defined(__HIPCC__)
    fllm_cublaslt_set_mode(mode);
#else
    (void)mode;
#endif
    return 0;
}
// W8A8 product of FP8 weights in the W8A8 layout; wblock, amode: see fllm_fp8_w8a8_gemm
int fllm_cuda_linear_fp8_w8a8(FllmStream *st, const void *w, const float *wscale, int wblock, const float *x, float *y, const float *bias, int rows,
                              int cols, int batch, int amode, float static_scale) {
#if defined(__HIPCC__)
    (void)st; (void)w; (void)wscale; (void)wblock; (void)x; (void)y; (void)bias; (void)rows; (void)cols; (void)batch; (void)amode; (void)static_scale;
    return fllm_set_error("FP8 W8A8 GEMM is not available in the ROCm build");
#else
    PREP(st);
    const int rc = fllm_fp8_w8a8_gemm(st->s, w, wscale, wblock, x, y, bias, rows, cols, batch, amode, static_scale);
    if (rc < 0) return fllm_set_error("FP8 W8A8 GEMM rejected %d x %d, batch %d (device, alignment or scale mode)", rows, cols, batch);
    return rc;
#endif
}
// Block-wise FP8 weight (FP8B; fp8_block.cu): `w` is the device buffer of the weight, tiles of 2^lbr rows x 2^lbc columns
int fllm_cuda_linear_fp8_block(FllmStream *st, const void *w, const float *x, float *y, const float *bias, int rows, int cols, int batch, int lbr,
                               int lbc) {
#if defined(__HIPCC__)
    (void)st; (void)w; (void)x; (void)y; (void)bias; (void)rows; (void)cols; (void)batch; (void)lbr; (void)lbc;
    return fllm_set_error("block-wise FP8 weights are not available in the ROCm build");
#else
    PREP(st);
    return fllm_fp8_block_gemm(st->s, w, x, y, bias, rows, cols, batch, lbr, lbc);
#endif
}
// int8 weights [rows][cols] with one FP32 scale per row, per-token activations (the SmoothQuant / compressed-tensors layout)
int fllm_cuda_linear_int8_plain(FllmStream *st, const void *w, const float *wscale, const float *x, float *y, const float *bias, int rows, int cols,
                                int batch) {
#if defined(__HIPCC__)
    (void)st; (void)w; (void)wscale; (void)x; (void)y; (void)bias; (void)rows; (void)cols; (void)batch;
    return fllm_set_error("int8 W8A8 GEMM is not available in the ROCm build");
#else
    PREP(st);
    const int rc = fllm_int8_plain_gemm(st->s, w, wscale, x, y, bias, rows, cols, batch);
    if (rc < 0) return fllm_set_error("int8 W8A8 GEMM rejected %d x %d, batch %d (device or alignment)", rows, cols, batch);
    return rc;
#endif
}
int fllm_cuda_linear_nvfp4(FllmStream *st, const void *w, const float *x, float *y, const float *bias, float gscale,
                           int rows, int cols, int batch) {
    CHECK(cudaSetDevice(st->dev), "cudaSetDevice");
    if (int e = flush_add(st)) return e;
    st->q4z_ready = false;
    // a deferred RMSNorm / FP16 attention output whose result this GEMM reads: fused into the activation quantization when the CUTLASS GEMM takes the product
    const int cutlass_mode = (fllm_fp4_acc ? 1 : 0) | (fllm_fp4_f16out ? 2 : 0);
    const bool glu_first = st->glu_pending;
    bool norm_fused = false;
    const void *attn_half = nullptr;
    if (!glu_first && st->norm.pending && st->norm.out == x && st->norm.rows == batch && st->norm.cols == cols && bias == nullptr) {
        st->norm.pending = false;
        norm_fused = fllm_nvfp4_norm_quant_try(st->s, st->norm.x, st->norm.w, st->norm.out, batch, cols, st->norm.eps, w, rows, cutlass_mode) != 0;
        if (!norm_fused) {
            if (int e = fllm_rms_norm_launch(st->s, st->norm.x, st->norm.w, st->norm.out, st->norm.rows, st->norm.cols, st->norm.eps)) return e;
        } else {
            st->fp4_ready = true;
            st->fp4_x = x;
            st->fp4_batch = batch;
            st->fp4_cols = cols;
        }
    } else if (!glu_first && fllm_pending_attn.o16 && fllm_pending_attn.out == x && fllm_pending_attn.count == (size_t)batch * cols && bias == nullptr &&
               fllm_nvfp4_cutlass_ok(x, y, bias, rows, cols, batch, cutlass_mode)) {
        attn_half = fllm_pending_attn.o16;
        fllm_pending_attn = FllmPendingAttn();
    }
    bool fuse = glu_fusable(st, 40, w, x, y, cols, batch);
    // an FP16 glu is only folded into the CUTLASS GEMM (the built-in kernels read FP32): any other case computes it first
    if (fuse && st->glu_half && !fllm_nvfp4_cutlass_ok(st->glu_gate, y, bias, rows, cols, batch, fllm_fp4_acc ? 1 : 0)) fuse = false;
    if (!fuse) {
        if (int e = flush_glu(st)) return e;
    }
    // quantized activations (nibbles + one scale byte per 16 values) and split-K partials of narrow batches
    const size_t need = (((size_t)batch * (cols >> 1) + 255) & ~size_t(255)) + (((size_t)batch * (cols >> 4) + 255) & ~size_t(255)) +
                        (batch <= 64 ? (size_t)4 << 20 : (size_t)16 << 20);
    if (need > st->x16_bytes) {
        if (st->capture) {
            return fllm_set_error("cuda_moe_linear: scratch buffer must not reallocate during graph capture (needs %zu, has %zu)", need, st->x16_bytes);
        }
        CHECK(cudaStreamSynchronize(st->s), "cudaStreamSynchronize");
        cudaFree(st->x16);
        st->epoch++;
        st->x16 = nullptr;
        st->x16_bytes = (need * 2 + 255) & ~size_t(255);
        CHECK(cudaMalloc(&st->x16, st->x16_bytes), "cudaMalloc(activation scratch)");
        st->fp4_ready = false;
    }
    st->x16_ready = false;
    st->x8_ready = false;
    const bool reuse = !fuse && st->fp4_ready && st->fp4_x == x && st->fp4_batch == batch && st->fp4_cols == cols;
    if (fuse) {
        fllm_glu_up = st->glu_up;
        fllm_glu_mode = st->glu_mode;
        fllm_glu_half = st->glu_half;
        st->glu_pending = false;
    }
    fllm_x_half = attn_half;
    const bool ok = fllm_nvfp4_try(st->s, w, fuse ? st->glu_gate : x, y, bias, gscale, rows, cols, batch, st->x16, st->x16_bytes, reuse ? 0 : 1) != 0;
    fllm_x_half = nullptr;
    fllm_glu_up = nullptr;
    fllm_glu_half = false;
    if (!ok)
        return fllm_set_error("NVFP4 GEMM unsupported for %d x %d (needs sm_12x and k %% 256 == 0)", rows, cols);
    st->fp4_x = x;
    st->fp4_batch = batch;
    st->fp4_cols = cols;
    st->fp4_ready = (const void *)y != (const void *)x;
    return fllm_check_launch("nvfp4");
}
int fllm_cuda_embedding(FllmStream *st, const void *w, int kind, const int64_t *ids, int n, float *out, int cols) {
    PREP(st);
    void *d = nullptr;
    if (int e = stage(st, -1, ids, (size_t)n * sizeof(int64_t), &d)) return e;
    return fllm_embedding_launch(st->s, w, kind, static_cast<const int64_t *>(d), out, cols, n);
}
// ---- CUDA graphs: a decode step is captured once per batch size and replayed (vLLM captures its forward the same way, with fixed input
// buffers; here the buffers are the four staging slots above). The kernels of a step are about 700 small launches; one graph launch
// replaces them.
static void graph_free(FllmGraph *g) {
    if (g->exec) cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(g->exec));
    g->exec = nullptr;
    for (auto &sl : g->slot) {
        cudaFree(sl.dev);
        cudaFreeHost(sl.pinned);
        sl = GraphSlot();
    }
}
int fllm_cuda_graph_supported(FllmStream *st) {
    int v = 0;
#if defined(__HIPCC__)
    // hipGraph capture of the decode step has never run on an AMD GPU (test level L0): opt in with FLLM_HIP_GRAPH=1 (the Rust side then
    // follows the rules of the CUDA graphs, FLLM_CUDA_GRAPH=0 still switches them off)
    const char *opt = getenv("FLLM_HIP_GRAPH");
    return opt && opt[0] == '1' && hipRuntimeGetVersion(&v) == hipSuccess && v >= 60000000 ? 1 : 0;
#else
    return cudaDriverGetVersion(&v) == cudaSuccess && v >= 12000 ? 1 : 0;
#endif
}
// Start recording the work of the stream into graph `id`. `sizes` are the byte sizes of the four staged arrays; `max_len` is the
// context length the attention kernels are sized for.
int fllm_cuda_graph_begin(FllmStream *st, int id, size_t pos_bytes, size_t slot_bytes, size_t table_bytes, size_t len_bytes, int max_len) {
    CHECK(cudaSetDevice(st->dev), "cudaSetDevice");
    if (st->capture) return fllm_set_error("a graph capture is already running");
    if (id < 0) return fllm_set_error("invalid graph id");
    if (int e = flush_glu(st)) return e;
    CHECK(cudaStreamSynchronize(st->s), "cudaStreamSynchronize");
    if ((size_t)id >= st->graphs.size()) st->graphs.resize((size_t)id + 1, nullptr);
    if (!st->graphs[id]) st->graphs[id] = new FllmGraph();
    FllmGraph *g = st->graphs[id];
    graph_free(g);
    const size_t sizes[4] = {pos_bytes, slot_bytes, table_bytes, len_bytes};
    for (int k = 0; k < 4; k++) {
        const size_t n = sizes[k] > 16 ? sizes[k] : 16;
        CHECK(cudaMalloc(&g->slot[k].dev, n), "cudaMalloc(graph slot)");
        CHECK(cudaHostAlloc(&g->slot[k].pinned, n, cudaHostAllocDefault), "cudaHostAlloc(graph slot)");
        g->slot[k].cap = n;
    }
    g->max_len = max_len;
    // nothing cached from earlier calls may be baked into the graph
    st->fp4_ready = st->x16_ready = st->x8_ready = st->q4z_ready = false;
    cudaError_t e = cudaStreamBeginCapture(st->s, cudaStreamCaptureModeThreadLocal);
    if (e != cudaSuccess) {
        graph_free(g);
        return fllm_set_error("cudaStreamBeginCapture: %s", cudaGetErrorString(e));
    }
    st->capture = g;
    return 0;
}
// Finish (or, with `abort`, drop) the capture and make the graph launchable.
int fllm_cuda_graph_end(FllmStream *st, int id, int abort) {
    FllmGraph *g = st->capture;
    if (!g || id < 0 || (size_t)id >= st->graphs.size() || st->graphs[id] != g) return fllm_set_error("no graph capture is running for id %d", id);
    int rc = abort ? 0 : flush_glu(st);
    st->glu_pending = false;
    cudaGraph_t graph = nullptr;
    cudaError_t e = cudaStreamEndCapture(st->s, &graph);
    st->capture = nullptr;
    st->fp4_ready = st->x16_ready = st->x8_ready = st->q4z_ready = false;
    if (abort || rc != 0 || e != cudaSuccess) {
        if (graph) cudaGraphDestroy(graph);
        graph_free(g);
        cudaGetLastError();
        if (abort) return 0;
        return rc != 0 ? rc : fllm_set_error("graph capture failed: %s", cudaGetErrorString(e));
    }
    cudaGraphExec_t exec = nullptr;
    e = cudaGraphInstantiate(&exec, graph, 0);
    cudaGraphDestroy(graph);
    if (e != cudaSuccess) {
        graph_free(g);
        return fllm_set_error("cudaGraphInstantiate: %s", cudaGetErrorString(e));
    }
    g->exec = exec;
    g->epoch = st->epoch;
    return 0;
}
// New contents of one staged array for the next launch (same size as during the capture).
int fllm_cuda_graph_set(FllmStream *st, int id, int kind, const void *data, size_t bytes) {
    if (id < 0 || (size_t)id >= st->graphs.size() || !st->graphs[id] || !st->graphs[id]->exec || kind < 0 || kind > 3)
        return fllm_set_error("graph %d is not ready", id);
    GraphSlot &sl = st->graphs[id]->slot[kind];
    if (bytes != sl.bytes) return fllm_set_error("graph %d: array %d has %zu bytes, %zu were captured", id, kind, bytes, sl.bytes);
    memcpy(sl.pinned, data, bytes);
    return 0;
}
// Returns 0 on success and 2 when the graph is stale (a buffer it points to was reallocated): capture it again.
int fllm_cuda_graph_launch(FllmStream *st, int id) {
    CHECK(cudaSetDevice(st->dev), "cudaSetDevice");
    if (id < 0 || (size_t)id >= st->graphs.size() || !st->graphs[id] || !st->graphs[id]->exec) return fllm_set_error("graph %d is not ready", id);
    FllmGraph *g = st->graphs[id];
    if (g->epoch != st->epoch) return 2;
    if (int e = flush_glu(st)) return e;
    st->fp4_ready = st->x16_ready = st->x8_ready = st->q4z_ready = false;
    CHECK(cudaGraphLaunch(static_cast<cudaGraphExec_t>(g->exec), st->s), "cudaGraphLaunch");
    return 0;
}

// Switch the accumulate mode of the NVFP4 CUTLASS GEMM on or off: the next product is multiplied by `scale` and added to y (the residual
// stream) by the epilogue. The caller asks fllm_cuda_nvfp4_cutlass_ok first; a GEMM that cannot do it fails instead of overwriting y.
int fllm_cuda_nvfp4_accumulate(int on, float scale) {
    fllm_fp4_acc = on != 0;
    fllm_fp4_acc_scale = scale;
    return 0;
}
int fllm_cuda_nvfp4_f16out(int on) {
    fllm_fp4_f16out = on != 0;
    return 0;
}
// mode: bit 0 = the product is added to y, bit 1 = the result is FP16 (see fllm_nvfp4_cutlass_ok)
int fllm_cuda_nvfp4_cutlass_ok(const float *x, const float *y, const float *bias, int rows, int cols, int batch, int mode) {
    return fllm_nvfp4_cutlass_ok(x, y, bias, rows, cols, batch, mode);
}
int fllm_cuda_q4z_f16out(int on) {
    fllm_q4z_f16out = on != 0;
    return 0;
}
// Bias, rotation and cache write of the FP16 q/k/v left by three Marlin GEMMs (see rope_kv_f16_kernel).
int fllm_cuda_rope_kv_f16(FllmStream *st, const uint16_t *q16, const uint16_t *k16, const uint16_t *v16, const float *qb, const float *kb,
                          const float *vb, float *q_out, uint16_t *pool_k_layer, uint16_t *pool_v_layer, const int32_t *positions,
                          const int32_t *slots, int rows, const float *cs, const float *sn, int heads, int kv_heads, int dk, int dv,
                          int half_, int neox) {
    PREP(st);
    void *dp = nullptr, *ds = nullptr;
    if (int e = stage(st, 0, positions, (size_t)rows * sizeof(int32_t), &dp)) return e;
    if (int e = stage(st, 1, slots, (size_t)rows * sizeof(int32_t), &ds)) return e;
    return fllm_rope_kv_f16_launch(st->s, q16, k16, v16, qb, kb, vb, q_out, pool_k_layer, pool_v_layer, static_cast<const int32_t *>(dp),
                                   static_cast<const int32_t *>(ds), cs, sn, rows, heads, kv_heads, dk, dv, half_, neox);
}
#if !defined(__HIPCC__)
// ---- fused decode kernels (fused.cu): 3 = declined (nothing was launched, the caller runs the unfused kernels), 0 = done
static const int FLLM_DECLINED = 3;

// Where the quantized activations of the GEMM that follows go (q0, q1) and which format: the quant argument if it applies, else 0.
static int fused_quant_target(FllmStream *st, int quant, int kind, int rows, int cols, void **q0, void **q1, const void *next_w = nullptr,
                              int next_rows = 0, const float **gs = nullptr) {
    if (quant == 0 || !fllm_fuse_enabled(FLLM_FUSE_NORM_QUANT)) return 0;
    if (quant == 1) {
        if (!st->q4z_io || (size_t)rows * cols * 2 > st->q4z_io_bytes) return 0;
        *q0 = st->q4z_io;
        return 1;
    }
    if (!st->x16) return 0;
    if (quant == 2) {
        if (!fllm_nvfp4_prep_ok(cols, rows)) return 0;
        // the activation factor of the weight that reads the result (FLLM_NVFP4_ACT_SCALE): the fused kernel must quantize exactly like nvfp4_quantize_kernel
        const float *gs_dev = nullptr;
        if (gs == nullptr || !fllm_nvfp4_prep_scale(next_w, next_rows, cols, &gs_dev)) return 0;
        *gs = gs_dev;
        const size_t xq = ((size_t)rows * (cols >> 1) + 255) & ~size_t(255), xs = ((size_t)rows * (cols >> 4) + 255) & ~size_t(255);
        if (xq + xs > st->x16_bytes) return 0;
        *q0 = st->x16;
        *q1 = static_cast<char *>(st->x16) + xq;
        return 2;
    }
    if (quant == 3) {
        if (!fllm_mmvq_prep_ok(kind, cols, rows)) return 0;
        const size_t xq = ((size_t)rows * cols + 255) & ~size_t(255);
        if (xq + (size_t)rows * (cols / 32) * sizeof(float2) > st->x16_bytes) return 0;
        *q0 = st->x16;
        *q1 = static_cast<char *>(st->x16) + xq;
        return 3;
    }
    return 0;
}
// the next GEMM finds its activations prepared when it reads `out`
static void fused_quant_done(FllmStream *st, int quant, const float *out, int rows, int cols) {
    if (quant == 1) {
        st->q4z_x = out;
        st->q4z_batch = rows;
        st->q4z_cols = cols;
        st->q4z_perm = nullptr;
        st->q4z_ready = true;
    } else if (quant == 2) {
        st->fp4_x = out;
        st->fp4_batch = rows;
        st->fp4_cols = cols;
        st->fp4_ready = true;
        fllm_nvfp4_scratch_filled();
    } else if (quant == 3) {
        st->x8_x = out;
        st->x8_batch = rows;
        st->x8_cols = cols;
        st->x8_ready = true;
    }
}

// x += scale * branch; out = rms_norm(x) * w; the activations of the next GEMM quantized (see fused.cu)
int fllm_cuda_add_rms_norm(FllmStream *st, float *x, const float *branch, float scale, const float *w, float *out, int rows, int cols,
                           float eps, int quant, int kind, const void *next_w, int next_rows) {
    // decode batches only: from FLLM_PREFILL_MIN_ROWS rows on, the prompt path (rms_norm_prompt_kernel, FLLM_FUSE_PREFILL_*) owns this fusion
    if (!fllm_fuse_enabled(FLLM_FUSE_ADD_NORM) || rows <= 0 || rows >= FLLM_PREFILL_MIN_ROWS || cols <= 0 || (cols & 3) != 0 || cols > 8192) return FLLM_DECLINED;
    if ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(branch) | reinterpret_cast<uintptr_t>(w) | reinterpret_cast<uintptr_t>(out)) % 16 != 0)
        return FLLM_DECLINED;
    PREP(st);
    void *q0 = nullptr, *q1 = nullptr;
    const float *gs = nullptr;
    const int q = fused_quant_target(st, quant, kind, rows, cols, &q0, &q1, next_w, next_rows, &gs);
    if (fllm_add_rms_norm_launch(st->s, x, branch, 1, nullptr, scale, w, out, rows, cols, eps, q, q0, q1, gs) != 0) return FLLM_DECLINED;
    if (int e = fllm_check_launch("add_rms_norm")) return e;
    fused_quant_done(st, q, out, rows, cols);
    return 0;
}

// Marlin 4-bit output projection and residual update, norm and activation conversion of the next GEMM in two kernels: the GEMM leaves
// its FP16 result in the io buffer, the norm kernel adds scale * (result + bias) to `res`, normalizes and converts. rows = output width
// (= the width of the norm), cols = input width.
int fllm_cuda_linear_q4z_add_norm(FllmStream *st, const void *w, int group, const float *act, float *res, const float *bias, int rows,
                                  int cols, int batch, const int *perm, float scale, const float *nw, float *out, float eps, int quant,
                                  int kind) {
    // decode batches only (see fllm_cuda_add_rms_norm): prompt batches leave the residual add pending for the prompt RMS norm (PendingAdd)
    if (!fllm_fuse_enabled(FLLM_FUSE_ADD_NORM) || rows <= 0 || cols <= 0 || batch <= 0 || batch >= FLLM_PREFILL_MIN_ROWS || (rows & 3) != 0 || rows > 8192)
        return FLLM_DECLINED;
    if ((reinterpret_cast<uintptr_t>(res) | reinterpret_cast<uintptr_t>(nw) | reinterpret_cast<uintptr_t>(out) | reinterpret_cast<uintptr_t>(bias)) % 16 != 0)
        return FLLM_DECLINED;
    fllm_q4z_skip_result = true;
    const int rc = fllm_cuda_linear_q4z(st, w, group, act, res, nullptr, rows, cols, batch, perm);
    fllm_q4z_skip_result = false;
    if (rc != 0) return rc;
    // the FP16 result sits behind the converted activations of the GEMM; the converted activations of the next GEMM (batch x rows halves
    // from the start of the buffer) must not reach it
    const size_t y16_off = ((size_t)batch * cols * 2 + 255) & ~size_t(255);
    const uint8_t *y16 = static_cast<const uint8_t *>(st->q4z_io) + y16_off;
    void *q0 = nullptr, *q1 = nullptr;
    int q = (size_t)batch * rows * 2 <= y16_off ? fused_quant_target(st, quant, kind, batch, rows, &q0, &q1) : 0;
    if (q != 1) q = 0;  // only the Marlin conversion shares this buffer; the other formats would need their own scratch
    if (fllm_add_rms_norm_launch(st->s, res, y16, 2, bias, scale, nw, out, batch, rows, eps, q, q0, q1) != 0)
        return fllm_set_error("fused Marlin output projection and norm: unsupported shape after the GEMM ran");
    if (int e = fllm_check_launch("q4z add_rms_norm")) return e;
    fused_quant_done(st, q, out, batch, rows);
    return 0;
}

// rotary embedding of q (in place) and the KV cache write of k and v in one kernel (f16 cache)
int fllm_cuda_rope_kv_f32(FllmStream *st, float *q, const float *k, const float *v, uint16_t *pool_k_layer, uint16_t *pool_v_layer,
                          const int32_t *positions, const int32_t *slots, int rows, const float *cs, const float *sn, int heads,
                          int kv_heads, int dk, int dv, int half_, int neox) {
    if (!fllm_fuse_enabled(FLLM_FUSE_ROPE_KV)) return FLLM_DECLINED;
    PREP(st);
    void *dp = nullptr, *ds = nullptr;
    if (int e = stage(st, 0, positions, (size_t)rows * sizeof(int32_t), &dp)) return e;
    if (int e = stage(st, 1, slots, (size_t)rows * sizeof(int32_t), &ds)) return e;
    return fllm_rope_kv_f32_launch(st->s, q, k, v, pool_k_layer, pool_v_layer, static_cast<const int32_t *>(dp), static_cast<const int32_t *>(ds),
                                   cs, sn, rows, heads, kv_heads, dk, dv, half_, neox);
}
#else
int fllm_cuda_add_rms_norm(FllmStream *, float *, const float *, float, const float *, float *, int, int, float, int, int, const void *, int) {
    return 3;
}
int fllm_cuda_linear_q4z_add_norm(FllmStream *, const void *, int, const float *, float *, const float *, int, int, int, const int *, float,
                                  const float *, float *, float, int, int) {
    return 3;
}
int fllm_cuda_rope_kv_f32(FllmStream *, float *, const float *, const float *, uint16_t *, uint16_t *, const int32_t *, const int32_t *, int,
                          const float *, const float *, int, int, int, int, int, int) {
    return 3;
}
#endif
int fllm_cuda_set_fuse(int which, int on) {
    fllm_fuse_force(which, on);
    return 0;
}
// Self test: pretend the device allows `bytes` of dynamic shared memory per block (0: the real limit).
int fllm_cuda_force_smem(int bytes) {
    g_forced_smem.store(bytes > 0 ? bytes : 0, std::memory_order_relaxed);
    return 0;
}
// The index of the largest logit of each row, copied to `host_out` (rows ints): greedy sampling without moving the logits.
int fllm_cuda_argmax(FllmStream *st, const float *logits, int rows, int vocab, int *host_out) {
    PREP(st);
    if (rows <= 0 || vocab <= 0) return fllm_set_error("argmax: empty input");
    if ((size_t)rows > st->argmax_cap) {
        CHECK(cudaStreamSynchronize(st->s), "cudaStreamSynchronize");
        cudaFree(st->argmax_out);
        st->argmax_out = nullptr;
        st->argmax_cap = (size_t)rows * 2 > 1024 ? (size_t)rows * 2 : 1024;
        CHECK(cudaMalloc(&st->argmax_out, st->argmax_cap * sizeof(int)), "cudaMalloc(argmax)");
    }
    if (int e = fllm_argmax_launch(st->s, logits, rows, vocab, st->argmax_out)) return e;
    CHECK(cudaMemcpyAsync(host_out, st->argmax_out, (size_t)rows * sizeof(int), cudaMemcpyDeviceToHost, st->s), "argmax download");
    CHECK(cudaStreamSynchronize(st->s), "argmax sync");
    return 0;
}
// Sampling a batch of rows on the device (sampling.cu). `params` is `rows` FllmSampleRow structs on the host; the lists are host arrays
// that the call uploads. The logits stay unchanged: the kernel works on a copy, a few rows at a time (at most 64 MiB of work space).
// Results are copied back.
int fllm_cuda_sample(FllmStream *st, const float *logits, int rows, int vocab, const void *params, const int32_t *pen_ids,
                     const int32_t *pen_counts, int pen_total, const int32_t *bias_ids, const float *bias_vals, int bias_total,
                     const uint32_t *mask, int mask_total, int mask_words, int top_stride, int32_t *ids_out, float *logprob_out,
                     int32_t *top_ids_out, float *top_logprob_out) {
    PREP(st);
    if (rows <= 0 || vocab <= 0) return fllm_set_error("sample: empty input");
    if (st->capture) return fllm_set_error("sample: not allowed while a CUDA graph is recorded");
    auto up = [](size_t n) { return (n + 255) & ~(size_t)255; };
    const size_t params_b = up((size_t)rows * 80), pen_b = up((size_t)(pen_total > 0 ? pen_total : 1) * 4),
                 bias_b = up((size_t)(bias_total > 0 ? bias_total : 1) * 4), mask_b = up((size_t)(mask_total > 0 ? mask_total : 1) * 4),
                 ids_b = up((size_t)rows * 4), lp_b = up((size_t)rows * 4), top_b = up((size_t)rows * (top_stride > 0 ? top_stride : 1) * 4);
    size_t chunk = (size_t)64 << 20;
    chunk = chunk / ((size_t)vocab * 4);
    if (chunk < 1) chunk = 1;
    if (chunk > (size_t)rows) chunk = (size_t)rows;
    const size_t work_b = up(chunk * (size_t)vocab * 4);
    const size_t total = params_b + 2 * pen_b + 2 * bias_b + mask_b + ids_b + lp_b + 2 * top_b + work_b;
    if (total > st->sample_cap) {
        CHECK(cudaStreamSynchronize(st->s), "cudaStreamSynchronize");
        cudaFree(st->sample_scratch);
        st->sample_scratch = nullptr;
        st->sample_cap = total + total / 4;
        CHECK(cudaMalloc(&st->sample_scratch, st->sample_cap), "cudaMalloc(sample)");
    }
    char *d_params = static_cast<char *>(st->sample_scratch);
    int32_t *d_pen_ids = reinterpret_cast<int32_t *>(d_params + params_b);
    int32_t *d_pen_counts = reinterpret_cast<int32_t *>(reinterpret_cast<char *>(d_pen_ids) + pen_b);
    int32_t *d_bias_ids = reinterpret_cast<int32_t *>(reinterpret_cast<char *>(d_pen_counts) + pen_b);
    float *d_bias_vals = reinterpret_cast<float *>(reinterpret_cast<char *>(d_bias_ids) + bias_b);
    uint32_t *d_mask = reinterpret_cast<uint32_t *>(reinterpret_cast<char *>(d_bias_vals) + bias_b);
    int32_t *d_ids = reinterpret_cast<int32_t *>(reinterpret_cast<char *>(d_mask) + mask_b);
    float *d_lp = reinterpret_cast<float *>(reinterpret_cast<char *>(d_ids) + ids_b);
    int32_t *d_top_ids = reinterpret_cast<int32_t *>(reinterpret_cast<char *>(d_lp) + lp_b);
    float *d_top_lp = reinterpret_cast<float *>(reinterpret_cast<char *>(d_top_ids) + top_b);
    float *d_work = reinterpret_cast<float *>(reinterpret_cast<char *>(d_top_lp) + top_b);
    CHECK(cudaMemcpyAsync(d_params, params, (size_t)rows * 80, cudaMemcpyHostToDevice, st->s), "sample params upload");
    if (pen_total > 0) {
        CHECK(cudaMemcpyAsync(d_pen_ids, pen_ids, (size_t)pen_total * 4, cudaMemcpyHostToDevice, st->s), "sample penalties upload");
        CHECK(cudaMemcpyAsync(d_pen_counts, pen_counts, (size_t)pen_total * 4, cudaMemcpyHostToDevice, st->s), "sample penalties upload");
    }
    if (bias_total > 0) {
        CHECK(cudaMemcpyAsync(d_bias_ids, bias_ids, (size_t)bias_total * 4, cudaMemcpyHostToDevice, st->s), "sample bias upload");
        CHECK(cudaMemcpyAsync(d_bias_vals, bias_vals, (size_t)bias_total * 4, cudaMemcpyHostToDevice, st->s), "sample bias upload");
    }
    if (mask_total > 0) CHECK(cudaMemcpyAsync(d_mask, mask, (size_t)mask_total * 4, cudaMemcpyHostToDevice, st->s), "sample mask upload");
    for (size_t first = 0; first < (size_t)rows; first += chunk) {
        const int n = (int)(((size_t)rows - first) < chunk ? ((size_t)rows - first) : chunk);
        if (int e = fllm_sample_launch(st->s, logits + first * (size_t)vocab, d_work, n, vocab, d_params + first * 80, d_pen_ids, d_pen_counts,
                                       d_bias_ids, d_bias_vals, d_mask, mask_words, d_ids + first, d_lp + first,
                                       d_top_ids + first * (size_t)(top_stride > 0 ? top_stride : 1),
                                       d_top_lp + first * (size_t)(top_stride > 0 ? top_stride : 1), top_stride > 0 ? top_stride : 1))
            return e;
    }
    CHECK(cudaMemcpyAsync(ids_out, d_ids, (size_t)rows * 4, cudaMemcpyDeviceToHost, st->s), "sample download");
    CHECK(cudaMemcpyAsync(logprob_out, d_lp, (size_t)rows * 4, cudaMemcpyDeviceToHost, st->s), "sample download");
    if (top_stride > 0) {
        CHECK(cudaMemcpyAsync(top_ids_out, d_top_ids, (size_t)rows * top_stride * 4, cudaMemcpyDeviceToHost, st->s), "sample download");
        CHECK(cudaMemcpyAsync(top_logprob_out, d_top_lp, (size_t)rows * top_stride * 4, cudaMemcpyDeviceToHost, st->s), "sample download");
    }
    CHECK(cudaStreamSynchronize(st->s), "sample sync");
    return 0;
}

int fllm_cuda_rms_norm(FllmStream *st, const float *x, const float *w, float *out, int rows, int cols, float eps) {
    // Prompt batches: the residual add that an output projection left pending (FLLM_FUSE_PREFILL_ADD_NORM) and the FP16 copy of the result
    // that the next Marlin GEMM would convert itself (FLLM_FUSE_PREFILL_NORM_F16) are done in the norm pass. Both are bit-identical to the
    // separate kernels (see rms_norm_prompt_kernel).
    if (rows >= FLLM_PREFILL_MIN_ROWS && !st->capture && cudaSetDevice(st->dev) == cudaSuccess && fllm_cc() >= 75 &&
        fllm_rms_norm_prompt_ok(x, w, out, cols)) {
        const FllmStream::PendingAdd pend = st->add;
        const bool add = pend.on && pend.x == x && pend.batch == rows && pend.width == cols && fllm_opt(FLLM_OPT_FUSE_ADD_NORM) != 0;
        const size_t bytes16 = (size_t)rows * cols * 2;
        // the FP16 rows go to the start of the Marlin activation scratch, which must not overlap the pending FP16 result
        bool f16 = fllm_opt(FLLM_OPT_FUSE_NORM_F16) != 0 && st->q4z_io != nullptr && bytes16 <= st->q4z_io_bytes;
        if (f16 && add) f16 = bytes16 <= (size_t)(static_cast<const char *>(pend.y16) - static_cast<const char *>(st->q4z_io));
        if (add || f16) {
            if (add) st->add.on = false;
            PREP(st);
            if (int e = fllm_rms_norm_prompt_launch(st->s, add ? pend.y16 : nullptr, pend.bias, pend.scale, const_cast<float *>(x), w, out,
                                                    f16 ? st->q4z_io : nullptr, rows, cols, eps))
                return e;
            if (f16) {
                st->q4z_x = out;
                st->q4z_batch = rows;
                st->q4z_cols = cols;
                st->q4z_perm = nullptr;
                st->q4z_ready = true;
            }
            return 0;
        }
    }
    PREP(st);
    // prompt-sized batches of NVFP4 models: wait for the GEMM that reads the result, which quantizes it in the same kernel (fllm_nvfp4_norm_quant_try)
    if (fllm_fuse_nvfp4_norm() && rows > 64 && cols % 256 == 0 && cols <= 8192 && !st->capture) {
        st->norm.pending = true;
        st->norm.x = x;
        st->norm.w = w;
        st->norm.out = out;
        st->norm.rows = rows;
        st->norm.cols = cols;
        st->norm.eps = eps;
        return 0;
    }
    return fllm_rms_norm_launch(st->s, x, w, out, rows, cols, eps);
}
int fllm_cuda_rms_norm_grouped(FllmStream *st, const float *x, const float *w, float *out, int rows, int cols, int groups, float eps) {
    PREP(st);
    return fllm_rms_norm_grouped_launch(st->s, x, w, out, rows, cols, groups, eps);
}
int fllm_cuda_layer_norm(FllmStream *st, const float *x, const float *w, const float *b, float *out, int rows, int cols,
                         float eps) {
    PREP(st);
    return fllm_layer_norm_launch(st->s, x, w, b, out, rows, cols, eps);
}
int fllm_cuda_glu(FllmStream *st, const float *gate, const float *up, float *out, size_t count, int mode) {
    PREP(st);
    static const bool no_fuse = getenv("FLLM_CUDA_NO_GLU_FUSE") != nullptr;
    if (no_fuse) return fllm_glu_launch(st->s, gate, up, out, count, mode);
    st->glu_gate = gate;
    st->glu_up = up;
    st->glu_out = out;
    st->glu_count = count;
    st->glu_mode = mode;
    st->glu_half = false;
    st->glu_pending = true;
    return 0;
}
// glu of FP16 gate and up (the results of NVFP4 GEMMs run with fllm_cuda_nvfp4_f16out): folded into the activation quantization of the CUTLASS GEMM
// that follows, or computed (FP32 result) when anything else needs it.
int fllm_cuda_glu_f16(FllmStream *st, const void *gate16, const void *up16, float *out, size_t count, int mode) {
    PREP(st);
    static const bool no_fuse = getenv("FLLM_CUDA_NO_GLU_FUSE") != nullptr;
    if (no_fuse) return fllm_glu_launch_f16(st->s, gate16, up16, out, count, mode);
    st->glu_gate = static_cast<const float *>(gate16);
    st->glu_up = static_cast<const float *>(up16);
    st->glu_out = out;
    st->glu_count = count;
    st->glu_mode = mode;
    st->glu_half = true;
    st->glu_pending = true;
    return 0;
}
int fllm_cuda_repack(FllmStream *st, void *dst, const void *src, int kind, size_t rows, size_t cols) {
    PREP(st);
    return fllm_repack_launch(st->s, dst, src, kind, rows, cols);
}
// idx (n int32) and, when `w` is not null, the n row weights travel in one staged block.
static int stage_rows(FllmStream *st, const int32_t *idx, const float *w, int n, const int32_t **dev_idx, const float **dev_w) {
    std::vector<int32_t> packed((size_t)n * (w ? 2 : 1));
    memcpy(packed.data(), idx, (size_t)n * sizeof(int32_t));
    if (w) memcpy(packed.data() + n, w, (size_t)n * sizeof(float));
    void *d = nullptr;
    if (int e = stage(st, -1, packed.data(), packed.size() * sizeof(int32_t), &d)) return e;
    *dev_idx = static_cast<const int32_t *>(d);
    *dev_w = w ? reinterpret_cast<const float *>(static_cast<const int32_t *>(d) + n) : nullptr;
    return 0;
}
int fllm_cuda_gather_rows(FllmStream *st, float *dst, const float *src, const int32_t *idx, const float *w, int n, int row_len) {
    PREP(st);
    const int32_t *di = nullptr;
    const float *dw = nullptr;
    if (int e = stage_rows(st, idx, w, n, &di, &dw)) return e;
    return fllm_gather_rows_launch(st->s, dst, src, di, dw, n, row_len);
}
int fllm_cuda_scatter_add_rows(FllmStream *st, float *dst, const float *src, const int32_t *idx, const float *w, int n, int row_len) {
    PREP(st);
    const int32_t *di = nullptr;
    const float *dw = nullptr;
    if (int e = stage_rows(st, idx, w, n, &di, &dw)) return e;
    return fllm_scatter_add_rows_launch(st->s, dst, src, di, dw, n, row_len);
}
int fllm_cuda_zero(FllmStream *st, void *ptr, size_t bytes) {
    PREP(st);
    CHECK(cudaMemsetAsync(ptr, 0, bytes, st->s), "cudaMemsetAsync");
    return 0;
}
// Page-lock host memory (a read-only file mapping) so host-to-device copies from it run asynchronously at full speed.
int fllm_cuda_host_register(int dev, void *ptr, size_t bytes) {
    CHECK(cudaSetDevice(dev), "cudaSetDevice");
    cudaError_t e = cudaHostRegister(ptr, bytes, cudaHostRegisterReadOnly | cudaHostRegisterPortable);
    if (e != cudaSuccess) {
        cudaGetLastError();
        return fllm_set_error("cudaHostRegister: %s", cudaGetErrorString(e));
    }
    return 0;
}
int fllm_cuda_host_unregister(void *ptr) {
    cudaHostUnregister(ptr);
    cudaGetLastError();
    return 0;
}
int fllm_cuda_moe_route(FllmStream *st, const float *logits, int32_t *ids, float *w, int rows, int experts, int k,
                        int normalize, float scale) {
    PREP(st);
    return fllm_moe_route_launch(st->s, logits, ids, w, rows, experts, k, normalize, scale);
}
int fllm_cuda_moe_combine(FllmStream *st, float *out, const float *y, const float *w, int tokens, int k, int hidden) {
    PREP(st);
    return fllm_moe_combine_launch(st->s, out, y, w, tokens, k, hidden);
}
int fllm_cuda_moe_linear(FllmStream *st, const void *const *tab, const void *const *btab, int kind, const int32_t *ids,
                         const float *x, float *y, int rows, int cols, int slots, int xdiv) {
    PREP(st);
    // the int8 activations live in the same scratch as the FP16 copies of the GEMMs
    const int inputs = (slots + xdiv - 1) / xdiv;
    const size_t need = (((size_t)inputs * cols + 255) & ~size_t(255)) + (size_t)inputs * (cols / 32) * 8 + ((size_t)4 << 20);
    if (need > st->x16_bytes) {
        if (st->capture) {
            return fllm_set_error("cuda_moe_linear: scratch buffer must not reallocate during graph capture (needs %zu, has %zu)", need, st->x16_bytes);
        }
        CHECK(cudaStreamSynchronize(st->s), "cudaStreamSynchronize");
        cudaFree(st->x16);
        st->epoch++;
        st->x16 = nullptr;
        st->x16_bytes = (need * 2 + 255) & ~size_t(255);
        CHECK(cudaMalloc(&st->x16, st->x16_bytes), "cudaMalloc(activation scratch)");
    }
    return fllm_moe_linear_launch(st->s, tab, kind, ids, x, y, rows, cols, slots, xdiv, btab, st->x16, st->x16_bytes);
}
int fllm_cuda_moe_route2(FllmStream *st, const float *logits, const float *bias, int32_t *ids, float *w, int rows, int experts, int k,
                         int sigmoid, int normalize, float scale, float eps, int groups, int groups_used, int top2) {
    PREP(st);
    return fllm_moe_route2_launch(st->s, logits, bias, ids, w, rows, experts, k, sigmoid, normalize, scale, eps, groups, groups_used, top2);
}
int fllm_cuda_moe_align(FllmStream *st, const int32_t *ids, int32_t *sorted, int32_t *block_expert, int32_t *num_post, int pairs,
                        int expert_offset, int num_local, int block) {
    PREP(st);
    return fllm_moe_align_launch(st->s, ids, sorted, block_expert, num_post, pairs, expert_offset, num_local, block);
}
int fllm_cuda_moe_grouped_supported(int kind, int cols, int block) { return fllm_moe_grouped_supported((uint32_t)kind, cols, block); }
int fllm_cuda_moe_grouped(FllmStream *st, int kind, int glu, const void *const *wtab, const void *const *wtab2, const void *const *btab,
                          const void *const *btab2, const float *x, const float *pw, float *y, const int32_t *sorted,
                          const int32_t *block_expert, const int32_t *num_post, int rows, int cols, int pairs, int xdiv, int block,
                          int max_blocks, int glu_mode, float alpha, float limit) {
    PREP(st);
    return fllm_moe_grouped_launch(st->s, (uint32_t)kind, glu, wtab, wtab2, btab, btab2, x, pw, y, sorted, block_expert, num_post, rows, cols, pairs,
                                   xdiv, block, max_blocks, glu_mode, alpha, limit);
}
int fllm_cuda_moe_combine2(FllmStream *st, float *out, const float *y, const float *w, const float *shared, const float *gate, int tokens,
                           int k, int hidden) {
    PREP(st);
    return fllm_moe_combine2_launch(st->s, out, y, w, shared, gate, tokens, k, hidden);
}
int fllm_cuda_moe_gate_scale(FllmStream *st, float *gate, const float *x, const float *g, int tokens, int hidden) {
    PREP(st);
    return fllm_moe_gate_scale_launch(st->s, gate, x, g, tokens, hidden);
}
int fllm_cuda_swiglu_oai(FllmStream *st, const float *gate, const float *up, float *out, size_t count, float alpha, float limit) {
    PREP(st);
    return fllm_swiglu_oai_launch(st->s, gate, up, out, count, alpha, limit);
}
int fllm_cuda_residual(FllmStream *st, float *x, const float *res, size_t count, float scale) {
    PREP(st);
    return fllm_residual_launch(st->s, x, res, count, scale);
}
int fllm_cuda_split_q_gate(FllmStream *st, const float *qg, float *q, float *gate, int rows, int heads, int dk, int per_head) {
    PREP(st);
    return fllm_split_q_gate_launch(st->s, qg, q, gate, rows, heads, dk, per_head);
}
// AMD build: SSM / DeltaNet / Mamba / ShortConv layers stay on the host path by default until verified on AMD hardware;
// FLLM_HIP_SSM=1 enables them on the device, and FLLM_HIP_NO_SSM=1 explicitly disables them.
static bool ssm_kernels_off() {
#if defined(__HIPCC__)
    static const bool off = [] {
        const char *no_ssm = getenv("FLLM_HIP_NO_SSM");
        if (no_ssm && no_ssm[0] == '1') return true;
        const char *ssm = getenv("FLLM_HIP_SSM");
        return !(ssm && ssm[0] == '1');
    }();
    return off;
#else
    return false;
#endif
}
int fllm_cuda_delta_supported(int kh, int vh, int hk, int hv, int ksz) { return !ssm_kernels_off() && fllm_delta_supported(kh, vh, hk, hv, ksz); }
// Gated DeltaNet layer: `runs` is nruns DeltaRun records (24 bytes each, see deltanet.cu) in host memory.
int fllm_cuda_delta_layer(FllmStream *st, const void *runs, int nruns, int max_n, int kh, int vh, int hk, int hv, int ksz, int tiled,
                          float beta_scale, float eps, size_t state_off, size_t rec_bytes, const float *qkv, const float *z,
                          const float *beta, const float *alpha, const float *convw, const float *dt_bias, const float *decay,
                          const float *norm, float *mixed, float *out) {
    PREP(st);
    void *d = nullptr;
    if (int e = stage(st, 4, runs, (size_t)nruns * 24, &d)) return e;
    return fllm_delta_launch(st->s, d, nruns, max_n, kh, vh, hk, hv, ksz, tiled, beta_scale, eps, state_off, rec_bytes, qkv, z, beta, alpha,
                             convw, dt_bias, decay, norm, mixed, out);
}
int fllm_cuda_kda_supported(int heads, int head_dim, int ksz) { return fllm_kda_supported(heads, head_dim, ksz); }
// Kimi Delta Attention (KDA) layer: `runs` is nruns DeltaRun records in host memory.
int fllm_cuda_kda_layer(FllmStream *st, const void *runs, int nruns, int max_n, int heads, int head_dim, int ksz,
                        float lower_bound, float eps, size_t state_off, size_t rec_bytes,
                        const float *qkv, const float *forget, const float *beta, const float *gate,
                        const float *convw, const float *dt_bias, const float *rate, const float *norm,
                        float *mixed, float *out) {
    PREP(st);
    void *d = nullptr;
    if (int e = stage(st, 4, runs, (size_t)nruns * 24, &d)) return e;
    return fllm_kda_launch(st->s, d, nruns, max_n, heads, head_dim, ksz, lower_bound, eps, state_off, rec_bytes,
                           qkv, forget, beta, gate, convw, dt_bias, rate, norm, mixed, out);
}
int fllm_cuda_act(FllmStream *st, const float *x, float *out, size_t n, int mode) {
    PREP(st);
    return fllm_act_launch(st->s, x, out, n, mode);
}
int fllm_cuda_act_glu(FllmStream *st, const float *gate, const float *up, float *out, size_t n, int mode) {
    PREP(st);
    return fllm_act_glu_launch(st->s, gate, up, out, n, mode);
}
int fllm_cuda_softplus_mul(FllmStream *st, float *x, const float *g, size_t n, float beta) {
    PREP(st);
    return fllm_softplus_mul_launch(st->s, x, g, n, beta);
}
int fllm_cuda_clamp(FllmStream *st, float *x, size_t n, float limit) {
    PREP(st);
    return fllm_clamp_launch(st->s, x, n, limit);
}
int fllm_cuda_mla_latent(FllmStream *st, const float *kv_a, float *latent, int rows, int kv_lora, int width) {
    PREP(st);
    return fllm_mla_latent_launch(st->s, kv_a, latent, rows, kv_lora, width);
}
int fllm_cuda_mla_assemble(FllmStream *st, const float *q_full, const float *kv_a, const float *kv, float *q, float *k, float *v, int rows,
                           int heads, int nope, int rope, int value, int kv_lora) {
    PREP(st);
    return fllm_mla_assemble_launch(st->s, q_full, kv_a, kv, q, k, v, rows, heads, nope, rope, value, kv_lora);
}
int fllm_cuda_mamba2_supported(int inner, int state, int heads, int groups, int ksz, int norm_groups) {
    return !ssm_kernels_off() && fllm_mamba2_supported(inner, state, heads, groups, ksz, norm_groups);
}
// Mamba-2 layer: `runs` is nruns SsmRun records (24 bytes each, see mamba2.cu) in host memory.
int fllm_cuda_mamba2_layer(FllmStream *st, const void *runs, int nruns, int rows, int inner, int state, int heads, int groups, int ksz,
                           int norm_groups, float step_min, float eps, size_t state_off, size_t rec_bytes, const float *proj,
                           const float *convw, const float *convb, const float *dt_bias, const float *decay, const float *skip,
                           const float *norm, float *mixed, float *out) {
    PREP(st);
    void *d = nullptr;
    if (int e = stage(st, 4, runs, (size_t)nruns * 24, &d)) return e;
    return fllm_mamba2_launch(st->s, d, nruns, rows, inner, state, heads, groups, ksz, norm_groups, step_min, eps, state_off, rec_bytes, proj,
                              convw, convb, dt_bias, decay, skip, norm, mixed, out);
}
int fllm_cuda_shortconv_supported(int hidden, int ksz) { return !ssm_kernels_off() && fllm_shortconv_supported(hidden, ksz); }
// ShortConv layer: `runs` is nruns ConvRun records (24 bytes each, see shortconv.cu) in host memory.
int fllm_cuda_shortconv_layer(FllmStream *st, const void *runs, int nruns, int hidden, int ksz, size_t state_off, const float *proj,
                              const float *convw, const float *convb, float *out) {
    PREP(st);
    void *d = nullptr;
    if (int e = stage(st, 4, runs, (size_t)nruns * 24, &d)) return e;
    return fllm_shortconv_launch(st->s, d, nruns, hidden, ksz, state_off, proj, convw, convb, out);
}
int fllm_cuda_mamba1_supported(int inner, int state, int rank, int ksz) { return !ssm_kernels_off() && fllm_mamba1_supported(inner, state, rank, ksz); }
// Mamba-1 layer, three launches around the GEMMs of x_proj and dt_proj; `runs` is nruns SsmRun records (24 bytes each) in host memory.
int fllm_cuda_mamba1_conv(FllmStream *st, const void *runs, int nruns, int inner, int ksz, size_t state_off, size_t rec_bytes, const float *proj,
                          const float *convw, const float *convb, float *mixed) {
    PREP(st);
    void *d = nullptr;
    if (int e = stage(st, 4, runs, (size_t)nruns * 24, &d)) return e;
    return fllm_mamba1_conv_launch(st->s, d, nruns, inner, ksz, state_off, rec_bytes, proj, convw, convb, mixed);
}
int fllm_cuda_mamba1_prep(FllmStream *st, float *split, float *dt_in, int rows, int rank, int state, float eps) {
    PREP(st);
    return fllm_mamba1_prep_launch(st->s, split, dt_in, rows, rank, state, eps);
}
int fllm_cuda_mamba1_scan(FllmStream *st, const void *runs, int nruns, int inner, int state, int rank, size_t state_off, const float *split,
                          const float *dt, const float *mixed, const float *proj, const float *decay, const float *skip, float *out) {
    PREP(st);
    void *d = nullptr;
    if (int e = stage(st, 4, runs, (size_t)nruns * 24, &d)) return e;
    return fllm_mamba1_scan_launch(st->s, d, nruns, inner, state, rank, state_off, split, dt, mixed, proj, decay, skip, out);
}
int fllm_cuda_sigmoid_mul(FllmStream *st, float *x, const float *g, size_t n) {
    PREP(st);
    return fllm_sigmoid_mul_launch(st->s, x, g, n);
}
int fllm_cuda_rope(FllmStream *st, float *q, float *k, const int32_t *positions, int rows, const float *cs,
                   const float *sn, int heads, int kv_heads, int dim, int half_, int neox) {
    PREP(st);
    void *d = nullptr;
    if (int e = stage(st, 0, positions, (size_t)rows * sizeof(int32_t), &d)) return e;
    return fllm_rope_launch(st->s, q, k, static_cast<const int32_t *>(d), cs, sn, rows, heads, kv_heads, dim, half_, neox);
}
int fllm_cuda_kv_write(FllmStream *st, const float *src, uint16_t *pool_layer, const int32_t *slots, int rows, int kv,
                       int kind, int group, int scale_base_words) {
    PREP(st);
    void *d = nullptr;
    if (int e = stage(st, 1, slots, (size_t)rows * sizeof(int32_t), &d)) return e;
    return fllm_kv_write_launch(st->s, src, pool_layer, static_cast<const int32_t *>(d), rows, kv, kind, group,
                                scale_base_words);
}
int fllm_cuda_attention(FllmStream *st, const float *q, const uint16_t *kpool_layer, const uint16_t *vpool_layer,
                        float *out, const int64_t *tables_host, size_t tables_len, const int64_t *lengths_host,
                        int rows, const float *sinks, const float *alibi, int heads, int kv_heads, int dk, int dv,
                        int stride, int block_size, int kind, int kgroup, int vgroup, int kscale_base,
                        int vscale_base, float scale, float softcap, int window, int chunk) {
    PREP(st);
    void *dt = nullptr, *dl = nullptr;
    const bool cached = step_cached(st);
    // (inside a forward pass the tables are staged once, and only when an attention kernel needs the table of every row)
    if (!cached) {
        if (int e = stage(st, 2, tables_host, tables_len * sizeof(int64_t), &dt)) return e;
        if (int e = stage(st, 3, lengths_host, (size_t)rows * sizeof(int64_t), &dl)) return e;
    } else {
        if (int e = stage_step(st, st->step.lengths, 3, lengths_host, (size_t)rows * sizeof(int64_t), &dl)) return e;
    }
    static const bool old_attention = getenv("FLLM_CUDA_OLD_ATTN") != nullptr;
    if (!old_attention && dk == dv && (kind == 1 || kind == 2)) {
        if (!st->ws) {
            st->ws_bytes = 16u << 20;
            CHECK(cudaMalloc(reinterpret_cast<void **>(&st->ws), st->ws_bytes), "cudaMalloc(attention workspace)");
        }
        // sliding window / soft cap / sinks, ragged batches, tensor-core decode and cascade attention through FlashInfer (fi_attn.cu; off by default)
        if (!st->capture) {
            const int rcf = fllm_fi_attention_try(st->s, q, kpool_layer, vpool_layer, out, tables_host, lengths_host, rows, sinks, alibi != nullptr, heads,
                                                  kv_heads, dk, dv, stride, block_size, kind, scale, softcap, window, chunk);
            if (rcf >= 0) return rcf;
        }
        static const bool no_flash = getenv("FLLM_CUDA_NO_FLASH") != nullptr;
        // The per-row decode kernels (attention4, attention2) for rows [r0, r1) of the batch: rows that are no part of a run of 16 or more
        // rows of one sequence (decode rows and verify rows), or all rows when the tensor-core kernel does not cover the shape.
        auto decode_rows = [&](int r0, int r1) -> int {
            if (cached && !dt) {
                if (int e = stage_step(st, st->step.tables, 2, tables_host, tables_len * sizeof(int64_t), &dt)) return e;
            }
            int max_len = 1;
            for (int r = r0; r < r1; r++) max_len = lengths_host[r] > max_len ? (int)lengths_host[r] : max_len;
            if (st->capture) max_len = st->capture->max_len;  // a graph is replayed with other lengths: size the splits for the longest context
            const int64_t *tabs = static_cast<const int64_t *>(dt) + (size_t)r0 * stride;
            const int64_t *lens = static_cast<const int64_t *>(dl) + r0;
            const float *qr = q + (size_t)r0 * heads * dk;
            float *outr = out + (size_t)r0 * heads * dv;
            const int rc4 = fllm_attention4_try(st->s, qr, kpool_layer, vpool_layer, outr, tabs, lens, max_len, sinks, alibi, r1 - r0, heads, kv_heads, dk, dv,
                                                stride, block_size, kind, scale, softcap, window, chunk, st->ws, st->ws_bytes, kscale_base, vscale_base);
            if (rc4 >= 0) return rc4;
            const int rc = fllm_attention2_try(st->s, qr, kpool_layer, vpool_layer, outr, tabs, lens, max_len, sinks, alibi, r1 - r0, heads, kv_heads, dk, dv,
                                               stride, block_size, kind, scale, softcap, window, chunk, st->ws, st->ws_bytes, kscale_base, vscale_base);
            if (rc >= 0) return rc;
            return -1;
        };
        if (!st->capture && !no_flash && !alibi && softcap == 0.f && window == 0 && chunk == 0 && rows >= 16 && !fllm_exact_math()) {
            // rows of one sequence (same block table, lengths growing by one) go to the tensor-core kernel; the other rows of the batch
            // (the decode rows next to a prompt chunk) go to the decode kernels, so a running request does not slow a prompt down
            std::vector<std::pair<int, int>> rest;  // row ranges for the decode kernels
            auto later = [&](int r0, int r1) {
                if (!rest.empty() && rest.back().second == r0) rest.back().second = r1;
                else rest.emplace_back(r0, r1);
            };
            if (cached) {
                // the rows are grouped once per forward pass; a run needs only its first block table, so one table per run is staged
                step_runs(st, tables_host, lengths_host, rows, stride);
                const auto &runs = st->step.runs;
                bool all = true, any = false;
                for (const auto &run : runs) {
                    if (run.second - run.first < 16) all = false;
                    else any = true;
                }
                void *dc = nullptr;
                if (any) {
                    auto &compact = st->step.compact_tables;
                    if (int e = stage_step(st, st->step.compact, 2, compact.data(), compact.size() * sizeof(int64_t), &dc)) return e;
                }
                // the FP16 attention result is also left in the Marlin scratch when one run covers all rows and the output projection
                // comes next (FLLM_FUSE_PREFILL_ATTN_F16)
                const size_t out16 = (size_t)rows * heads * dv * 2;
                const bool f16 = all && runs.size() == 1 && fllm_opt(FLLM_OPT_FUSE_ATTN_F16) != 0 && st->q4z_io != nullptr && out16 <= st->q4z_io_bytes;
                for (size_t i = 0; any && i < runs.size(); i++) {
                    const int r0 = runs[i].first, r1 = runs[i].second;
                    if (r1 - r0 < 16) {
                        later(r0, r1);
                        continue;
                    }
                    fllm_attn_x16 = f16 ? st->q4z_io : nullptr;
                    fllm_attn_x16_written = false;
                    const int rc = fllm_attention3_try(st->s, q + (size_t)r0 * heads * dk, kpool_layer, vpool_layer,
                                                       out + (size_t)r0 * heads * dv, static_cast<const int64_t *>(dc) + i * (size_t)stride,
                                                       static_cast<const int64_t *>(dl) + r0, sinks, r1 - r0, heads, kv_heads, dk, dv,
                                                       stride, block_size, kind, scale, kscale_base, vscale_base,
                                                       (int)lengths_host[r1 - 1]);
                    fllm_attn_x16 = nullptr;
                    if (rc < 0) {
                        later(r0, r1);
                        continue;
                    }
                    if (rc > 0) return rc;
                    if (fllm_attn_x16_written) {
                        st->q4z_x = out;
                        st->q4z_batch = rows;
                        st->q4z_cols = heads * dv;
                        st->q4z_perm = nullptr;
                        st->q4z_ready = true;
                    }
                }
                if (any) {
                    for (const auto &range : rest) {
                        const int rc = decode_rows(range.first, range.second);
                        if (rc != 0) return rc;
                    }
                    return 0;
                }
            } else {
                int r0 = 0;
                bool any = false;
                while (r0 < rows) {
                    int r1 = r0 + 1;
                    while (r1 < rows && lengths_host[r1] == lengths_host[r1 - 1] + 1 &&
                           memcmp(tables_host + (size_t)r1 * stride, tables_host + (size_t)r0 * stride,
                                  (size_t)stride * sizeof(int64_t)) == 0)
                        r1++;
                    const int n = r1 - r0;
                    int rc = -1;
                    if (n >= 16)
                        rc = fllm_attention3_try(st->s, q + (size_t)r0 * heads * dk, kpool_layer, vpool_layer,
                                                 out + (size_t)r0 * heads * dv,
                                                 static_cast<const int64_t *>(dt) + (size_t)r0 * stride,
                                                 static_cast<const int64_t *>(dl) + r0, sinks, n, heads, kv_heads, dk, dv,
                                                 stride, block_size, kind, scale, kscale_base, vscale_base,
                                                 (int)lengths_host[r1 - 1]);
                    if (rc < 0) {
                        later(r0, r1);
                    } else {
                        if (rc > 0) return rc;
                        any = true;
                    }
                    r0 = r1;
                }
                if (any) {
                    for (const auto &range : rest) {
                        const int rc = decode_rows(range.first, range.second);
                        if (rc != 0) return rc;
                    }
                    return 0;
                }
            }
        }
        {
            const int rc = decode_rows(0, rows);
            if (rc >= 0) return rc;
        }
    }
    if (cached && !dt) {
        if (int e = stage_step(st, st->step.tables, 2, tables_host, tables_len * sizeof(int64_t), &dt)) return e;
    }
    return fllm_attention_launch(st->s, q, kpool_layer, vpool_layer, out, static_cast<const int64_t *>(dt),
                                 static_cast<const int64_t *>(dl), sinks, alibi, rows, heads, kv_heads, dk, dv, stride,
                                 block_size, kind, kgroup, vgroup, kscale_base, vscale_base, scale, softcap, window,
                                 chunk);
}

int fllm_cuda_kvq_write(FllmStream *st, const float *src, uint16_t *pool_layer, const int32_t *slots, int rows,
                        const FllmKvDesc *desc) {
    PREP(st);
    void *d = nullptr;
    if (int e = stage(st, 1, slots, (size_t)rows * sizeof(int32_t), &d)) return e;
    return fllm_kvq_write_launch(st->s, src, pool_layer, static_cast<const int32_t *>(d), rows, *desc);
}
int fllm_cuda_kvq_attention(FllmStream *st, const float *q, const uint16_t *kpool_layer, const uint16_t *vpool_layer,
                            float *out, const int64_t *tables_host, size_t tables_len, const int64_t *lengths_host,
                            int rows, const float *sinks, const float *alibi, int heads, int kv_heads, int dk, int dv,
                            int stride, int block_size, const FllmKvDesc *kd, const FllmKvDesc *vd, float scale,
                            float softcap, int window, int chunk) {
    PREP(st);
    void *dt = nullptr, *dl = nullptr;
    if (int e = stage_step(st, st->step.tables, 2, tables_host, tables_len * sizeof(int64_t), &dt)) return e;
    if (int e = stage_step(st, st->step.lengths, 3, lengths_host, (size_t)rows * sizeof(int64_t), &dl)) return e;
    if (!st->ws) {
        st->ws_bytes = 16u << 20;
        CHECK(cudaMalloc(reinterpret_cast<void **>(&st->ws), st->ws_bytes), "cudaMalloc(attention workspace)");
    }
    return fllm_kvq_attention(st->s, st, q, kpool_layer, vpool_layer, out, static_cast<const int64_t *>(dt),
                              static_cast<const int64_t *>(dl), tables_host, lengths_host, sinks, alibi, rows, heads, kv_heads,
                              dk, dv, stride, block_size, *kd, *vd, scale, softcap, window, chunk, st->ws, st->ws_bytes,
                              st->capture != nullptr, st->capture ? st->capture->max_len : 0);
}

#if !defined(__HIPCC__)
#include "act_api.inc"
#endif

// ---- latent-cache MLA (mla_latent.cu, mla_decode.cu; docs/changes/hopper-mla.md) ----
int fllm_cuda_mla_latent_supported(int dev, int kv_lora, int rope, int nope, int value, int kv_fmt) {
    if (cudaSetDevice(dev) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return fllm_mla_latent_supported(kv_lora, rope, nope, value, kv_fmt);
}
int fllm_cuda_mla_kv(FllmStream *st, const float *kv_a, const float *latent, const int32_t *positions, const float *cs, const float *sn, float *k_row,
                     int rows, int kv_lora, int rope, int neox) {
    PREP(st);
    void *d = nullptr;
    if (int e = stage(st, 0, positions, (size_t)rows * sizeof(int32_t), &d)) return e;
    return fllm_mla_kv_launch(st->s, kv_a, latent, static_cast<const int32_t *>(d), cs, sn, k_row, rows, kv_lora, rope, neox);
}
int fllm_cuda_mla_query(FllmStream *st, const float *q_full, const uint16_t *w, const int32_t *positions, const float *cs, const float *sn, uint16_t *q_lat,
                        int rows, int heads, int nope, int rope, int value, int kv_lora, int neox) {
    PREP(st);
    void *d = nullptr;
    if (int e = stage(st, 0, positions, (size_t)rows * sizeof(int32_t), &d)) return e;
    return fllm_mla_query_launch(st->s, q_full, w, static_cast<const int32_t *>(d), cs, sn, q_lat, rows, heads, nope, rope, value, kv_lora, neox);
}
int fllm_cuda_mla_output(FllmStream *st, const uint16_t *o_lat, const uint16_t *w, float *out, int rows, int heads, int nope, int value, int kv_lora) {
    PREP(st);
    return fllm_mla_output_launch(st->s, o_lat, w, out, rows, heads, nope, value, kv_lora);
}
int fllm_cuda_mla_attention(FllmStream *st, const uint16_t *q_lat, const uint16_t *pool, uint16_t *o_lat, const int64_t *tables_host, size_t tables_len,
                            const int64_t *lengths_host, int rows, int heads, int kv_lora, int rope, int stride, int block_size, const FllmKvDesc *kd,
                            float scale, int mode) {
    PREP(st);
    void *dt = nullptr, *dl = nullptr;
    if (int e = stage(st, 2, tables_host, tables_len * sizeof(int64_t), &dt)) return e;
    if (int e = stage(st, 3, lengths_host, (size_t)rows * sizeof(int64_t), &dl)) return e;
    const int rc = fllm_mla_attention_launch(st->s, q_lat, pool, o_lat, static_cast<const int64_t *>(dt), static_cast<const int64_t *>(dl), tables_host,
                                             lengths_host, rows, heads, kv_lora, rope, stride, block_size, kd, scale, mode);
    if (rc < 0) return fllm_set_error("latent MLA attention does not cover %d + %d values per token in cache format %d", kv_lora, rope, kd->fmt);
    return rc;
}
int fllm_cuda_tree_attention(FllmStream *st, const float *q, const uint16_t *kpool_layer,
                             const uint16_t *vpool_layer, float *out, const int64_t *tables,
                             const int64_t *lengths, const uint8_t *tree_mask, int prefix_len,
                             int num_nodes, int rows, int heads, int kv_heads, int dk, int dv,
                             int stride, int block_size, int kind, float scale, float softcap) {
    PREP(st);
    return fllm_tree_attention_launch(st->s, q, kpool_layer, vpool_layer, out, tables, lengths,
                                      tree_mask, prefix_len, num_nodes, rows, heads, kv_heads,
                                      dk, dv, stride, block_size, kind, scale, softcap);
}

int fllm_cuda_spec_verify(FllmStream *st, const int32_t *draft, const int32_t *target,
                          int num_draft, int32_t *out_accepted, int32_t *out_next,
                          int64_t *seq_length) {
    PREP(st);
    return fllm_spec_verify_launch(st->s, draft, target, num_draft, out_accepted, out_next, seq_length);
}

}  // extern "C"

int fllm_cuda_attn_encoder(FllmStream *st, const float *q, const float *k, const float *v, float *out,
                           const int32_t *cu_seqlens_host, int batch_size, int total_tokens,
                           int heads, int kv_heads, int head_dim, float scale, int window,
                           const int32_t *global_tokens_host, int num_global) {
    PREP(st);
    void *d_cu = nullptr;
    if (int e = stage(st, 4, cu_seqlens_host, (size_t)(batch_size + 1) * sizeof(int32_t), &d_cu)) return e;
    void *d_globals = nullptr;
    if (num_global > 0 && global_tokens_host != nullptr) {
        if (int e = stage(st, 4, global_tokens_host, (size_t)num_global * sizeof(int32_t), &d_globals)) return e;
    }
    return fllm_attn_encoder_launch(st->s, q, k, v, out, static_cast<const int32_t *>(d_cu), batch_size,
                                    total_tokens, heads, kv_heads, head_dim, scale, window,
                                    static_cast<const int32_t *>(d_globals), num_global);
}
