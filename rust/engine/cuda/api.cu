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

static thread_local char g_error[512] = "";

// state of the NVFP4 / Marlin FP16 paths (declared in fllm_cuda_internal.h; defined outside of extern "C" like fllm_glu_up)
thread_local bool fllm_fp4_acc = false;
thread_local float fllm_fp4_acc_scale = 0.f;
thread_local bool fllm_fp4_f16out = false;
thread_local bool fllm_glu_half = false;
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
int fllm_smem_limit() {
    static int cache[16] = {0};
    int dev = 0;
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
    *cc_minor = 0;
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
    return 0;
}
int fllm_cuda_memset(int dev, void *ptr, int value, size_t bytes) {
    CHECK(cudaSetDevice(dev), "cudaSetDevice");
    CHECK(cudaMemset(ptr, value, bytes), "cudaMemset");
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

#if !defined(__HIPCC__)
static void graph_free(FllmGraph *g);
#endif

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
    // Marlin (Q4Z weights): FP16 activations and result, split-K partials and the lock workspace (zero between calls)
    void *q4z_io = nullptr, *q4z_ctmp = nullptr, *q4z_ws = nullptr;
    size_t q4z_io_bytes = 0, q4z_ctmp_bytes = 0, q4z_ws_bytes = 0;
    int sms = 0;
    int *argmax_out = nullptr;  // device ids of the last argmax (rows x int)
    size_t argmax_cap = 0;
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
};

static int flush_add(FllmStream *st) {
    if (!st->add.on) return 0;
    st->add.on = false;
    return fllm_q4z_acc_launch(st->s, st->add.y16, st->add.bias, st->add.x, st->add.width, st->add.batch, st->add.scale);
}
static int flush_glu(FllmStream *st) {
    if (int e = flush_add(st)) return e;
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
#if !defined(__HIPCC__)
    for (auto *g : st->graphs) {
        if (g) {
            graph_free(g);
            delete g;
        }
    }
#endif
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
        cudaFree(st->arena);
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
    CHECK(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, st->s), "upload");
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
    if (st->glu_half && kind != 40) return false;
    if ((reinterpret_cast<uintptr_t>(st->glu_gate) & 15) != 0 || (reinterpret_cast<uintptr_t>(st->glu_up) & 15) != 0 ||
        (reinterpret_cast<uintptr_t>(w) & 15) != 0)
        return false;
    const bool narrow_ok = kind == 2 || kind == 3 || kind == 6 || kind == 7 || kind == 8 || kind == 20 || kind == 39;
    if ((cols & (narrow_ok ? 31 : 255)) != 0) return false;
    if (kind == 40) return true;
    static const bool no_qgemm = getenv("FLLM_CUDA_NO_QGEMM") != nullptr;
    if (batch < 16 || no_qgemm || fllm_exact_math() || fllm_cc() < 75) return false;
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
    if (!st->glu_pending || st->glu_half || x != st->glu_out || (const void *)y == (const void *)x || (size_t)batch * cols != st->glu_count)
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
        fllm_glu_mode = st->glu_mode;
        st->glu_pending = false;
    }
    const int rc = fllm_linear_launch(st->s, w, kind, fuse ? st->glu_gate : x, y, bias, rows, cols, batch, st->x16, st->x16_bytes);
    fllm_glu_up = nullptr;
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
int fllm_cuda_linear_nvfp4(FllmStream *st, const void *w, const float *x, float *y, const float *bias, float gscale,
                           int rows, int cols, int batch) {
    CHECK(cudaSetDevice(st->dev), "cudaSetDevice");
    if (int e = flush_add(st)) return e;
    st->q4z_ready = false;
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
    const bool ok = fllm_nvfp4_try(st->s, w, fuse ? st->glu_gate : x, y, bias, gscale, rows, cols, batch, st->x16, st->x16_bytes, reuse ? 0 : 1) != 0;
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
#if !defined(__HIPCC__)
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
    return cudaDriverGetVersion(&v) == cudaSuccess && v >= 12000 ? 1 : 0;
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
#else
int fllm_cuda_graph_supported(FllmStream *) { return 0; }
int fllm_cuda_graph_begin(FllmStream *, int, size_t, size_t, size_t, size_t, int) { return fllm_set_error("CUDA graphs need an NVIDIA GPU"); }
int fllm_cuda_graph_end(FllmStream *, int, int) { return fllm_set_error("CUDA graphs need an NVIDIA GPU"); }
int fllm_cuda_graph_set(FllmStream *, int, int, const void *, size_t) { return fllm_set_error("CUDA graphs need an NVIDIA GPU"); }
int fllm_cuda_graph_launch(FllmStream *, int) { return fllm_set_error("CUDA graphs need an NVIDIA GPU"); }
#endif

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
static int fused_quant_target(FllmStream *st, int quant, int kind, int rows, int cols, void **q0, void **q1) {
    if (quant == 0 || !fllm_fuse_enabled(FLLM_FUSE_NORM_QUANT)) return 0;
    if (quant == 1) {
        if (!st->q4z_io || (size_t)rows * cols * 2 > st->q4z_io_bytes) return 0;
        *q0 = st->q4z_io;
        return 1;
    }
    if (!st->x16) return 0;
    if (quant == 2) {
        if (!fllm_nvfp4_prep_ok(cols, rows)) return 0;
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
                           float eps, int quant, int kind) {
    if (!fllm_fuse_enabled(FLLM_FUSE_ADD_NORM) || rows <= 0 || cols <= 0 || (cols & 3) != 0 || cols > 8192) return FLLM_DECLINED;
    if ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(branch) | reinterpret_cast<uintptr_t>(w) | reinterpret_cast<uintptr_t>(out)) % 16 != 0)
        return FLLM_DECLINED;
    PREP(st);
    void *q0 = nullptr, *q1 = nullptr;
    const int q = fused_quant_target(st, quant, kind, rows, cols, &q0, &q1);
    if (fllm_add_rms_norm_launch(st->s, x, branch, 1, nullptr, scale, w, out, rows, cols, eps, q, q0, q1) != 0) return FLLM_DECLINED;
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
    if (!fllm_fuse_enabled(FLLM_FUSE_ADD_NORM) || rows <= 0 || cols <= 0 || batch <= 0 || (rows & 3) != 0 || rows > 8192) return FLLM_DECLINED;
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
int fllm_cuda_add_rms_norm(FllmStream *, float *, const float *, float, const float *, float *, int, int, float, int, int) { return 3; }
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
    return fllm_rms_norm_launch(st->s, x, w, out, rows, cols, eps);
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
        CHECK(cudaStreamSynchronize(st->s), "cudaStreamSynchronize");
        cudaFree(st->x16);
        st->epoch++;
        st->x16 = nullptr;
        st->x16_bytes = (need * 2 + 255) & ~size_t(255);
        CHECK(cudaMalloc(&st->x16, st->x16_bytes), "cudaMalloc(activation scratch)");
    }
    return fllm_moe_linear_launch(st->s, tab, kind, ids, x, y, rows, cols, slots, xdiv, btab, st->x16, st->x16_bytes);
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
int fllm_cuda_delta_supported(int kh, int vh, int hk, int hv, int ksz) { return fllm_delta_supported(kh, vh, hk, hv, ksz); }
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
    return fllm_mamba2_supported(inner, state, heads, groups, ksz, norm_groups);
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
int fllm_cuda_shortconv_supported(int hidden, int ksz) { return fllm_shortconv_supported(hidden, ksz); }
// ShortConv layer: `runs` is nruns ConvRun records (24 bytes each, see shortconv.cu) in host memory.
int fllm_cuda_shortconv_layer(FllmStream *st, const void *runs, int nruns, int hidden, int ksz, size_t state_off, const float *proj,
                              const float *convw, const float *convb, float *out) {
    PREP(st);
    void *d = nullptr;
    if (int e = stage(st, 4, runs, (size_t)nruns * 24, &d)) return e;
    return fllm_shortconv_launch(st->s, d, nruns, hidden, ksz, state_off, proj, convw, convb, out);
}
int fllm_cuda_mamba1_supported(int inner, int state, int rank, int ksz) { return fllm_mamba1_supported(inner, state, rank, ksz); }
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
        static const bool no_flash = getenv("FLLM_CUDA_NO_FLASH") != nullptr;
        if (!st->capture && !no_flash && !alibi && softcap == 0.f && window == 0 && chunk == 0 && rows >= 16 && !fllm_exact_math()) {
            // rows of one sequence (same block table, lengths growing by one) go to the tensor-core kernel
            if (cached) {
                // the rows are grouped once per forward pass; a run needs only its first block table, so one table per run is staged
                step_runs(st, tables_host, lengths_host, rows, stride);
                const auto &runs = st->step.runs;
                bool all = true;
                for (const auto &run : runs)
                    if (run.second - run.first < 16) all = false;
                void *dc = nullptr;
                if (all) {
                    auto &compact = st->step.compact_tables;
                    if (int e = stage_step(st, st->step.compact, 2, compact.data(), compact.size() * sizeof(int64_t), &dc)) return e;
                }
                // the FP16 attention result is also left in the Marlin scratch when one run covers all rows and the output projection
                // comes next (FLLM_FUSE_PREFILL_ATTN_F16)
                const size_t out16 = (size_t)rows * heads * dv * 2;
                const bool f16 = all && runs.size() == 1 && fllm_opt(FLLM_OPT_FUSE_ATTN_F16) != 0 && st->q4z_io != nullptr && out16 <= st->q4z_io_bytes;
                for (size_t i = 0; all && i < runs.size(); i++) {
                    const int r0 = runs[i].first, r1 = runs[i].second;
                    fllm_attn_x16 = f16 ? st->q4z_io : nullptr;
                    fllm_attn_x16_written = false;
                    const int rc = fllm_attention3_try(st->s, q + (size_t)r0 * heads * dk, kpool_layer, vpool_layer,
                                                       out + (size_t)r0 * heads * dv, static_cast<const int64_t *>(dc) + i * (size_t)stride,
                                                       static_cast<const int64_t *>(dl) + r0, sinks, r1 - r0, heads, kv_heads, dk, dv,
                                                       stride, block_size, kind, scale, kscale_base, vscale_base,
                                                       (int)lengths_host[r1 - 1]);
                    fllm_attn_x16 = nullptr;
                    if (rc < 0) {
                        all = false;
                        break;
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
                if (all) return 0;
            } else {
                int r0 = 0;
                bool all = true;
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
                        all = false;
                        break;
                    }
                    if (rc > 0) return rc;
                    r0 = r1;
                }
                if (all) return 0;
            }
        }
        if (cached) {
            if (int e = stage_step(st, st->step.tables, 2, tables_host, tables_len * sizeof(int64_t), &dt)) return e;
        }
        int max_len = 1;
        for (int r = 0; r < rows; r++) max_len = lengths_host[r] > max_len ? (int)lengths_host[r] : max_len;
        if (st->capture) max_len = st->capture->max_len;  // a graph is replayed with other lengths: size the splits for the longest context
        const int rc4 = fllm_attention4_try(st->s, q, kpool_layer, vpool_layer, out, static_cast<const int64_t *>(dt),
                                            static_cast<const int64_t *>(dl), max_len, sinks, alibi, rows, heads,
                                            kv_heads, dk, dv, stride, block_size, kind, scale, softcap, window, chunk,
                                            st->ws, st->ws_bytes, kscale_base, vscale_base);
        if (rc4 >= 0) return rc4;
        const int rc = fllm_attention2_try(st->s, q, kpool_layer, vpool_layer, out, static_cast<const int64_t *>(dt),
                                           static_cast<const int64_t *>(dl), max_len, sinks, alibi, rows, heads,
                                           kv_heads, dk, dv, stride, block_size, kind, scale, softcap, window, chunk,
                                           st->ws, st->ws_bytes, kscale_base, vscale_base);
        if (rc >= 0) return rc;
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
#if defined(__HIPCC__)
    (void)st; (void)src; (void)pool_layer; (void)slots; (void)rows; (void)desc;
    return fllm_set_error("the extra KV cache formats are not available in the ROCm build");
#else
    PREP(st);
    void *d = nullptr;
    if (int e = stage(st, 1, slots, (size_t)rows * sizeof(int32_t), &d)) return e;
    return fllm_kvq_write_launch(st->s, src, pool_layer, static_cast<const int32_t *>(d), rows, *desc);
#endif
}
int fllm_cuda_kvq_attention(FllmStream *st, const float *q, const uint16_t *kpool_layer, const uint16_t *vpool_layer,
                            float *out, const int64_t *tables_host, size_t tables_len, const int64_t *lengths_host,
                            int rows, const float *sinks, const float *alibi, int heads, int kv_heads, int dk, int dv,
                            int stride, int block_size, const FllmKvDesc *kd, const FllmKvDesc *vd, float scale,
                            float softcap, int window, int chunk) {
#if defined(__HIPCC__)
    (void)st; (void)q; (void)kpool_layer; (void)vpool_layer; (void)out; (void)tables_host; (void)tables_len; (void)lengths_host;
    (void)rows; (void)sinks; (void)alibi; (void)heads; (void)kv_heads; (void)dk; (void)dv; (void)stride; (void)block_size;
    (void)kd; (void)vd; (void)scale; (void)softcap; (void)window; (void)chunk;
    return fllm_set_error("the extra KV cache formats are not available in the ROCm build");
#else
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
#endif
}

}  // extern "C"
