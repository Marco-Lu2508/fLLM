// fLLM XDNA shim: a flat C interface over the C++ API of XRT (xrt::device, xrt::hw_context, xrt::kernel, xrt::bo).
//
// Why a shim: the amdxdna driver only accepts hardware contexts, which XRT exposes in its C++ API alone (the C functions
// xrtDeviceLoadXclbin* fail with "Operation not supported" on XDNA). The shim is the only C++ in fLLM; it is optional at build time
// (rust/engine/build.rs compiles it when the XRT headers are found, scripts/build-xdna-shim.sh does it by hand) and at run time
// (the Rust side opens the library with dlopen/LoadLibrary and reports "unavailable" when it or XRT is missing).
//
// Copyright 2026 the fLLM authors. Apache-2.0.
#include <chrono>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_hw_context.h"
#include "xrt/xrt_kernel.h"
#include "xrt/experimental/xrt_system.h"
#include "xrt/experimental/xrt_xclbin.h"

#if defined(_WIN32)
#define FX_API extern "C" __declspec(dllexport)
#else
#define FX_API extern "C" __attribute__((visibility("default")))
#endif

namespace {
thread_local std::string g_last_error;

void set_error(const char* where, const std::exception& e) { g_last_error = std::string(where) + ": " + e.what(); }
void set_error(const char* where, const char* what) { g_last_error = std::string(where) + ": " + what; }

struct Ctx {
    xrt::device dev;
    xrt::xclbin xb;
    xrt::hw_context hw;
    xrt::kernel kern;
    std::string kernel_name;
};

struct Buf {
    xrt::bo bo;
    size_t size = 0;
    void* ptr = nullptr;
};

struct RunH {
    xrt::run run;
};
}  // namespace

// ABI version of this interface; the Rust side refuses a different one.
FX_API uint32_t fx_abi_version() { return 1; }

// Text of the last error on this thread ("" when none).
FX_API const char* fx_last_error() { return g_last_error.c_str(); }

// Number of NPU devices XRT sees (-1 on error).
FX_API int32_t fx_device_count() {
    try {
        return static_cast<int32_t>(xrt::system::enumerate_devices());
    } catch (const std::exception& e) {
        set_error("enumerate_devices", e);
        return -1;
    }
}

// Opens device `index`, registers the xclbin (bytes in memory) and creates the hardware context and the kernel.
FX_API void* fx_open(uint32_t index, const void* xclbin, size_t xclbin_len) {
    try {
        auto c = std::make_unique<Ctx>();
        c->dev = xrt::device(index);
        const char* b = static_cast<const char*>(xclbin);
        std::vector<char> data(b, b + xclbin_len);
        c->xb = xrt::xclbin(data);
        auto kernels = c->xb.get_kernels();
        if (kernels.empty()) {
            set_error("fx_open", "the xclbin lists no kernel");
            return nullptr;
        }
        c->kernel_name = kernels[0].get_name();
        c->dev.register_xclbin(c->xb);
        c->hw = xrt::hw_context(c->dev, c->xb.get_uuid());
        c->kern = xrt::kernel(c->hw, c->kernel_name);
        return c.release();
    } catch (const std::exception& e) {
        set_error("fx_open", e);
        return nullptr;
    }
}

FX_API void fx_close(void* ctx) { delete static_cast<Ctx*>(ctx); }

// Buffer for kernel argument `arg` (1 = instruction stream, 3 = A, 4 = B, 5 = C). `cacheable` is for the instruction stream.
FX_API void* fx_buf_new(void* ctx, int32_t arg, size_t size, int32_t cacheable) {
    try {
        auto* c = static_cast<Ctx*>(ctx);
        auto b = std::make_unique<Buf>();
        auto flags = cacheable ? xrt::bo::flags::cacheable : xrt::bo::flags::host_only;
        b->bo = xrt::bo(c->dev, size, flags, c->kern.group_id(arg));
        b->size = size;
        b->ptr = b->bo.map<void*>();
        return b.release();
    } catch (const std::exception& e) {
        set_error("fx_buf_new", e);
        return nullptr;
    }
}

FX_API void fx_buf_free(void* buf) { delete static_cast<Buf*>(buf); }

// Host mapping of the buffer.
FX_API void* fx_buf_ptr(void* buf) { return static_cast<Buf*>(buf)->ptr; }

FX_API size_t fx_buf_size(void* buf) { return static_cast<Buf*>(buf)->size; }

// Flushes (to_device != 0) or invalidates (to_device == 0) the first `size` bytes (0 = all). Returns 0 on success.
FX_API int32_t fx_buf_sync(void* buf, int32_t to_device, size_t size) {
    try {
        auto* b = static_cast<Buf*>(buf);
        if (size == 0 || size > b->size) size = b->size;
        b->bo.sync(to_device ? XCL_BO_SYNC_BO_TO_DEVICE : XCL_BO_SYNC_BO_FROM_DEVICE, size, 0);
        return 0;
    } catch (const std::exception& e) {
        set_error("fx_buf_sync", e);
        return -1;
    }
}

// Starts the kernel with the calling convention of mlir-aie / IRON designs: opcode 3, the instruction buffer and its length in
// words, then the data buffers a, b, c. Returns a run handle (finish it with fx_run_wait) or null.
FX_API void* fx_run_start(void* ctx, void* instr, uint32_t instr_words, void* a, void* b, void* c) {
    try {
        auto* x = static_cast<Ctx*>(ctx);
        auto r = std::make_unique<RunH>();
        r->run = x->kern(3u, static_cast<Buf*>(instr)->bo, instr_words, static_cast<Buf*>(a)->bo, static_cast<Buf*>(b)->bo,
                         static_cast<Buf*>(c)->bo);
        return r.release();
    } catch (const std::exception& e) {
        set_error("fx_run_start", e);
        return nullptr;
    }
}

// Waits for the run (timeout_ms 0 = no timeout) and frees the handle. Returns 0 when the run completed.
FX_API int32_t fx_run_wait(void* run, uint32_t timeout_ms) {
    std::unique_ptr<RunH> r(static_cast<RunH*>(run));
    try {
        auto st = r->run.wait(std::chrono::milliseconds(timeout_ms));
        if (st != ERT_CMD_STATE_COMPLETED) {
            g_last_error = "fx_run_wait: the kernel run ended in state " + std::to_string(static_cast<int>(st));
            return -1;
        }
        return 0;
    } catch (const std::exception& e) {
        set_error("fx_run_wait", e);
        return -1;
    }
}
