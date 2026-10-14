// SPDX-License-Identifier: Apache-2.0
// fLLM Intel GPU backend (SYCL / oneAPI): device enumeration, USM memory, streams (in-order queues), copies.
// Verification level: L0 (compiles with the DPC++ front end; never run on an Intel GPU).
#include <atomic>
#include <cstdlib>
#include <cstring>

#include "common.hpp"

namespace fllm {

// The Rust side (ffi.rs `DeviceProps`) mirrors this struct byte for byte.
static_assert(sizeof(FllmSyclDeviceInfo) == 520, "FllmSyclDeviceInfo changed: update DeviceProps in ffi.rs");

namespace {

thread_local std::string t_error;
std::mutex g_async_mu;
std::string g_async_error;
std::atomic<bool> g_exact{false};

void async_handler(sycl::exception_list list) {
    for (const std::exception_ptr& e : list) {
        try {
            std::rethrow_exception(e);
        } catch (const std::exception& ex) {
            std::lock_guard<std::mutex> lock(g_async_mu);
            g_async_error += std::string(ex.what()) + "; ";
        }
    }
}

void copy_text(char* dst, size_t cap, const std::string& s) {
    std::strncpy(dst, s.c_str(), cap - 1);
    dst[cap - 1] = 0;
}

struct Registry {
    std::vector<std::unique_ptr<Device>> devices;
    std::string error;
};

#ifdef SYCL_EXT_ONEAPI_MATRIX
namespace mx = sycl::ext::oneapi::experimental::matrix;

// Matrix engine shapes of the device for fp16 inputs with an FP32 accumulator: 16 (8x16x16, sub-group 16), 8 (8x8x16, sub-group 8).
int xmx_shape(const sycl::device& dev, bool sg8, bool sg16) {
    int best = 0;
    try {
        for (const mx::combination& c : dev.get_info<sycl::ext::oneapi::experimental::info::device::matrix_combinations>()) {
            if (c.atype != mx::matrix_type::fp16 || c.btype != mx::matrix_type::fp16 || c.ctype != mx::matrix_type::fp32 ||
                c.dtype != mx::matrix_type::fp32)
                continue;
            const bool m_ok = c.msize == 8 || (c.msize == 0 && c.max_msize >= 8);
            const bool k_ok = c.ksize == 16 || (c.ksize == 0 && c.max_ksize >= 16);
            const bool n16 = c.nsize == 16 || (c.nsize == 0 && c.max_nsize >= 16);
            const bool n8 = c.nsize == 8;
            if (m_ok && k_ok && n16 && sg16) best = 16;
            else if (m_ok && k_ok && n8 && sg8 && best == 0) best = 8;
        }
    } catch (...) {
        best = 0;
    }
    return best;
}
#else
int xmx_shape(const sycl::device&, bool, bool) { return 0; }
#endif

std::string arch_name(const sycl::device& dev) {
#ifdef SYCL_EXT_ONEAPI_DEVICE_ARCHITECTURE
    namespace ex = sycl::ext::oneapi::experimental;
    try {
        switch (dev.get_info<ex::info::device::architecture>()) {
            case ex::architecture::intel_gpu_tgllp: return "tgllp";
            case ex::architecture::intel_gpu_dg1: return "dg1";
            case ex::architecture::intel_gpu_acm_g10: return "acm-g10";
            case ex::architecture::intel_gpu_acm_g11: return "acm-g11";
            case ex::architecture::intel_gpu_acm_g12: return "acm-g12";
            case ex::architecture::intel_gpu_pvc: return "pvc";
            default: return "";
        }
    } catch (...) {
    }
#else
    (void)dev;
#endif
    return "";
}

void fill_info(Device& d) {
    FllmSyclDeviceInfo& i = d.info;
    std::memset(&i, 0, sizeof(i));
    const sycl::device& dev = d.dev;
    copy_text(i.name, sizeof(i.name), dev.get_info<sycl::info::device::name>());
    copy_text(i.driver, sizeof(i.driver), dev.get_info<sycl::info::device::driver_version>());
    copy_text(i.arch, sizeof(i.arch), arch_name(dev));
    i.total_memory = dev.get_info<sycl::info::device::global_mem_size>();
    i.max_alloc = dev.get_info<sycl::info::device::max_mem_alloc_size>();
    i.compute_units = static_cast<int32_t>(dev.get_info<sycl::info::device::max_compute_units>());
    i.max_wg_size = static_cast<int32_t>(dev.get_info<sycl::info::device::max_work_group_size>());
    i.local_mem_bytes = static_cast<int32_t>(dev.get_info<sycl::info::device::local_mem_size>());
    i.is_gpu = dev.is_gpu() ? 1 : 0;
    i.has_fp16 = dev.has(sycl::aspect::fp16) ? 1 : 0;
    i.vendor_id = static_cast<uint32_t>(dev.get_info<sycl::info::device::vendor_id>());
    i.level_zero = dev.get_backend() == sycl::backend::ext_oneapi_level_zero ? 1 : 0;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    try {
        i.is_integrated = dev.get_info<sycl::info::device::host_unified_memory>() ? 1 : 0;
    } catch (...) {
    }
#pragma GCC diagnostic pop
#ifdef SYCL_EXT_INTEL_DEVICE_INFO
    try {
        if (dev.has(sycl::aspect::ext_intel_gpu_eu_count)) i.eu_count = static_cast<int32_t>(dev.get_info<sycl::ext::intel::info::device::gpu_eu_count>());
    } catch (...) {
    }
#endif
    bool sg8 = false, sg16 = false;
    for (size_t s : dev.get_info<sycl::info::device::sub_group_sizes>()) {
        sg8 |= s == 8;
        sg16 |= s == 16;
    }
    i.has_sg16 = sg16 ? 1 : 0;
    d.xmx_n = xmx_shape(dev, sg8, sg16);
    i.xmx_n16 = d.xmx_n == 16 ? 1 : 0;
    i.xmx_n8 = d.xmx_n == 8 ? 1 : 0;
}

// Free memory is a property of the moment: queried by device_info, not cached.
uint64_t free_memory(const sycl::device& dev) {
#ifdef SYCL_EXT_INTEL_DEVICE_INFO
    try {
        if (dev.has(sycl::aspect::ext_intel_free_memory)) return dev.get_info<sycl::ext::intel::info::device::free_memory>();
    } catch (...) {
    }
#else
    (void)dev;
#endif
    return 0;
}

Registry build_registry() {
    Registry r;
    try {
        std::vector<sycl::device> gpus = sycl::device::get_devices(sycl::info::device_type::gpu);
        std::vector<sycl::device> l0, other;
        for (const sycl::device& d : gpus) (d.get_backend() == sycl::backend::ext_oneapi_level_zero ? l0 : other).push_back(d);
        // Level Zero is the native Intel path; the OpenCL devices are the same GPUs again. FLLM_SYCL_ALL_BACKENDS=1 lists both.
        const bool all = std::getenv("FLLM_SYCL_ALL_BACKENDS") != nullptr;
        std::vector<sycl::device> chosen = l0;
        if (all || l0.empty()) chosen.insert(chosen.end(), other.begin(), other.end());
        for (const sycl::device& d : chosen) {
            auto dev = std::make_unique<Device>();
            dev->dev = d;
            dev->ctx = sycl::context(d, async_handler);
            fill_info(*dev);
            // every kernel needs sub-group size 16 (or 8 for the Alchemist matrix engine) and fp16 conversions
            if (!dev->info.has_fp16 || !(dev->info.has_sg16 || dev->xmx_n == 8)) {
                r.error += std::string(dev->info.name) + " skipped (needs fp16 and sub-group size 16); ";
                continue;
            }
            dev->setup_queue = std::make_unique<sycl::queue>(dev->ctx, d, async_handler, sycl::property_list{sycl::property::queue::in_order()});
            r.devices.push_back(std::move(dev));
        }
        if (r.devices.empty() && r.error.empty()) r.error = "the SYCL runtime reports no GPU device (is the Level Zero driver installed?)";
    } catch (const std::exception& e) {
        r.error = std::string("SYCL device discovery failed: ") + e.what();
        r.devices.clear();
    } catch (...) {
        r.error = "SYCL device discovery failed";
        r.devices.clear();
    }
    return r;
}

Registry& registry() {
    static Registry r = build_registry();
    return r;
}

}  // namespace

void set_error(const std::string& msg) { t_error = msg; }

std::string take_async_error() {
    std::lock_guard<std::mutex> lock(g_async_mu);
    std::string e;
    e.swap(g_async_error);
    return e;
}

bool exact_math() { return g_exact.load() || std::getenv("FLLM_SYCL_EXACT") != nullptr; }

int device_count_or_throw() {
    Registry& r = registry();
    if (r.devices.empty()) throw std::runtime_error(r.error.empty() ? "no SYCL GPU device" : r.error);
    return static_cast<int>(r.devices.size());
}

Device& get_device(int index) {
    const int n = device_count_or_throw();
    if (index < 0 || index >= n) throw std::runtime_error("SYCL device " + std::to_string(index) + " requested, " + std::to_string(n) + " present");
    return *registry().devices[static_cast<size_t>(index)];
}

// ---- Stream -------------------------------------------------------------------------------------------------------------------

Stream::Stream(Device& d)
    : dev(&d), q(d.ctx, d.dev, async_handler, sycl::property_list{sycl::property::queue::in_order()}) {}

Stream::~Stream() {
    try {
        q.wait();
    } catch (...) {
    }
    if (host_arena != nullptr) sycl::free(host_arena, dev->ctx);
    if (dev_arena != nullptr) sycl::free(dev_arena, dev->ctx);
    if (scratch != nullptr) sycl::free(scratch, dev->ctx);
    for (void* p : deferred) sycl::free(p, dev->ctx);
    std::lock_guard<std::mutex> lock(dev->mu);
    dev->streams.erase(this);
}

static void ensure_arena(Stream& s) {
    if (s.host_arena != nullptr) return;
    s.host_arena = static_cast<char*>(sycl::malloc_host(Stream::kArena, s.dev->ctx));
    s.dev_arena = static_cast<char*>(sycl::malloc_device(Stream::kArena, s.dev->dev, s.dev->ctx));
    if (s.host_arena == nullptr || s.dev_arena == nullptr) throw std::runtime_error("cannot allocate the staging arena");
}

void Stream::drain() {
    q.wait_and_throw();
    for (void* p : deferred) sycl::free(p, dev->ctx);
    deferred.clear();
    arena_off = 0;
    const std::string err = take_async_error();
    if (!err.empty()) throw std::runtime_error("asynchronous kernel error: " + err);
}

void* Stream::stage(const void* host, size_t bytes) {
    const size_t need = round_up(std::max<size_t>(bytes, 1), 256);
    if (need > kArena / 2) {
        void* d = sycl::malloc_device(need, dev->dev, dev->ctx);
        if (d == nullptr) throw std::runtime_error("out of device memory while staging a host array");
        q.memcpy(d, host, bytes);
        q.wait();  // the host source may be pageable and is only guaranteed until this call returns
        deferred.push_back(d);
        return d;
    }
    ensure_arena(*this);
    if (arena_off + need > kArena) drain();
    std::memcpy(host_arena + arena_off, host, bytes);
    q.memcpy(dev_arena + arena_off, host_arena + arena_off, bytes);
    void* r = dev_arena + arena_off;
    arena_off += need;
    return r;
}

void Stream::upload_async(void* dst, const void* src, size_t bytes) {
    if (bytes == 0) return;
    const size_t need = round_up(bytes, 256);
    if (need > kArena / 4) {
        q.memcpy(dst, src, bytes);
        q.wait();
        return;
    }
    ensure_arena(*this);
    if (arena_off + need > kArena) drain();
    std::memcpy(host_arena + arena_off, src, bytes);
    q.memcpy(dst, host_arena + arena_off, bytes);
    arena_off += need;
}

void* Stream::get_scratch(size_t bytes) {
    if (bytes <= scratch_bytes) return scratch;
    q.wait();
    if (scratch != nullptr) sycl::free(scratch, dev->ctx);
    const size_t grown = std::max(bytes, scratch_bytes * 2);
    scratch = sycl::malloc_device(grown, dev->dev, dev->ctx);
    if (scratch == nullptr) {
        scratch_bytes = 0;
        throw std::runtime_error("out of device memory for kernel scratch (" + std::to_string(grown) + " bytes)");
    }
    scratch_bytes = grown;
    return scratch;
}

}  // namespace fllm

using fllm::guarded;
using fllm::Stream;

#define FLLM_STREAM Stream& s = *reinterpret_cast<Stream*>(stp)

extern "C" {

const char* fllm_sycl_last_error(void) {
    static thread_local std::string copy;
    copy = fllm::t_error;
    return copy.c_str();
}

int fllm_sycl_device_count(void) {
    fllm::Registry& r = fllm::registry();
    if (r.devices.empty()) {
        fllm::set_error(r.error.empty() ? "no SYCL GPU device" : r.error);
        return -1;
    }
    return static_cast<int>(r.devices.size());
}

int fllm_sycl_device_info(int dev, FllmSyclDeviceInfo* out) {
    return guarded("device_info", [&] {
        fllm::Device& d = fllm::get_device(dev);
        *out = d.info;
        out->free_memory = fllm::free_memory(d.dev);
    });
}

int fllm_sycl_malloc(int dev, void** ptr, size_t bytes) {
    return guarded("malloc", [&] {
        fllm::Device& d = fllm::get_device(dev);
        const size_t n = std::max<size_t>(bytes, 64);
        void* p = nullptr;
        std::string why;
        try {
            p = sycl::malloc_device(n, d.dev, d.ctx);
        } catch (const sycl::exception& e) {
            why = e.what();
        }
        if (p == nullptr) {
            std::string msg = "cannot allocate " + std::to_string(n) + " bytes of device memory" + (why.empty() ? "" : ": " + why);
            // the reported limit is not enforced when the driver is told to relax it, so it is only a hint
            if (n > d.info.max_alloc)
                msg += " (the device reports " + std::to_string(d.info.max_alloc) +
                       " bytes as the limit of one allocation; Level Zero: export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1)";
            throw std::runtime_error(msg);
        }
        *ptr = p;
    });
}

int fllm_sycl_free(int dev, void* ptr) {
    return guarded("free", [&] {
        if (ptr == nullptr) return;
        fllm::Device& d = fllm::get_device(dev);
        {
            // a kernel enqueued earlier may still use the memory: let every live stream finish first (the lock also keeps a
            // stream that is being destroyed from vanishing under us)
            std::lock_guard<std::mutex> lock(d.mu);
            for (Stream* s : d.streams) s->q.wait();
        }
        sycl::free(ptr, d.ctx);
    });
}
int fllm_sycl_h2d_sync(int dev, void* dst, const void* src, size_t bytes) {
    return guarded("h2d", [&] {
        fllm::Device& d = fllm::get_device(dev);
        std::lock_guard<std::mutex> lock(d.mu);
        d.setup_queue->memcpy(dst, src, bytes).wait_and_throw();
        const std::string err = fllm::take_async_error();
        if (!err.empty()) throw std::runtime_error(err);
    });
}

int fllm_sycl_memset(int dev, void* dst, int value, size_t bytes) {
    return guarded("memset", [&] {
        fllm::Device& d = fllm::get_device(dev);
        std::lock_guard<std::mutex> lock(d.mu);
        d.setup_queue->memset(dst, value, bytes).wait_and_throw();
    });
}

int fllm_sycl_set_exact(int on) {
    fllm::g_exact.store(on != 0);
    return 0;
}

int fllm_sycl_is_exact(void) { return fllm::exact_math() ? 1 : 0; }

FllmSyclStream* fllm_sycl_stream_create(int dev) {
    try {
        fllm::Device& d = fllm::get_device(dev);
        auto s = std::make_unique<Stream>(d);
        {
            std::lock_guard<std::mutex> lock(d.mu);
            d.streams.insert(s.get());
        }
        return reinterpret_cast<FllmSyclStream*>(s.release());
    } catch (const std::exception& e) {
        fllm::set_error(std::string("stream_create: ") + e.what());
    } catch (...) {
        fllm::set_error("stream_create: unknown exception");
    }
    return nullptr;
}

void fllm_sycl_stream_destroy(FllmSyclStream* stp) { delete reinterpret_cast<Stream*>(stp); }

int fllm_sycl_stream_sync(FllmSyclStream* stp) {
    return guarded("synchronize", [&] {
        FLLM_STREAM;
        s.drain();
    });
}

int fllm_sycl_upload(FllmSyclStream* stp, void* dst, const void* src_host, size_t bytes) {
    return guarded("upload", [&] {
        FLLM_STREAM;
        s.upload_async(dst, src_host, bytes);
    });
}

int fllm_sycl_download(FllmSyclStream* stp, void* dst_host, const void* src, size_t bytes) {
    return guarded("download", [&] {
        FLLM_STREAM;
        if (bytes == 0) return;
        s.q.memcpy(dst_host, src, bytes);
        s.drain();
    });
}

int fllm_sycl_copy(FllmSyclStream* stp, void* dst, const void* src, size_t bytes) {
    return guarded("copy", [&] {
        FLLM_STREAM;
        if (bytes > 0) s.q.memcpy(dst, src, bytes);
    });
}

int fllm_sycl_zero(FllmSyclStream* stp, void* ptr, size_t bytes) {
    return guarded("zero", [&] {
        FLLM_STREAM;
        if (bytes > 0) s.q.memset(ptr, 0, bytes);
    });
}

}  // extern "C"