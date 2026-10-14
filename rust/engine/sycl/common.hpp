// SPDX-License-Identifier: Apache-2.0
// fLLM Intel GPU backend (SYCL / oneAPI): shared host-side declarations (devices, streams, error handling, launch helpers).
// Verification level: L0 (compiles with the DPC++ front end; never run on an Intel GPU).
#pragma once

#include <sycl/sycl.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "fllm_sycl.h"

namespace fllm {

// ---- errors --------------------------------------------------------------------------------------------------------
void set_error(const std::string& msg);
std::string take_async_error();

// Runs `f`; converts exceptions into the nonzero return code + last-error text of the C interface.
template <class F>
int guarded(const char* what, F&& f) {
    try {
        f();
        return 0;
    } catch (const sycl::exception& e) {
        set_error(std::string(what) + ": SYCL error: " + e.what());
    } catch (const std::exception& e) {
        set_error(std::string(what) + ": " + e.what());
    } catch (...) {
        set_error(std::string(what) + ": unknown exception");
    }
    return 1;
}

// ---- devices -------------------------------------------------------------------------------------------------------
struct Stream;

struct Device {
    sycl::device dev;
    sycl::context ctx;
    std::unique_ptr<sycl::queue> setup_queue;  // in-order queue for synchronous copies / memsets
    std::mutex mu;                             // guards setup_queue and streams
    std::set<Stream*> streams;                 // live streams: waited for before a device allocation is freed
    FllmSyclDeviceInfo info{};
    int xmx_n = 0;  // 16: 8x16x16 joint_matrix with sub-group 16, 8: 8x8x16 with sub-group 8, 0: none
};

int device_count_or_throw();  // throws std::runtime_error with the reason when there is no device
Device& get_device(int index);

// ---- streams -------------------------------------------------------------------------------------------------------
struct Stream {
    Device* dev;
    sycl::queue q;

    explicit Stream(Device& d);

    // Staging of small host arrays (positions, slots, block tables ...): a host-USM arena the CPU fills, a device arena the
    // queue copies it into. Regions are only reused after the queue drained (wait), so in-flight kernels never see them change.
    static constexpr size_t kArena = 16u << 20;
    char* host_arena = nullptr;
    char* dev_arena = nullptr;
    size_t arena_off = 0;
    std::vector<void*> deferred;  // one-off staging buffers released at the next synchronization

    void* scratch = nullptr;
    size_t scratch_bytes = 0;

    ~Stream();
    // Copy `bytes` of host memory to the device arena and return the device address (valid for kernels enqueued afterwards).
    void* stage(const void* host, size_t bytes);
    // Copy `bytes` of host memory to device memory `dst` without waiting (small copies go through the host arena).
    void upload_async(void* dst, const void* src, size_t bytes);
    // Device scratch of at least `bytes` bytes (kept between calls; contents are undefined at entry).
    void* get_scratch(size_t bytes);
    // Wait for all work, release one-off buffers, rewind the arena; throws on asynchronous kernel errors.
    void drain();
};

inline size_t round_up(size_t a, size_t b) { return (a + b - 1) / b * b; }
inline size_t ceil_div(size_t a, size_t b) { return (a + b - 1) / b; }

// Number of work-groups of size `wg` that cover n items.
inline size_t groups_for(size_t n, size_t wg) { return n == 0 ? 1 : ceil_div(n, wg); }

// ---- weight layout (shared by the host packer and the kernels' launch code) ------------------------------------------
// Returns false when `kind` is not supported or `cols` is not a multiple of the block size.
bool weight_layout(int kind, size_t cols, size_t* nbp, size_t* row_bytes);
size_t raw_row_bytes(int kind, size_t cols);
int device_kind(int kind);  // IQ4_XS is stored as IQ4_NL

// ---- operations implemented in the *.cpp files -----------------------------------------------------------------------
// linear_small.cpp / linear_tiled.cpp / linear_xmx.cpp: return false when the format is not handled.
bool launch_linear_small(Stream& s, const void* w, int kind, int rows, int cols, const float* x, float* y, const float* bias,
                         int batch);
bool launch_linear_tiled(Stream& s, const void* w, int kind, int rows, int cols, const float* x, float* y, const float* bias,
                         int batch);
bool launch_linear_xmx(Stream& s, const void* w, int kind, int rows, int cols, const float* x, float* y, const float* bias,
                       int batch);
bool launch_moe_linear(Stream& s, const void* const* tab, const float* const* btab, int kind, const int32_t* ids, const float* x,
                       float* y, int rows, int cols, int slots, int x_div);
bool launch_embedding(Stream& s, const void* w, int kind, const int64_t* ids_dev, int n, float* out, int cols);

bool exact_math();

}  // namespace fllm
