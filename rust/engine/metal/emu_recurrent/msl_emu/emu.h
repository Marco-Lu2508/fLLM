// CPU emulation of the small part of Metal Shading Language that the kernels of package 6c use (moe_*, gate_*, delta_*, mamba_*), so that the
// REAL kernel text can run on a CPU test machine: every thread of a threadgroup is a std::thread, threadgroup memory is a static array, and
// threadgroup_barrier is a real barrier. Built with -fsanitize=thread, a missing barrier shows up as a data race.
//
// What this checks: the algorithm, the index arithmetic, the barrier discipline and the block layouts of the kernels.
// What it does NOT check: address-space rules, attribute indices, Metal compiler acceptance, GPU limits, the fast-math behaviour of the GPU,
// simdgroup semantics (none are used), memory-model details beyond the barrier (the kernels need only barrier-ordered threadgroup memory).
//
// Source transformation (run_emulation.sh): [[...]] attributes removed, `[[kernel]]` / `[[host_name()]]` removed, indented
// `threadgroup T name[..];` declarations (kernel-scope threadgroup memory) become `static T name[..];`, the macros below handle the rest.
#pragma once
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;
typedef _Float16 half;

struct uint2 {
    uint x, y;
};
struct uint3 {
    uint x, y, z;
};

namespace metal {

enum class mem_flags { mem_none, mem_device, mem_threadgroup, mem_texture };

// ---- barrier ------------------------------------------------------------------------------------------------------------------------
struct EmuBarrier {
    std::mutex m;
    std::condition_variable cv;
    int total = 0;
    int waiting = 0;
    unsigned generation = 0;
    bool timed_out = false;
    void arrive() {
        std::unique_lock<std::mutex> lock(m);
        const unsigned gen = generation;
        if (++waiting == total) {
            waiting = 0;
            generation++;
            cv.notify_all();
            return;
        }
        if (!cv.wait_for(lock, std::chrono::seconds(60), [&] { return generation != gen; })) {
            std::fprintf(stderr, "EMU: barrier deadlock (a thread left the kernel before a barrier, or barriers are not uniform)\n");
            std::abort();
        }
    }
};
inline EmuBarrier *&emu_barrier() {
    static EmuBarrier *b = nullptr;
    return b;
}
inline void threadgroup_barrier(mem_flags) { emu_barrier()->arrive(); }

// ---- math ---------------------------------------------------------------------------------------------------------------------------
template <class T>
inline T min(T a, T b) {
    return b < a ? b : a;
}
template <class T>
inline T max(T a, T b) {
    return a < b ? b : a;
}
inline float exp(float x) { return std::exp(x); }
inline float log(float x) { return std::log(x); }
inline float sqrt(float x) { return std::sqrt(x); }
namespace precise {
inline float exp(float x) { return std::exp(x); }
inline float log(float x) { return std::log(x); }
inline float sqrt(float x) { return std::sqrt(x); }
}  // namespace precise
namespace fast = precise;

template <class To, class From>
inline To as_type(From f) {
    static_assert(sizeof(To) == sizeof(From), "as_type: size mismatch");
    To t;
    std::memcpy(&t, &f, sizeof(To));
    return t;
}

}  // namespace metal
using namespace metal;

// ---- launching ----------------------------------------------------------------------------------------------------------------------
// Run `body(lane)` for lane = 0 .. threads-1 as concurrent threads sharing one barrier (one threadgroup).
inline void emu_threadgroup(uint threads, const std::function<void(uint)> &body) {
    metal::EmuBarrier barrier;
    barrier.total = int(threads);
    metal::emu_barrier() = &barrier;
    std::vector<std::thread> pool;
    pool.reserve(threads);
    for (uint lane = 0; lane < threads; lane++) pool.emplace_back([&body, lane] { body(lane); });
    for (auto &t : pool) t.join();
    metal::emu_barrier() = nullptr;
}

// MSL address-space and function qualifiers; defined last so that the C++ headers above are unaffected (include the C++ headers BEFORE this file).
#define device
#define constant const
#define thread
#define threadgroup
#define kernel
