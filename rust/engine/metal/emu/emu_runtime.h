// CPU emulation runtime of the p06a kernel check: threads that play the threads of a threadgroup. Included BEFORE the shim (the shim
// defines `thread`, `device` ... as macros, which would break <thread>).
//
// One std::thread per Metal thread, std::barrier for threadgroup_barrier and for the lock-step of a simdgroup (simd_sum).
// Threadgroups run one after the other. Not a GPU test: no memory model, no divergence, no performance, plain C++ float semantics.
#pragma once
#include <array>
#include <atomic>
#include <barrier>
#include <cstdint>
#include <functional>
#include <memory>
#include <thread>
#include <vector>

namespace emu {

struct Group {
    unsigned nthreads = 0;
    unsigned simd_width = 32;
    std::unique_ptr<std::barrier<>> tg;
    std::vector<std::unique_ptr<std::barrier<>>> sg;
    std::vector<float> slots;
};

struct Ctx {
    Group *group = nullptr;
    unsigned tid = 0;
    unsigned sg = 0;
    unsigned lane = 0;
};

inline thread_local Ctx ctx;

// Runs `body(tid, simdgroup index, lane)` on `nthreads` threads and waits for them.
template <class Body>
void run_group(unsigned nthreads, unsigned simd_width, Body &&body) {
    Group g;
    g.nthreads = nthreads;
    g.simd_width = simd_width;
    g.tg = std::make_unique<std::barrier<>>(nthreads);
    for (unsigned i = 0; i < nthreads / simd_width; ++i) g.sg.push_back(std::make_unique<std::barrier<>>(simd_width));
    g.slots.assign(nthreads, 0.0f);
    std::vector<std::thread> threads;
    threads.reserve(nthreads);
    for (unsigned tid = 0; tid < nthreads; ++tid) {
        threads.emplace_back([&, tid] {
            ctx = Ctx{&g, tid, tid / simd_width, tid % simd_width};
            body(tid, tid / simd_width, tid % simd_width);
        });
    }
    for (auto &t : threads) t.join();
}

}  // namespace emu
