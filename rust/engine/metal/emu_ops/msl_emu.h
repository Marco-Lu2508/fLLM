// msl_emu.h - runs the MSL kernels of fLLM on the CPU (package p06b). NOT a Metal implementation, a test aid for hosts without a Mac.
//
// The MSL text is compiled as C++ against the syntax shim of the toolchain package (msl_shim.h, found through -I) - which is how mslcheck.sh checks
// syntax - but this header replaces the placeholder bodies of the shim for everything that has runtime meaning:
//   * a threadgroup runs as one std::thread per Metal thread; `threadgroup_barrier` is a real barrier, `simdgroup_barrier` a barrier of the 32 lanes;
//   * simd_sum / simd_max / simd_min / simd_shuffle* / simd_any / simd_all exchange values between the 32 lanes of a simdgroup;
//   * simdgroup_matrix<T, 8, 8> is held REPLICATED in every lane (every lane has the whole 8x8 tile); simdgroup_load reads the tile, simdgroup_store
//     is executed by lane 0 only, simdgroup_multiply_accumulate computes the whole product in every lane. This is correct for code that only uses
//     the documented API (no thread_elements()), as the kernels of this project do, and needs the simdgroup_barrier the kernels place after a store;
//   * `threadgroup` variables declared in a kernel are turned into `static` ones by the preprocessing of run_emu.sh (threadgroups are run one after
//     the other, so the variable is shared by the threads of the running group).
// What it cannot find: address space errors, bank conflicts, register / threadgroup memory limits, precision of the GPU's fast math, the real
// rounding of simd_sum (the lanes are added in order), alignment rules of packed vectors (packed_float4 is 16-byte aligned here: test data keeps
// row starts 16-byte aligned), anything about performance. SIMD width 32 and at most 1024 threads per group are assumed.
#ifndef FLLM_MSL_EMU_H
#define FLLM_MSL_EMU_H

#include <array>
#include <condition_variable>
#include <functional>
#include <cstdint>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace emu {

struct Barrier {
    std::mutex m;
    std::condition_variable cv;
    int n;
    int count = 0;
    unsigned generation = 0;
    explicit Barrier(int participants) : n(participants) {}
    void wait() {
        std::unique_lock<std::mutex> lock(m);
        const unsigned gen = generation;
        if (++count == n) {
            count = 0;
            ++generation;
            cv.notify_all();
        } else {
            cv.wait(lock, [&] { return gen != generation; });
        }
    }
};

struct Group {
    int nthreads;
    int nsg;
    Barrier tg;
    std::vector<std::unique_ptr<Barrier>> sgb;
    std::vector<std::array<uint64_t, 32>> slots;
    explicit Group(int threads) : nthreads(threads), nsg((threads + 31) / 32), tg(threads), slots(static_cast<size_t>((threads + 31) / 32)) {
        for (int s = 0; s < nsg; s++) sgb.push_back(std::make_unique<Barrier>(32));
    }
};

struct Ctx {
    Group *g;
    unsigned tid, sg, lane;
};
inline thread_local Ctx *tctx = nullptr;

template <class T>
inline uint64_t to_bits(T x) {
    static_assert(sizeof(T) <= 8);
    uint64_t b = 0;
    std::memcpy(&b, &x, sizeof(T));
    return b;
}
template <class T>
inline T from_bits(uint64_t b) {
    T x;
    std::memcpy(&x, &b, sizeof(T));
    return x;
}

// every lane stores `x`, all lanes meet, `pick` reads the 32 stored values, all lanes meet again (the slots can be reused)
template <class T, class F>
inline auto collective(T x, F pick) {
    Ctx &c = *tctx;
    auto &s = c.g->slots[c.sg];
    s[c.lane] = to_bits(x);
    c.g->sgb[c.sg]->wait();
    auto r = pick(s, c.lane);
    c.g->sgb[c.sg]->wait();
    return r;
}

}  // namespace emu

namespace emu {
// runs body(tid, group) on one std::thread per Metal thread of a threadgroup (defined before the shim, which erases the identifier `thread`)
inline void run_threads(int nthreads, const std::function<void(unsigned, Group &)> &body) {
    Group g(nthreads);
    std::vector<std::thread> threads;
    threads.reserve(static_cast<size_t>(nthreads));
    for (int t = 0; t < nthreads; t++) {
        threads.emplace_back([&, t] {
            Ctx c{&g, static_cast<unsigned>(t), static_cast<unsigned>(t) / 32u, static_cast<unsigned>(t) % 32u};
            tctx = &c;
            body(static_cast<unsigned>(t), g);
        });
    }
    for (auto &th : threads) th.join();
}
}  // namespace emu
// Rename the placeholders of the shim; the real ones are defined below in namespace metal.
#define threadgroup_barrier shim_unused_threadgroup_barrier
#define simdgroup_barrier shim_unused_simdgroup_barrier
#define simd_sum shim_unused_simd_sum
#define simd_max shim_unused_simd_max
#define simd_min shim_unused_simd_min
#define simd_shuffle shim_unused_simd_shuffle
#define simd_shuffle_xor shim_unused_simd_shuffle_xor
#define simd_shuffle_down shim_unused_simd_shuffle_down
#define simd_shuffle_up shim_unused_simd_shuffle_up
#define simd_all shim_unused_simd_all
#define simd_any shim_unused_simd_any
#define simdgroup_matrix shim_unused_simdgroup_matrix
#define make_filled_simdgroup_matrix shim_unused_make_filled_simdgroup_matrix
#define simdgroup_load shim_unused_simdgroup_load
#define simdgroup_store shim_unused_simdgroup_store
#define simdgroup_multiply shim_unused_simdgroup_multiply
#define simdgroup_multiply_accumulate shim_unused_simdgroup_multiply_accumulate
#include <metal_stdlib>
#undef threadgroup_barrier
#undef simdgroup_barrier
#undef simd_sum
#undef simd_max
#undef simd_min
#undef simd_shuffle
#undef simd_shuffle_xor
#undef simd_shuffle_down
#undef simd_shuffle_up
#undef simd_all
#undef simd_any
#undef simdgroup_matrix
#undef make_filled_simdgroup_matrix
#undef simdgroup_load
#undef simdgroup_store
#undef simdgroup_multiply
#undef simdgroup_multiply_accumulate

namespace metal {

inline void threadgroup_barrier(mem_flags) { emu::tctx->g->tg.wait(); }
inline void simdgroup_barrier(mem_flags) { emu::tctx->g->sgb[emu::tctx->sg]->wait(); }

template <class T>
inline T simd_sum(T x) {
    return emu::collective(x, [](auto &s, unsigned) {
        T r = emu::from_bits<T>(s[0]);
        for (int i = 1; i < 32; i++) r = r + emu::from_bits<T>(s[i]);
        return r;
    });
}
template <class T>
inline T simd_max(T x) {
    return emu::collective(x, [](auto &s, unsigned) {
        T r = emu::from_bits<T>(s[0]);
        for (int i = 1; i < 32; i++) r = metal::max(r, emu::from_bits<T>(s[i]));
        return r;
    });
}
template <class T>
inline T simd_min(T x) {
    return emu::collective(x, [](auto &s, unsigned) {
        T r = emu::from_bits<T>(s[0]);
        for (int i = 1; i < 32; i++) r = metal::min(r, emu::from_bits<T>(s[i]));
        return r;
    });
}
template <class T>
inline T simd_shuffle(T x, ushort lane) {
    return emu::collective(x, [lane](auto &s, unsigned) { return emu::from_bits<T>(s[lane & 31u]); });
}
template <class T>
inline T simd_shuffle_xor(T x, ushort mask) {
    return emu::collective(x, [mask, x](auto &s, unsigned me) { return emu::from_bits<T>(s[(me ^ mask) & 31u]); });
}
template <class T>
inline T simd_shuffle_down(T x, ushort delta) {
    return emu::collective(x, [delta, x](auto &s, unsigned me) { return me + delta < 32u ? emu::from_bits<T>(s[me + delta]) : x; });
}
template <class T>
inline T simd_shuffle_up(T x, ushort delta) {
    return emu::collective(x, [delta, x](auto &s, unsigned me) { return me >= delta ? emu::from_bits<T>(s[me - delta]) : x; });
}
inline bool simd_all(bool b) {
    return emu::collective(b, [](auto &s, unsigned) {
        for (int i = 0; i < 32; i++)
            if (!emu::from_bits<bool>(s[i])) return false;
        return true;
    });
}
inline bool simd_any(bool b) {
    return emu::collective(b, [](auto &s, unsigned) {
        for (int i = 0; i < 32; i++)
            if (emu::from_bits<bool>(s[i])) return true;
        return false;
    });
}

// ---- simdgroup_matrix: the whole tile in every lane, m[row][col] --------------------------------------------------------
template <class T, int Cols, int Rows>
struct simdgroup_matrix {
    T m[Rows][Cols];
};
template <class T, int C, int R>
inline simdgroup_matrix<T, C, R> make_filled_simdgroup_matrix(T v) {
    simdgroup_matrix<T, C, R> r;
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++) r.m[i][j] = v;
    return r;
}
// origin = (column, row) of the first element inside the source matrix of `elements_per_row` columns
template <class T, int C, int R, class U>
inline void simdgroup_load(simdgroup_matrix<T, C, R> &d, const U *src, ulong elements_per_row, ulong2 origin = ulong2{0, 0}, bool transpose = false) {
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++) {
            const ulong rr = transpose ? ulong(j) : ulong(i);
            const ulong cc = transpose ? ulong(i) : ulong(j);
            d.m[i][j] = static_cast<T>(src[(origin[1] + rr) * elements_per_row + origin[0] + cc]);
        }
}
template <class T, int C, int R, class U>
inline void simdgroup_store(const simdgroup_matrix<T, C, R> &a, U *dst, ulong elements_per_row, ulong2 origin = ulong2{0, 0}, bool transpose = false) {
    if (emu::tctx->lane != 0) return;  // one lane writes the tile; the kernel's simdgroup_barrier orders it before the reads of the others
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++) {
            const ulong rr = transpose ? ulong(j) : ulong(i);
            const ulong cc = transpose ? ulong(i) : ulong(j);
            dst[(origin[1] + rr) * elements_per_row + origin[0] + cc] = static_cast<U>(a.m[i][j]);
        }
}
template <class T, class A, class B, class Cc, int C, int R>
inline void simdgroup_multiply_accumulate(simdgroup_matrix<T, C, R> &d, const simdgroup_matrix<A, C, R> &a, const simdgroup_matrix<B, C, R> &b,
                                          const simdgroup_matrix<Cc, C, R> &c) {
    simdgroup_matrix<T, C, R> r;
    for (int i = 0; i < R; i++)
        for (int j = 0; j < C; j++) {
            T acc = static_cast<T>(c.m[i][j]);
            for (int k = 0; k < C; k++) acc += static_cast<T>(a.m[i][k]) * static_cast<T>(b.m[k][j]);
            r.m[i][j] = acc;
        }
    d = r;
}

}  // namespace metal

namespace emu {

// attributes of one thread ([[thread_position_in_grid]] ...)
struct Attrs {
    metal::uint3 tgpos, tgrid, tpg, tidv, gid;
    unsigned tid, sg, lane, nsg;
};

enum AttrKind {
    K_TGPOS,   // threadgroup_position_in_grid
    K_GID,     // thread_position_in_grid
    K_TGRID,   // threadgroups_per_grid
    K_TPG,     // threads_per_threadgroup
    K_TIDV,    // thread_position_in_threadgroup
    K_TIDX,    // thread_index_in_threadgroup
    K_SG,      // simdgroup_index_in_threadgroup
    K_LANE,    // thread_index_in_simdgroup
    K_NSG,     // simdgroups_per_threadgroup
    K_TPSG,    // threads_per_simdgroup
};

struct AttrTag {
    const Attrs *a;
    AttrKind kind;
    template <class P>
    operator P() const {
        metal::uint3 v{};
        unsigned scalar = 0;
        bool is_vec_kind = true;
        switch (kind) {
            case K_TGPOS: v = a->tgpos; break;
            case K_GID: v = a->gid; break;
            case K_TGRID: v = a->tgrid; break;
            case K_TPG: v = a->tpg; break;
            case K_TIDV: v = a->tidv; break;
            case K_TIDX: scalar = a->tid; is_vec_kind = false; break;
            case K_SG: scalar = a->sg; is_vec_kind = false; break;
            case K_LANE: scalar = a->lane; is_vec_kind = false; break;
            case K_NSG: scalar = a->nsg; is_vec_kind = false; break;
            case K_TPSG: scalar = 32; is_vec_kind = false; break;
        }
        if constexpr (std::is_same_v<P, unsigned>) {
            if (!is_vec_kind) return scalar;
            return kind == K_TPG ? v.x * v.y * v.z : v.x;  // a scalar threads_per_threadgroup is the size of the group
        } else if constexpr (__builtin_vectorelements(P) == 2) {
            return P{v.x, v.y};
        } else {
            return P{v.x, v.y, v.z};
        }
    }
};

struct Env {
    std::vector<unsigned char *> bufs;
};

struct BufTag {
    int n;
    Env *e;
    template <class P>
    operator P() const {
        return reinterpret_cast<P>(e->bufs[static_cast<size_t>(n)]);
    }
};

// runs `kfn(const Attrs&)` for every thread of every threadgroup of the grid; the groups one after the other, the threads of a group concurrently
template <class F>
inline void run_grid(const unsigned grid[3], const unsigned tpg[3], F &&kfn) {
    const int nthreads = static_cast<int>(tpg[0] * tpg[1] * tpg[2]);
    for (unsigned z = 0; z < grid[2]; z++)
        for (unsigned y = 0; y < grid[1]; y++)
            for (unsigned x = 0; x < grid[0]; x++) {
                run_threads(nthreads, [&](unsigned lt, Group &g) {
                    Attrs a{};
                    a.tgpos = metal::uint3{x, y, z};
                    a.tgrid = metal::uint3{grid[0], grid[1], grid[2]};
                    a.tpg = metal::uint3{tpg[0], tpg[1], tpg[2]};
                    a.tidv = metal::uint3{lt % tpg[0], (lt / tpg[0]) % tpg[1], lt / (tpg[0] * tpg[1])};
                    a.gid = metal::uint3{x * tpg[0] + a.tidv.x, y * tpg[1] + a.tidv.y, z * tpg[2] + a.tidv.z};
                    a.tid = lt;
                    a.sg = lt / 32u;
                    a.lane = lt % 32u;
                    a.nsg = static_cast<unsigned>(g.nsg);
                    kfn(a);
                });
            }
}
}  // namespace emu

#endif  // FLLM_MSL_EMU_H
