// CPU emulation of the small part of Metal Shading Language that the element-wise kernels of act_glu.metal, arch_ops.metal, misc_*.metal and
// shortconv.metal use, so that the REAL kernel text can run on a CPU test machine. A kernel is called as an ordinary function, once per grid
// thread (the grid-stride loops of the kernels are exercised with the thread counts the Rust planners compute). Derived from
// emu_recurrent/msl_emu/emu.h; reductions over threadgroups are not emulated (the kernels of this directory have none).
//
// What this checks: the algorithm, the index arithmetic, the layouts and the numerics of the formulas against references in double.
// What it does NOT check: address-space rules, attribute rules, Metal compiler acceptance, GPU limits, fast-math behaviour, simdgroup semantics.
//
// Source transformation (run_emulation.sh): [[...]] attributes removed, template instantiation lines removed, the macros below handle the rest.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

template <class T>
inline T min(T a, T b) {
    return b < a ? b : a;
}
template <class T>
inline T max(T a, T b) {
    return a < b ? b : a;
}
template <class T>
inline T clamp(T x, T lo, T hi) {
    return min(max(x, lo), hi);
}
inline float exp(float x) { return std::exp(x); }
inline float log(float x) { return std::log(x); }
inline float sqrt(float x) { return std::sqrt(x); }
inline float tanh(float x) { return std::tanh(x); }
inline float fabs(float x) { return std::fabs(x); }
inline float abs(float x) { return std::fabs(x); }
inline float rsqrt(float x) { return 1.0f / std::sqrt(x); }
inline float simd_sum(float v) { return v; }
inline float simd_max(float v) { return v; }
inline void threadgroup_barrier(mem_flags) {}
namespace precise {
inline float exp(float x) { return std::exp(x); }
inline float log(float x) { return std::log(x); }
inline float sqrt(float x) { return std::sqrt(x); }
inline float tanh(float x) { return std::tanh(x); }
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

// MSL address-space and function qualifiers; defined last so that the C++ headers above are unaffected (include the C++ headers BEFORE this file).
#define device
#define constant const
#define thread
#define threadgroup
#define kernel
