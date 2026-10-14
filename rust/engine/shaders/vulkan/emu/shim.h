// CPU emulation shim for the GLSL compute shaders of package 2: just enough of GLSL (vec4/ivec4, builtins, barrier) so that the
// preprocessed shader text compiles as C++20 and runs with one std::thread per invocation of a workgroup (std::barrier = barrier()).
#pragma once
#include <algorithm>
#include <barrier>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <thread>
#include <vector>

typedef unsigned int uint;
struct uvec3 { uint x, y, z; };
struct ivec4;
struct vec2 {
    float x, y;
    float operator[](uint i) const { return i ? y : x; }
};
struct vec4 {
    float x, y, z, w;
    vec4() : x(0), y(0), z(0), w(0) {}
    explicit vec4(float s) : x(s), y(s), z(s), w(s) {}
    vec4(float a, float b, float c, float d) : x(a), y(b), z(c), w(d) {}
    explicit vec4(const ivec4& v);
    float& operator[](uint i) { return (&x)[i]; }
    const float& operator[](uint i) const { return (&x)[i]; }
};
struct ivec4 {
    int x, y, z, w;
    ivec4() : x(0), y(0), z(0), w(0) {}
    explicit ivec4(int s) : x(s), y(s), z(s), w(s) {}
    ivec4(int a, int b, int c, int d) : x(a), y(b), z(c), w(d) {}
};
inline vec4::vec4(const ivec4& v) : x((float)v.x), y((float)v.y), z((float)v.z), w((float)v.w) {}
inline ivec4 operator<<(ivec4 a, ivec4 b) { return ivec4((int)((unsigned)a.x << b.x), (int)((unsigned)a.y << b.y), (int)((unsigned)a.z << b.z), (int)((unsigned)a.w << b.w)); }
inline ivec4 operator>>(ivec4 a, int s) { return ivec4(a.x >> s, a.y >> s, a.z >> s, a.w >> s); }
inline vec4 operator+(vec4 a, vec4 b) { return vec4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w); }
inline vec4 operator+(vec4 a, float s) { return vec4(a.x + s, a.y + s, a.z + s, a.w + s); }
inline vec4 operator*(vec4 a, vec4 b) { return vec4(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }
inline vec4 operator*(vec4 a, float s) { return vec4(a.x * s, a.y * s, a.z * s, a.w * s); }
inline float dot(vec4 a, vec4 b) { return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w; }

static float half_to_float(uint16_t h) {
    uint s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v;
    if (e == 0) v = std::ldexp((float)m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp((float)(m + 1024), (int)e - 25);
    return s ? -v : v;
}
inline vec2 unpackHalf2x16(uint v) { return vec2{half_to_float(v & 0xffff), half_to_float(v >> 16)}; }
inline float uintBitsToFloat(uint v) { float f; memcpy(&f, &v, 4); return f; }
inline uint floatBitsToUint(float f) { uint v; memcpy(&v, &f, 4); return v; }
inline int bitfieldExtract(int v, int off, int bits) { return (int)((unsigned)v << (32 - off - bits)) >> (32 - bits); }
inline float roundEven(float x) { return std::nearbyint(x); }
inline int clamp(int x, int lo, int hi) { return x < lo ? lo : (x > hi ? hi : x); }
using std::abs;
using std::max;

inline uvec3 gl_NumWorkGroups;
inline uvec3 gl_WorkGroupID;
inline thread_local uvec3 gl_LocalInvocationID;
inline thread_local uvec3 gl_GlobalInvocationID;
inline std::barrier<>* g_bar;
inline void barrier() { g_bar->arrive_and_wait(); }

// ---- subgroups and cooperative matrices (emulated redundantly: every invocation holds whole 16 x 16 matrices) ----
#ifndef SUBGROUP_SIZE
#define SUBGROUP_SIZE 32
#endif
inline thread_local uint gl_SubgroupID;
inline thread_local uint gl_SubgroupInvocationID;
constexpr uint gl_SubgroupSize = SUBGROUP_SIZE;
inline void subgroupBarrier() { barrier(); }            // every invocation of the workgroup reaches it in the emulated shader
inline void subgroupMemoryBarrierShared() {}
struct float16_t {
    float v;
    float16_t() : v(0) {}
    float16_t(float x) : v((float)(_Float16)x) {}
    operator float() const { return v; }
};
constexpr int gl_ScopeSubgroup = 3, gl_MatrixUseA = 0, gl_MatrixUseB = 1, gl_MatrixUseAccumulator = 2;
constexpr int gl_CooperativeMatrixLayoutRowMajor = 0, gl_CooperativeMatrixLayoutColumnMajor = 1;
template <class T, int Scope, int R, int C, int Use>
struct coopmat {
    T v[R][C];
    coopmat() { for (int i = 0; i < R; ++i) for (int j = 0; j < C; ++j) v[i][j] = T(0.0f); }
    explicit coopmat(double s) { for (int i = 0; i < R; ++i) for (int j = 0; j < C; ++j) v[i][j] = T((float)s); }
};
template <class M, class T>
void coopMatLoad(M& m, T* arr, uint off, uint stride, int layout) {
    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 16; ++c) m.v[r][c] = layout == 0 ? arr[off + r * stride + c] : arr[off + c * stride + r];
}
template <class M, class T>
void coopMatStore(const M& m, T* arr, uint off, uint stride, int layout) {
    for (int r = 0; r < 16; ++r)
        for (int c = 0; c < 16; ++c) { if (layout == 0) arr[off + r * stride + c] = m.v[r][c]; else arr[off + c * stride + r] = m.v[r][c]; }
}
template <class MA, class MB, class MC>
MC coopMatMulAdd(const MA& a, const MB& b, const MC& c) {
    MC out = c;
    for (int i = 0; i < 16; ++i)
        for (int j = 0; j < 16; ++j) {
            float s = out.v[i][j];
            for (int k = 0; k < 16; ++k) s += (float)a.v[i][k] * (float)b.v[k][j];
            out.v[i][j] = s;
        }
    return out;
}

// runs `fn` (the shader's main) for every workgroup of a gx x gy x 1 grid, one thread per invocation
template <class F>
void dispatch(F fn, uint gx, uint gy, uint wg) {
    gl_NumWorkGroups = {gx, gy, 1};
    for (uint y = 0; y < gy; ++y)
        for (uint x = 0; x < gx; ++x) {
            gl_WorkGroupID = {x, y, 0};
            std::barrier<> bar(wg);
            g_bar = &bar;
            uvec3 id = gl_WorkGroupID;
            std::vector<std::thread> threads;
            for (uint t = 0; t < wg; ++t)
                threads.emplace_back([=] {
                    gl_LocalInvocationID = {t, 0, 0};
                    gl_GlobalInvocationID = {id.x * wg + t, id.y, 0};
                    gl_SubgroupID = t / gl_SubgroupSize;
                    gl_SubgroupInvocationID = t % gl_SubgroupSize;
                    fn();
                });
            for (auto& th : threads) th.join();
        }
}
