// Host check of the NVFP4 activation arithmetic in rust/engine/cuda/nvfp4_quant.cuh (no GPU needed): UE4M3 block scale encoding and E2M1 rounding (ties to even) against
// brute-force nearest-value references, bit for bit, on random blocks from 1e-4 to 1e4 with per-tensor factors from 1e-3 to 1e3 and on the exact ties.
// The device intrinsics are emulated by plain C++ (IEEE arithmetic; the kernels divide with __fdiv_rn, so this is the arithmetic of the GPU).
//   g++ -O2 -std=c++17 scripts/test_nvfp4_quant_host.cpp -o /tmp/test_nvfp4_quant_host && /tmp/test_nvfp4_quant_host
// It found that the encoder saturated at a scale of 256 instead of 448 (codes 0x78..0x7D).
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#define __device__
#define __forceinline__ inline
#define __fdiv_rn(a, b) ((a) / (b))
static inline int __float2int_rn(float x) { return (int)std::nearbyintf(x); }
#include "../rust/engine/cuda/nvfp4_quant.cuh"
static double ue_val(int c) { int e = (c >> 3) & 15, m = c & 7; return e == 0 ? m * 0.125 * std::ldexp(1.0, -6) : (1 + m * 0.125) * std::ldexp(1.0, e - 7); }
static int ue_ref(double x) {
    if (!(x > 0)) return 0;
    if (x >= 448) return 0x7E;
    int best = 0;
    for (int c = 1; c <= 0x7E; c++) { double d = std::fabs(ue_val(c) - x), db = std::fabs(ue_val(best) - x); if (d < db || (d == db && c % 2 == 0)) best = c; }
    return best;
}
static const double G[8] = {0, .5, 1, 1.5, 2, 3, 4, 6};
static int e2_ref(double a) {
    int best = 0;
    for (int i = 1; i < 8; i++) { double d = std::fabs(G[i] - a), db = std::fabs(G[best] - a); if (d < db || (d == db && i % 2 == 0)) best = i; }
    return best;
}
int main() {
    std::mt19937 rng(1);
    std::uniform_real_distribution<float> U(-1, 1);
    long bad_sc = 0, bad_q = 0, n = 0;
    // exact ties and neighbours of the E2M1 grid
    const float mids[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.f};
    for (int i = 0; i < 7; i++) if ((int)e2m1_idx(mids[i]) != e2_ref(mids[i])) { printf("tie %g: %u vs %d\n", mids[i], e2m1_idx(mids[i]), e2_ref(mids[i])); bad_q++; }
    for (int code = 0; code <= 0x7E; code++) if ((int)f_to_ue4m3((float)ue_val(code)) != code) { printf("code %d\n", code); bad_sc++; }
    for (int it = 0; it < 400000; it++) {
        float v[16];
        const float mag = std::pow(10.f, U(rng) * 4.f);  // 1e-4 .. 1e4
        const float gs = (it % 3 == 0) ? 1.f : std::pow(10.f, U(rng) * 3.f);
        for (int i = 0; i < 16; i++) v[i] = U(rng) * mag * ((rng() % 8 == 0) ? 4.f : 1.f);
        if (it % 50 == 0) for (int i = 0; i < 16; i++) v[i] = 0;
        uint32_t w[2];
        const uint32_t sb = fp4_quantize_values(v, w, gs);
        float amax = 0; for (int i = 0; i < 16; i++) amax = std::fmax(amax, std::fabs(v[i]));
        const int rc = ue_ref((double)(amax * gs * (1.f / 6.f)));
        if ((int)sb != rc) { bad_sc++; if (bad_sc < 5) printf("scale %u vs %d (x=%g)\n", sb, rc, amax * gs / 6); }
        const float sd = (float)ue_val(rc), inv = sd > 0 ? gs / sd : 0.f;
        for (int h = 0; h < 2; h++) for (int q = 0; q < 8; q++) {
            const float t = v[8 * h + q] * inv;
            const int ref = e2_ref(std::fabs((double)t)) | (t < 0 ? 8 : 0);
            const int got = (w[h] >> (4 * q)) & 15;
            n++;
            if (ref != got) { bad_q++; if (bad_q < 5) printf("q %d vs %d (t=%g)\n", got, ref, t); }
        }
    }
    printf("values %ld, scale mismatches %ld, nibble mismatches %ld\n", n, bad_sc, bad_q);
    return bad_sc || bad_q;
}
