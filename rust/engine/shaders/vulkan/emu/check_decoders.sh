#!/bin/bash
# CPU check of the weight chunk decoders of shaders/vulkan/_q_fmt_*.glsl: the GLSL function bodies are turned into C++ by a mechanical text
# transform (sed) and run on the device-layout bytes + reference dequantization that the Rust test `dump_vectors_for_the_glsl_transpile_check`
# writes. Not a GPU test: it executes the shader TEXT on the CPU (level L0 stays L0).
#   usage: check_decoders.sh            (dumps in $DUMP_DIR, default $EMU_WORK/dump; see README.md for how to create them)
# needs: g++ (C++17), sed
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
SH=$(dirname "$HERE")
W=${EMU_WORK:-/tmp/fllm-vk-emu}
D=${DUMP_DIR:-$W/dump}
mkdir -p "$W/dec" && cd "$W/dec"
cat > shim.h <<'EOF'
#include <cstdint>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
typedef unsigned int uint;
static uint* w;
static struct { uint off1, off2, off3; } p;
struct vec2 { float x, y; float operator[](uint i) const { return i ? y : x; } };
static float half_to_float(uint16_t h) {
    uint s = (h >> 15) & 1, e = (h >> 10) & 31, m = h & 1023;
    float v;
    if (e == 0) v = std::ldexp((float)m, -24);
    else if (e == 31) v = m ? NAN : INFINITY;
    else v = std::ldexp((float)(m + 1024), (int)e - 25);
    return s ? -v : v;
}
static vec2 unpackHalf2x16(uint v) { return vec2{half_to_float(v & 0xffff), half_to_float(v >> 16)}; }
static float uintBitsToFloat(uint v) { float f; memcpy(&f, &v, 4); return f; }
static int bitfieldExtract(int v, int off, int bits) { return (int)((unsigned)v << (32 - off - bits)) >> (32 - bits); }
// _q_common.glsl helpers, textually the same bodies
static float f16at(uint base, uint i) { return unpackHalf2x16(w[base + (i >> 1)])[i & 1u]; }
static uint sub8(uint x, uint c) { return ((x | 0x80808080u) - c * 0x01010101u) ^ 0x80808080u; }
EOF
run_fmt() { # tag file-stem extra-g++-flags...
  tag=$1; f=$2; shift 2
  sed -e 's/out \(uint\|float\) \([a-z0-9]*\)/\1\& \2/g' \
      -e '/^const int KV/ s/= int\[16\](/= {/' -e '/^const int KV/ s/);$/};/' \
      "$SH/_q_fmt_$f.glsl" > fmt_$tag.h
  cat > run_$tag.cpp <<EOF
#include "shim.h"
#include "fmt_$tag.h"
int main(int argc, char** argv) {
    FILE* fp = fopen(argv[1], "rb"); if (!fp) { perror("open"); return 2; }
    uint hdr[8]; if (fread(hdr, 4, 8, fp) != 8) return 2;
    uint rows = hdr[0], cols = hdr[1], qk = hdr[2], nb = hdr[3]; p.off1 = hdr[4]; p.off2 = hdr[5]; p.off3 = hdr[6];
    std::vector<uint> buf(hdr[7] / 4); if (fread(buf.data(), 4, buf.size(), fp) != buf.size()) return 2;
    std::vector<float> ref((size_t)rows * cols); if (fread(ref.data(), 4, ref.size(), fp) != ref.size()) return 2;
    w = buf.data();
    if (qk != QK) { printf("QK mismatch %u vs %d\n", qk, QK); return 3; }
    size_t bad = 0, n = 0; double worst = 0;
    for (uint r = 0; r < rows; ++r) for (uint kb = 0; kb < nb; ++kb) for (uint cin = 0; cin < qk / 8; ++cin) {
        uint p0, p1; float ws, wm;
        wchunk(r * nb + kb, cin, p0, p1, ws, wm);
        uint pk[2] = {p0, p1};
        for (int i = 0; i < 8; ++i) {
            int q = bitfieldExtract((int)pk[i / 4], 8 * (i % 4), 8);
            float got = ws * (float)q + wm, want = ref[(size_t)r * cols + kb * qk + cin * 8 + i];
            double tol = 1e-6 * fmax(fabs(ws * (float)q), fabs(wm)) + 1e-9;
            double err = fabs((double)got - want);
            if (err > tol) { if (bad < 5) printf("  MISMATCH r%u kb%u c%u i%d: got %g want %g\n", r, kb, cin, i, got, want); ++bad; }
            worst = fmax(worst, err); ++n;
        }
    }
    printf("%-8s values=%zu mismatches=%zu worst_abs_err=%g\n", "$tag", n, bad, worst);
    return bad ? 1 : 0;
}
EOF
  g++ -std=c++17 -O1 "$@" run_$tag.cpp -o run_$tag 2> err_$tag.txt || { echo "C++ build failed for $tag"; head -20 err_$tag.txt; return 1; }
  ./run_$tag "$D/$tag.bin"
}
fails=0
for t in q4_0 q4_1 q5_0 q5_1 q8_0 q2_k q3_k q4_k q5_k q6_k iq4_nl mxfp4 nvfp4; do run_fmt $t $t || fails=$((fails+1)); done
for g in 32 64 128; do run_fmt q4z$g q4z -DQ4Z_GROUP=$g || fails=$((fails+1)); done
echo "decoder failures=$fails"
[ $fails = 0 ]
