#!/bin/bash
# CPU emulation of the kernels of package 6c (moe_*, gate_*, delta_*, mamba_*): runs the REAL .metal text on the CPU, one std::thread per
# GPU thread, with a real threadgroup barrier, and compares with independent references (see emu_main.cpp, msl_emu/emu.h).
#
# usage: run_emulation.sh <vectors.bin>        (vectors.bin: see harness/src/bin/gen_vectors.rs)
# environment: CXX (default g++), TSAN=1 (build with -fsanitize=thread: a missing barrier becomes a reported data race),
#              EMU_BUILD (work directory, default /tmp/p06c-emu), EMU_SEED (random seed),
#              EMU_METAL (directory with the .metal sources, default the parent of this directory; used for mutation tests)
# Needs only a C++20 compiler; nothing is installed and no GPU is used. Exit code 0 only if every check passes.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
METAL=${EMU_METAL:-$(cd "$HERE/.." && pwd)}
B=${EMU_BUILD:-/tmp/p06c-emu}
mkdir -p "$B"
CXX=${CXX:-g++}
VECTORS=${1:-}

# MSL -> C++: drop the template instantiation lines (checked by mslcheck.sh instead), the loop-unroll pragma and every [[attribute]];
# `static constant constexpr const T` loses its duplicate `const`; indented `threadgroup T name[..]` (kernel-scope threadgroup memory)
# becomes a static array shared by the emulated threads of the (one at a time) threadgroup.
prep() {
  sed -E \
    -e '/^template \[\[host_name/d' \
    -e 's/_Pragma\("clang loop unroll\(full\)"\)//' \
    -e 's/\[\[[^]]*\]\]//g' \
    -e 's/constant constexpr const /constexpr /' \
    -e 's/^([[:space:]]+)threadgroup (float|uint) /\1static \2 /' "$1"
}
for f in common_rec delta_conv delta_rec gate_attention mamba_conv mamba_norm mamba_scan1 mamba_scan2 moe_combine moe_linear moe_route moe_rows; do
  prep "$METAL/$f.metal" > "$B/$f.inc"
done

FLAGS="-std=c++20 -O1 -g -pthread -Wall -Wno-unknown-pragmas -Wno-attributes -Wno-unused-variable -Wno-unused-but-set-variable"
if [ "${TSAN:-0}" = 1 ]; then FLAGS="$FLAGS -fsanitize=thread"; fi
"$CXX" $FLAGS -I"$HERE/msl_emu" -I"$B" "$HERE/emu_main.cpp" -o "$B/emu_main"
if [ "${TSAN:-0}" = 1 ]; then
  # ThreadSanitizer of recent compilers does not start with high-entropy ASLR (kernels with vm.mmap_rnd_bits > 28): run without address randomisation
  exec setarch "$(uname -m)" -R "$B/emu_main" "$VECTORS" ${EMU_SEED:-}
fi
exec "$B/emu_main" "$VECTORS" ${EMU_SEED:-}
