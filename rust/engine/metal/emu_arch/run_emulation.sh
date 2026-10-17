#!/bin/bash
# CPU emulation of the element-wise kernels act_glu.metal and arch_ops.metal: runs the REAL .metal text on the CPU and compares it with
# references in double (see emu_main.cpp and msl_emu/emu.h for what that proves and what it does not).
#
# usage: run_emulation.sh
# environment: CXX (default g++; needs C++20 and _Float16: g++ >= 12 or clang >= 15), EMU_BUILD (work directory),
#              EMU_METAL (directory with the .metal sources, default the parent of this directory; used for mutation tests)
# Exit code 0 only if every check passes.
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
METAL=${EMU_METAL:-$(cd "$HERE/.." && pwd)}
B=${EMU_BUILD:-/tmp/metal-emu-arch}
CXX=${CXX:-g++}
mkdir -p "$B/msl_emu"
cp "$HERE"/msl_emu/* "$B/msl_emu/"
# MSL -> C++: drop the template instantiation lines, the loop-unroll pragma and every [[attribute]]; a duplicate `const` after
# `constant constexpr` is removed.
prep() {
  sed -E \
    -e '/^template \[\[host_name/d' \
    -e 's/_Pragma\("clang loop unroll\(full\)"\)//' \
    -e 's/\[\[[^]]*\]\]//g' \
    -e 's/constant constexpr const /constexpr /' "$1"
}
for f in common_ops act_glu arch_ops; do
  prep "$METAL/$f.metal" > "$B/$f.inc"
done
cp "$HERE/emu_main.cpp" "$B/"
FLAGS="-std=c++20 -O1 -g -Wall -Wno-unknown-pragmas -Wno-attributes -Wno-unused-variable -Wno-unused-but-set-variable -Wno-unused-function"
"$CXX" $FLAGS -I"$B/msl_emu" -I"$B" "$B/emu_main.cpp" -o "$B/emu_main"
exec "$B/emu_main"
