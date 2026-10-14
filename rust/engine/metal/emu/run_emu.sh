#!/bin/bash
# CPU emulation of the quantized linear kernels (package p06a): builds the MSL text as C++ and runs the real kernel templates on emulated
# threadgroups against the vectors dumped by the Rust test `dump_vectors_for_the_msl_emulation`. No GPU, no Metal compiler involved.
#   usage: run_emu.sh [tag...]            (default: every tag found in $DUMP_DIR)
# environment: EMU_WORK (default /tmp/fllm-metal-emu), DUMP_DIR (default $EMU_WORK/dump), JOBS (default 6),
#              MSL_SHIM (default ~/fllm-tools/tools/msl-shim, the syntax shim of mslcheck.sh), CXX (a clang++ with _Float16 and
#              __builtin_elementwise_*; default the ROCm clang of ~/fllm-tools/tools).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
METAL=$(dirname "$HERE")
W=${EMU_WORK:-/tmp/fllm-metal-emu}
D=${DUMP_DIR:-$W/dump}
SHIM=${MSL_SHIM:-$HOME/fllm-tools/tools/msl-shim}
LLVM=$HOME/fllm-tools/tools/rocm-root/opt/rocm-7.2.4/lib/llvm
CXX=${CXX:-$LLVM/bin/clang++}
export LD_LIBRARY_PATH=$LLVM/lib:${LD_LIBRARY_PATH:-}
mkdir -p "$W"

# the same text preparation as mslcheck.sh (attributes that C++ does not know, half literals), plus: the function constant becomes a
# plain global that the driver sets
prep() {
  printf '#line 1 "%s"\n' "$1"
  sed -E \
    -e 's/^constant uint q_fc_nb \[\[function_constant\(500\)\]\];/uint q_fc_nb = 1;/' \
    -e 's/\[\[[[:space:]]*host_name[[:space:]]*\([^)]*\)[[:space:]]*\]\]//g' \
    -e 's/\[\[[[:space:]]*kernel[[:space:]]*\]\]//g' \
    -e 's/[[:space:]]*\[\[[[:space:]]*function_constant[[:space:]]*\([^)]*\)[[:space:]]*\]\][[:space:]]*;/ = {};/g' \
    -e 's/\[\[[[:space:]]*threadgroup[[:space:]]*\(/[[tg_buffer(/g' \
    -e 's/(^|[^A-Za-z0-9_.])([0-9]+\.[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?[hH]\b/\1\2\3f16/g' "$METAL/$1"
  printf '\n'
}
{ for f in common.metal common_q_gemm.metal common_q_gemv.metal q_prepass.metal; do prep $f; done; } > "$W/unit.cpp"

"$CXX" -x c++ -std=c++23 -O1 -g0 -pthread -fsanitize=alignment -fsanitize-trap=alignment \
  -Wno-attributes -Wno-unknown-attributes -Wno-duplicate-decl-specifier -Wno-c++26-extensions -Wno-unused-value -Wno-unused-variable \
  -Wno-pragma-once-outside-header -Wno-unknown-pragmas -Wno-unused-parameter -ferror-limit=30 \
  -I"$HERE" -I"$SHIM" -I"$W" "$HERE/harness.cpp" -o "$W/emu" 2> "$W/build.log" || { echo "BUILD FAILED"; head -60 "$W/build.log"; exit 1; }
echo "built $W/emu ($(grep -c warning "$W/build.log" || true) warnings)"

if [ $# -gt 0 ]; then TAGS="$*"; else TAGS=$(cd "$D" && ls *.bin | sed -E 's/(_big|_tail|_odd)?\.bin$//' | sort -u | tr '\n' ' '); fi
rm -f "$W"/out_*.txt
echo $TAGS | tr ' ' '\n' | xargs -P "${JOBS:-6}" -I{} sh -c "'$W/emu' '$D' {} > '$W/out_{}.txt' 2>&1; echo \"{} exit=\$?\""
echo "---- summary"
cat "$W"/out_*.txt | grep -E "FAIL|MISMATCH|STRAY|checks," | grep -v " 0 failures" | head -80 || true
grep -h "^TOTAL" "$W"/out_*.txt | awk '{c+=$2; f+=$4} END {print "ALL TAGS: " c " checks, " f " failures"}'
