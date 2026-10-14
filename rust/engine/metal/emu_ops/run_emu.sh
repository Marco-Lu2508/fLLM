#!/bin/bash
# run_emu.sh - runs the MSL kernels of fLLM (rust/engine/metal/*.metal) on the CPU against the engine's CPU backend (package p06b).
#
#   emu_ops/run_emu.sh [fixture-dir] [case-filter]
#
# 1. the unit test `emu_fixtures` of the planning layer (src/backend/metal/ops/mirror.rs) writes cases: inputs, the plan (kernel name, grid,
#    threadgroup size, parameter bytes) and the output of the CPU backend; it needs the harness crate that includes ops.rs on a Linux host
#    (HARNESS=<dir of the crate>, see the report of package p06b) - skipped when FIXTURES is given and exists;
# 2. the MSL sources are turned into C++ (`kernel` / address spaces erased by the syntax shim, `threadgroup` variables made `static`),
#    a dispatch table is generated from the kernel signatures (gen_glue.py) and everything is compiled with clang++ against msl_emu.h;
# 3. emu_run executes every case thread by thread and compares.
# Environment: CXX (clang++ with C++23), SHIM (directory with the metal_stdlib shim of mslcheck.sh), HARNESS, OUT (build directory).
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
METAL=$(dirname "$HERE")
T=$HOME/fllm-tools/tools
CXX=${CXX:-$T/rocm-root/opt/rocm-7.2.4/lib/llvm/bin/clang++}
SHIM=${SHIM:-$T/msl-shim}
export LD_LIBRARY_PATH=$(dirname "$CXX")/../lib:${LD_LIBRARY_PATH:-}
OUT=${OUT:-/tmp/p06b_emu}
FIX=${1:-$OUT/fixtures}
FILTER=${2:-}
mkdir -p "$OUT"

# --- 1. fixtures
if [ ! -f "$FIX/cases.txt" ]; then
  : "${HARNESS:?set HARNESS to the harness crate (see the report of p06b) or pass an existing fixture directory}"
  rm -rf "$FIX"; mkdir -p "$FIX"
  (cd "$HARNESS" && FLLM_EMU_DIR="$FIX" ${CS:-cargo} ${CS:+p06b-harness} test --release emu_fixtures -- --nocapture 2>&1 | tail -5)
fi

# --- 2. the MSL sources as one C++ unit: common files first, then the kernel files
KERNELS="$(ls "$METAL"/norm_*.metal "$METAL"/rope_*.metal "$METAL"/act_*.metal "$METAL"/misc_*.metal "$METAL"/kv_*.metal "$METAL"/embed_*.metal "$METAL"/attn_*.metal)"
prep() {
  sed -E \
    -e 's/\[\[[[:space:]]*host_name[[:space:]]*\([^)]*\)[[:space:]]*\]\]//g' \
    -e 's/\[\[[[:space:]]*kernel[[:space:]]*\]\]//g' \
    -e 's/^([[:space:]]*)threadgroup[[:space:]]+([A-Za-z_0-9]+[[:space:]]+[A-Za-z_0-9]+(\[[^]]*\])?;)/\1static \2/' \
    "$1"
}
{
  echo '#define FLLM_HAS_SIMDGROUP_MATRIX 1'
  prep "$METAL/common_ops.metal"
  for f in $KERNELS; do prep "$f"; done
} > "$OUT/unit.inc"
python3 "$HERE/gen_glue.py" "$OUT/glue.inc" $KERNELS
cp "$HERE/msl_emu.h" "$HERE/emu_main.cpp" "$OUT/"
"$CXX" -x c++ -std=c++23 -O1 -g0 -pthread -I"$SHIM" -I"$OUT" -Wno-unknown-attributes -Wno-attributes -Wno-unused-value -Wno-unused-variable \
  -Wno-pragma-once-outside-header -Wno-unknown-pragmas -Wno-c++26-extensions -Wno-duplicate-decl-specifier -Wno-unused-but-set-variable \
  -Wno-unused-lambda-capture -Wno-pass-failed -ferror-limit=30 "$OUT/emu_main.cpp" -o "$OUT/emu_run"

# --- 3. run
"$OUT/emu_run" "$FIX" $FILTER
