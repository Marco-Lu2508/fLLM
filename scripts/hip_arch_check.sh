#!/usr/bin/env bash
# Checks rust/engine/hip/arch_table.csv against the compiler: for every row the device pass of hipcc is run through the preprocessor with
# --offload-arch=<gfx> and the macros of cuda/hip_compat.h (FLLM_HIP_CDNA / _RDNA3 / _RDNA4 / _SDOT4 / _SUDOT4 / _DOT2) and the wave size are
# compared with the row. The generic target of every row is checked too: its macros may only promise what every member of the family has.
# Also builds and runs the host-only test of hip/hip_arch.h against the csv. Needs hipcc (ROCm) and a C++ compiler; no GPU.
#   scripts/hip_arch_check.sh
set -u
here="$(cd "$(dirname "$0")/.." && pwd)"
eng="$here/rust/engine"
rocm="${ROCM_PATH:-/opt/rocm}"
cc="$(command -v hipcc || true)"; [ -n "$cc" ] || cc="$rocm/lib/llvm/bin/clang++"
[ -x "$cc" ] || cc="$rocm/core/lib/llvm/bin/clang++"
work="$(mktemp -d)"; trap 'rm -rf "$work"' EXIT
fail=0

echo "== hip_arch.h against arch_table.csv"
"${CXX:-c++}" -std=c++17 -Wall -o "$work/hip_arch_test" "$eng/hip/hip_arch_test.cpp" && "$work/hip_arch_test" "$eng/hip/arch_table.csv" || fail=1

macros_of() {  # prints "use dot4 dot2 wave" for a target, or "unsupported"
  local t="$1" out
  echo '#include "cuda/hip_compat.h"' > "$work/m.hip"
  out="$("$cc" -x hip --cuda-device-only -E -dM --offload-arch="$t" -I"$eng" "$work/m.hip" 2>/dev/null)" || { echo unsupported; return; }
  local use=other dot4=none dot2=0 wave=64
  grep -q '^#define FLLM_HIP_CDNA ' <<<"$out" && use=cdna
  grep -q '^#define FLLM_HIP_RDNA3 ' <<<"$out" && use=rdna3
  grep -q '^#define FLLM_HIP_RDNA4 ' <<<"$out" && use=rdna4
  grep -q '^#define FLLM_HIP_SDOT4 ' <<<"$out" && dot4=sdot4
  grep -q '^#define FLLM_HIP_SUDOT4 ' <<<"$out" && dot4=sudot4
  grep -q '^#define FLLM_HIP_DOT2 ' <<<"$out" && dot2=1
  # the wave size the target compiles for: .amdhsa_wavefront_size32 of the assembly of an empty kernel
  echo '__global__ void k(int *o) { o[0] = 1; }' > "$work/w.hip"
  wave32="$("$cc" -x hip --cuda-device-only -S -O1 --offload-arch="$t" "$work/w.hip" -o - 2>/dev/null | grep -o 'amdhsa_wavefront_size32 [01]' | head -1 | awk '{print $2}')"
  if [ "$wave32" = 1 ]; then wave=32; elif [ "$wave32" = 0 ] || [ -z "$wave32" ]; then wave=64; fi
  echo "$use $dot4 $dot2 $wave"
}

echo "== csv rows against the compiler ($("$cc" --version | head -1))"
declare -A generic_of_row generic_seen
tail -n +2 "$eng/hip/arch_table.csv" | tr -d '\r' > "$work/rows.csv"
while IFS=, read -r gfx family wave matrix use dot4 dot2 generic; do
  got="$(macros_of "$gfx")"
  if [ "$got" = unsupported ]; then echo "SKIP $gfx (compiler does not know it)"; continue; fi
  want="$use $dot4 $dot2 $wave"
  if [ "$got" = "$want" ]; then echo "ok   $gfx: $got"; else echo "FAIL $gfx: compiler says '$got', csv says '$want'"; fail=1; fi
  [ "$generic" = "-" ] || generic_seen[$generic]=1
done < "$work/rows.csv"

echo "== generic targets may only promise what all their members have"
for g in "${!generic_seen[@]}"; do
  got="$(macros_of "$g")"
  if [ "$got" = unsupported ]; then echo "SKIP $g (compiler does not know it)"; continue; fi
  read -r guse gdot4 gdot2 gwave <<<"$got"
  bad=""
  while IFS=, read -r gfx family wave matrix use dot4 dot2 generic; do
    [ "$generic" = "$g" ] || continue
    [ "$guse" = other ] || [ "$guse" = "$use" ] || bad="$bad $gfx(use $use)"
    [ "$gdot4" = none ] || [ "$gdot4" = "$dot4" ] || bad="$bad $gfx(dot4 $dot4)"
    [ "$gdot2" = 0 ] || [ "$dot2" = 1 ] || bad="$bad $gfx(dot2)"
    [ "$gwave" = "$wave" ] || bad="$bad $gfx(wave $wave)"
  done < "$work/rows.csv"
  if [ -z "$bad" ]; then echo "ok   $g: $got"; else echo "FAIL $g ($got) promises more than:$bad"; fail=1; fi
done
exit $fail
