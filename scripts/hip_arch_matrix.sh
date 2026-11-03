#!/usr/bin/env bash
# Compile-only matrix of the HIP kernels: every source of rust/engine/{cuda,hip} is compiled for ONE gfx target at a time (device pass only), so
# a broken GPU family shows up as its own line. No GPU is needed and nothing is run.
#
#   scripts/hip_arch_matrix.sh [--opt O0|O1|O3] [--jobs N] [--list] [--summary FILE] [--keep DIR] gfx1030 gfx11-generic ...
#   scripts/hip_arch_matrix.sh --default                    # the default list of rust/engine/build.rs (Linux)
#
# Targets the installed compiler does not know are reported as SKIP (not as a failure of fLLM). Exit status 1 when a target FAILs.
# Needs hipcc (or the ROCm clang++) in PATH / ROCM_PATH.
set -u
here="$(cd "$(dirname "$0")/.." && pwd)"
opt=O1; jobs=2; summary=""; keep=""; archs=()
while [ $# -gt 0 ]; do
  case "$1" in
    --opt) opt="$2"; shift 2 ;;
    --jobs) jobs="$2"; shift 2 ;;
    --summary) summary="$2"; shift 2 ;;
    --keep) keep="$2"; shift 2 ;;
    -h|--help) sed -n 2,10p "$0"; exit 0 ;;
    *) archs+=("$1"); shift ;;
  esac
done
[ ${#archs[@]} -gt 0 ] || { echo "no targets given" >&2; exit 2; }
rocm="${ROCM_PATH:-/opt/rocm}"
cc="$(command -v hipcc || true)"; [ -n "$cc" ] || cc="$rocm/lib/llvm/bin/clang++"
[ -x "$cc" ] || cc="$rocm/core/lib/llvm/bin/clang++"
work="$(mktemp -d)"; [ -z "$keep" ] || mkdir -p "$keep"
# the source list of hip_kernels() in rust/engine/build.rs, so that this check never drifts from the build
sources="$(python3 "$here/scripts/hip_sources.py" "$here/rust/engine/build.rs")"
[ -n "$sources" ] || { echo "cannot read the source list from build.rs" >&2; exit 2; }
echo "compiler: $("$cc" --version | head -1)"

one_target() {
  arch="$1"
  log="$work/$arch.log"; : > "$log"
  # does the compiler know the target at all?
  probe="$work/$arch.probe.hip"; echo '__global__ void k() {}' > "$probe"
  if ! "$cc" -x hip --cuda-device-only -c --offload-arch="$arch" "$probe" -o "$work/$arch.probe.o" >> "$log" 2>&1; then
    echo "SKIP $arch (the compiler or its device libraries do not know this target)"; return 0
  fi
  failed=""
  for s in $sources; do
    flags="-ffast-math"; [ "$s" = "cuda/kv_quant.cu" ] && flags=""
    if ! (cd "$here/rust/engine" && "$cc" -x hip --cuda-device-only -c -$opt -std=c++17 -DFLLM_NO_FP4 -Wno-unused-value -Wno-unused-result $flags \
          --offload-arch="$arch" "$s" -o "$work/$arch.$(basename "$s").o") >> "$log" 2>&1; then
      failed="$failed $s"
    elif [ -n "$keep" ]; then
      cp "$work/$arch.$(basename "$s").o" "$keep/" 2>/dev/null
    fi
    rm -f "$work/$arch.$(basename "$s").o"
  done
  if [ -z "$failed" ]; then echo "PASS $arch"; else echo "FAIL $arch:$failed"; cp "$log" "${keep:-$work}/$arch.log" 2>/dev/null; fi
}
export -f one_target
export cc here opt work keep sources

out="$work/summary.txt"
printf '%s\n' "${archs[@]}" | xargs -P "$jobs" -I{} bash -c 'one_target {}' | sort > "$out"
cat "$out"
[ -z "$summary" ] || cp "$out" "$summary"
[ -n "$keep" ] || echo "(logs of failed targets: $work/<target>.log)"
! grep -q '^FAIL' "$out"
