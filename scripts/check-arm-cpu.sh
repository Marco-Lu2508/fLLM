#!/usr/bin/env bash
# Verification run for the ARM CPU kernels (docs/cpu-arm.md). Run it on an AArch64 machine
# (Linux, Android/Termux, macOS on Apple silicon): it builds the engine, runs the CPU unit tests
# and `fllm-rs selftest-cpu` for every tier the CPU supports, and prints a summary.
#
#   scripts/check-arm-cpu.sh            # tests + selftest for every supported tier
#   scripts/check-arm-cpu.sh --quick    # selftest only (skip cargo test)
#
# Logs go to target/check-arm-cpu/. The exit status is non-zero when any step failed.
set -u

cd "$(dirname "$0")/.." || exit 2
quick=0
[ "${1:-}" = "--quick" ] && quick=1

case "$(uname -m)" in
  aarch64 | arm64) ;;
  *)
    echo "check-arm-cpu.sh: this is $(uname -m), not an AArch64 machine; the ARM kernels would not run." >&2
    exit 2
    ;;
esac

logs=target/check-arm-cpu
mkdir -p "$logs"
failed=0
summary=()

step() { # step <name> <command...>: run, log, record
  local name=$1
  shift
  if "$@" >"$logs/$name.log" 2>&1; then
    summary+=("ok      $name")
  else
    summary+=("FAILED  $name  (see $logs/$name.log)")
    failed=1
  fi
}

if [ "$quick" = 0 ]; then
  step unit-cpu-arm cargo test --release -p fllm-engine cpu_arm
  step unit-cpu-delta cargo test --release -p fllm-engine cpu_delta
  step unit-cpu-all cargo test --release -p fllm-engine cpu_
fi
step build cargo build --release -p fllm-engine --bin fllm-rs

bin=target/release/fllm-rs
if [ ! -x "$bin" ]; then
  printf '%s\n' "${summary[@]}"
  echo "no binary at $bin" >&2
  exit 1
fi

# Features and tiers as the engine sees them.
"$bin" selftest-cpu --help >"$logs/help.log" 2>&1
tiers=$("$bin" selftest-cpu 2>&1 | tee "$logs/selftest-all.log" | sed -n 's/^supported tiers: //p')
if grep -q ' 0 failed' "$logs/selftest-all.log" && grep -q 'checks, ' "$logs/selftest-all.log"; then
  summary+=("ok      selftest-cpu (all tiers: ${tiers:-none})")
else
  summary+=("FAILED  selftest-cpu (see $logs/selftest-all.log)")
  failed=1
fi

# Every tier on its own, so a failure names the tier even when the full run is long.
for tier in $tiers; do
  step "selftest-$tier" "$bin" selftest-cpu --tier "$tier"
done

# The SVE tier is opt-in; selftest-cpu enables it itself on CPUs with 256-bit or wider vectors.
if grep -q 'SVE vector length' "$logs/selftest-all.log"; then
  summary+=("info    $(grep 'SVE vector length' "$logs/selftest-all.log")")
fi

# FLLM_CPU_ISA must step the dispatch down: the lowest tier must still pass.
FLLM_CPU_ISA=neon step selftest-cap-neon "$bin" selftest-cpu --tier neon

echo
echo "ARM CPU kernel check on $(uname -sm), CPU: $(grep -m1 -i -E 'model name|Features' /proc/cpuinfo 2>/dev/null || sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown)"
printf '%s\n' "${summary[@]}"
exit "$failed"
