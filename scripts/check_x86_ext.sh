#!/usr/bin/env bash
# Hardware check of the x86 extension kernels (docs/cpu-avx512-amx.md): AVX-512 VNNI/BF16 and AMX.
#
# Run it on an x86-64 Linux machine with AVX-512 (Zen 4/5, Ice Lake or later) and, for the AMX
# checks, a Sapphire Rapids or later Xeon. It builds with `--features avx512`, runs the unit tests of
# the new module (the AVX-512 and AMX cases skip themselves when the CPU lacks the feature),
# then runs `fllm-rs selftest-cpu` for every tier the CPU supports and for a few policy overrides.
# Exit status 0 means every check passed. Nothing here downloads or runs a model; the L2 check
# (same greedy text as the avx2 tier) is described at the end of the docs page.
set -u -o pipefail
cd "$(dirname "$0")/.."
fail=0
step() { printf '\n== %s\n' "$*"; }

step "unit tests (cpu_avx512)"
cargo test --release -p fllm-engine --features avx512 cpu_avx512 || fail=1

step "build fllm-rs"
cargo build --release -p fllm-engine --features avx512 --bin fllm-rs || exit 1
bin="${CARGO_TARGET_DIR:-target}/release/fllm-rs"
[ -x "$bin" ] || { echo "cannot find $bin"; exit 1; }

step "selftest-cpu, every supported tier"
log=$(mktemp)
"$bin" selftest-cpu | tee "$log" || fail=1
tiers=$(sed -n 's/^supported tiers: //p' "$log" | head -1)
echo "tiers: $tiers"

for tier in $tiers; do
  [ "$tier" = scalar ] && continue
  step "selftest-cpu --tier $tier"
  "$bin" selftest-cpu --tier "$tier" | tail -3 || fail=1
done

step "policy overrides (each must still pass)"
run_with() {
  echo "-- $*"
  env "$@" "$bin" selftest-cpu | tail -2 || fail=1
}
run_with FLLM_X86_WIDE_FLOAT=1
run_with FLLM_X86_WIDE_FLOAT=0
run_with FLLM_X86_WIDE_INT=0
run_with FLLM_X86_BF16=0
run_with FLLM_VNNI_GEMM=all
run_with FLLM_VNNI_GEMM=0
# AMX at every prompt length, down to a single token; harmless without AMX
run_with FLLM_AMX_MIN_BATCH=1
run_with FLLM_CPU_ISA=avx512-vnni
run_with FLLM_CPU_ISA=avx512-bf16

if grep -q ' FAIL' "$log"; then fail=1; fi
rm -f "$log"
if [ "$fail" = 0 ]; then echo; echo "x86 extension checks: all passed (level L1 on this machine)"; else echo; echo "x86 extension checks: FAILURES"; fi
exit "$fail"
