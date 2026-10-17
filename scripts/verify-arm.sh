#!/usr/bin/env bash
# Verification run of the ARM CPU kernels (NEON, dotprod, i8mm, SVE) on an AArch64 machine: Apple M-series, Graviton, Raspberry Pi,
# Ampere, Snapdragon X. See docs/backends-verify.md, section ARM. There is no GPU here: the "device" is the CPU, the reference is the
# lowest tier (FLLM_CPU_ISA=neon), and every tier the CPU supports is compared with it.
#
#   scripts/verify-arm.sh --model /models/llama-q4_k_m.gguf
#   V_ARM_QUICK=1 scripts/verify-arm.sh --skip bench          # check-arm-cpu.sh --quick: selftest only
#
# check-arm-cpu.sh builds, runs the CPU unit tests and `fllm-rs selftest-cpu` for every tier; its logs go to
# target/check-arm-cpu/. The SVE tier is opt-in and is only compared when the CPU reports 256-bit or wider vectors (FLLM_ARM_SVE=1).
# Level of this script: L0.
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=scripts/verify-common.sh
. "$here/verify-common.sh"

V_BACKEND=arm
V_FEATURES=
V_DEVICE=cpu
V_REF_ENV="FLLM_CPU_ISA=neon"
V_PHASES="build devices pre text bench"

pre_steps() {
  local script="$V_ROOT/scripts/check-arm-cpu.sh"
  if [ ! -f "$script" ]; then v_skip check-arm-cpu "scripts/check-arm-cpu.sh is not in this tree"; return; fi
  if [ "${V_ARM_QUICK:-0}" = 1 ]; then v_step check-arm-cpu bash "$script" --quick; else v_step check-arm-cpu bash "$script"; fi
}

# one text case per supported tier above the reference tier
case_hook() {
  local tiers tier
  [ -x "$V_BIN" ] || return 0
  tiers=$("$V_BIN" selftest-cpu 2>&1 | sed -n 's/^supported tiers: //p' | head -n 1)
  for tier in $tiers; do
    case "$tier" in
      scalar | neon) ;;
      neon-sve) echo "tier-$tier|cpu|FLLM_CPU_ISA=$tier FLLM_ARM_SVE=1" ;;
      *) echo "tier-$tier|cpu|FLLM_CPU_ISA=$tier" ;;
    esac
  done
}

v_main "$@"
