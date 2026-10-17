#!/usr/bin/env bash
# Verification run of the x86 extension kernels (cargo feature `avx512`: AVX-512 VNNI/BF16 and AMX) on an x86-64 Linux machine with
# AVX-512 (Zen 4/5, Ice Lake or newer) and, for the AMX paths, a Sapphire Rapids or newer Xeon with Linux 5.16 or newer.
# See docs/backends-verify.md, section x86. There is no GPU here: the "device" is the CPU, the reference is the AVX2 tier
# (FLLM_CPU_ISA=avx2), and every tier the CPU supports is compared with it.
#
#   scripts/verify-avx512.sh --model /models/llama-q4_k_m.gguf
#
# check_x86_ext.sh builds with --features avx512, runs the unit tests of cpu_avx512, `fllm-rs selftest-cpu` for every
# tier and for the policy overrides (FLLM_X86_WIDE_FLOAT, FLLM_X86_WIDE_INT, FLLM_X86_BF16, FLLM_VNNI_GEMM, FLLM_AMX_MIN_BATCH).
# Level of this script: L0.
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=scripts/verify-common.sh
. "$here/verify-common.sh"

V_BACKEND=avx512
V_FEATURES=avx512
V_DEVICE=cpu
V_REF_ENV="FLLM_CPU_ISA=avx2"
V_PHASES="build devices pre text bench"
# the policy overrides of check_x86_ext.sh as text cases (each must give the same greedy text up to rounding)
V_EXTRA_CASES="wide-float||FLLM_X86_WIDE_FLOAT=1
no-wide-float||FLLM_X86_WIDE_FLOAT=0
no-bf16||FLLM_X86_BF16=0
vnni-gemm-all||FLLM_VNNI_GEMM=all
amx-from-1-token||FLLM_AMX_MIN_BATCH=1"

pre_steps() {
  local script="$V_ROOT/scripts/check_x86_ext.sh"
  if [ ! -f "$script" ]; then v_skip check-x86-ext "scripts/check_x86_ext.sh is not in this tree"; return; fi
  v_step check-x86-ext bash "$script"
}

# one text case per supported tier above the reference tier
case_hook() {
  local tiers tier
  [ -x "$V_BIN" ] || return 0
  tiers=$("$V_BIN" selftest-cpu 2>&1 | sed -n 's/^supported tiers: //p' | head -n 1)
  for tier in $tiers; do
    case "$tier" in
      scalar | avx2) ;;
      *) echo "tier-$tier|cpu|FLLM_CPU_ISA=$tier" ;;
    esac
  done
}

v_main "$@"
