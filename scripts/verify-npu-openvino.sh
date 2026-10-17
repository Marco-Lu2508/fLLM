#!/usr/bin/env bash
# Verification run of the Intel NPU backend over OpenVINO (cargo feature `npu-openvino`) on a Core Ultra machine with an NPU.
# See docs/backends-verify.md, section Intel NPU. Needs the NPU driver and OpenVINO 2025 or newer with the NPU plugin; the C library is
# found through FLLM_OPENVINO_C_LIB, FLLM_OPENVINO_DIR, INTEL_OPENVINO_DIR or the library path.
#
#   scripts/verify-npu-openvino.sh --model /models/llama-q4_0.gguf
#   V_NPU_PROBES="xdna hexagon ane" scripts/verify-npu-openvino.sh --skip text,bench   # also the probes of the XDNA, Hexagon and ANE research notes
#
# Expected: selftest-gpu reports the same 7 failed q8 attention checks as the CPU backend itself (tolerance tighter than the q8 cache
# error), nothing else; the script therefore exits 1 on a healthy run. Only prompt-sized
# quantized_linear calls go to the NPU: the text case `npu-forced` lowers the thresholds so that short prompts reach it too.
# Hybrid backend: dense models only. The probes of XDNA, Hexagon and ANE exist only to measure one MatMul through the vendor stack.
# Level of this script: L0.
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=scripts/verify-common.sh
. "$here/verify-common.sh"

V_BACKEND=npu-openvino
V_FEATURES=npu-openvino
V_DEVICE=npu
V_REF_ENV="FLLM_NPU=0"
# every shape compiles its own model: the thresholds must be lowered for the self test to reach the NPU at all
V_SELFTEST_ENV="FLLM_NPU_MIN_BATCH=1 FLLM_NPU_MIN_ELEMS=1"
V_EXTRA_CASES="npu-forced||FLLM_NPU_MIN_BATCH=1 FLLM_NPU_MIN_ELEMS=1"
# the host hook (the `host_engine` function of compute.rs) is optional; without it FLLM_NPU_HOST=1 does nothing
if grep -q 'host_engine' "$here/../rust/engine/src/compute.rs" 2>/dev/null; then
  V_EXTRA_CASES="$V_EXTRA_CASES
cpu-npu-host|cpu|FLLM_NPU_HOST=1 FLLM_NPU_MIN_BATCH=1 FLLM_NPU_MIN_ELEMS=1"
fi

pre_steps() {
  local probe feature
  if command -v python3 >/dev/null 2>&1; then
    v_step openvino-devices python3 -c "import openvino as ov; print(ov.Core().available_devices)"
  else
    v_skip openvino-devices "python3 not found"
  fi
  # compares a 192x512 Q4_0 projection on the NPU with the host result (threshold 2 %)
  v_step npu-hardware-selfcheck cargo test --release -p fllm-engine --features npu-openvino --lib npu_hardware_selfcheck -- --ignored --nocapture
  for probe in ${V_NPU_PROBES:-}; do
    case "$probe" in
      xdna) feature=npu-xdna ;;
      hexagon) feature=npu-qnn ;;
      ane) feature=npu-ane ;;
      *) v_skip "probe-$probe" "unknown probe (xdna, hexagon, ane)"; continue ;;
    esac
    v_step "probe-$probe-list" cargo run --release -p fllm-engine --example npu_probe --features "$feature" -- list
    v_step "probe-$probe-1x4096x4096" cargo run --release -p fllm-engine --example npu_probe --features "$feature" -- "$probe" 1 4096 4096
    v_step "probe-$probe-128x4096x4096" cargo run --release -p fllm-engine --example npu_probe --features "$feature" -- "$probe" 128 4096 4096
  done
}

v_main "$@"
