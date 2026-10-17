#!/usr/bin/env bash
# Verification run of the Intel SYCL backend (cargo feature `sycl`) on a machine with an Intel GPU (Arc, Xe, Level Zero driver).
# See docs/backends-verify.md, section Intel. Needs oneAPI DPC++ (icpx).
#
#   scripts/verify-sycl.sh --model-uri hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf
#   ONEAPI_DEVICE_SELECTOR=level_zero:0 scripts/verify-sycl.sh --skip build --bin target/release/fllm-rs
#
# The self test (batch <= 5) reaches only the exact kernels; the tiled and XMX (joint_matrix) paths of the prompt are checked by the
# unit test `linear_matches_cpu_for_every_batch_path` in the pre phase.
# Level of this script: L0.
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=scripts/verify-common.sh
. "$here/verify-common.sh"

if [ -z "${ONEAPI_ROOT:-}" ] && [ -f /opt/intel/oneapi/setvars.sh ]; then
  # shellcheck disable=SC1091
  . /opt/intel/oneapi/setvars.sh >/dev/null 2>&1
fi

V_BACKEND=sycl
V_FEATURES=sycl
V_DEVICE=sycl:0
V_EXACT_ENV="FLLM_SYCL_EXACT=1"
V_DIAG_ENVS="FLLM_SYCL_EXACT=1"
V_SELFTEST_TWICE=0

pre_steps() {
  v_step sycl-linear-vs-cpu cargo test --release -p fllm-engine --features sycl --lib linear_matches_cpu -- --nocapture
}

v_main "$@"
