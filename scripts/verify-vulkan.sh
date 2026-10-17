#!/usr/bin/env bash
# Verification run of the native Vulkan backend (cargo feature `vulkan`) on a machine with a Vulkan GPU: AMD, NVIDIA, Intel.
# See docs/backends-verify.md. Needs the Vulkan SDK or glslc at build time; check_all.sh also needs spirv-val.
#
#   scripts/verify-vulkan.sh --model-uri hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf
#   scripts/verify-vulkan.sh --device vulkan:1 --skip build --bin target/release/fllm-rs    # second GPU
#
# Every run uses FLLM_VULKAN_VALIDATE=1 (validation layer, if installed). The self test runs twice: FLLM_VULKAN_EXACT=1 (only the
# scalar shaders, FP32 tolerance) and then the fast variants (int8 dot product, cooperative matrix), see the Vulkan notes in the docs.
# Level of this script: L0.
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=scripts/verify-common.sh
. "$here/verify-common.sh"

V_BACKEND=vulkan
V_FEATURES=vulkan
V_COMMON_ENV="FLLM_VULKAN_VALIDATE=1"
V_EXACT_ENV="FLLM_VULKAN_EXACT=1"
V_DIAG_ENVS="FLLM_VULKAN_NO_COOP=1 FLLM_VULKAN_NO_IDOT=1"
V_SELFTEST_TWICE=1
V_NATIVE_GUARD=wgpu
# Decoder::set_device falls back from vulkan to wgpu when the native backend cannot be created; vulkan-native refuses to
if grep -q 'vulkan-native' "$here/../rust/engine/src/devices.rs" 2>/dev/null; then
  V_DEVICE=vulkan-native:0
else
  V_DEVICE=vulkan:0
  echo "[verify-vulkan] note: this tree has no 'vulkan-native' device name; a fallback to wgpu would not fail the run, read the server logs" >&2
fi

pre_steps() {
  local checker="$V_ROOT/rust/engine/shaders/vulkan/check_all.sh"
  if [ -f "$checker" ]; then
    # compiles and validates every shader of the registry with the local glslc
    v_step shader-check bash "$checker"
  else
    v_skip shader-check "rust/engine/shaders/vulkan/check_all.sh is not in this tree"
  fi
  # hardware tests of the core (instance, memory, pipelines, recorder) and of the backend; both are #[ignore]d without a device
  # shellcheck disable=SC2086
  v_step vk-smoke $V_COMMON_ENV cargo test --release -p fllm-engine --features vulkan --lib backend::vulkan::instance::smoke -- --ignored --nocapture --test-threads=1
  # shellcheck disable=SC2086
  v_step vk-hw-tests $V_COMMON_ENV cargo test --release -p fllm-engine --features vulkan --lib backend::vulkan::backend::hw_tests -- --ignored --nocapture --test-threads=1
}

v_main "$@"
