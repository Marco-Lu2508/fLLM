#!/usr/bin/env bash
# Verification run of the native Metal backend (cargo feature `metal`) on an Apple-silicon Mac. See docs/backends-verify.md, section Mac.
# Needs Rust and the Apple SDK (Xcode or the Command Line Tools); macOS 15 or newer for residency sets and MSL 3.2.
#
#   scripts/verify-metal.sh --model-uri hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf
#
# Order: metal_core_smoke, the quantized kernels with all libraries compiled at start (read the compiler messages in
# metal-quant.log first), the ops kernels, devices, selftest-gpu, text check against the CPU, benchmark. When the self test or the text
# differs, the script repeats it once per switch of V_DIAG_ENVS: note which switch makes the failure go away.
# Level of this script: L0. Works with the bash 3.2 of macOS.
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=scripts/verify-common.sh
. "$here/verify-common.sh"

V_BACKEND=metal
V_FEATURES=metal
V_EXACT_ENV="FLLM_METAL_EXACT=1"
V_DIAG_ENVS="FLLM_METAL_SERIAL=1 FLLM_METAL_FAST_MATH=0 FLLM_METAL_EXACT=1 FLLM_METAL_NO_NOCOPY=1 FLLM_METAL_SINGLE_LIB=1 FLLM_METAL_MSL=3.0 FLLM_METAL_MSL=3.1"
V_NATIVE_GUARD=wgpu
# the file-mapping (no-copy) path against the copy path, (the copy path is the reference)
V_EXTRA_CASES="no-nocopy||FLLM_METAL_NO_NOCOPY=1"
if grep -q 'metal-native' "$here/../rust/engine/src/devices.rs" 2>/dev/null; then
  V_DEVICE=metal-native:0
else
  V_DEVICE=metal:0
  echo "[verify-metal] note: this tree has no 'metal-native' device name; a fallback to wgpu would not fail the run, read the server logs" >&2
fi

pre_steps() {
  # (a) the nine checks of the core (device, buffers, no-copy mapping, pipelines, command buffers, hazards, blits)
  v_step metal-core-smoke cargo test --release -p fllm-engine --features metal metal_core_smoke -- --nocapture
  # (b)+(c) every library is compiled at start (FLLM_METAL_PREWARM), compiler errors per file go to the log (FLLM_METAL_DEBUG);
  # then the quantized kernels against the CPU reference, one line per format and batch: [ok] / [FAIL]
  v_step metal-quant FLLM_METAL_PREWARM=1 FLLM_METAL_DEBUG=1 cargo test --release -p fllm-engine --features metal metal_quant_kernels -- --nocapture
  # (d) attention, norms, RoPE, activations, embedding, KV write, argmax
  v_step metal-ops cargo test --release -p fllm-engine --features metal metal_ops_kernels -- --nocapture
}

v_main "$@"
