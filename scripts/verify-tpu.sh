#!/usr/bin/env bash
# Verification run of the Google TPU backend (cargo feature `tpu-pjrt`) on a Colab TPU runtime (v5e-1, or v2-8 when offered) or a TPU VM.
# See docs/backends-verify.md, section TPU. Needs the pip package `libtpu` (or FLLM_TPU_LIBRARY) at run time, not at build time.
# Restart the runtime first: a TPU can be opened by one process only, an earlier `import jax` in the notebook kernel blocks it.
#
#   !bash fllm/scripts/verify-tpu.sh --model-uri hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf
#
# When scripts/colab/tpu_check.py exists, it does build, device list, self test (FLLM_TPU_EXACT=1) and the greedy text
# comparison of TPU against CPU; its files (tpu_check_report.json, tpu_selftest.json, tpu_check_*.log) are copied to the output
# directory, and this script adds the timed requests, fllm-rs bench and the JSON report. Without it the common phases run instead.
# Level of this script: L0.
here=$(cd "$(dirname "$0")" && pwd)
# shellcheck source=scripts/verify-common.sh
. "$here/verify-common.sh"

V_BACKEND=tpu
V_FEATURES=tpu-pjrt
V_DEVICE=tpu
# the self test bounds are those of float32 arithmetic: f32 weights and full-precision products
V_COMMON_ENV="FLLM_TPU_EXACT=1"
if [ -z "${FLLM_TPU_LIBRARY:-}" ] && command -v python3 >/dev/null 2>&1; then
  lib=$(python3 -c 'import libtpu; print(libtpu.get_library_path())' 2>/dev/null)
  if [ -n "$lib" ] && [ -f "$lib" ]; then export FLLM_TPU_LIBRARY="$lib"; fi
fi
if [ -f "$here/colab/tpu_check.py" ]; then V_PHASES="pre bench"; fi

pre_steps() {
  local checker="$V_ROOT/scripts/colab/tpu_check.py" args f
  if [ ! -f "$checker" ]; then v_skip tpu-check "scripts/colab/tpu_check.py is not in this tree"; return; fi
  args=(--report "$V_OUT/tpu_check_report.json" --port "$V_PORT" --tokens "$V_TOKENS" --prompt "$V_PROMPT")
  if [ -n "$V_BIN_GIVEN" ] && [ "$V_BIN_GIVEN" = 1 ]; then args+=(--skip build --bin "$V_BIN"); fi
  if [ -n "$V_MODEL" ]; then args+=(--model "$V_MODEL"); elif [ -n "$V_MODEL_URI" ]; then args+=(--model-uri "$V_MODEL_URI"); fi
  v_step tpu-check python3 "$checker" "${args[@]}"
  if [ "$V_DRY" != 1 ]; then
    for f in "$V_ROOT"/scripts/colab/tpu_selftest.json "$V_ROOT"/scripts/colab/tpu_check_*.log; do
      if [ -f "$f" ]; then cp "$f" "$V_OUT/"; fi
    done
  fi
}

v_main "$@"
