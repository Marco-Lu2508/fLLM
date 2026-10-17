#!/usr/bin/env bash
# Runs the SDK tests against a running fllm-rs server that serves the tiny test model.
#
#   fllm-rs make-tiny-model /tmp/tiny.gguf
#   fllm-rs --model /tmp/tiny.gguf --model-name tiny --device cpu --port 8000 --api-key test-key &
#   FLLM_API_KEY=test-key scripts/api-tests/run_all.sh
#
# Environment: FLLM_BASE_URL (default http://127.0.0.1:8000), FLLM_API_KEY, FLLM_MODEL (default tiny),
# VENV (default .venv-api-tests next to this script; set VENV= to use the current interpreter).
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
venv="${VENV-$here/.venv-api-tests}"
if [ -n "$venv" ]; then
    [ -d "$venv" ] || python3 -m venv "$venv"
    # shellcheck disable=SC1091
    . "$venv/bin/activate"
fi
python3 -m pip install --quiet --disable-pip-version-check -r "$here/requirements.txt"
cd "$here"
exec python3 -m pytest -v "$@"
