#!/bin/bash
# Start an fllm-rs server, run all API client tests, stop the server.
# Usage: run_all.sh [--bin PATH] [--model PATH] [--port 8991]
set -u
BIN="${BIN:-./target/release/fllm-rs}"
MODEL="${MODEL:?set MODEL to a small .gguf file}"
PORT="${1:-8991}"
for a in "$@"; do case "$a" in --bin=*) BIN="${a#--bin=}";; --model=*) MODEL="${a#--model=}";; --port=*) PORT="${a#--port=}";; 89*) PORT="$a";; esac; done
NAME="fllm-api-test"
LOG="${TMPDIR:-/tmp}/${NAME}.log"
VENV="${VENV:-.venv}"
export FLLM_URL="http://127.0.0.1:${PORT}"
export FLLM_MODEL="test-model"
HERE="$(cd "$(dirname "$0")" && pwd)"
pass=0; fail=0
"$BIN" serve --model "$MODEL" --model-name "$FLLM_MODEL" \
  --host 127.0.0.1 --port "$PORT" --ctx 1024 --threads 4 --device cpu \
  >"$LOG" 2>&1 &
SRV=$!
cleanup() { kill "$SRV" 2>/dev/null; wait "$SRV" 2>/dev/null; }
trap cleanup EXIT
for i in $(seq 1 120); do
  curl -fsS "$FLLM_URL/health" >/dev/null 2>&1 && break
  kill -0 "$SRV" 2>/dev/null || { echo "server died; log:"; tail -30 "$LOG"; exit 1; }
  sleep 2
done
curl -fsS "$FLLM_URL/health" >/dev/null || { echo "server not ready"; tail -30 "$LOG"; exit 1; }
# shellcheck disable=1091
source "$VENV/bin/activate"
for t in openai_client.py anthropic_client.py ollama_client.py; do
  echo "=== $t ==="
  if python "$HERE/$t"; then pass=$((pass+1)); else fail=$((fail+1)); fi
done
echo "suites passed=$pass failed=$fail"
[ "$fail" -eq 0 ]
