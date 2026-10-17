#!/usr/bin/env bash
set -euo pipefail
BIN="${BIN:-./target/release/fllm-rs}"
OUT="${1:-docs/cli-reference.md}"
{
  echo "# CLI reference (generated)"
  echo
  for cmd in serve run pull list rm ps show search convert quantize inspect formats chat bench devices selftest-cpu selftest-gpu completions; do
    echo "## fllm-rs $cmd"
    echo '```'
    "$BIN" "$cmd" --help 2>&1 || true
    echo '```'
    echo
  done
} > "$OUT"
