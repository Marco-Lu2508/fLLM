#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FLLM_BIN="$DIR/.local/bin/fllm-rs"
if [ ! -x "$FLLM_BIN" ]; then
    "$DIR/install.sh"
fi
exec "$FLLM_BIN" "$@"
