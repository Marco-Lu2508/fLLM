#!/usr/bin/env bash
# Runs fllm-rs. Looks for the binary in $FLLM_INSTALL_DIR, ./.local/bin (older install.sh), the PATH and ~/.local/bin
# (where install.sh puts it by default); installs it with install.sh when none exists.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

find_fllm() {
    local candidate
    for candidate in "${FLLM_INSTALL_DIR:+$FLLM_INSTALL_DIR/fllm-rs}" "$DIR/.local/bin/fllm-rs"; do
        if [ -n "$candidate" ] && [ -x "$candidate" ]; then
            printf '%s\n' "$candidate"
            return 0
        fi
    done
    if candidate="$(command -v fllm-rs 2>/dev/null)" && [ -x "$candidate" ]; then
        printf '%s\n' "$candidate"
        return 0
    fi
    if [ -x "$HOME/.local/bin/fllm-rs" ]; then
        printf '%s\n' "$HOME/.local/bin/fllm-rs"
        return 0
    fi
    return 1
}

if ! FLLM_BIN="$(find_fllm)"; then
    "$DIR/install.sh"
    FLLM_BIN="$(find_fllm)" || { echo "run.sh: fllm-rs is still not found after install.sh" >&2; exit 1; }
fi
exec "$FLLM_BIN" "$@"
