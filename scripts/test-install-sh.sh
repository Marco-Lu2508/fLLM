#!/bin/sh
# Test of install.sh against a fake release server. Run it by hand or from CI:
#
#   scripts/test-install-sh.sh            # builds the fake release tree, starts python3 -m http.server, runs the cases
#   TEST_SHELL=dash scripts/test-install-sh.sh
#
# Needs python3, curl (or wget), tar and sha256sum on the machine that runs it. Nothing leaves the machine: the release tree is
# served from 127.0.0.1 and every case runs with its own temporary HOME.
set -eu

ROOT=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
SHELL_BIN=${TEST_SHELL:-sh}
PORT=${TEST_PORT:-18765}
WORK=$(mktemp -d)
SRV_PID=""
cleanup() { [ -z "$SRV_PID" ] || kill "$SRV_PID" 2>/dev/null || true; rm -rf "$WORK"; }
trap cleanup EXIT

OS=$(uname -s); case "$OS" in Linux) OSN=linux ;; Darwin) OSN=macos ;; *) echo "unsupported OS for this test"; exit 1 ;; esac
case "$(uname -m)" in x86_64|amd64) ARCH=x86_64 ;; aarch64|arm64) ARCH=aarch64 ;; *) echo "unsupported arch"; exit 1 ;; esac
if [ "$OSN" = linux ]; then ASSET="fllm-linux-$ARCH-musl.tar.gz"; else ASSET="fllm-macos-$ARCH.tar.gz"; fi

REL="$WORK/www/o/n/releases/download"
API="$WORK/www/api/v1/repos/o/n/releases"
mkdir -p "$API/tags"

make_release() { # TAG
    d="$REL/$1"; mkdir -p "$d" "$WORK/pkg-$1"
    printf '#!/bin/sh\necho "fllm-rs %s (git fake, 2026-01-01) features: none backends: cpu"\n' "$1" > "$WORK/pkg-$1/fllm-rs"
    chmod 0755 "$WORK/pkg-$1/fllm-rs"
    tar -czf "$d/$ASSET" -C "$WORK/pkg-$1" fllm-rs
}
sum_of() { sha256sum "$1" 2>/dev/null | cut -d' ' -f1 || shasum -a 256 "$1" | cut -d' ' -f1; }

# v9.9.9: good .sha256 file, notes carry the same list
make_release v9.9.9
H=$(sum_of "$REL/v9.9.9/$ASSET")
echo "$H  $ASSET" > "$REL/v9.9.9/$ASSET.sha256"
printf '{"tag_name":"v9.9.9","body":"## Checksums\\r\\n%s  %s\\r\\n"}' "$H" "$ASSET" > "$API/tags/v9.9.9"
printf '{"tag_name":"v9.9.9","body":"x"}' > "$API/latest"
# v9.9.8: the .sha256 file does not match the asset
make_release v9.9.8
echo "0000000000000000000000000000000000000000000000000000000000000000  $ASSET" > "$REL/v9.9.8/$ASSET.sha256"
# v9.9.7: notes list differs from the .sha256 file
make_release v9.9.7
H7=$(sum_of "$REL/v9.9.7/$ASSET")
echo "$H7  $ASSET" > "$REL/v9.9.7/$ASSET.sha256"
printf '{"tag_name":"v9.9.7","body":"1111111111111111111111111111111111111111111111111111111111111111  %s"}' "$ASSET" > "$API/tags/v9.9.7"
# v9.9.6: no .sha256 file, the notes carry the right hash
make_release v9.9.6
H6=$(sum_of "$REL/v9.9.6/$ASSET")
printf '{"tag_name":"v9.9.6","body":"%s  %s"}' "$H6" "$ASSET" > "$API/tags/v9.9.6"
# v9.9.5: nothing to verify against
make_release v9.9.5

(cd "$WORK/www" && python3 -m http.server "$PORT" --bind 127.0.0.1 >/dev/null 2>&1) &
SRV_PID=$!
i=0; until (curl -fsS "http://127.0.0.1:$PORT/api/v1/repos/o/n/releases/latest" >/dev/null 2>&1 || wget -qO- "http://127.0.0.1:$PORT/api/v1/repos/o/n/releases/latest" >/dev/null 2>&1); do
    i=$((i+1)); [ $i -lt 50 ] || { echo "fake server did not start"; exit 1; }; sleep 0.2
done

FAILS=0
t_ok() { echo "ok   - $1"; }
t_fail() { echo "FAIL - $1"; FAILS=$((FAILS+1)); }
check() { # NAME CONDITION-COMMAND...
    n=$1; shift
    if "$@"; then t_ok "$n"; else t_fail "$n"; fi
}

run_install() { # HOME-DIR ARGS... ; output in $WORK/out ; exit code in $WORK/rc
    h=$1; shift
    mkdir -p "$h"
    set +e
    HOME="$h" SHELL=/bin/bash FLLM_REPO_URL="http://127.0.0.1:$PORT/o/n" FLLM_FORGE=gitea PATH="$PATH" \
        "$SHELL_BIN" "$ROOT/install.sh" "$@" > "$WORK/out" 2>&1
    echo $? > "$WORK/rc"
    set -e
}
run_pipe() { # HOME-DIR ARGS...
    h=$1; shift
    mkdir -p "$h"
    set +e
    cat "$ROOT/install.sh" | HOME="$h" SHELL=/bin/zsh FLLM_REPO_URL="http://127.0.0.1:$PORT/o/n" FLLM_FORGE=gitea "$SHELL_BIN" -s -- "$@" > "$WORK/out" 2>&1
    echo $? > "$WORK/rc"
    set -e
}
rc_is() { [ "$(cat "$WORK/rc")" = "$1" ]; }
out_has() { grep -qF -- "$1" "$WORK/out"; }

# 1. pipe mode, newest release from the API, default directory, PATH hint only
H1="$WORK/h1"; run_pipe "$H1"
check "pipe: exit 0" rc_is 0
check "pipe: binary in ~/.local/bin" test -x "$H1/.local/bin/fllm-rs"
check "pipe: installed version runs" sh -c "[ \"\$($H1/.local/bin/fllm-rs --version | cut -d' ' -f2)\" = v9.9.9 ]"
check "pipe: checksum accepted from file and notes" out_has "matches the .sha256 file and the release notes"
check "pipe: prints the PATH line" out_has 'export PATH='
check "pipe: no rc file written without --modify-path" test ! -e "$H1/.zshrc"

# 2. --modify-path is idempotent (bash)
H2="$WORK/h2"; run_install "$H2" --version v9.9.9 --modify-path
check "modify-path: exit 0" rc_is 0
check "modify-path: .bashrc has the block" grep -qF '# >>> fllm >>>' "$H2/.bashrc"
run_install "$H2" --version v9.9.9 --modify-path
check "modify-path: second run keeps one block" sh -c "[ \"\$(grep -cF '# >>> fllm >>>' $H2/.bashrc)\" = 1 ]"
check "modify-path: second run says so" out_has "already present"
# shellcheck disable=SC2016
check "modify-path: line uses \$HOME" grep -qF '$HOME/.local/bin' "$H2/.bashrc"
# the line works in sh and leaves PATH containing the directory exactly once
check "modify-path: sourcing adds the dir once" sh -c "HOME=$H2 PATH=/usr/bin:/bin; . $H2/.bashrc; . $H2/.bashrc; [ \"\$(echo \$PATH | tr : '\n' | grep -cx $H2/.local/bin)\" = 1 ]"

# 3. checksum mismatch aborts and installs nothing
H3="$WORK/h3"; run_install "$H3" --version v9.9.8
check "mismatch: non-zero exit" rc_is 1
check "mismatch: says CHECKSUM MISMATCH" out_has "CHECKSUM MISMATCH"
check "mismatch: nothing installed" test ! -e "$H3/.local/bin/fllm-rs"

# 4. notes disagree with the .sha256 file
H4="$WORK/h4"; run_install "$H4" --version v9.9.7
check "disagree: non-zero exit" rc_is 1
check "disagree: nothing installed" test ! -e "$H4/.local/bin/fllm-rs"

# 5. only the release notes carry the hash
H5="$WORK/h5"; run_install "$H5" --version v9.9.6
check "notes only: installs" rc_is 0
check "notes only: says where the hash came from" out_has "matches the release notes"

# 6. nothing to verify against
H6="$WORK/h6"; run_install "$H6" --version v9.9.5
check "no checksum: refuses" rc_is 1
check "no checksum: names --no-verify" out_has "--no-verify"
check "no checksum: nothing installed" test ! -e "$H6/.local/bin/fllm-rs"
run_install "$H6" --version v9.9.5 --no-verify
check "no checksum: --no-verify installs" rc_is 0
check "no checksum: --no-verify warns" out_has "WITHOUT checksum verification"

# 7. missing asset / tag, and the CUDA variant
H7="$WORK/h7"; run_install "$H7" --version v0.0.1
check "missing tag: non-zero exit" rc_is 1
check "missing tag: message names the asset" out_has "has no asset"
run_install "$H7" --version v9.9.9 --variant cuda
check "cuda missing: non-zero exit" rc_is 1
check "cuda missing: Docker alternative printed" out_has "docker run --rm --gpus all ghcr.io/marco-lu2508/fllm:cuda"

# 8. --prefix and FLLM_INSTALL_DIR
H8="$WORK/h8"; run_install "$H8" --version v9.9.9 --prefix "$WORK/prefix"
check "prefix: DIR/bin" test -x "$WORK/prefix/bin/fllm-rs"
mkdir -p "$WORK/h9"
set +e; HOME="$WORK/h9" FLLM_INSTALL_DIR="$WORK/exact" FLLM_REPO_URL="http://127.0.0.1:$PORT/o/n" FLLM_FORGE=gitea "$SHELL_BIN" "$ROOT/install.sh" v9.9.9 >/dev/null 2>&1; set -e
check "FLLM_INSTALL_DIR: exact directory, bare tag argument" test -x "$WORK/exact/fllm-rs"

# 9. uninstall removes the binary and the PATH block, keeps the rest of the rc file
printf 'export KEEP=1\n' > "$H2/.bashrc.keep"
{ printf 'export BEFORE=1\n'; cat "$H2/.bashrc"; printf 'export AFTER=1\n'; } > "$H2/.bashrc.new" && mv "$H2/.bashrc.new" "$H2/.bashrc"
run_install "$H2" --uninstall
check "uninstall: exit 0" rc_is 0
check "uninstall: binary gone" test ! -e "$H2/.local/bin/fllm-rs"
check "uninstall: block gone" sh -c "! grep -qF 'fllm' $H2/.bashrc"
check "uninstall: other lines kept" sh -c "grep -q BEFORE $H2/.bashrc && grep -q AFTER $H2/.bashrc"
run_install "$H2" --uninstall
check "uninstall: second run is harmless" rc_is 0

# 10. fish
H10="$WORK/h10"; mkdir -p "$H10"
set +e; HOME="$H10" SHELL=/usr/bin/fish FLLM_REPO_URL="http://127.0.0.1:$PORT/o/n" FLLM_FORGE=gitea "$SHELL_BIN" "$ROOT/install.sh" --version v9.9.9 --modify-path >/dev/null 2>&1; set -e
check "fish: config.fish written" grep -qF 'set -gx PATH' "$H10/.config/fish/config.fish"

# 11. --help and an unknown option
run_install "$WORK/h11" --help
check "help: exit 0" rc_is 0
run_install "$WORK/h11" --bogus
check "unknown option: exit 2" rc_is 2

echo ""
if [ "$FAILS" = 0 ]; then echo "all install.sh cases passed (shell: $SHELL_BIN)"; else echo "$FAILS case(s) failed (shell: $SHELL_BIN)"; exit 1; fi
