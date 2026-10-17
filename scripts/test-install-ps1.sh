#!/bin/sh
# Test of install.ps1 on PowerShell 7 (Linux container) against a fake release server, with faked Windows environment variables
# (LOCALAPPDATA, SystemRoot, PROCESSOR_ARCHITECTURE, and a file standing in for the user PATH). Windows PowerShell 5.1 cannot run
# here: for it the script is only checked for the constructs that 5.1 lacks (see docs/changes/hygiene-a.md).
#
#   scripts/test-install-ps1.sh          # needs docker and python3 on the host; uses mcr.microsoft.com/powershell
set -eu

ROOT=$(CDPATH='' cd -- "$(dirname -- "$0")/.." && pwd)
PORT=${TEST_PORT:-18766}

if [ "${1:-}" != "--inner" ]; then
    WORK=$(mktemp -d)
    SRV_PID=""
    cleanup() { [ -z "$SRV_PID" ] || kill "$SRV_PID" 2>/dev/null || true; rm -rf "$WORK"; }
    trap cleanup EXIT
    REL="$WORK/www/o/n/releases/download"
    API="$WORK/www/api/v1/repos/o/n/releases"
    mkdir -p "$API/tags" "$REL"
    python3 - "$REL" "$API" <<'PYEOF'
import hashlib, json, os, sys, zipfile
rel, api = sys.argv[1], sys.argv[2]
def make(tag, assets, sums="file", bad=False, notes=False):
    d = os.path.join(rel, tag); os.makedirs(d, exist_ok=True)
    lines = []
    for a in assets:
        p = os.path.join(d, a)
        with zipfile.ZipFile(p, "w") as z:
            info = zipfile.ZipInfo("fllm-rs.exe"); info.external_attr = 0o755 << 16
            z.writestr(info, '#!/bin/sh\necho "fllm-rs %s (git fake, 2026-01-01) features: none backends: cpu [%s]"\n' % (tag, a))
        h = hashlib.sha256(open(p, "rb").read()).hexdigest()
        if bad: h = "0" * 64
        if sums == "file": open(p + ".sha256", "w").write(h + "  " + a + "\n")
        lines.append(h + "  " + a)
    if notes:
        json.dump({"tag_name": tag, "body": "Checksums\r\n" + "\r\n".join(lines)}, open(os.path.join(api, "tags", tag), "w"))
make("v9.9.9", ["fllm-windows-x86_64-msvc.zip", "fllm-windows-x86_64-msvc-cuda.zip"], notes=True)
make("v9.9.8", ["fllm-windows-x86_64-msvc.zip"], bad=True)
make("v9.9.4", ["fllm-windows-x86_64-msvc.zip"])
make("v9.9.3", ["fllm-windows-x86_64-msvc.zip"], sums="none")
json.dump({"tag_name": "v9.9.9", "body": "x"}, open(os.path.join(api, "latest"), "w"))
PYEOF
    (cd "$WORK/www" && python3 -m http.server "$PORT" --bind 127.0.0.1 >/dev/null 2>&1) &
    SRV_PID=$!
    sleep 1
    docker run --rm --network host -v "$ROOT:/repo:ro" -e TEST_PORT="$PORT" mcr.microsoft.com/powershell:latest bash /repo/scripts/test-install-ps1.sh --inner
    exit $?
fi

# ------------------------------------------------------------------ inside the PowerShell container
T=$(mktemp -d)
FAILS=0
ok() { echo "ok   - $1"; }
bad() { echo "FAIL - $1"; FAILS=$((FAILS+1)); }
check() { n=$1; shift; if "$@"; then ok "$n"; else bad "$n"; fi; }
out_has() { grep -qF -- "$1" "$T/out"; }
rc_is() { [ "$(cat "$T/rc")" = "$1" ]; }

mkdir -p "$T/win/System32"
export PROCESSOR_ARCHITECTURE=AMD64 SystemRoot="$T/win" FLLM_FORGE=gitea FLLM_REPO_URL="http://127.0.0.1:$PORT/o/n"
unset PROCESSOR_ARCHITEW6432 FLLM_INSTALL_DIR || true

runps() { # LOCALAPPDATA-DIR then arguments for install.ps1 (file mode)
    la=$1; shift
    mkdir -p "$la"
    set +e
    LOCALAPPDATA="$la" FLLM_TEST_USERPATH="$T/userpath" pwsh -NoProfile -NonInteractive -File /repo/install.ps1 "$@" > "$T/out" 2>&1
    echo $? > "$T/rc"
    set -e
}

# 0. parse
pwsh -NoProfile -Command '$e=$null; [void][System.Management.Automation.Language.Parser]::ParseFile("/repo/install.ps1",[ref]$null,[ref]$e); if ($e) { $e; exit 1 }' > "$T/out" 2>&1 && ok "parses" || { bad "parses"; cat "$T/out"; }

# 1. default install, no GPU, no PATH change without consent
rm -f "$T/userpath"; runps "$T/la1" -Version v9.9.9
check "default: exit 0" rc_is 0
check "default: fllm-rs.exe in LOCALAPPDATA/fllm/bin" test -f "$T/la1/fllm/bin/fllm-rs.exe"
check "default: default asset" out_has "asset: fllm-windows-x86_64-msvc.zip"
check "default: checksum from file and notes" out_has "matches the .sha256 file and the release notes"
check "default: user PATH untouched without consent" test ! -s "$T/userpath"
check "default: tells how to add PATH" out_has "-AddToPath"
check "default: unsigned note" out_has "not code-signed"

# 2. -AddToPath, twice: one entry
runps "$T/la1" -Version v9.9.9 -AddToPath
check "AddToPath: exit 0" rc_is 0
check "AddToPath: entry written" grep -qF "$T/la1/fllm/bin" "$T/userpath"
runps "$T/la1" -Version v9.9.9 -AddToPath
check "AddToPath twice: still one entry" sh -c "[ \"\$(tr ';' '\n' < $T/userpath | grep -cxF $T/la1/fllm/bin)\" = 1 ]"
check "AddToPath twice: says already on PATH" out_has "already on your user PATH"

# 3. NVIDIA GPU: nvcuda.dll in System32 selects the CUDA asset when the release has it
: > "$T/win/System32/nvcuda.dll"
runps "$T/la2" -Version v9.9.9
check "nvidia: exit 0" rc_is 0
check "nvidia: cuda asset chosen" out_has "asset: fllm-windows-x86_64-msvc-cuda.zip"
check "nvidia: reason shown" out_has "NVIDIA GPU detected"
runps "$T/la3" -Version v9.9.4
check "nvidia, release without cuda asset: falls back" out_has "asset: fllm-windows-x86_64-msvc.zip"
check "nvidia, release without cuda asset: says so" out_has "no release asset 'fllm-windows-x86_64-msvc-cuda.zip'"
runps "$T/la3b" -Version v9.9.4 -Variant default
check "explicit default variant ignores the GPU" out_has "Variant: default"
rm -f "$T/win/System32/nvcuda.dll"

# 4. checksum mismatch, missing tag, no checksum
runps "$T/la4" -Version v9.9.8
check "mismatch: exit 1" rc_is 1
check "mismatch: message" out_has "CHECKSUM MISMATCH"
check "mismatch: nothing installed" test ! -e "$T/la4/fllm/bin/fllm-rs.exe"
runps "$T/la5" -Version v0.0.1
check "missing tag: exit 1" rc_is 1
check "missing tag: message" out_has "no release asset"
runps "$T/la6" -Version v9.9.3
check "no checksum: refuses" rc_is 1
check "no checksum: names -NoVerify" out_has "-NoVerify"
runps "$T/la6" -Version v9.9.3 -NoVerify
check "-NoVerify installs" test -f "$T/la6/fllm/bin/fllm-rs.exe"

# 5. newest release through the API, bare version without v
runps "$T/la7"
check "latest: resolves v9.9.9" out_has "Newest release: v9.9.9"
runps "$T/la8" -Version 9.9.4
check "bare version gets a v" out_has "Installed v9.9.4"

# 6. uninstall removes the exe and the PATH entry
runps "$T/la1" -Uninstall
check "uninstall: exe gone" test ! -e "$T/la1/fllm/bin/fllm-rs.exe"
check "uninstall: PATH entry gone" sh -c "! grep -qF $T/la1/fllm/bin $T/userpath"

# 7. through iex and through a script block with parameters (the two documented one-liners)
set +e
LOCALAPPDATA="$T/la9" FLLM_TEST_USERPATH="$T/userpath" pwsh -NoProfile -NonInteractive -Command 'Get-Content /repo/install.ps1 -Raw | Invoke-Expression; if (Test-Path variable:ErrorActionPreference) { "EAP=$ErrorActionPreference" }' > "$T/out" 2>&1
echo $? > "$T/rc"; set -e
check "iex: installs the newest release" test -f "$T/la9/fllm/bin/fllm-rs.exe"
check "iex: session settings untouched" out_has "EAP=Continue"
set +e
LOCALAPPDATA="$T/la10" FLLM_TEST_USERPATH="$T/userpath" pwsh -NoProfile -NonInteractive -Command '& ([scriptblock]::Create((Get-Content /repo/install.ps1 -Raw))) -Version v9.9.4 -AddToPath' > "$T/out" 2>&1
echo $? > "$T/rc"; set -e
check "scriptblock with parameters: installs v9.9.4" out_has "Installed v9.9.4"
check "scriptblock with parameters: PATH entry" grep -qF "$T/la10/fllm/bin" "$T/userpath"
set +e
LOCALAPPDATA="$T/la11" FLLM_TEST_USERPATH="$T/userpath" pwsh -NoProfile -NonInteractive -Command 'Get-Content /repo/install.ps1 -Raw | Invoke-Expression; "still alive"' > "$T/out" 2>&1
set -e
runps_fail_iex() { LOCALAPPDATA="$T/la12" FLLM_REPO_URL="http://127.0.0.1:1/o/n" pwsh -NoProfile -NonInteractive -Command 'Get-Content /repo/install.ps1 -Raw | Invoke-Expression; "still alive"' > "$T/out" 2>&1; }
runps_fail_iex || true
check "iex with an unreachable server: error shown, session survives" sh -c "grep -q 'install.ps1:' $T/out && grep -q 'still alive' $T/out"

echo ""
if [ "$FAILS" = 0 ]; then echo "all install.ps1 cases passed (pwsh $(pwsh -NoProfile -Command '$PSVersionTable.PSVersion.ToString()'))"; else echo "$FAILS case(s) failed"; exit 1; fi
