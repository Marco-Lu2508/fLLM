#!/usr/bin/env bash
# Verification run on Windows from Git Bash or WSL: hands over to scripts/verify-windows.ps1 (PowerShell), which does the work.
# The options are the same as those of the other verify scripts and are translated to the PowerShell names.
#
#   scripts/verify-windows.sh --model-uri hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf
#   scripts/verify-windows.sh --device cuda:0 --build-args=-Cuda,-CudaArchs,89
#
# Level of this script: L0.
here=$(cd "$(dirname "$0")" && pwd)
ps=$(command -v powershell.exe || command -v pwsh.exe || command -v pwsh)
if [ -z "$ps" ]; then
  echo "verify-windows.sh: PowerShell not found (powershell.exe or pwsh)" >&2
  exit 2
fi
target="$here/verify-windows.ps1"
if command -v cygpath >/dev/null 2>&1; then target=$(cygpath -w "$target"); fi

args=()
while [ $# -gt 0 ]; do
  case "$1" in
    --model) args+=(-Model "$2"); shift 2 ;;
    --model-uri) args+=(-ModelUri "$2"); shift 2 ;;
    --bin) args+=(-Bin "$2"); shift 2 ;;
    --device) args+=(-Device "$2"); shift 2 ;;
    --out) args+=(-Out "$2"); shift 2 ;;
    --port) args+=(-Port "$2"); shift 2 ;;
    --tokens) args+=(-Tokens "$2"); shift 2 ;;
    --prompt) args+=(-Prompt "$2"); shift 2 ;;
    --skip) args+=(-Skip "$2"); shift 2 ;;
    --build-args) args+=(-BuildArgs "$2"); shift 2 ;;
    --build-args=*) args+=(-BuildArgs "${1#--build-args=}"); shift ;;
    --dry-run) args+=(-DryRun); shift ;;
    -h | --help) args+=(-Help); shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done
exec "$ps" -NoProfile -ExecutionPolicy Bypass -File "$target" "${args[@]}"
