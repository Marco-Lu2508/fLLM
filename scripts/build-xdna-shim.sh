#!/usr/bin/env bash
# Builds the optional XDNA shim (rust/engine/xdna/shim/fllm_xdna_shim.cpp) that fLLM's native AMD NPU backend loads at run time.
#
#   scripts/build-xdna-shim.sh [OUT_DIR]
#
# Needs a C++17 compiler and the XRT development headers (Debian/Ubuntu: `apt install libxrt-dev`; or the XRT of the Ryzen AI
# software). Environment: XRT_INCLUDE (directory that contains xrt/xrt_kernel.h), XRT_LIB (directory with libxrt_coreutil), CXX.
# The result is libfllm_xdna_shim.so in OUT_DIR (default: ~/.cache/fllm/npu). fLLM finds it there, next to the executable, or through
# FLLM_XDNA_SHIM=/path/to/the/file.
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
src="$here/../rust/engine/xdna/shim/fllm_xdna_shim.cpp"
out="${1:-${XDG_CACHE_HOME:-$HOME/.cache}/fllm/npu}"
cxx="${CXX:-c++}"
inc="${XRT_INCLUDE:-}"
if [ -z "$inc" ]; then
  for d in /opt/xilinx/xrt/include /usr/include /usr/local/include; do
    if [ -f "$d/xrt/xrt_kernel.h" ]; then inc="$d"; break; fi
  done
fi
if [ -z "$inc" ]; then
  echo "XRT headers not found (xrt/xrt_kernel.h). Install libxrt-dev or set XRT_INCLUDE." >&2
  exit 1
fi
libdir="${XRT_LIB:-}"
link=(-lxrt_coreutil)
if [ -n "$libdir" ]; then
  link=("-L$libdir" -lxrt_coreutil)
elif ! ls /usr/lib/*/libxrt_coreutil.so /usr/lib/libxrt_coreutil.so /opt/xilinx/xrt/lib/libxrt_coreutil.so >/dev/null 2>&1; then
  # runtime package only (no development symlink): link against the versioned file
  for f in /usr/lib/*/libxrt_coreutil.so.2 /opt/xilinx/xrt/lib/libxrt_coreutil.so.2; do
    if [ -f "$f" ]; then link=("-l:$(basename "$f")" "-L$(dirname "$f")"); break; fi
  done
fi
mkdir -p "$out"
"$cxx" -std=c++17 -O2 -fPIC -shared -fvisibility=hidden -I"$inc" "$src" -o "$out/libfllm_xdna_shim.so" "${link[@]}" -Wl,-rpath,'$ORIGIN'
echo "built $out/libfllm_xdna_shim.so"
