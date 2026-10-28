#!/bin/bash
# Compiles common.metal as C++ (mechanical sed, mini_stdlib instead of the full shim) and runs dec8_mlx.cpp: the MLX affine chunk decoder of
# every bit width and group size against an independent bit-by-bit packer. No GPU, no Metal compiler.
#   usage: run_dec8_mlx.sh           (CXX: a C++20 compiler with _Float16, e.g. g++ >= 12 or clang++; default c++)
set -e
HERE=$(cd "$(dirname "$0")" && pwd)
METAL=$(dirname "$HERE")
W=${EMU_WORK:-/tmp/fllm-metal-dec8}
mkdir -p "$W"
{
  printf '#line 1 "common.metal"\n'
  sed -E \
    -e 's/\[\[[[:space:]]*host_name[[:space:]]*\([^)]*\)[[:space:]]*\]\]//g' \
    -e 's/\[\[[[:space:]]*kernel[[:space:]]*\]\]//g' \
    -e 's/(^|[^A-Za-z0-9_.])([0-9]+\.[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?[hH]\b/\1\2\3f16/g' \
    -e 's/^using namespace metal;//' "$METAL/common.metal"
  printf '\n'
} > "$W/unit.cpp"
"${CXX:-c++}" -x c++ -std=c++20 -O1 -fsanitize=alignment -fsanitize-undefined-trap-on-error -Wno-attributes -Wno-unused-value -Wno-pragma-once-outside-header \
  -Wno-unknown-pragmas -I"$HERE/mini_stdlib" -I"$W" "$HERE/dec8_mlx.cpp" -o "$W/dec8_mlx"
"$W/dec8_mlx"
