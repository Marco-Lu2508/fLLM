#!/bin/bash
# Compiles common.metal as C++ (mechanical sed, mini_stdlib instead of the full shim) and runs the standalone chunk-decoder checks dec8_<name>.cpp:
#   mlx   the MLX affine decoder of every bit width and group size
#   fp8b  the block-wise FP8 decoder (tile scale lookup from the row and column) through the q_dec8 selector
#   bnb   the bitsandbytes NF4 / FP4 / INT8R decoders and their code books
# No GPU, no Metal compiler.
#   usage: run_dec8.sh [name...]       (default: all; CXX: a C++20 compiler with _Float16, e.g. g++ >= 12 or clang++; default c++)
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
# the matrix-vector kernel body for kernel_mv.cpp: the function constant becomes a plain global that the driver sets
{
  printf '#line 1 "common_q_gemv.metal"\n'
  sed -E \
    -e 's/^constant uint q_fc_nb \[\[function_constant\(500\)\]\];/uint q_fc_nb = 1;/' \
    -e 's/\[\[[[:space:]]*host_name[[:space:]]*\([^)]*\)[[:space:]]*\]\]//g' \
    -e 's/\[\[[[:space:]]*kernel[[:space:]]*\]\]//g' \
    -e 's/(^|[^A-Za-z0-9_.])([0-9]+\.[0-9]*|\.[0-9]+)([eE][-+]?[0-9]+)?[hH]\b/\1\2\3f16/g' "$METAL/common_q_gemv.metal"
  printf '\n'
} > "$W/gemv.cpp"
NAMES=${*:-mlx fp8b bnb kernel_mv}
for n in $NAMES; do
  SRC=dec8_$n.cpp; [ -f "$HERE/$SRC" ] || SRC=$n.cpp
  "${CXX:-c++}" -x c++ -std=c++20 -O1 -pthread -fsanitize=alignment -fsanitize-undefined-trap-on-error -Wno-attributes -Wno-unused-value -Wno-pragma-once-outside-header \
    -idirafter "$HERE" \
    -Wno-unknown-pragmas -I"$HERE/mini_stdlib" -I"$W" "$HERE/$SRC" -o "$W/$n.out"
  "$W/$n.out"
done
