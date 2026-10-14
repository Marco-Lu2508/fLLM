#!/bin/sh
# Host-only check of the weight packer and device decoders (no GPU needed): packs random GGUF blocks of every supported format,
# decodes them with the kernels' own F::decode and compares with a reference written from the engine's CPU decoders.
# usage: tests/run_decode_check.sh   (CXX=icpx by default; with the open-source DPC++ set CXX=/path/to/clang++ and LD_LIBRARY_PATH)
set -e
cd "$(dirname "$0")"
CXX=${CXX:-icpx}
OUT=${TMPDIR:-/tmp}/fllm_sycl_decode_check
"$CXX" -fsycl -std=c++17 -O1 -I.. decode_check.cpp ../weights.cpp ../device.cpp -o "$OUT"
"$OUT"
rm -f "$OUT"