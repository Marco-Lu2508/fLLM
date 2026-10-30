#!/usr/bin/env bash
# Regenerates the NPU programs of the native AMD XDNA backend (rust/engine/xdna/kernels) from the open mlir-aie / Peano toolchain.
#
#   scripts/xdna-build-kernels.sh TOOLS_DIR [OUT_DIR]
#
# TOOLS_DIR is a prefix of your own (it is created): a Python virtual environment with the mlir-aie wheel and the Peano (llvm-aie)
# wheel, a checkout of the mlir-aie sources (for the whole_array.py design), and the XRT utilities (xclbinutil) that aiecc calls.
# Needs: python3 (3.10 - 3.14), uv or pip, git, curl, and `xclbinutil` on the PATH (Debian/Ubuntu: `apt install libxrt-utils`).
# This is a developer tool; users of fLLM do not run it, the results are in the repository.
#
# Versions used for the shipped files (docs/npu-xdna.md): mlir-aie 1.4.4, llvm-aie 22.0.0.2026102001+4f5df13d.
# Both are Apache-2.0 WITH LLVM-exception (mlir-aie) / Apache-2.0 WITH LLVM-exception (llvm-aie).
set -euo pipefail
tools="${1:?usage: xdna-build-kernels.sh TOOLS_DIR [OUT_DIR]}"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
out="${2:-$here/../rust/engine/xdna/kernels}"
mlir_aie_version="${MLIR_AIE_VERSION:-1.4.4}"
mkdir -p "$tools" "$out"
cd "$tools"

if [ ! -d venv ]; then
  python3 -m venv venv
fi
# shellcheck disable=SC1091
. venv/bin/activate
py_tag="cp$(python3 -c 'import sys; print(f"{sys.version_info[0]}{sys.version_info[1]}")')"
if ! python3 -c 'import aie' 2>/dev/null; then
  mkdir -p dl && cd dl
  curl -fLO "https://github.com/Xilinx/mlir-aie/releases/download/v${mlir_aie_version}/mlir_aie-${mlir_aie_version}-${py_tag}-${py_tag}-manylinux_2_35_x86_64.whl"
  url="$(curl -fs https://api.github.com/repos/Xilinx/llvm-aie/releases/tags/nightly \
    | python3 -c 'import json,sys; print(sorted(a["browser_download_url"] for a in json.load(sys.stdin)["assets"] if "x86_64" in a["name"] and a["name"].startswith("llvm_aie"))[-1])')"
  curl -fL -o "llvm_aie-$(basename "${url//%2B/+}")" "$url"
  cd ..
  python3 -m pip install dl/mlir_aie-*.whl dl/llvm_aie-*.whl numpy ml_dtypes
fi
if [ ! -d src-mlir-aie ]; then
  git clone --depth 1 --branch "v${mlir_aie_version}" https://github.com/Xilinx/mlir-aie.git src-mlir-aie
fi
site="$(python3 -c 'import site; print(site.getsitepackages()[0])')"
export PEANO_INSTALL_DIR="$site/llvm-aie"
export MLIR_AIE_DIR="$site/mlir_aie"
export PATH="$MLIR_AIE_DIR/bin:$PEANO_INSTALL_DIR/bin:$PATH"
command -v xclbinutil >/dev/null || { echo "xclbinutil not found (apt install libxrt-utils)" >&2; exit 1; }

design="$tools/src-mlir-aie/programming_examples/basic/matrix_multiplication/whole_array/whole_array.py"
# bf16 inputs, f32 accumulation and output, B stored [n][k], 8 columns, tile m=32 k=64 n=64. The xclbin does not depend on the shape; the
# instruction stream does: one per (m, k) with n = 1024.
for shape in "512 1024" "512 2048" "512 3072" "512 4096" "128 1024" "128 2048" "128 3072" "128 4096"; do
  set -- $shape
  work="$tools/build/m$1_k$2_n1024"
  mkdir -p "$work"
  (cd "$work" && python3 "$design" --dev npu2 -M "$1" -K "$2" -N 1024 -m 32 -k 64 -n 64 --n-aie-cols 8 --b-col-maj 1 \
      --emulate-bf16-mmul-with-bfp16 0 --dtype_in bf16 --dtype_out f32 --xclbin-path "$work/a.xclbin" --insts-path "$work/a.bin")
  cp "$work/a.bin" "$out/m$1_k$2_n1024.insts"
  if [ "$1 $2" = "512 1024" ]; then cp "$work/a.xclbin" "$out/gemm_bf16_f32_8col.xclbin"; fi
done
echo "wrote $out"

# ---- programs with 4-bit weights expanded inside the compute tiles (rust/engine/xdna/kernel_src/) ----------------------
# dq:   bf16 products, weight tiles packed 4-bit (fllm_dq.cc, whole_array_dq.py), tile m = 32 (128 / 512 rows per call)
# dqb:  block-floating-point products, f32 sums (fllm_mm_bfp.cc, whole_array_dqbfp.py), tile m = 64 (256 / 512 rows per call)
# dqbs: the same with m = 16 (64 rows per call)
src="$here/../rust/engine/xdna/kernel_src"
export FLLM_DQ_SRC="$(cd "$src" && pwd)"
build_dq() { # prefix design m_tile rows...
  local prefix="$1" design="$2" mt="$3"
  shift 3
  for rows in "$@"; do
    for k in 1024 2048 3072 4096; do
      work="$tools/build/${prefix}_m${rows}_k${k}"
      mkdir -p "$work"
      (cd "$work" && python3 "$FLLM_DQ_SRC/$design.py" --dev npu2 -M "$rows" -K "$k" -N 1024 -m "$mt" -k 64 -n 64 --n-aie-cols 8 \
          --dtype_in bf16 --dtype_out f32 --xclbin-path "$work/a.xclbin" --insts-path "$work/a.bin")
      cp "$work/a.bin" "$out/${prefix}_m${rows}_k${k}_n1024.insts"
      if [ "$k" = 1024 ] && [ ! -e "$out/gemm_${prefix}_q4_8col.xclbin" ]; then cp "$work/a.xclbin" "$out/gemm_${prefix}_q4_8col.xclbin"; fi
    done
  done
}
build_dq dq whole_array_dq 32 512 128
build_dq dqb whole_array_dqbfp 64 512 256
build_dq dqbs whole_array_dqbfp 16 64
echo "wrote the 4-bit programs to $out"
