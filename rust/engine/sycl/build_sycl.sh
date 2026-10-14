#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Builds libfllmsycl.a (the SYCL kernels of the Intel GPU backend) outside of cargo; build.rs does the same for `--features sycl`.
#
#   rust/engine/sycl/build_sycl.sh [OUT_DIR]        (default OUT_DIR: rust/engine/sycl/build)
#
# Environment:
#   CXX                  the DPC++ compiler (default: icpx from the oneAPI environment; `source /opt/intel/oneapi/setvars.sh` first;
#                        the open-source intel/llvm clang++ works as well)
#   FLLM_SYCL_AOT        ahead-of-time targets for Intel GPUs, e.g. "acm-g10,pvc,bmg-g21" (needs `ocloc`); default is JIT from SPIR-V,
#                        which works on every device the driver supports and compiles each kernel on first use
#   FLLM_SYCL_EXTRA_FLAGS  further flags for every compiler call
#   JOBS                 parallel compiler calls (default: number of CPUs)
#
# Steps (the order matters, see docs/gpu-backends.md "SYCL"):
#   1. every *.cpp -> *.o            (icpx -fsycl -c; host code plus the device code as an offload image)
#   2. all objects -> fllm_sycl_dev.o (icpx -fsycl -fsycl-link: pre-links and registers the device images, so that the final
#                                      link by cc / rustc needs no SYCL driver)
#   3. ar -> libfllmsycl.a            (+ the dev object)
# The consumer links libfllmsycl.a, libsycl (dylib) and libstdc++; at run time libsycl finds the Level Zero loader (libze_loader).
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-$HERE/build}
CXX=${CXX:-icpx}
JOBS=${JOBS:-$(nproc 2>/dev/null || echo 4)}
mkdir -p "$OUT"

# one device image per kernel: the driver then compiles only the kernels that are launched (about 170 instances exist)
split=(-fsycl-device-code-split=per_kernel)
flags=(-fsycl -std=c++17 -O3 -ffp-model=precise -fPIC -I"$HERE" "${split[@]}")
aot=()
if [ -n "${FLLM_SYCL_AOT:-}" ]; then
    aot=(-fsycl-targets=spir64_gen -Xsycl-target-backend=spir64_gen "-device ${FLLM_SYCL_AOT}")
fi
# shellcheck disable=SC2206
extra=(${FLLM_SYCL_EXTRA_FLAGS:-})

sources=()
for f in "$HERE"/*.cpp; do sources+=("$f"); done
if [ "${#sources[@]}" -eq 0 ]; then echo "no sources in $HERE" >&2; exit 1; fi

echo "== compiling ${#sources[@]} files with $CXX"
running=0
pids=()
for src in "${sources[@]}"; do
    obj="$OUT/$(basename "${src%.cpp}").o"
    "$CXX" "${flags[@]}" "${aot[@]}" "${extra[@]}" -c "$src" -o "$obj" &
    pids+=($!)
    running=$((running + 1))
    if [ "$running" -ge "$JOBS" ]; then
        wait "${pids[0]}"
        pids=("${pids[@]:1}")
        running=$((running - 1))
    fi
done
for pid in "${pids[@]}"; do wait "$pid"; done

objs=()
for src in "${sources[@]}"; do objs+=("$OUT/$(basename "${src%.cpp}").o"); done

echo "== device link"
"$CXX" -fsycl -fsycl-link -fPIC "${split[@]}" "${aot[@]}" "${extra[@]}" "${objs[@]}" -o "$OUT/fllm_sycl_dev.o"

echo "== archive"
rm -f "$OUT/libfllmsycl.a"
ar crs "$OUT/libfllmsycl.a" "${objs[@]}" "$OUT/fllm_sycl_dev.o"
echo "built $OUT/libfllmsycl.a"
