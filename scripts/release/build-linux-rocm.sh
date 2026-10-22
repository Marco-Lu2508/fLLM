#!/bin/sh
# Builds the Linux x86_64 fLLM binary with the ROCm / HIP kernels (builder stage of docker/Dockerfile.rocm) and copies it to
# dist-rocm/fllm-rs. The release workflow (.github/workflows/release.yml, job linux-rocm) calls this script, so the same commands run on a
# CI runner and on a local machine. hipcc compiles the kernels without a GPU. The binary loads libamdhip64 at run time from the ROCm
# installation of the user (the amdgpu driver and the HIP runtime must be there; the Docker image ghcr.io/<owner>/fllm:rocm brings it).
# The kernels have not run on an AMD GPU yet (test level L0, docs/changes/hip.md).
#
#   scripts/release/build-linux-rocm.sh
#
# Environment overrides:
#   HIP_ARCHS  gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1151;gfx1200;gfx1201   offload architectures
#   OUT        dist-rocm   output directory that receives fllm-rs
#   JOBS       4           cargo job slots
set -eu

ROOT=$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd)
HIP_ARCHS=${HIP_ARCHS:-"gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1151;gfx1200;gfx1201"}
OUT=${OUT:-dist-rocm}
JOBS=${JOBS:-4}
IMAGE=fllm-rocm-build

if ! command -v docker >/dev/null 2>&1; then
    echo "build-linux-rocm.sh: docker is required" >&2
    exit 1
fi

mkdir -p "$ROOT/$OUT"
docker build -f "$ROOT/docker/Dockerfile.rocm" --target builder \
    --build-arg "HIP_ARCHS=$HIP_ARCHS" --build-arg "JOBS=$JOBS" -t "$IMAGE" "$ROOT"
id=$(docker create "$IMAGE")
trap 'docker rm -f "$id" >/dev/null 2>&1 || true' EXIT
docker cp "$id:/build/fLLM/target/release/fllm-rs" "$ROOT/$OUT/fllm-rs"
chmod 0755 "$ROOT/$OUT/fllm-rs"
echo "built: $OUT/fllm-rs"
