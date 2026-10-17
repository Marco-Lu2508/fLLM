#!/bin/sh
# Builds the Linux x86_64 fLLM binary with the CUDA kernels inside the CUDA devel container and copies it to
# dist-cuda/fllm-rs. The release workflow (.github/workflows/release.yml, job linux-cuda) calls this script so
# that the same commands run on a CI runner and on a local machine.
#
#   scripts/release/build-linux-cuda.sh
#
# Everything happens inside the container: the packages, the Rust toolchain (rustup) and the CUTLASS and
# FlashInfer headers (third_party/, cloned when missing) are fetched on the first run. Needs docker and
# network access. The full architecture list took about 21 minutes on a 10-core machine (JOBS=3); a slower
# CI runner takes longer, but a single architecture alone only a few minutes.
#
# Environment overrides:
#   CUDA_IMAGE    nvidia/cuda:12.8.1-devel-ubuntu22.04  toolkit used for nvcc; the 22.04 base keeps the glibc floor at 2.35
#   CUDA_ARCHS    75;80;86;89;90;100;120                GPU architectures; build.rs embeds the PTX of the newest one
#   RUST_FEATURES rust-gpu,cuda                         cargo features (wgpu paths plus the CUDA backend)
#   TARGET_DIR    target-linux-cuda                     cargo target directory inside the repository
#   OUT           dist-cuda                             output directory that receives fllm-rs
#   JOBS          3                                     cargo job slots (nvcc parallelises its own work)
#   NAME          fllm-linux-cuda-build                 container name
set -eu

ROOT=$(CDPATH='' cd -- "$(dirname -- "$0")/../.." && pwd)
CUDA_IMAGE=${CUDA_IMAGE:-nvidia/cuda:12.8.1-devel-ubuntu22.04}
CUDA_ARCHS=${CUDA_ARCHS:-75;80;86;89;90;100;120}
RUST_FEATURES=${RUST_FEATURES:-rust-gpu,cuda}
TARGET_DIR=${TARGET_DIR:-target-linux-cuda}
OUT=${OUT:-dist-cuda}
JOBS=${JOBS:-3}
NAME=${NAME:-fllm-linux-cuda-build}

if ! command -v docker >/dev/null 2>&1; then
    echo "build-linux-cuda.sh: docker is required" >&2
    exit 1
fi

mkdir -p "$ROOT/$OUT"

# The container runs as root (the CUDA image has no matching user); the build directory and the output are
# chowned back to the calling user at the end, so a local checkout stays clean.
docker run --rm --name "$NAME" \
    -v "$ROOT:/src" -w /src \
    -v fllm-cargo-cache:/root/.cargo/registry \
    -e CARGO_TARGET_DIR="/src/$TARGET_DIR" \
    -e CARGO_BUILD_JOBS="$JOBS" \
    -e FLLM_CUDA_ARCHS="$CUDA_ARCHS" \
    -e RUST_FEATURES="$RUST_FEATURES" \
    -e HOST_UID="$(id -u)" -e HOST_GID="$(id -g)" \
    -e OUT="$OUT" \
    "$CUDA_IMAGE" \
    bash -lc 'set -e
        export DEBIAN_FRONTEND=noninteractive
        apt-get update -qq
        apt-get install -y -qq --no-install-recommends \
            curl ca-certificates pkg-config cmake clang build-essential git binutils >/dev/null
        if [ ! -x "$HOME/.cargo/bin/cargo" ]; then
            curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
        fi
        . "$HOME/.cargo/env"
        # CUTLASS (NVFP4 prompt GEMM on Blackwell) and FlashInfer (prompt attention); build.rs warns and
        # builds slower fallbacks when they are missing.
        if [ ! -f third_party/cutlass/include/cutlass/cutlass.h ] \
           || [ ! -f third_party/flashinfer/include/flashinfer/attention/prefill.cuh ]; then
            mkdir -p third_party
            git clone -q --depth 1 --branch v4.8.0 https://github.com/NVIDIA/cutlass third_party/cutlass
            git clone -q --depth 1 --branch v0.6.5 https://github.com/flashinfer-ai/flashinfer third_party/flashinfer
        fi
        nice -n 19 cargo build --release --locked -p fllm-engine --features "$RUST_FEATURES" --bin fllm-rs
        install -m 0755 "$CARGO_TARGET_DIR/release/fllm-rs" "/src/$OUT/fllm-rs"
        chown -R "$HOST_UID:$HOST_GID" "/src/$OUT" "$CARGO_TARGET_DIR"
        echo "built: $OUT/fllm-rs"'
