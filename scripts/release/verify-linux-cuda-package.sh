#!/bin/sh
# Verifies a fllm-linux-x86_64-cuda.tar.gz package the way a user machine would see it:
#   - the CUDA runtime must be linked statically (no libcudart.so in the dynamic section),
#   - the binary must start in a clean ubuntu:22.04 container without any CUDA toolkit,
#   - `devices` must list the cuda backend among the compiled backends,
#   - without an NVIDIA driver there must be a clear message instead of a crash.
# Used by .github/workflows/release.yml and can be run locally: ./scripts/release/verify-linux-cuda-package.sh dist-cuda/../fllm-linux-x86_64-cuda.tar.gz
set -eu

PKG=${1:-}
if [ -z "$PKG" ] || [ ! -f "$PKG" ]; then
  echo "usage: $0 <fllm-linux-x86_64-cuda.tar.gz>" >&2
  exit 2
fi
PKG=$(cd "$(dirname "$PKG")" && pwd)/$(basename "$PKG")

IMAGE=${SMOKE_IMAGE:-ubuntu:22.04}
command -v docker >/dev/null 2>&1 || { echo "FAIL: docker is required for the smoke test" >&2; exit 1; }
command -v readelf >/dev/null 2>&1 || { echo "FAIL: readelf is required (binutils)" >&2; exit 1; }

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
tar -xzf "$PKG" -C "$TMP"
[ -x "$TMP/fllm-rs" ] || { echo "FAIL: the package does not contain an executable fllm-rs" >&2; exit 1; }

echo "== dynamic dependencies (must not contain libcudart: the CUDA runtime is linked statically) =="
readelf -d "$TMP/fllm-rs" | grep NEEDED || true
if readelf -d "$TMP/fllm-rs" | grep -q 'NEEDED.*libcudart'; then
  echo "FAIL: the binary needs libcudart.so at run time (expected the static CUDA runtime)" >&2
  exit 1
fi
file "$TMP/fllm-rs"

run() { # run <name> <expected-exit> <command...>
  name=$1; expect=$2; shift 2
  set +e
  out=$(docker run --rm -v "$TMP:/pkg:ro" "$IMAGE" /pkg/fllm-rs "$@" 2>&1)
  code=$?
  set -e
  echo "== $name (exit $code) =="
  printf '%s\n' "$out"
  if [ "$expect" = zero ] && [ "$code" -ne 0 ]; then
    echo "FAIL: $name exited with $code" >&2; exit 1
  fi
  if [ "$expect" = nonzero ] && [ "$code" -eq 0 ]; then
    echo "FAIL: $name was expected to fail without a GPU" >&2; exit 1
  fi
  case "$out" in
    *panicked*|*RUST_BACKTRACE*) echo "FAIL: $name crashed (panic) instead of reporting the missing device" >&2; exit 1 ;;
  esac
  if [ "$code" -ge 128 ]; then
    echo "FAIL: $name was killed by a signal (exit $code)" >&2; exit 1
  fi
  LAST_OUT=$out
}

# No --version in this CLI; --help proves that the binary starts inside the clean image.
run "fllm-rs --help" zero --help

run "fllm-rs devices" zero devices
printf '%s\n' "$LAST_OUT" | grep -q 'Compiled backends:.*cuda' \
  || { echo "FAIL: the cuda backend is not compiled into this binary" >&2; exit 1; }
printf '%s\n' "$LAST_OUT" | grep -q 'cpu' \
  || { echo "FAIL: devices did not list the CPU backend" >&2; exit 1; }

# Without the NVIDIA driver the CUDA path must report clearly that no device exists.
run "fllm-rs selftest-gpu" nonzero selftest-gpu
printf '%s\n' "$LAST_OUT" | grep -qi 'not compiled into this binary' \
  && { echo "FAIL: selftest-gpu says the backend is missing; the CUDA build is broken" >&2; exit 1; }
printf '%s\n' "$LAST_OUT" | grep -qi 'no gpu device' \
  || { echo "FAIL: selftest-gpu did not print a clear no-device message" >&2; exit 1; }

run "fllm-rs selftest-gpu --device cuda" nonzero selftest-gpu --device cuda
printf '%s\n' "$LAST_OUT" | grep -qi 'not compiled into this binary' \
  && { echo "FAIL: --device cuda says the backend is missing; the CUDA build is broken" >&2; exit 1; }

echo "PASS: $PKG starts in $IMAGE without a CUDA toolkit and reports the missing GPU instead of crashing"
