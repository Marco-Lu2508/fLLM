#!/usr/bin/env bash
# Runs the native Vulkan backend on a software Vulkan driver (Mesa lavapipe) inside docker: no GPU needed.
# It validates the shaders and the host code against the CPU reference (not performance, not vendor-driver quirks).
#
#   scripts/vulkan-soft-test.sh [selftest|unit|model|all] [extra selftest-gpu args...]
#
# selftest  fllm-rs selftest-gpu --device vulkan on lavapipe, JSON report in $OUT/selftest-vulkan.json
# unit      cargo test --features vulkan (includes the whole-model test, which needs a device)
# model     only the whole-model test (tiny random model, 8 greedy tokens, Vulkan vs CPU)
# all       unit + selftest (default)
#
# Environment: FLLM_VK_IMAGE (default fllm-vulkan-soft), OUT (default ./vulkan-soft-out), LP_NUM_THREADS (default 4),
#              CARGO_BUILD_JOBS (default 3), WRAP (command prefix for the container run, e.g. a build-slot wrapper),
#              CACHE_VOLUME (cargo registry volume, default fllm-cargo-cache).
set -euo pipefail

mode=${1:-all}
[ $# -gt 0 ] && shift || true
root=$(cd "$(dirname "$0")/.." && pwd)
image=${FLLM_VK_IMAGE:-fllm-vulkan-soft}
out=${OUT:-$root/vulkan-soft-out}
mkdir -p "$out"

if ! docker image inspect "$image" >/dev/null 2>&1; then
  docker build -f "$root/docker/Dockerfile.vulkan-soft" -t "$image" "$root/docker"
fi

# FLLM_ALLOW_SOFTWARE_GPU: lavapipe is refused as a "GPU" otherwise. The target dir lives inside the mount so rebuilds are incremental.
inner='set -e
cd /src
export CARGO_TARGET_DIR=/src/target-vksoft
case "$MODE" in
  unit|all)
    cargo test --release --locked -p fllm-engine --lib --features vulkan,rust-gpu -- --test-threads=1 vulkan wgpu_backend device_fallback selftest::
    ;;
  model)
    cargo test --release --locked -p fllm-engine --lib --features vulkan,rust-gpu -- --test-threads=1 --nocapture vulkan_whole_model
    ;;
esac
case "$MODE" in
  selftest|all)
    cargo build --release --locked -p fllm-engine --features vulkan,rust-gpu --bin fllm-rs
    vulkaninfo --summary 2>/dev/null | grep -E "deviceName|driverName|apiVersion" | head -4 || true
    set +e
    "$CARGO_TARGET_DIR/release/fllm-rs" selftest-gpu --device vulkan --json /out/selftest-vulkan.json "$@" 2>&1 | tee /out/selftest-vulkan.txt
    rc=${PIPESTATUS[0]}
    set -e
    python3 - <<PY
import json
d = json.load(open("/out/selftest-vulkan.json"))
print("SUMMARY", {k: d.get(k) for k in ("device_name", "checks", "ok_count", "failed", "skipped")})
for g, c in d.get("groups", {}).items():
    print("  %-12s %s" % (g, c))
PY
    exit $rc
    ;;
esac'

# shellcheck disable=SC2086
${WRAP:-} docker run --rm --name "vkwgpu-soft-$$" \
  -v "$root":/src -v "${CACHE_VOLUME:-fllm-cargo-cache}":/usr/local/cargo/registry -v "$out":/out \
  -e MODE="$mode" -e CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-3}" -e LP_NUM_THREADS="${LP_NUM_THREADS:-4}" \
  -e FLLM_ALLOW_SOFTWARE_GPU=1 \
  "$image" nice -n 19 bash -c "$inner" bash "$@"
