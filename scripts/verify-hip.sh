#!/usr/bin/env bash
# Verification of the AMD (ROCm / HIP) backend on a machine WITH an AMD GPU. Everything in the HIP backend has so far only been compiled
# (test level L0, see docs/changes/hip.md); this script produces the evidence that moves it to L1 (selftest on the device), L2 (text equal
# to the CPU) and a first L3 figure (throughput).
#
#   scripts/verify-hip.sh [MODEL.gguf] [OUT_DIR]
#
# Steps (each logs to OUT_DIR, default verify-out/hip): 0 environment, 1 build, 2 devices, 3 selftest-gpu (default kernels, then each
# experimental switch), 4 greedy text on the CPU and on the GPU (with MODEL), 5 bench (with MODEL). Send back the whole OUT_DIR and
# the line of `fllm-rs devices` that names the GPU. Do not run other GPU jobs at the same time.
#
# Environment
#   ROCM_PATH        ROCm install (default /opt/rocm)
#   FLLM_HIP_ARCHS   gfx targets to compile (default: only the GPU found by rocminfo, which makes the build much faster)
#   HSA_OVERRIDE_GFX_VERSION   for a card the ROCm release does not list: the version of the closest supported chip, e.g. 11.0.0
#                    for gfx1101 / gfx1102 / gfx1103, 10.3.0 for gfx1031 / gfx1032 (then also set FLLM_HIP_ARCHS=gfx1100 / gfx1030)
#   SKIP_BUILD=1     use the existing target/release/fllm-rs
set -u
MODEL="${1:-}"
OUT="${2:-verify-out/hip}"
export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
export PATH="$ROCM_PATH/bin:$PATH"
mkdir -p "$OUT"
BIN=target/release/fllm-rs
fail=0
step() { echo; echo "== $*"; }

step "0 environment"
{
    date
    uname -a
    command -v rocminfo >/dev/null && rocminfo | grep -E "Marketing Name|Name:.*gfx|Compute Unit|Wavefront" | head -20
    command -v hipcc >/dev/null && hipcc --version | head -3
    ls /opt/rocm*/.info/version 2>/dev/null && cat /opt/rocm*/.info/version
    echo "HSA_OVERRIDE_GFX_VERSION=${HSA_OVERRIDE_GFX_VERSION:-}"
    ls -l /dev/kfd /dev/dri 2>&1
    id
} 2>&1 | tee "$OUT/00-env.log"
GFX="$(command -v rocminfo >/dev/null && rocminfo 2>/dev/null | grep -o -E 'gfx[0-9a-f]+' | sort -u | head -1)"
export FLLM_HIP_ARCHS="${FLLM_HIP_ARCHS:-${GFX:-gfx1100}}"
echo "building for: $FLLM_HIP_ARCHS" | tee -a "$OUT/00-env.log"

if [ "${SKIP_BUILD:-0}" != 1 ]; then
    step "1 build (FLLM_HIP=1 cargo build --features hip)"
    if FLLM_HIP=1 cargo build --release --locked -p fllm-engine --features hip --bin fllm-rs >"$OUT/01-build.log" 2>&1; then
        echo "build ok"
    else
        tail -40 "$OUT/01-build.log"; echo "BUILD FAILED"; exit 1
    fi
fi

step "2 devices"
$BIN devices 2>&1 | tee "$OUT/02-devices.log"
DEV="hip:0"
$BIN devices 2>&1 | grep -q "ROCm" || { echo "no ROCm device listed: driver / permissions (groups video, render) / HSA_OVERRIDE_GFX_VERSION?"; fail=1; }

step "3 selftest-gpu --device $DEV"
$BIN selftest-gpu --device "$DEV" --json "$OUT/03-selftest.json" 2>&1 | tee "$OUT/03-selftest.log" | tail -25
[ "${PIPESTATUS[0]}" = 0 ] || { echo "SELFTEST FAILED"; fail=1; }
# the switches that select experimental kernels: each run must pass on its own, otherwise the log says which kernel to blame
for sw in FLLM_HIP_NO_Q4Z=1 FLLM_HIP_NO_Q4Z_SKINNY=1 FLLM_HIP_GRAPH=1; do
    echo "-- with $sw"
    env "$sw" $BIN selftest-gpu --device "$DEV" --json "$OUT/03-selftest-${sw%%=*}.json" >"$OUT/03-selftest-${sw%%=*}.log" 2>&1
    rc=$?
    tail -3 "$OUT/03-selftest-${sw%%=*}.log"
    [ $rc = 0 ] || { echo "FAILED with $sw"; fail=1; }
done

if [ -n "$MODEL" ]; then
    step "4 text: CPU against the GPU (greedy, 64 tokens)"
    ASK='{"model":"m","messages":[{"role":"user","content":"Write one sentence about the sea."}],"temperature":0,"max_tokens":64}'
    ask() { # device port outfile
        $BIN serve --model "$MODEL" --device "$1" --port "$2" --ctx 2048 --max-seqs 2 >"$OUT/04-serve-$1.log" 2>&1 &
        pid=$!
        for _ in $(seq 1 180); do curl -sf "http://127.0.0.1:$2/v1/models" >/dev/null && break; sleep 2; done
        curl -s "http://127.0.0.1:$2/v1/chat/completions" -H 'Content-Type: application/json' -d "$ASK" \
            | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' >"$3" 2>>"$OUT/04-serve-$1.log"
        kill $pid 2>/dev/null; wait $pid 2>/dev/null
    }
    ask cpu 18001 "$OUT/04-cpu.txt"
    FLLM_CUDA_EXACT=1 ask "$DEV" 18002 "$OUT/04-gpu-exact.txt"
    ask "$DEV" 18003 "$OUT/04-gpu-fast.txt"
    cmp "$OUT/04-cpu.txt" "$OUT/04-gpu-exact.txt" && echo "exact mode: identical to the CPU" || { echo "exact mode differs from the CPU (cmp names the first byte)"; fail=1; }
    cmp "$OUT/04-cpu.txt" "$OUT/04-gpu-fast.txt" && echo "default mode: identical to the CPU" || echo "default mode differs (FP16 matrix cores round; note the first different byte, this is not a failure by itself)"

    step "5 bench (prefill 512 / decode 128)"
    $BIN serve --model "$MODEL" --device "$DEV" --port 18004 --ctx 4096 --max-seqs 2 >"$OUT/05-serve.log" 2>&1 &
    pid=$!
    for _ in $(seq 1 180); do curl -sf http://127.0.0.1:18004/v1/models >/dev/null && break; sleep 2; done
    $BIN bench --url http://127.0.0.1:18004 --scenarios prefill,decode-single --prompt-lens 512 --gen-len 128 --warmup 1 --repeats 3 \
        --label "hip ${GFX:-unknown}" --json "$OUT/05-bench.json" 2>&1 | tee "$OUT/05-bench.log" | tail -15
    kill $pid 2>/dev/null; wait $pid 2>/dev/null
else
    echo; echo "(no MODEL given: steps 4 and 5 skipped; pass a small GGUF, e.g. a Q4_K_M or Q8_0 of Qwen2.5-0.5B-Instruct)"
fi

echo
[ $fail = 0 ] && echo "verify-hip: no failure detected, results are in $OUT" || echo "verify-hip: FAILURES, see $OUT"
exit $fail
