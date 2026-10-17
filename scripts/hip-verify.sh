#!/usr/bin/env bash
# Check the AMD (hip) backend of fllm-rs on a machine with a ROCm GPU: device listing, selftest-gpu, and greedy text of a real
# model on the CPU against the GPU with every AMD-specific kernel family switched on or off.
#
#   scripts/hip-verify.sh MODEL.gguf [path/to/fllm-rs] [device (default hip:0)]
#
# Build first:  FLLM_HIP=1 ROCM_PATH=/opt/rocm FLLM_HIP_ARCHS="gfx1100;gfx90a" cargo build --release -p fllm-engine --features hip
# Switches (environment variables read by hip/*.cu): FLLM_HIP_NO_MMA (matrix-core GEMM), FLLM_HIP_NO_SKINNY (F16/BF16 skinny GEMM),
# FLLM_HIP_NO_Q4Z (AWQ/GPTQ kernels; the weights are then expanded to F32). FLLM_CUDA_NO_ATTN4 turns the GQA decode attention off.
# Every run uses temperature 0. The matrix-core paths round weights and activations to FP16, so a later token can differ from the
# FP32 reference; the script reports the position of the first difference instead of only pass/fail.
# Level: this script has not been run on AMD hardware yet.
set -u
MODEL=${1:?usage: hip-verify.sh MODEL.gguf [fllm-rs] [device]}
BIN=${2:-target/release/fllm-rs}
DEVICE=${3:-hip:0}
PORT=${PORT:-18999}
PROMPT=${PROMPT:-"Write one sentence about the sea."}
TOKENS=${TOKENS:-64}
OUT=${OUT:-hip-verify-out}
mkdir -p "$OUT"
fail=0

echo "== devices"
"$BIN" devices 2>&1 | tee "$OUT/devices.txt" | head -40

echo "== selftest-gpu --device $DEVICE"
if "$BIN" selftest-gpu --device "$DEVICE" --json "$OUT/selftest.json" > "$OUT/selftest.txt" 2>&1; then
  echo "selftest-gpu: OK ($(tail -1 "$OUT/selftest.txt"))"
else
  echo "selftest-gpu: FAILED, see $OUT/selftest.txt"; tail -15 "$OUT/selftest.txt"; fail=1
fi

# run_case NAME DEVICE [ENV=VALUE ...]: start the server, ask for a greedy completion, store the text in $OUT/NAME.txt
run_case() {
  local name=$1 dev=$2; shift 2
  env "$@" "$BIN" serve --model "$MODEL" --device "$dev" --port "$PORT" --ctx 2048 --max-seqs 2 > "$OUT/$name.log" 2>&1 &
  local pid=$!
  local up=0
  for _ in $(seq 1 300); do
    if curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null 2>&1; then up=1; break; fi
    kill -0 "$pid" 2> /dev/null || break
    sleep 1
  done
  if [ "$up" != 1 ]; then echo "$name: server did not start (see $OUT/$name.log)"; kill "$pid" 2> /dev/null; wait "$pid" 2> /dev/null; return 1; fi
  local start; start=$(date +%s.%N)
  curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "$(python3 -c 'import json,sys; print(json.dumps({"model":"m","messages":[{"role":"user","content":sys.argv[1]}],"temperature":0,"max_tokens":int(sys.argv[2])}))' "$PROMPT" "$TOKENS")" \
    | python3 -c 'import json,sys; print(json.load(sys.stdin)["choices"][0]["message"]["content"])' > "$OUT/$name.txt"
  local rc=$?
  local end; end=$(date +%s.%N)
  kill "$pid" 2> /dev/null; wait "$pid" 2> /dev/null
  printf '%-22s rc=%s  %s s  %s chars\n' "$name" "$rc" "$(python3 -c "print(f'{$end - $start:.1f}')")" "$(wc -c < "$OUT/$name.txt")"
  return "$rc"
}

# first position where two texts differ ("same" if they do not)
compare() {
  python3 - "$1" "$2" <<'EOF'
import sys
a = open(sys.argv[1], encoding="utf-8", errors="replace").read(); b = open(sys.argv[2], encoding="utf-8", errors="replace").read()
if a == b: print("same")
else:
    n = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
    print(f"first difference at character {n} of {len(a)} / {len(b)}")
EOF
}

echo "== greedy text, $TOKENS tokens"
run_case cpu cpu || fail=1
declare -a CASES=("hip-default" "hip-no-mma:FLLM_HIP_NO_MMA=1" "hip-no-skinny:FLLM_HIP_NO_SKINNY=1" "hip-no-q4z:FLLM_HIP_NO_Q4Z=1" \
                  "hip-no-attn4:FLLM_CUDA_NO_ATTN4=1" "hip-exact:FLLM_CUDA_EXACT=1")
for entry in "${CASES[@]}"; do
  name=${entry%%:*}; envv=${entry#*:}; [ "$envv" = "$entry" ] && envv=FLLM_HIP_VERIFY=1
  if run_case "$name" "$DEVICE" "$envv"; then
    printf '   against cpu: %s\n' "$(compare "$OUT/cpu.txt" "$OUT/$name.txt")"
  else
    fail=1
  fi
done
echo "== done (outputs in $OUT/); exit $fail"
exit "$fail"
