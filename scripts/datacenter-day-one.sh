#!/usr/bin/env bash
# The first day on a multi-GPU node: checks first, then the same agent load against fLLM and, when given, vLLM and SGLang.
#
#   MODEL_GGUF=/models/Llama-3.1-70B-Instruct-Q4_K_M.gguf DEVICES=hip:0,hip:1,hip:2,hip:3,hip:4,hip:5,hip:6,hip:7 TP=8 \
#   VLLM_CMD='vllm serve meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size 8 --port {PORT}' \
#   SGLANG_CMD='python3 -m sglang.launch_server --model-path meta-llama/Llama-3.1-70B-Instruct --tp 8 --port {PORT}' \
#   scripts/datacenter-day-one.sh
#
# What it does, in this order, and why it stops where it stops:
#   1. Gates (fLLM only): devices, doctor, the self test on every GPU, the collectives. The first gate that fails ends the run: a
#      benchmark on a node whose collectives are broken measures the breakage. Nothing after a failed gate is run.
#   2. Every engine in turn, started fresh (so the first pass of bench-agent is cold): wait for /health, `bench-agent` with cold and warm
#      pass, `bench-serve` sweep, stop the engine. The same options and seed for every engine, so the prompts are byte-identical.
#   3. A summary with the files to send back. Nothing is uploaded anywhere.
#
# Nothing in here has run on a GPU node yet (the machine it was written on has no GPU): the gates and the benchmark phases were run on a
# CPU host with DEVICES=cpu to see that the logic works, and the commands of the GPU checks are the ones of docs/datacenter-gap.md.
#
# Settings (environment):
#   MODEL_GGUF   model file for fLLM (required)          DEVICES   fLLM --device value (default hip:0..7, "cpu" skips GPU gates)
#   TP           tensor parallel size for fLLM (default 8)    FLLM  the binary (default fllm-rs)
#   VLLM_CMD / SGLANG_CMD  command that starts the engine, {PORT} is replaced; unset = that engine is skipped
#   MODEL_NAME   model id the other engines serve under, when it differs from what /v1/models lists first
#   OUT          output directory (default ./day-one-<time>)    PORT  first port (default 8200)
#   USERS TURNS SYSTEM_WORDS ANSWER_TOKENS SEED   bench-agent options (defaults 32 8 3000 128 42)
#   SWEEP        bench-serve concurrency list (default 1,4,16,64)
#   OUTPUT_CACHE on = keep fLLM's output cache (default off; it distorts the warm pass of a comparison, see the comment below)
#   READY_SECONDS  wait for an engine to answer /health (default 1800: a 70B model loads for minutes)
#   DRY_RUN=1    print every command and run none of them
set -u

FLLM="${FLLM:-fllm-rs}"
DEVICES="${DEVICES:-hip:0,hip:1,hip:2,hip:3,hip:4,hip:5,hip:6,hip:7}"
TP="${TP:-8}"
OUT="${OUT:-./day-one-$(date +%Y%m%d-%H%M%S)}"
PORT="${PORT:-8200}"
USERS="${USERS:-32}"; TURNS="${TURNS:-8}"; SYSTEM_WORDS="${SYSTEM_WORDS:-3000}"; ANSWER_TOKENS="${ANSWER_TOKENS:-128}"; SEED="${SEED:-42}"
SWEEP="${SWEEP:-1,4,16,64}"
READY_SECONDS="${READY_SECONDS:-1800}"
DRY_RUN="${DRY_RUN:-0}"
MODEL_GGUF="${MODEL_GGUF:-}"
SERVER_PID=""
# fLLM replays the whole stream of an identical request from its output cache; vLLM and SGLang have no such thing. With identical prompts in
# the warm pass that would make fLLM look faster for a reason that has nothing to do with the prefix cache, so it is off here (the cache of
# the prefix, the KV blocks, stays on). OUTPUT_CACHE=on measures the output cache too; say so when the numbers are shown.
if [ "${OUTPUT_CACHE:-off}" = on ]; then unset FLLM_OUTPUT_CACHE; else export FLLM_OUTPUT_CACHE=off; fi

say() { printf '\n== %s\n' "$*"; }
die() { printf 'STOP: %s\n' "$*" >&2; stop_engine; summary "stopped: $*"; exit 1; }

stop_engine() {
  if [ -n "$SERVER_PID" ] && kill -0 "$SERVER_PID" 2>/dev/null; then
    kill "$SERVER_PID" 2>/dev/null
    for _ in $(seq 1 60); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$SERVER_PID" 2>/dev/null
  fi
  SERVER_PID=""
}
trap 'stop_engine' EXIT

# run NAME COMMAND...: the output goes to $OUT/NAME.log as well as to the screen; the exit code of the command is the result.
run() {
  local name="$1"; shift
  if [ "$DRY_RUN" = 1 ]; then printf '[dry-run] %s\n' "$*"; return 0; fi
  printf '$ %s\n' "$*" | tee "$OUT/$name.log"
  "$@" 2>&1 | tee -a "$OUT/$name.log"
  return "${PIPESTATUS[0]}"
}

summary() {
  [ "$DRY_RUN" = 1 ] && return 0
  {
    echo "day-one run: $(date -u +%FT%TZ)"
    echo "result: $1"
    echo "options: users=$USERS turns=$TURNS system_words=$SYSTEM_WORDS answer_tokens=$ANSWER_TOKENS seed=$SEED sweep=$SWEEP"
    echo "files:"
    (cd "$OUT" && ls -1 | sed 's/^/  /')
  } > "$OUT/SUMMARY.txt"
  echo; cat "$OUT/SUMMARY.txt"
}

wait_ready() { # PORT, PID
  [ "$DRY_RUN" = 1 ] && return 0
  local waited=0
  while [ "$waited" -lt "$READY_SECONDS" ]; do
    kill -0 "$2" 2>/dev/null || return 2           # the engine exited while loading
    curl -s -m 3 -o /dev/null "http://127.0.0.1:$1/health" && return 0
    curl -s -m 3 -o /dev/null "http://127.0.0.1:$1/v1/models" && return 0
    sleep 5; waited=$((waited + 5))
  done
  return 1
}

# measure ENGINE PORT: the same load for every engine
measure() {
  local engine="$1" port="$2" model_args=()
  [ -n "${MODEL_NAME:-}" ] && [ "$engine" != fllm ] && model_args=(--model "$MODEL_NAME")
  run "$engine-agent" "$FLLM" bench-agent --url "http://127.0.0.1:$port" "${model_args[@]}" --users "$USERS" --turns "$TURNS" \
      --system-words "$SYSTEM_WORDS" --answer-tokens "$ANSWER_TOKENS" --passes 2 --seed "$SEED" --label "$engine" \
      --json "$OUT/$engine-agent.json" --markdown "$OUT/$engine-agent.md" || echo "bench-agent against $engine failed (see $engine-agent.log)"
  run "$engine-serve" "$FLLM" bench-serve --url "http://127.0.0.1:$port" "${model_args[@]}" --concurrency "$SWEEP" --input-len 512 --output-len 128 \
      --json "$OUT/$engine-serve.json" || echo "bench-serve against $engine failed (see $engine-serve.log)"
}

# start_and_measure ENGINE PORT COMMAND...
start_and_measure() {
  local engine="$1" port="$2"; shift 2
  say "$engine on port $port"
  if [ "$DRY_RUN" = 1 ]; then printf '[dry-run] %s\n' "$*"; measure "$engine" "$port"; return 0; fi
  "$@" > "$OUT/$engine-server.log" 2>&1 &
  SERVER_PID=$!
  wait_ready "$port" "$SERVER_PID"
  case $? in
    0) measure "$engine" "$port" ;;
    2) echo "$engine exited while loading; the end of its log:"; tail -n 15 "$OUT/$engine-server.log"; ENGINE_FAILED="${ENGINE_FAILED:-} $engine" ;;
    *) echo "$engine did not answer in ${READY_SECONDS}s"; ENGINE_FAILED="${ENGINE_FAILED:-} $engine" ;;
  esac
  stop_engine
}

[ -n "$MODEL_GGUF" ] || { echo "MODEL_GGUF is not set (the model file for fLLM)"; exit 2; }
[ "$DRY_RUN" = 1 ] || [ -f "$MODEL_GGUF" ] || { echo "MODEL_GGUF: $MODEL_GGUF does not exist"; exit 2; }
[ "$DRY_RUN" = 1 ] || command -v "$FLLM" >/dev/null 2>&1 || { echo "$FLLM is not in PATH (FLLM=...)"; exit 2; }
[ "$DRY_RUN" = 1 ] || mkdir -p "$OUT"

say "environment"
if [ "$DRY_RUN" != 1 ]; then
  {
    date -u +%FT%TZ; uname -a
    echo "--- fllm-rs"; "$FLLM" --version 2>&1 | head -2
    echo "--- rocm"; (command -v rocm-smi >/dev/null && { rocm-smi --showproductname --showdriverversion 2>&1 | head -40; rocm-smi --showtopo 2>&1 | head -60; }) || echo "no rocm-smi"
    echo "--- nvidia"; (command -v nvidia-smi >/dev/null && { nvidia-smi 2>&1 | head -30; nvidia-smi topo -m 2>&1 | head -30; }) || echo "no nvidia-smi"
    echo "--- vllm / sglang"; (python3 -m pip list 2>/dev/null | grep -iE '^(vllm|sglang|torch|flashinfer|aiter)' ) || true
  } > "$OUT/environment.txt" 2>&1
  echo "written to $OUT/environment.txt"
fi

if [ "$DEVICES" != cpu ]; then
  say "gates (the first one that fails ends the run)"
  run devices "$FLLM" devices || die "fllm-rs devices"
  run doctor "$FLLM" doctor --gpu || die "fllm-rs doctor --gpu"
  IFS=, read -r -a LIST <<< "$DEVICES"
  # Both commands end with exit code 0 on a machine without a GPU or with a binary that has no GPU backend, so the exit code is no gate:
  # read what they say. The backend of every requested device must be compiled in, and the devices must be listed.
  if [ "$DRY_RUN" != 1 ]; then
    compiled=$(grep -m1 '^Compiled backends:' "$OUT/devices.log" | sed 's/^Compiled backends: *//')
    for dev in "${LIST[@]}"; do
      backend="${dev%%:*}"
      case ",$(echo "$compiled" | tr -d ' ')," in
        *",$backend,"*) ;;
        *) die "this binary has no '$backend' backend (compiled: ${compiled:-unknown}); build it with that GPU backend (docs/hip.md for AMD)" ;;
      esac
    done
    grep -q 'no GPU found' "$OUT/doctor.log" && die "doctor --gpu: no GPU found (driver missing or not visible to this user)"
    listed=$(grep -c '^  \[[0-9]*\] ' "$OUT/devices.log")
    [ "$listed" -ge "${#LIST[@]}" ] || die "fllm-rs devices lists $listed device(s), ${#LIST[@]} requested"
  fi
  for dev in "${LIST[@]}"; do
    run "selftest-$dev" "$FLLM" selftest-gpu --device "$dev" --failures-only || die "selftest-gpu on $dev"
  done
  run selftest-comm "$FLLM" selftest-comm || die "selftest-comm (RCCL / NCCL part)"
  FLLM_CUSTOM_ALLREDUCE=1 run selftest-comm-custom "$FLLM" selftest-comm || die "selftest-comm with FLLM_CUSTOM_ALLREDUCE=1"
  say "gates passed"
else
  say "DEVICES=cpu: the GPU gates are skipped"
fi

ENGINE_FAILED=""
if [ "$DEVICES" = cpu ]; then
  start_and_measure fllm "$PORT" "$FLLM" serve --model "$MODEL_GGUF" --device cpu --port "$PORT"
else
  start_and_measure fllm "$PORT" "$FLLM" serve --model "$MODEL_GGUF" --device "$DEVICES" --tensor-parallel "$TP" --port "$PORT"
  # the same run with the old host path, so the effect of the collectives shows in the numbers
  FLLM_HOST_ALLREDUCE=1 start_and_measure fllm-host-allreduce "$((PORT + 3))" "$FLLM" serve --model "$MODEL_GGUF" --device "$DEVICES" --tensor-parallel "$TP" --port "$((PORT + 3))"
fi

if [ -n "${VLLM_CMD:-}" ]; then
  # shellcheck disable=SC2086
  start_and_measure vllm "$((PORT + 1))" bash -c "${VLLM_CMD//\{PORT\}/$((PORT + 1))}"
fi
if [ -n "${SGLANG_CMD:-}" ]; then
  start_and_measure sglang "$((PORT + 2))" bash -c "${SGLANG_CMD//\{PORT\}/$((PORT + 2))}"
fi

if [ -n "$ENGINE_FAILED" ]; then summary "finished, but did not start:$ENGINE_FAILED"; exit 1; fi
summary "finished"
