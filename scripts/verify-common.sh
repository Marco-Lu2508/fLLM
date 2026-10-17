#!/usr/bin/env bash
# Shared driver of scripts/verify-<backend>.sh (see docs/backends-verify.md).
#
# A wrapper sets the V_* variables below, optionally defines the hooks `pre_steps` and `case_hook`, and calls `v_main "$@"`.
# Phases (all on by default, `--skip build,bench` leaves some out):
#   build     cargo build --release -p fllm-engine --features $V_FEATURES --bin fllm-rs
#   devices   fllm-rs devices
#   pre       backend specific steps: the function `pre_steps` (usually calls to the scripts and tests of the package)
#   selftest  fllm-rs selftest-gpu --device $V_DEVICE --json ... (exact math first and fast math second when V_SELFTEST_TWICE=1)
#   text      greedy text (temperature 0) of the reference (cpu) against the device: default math, exact math, extra cases;
#             when the default differs, one more case per switch in V_DIAG_ENVS
#   bench     timed chat requests and `fllm-rs bench` against a server on the device
# Output: $V_OUT (default verify-out/<backend>) holds one log per step and report.json with device, commit, command, exit code,
# duration and tokens/s. The exit status is 1 when a step failed; a text that differs from the CPU is reported, not counted as a failure
# (the fast variants round differently, see docs/backends-verify.md).
#
# Level of this script: L0. It has been syntax checked (bash -n) and never run. Works with bash 3.2 (macOS).

V_RECORDS=()
V_FAILED=0
V_TOK_S=""
V_PID=""
V_LAST_SAME=""
V_START=$SECONDS

v_log() { printf '[verify-%s] %s\n' "${V_BACKEND:-?}" "$*" >&2; }

# absolute form of a path given relative to the directory the script was started in (empty stays empty)
v_abs() {
  case "$1" in
    "" | /*) printf '%s' "$1" ;;
    *) printf '%s/%s' "$PWD" "$1" ;;
  esac
}

# JSON string literal of the argument
v_json_str() {
  printf '%s' "$1" | tr '\t\r' '  ' | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' |
    awk 'BEGIN { printf "\"" } NR > 1 { printf "\\n" } { printf "%s", $0 } END { printf "\"" }'
}

# v_record NAME COMMAND EXIT SECONDS LOG NOTE [EXTRA_JSON_FIELDS]
v_record() {
  local name=$1 cmd=$2 rc=$3 secs=$4 log=$5 note=${6:-} extra=${7:-} obj
  obj="{\"name\": $(v_json_str "$name"), \"command\": $(v_json_str "$cmd"), \"exit_code\": $rc, \"seconds\": $secs, \"log\": $(v_json_str "$log"), \"note\": $(v_json_str "$note")"
  if [ -n "$extra" ]; then obj="$obj, $extra"; fi
  V_RECORDS+=("$obj}")
}

v_skip() { # v_skip NAME REASON
  v_log "skip $1: $2"
  v_record "$1" "" 0 0 "" "skipped: $2" "\"skipped\": true"
}

# v_step NAME [ENV=VALUE ...] COMMAND [ARGS...]  - run in the repository root, log to $V_OUT/NAME.log, record exit code and duration.
# A failing step is recorded and the run continues. COMMAND must be an executable, not a shell function.
v_step() {
  local name=$1 start rc log
  shift
  log="$V_OUT/$name.log"
  v_log "step $name: $*"
  if [ "$V_DRY" = 1 ]; then
    v_record "$name" "$*" 0 0 "$log" "dry run"
    return 0
  fi
  start=$SECONDS
  (cd "$V_ROOT" && env "$@") >"$log" 2>&1
  rc=$?
  v_record "$name" "$*" "$rc" $((SECONDS - start)) "$log"
  if [ "$rc" -ne 0 ]; then
    V_FAILED=1
    v_log "  FAILED (exit $rc), see $log"
    tail -n 5 "$log" >&2
  fi
  return "$rc"
}

v_phase_on() { case " $V_PHASES " in *" $1 "*) return 0 ;; *) return 1 ;; esac; }

v_usage() {
  cat <<EOF
usage: scripts/verify-${V_BACKEND:-BACKEND}.sh [options]
  --model FILE.gguf      model for the text check and the benchmark (env MODEL); without one both are skipped
  --model-uri URI        fetch the model with 'fllm-rs pull URI' (env MODEL_URI), e.g. hf://Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q8_0.gguf
  --bin PATH             use this fllm-rs instead of building one (env BIN; it must have been built with the features of this backend)
  --device NAME          device argument (default ${V_DEVICE:-?}; env DEVICE), e.g. vulkan:1
  --out DIR              output directory (default verify-out/${V_BACKEND:-BACKEND})
  --port N               server port (default 18080)
  --tokens N             tokens of the text check (default 64)
  --prompt TEXT          prompt of the text check
  --skip a,b             leave out phases: build devices pre selftest text bench
  --dry-run              print the plan, run nothing
Environment: V_READY_TIMEOUT (seconds until the server answers, default 900), V_REQUEST_TIMEOUT (default 900).
EOF
}

v_parse_args() {
  V_MODEL=${MODEL:-}
  V_MODEL_URI=${MODEL_URI:-}
  V_BIN=${BIN:-}
  [ -n "${DEVICE:-}" ] && V_DEVICE=$DEVICE
  V_OUT=${OUT:-}
  V_PORT=${PORT:-18080}
  V_TOKENS=${TOKENS:-64}
  V_PROMPT=${PROMPT:-"Write one sentence about the sea."}
  V_DRY=0
  local skip=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --model) V_MODEL=$2; shift 2 ;;
      --model-uri) V_MODEL_URI=$2; shift 2 ;;
      --bin) V_BIN=$2; shift 2 ;;
      --device) V_DEVICE=$2; shift 2 ;;
      --out) V_OUT=$2; shift 2 ;;
      --port) V_PORT=$2; shift 2 ;;
      --tokens) V_TOKENS=$2; shift 2 ;;
      --prompt) V_PROMPT=$2; shift 2 ;;
      --skip) skip=$2; shift 2 ;;
      --dry-run) V_DRY=1; shift ;;
      -h | --help) v_usage; exit 0 ;;
      *) echo "unknown option: $1" >&2; v_usage >&2; exit 2 ;;
    esac
  done
  V_BIN_GIVEN=0
  [ -n "$V_BIN" ] && V_BIN_GIVEN=1
  local p
  for p in $(printf '%s' "$skip" | tr ',' ' '); do
    V_PHASES=$(printf ' %s ' "$V_PHASES" | sed "s/ $p / /")
  done
}

v_cleanup() {
  if [ -n "$V_PID" ]; then
    kill "$V_PID" 2>/dev/null
    wait "$V_PID" 2>/dev/null
    V_PID=""
  fi
}

# ---------------------------------------------------------------------------------------------------------------- helpers (python)
v_write_helpers() {
  # ask.py URL PROMPT TOKENS TIMEOUT OUT_TXT REPEATS: greedy chat request(s); the first text goes to OUT_TXT, the last request is timed
  cat >"$V_OUT/ask.py" <<'PY'
import json, sys, time, urllib.request
url, prompt, tokens, timeout, out_txt, repeats = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], int(sys.argv[6])
texts, last = [], {}
for i in range(repeats):
    body = json.dumps({"model": "m", "messages": [{"role": "user", "content": prompt}], "temperature": 0, "max_tokens": tokens}).encode()
    request = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    start = time.time()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.load(response)
    seconds = time.time() - start
    texts.append(data["choices"][0]["message"]["content"])
    n = (data.get("usage") or {}).get("completion_tokens")
    last = {"seconds": round(seconds, 3), "completion_tokens": n, "tok_s": round(n / seconds, 2) if n and seconds > 0 else None}
with open(out_txt, "w", encoding="utf-8") as f:
    f.write(texts[0])
last["repeat_equal"] = all(t == texts[0] for t in texts)
last["requests"] = repeats
print(json.dumps(last))
PY
  # cmp.py A B: "same" or the first position where two texts differ
  cat >"$V_OUT/cmp.py" <<'PY'
import sys
a = open(sys.argv[1], encoding="utf-8", errors="replace").read()
b = open(sys.argv[2], encoding="utf-8", errors="replace").read()
if a == b:
    print("same")
else:
    n = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a), len(b)))
    print("first difference at character %d (lengths %d / %d)" % (n, len(a), len(b)))
PY
}

# v_start_server NAME DEVICE [ENV=VALUE ...]  - starts `fllm-rs serve` in the background, waits until /v1/models answers
v_start_server() {
  local name=$1 dev=$2 log waited=0
  shift 2
  log="$V_OUT/server-$name.log"
  (cd "$V_ROOT" && exec env "$@" "$V_BIN" serve --model "$V_MODEL" --device "$dev" --host 127.0.0.1 --port "$V_PORT" --ctx 2048 --max-seqs 2) >"$log" 2>&1 &
  V_PID=$!
  while [ "$waited" -lt "${V_READY_TIMEOUT:-900}" ]; do
    if curl -sf "http://127.0.0.1:$V_PORT/v1/models" >/dev/null 2>&1; then return 0; fi
    kill -0 "$V_PID" 2>/dev/null || break
    sleep 2
    waited=$((waited + 2))
  done
  v_cleanup
  return 1
}

# v_ask NAME REPEATS  - asks the running server; prints the JSON of the last request, text of the first in $V_OUT/text-NAME.txt
v_ask() {
  python3 "$V_OUT/ask.py" "http://127.0.0.1:$V_PORT/v1/chat/completions" "$V_PROMPT" "$V_TOKENS" "${V_REQUEST_TIMEOUT:-900}" "$V_OUT/text-$1.txt" "$2" 2>"$V_OUT/ask-$1.err"
}

# v_text_case NAME DEVICE [ENV=VALUE ...]  - one server, two requests (the first warms up kernels and pipelines), text compared with the reference
v_text_case() {
  local name=$1 dev=$2 start res cmp extra guard=""
  shift 2
  V_LAST_SAME=""
  v_log "text case $name: --device $dev $*"
  if [ "$V_DRY" = 1 ]; then
    v_record "text-$name" "serve --device $dev $*" 0 0 "" "dry run"
    return 0
  fi
  start=$SECONDS
  if ! v_start_server "$name" "$dev" "$@"; then
    V_FAILED=1
    V_LAST_SAME="error"
    v_record "text-$name" "serve --device $dev $*" 1 $((SECONDS - start)) "$V_OUT/server-$name.log" "server did not start"
    tail -n 8 "$V_OUT/server-$name.log" >&2
    return 1
  fi
  res=$(v_ask "$name" 2)
  if [ -z "$res" ]; then
    V_FAILED=1
    V_LAST_SAME="error"
    v_cleanup
    v_record "text-$name" "serve --device $dev $*" 1 $((SECONDS - start)) "$V_OUT/ask-$name.err" "request failed"
    return 1
  fi
  v_cleanup
  if [ "$name" = "ref" ]; then
    cmp="reference"
  else
    cmp=$(python3 "$V_OUT/cmp.py" "$V_OUT/text-ref.txt" "$V_OUT/text-$name.txt")
  fi
  V_LAST_SAME=$cmp
  # the native Vulkan and Metal backends can fall back to wgpu (Decoder::set_device); a log that mentions wgpu may mean that
  if [ "$name" != "ref" ] && [ -n "${V_NATIVE_GUARD:-}" ] && grep -qi "$V_NATIVE_GUARD" "$V_OUT/server-$name.log" 2>/dev/null; then
    guard=", \"log_mentions_$V_NATIVE_GUARD\": true"
  fi
  extra="\"text_vs_cpu\": $(v_json_str "$cmp"), \"request\": $res$guard"
  v_record "text-$name" "serve --device $dev $*" 0 $((SECONDS - start)) "$V_OUT/server-$name.log" "" "$extra"
  if [ "$dev" = "${V_DEVICE:-}" ] && [ "$name" != "ref" ]; then
    V_TOK_S=$(printf '%s' "$res" | sed -n 's/.*"tok_s": \([0-9.]*\).*/\1/p')
  fi
  v_log "  $name: $cmp"
  return 0
}

# ---------------------------------------------------------------------------------------------------------------- phases
v_find_bin() {
  if [ -z "$V_BIN" ]; then V_BIN="${CARGO_TARGET_DIR:-$V_ROOT/target}/release/fllm-rs"; fi
}

v_build() {
  local feat=()
  if [ "$V_BIN_GIVEN" = 1 ]; then v_skip build "--bin given"; return; fi
  if [ -n "$V_FEATURES" ]; then feat=(--features "$V_FEATURES"); fi
  # shellcheck disable=SC2086  # V_BUILD_ENV is a list of NAME=VALUE words
  v_step build $V_BUILD_ENV cargo build --release -p fllm-engine "${feat[@]}" --bin fllm-rs
}

v_devices() {
  if [ "$V_DRY" != 1 ] && [ ! -x "$V_BIN" ]; then v_skip devices "no binary at $V_BIN"; return; fi
  # shellcheck disable=SC2086
  v_step devices $V_COMMON_ENV "$V_BIN" devices
  if [ "$V_DRY" != 1 ]; then head -n 60 "$V_OUT/devices.log" >&2; fi
}

v_selftest() {
  local e name rc
  if [ "$V_DRY" != 1 ] && [ ! -x "$V_BIN" ]; then v_skip selftest "no binary at $V_BIN"; return; fi
  if [ -n "$V_EXACT_ENV" ] && [ "$V_SELFTEST_TWICE" = 1 ]; then
    # shellcheck disable=SC2086
    v_step selftest-exact $V_COMMON_ENV $V_SELFTEST_ENV $V_EXACT_ENV "$V_BIN" selftest-gpu --device "$V_DEVICE" --json "$V_OUT/selftest-exact.json"
    # shellcheck disable=SC2086
    v_step selftest-fast $V_COMMON_ENV $V_SELFTEST_ENV "$V_BIN" selftest-gpu --device "$V_DEVICE" --json "$V_OUT/selftest-fast.json"
    return
  fi
  # shellcheck disable=SC2086
  v_step selftest $V_COMMON_ENV $V_SELFTEST_ENV "$V_BIN" selftest-gpu --device "$V_DEVICE" --json "$V_OUT/selftest.json"
  rc=$?
  if [ "$rc" -ne 0 ] && [ "$V_DRY" != 1 ]; then
    # find the switch that makes the failure go away; these runs do not change the verdict above
    for e in $V_DIAG_ENVS $V_EXACT_ENV; do
      name="selftest-with-$(printf '%s' "$e" | sed 's/[^A-Za-z0-9_=.-]/_/g')"
      local before=$V_FAILED
      # shellcheck disable=SC2086
      v_step "$name" $V_COMMON_ENV $V_SELFTEST_ENV "$e" "$V_BIN" selftest-gpu --device "$V_DEVICE" --json "$V_OUT/$name.json"
      V_FAILED=$before
    done
  fi
}

v_pull() {
  local path
  if [ -n "$V_MODEL" ] || [ -z "$V_MODEL_URI" ]; then return; fi
  if [ "$V_DRY" = 1 ]; then v_record pull "$V_BIN pull $V_MODEL_URI" 0 0 "" "dry run"; return; fi
  if v_step pull "$V_BIN" pull "$V_MODEL_URI"; then
    path=$(tail -n 1 "$V_OUT/pull.log")
    if [ -f "$path" ]; then V_MODEL=$path; else v_log "pull printed '$path', which is not a file"; fi
  fi
}

v_text() {
  local line name dev envs e before
  if [ -z "$V_MODEL" ] && [ "$V_DRY" != 1 ]; then v_skip text "no model (--model or --model-uri)"; return; fi
  if ! command -v python3 >/dev/null 2>&1; then v_skip text "python3 not found"; return; fi
  if ! command -v curl >/dev/null 2>&1; then v_skip text "curl not found"; return; fi
  v_text_case ref cpu $V_REF_ENV || { v_skip text-cases "reference run failed"; return; }
  # shellcheck disable=SC2086
  v_text_case default "$V_DEVICE" $V_COMMON_ENV
  local default_result=$V_LAST_SAME
  if [ -n "$V_EXACT_ENV" ]; then
    # shellcheck disable=SC2086
    v_text_case exact "$V_DEVICE" $V_COMMON_ENV $V_EXACT_ENV
  fi
  # extra cases: "name|device|ENV=VALUE ENV=VALUE" per line (V_EXTRA_CASES, then the output of case_hook)
  local cases="$V_EXTRA_CASES"
  if [ "$(type -t case_hook)" = "function" ]; then cases="$cases
$(case_hook)"; fi
  while IFS='|' read -r name dev envs; do
    [ -n "$name" ] || continue
    # shellcheck disable=SC2086
    v_text_case "$name" "${dev:-$V_DEVICE}" $envs
  done <<EOF
$cases
EOF
  if [ -n "$V_DIAG_ENVS" ] && [ "$default_result" != "same" ] && [ "$default_result" != "" ]; then
    v_log "default text differs from the CPU: one case per switch ($V_DIAG_ENVS)"
    for e in $V_DIAG_ENVS; do
      # shellcheck disable=SC2086
      v_text_case "diag-$(printf '%s' "$e" | sed 's/[^A-Za-z0-9_=.-]/_/g')" "$V_DEVICE" $V_COMMON_ENV "$e"
    done
  fi
}

v_bench() {
  local start res
  if [ -z "$V_MODEL" ] && [ "$V_DRY" != 1 ]; then v_skip bench "no model (--model or --model-uri)"; return; fi
  if [ "$V_DRY" = 1 ]; then
    v_record bench "$V_BIN bench --url http://127.0.0.1:$V_PORT --scenarios prefill,decode-single --json $V_OUT/bench.json" 0 0 "" "dry run"
    return
  fi
  if ! command -v python3 >/dev/null 2>&1 || ! command -v curl >/dev/null 2>&1; then v_skip bench "python3 or curl not found"; return; fi
  start=$SECONDS
  # shellcheck disable=SC2086
  if ! v_start_server bench "$V_DEVICE" $V_COMMON_ENV; then
    V_FAILED=1
    v_record bench "serve --device $V_DEVICE" 1 $((SECONDS - start)) "$V_OUT/server-bench.log" "server did not start"
    return
  fi
  res=$(v_ask bench-timed 3)
  if [ -n "$res" ]; then
    v_record bench-timed-request "chat completion, 3 requests, last one timed (includes the prompt)" 0 0 "$V_OUT/ask-bench-timed.err" "" "\"request\": $res"
    V_TOK_S=$(printf '%s' "$res" | sed -n 's/.*"tok_s": \([0-9.]*\).*/\1/p')
  else
    V_FAILED=1
    v_record bench-timed-request "chat completion" 1 0 "$V_OUT/ask-bench-timed.err" "request failed"
  fi
  # the server of this phase stays up for fllm-rs bench; v_step runs the client
  v_step bench "$V_BIN" bench --url "http://127.0.0.1:$V_PORT" --scenarios prefill,decode-single --prompt-lens 512 --gen-len 128 --warmup 1 --repeats 2 --label "$V_BACKEND $V_DEVICE" --json "$V_OUT/bench.json"
  v_cleanup
}

v_finish() {
  local out="$V_OUT/report.json" commit dirty i n sep="" duration level
  commit=$(git -C "$V_ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)
  dirty=false
  if [ -n "$(git -C "$V_ROOT" status --porcelain 2>/dev/null)" ]; then dirty=true; fi
  duration=$((SECONDS - V_START))
  {
    printf '{\n'
    printf '  "backend": %s,\n' "$(v_json_str "$V_BACKEND")"
    printf '  "device": %s,\n' "$(v_json_str "$V_DEVICE")"
    printf '  "commit": %s,\n' "$(v_json_str "$commit")"
    printf '  "tree_dirty": %s,\n' "$dirty"
    printf '  "command": %s,\n' "$(v_json_str "$V_COMMAND")"
    printf '  "features": %s,\n' "$(v_json_str "$V_FEATURES")"
    printf '  "model": %s,\n' "$(v_json_str "$V_MODEL")"
    printf '  "host": %s,\n' "$(v_json_str "$(uname -srm 2>/dev/null)")"
    printf '  "date_utc": %s,\n' "$(v_json_str "$(date -u +%Y-%m-%dT%H:%M:%SZ)")"
    printf '  "exit_code": %s,\n' "$V_FAILED"
    printf '  "duration_seconds": %s,\n' "$duration"
    printf '  "tok_s": %s,\n' "${V_TOK_S:-null}"
    printf '  "tok_s_note": "chat completion of the device server, includes the prompt; not a benchmark (see bench.json for fllm-rs bench)",\n'
    printf '  "script_level": "L0 (syntax checked only before its first run)",\n'
    printf '  "steps": [\n'
    n=${#V_RECORDS[@]}
    i=0
    while [ "$i" -lt "$n" ]; do
      if [ "$i" -lt $((n - 1)) ]; then sep=","; else sep=""; fi
      printf '    %s%s\n' "${V_RECORDS[$i]}" "$sep"
      i=$((i + 1))
    done
    printf '  ]\n}\n'
  } >"$out"
  v_log "report: $out (exit code $V_FAILED, $duration s)"
  if [ "$V_DRY" = 1 ]; then v_log "dry run: nothing was executed"; fi
  level="no step failed"
  if [ "$V_FAILED" -ne 0 ]; then level="at least one step failed: the matrix cell stays at L0"; fi
  v_log "$level. Levels are assigned by hand from the report: docs/backends-verify.md, section 'Auswertung'."
}

# ---------------------------------------------------------------------------------------------------------------- main
v_main() {
  V_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
  V_COMMAND="$0 $*"
  : "${V_BACKEND:?wrapper must set V_BACKEND}"
  V_FEATURES=${V_FEATURES:-}
  V_BUILD_ENV=${V_BUILD_ENV:-}
  V_DEVICE=${V_DEVICE:-auto}
  V_COMMON_ENV=${V_COMMON_ENV:-}
  V_SELFTEST_ENV=${V_SELFTEST_ENV:-}
  V_EXACT_ENV=${V_EXACT_ENV:-}
  V_DIAG_ENVS=${V_DIAG_ENVS:-}
  V_REF_ENV=${V_REF_ENV:-}
  V_EXTRA_CASES=${V_EXTRA_CASES:-}
  V_SELFTEST_TWICE=${V_SELFTEST_TWICE:-0}
  V_PHASES=${V_PHASES:-"build devices pre selftest text bench"}
  v_parse_args "$@"
  V_MODEL=$(v_abs "$V_MODEL")
  V_BIN=$(v_abs "$V_BIN")
  V_OUT=$(v_abs "$V_OUT")
  : "${V_OUT:=$V_ROOT/verify-out/$V_BACKEND}"
  mkdir -p "$V_OUT"
  trap v_cleanup EXIT INT TERM
  v_find_bin
  v_log "root $V_ROOT, device $V_DEVICE, features '${V_FEATURES}', output $V_OUT"
  v_write_helpers

  if v_phase_on build; then v_build; fi
  if v_phase_on devices; then v_devices; fi
  if v_phase_on pre; then
    if [ "$(type -t pre_steps)" = "function" ]; then pre_steps; else v_skip pre "this backend has no extra steps"; fi
  fi
  if v_phase_on selftest; then v_selftest; fi
  if v_phase_on text || v_phase_on bench; then v_pull; fi
  if v_phase_on text; then v_text; fi
  if v_phase_on bench; then v_bench; fi
  v_finish
  exit "$V_FAILED"
}
