#!/bin/bash
# CLI acceptance test against live CPU servers.
# Expects: servers on 8990 (no key) and 8991 (--api-key secret123), model test-model.
# Usage: FLLM_TEST_MODEL=/path/model.gguf ./cli-tests.sh [BIN] [BASE0] [BASE1]
# (start the two servers first, for example with `fllm-rs serve --model M --model-name test-model --port 8990 --device cpu` and the same
# with `--port 8991 --api-key secret123`; any small GGUF works)
set -u
BIN="${1:-${FLLM_RS:-./target/release/fllm-rs}}"
MODEL="${FLLM_TEST_MODEL:?set FLLM_TEST_MODEL to a small .gguf file}"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
BASE0="${2:-http://127.0.0.1:8990}"
BASE1="${3:-http://127.0.0.1:8991}"
PASS=0; FAIL=0
ok()   { PASS=$((PASS+1)); echo "PASS: $1"; }
bad()  { FAIL=$((FAIL+1)); echo "FAIL: $1 -- $2"; }

# 1. top-level help
if "$BIN" --help 2>&1 | grep -qi "usage"; then ok "--help"; else bad "--help" "no usage text"; fi

# 2. devices lists CPU backend
if "$BIN" devices 2>&1 | grep -qi "cpu"; then ok "devices shows cpu"; else bad "devices" "no cpu line"; fi

# 3. formats lists storage types
if "$BIN" formats 2>&1 | grep -qi "q4_k_m\|q8_0"; then ok "formats lists quants"; else bad "formats" "no quant lines"; fi

# 4. ps exits 0 (managed-server listing; manually started servers may not appear)
if "$BIN" ps >/dev/null 2>&1; then ok "ps exit 0"; else bad "ps" "nonzero exit"; fi

# 5. serve --help documents key flags
if "$BIN" serve --help 2>&1 | grep -q -- "--api-key" && "$BIN" serve --help 2>&1 | grep -q -- "--ctx"; then
  ok "serve --help flags"
else
  bad "serve --help" "missing --api-key/--ctx"
fi

# 6. chat: piped single question, expect non-empty assistant reply
CHAT_OUT=$(printf 'Say hi in exactly five words\n' | timeout 120 "$BIN" chat --url "$BASE0" --model test-model --max-new 32 2>$TMP/chat.err | tail -20)
if [ -n "$CHAT_OUT" ]; then ok "chat piped reply: $(echo "$CHAT_OUT" | head -c 80)"; else bad "chat" "$(head -c 300 $TMP/chat.err)"; fi

# 7. bench smoke: prefill only, fits ctx-1024 server
if timeout 300 "$BIN" bench --url "$BASE0" --model test-model --scenarios prefill --prompt-lens 128 --gen-len 32 --repeats 1 --warmup 1 >$TMP/bench.log 2>&1; then
  ok "bench smoke done"
else
  bad "bench" "$(tail -c 300 $TMP/bench.log)"
fi

# 8. run: local gguf end-to-end (starts own server on free port, chat via pipe, quit on EOF)
RUN_OUT=$(printf 'hi\n' | timeout 170 "$BIN" run "$MODEL" --ctx 2048 --threads 4 >$TMP/run.log 2>&1; echo "exit=$?")
if grep -q "127.0.0.1" $TMP/run.log && ! grep -qi "error" $TMP/run.log; then
  ok "run starts server + chat reply ($RUN_OUT)"
else
  bad "run" "$(tail -c 400 $TMP/run.log) $RUN_OUT"
fi

# 9. keyed server: chat with --api-key
CHAT1=$(printf 'Say ok\n' | timeout 120 "$BIN" chat --url "$BASE1" --model test-model --api-key secret123 --max-new 16 2>$TMP/chat1.err | tail -5)
if [ -n "$CHAT1" ]; then ok "chat keyed reply"; else bad "chat keyed" "$(head -c 300 $TMP/chat1.err)"; fi

# 10. keyed server without key must fail
if printf 'hi\n' | timeout 60 "$BIN" chat --url "$BASE1" --model test-model --max-new 8 >/dev/null 2>$TMP/chat2.err; then
  bad "chat no-key-ok" "expected auth failure but got reply"
else
  ok "chat without key refused"
fi

echo "== cli-tests: $PASS passed, $FAIL failed =="
exit $([ "$FAIL" -eq 0 ])
