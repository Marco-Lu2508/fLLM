#!/usr/bin/env bash
# Starts fllm-rs on the GPU, sends N simultaneous chat requests and prints what came back and the end of the server log.
# usage: bash scripts/colab/multiuser_check.sh [BINARY] [MODEL.gguf] [USERS] [extra serve arguments ...]
B=${1:-/content/fllm_compare/fllm-src/target/release/fllm-rs}
M=${2:-/content/fllm_compare/Qwen3.5-2B-Q4_K_M.gguf}
N=${3:-16}
shift 3 2>/dev/null
PORT=9999
LOG=/tmp/fllm-multiuser.log
pkill -x fllm-rs 2>/dev/null; sleep 1
"$B" serve --model "$M" --device cuda --port $PORT --ctx 9024 --max-seqs "$N" "$@" > "$LOG" 2>&1 &
for _ in $(seq 1 120); do curl -sf localhost:$PORT/health > /dev/null && break; sleep 1; done
echo "server up, sending $N requests"
python3 - "$PORT" "$N" <<'PY'
import json, sys, threading, urllib.request
port, n = int(sys.argv[1]), int(sys.argv[2])
text = "The quick brown fox jumps over the lazy dog. " * 10
out = [None] * n
def one(i):
    body = {"messages": [{"role": "user", "content": f"Run {i}: count the words. {text}"}], "max_tokens": 32, "temperature": 0}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            d = json.load(r)
            out[i] = "ok: " + d["choices"][0]["message"]["content"].replace("\n", " ")[:60]
    except urllib.error.HTTPError as e:
        out[i] = f"HTTP {e.code}: " + e.read().decode(errors="replace")[:300]
    except Exception as e:
        out[i] = f"failed: {e}"
threads = [threading.Thread(target=one, args=(i,)) for i in range(n)]
[t.start() for t in threads]
[t.join() for t in threads]
for i, o in enumerate(out):
    print(f"request {i:2}: {o}")
PY
sleep 2
echo "---- server log (without INFO lines) ----"
grep -v " INFO " "$LOG" | grep -v "OpenVINO" | tail -30
pkill -x fllm-rs 2>/dev/null
