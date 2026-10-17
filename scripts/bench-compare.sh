#!/usr/bin/env bash
# Basic fLLM-only benchmark. Optional vLLM/llama.cpp paths are left as env vars.
set -euo pipefail
MODEL="${MODEL:?set MODEL to a .gguf file}"
BIN="${BIN:-./target/release/fllm-rs}"
PORT=8999
$BIN run "$MODEL" --device cpu --threads 4 --port $PORT >${TMPDIR:-/tmp}/bench.log 2>&1 &
PID=$!
trap "kill $PID 2>/dev/null || true" EXIT
for i in $(seq 1 60); do curl -s http://127.0.0.1:$PORT/health >/dev/null && break; sleep 1; done
python3 - <<'PY'
import time,json,subprocess,urllib.request
url="http://127.0.0.1:8999/v1/chat/completions"
prompt="word " * 355
body={"model":"m","messages":[{"role":"user","content":prompt}],"max_tokens":16,"temperature":0,"stream":True}
data=json.dumps(body).encode()
t0=time.time(); req=urllib.request.Request(url,data=data,headers={"Content-Type":"application/json"})
first=None; chunks=0
with urllib.request.urlopen(req) as r:
  for line in r:
    if b'"content"' in line:
      if first is None: first=time.time()
      chunks+=1
print("ttft_ms", round((first-t0)*1000,1) if first else None, "decode_chunks", chunks)
PY
