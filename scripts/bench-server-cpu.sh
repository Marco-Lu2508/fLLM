#!/usr/bin/env bash
# CPU comparison on a Linux box with Docker: fLLM (a binary built for this machine) against llama.cpp and Ollama (their official
# images), the same GGUF, the same requests (scripts/colab/compare.py's load test: prompt lengths x simultaneous users).
# usage: bash scripts/bench-server-cpu.sh FLLM_BINARY MODEL.gguf [sizes] [users] [engines]
#   e.g. bash scripts/bench-server-cpu.sh ~/cargo-target-shared/release/fllm-rs ~/bench-models/Qwen3.5-2B-Q4_K_M.gguf 100,500 1,4 fllm,llama
set -u
FLLM=$1
MODEL=$(readlink -f "$2")
SIZES=${3:-100,500,2000}
USERS=${4:-1,4,8}
ENGINES=${5:-fllm,llama,ollama}
HERE=$(cd "$(dirname "$0")" && pwd)
DIR=$(dirname "$MODEL")
NAME=$(basename "$MODEL")
THREADS=${THREADS:-$(nproc)}
MAXU=$(echo "$USERS" | tr ',' '\n' | sort -n | tail -1)
MAXS=$(echo "$SIZES" | tr ',' '\n' | sort -n | tail -1)
CTX=$(( (MAXS + 464) * MAXU ))
wait_up() { for _ in $(seq 1 300); do curl -sf "$1" > /dev/null && return 0; sleep 1; done; return 1; }
NO_THINK='{"chat_template_kwargs": {"enable_thinking": false}}'
measure() {
  local extra=${4:-$NO_THINK}
  python3 - "$1" "$2" "$SIZES" "$USERS" "$3" "$extra" <<'PY'
import importlib.util, json, sys
spec = importlib.util.spec_from_file_location("compare", sys.argv[5])
c = importlib.util.module_from_spec(spec); spec.loader.exec_module(c)
port, model = int(sys.argv[1]), sys.argv[2]
sizes = [int(x) for x in sys.argv[3].split(",")]; users = [int(x) for x in sys.argv[4].split(",")]
server = type("S", (), {"model": model})()
rows = c.measure(server, port, sizes, users, 64, json.loads(sys.argv[6]))
print(json.dumps(rows))
PY
}
for engine in ${ENGINES//,/ }; do
  echo "== $engine ($NAME, $THREADS threads)"
  case $engine in
    fllm)
      # FLLM_* variables of the caller reach the server (FLLM_THREADS is the --threads value)
      envs=$(env | grep -E '^FLLM_' | grep -v '^FLLM_THREADS=' | sed 's/^/-e /' | paste -sd ' ')
      docker run -d --rm --name bench-fllm --network host $envs -v "$FLLM":/fllm-rs:ro -v "$DIR":/m:ro rust:1-trixie \
        /fllm-rs serve --model "/m/$NAME" --device cpu --port 18100 --ctx "$CTX" --max-seqs "$MAXU" --model-name m --threads "${FLLM_THREADS:-auto}" > /dev/null
      wait_up http://127.0.0.1:18100/health && measure 18100 m "$HERE/colab/compare.py"
      docker logs bench-fllm > /tmp/bench-fllm.log 2>&1
      grep -E "threads:|ERROR" /tmp/bench-fllm.log | head -3
      docker stop bench-fllm > /dev/null ;;
    llama)
      docker run -d --rm --name bench-llama --network host -v "$DIR":/m:ro ghcr.io/ggml-org/llama.cpp:server \
        -m "/m/$NAME" -c "$CTX" -np "$MAXU" -t "$THREADS" --host 127.0.0.1 --port 18101 --alias m > /dev/null
      wait_up http://127.0.0.1:18101/health && measure 18101 m "$HERE/colab/compare.py"
      docker stop bench-llama > /dev/null ;;
    ollama)
      docker run -d --rm --name bench-ollama --network host -v "$DIR":/m:ro -e OLLAMA_HOST=127.0.0.1:18102 -e OLLAMA_NUM_PARALLEL="$MAXU" \
        -e OLLAMA_CONTEXT_LENGTH=$(( MAXS + 464 )) ollama/ollama:latest > /dev/null
      wait_up http://127.0.0.1:18102/api/version && docker exec bench-ollama sh -c "printf 'FROM /m/$NAME\n' > /tmp/Modelfile && OLLAMA_HOST=127.0.0.1:18102 ollama create bench -f /tmp/Modelfile" > /dev/null 2>&1
      measure 18102 bench "$HERE/colab/compare.py" '{"reasoning_effort": "none"}'
      docker stop bench-ollama > /dev/null ;;
  esac
done
