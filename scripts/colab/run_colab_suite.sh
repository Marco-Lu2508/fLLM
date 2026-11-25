#!/usr/bin/env bash
# End-to-end Google Colab GPU verification, hill-climb optimization, and vLLM/SGLang benchmark suite.
# Works directly inside Colab (`!bash fLLM/scripts/colab/run_colab_suite.sh`), via Antigravity CLI (`agy`),
# or via `google-colab-cli` (`colab ssh ...` / `colab exec ...`).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="${WORK_DIR:-/content/fllm_suite}"
mkdir -p "${WORK_DIR}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "ERROR: No NVIDIA GPU found (nvidia-smi missing). Switch Colab runtime to T4 / L4 / A100." >&2
  exit 1
fi

GPU_INFO="$(nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader | head -n1)"
GPU_NAME="$(echo "${GPU_INFO}" | cut -d',' -f1 | xargs)"
VRAM_MIB="$(echo "${GPU_INFO}" | cut -d',' -f2 | tr -dc '0-9')"
CC_RAW="$(echo "${GPU_INFO}" | cut -d',' -f3 | tr -dc '0-9')"
echo "Detected GPU: ${GPU_NAME} (${VRAM_MIB} MiB VRAM, sm_${CC_RAW})"

if [ "${CC_RAW:-80}" -lt 80 ]; then
  export FLLM_ACT_DTYPE="${FLLM_ACT_DTYPE:-fp16}"
  COMPARE_MODELS="${COMPARE_MODELS:-qwen7b-awq,qwen15-moe-gptq}"
  CLIMB_PRESET="${CLIMB_PRESET:-qwen3-8b-dspine}"
  CLIMB_GPU="t4"
elif [ "${VRAM_MIB:-24000}" -ge 38000 ]; then
  export FLLM_ACT_DTYPE="${FLLM_ACT_DTYPE:-bf16}"
  COMPARE_MODELS="${COMPARE_MODELS:-qwen7b-awq,qwen3-30b-a3b-gptq}"
  CLIMB_PRESET="${CLIMB_PRESET:-qwen3.5-27b-dspine}"
  CLIMB_GPU=$([ "${VRAM_MIB}" -ge 75000 ] && echo "a100-80g" || echo "a100-40g")
else
  export FLLM_ACT_DTYPE="${FLLM_ACT_DTYPE:-bf16}"
  COMPARE_MODELS="${COMPARE_MODELS:-qwen7b-awq,qwen15-moe-gptq}"
  CLIMB_PRESET="${CLIMB_PRESET:-qwen3-8b-dspine}"
  CLIMB_GPU="l4"
fi

echo "Selected FLLM_ACT_DTYPE=${FLLM_ACT_DTYPE}, CLIMB_PRESET=${CLIMB_PRESET} (${CLIMB_GPU}), COMPARE_MODELS=${COMPARE_MODELS}"

FLLM_BIN="${FLLM_BIN:-${WORK_DIR}/fllm-rs}"
if [ ! -x "${FLLM_BIN}" ]; then
  echo "Downloading colab-latest CUDA binary..."
  curl -L --retry 8 --retry-all-errors -o "${WORK_DIR}/fllm-colab-cuda.tar.gz" \
    "https://github.com/Marco-Lu2508/fLLM/releases/download/colab-latest/fllm-colab-cuda.tar.gz"
  tar -C "${WORK_DIR}" -xzf "${WORK_DIR}/fllm-colab-cuda.tar.gz"
  chmod +x "${FLLM_BIN}"
fi

echo "=== 1/3 verify_all.py ==="
python3 "${ROOT_DIR}/scripts/colab/verify_all.py" \
  --binary "${FLLM_BIN}" \
  --steps hw,devices,serve,speed,dspine,fp8kv \
  --out "${WORK_DIR}/verify_results.json"

echo "=== 2/3 hill_climb_colab.py (${CLIMB_PRESET} @ ${CLIMB_GPU}) ==="
python3 "${ROOT_DIR}/scripts/colab/hill_climb_colab.py" \
  --preset "${CLIMB_PRESET}" \
  --gpu "${CLIMB_GPU}" \
  --fllm-bin "${FLLM_BIN}" \
  --ledger "${WORK_DIR}/ledger.jsonl"

echo "=== 3/3 compare.py (fLLM vs vLLM vs SGLang, users 1,2,5,10,50) ==="
python3 "${ROOT_DIR}/scripts/colab/compare.py" \
  --work "${WORK_DIR}/compare" \
  --fllm "${FLLM_BIN}" \
  --models "${COMPARE_MODELS}" \
  --sizes "${COMPARE_SIZES:-2000,4000,8000}" \
  --users "${COMPARE_USERS:-1,2,5,10,50}" \
  --only fllm,vllm,sglang --no-llama --no-ik --no-ollama

echo "ALL DONE. Artifacts in ${WORK_DIR}:"
ls -lh "${WORK_DIR}/verify_results.json" "${WORK_DIR}/ledger.jsonl" "${WORK_DIR}/compare/compare_result.json" 2>/dev/null || true
