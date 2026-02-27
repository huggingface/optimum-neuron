#!/bin/bash
#
# Run a performance benchmark against a vLLM server using guidellm.
#
# Usage:
#   ./performance.sh [model] [concurrent_users]
#
# When model is omitted the script auto-detects it from the /v1/models
# endpoint (requires the vLLM server to be running).
#
# Environment variables:
#   VLLM_TARGET  – base URL of the vLLM server (default: http://localhost:8080/v1)

set -e

target=${VLLM_TARGET:-http://localhost:8080/v1}
users=${2:-128}

# Auto-detect the model name from the vLLM server when not provided.
if [ -z "$1" ]; then
  model=$(curl -s "${target}/models" \
    | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
  if [ -z "$model" ]; then
    echo "ERROR: Could not auto-detect model name from ${target}/models." >&2
    echo "Pass the model name as the first argument, or ensure the server is running." >&2
    exit 1
  fi
  echo "Auto-detected model: ${model}"
else
  model=$1
fi

date_str=$(date '+%Y-%m-%d-%H-%M-%S')
output_path="${model//\//_}#${date_str}_guidellm_report.json"

export HF_TOKEN=$(cat ~/.cache/huggingface/token)

export GUIDELLM__NUM_SWEEP_PROFILES=0
export GUIDELLM__MAX_CONCURRENCY=${users}
export GUIDELLM__REQUEST_TIMEOUT=60

guidellm \
  --target "${target}" \
  --model "${model}" \
  --data-type emulated \
  --data "prompt_tokens=1500,prompt_tokens_variance=150,generated_tokens=250,generated_tokens_variance=20" \
  --output-path "${output_path}"
