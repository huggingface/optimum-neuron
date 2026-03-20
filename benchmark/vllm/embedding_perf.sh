#!/bin/bash
#
# Run an embedding throughput benchmark against a vLLM server.
#
# Usage:
#   ./embedding_perf.sh [concurrent_users] [total_requests] [prompt_tokens]
#
# Environment variables:
#   VLLM_TARGET  – base URL of the vLLM server (default: http://localhost:8080/v1)

set -e

target=${VLLM_TARGET:-http://localhost:8080/v1}
users=${1:-32}
total=${2:-1000}
prompt_tokens=${3:-1500}

# Auto-detect the model name from the vLLM server.
model=$(curl -s "${target}/models" \
  | python3 -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
if [ -z "$model" ]; then
  echo "ERROR: Could not auto-detect model from ${target}/models." >&2
  exit 1
fi
echo "Model: ${model}"
echo "Concurrent users: ${users}, Total requests: ${total}, Prompt tokens: ~${prompt_tokens}"

date_str=$(date '+%Y-%m-%d-%H-%M-%S')
output_path="embed_${model//\//_}#${date_str}.json"

python3 "$(dirname "$0")/embedding_perf.py" \
  --target "${target}" \
  --model "${model}" \
  --concurrent "${users}" \
  --total "${total}" \
  --prompt-tokens "${prompt_tokens}" \
  --output "${output_path}"
