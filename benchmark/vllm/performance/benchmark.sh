#!/bin/bash

model=${1:-meta-llama/Meta-Llama-3.1-8B-Instruct}

date_str=$(date '+%Y-%m-%d-%H-%M-%S')
output_path="${model//\//_}#${date_str}_guidellm_report.json"

export HF_TOKEN=$(cat ~/.cache/huggingface/token)

export GUIDELLM__NUM_SWEEP_PROFILES=1
export GUIDELLM__MAX_CONCURRENCY=128
export GUIDELLM__REQUEST_TIMEOUT=60

guidellm \
  --target "http://localhost:8080/v1" \
  --model ${model} \
  --data-type emulated \
  --data "prompt_tokens=1500,prompt_tokens_variance=150,generated_tokens=250,generated_tokens_variance=20" \
  --output-path ${output_path} \
