#!/bin/bash

model=${1:-meta-llama/Llama-3.1-8B-Instruct}

date_str=$(date '+%Y-%m-%d-%H-%M-%S')
output_path="${model//\//_}#${date_str}_guidellm_report.json"

export HF_TOKEN=$(cat ~/.cache/huggingface/token)

export GUIDELLM__NUM_SWEEP_PROFILES=1
export GUIDELLM__MAX_CONCURRENCY=128
export GUIDELLM__REQUEST_TIMEOUT=60

guidellm benchmark \
  --target "http://localhost:8080/v1" \
  --model ${model} \
  --max-seconds 120 \
  --rate-type sweep \
  --data "prompt_tokens=1500,prompt_tokens_variance=150,output_tokens=250,outpu
t_tokens_variance=20" \
  --output-path ${output_path} \
