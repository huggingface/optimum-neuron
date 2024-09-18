#!/bin/bash

model=${1:-NousResearch/Llama-2-7b-chat-hf}
vu=${2:-1}

export HUGGINGFACE_API_BASE=http://127.0.0.1:8080
export HUGGINGFACE_API_KEY=EMPTY

benchmark_script=${LLMPerf}/token_benchmark_ray.py

if ! test -f ${benchmark_script}; then
  echo "LLMPerf script not found, please export LLMPerf=<path-to-llmperf>."
fi

max_requests=$(expr ${vu} \* 8 )
date_str=$(date '+%Y-%m-%d-%H-%M-%S')

python ${benchmark_script} \
       --model "huggingface/${model}" \
       --mean-input-tokens 1500 \
       --stddev-input-tokens 150 \
       --mean-output-tokens 245 \
       --stddev-output-tokens 20 \
       --max-num-completed-requests ${max_requests} \
       --timeout 7200 \
       --num-concurrent-requests ${vu} \
       --results-dir "tgi_bench_results/${date_str}" \
       --llm-api "litellm" \
       --additional-sampling-params '{}'
