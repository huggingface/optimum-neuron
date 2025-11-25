#!/bin/bash

model=${1:-meta-llama/Meta-Llama-3.1-8B-Instruct}
batch_size=${2:-32}
tasks=${3:-gsm8k}

lm_eval --model local-completions \
        --tasks ${tasks} \
        --model_args model=${model},base_url=http://127.0.0.1:8080/v1/completions,tokenized_requests=False \
        --batch_size ${batch_size}
