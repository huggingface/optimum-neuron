#!/bin/bash

# Neuron compilation/runtime flags
export NEURON_CC_FLAGS="--model-type transformer --retry_failed_compilation"
export PYTHONPATH="/home/ubuntu/optimum-neuron-grpo:${PYTHONPATH}"
export NEURON_FUSE_SOFTMAX=1
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3
export MALLOC_ARENA_MAX=64

# Minimal test configuration
PROCESSES_PER_NODE=8
TP_DEGREE=4
MODEL_NAME="Qwen/Qwen3-0.6B"
OUTPUT_DIR="$(echo $MODEL_NAME | cut -d'/' -f2)-grpo-test"

# Force extraction/graph mode to avoid full training
export NEURON_EXTRACT_GRAPHS_ONLY=1

torchrun --nproc_per_node $PROCESSES_PER_NODE finetune_qwen3_grpo_test.py \
  --model_id $MODEL_NAME \
  --do_train \
  --max_steps 5 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --learning_rate 8e-4 \
  --bf16 \
  --tensor_parallel_size $TP_DEGREE \
  --logging_steps 1 \
  --output_dir $OUTPUT_DIR \
  --overwrite_output_dir

