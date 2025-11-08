#!/bin/bash

# Flags for Neuron compilation
export NEURON_CC_FLAGS="--model-type transformer --retry_failed_compilation"
export NEURON_FUSE_SOFTMAX=1
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3
export MALLOC_ARENA_MAX=64

# Variables for training - Tiny configuration for 0.6B model (for quick testing)
PROCESSES_PER_NODE=32
NUM_EPOCHS=1
TP_DEGREE=8
BS=1
GRADIENT_ACCUMULATION_STEPS=4
LOGGING_STEPS=1
MODEL_NAME="Qwen/Qwen3-0.6B"
OUTPUT_DIR="$(echo $MODEL_NAME | cut -d'/' -f2)-grpo-gsm8k-tiny"
MAX_SAMPLES=100  # Very small dataset for quick testing
DISTRIBUTED_ARGS="--nproc_per_node $PROCESSES_PER_NODE"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Set PYTHONPATH to include the optimum-neuron-grpo directory
export PYTHONPATH="/home/ubuntu/optimum-neuron-grpo:${PYTHONPATH}"

if [ "$NEURON_EXTRACT_GRAPHS_ONLY" = "1" ]; then
    MAX_STEPS=5
else
    MAX_STEPS=-1
fi

torchrun --nproc_per_node $PROCESSES_PER_NODE grpo_qwen3_gsm8k.py \
  --model_id $MODEL_NAME \
  --model_size tiny \
  --num_train_epochs $NUM_EPOCHS \
  --do_train \
  --max_steps $MAX_STEPS \
  --max_samples $MAX_SAMPLES \
  --per_device_train_batch_size $BS \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --learning_rate 5e-5 \
  --bf16 \
  --tensor_parallel_size $TP_DEGREE \
  --zero_1 \
  --async_save \
  --logging_steps $LOGGING_STEPS \
  --output_dir $OUTPUT_DIR \
  --lr_scheduler_type "cosine" \
  --overwrite_output_dir \
  --save_steps 10 \
  --save_total_limit 1

