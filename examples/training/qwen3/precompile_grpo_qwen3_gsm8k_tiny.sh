#!/bin/bash

set -euo pipefail

# Precompile graphs for the tiny GRPO run with fixed shapes.
# This matches the training topology (TP=8, BS=1, total seq=2048) to ensure cache reuse.

# -----------------------------
# Neuron compiler/runtime flags
# -----------------------------
export NEURON_EXTRACT_GRAPHS_ONLY=1
export NEURON_PARALLEL_COMPILE=1
export NEURON_LOG_LEVEL=INFO

# Keep flags stable across precompile and training
export NEURON_CC_FLAGS="--model-type transformer --opt-level 2 --retry_failed_compilation --verbose"
export NEURON_FUSE_SOFTMAX=1
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3
export MALLOC_ARENA_MAX=64

# -----------------------------
# Run configuration (match training)
# -----------------------------
PROCESSES_PER_NODE=8       # Precompile with TP-only to reduce contention; TP must match
NUM_EPOCHS=1
TP_DEGREE=8
BS=1
GRADIENT_ACCUMULATION_STEPS=4
LOGGING_STEPS=1
MODEL_NAME="Qwen/Qwen3-0.6B"
OUTPUT_DIR="$(echo $MODEL_NAME | cut -d'/' -f2)-grpo-gsm8k-tiny-precompile"
MAX_SAMPLES=64

# Short run to trigger all graphs
MAX_STEPS=5

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Ensure PYTHONPATH includes project root
export PYTHONPATH="/home/ubuntu/optimum-neuron-grpo:${PYTHONPATH:-}"

cd "$SCRIPT_DIR"

echo "[Precompile] Starting graph extraction with PPN=$PROCESSES_PER_NODE TP=$TP_DEGREE BS=$BS MAX_STEPS=$MAX_STEPS"

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
  --save_steps 0 \
  --save_total_limit 0

echo "[Precompile] Completed. Unset NEURON_EXTRACT_GRAPHS_ONLY before real training."



