#!/bin/bash

# Flags for Neuron compilation
export NEURON_CC_FLAGS="--model-type transformer --retry_failed_compilation"
export NEURON_FUSE_SOFTMAX=1
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3 # Async Runtime
export MALLOC_ARENA_MAX=64 # Host OOM mitigation

# Variables for training
PROCESSES_PER_NODE=32
NUM_EPOCHS=3
TP_DEGREE=8
BS=1
GRADIENT_ACCUMULATION_STEPS=8
LOGGING_STEPS=2
MODEL_NAME="Qwen/Qwen3-8B" # Change this to the desired model name
OUTPUT_DIR="$(echo $MODEL_NAME | cut -d'/' -f2)-finetuned"
DISTRIBUTED_ARGS="--nproc_per_node $PROCESSES_PER_NODE"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

if [ "$NEURON_EXTRACT_GRAPHS_ONLY" = "1" ]; then
    MAX_STEPS=5
else
    MAX_STEPS=-1
fi

torchrun --nproc_per_node $PROCESSES_PER_NODE finetune_qwen3.py \
  --model_id $MODEL_NAME \
  --num_train_epochs $NUM_EPOCHS \
  --do_train \
  --max_steps $MAX_STEPS \
  --per_device_train_batch_size $BS \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --learning_rate 8e-4 \
  --bf16 \
  --tensor_parallel_size $TP_DEGREE \
  --zero_1 \
  --async_save \
  --logging_steps $LOGGING_STEPS \
  --output_dir $OUTPUT_DIR \
  --lr_scheduler_type "cosine" \
  --overwrite_output_dir
