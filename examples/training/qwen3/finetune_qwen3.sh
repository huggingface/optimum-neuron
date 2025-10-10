#!/bin/bash

# Flags for Neuron compilation
export NEURON_CC_FLAGS="--model-type transformer --retry_failed_compilation"
export NEURON_FUSE_SOFTMAX=1
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=7 # Async Runtime
export MALLOC_ARENA_MAX=128 # Host OOM mitigation

export XLA_DISABLE_FUNCTIONALIZATION=0

# Variables for training
PROCESSES_PER_NODE=8
NUM_EPOCHS=3
TP_DEGREE=2
PP_DEGREE=4
BS=1
GRADIENT_ACCUMULATION_STEPS=1
LOGGING_STEPS=2
MODEL_NAME="Qwen/Qwen3-8B" # Change this to the desired model name
# MODEL_NAME="meta-llama/Llama-3.1-8B-Instruct" # Change this to the desired model name
# MODEL_NAME="Qwen/Qwen3-0.6B" # Change this to the desired model name
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
  --learning_rate 5e-5 \
  --max_grad_norm 0.5 \
  --pipeline_parallel_num_microbatches 1 \
  --bf16 \
  --gradient_checkpointing false \
  --tensor_parallel_size $TP_DEGREE \
  --pipeline_parallel_size $PP_DEGREE \
  --zero_1 false \
  --async_save \
  --logging_steps $LOGGING_STEPS \
  --output_dir $OUTPUT_DIR \
  --lr_scheduler_type "cosine" \
  --overwrite_output_dir
