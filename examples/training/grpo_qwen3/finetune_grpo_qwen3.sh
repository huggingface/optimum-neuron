#!/bin/bash

# ============================================================================
# GRPO Fine-tuning Script for Qwen3 on AWS Trainium
# ============================================================================
# This script demonstrates how to fine-tune a Qwen3 model using GRPO
# (Group Relative Policy Optimization) on AWS Trainium devices.
#
# Prerequisites:
# 1. vLLM server running (or use mock vLLM for development)
# 2. Neuron SDK installed
# 3. Multi-node Trainium instance (e.g., trn1.32xlarge)
#
# For mock vLLM (development/testing):
#   - Set USE_MOCK_VLLM=True in grpo_trainer.py
#
# For real vLLM:
#   - Start vLLM server with: trl vllm-serve --model MODEL_NAME
# ============================================================================

# Flags for Neuron compilation
export NEURON_CC_FLAGS="--model-type transformer --retry_failed_compilation"
export NEURON_FUSE_SOFTMAX=1
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3 # Async Runtime
export MALLOC_ARENA_MAX=64 # Host OOM mitigation

# Variables for training
PROCESSES_PER_NODE=2
NUM_EPOCHS=1  # GRPO typically needs fewer epochs than SFT
TP_DEGREE=1
BS=1
GRADIENT_ACCUMULATION_STEPS=1  # Smaller for GRPO due to generation overhead
LOGGING_STEPS=2
MODEL_NAME="Qwen/Qwen3-0.6B"  # Use smaller model for testing
# MODEL_NAME="michaelbenayoun/qwen3-tiny-4kv-heads-4layers-random"
OUTPUT_DIR="$(echo $MODEL_NAME | cut -d'/' -f2)-grpo-finetuned"
DISTRIBUTED_ARGS="--nproc_per_node $PROCESSES_PER_NODE"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# GRPO-specific variables
NUM_GENERATIONS=4  # Number of completions per prompt (G in paper)
MAX_PROMPT_LENGTH=512
MAX_COMPLETION_LENGTH=256
TEMPERATURE=0.8
STEPS_PER_GENERATION=4  # Generate every N steps to amortize generation cost

if [ "$NEURON_EXTRACT_GRAPHS_ONLY" = "1" ]; then
    MAX_STEPS=5
else
    MAX_STEPS=100  # Limit steps for testing
fi

# Note: Adjust these parameters based on your hardware and task
# - Increase num_generations for better exploration (but slower training)
# - Adjust temperature for sampling diversity
# - Tune epsilon and beta for GRPO algorithm sensitivity

torchrun $DISTRIBUTED_ARGS finetune_grpo_qwen3.py \
  --model_id $MODEL_NAME \
  --num_train_epochs $NUM_EPOCHS \
  --do_train \
  --max_steps $MAX_STEPS \
  --per_device_train_batch_size $BS \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --gradient_checkpointing \
  --learning_rate 5e-5 \
  --bf16 \
  --tensor_parallel_size $TP_DEGREE \
  --zero_1 false \
  --optimizer_use_master_weights false \
  --optimizer_use_fp32_grad_acc false \
  --async_save \
  --logging_steps $LOGGING_STEPS \
  --output_dir $OUTPUT_DIR \
  --lr_scheduler_type "cosine" \
  --overwrite_output_dir \
  --num_generations $NUM_GENERATIONS \
  --max_prompt_length $MAX_PROMPT_LENGTH \
  --max_completion_length $MAX_COMPLETION_LENGTH \
  --temperature $TEMPERATURE \
  --steps_per_generation $STEPS_PER_GENERATION

echo "================================"
echo "Training completed!"
echo "Model saved to: $OUTPUT_DIR"
echo "================================"
