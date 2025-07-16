#!/bin/bash
set -ex

export NEURON_FUSE_SOFTMAX=1
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3
export MALLOC_ARENA_MAX=64 # limit the CPU allocation to avoid potential crashes
export NEURON_CC_FLAGS="--model-type=transformer --distribution-strategy=llm-training --enable-saturate-infinity --cache_dir=/home/ubuntu/cache_dir_neuron/"

PROCESSES_PER_NODE=8

TP_DEGREE=2
PP_DEGREE=1
BS=1
GRADIENT_ACCUMULATION_STEPS=8
LOGGING_STEPS=1
MODEL_NAME="meta-llama/Meta-Llama-3-8B"
OUTPUT_DIR=dolly_llama_output

if [ "$NEURON_EXTRACT_GRAPHS_ONLY" = "1" ]; then
    MAX_STEPS=10
    NUM_EPOCHS=1
else
    MAX_STEPS=-1
    NUM_EPOCHS=3
fi


torchrun --nproc_per_node $PROCESSES_PER_NODE docs/source/training_tutorials/sft_lora_finetune_llm.py \
  --model_id $MODEL_NAME \
  --num_train_epochs $NUM_EPOCHS \
  --do_train \
  --learning_rate 5e-5 \
  --warmup_ratio 0.03 \
  --max_steps $MAX_STEPS \
  --per_device_train_batch_size $BS \
  --per_device_eval_batch_size $BS \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --gradient_checkpointing true \
  --bf16 \
  --tensor_parallel_size $TP_DEGREE \
  --pipeline_parallel_size $PP_DEGREE \
  --logging_steps $LOGGING_STEPS \
  --save_total_limit 1 \
  --output_dir $OUTPUT_DIR \
  --lr_scheduler_type "constant" \
  --overwrite_output_dir
