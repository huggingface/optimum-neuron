#!/bin/bash
set -ex

# In PT2.1, functionalization is needed to close 3% convergence gap compared to PT1.13 for ZeRO1
export XLA_DISABLE_FUNCTIONALIZATION=1

export NEURON_FUSE_SOFTMAX=1
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3
# Limit memory allocation to prevent crashes
export MALLOC_ARENA_MAX=64
export NEURON_CC_FLAGS="--model-type=transformer --distribution-strategy=llm-training --enable-saturate-infinity --cache_dir=/home/ubuntu/cache_dir_neuron/"

# Distributed configs
PROCESSES_PER_NODE=32
WORLD_SIZE=1
DISTRIBUTED_ARGS="--nproc_per_node $PROCESSES_PER_NODE"
LOG_PATH=logs

# Create the log path
mkdir -p $LOG_PATH
echo $DISTRIBUTED_ARGS

# Parallelism configuration
GBS=512
NUM_EPOCHS=10
TP_DEGREE=8
PP_DEGREE=1
DP=$(($PROCESSES_PER_NODE * $WORLD_SIZE / $TP_DEGREE / $PP_DEGREE))
BS=1
GRADIENT_ACCUMULATION_STEPS=1
BLOCK_SIZE=2048
LOGGING_STEPS=1
MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.3"
OUTPUT_DIR="mistral_trained"

MAX_STEPS=-1

# Our script will first look in the working directory for a dataset matching the name, or download it from the Hugging Face hub
DATASET_NAME="dataset_formatted"

XLA_USE_BF16=1 torchrun $DISTRIBUTED_ARGS examples/run_clm.py \
    --model_name_or_path $MODEL_NAME \
    --num_train_epochs $NUM_EPOCHS \
    --dataset_name $DATASET_NAME \
    --do_train \
    --learning_rate 8e-6 \
    --warmup_steps 30 \
    --max_steps $MAX_STEPS \
    --per_device_train_batch_size $BS \
    --per_device_eval_batch_size $BS \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --gradient_checkpointing \
    --block_size $BLOCK_SIZE \
    --bf16 \
    --zero_1 false \
    --tensor_parallel_size $TP_DEGREE \
    --pipeline_parallel_size $PP_DEGREE \
    --logging_steps $LOGGING_STEPS \
   --save_total_limit 1 \
   --output_dir $OUTPUT_DIR \
   --overwrite_output_dir
