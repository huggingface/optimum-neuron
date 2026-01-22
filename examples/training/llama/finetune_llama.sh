#!/bin/bash

# Flags for Neuron compilation
export NEURON_CC_FLAGS="--model-type transformer --retry_failed_compilation"
export NEURON_FUSE_SOFTMAX=1
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3 # Async Runtime
export MALLOC_ARENA_MAX=64 # Host OOM mitigation

# Detect instance type and set parameters accordingly

# Get the total number of Neuron cores using neuron-ls
NEURON_CORES=$(neuron-ls | awk '/^\| [0-9]+/ {total += $4} END {print total}')

# Set parameters based on detected instance type
if [ "$NEURON_CORES" -eq 32 ]; then
    # trn1.32xlarge instance
    PROCESSES_PER_NODE=32
    TP_DEGREE=8
    echo "Detected trn1.32xlarge instance (32 cores)"
elif [ "$NEURON_CORES" -eq 64 ]; then
    # trn2.48xlarge instance  
    PROCESSES_PER_NODE=64
    TP_DEGREE=4
    echo "Detected trn2.48xlarge instance (64 cores)"
elif [ "$NEURON_CORES" -eq 4 ]; then
    # trn2.3xlarge instance
    PROCESSES_PER_NODE=4
    TP_DEGREE=4
    echo "Detected trn2.3xlarge instance (4 cores)"
else
    echo "Warning: Unrecognized instance type with $NEURON_CORES cores. Using default trn1.32xlarge settings."
    PROCESSES_PER_NODE=32
    TP_DEGREE=8
fi

# Variables for training
NUM_EPOCHS=3
BS=1
GRADIENT_ACCUMULATION_STEPS=16
LOGGING_STEPS=1
MODEL_NAME="meta-llama/Llama-3.1-8B" # Change this to the desired model name
OUTPUT_DIR="$(echo $MODEL_NAME | cut -d'/' -f2)-finetuned"
DISTRIBUTED_ARGS="--nproc_per_node $PROCESSES_PER_NODE"
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Verify gated access and download locally
echo "Downloading model: $MODEL_NAME"
if ! hf download $MODEL_NAME; then
    echo "ERROR: Failed to download model $MODEL_NAME"
    echo "Please ensure you have:"
    echo "1. Accepted the license agreement at https://huggingface.co/$MODEL_NAME"
    echo "2. Logged in with 'hf auth login' using a valid token"
    echo "3. Have access permissions to the gated model"
    exit 1
fi
echo "Model download completed successfully"

if [ "$NEURON_EXTRACT_GRAPHS_ONLY" = "1" ]; then
    MAX_STEPS=5
else
    MAX_STEPS=-1
fi

torchrun --nproc_per_node $PROCESSES_PER_NODE finetune_llama.py \
  --model_id $MODEL_NAME \
  --num_train_epochs $NUM_EPOCHS \
  --do_train \
  --max_steps $MAX_STEPS \
  --per_device_train_batch_size $BS \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --learning_rate 1e-4 \
  --bf16 \
  --tensor_parallel_size $TP_DEGREE \
  --async_save \
  --warmup_steps 5 \
  --logging_steps $LOGGING_STEPS \
  --output_dir $OUTPUT_DIR \
  --overwrite_output_dir
