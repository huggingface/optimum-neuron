#!/bin/bash
set -ex

# Configuration
MODEL_ID="bert-base-uncased"
LEARNING_RATE=5e-5
BATCH_SIZE=8
EPOCHS=3
MAX_LENGTH=128
OUTPUT_DIR="bert-emotion-model"

# Number of processes per node (2 Neuron cores on trn1.2xlarge)
PROCESSES_PER_NODE=2

# Launch training with torchrun for distributed training
torchrun --nproc_per_node $PROCESSES_PER_NODE fine_tune_bert.py \
  --model_id $MODEL_ID \
  --learning_rate $LEARNING_RATE \
  --batch_size $BATCH_SIZE \
  --epochs $EPOCHS \
  --max_length $MAX_LENGTH \
  --output_dir $OUTPUT_DIR