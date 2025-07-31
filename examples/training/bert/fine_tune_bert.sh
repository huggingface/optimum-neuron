#!/bin/bash
set -ex

# Configuration
MODEL_ID="bert-base-uncased"
LEARNING_RATE=5e-5
PER_DEVICE_TRAIN_BATCH_SIZE=8
PER_DEVICE_EVAL_BATCH_SIZE=8
EPOCHS=3
TRAIN_MAX_LENGTH=128
OUTPUT_DIR="bert-base-uncased-finetuned"

# Number of processes per node (2 Neuron cores on trn1.2xlarge)
PROCESSES_PER_NODE=2

# Launch training with torchrun for distributed training
torchrun --nproc_per_node $PROCESSES_PER_NODE fine_tune_bert.py \
  --model_id $MODEL_ID \
  --learning_rate $LEARNING_RATE \
  --per_device_train_batch_size $PER_DEVICE_TRAIN_BATCH_SIZE \
  --per_device_eval_batch_size $PER_DEVICE_EVAL_BATCH_SIZE \
  --epochs $EPOCHS \
  --train_max_length $TRAIN_MAX_LENGTH \
  --output_dir $OUTPUT_DIR