PROCESSES_PER_NODE=32

NUM_EPOCHS=3
TP_DEGREE=8
BS=1
GRADIENT_ACCUMULATION_STEPS=8
LOGGING_STEPS=2

MODEL_NAME="Qwen/Qwen3-8B"
OUTPUT_DIR="$(echo $MODEL_NAME | cut -d'/' -f2)-finetuned"

if [ "$NEURON_EXTRACT_GRAPHS_ONLY" = "1" ]; then
    export WANDB_MODE=disabled
    MAX_STEPS=$((LOGGING_STEPS + 5))
else
    MAX_STEPS=-1
fi

DISTRIBUTED_ARGS="--nproc_per_node $PROCESSES_PER_NODE"

torchrun $DISTRIBUTED_ARGS notebooks/text-generation/scripts/sft_finetuning_qwen3.py \
  --model_id $MODEL_NAME \
  --num_train_epochs $NUM_EPOCHS \
  --do_train \
  --max_steps $MAX_STEPS \
  --per_device_train_batch_size $BS \
  --per_device_eval_batch_size $BS \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --gradient_checkpointing true \
  --bf16 true \
  --save_steps 20 \
  --tensor_parallel_size $TP_DEGREE \
  --logging_steps $LOGGING_STEPS \
  --save_total_limit -1 \
  --output_dir $OUTPUT_DIR \
  --lr_scheduler_type "cosine" \
  --overwrite_output_dir \
  --async_save
