#!/bin/bash
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
sudo sysctl -w net.ipv4.ip_local_reserved_ports=44000,48620

export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1

export NEURON_FUSE_SOFTMAX=1
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=3
export MALLOC_ARENA_MAX=64
export NEURON_CC_FLAGS="--model-type=transformer --distribution-strategy=llm-training --enable-saturate-infinity --cache_dir=/home/ubuntu/cache_dir_neuron/"

PROCESSES_PER_NODE=32
WORLD_SIZE=1
NODEID=0
HOSTNAME=`hostname`
if [ -v SLURM_NTASKS ]; then
    # SLURM runs
    IPS=""
    for h in $(scontrol show hostname); do
        IPS="$IPS $(nslookup $h  | awk '/^Address: / { print $2 }')";
    done
    HOSTS=(${IPS//\ / })
    NODEID=$SLURM_NODEID
    NTASKS=$SLURM_NTASKS
    WORLD_SIZE=$SLURM_NTASKS
    JOB_ID=$SLURM_JOB_ID
    export NEMO_EXPM_VERSION=$SLURM_JOB_ID
    export EXPLICIT_LOGDIR=null
    LOG_PATH=logs/$SLURM_JOB_ID/$NODEID
    
    # MASTER_ADDR=${HOSTS[0]}
    MASTER_ADDR=(`scontrol show hostnames $SLURM_JOB_NODELIST`)
    MASTER_PORT=44000
    DISTRIBUTED_ARGS="--nproc_per_node $PROCESSES_PER_NODE --nnodes $NTASKS --node_rank $NODEID --master_addr $MASTER_ADDR --master_port $MASTER_PORT"
else
    DISTRIBUTED_ARGS="--nproc_per_node $PROCESSES_PER_NODE"
    LOG_PATH=logs
fi
mkdir -p $LOG_PATH
echo "Nodeinfo NODEID $NODEID hostname $HOSTNAME"
echo $DISTRIBUTED_ARGS

GBS=512
NUM_EPOCHS=3
TP_DEGREE=8
PP_DEGREE=1
DP=$(($PROCESSES_PER_NODE * $WORLD_SIZE / $TP_DEGREE / $PP_DEGREE))
BS=1
# GRADIENT_ACCUMULATION_STEPS=$(($GBS / $DP))
GRADIENT_ACCUMULATION_STEPS=8
BLOCK_SIZE=512
LOGGING_STEPS=1
# MODEL_NAME="meta-llama/Meta-Llama-3-8B"
# MODEL_NAME="codellama/CodeLlama-7b-hf"
# MODEL_NAME="michaelbenayoun/llama-2-tiny-4kv-heads-4layers-random"
# MODEL_NAME="meta-llama/Llama-3.2-1B-Instruct"
# MODEL_NAME="ibm-granite/granite-3.2-2b-instruct"
# MODEL_NAME="hf-internal-testing/tiny-random-GraniteForCausalLM"
# MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
MODEL_NAME="Qwen/Qwen3-0.6B"
OUTPUT_DIR=output-$SLURM_JOB_ID

WANDB_NAME=${1:-mistral_finetuning}

if [ "$NEURON_EXTRACT_GRAPHS_ONLY" = "1" ]; then
    MAX_STEPS=$((LOGGING_STEPS + 5))
    WANDB_MODE=disabled
else
    MAX_STEPS=-1
    WANDB_MODE=online
fi

#   --dataset_name wikicorpus \
#   --dataset_config_name raw_en \
#   --dataset_name wikitext \
#   --dataset_config_name wikitext-2-raw-v1 \

WANDB_MODE=$WANDB_MODE WANDB_NAME=$WANDB_NAME torchrun $DISTRIBUTED_ARGS test_custom_modeling.py \
  --model_id $MODEL_NAME \
  --num_train_epochs $NUM_EPOCHS \
  --do_train \
  --learning_rate 5e-4 \
  --warmup_ratio 0.03 \
  --max_steps $MAX_STEPS \
  --per_device_train_batch_size $BS \
  --per_device_eval_batch_size $BS \
  --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
  --gradient_checkpointing true \
  --bf16 true \
  --zero_1 false \
  --save_steps 100 \
  --tensor_parallel_size $TP_DEGREE \
  --pipeline_parallel_size $PP_DEGREE \
  --logging_steps $LOGGING_STEPS \
  --save_total_limit -1 \
  --output_dir $OUTPUT_DIR \
  --lr_scheduler_type "cosine" \
  --overwrite_output_dir

