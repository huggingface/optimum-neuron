#!/bin/bash
set -ex

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
sudo sysctl -w net.ipv4.ip_local_reserved_ports=44000,48620

export FI_EFA_USE_DEVICE_RDMA=1
export FI_PROVIDER=efa
export FI_EFA_FORK_SAFE=1

export NEURON_FUSE_SOFTMAX=1
export NEURON_RT_ASYNC_EXEC_MAX_INFLIGHT_REQUESTS=7
export MALLOC_ARENA_MAX=128
export XLA_DOWNCAST_BF16=1
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

GBS=256
NUM_EPOCHS=1
TP_DEGREE=8
PP_DEGREE=1
DP=$(($PROCESSES_PER_NODE * $WORLD_SIZE / $TP_DEGREE / $PP_DEGREE))
BS=$(($GBS / $DP))
BLOCK_SIZE=4096
# BLOCK_SIZE=16
# MODEL_NAME="michaelbenayoun/llama-2-tiny-16layers-32kv-heads-random"
# MODEL_NAME="michaelbenayoun/llama-2-tiny-4layers-random"
# MODEL_NAME="NousResearch/Llama-2-70b-chat-hf"
# MODEL_NAME="michaelbenayoun/llama-2-tiny-4kv-heads-16layers-random"
# MODEL_NAME="NousResearch/Llama-2-7b-chat-hf"
# MODEL_NAME="TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T"
MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.2"
OUTPUT_DIR=output-$SLURM_JOB_ID

if [ "$NEURON_EXTRACT_GRAPHS_ONLY" = "1" ]; then
    MAX_STEPS=10
    tb_dir="~/tensorboard/llama70B_compile"
else
    MAX_STEPS=-1
    tb_dir="~/tensorboard/llama70B_32nodes_${JOB_ID}"
    mkdir -p $tb_dir
fi

# --dataset_name wikitext \
# --dataset_config_name wikitext-2-raw-v1 \
NXD_LOG_LEVEL=info torchrun --tee=1 $DISTRIBUTED_ARGS test_llm_finetuning.py \
  --model_name_or_path $MODEL_NAME \
  --dataset_name Open-Orca/OpenOrca \
  --do_train \
  --do_eval false \
  --num_train_epochs $NUM_EPOCHS \
  --per_device_train_batch_size $BS \
  --per_device_eval_batch_size $BS \
  --max_steps $MAX_STEPS \
  --bf16 \
  --zero_1 \
  --gradient_checkpointing \
  --block_size $BLOCK_SIZE \
  --tensor_parallel_size $TP_DEGREE \
  --pipeline_parallel_size $PP_DEGREE \
  --logging_steps 1 \
  --output_dir $OUTPUT_DIR \
  --overwrite_output_dir

