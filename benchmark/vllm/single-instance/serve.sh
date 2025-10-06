#!/bin/bash
# Serve a neuron model using a single instance
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
configuration=${1:-${SCRIPT_DIR}/qwen3-30B-A3B}  # e.g. qwen3-30B-A3B

source ${configuration}/.env
optimum-cli neuron serve \
        -m ${MODEL_ID} \
        --batch_size ${BATCH_SIZE} \
        --sequence_length ${SEQUENCE_LENGTH} \
        --tensor_parallel_size ${TENSOR_PARALLEL_SIZE}
