#!/bin/bash
# Serve a neuron model, optionally with data parallelism.
set -e

env_file=${1:?Usage: serve.sh <path/to/serve-dpX-tpY.env>}

# Accept either a directory (error: must specify env file) or a direct path
if [ -d "$env_file" ]; then
    echo "ERROR: specify an env file, e.g. $env_file/serve-dp1-tp8.env" >&2
    exit 1
fi

source "${env_file}"

args=(
    -m "${MODEL_ID}"
    --batch_size "${BATCH_SIZE}"
    --sequence_length "${SEQUENCE_LENGTH}"
    --tensor_parallel_size "${TENSOR_PARALLEL_SIZE}"
)

if [ "${DATA_PARALLEL_SIZE:-1}" -gt 1 ]; then
    args+=(--data-parallel-size "${DATA_PARALLEL_SIZE}")
fi

optimum-cli neuron serve "${args[@]}"
