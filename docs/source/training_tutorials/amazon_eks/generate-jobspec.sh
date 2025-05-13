#!/bin/bash

# AWS and Registry Configuration
export AWS_REGION=$(aws ec2 describe-availability-zones --output text --query 'AvailabilityZones[0].[RegionName]')
export ACCOUNT=$(aws sts get-caller-identity --query Account --output text)
export REGISTRY=${ACCOUNT}.dkr.ecr.${AWS_REGION}.amazonaws.com/
export IMAGE=optimum-neuron-llama-pretraining
export TAG=:latest
export IMAGE_URI=${REGISTRY}${IMAGE}${TAG}

# Job Configuration
export JOB_NAME=llama-training-eks
export NUM_NODES=1
export INSTANCE_TYPE=ml.trn1.32xlarge
export EFA_PER_NODE=8
export NEURON_PER_NODE=16
export FI_PROVIDER=efa

# Storage Configuration
export FSX_CLAIM=fsx-claim

# Model and Dataset Configuration
export HF_ACCESS_TOKEN="<your_HF_token_here>"
export TOKENIZED_DATA_PATH=/fsx/cached_model_dir
export DATASET_NAME=wikicorpus
export DATASET_CONFIG_NAME=raw_en
export HF_MODEL_NAME=meta-llama/Llama-3.2-1B

# Training Configuration
export NEURON_CACHE_DIR=/fsx/neuron_cache
export CHECKPOINT_DIR=/fsx/output
export MAX_STEPS=1000
export BATCH_SIZE=1

# Generate the final yaml file from template
cat llama3_train.yaml-template | envsubst > llama_train.yaml
echo "Generated job spec: llama_train.yaml"
