#!/bin/bash
echo "Step: validate-neuron-devices"
neuron-ls

echo "Step: install-ubuntu-packages"
sudo apt-get update
sudo apt-get -y upgrade --only-upgrade systemd
sudo apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    cmake \
    curl \
    emacs \
    git \
    git-lfs \
    jq \
    software-properties-common \
    unzip \
    vim \
    wget \
    zlib1g-dev \
    libsndfile1-dev \
    ffmpeg \
    g++

sudo apt-get update # Update to fix broken packages
sudo apt-get install -y --no-install-recommends python-is-python3 python3-pip python3-dev python3-virtualenv

echo "Step: install-neuron-python-packages"
sudo -H -u ubuntu bash -c 'pip install --upgrade --no-cache-dir \
    --extra-index-url https://pip.repos.neuron.amazonaws.com \
    "protobuf==3.20.2" \
    torch-neuronx=="1.13.1.1.13.0" \
    "transformers-neuronx==0.9.474" \
    "neuronx_distributed==0.6.0" \
    "tensorboard-plugin-neuronx" \
    "torchvision==0.14.*"'

sudo -H -u ubuntu bash -c 'echo 'export PATH="${HOME}/.local/bin:$PATH"' >> ${HOME}/.bashrc'
sudo -H -u ubuntu bash -c 'python -c "import torch_neuronx"'