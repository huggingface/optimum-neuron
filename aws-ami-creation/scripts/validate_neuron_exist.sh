#!/bin/bash

echo "Step: validate-neuron-devices"
source /opt/aws_neuron_venv_pytorch/bin/activate
python -c 'import torch'
python -c 'import torch_neuronx'
neuron-ls
echo "Installing Tensorboard Plugin for Neuron"
pip install --upgrade --no-cache-dir \
    "tensorboard-plugin-neuronx"