#!/bin/bash
echo "Step: validate-neuron-devices"
neuron-ls

# Activate the neuron virtual environment
source /opt/aws_neuron_venv_pytorch/bin/activate

python -c 'import torch'
python -c 'import torch_neuronx'

echo "Installing Tensorboard Plugin for Neuron"
pip install --upgrade --no-cache-dir \
    "tensorboard-plugin-neuronx"