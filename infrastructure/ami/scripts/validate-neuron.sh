#!/bin/bash
echo "Step: validate-neuron-devices"
neuron-ls

# Activate the neuron virtual environment
source /opt/aws_neuronx_venv_pytorch_2_8/bin/activate

python -c 'import torch'
python -c 'import torch_neuronx'

echo "Installing Tensorboard Plugin for Neuron"
pip install --upgrade --no-cache-dir \
    "tensorboard-plugin-neuronx"
