#!/bin/bash
echo "Step: validate-neuron-devices"
neuron-ls

# Activate the neuron virtual environment
<<<<<<< HEAD
source /opt/aws_neuronx_venv_pytorch_2_6/bin/activate
=======
source /opt/aws_neuronx_venv_pytorch_2_7/bin/activate
>>>>>>> 3e99355966a976896c1950cc498beaa38e387b01

python -c 'import torch'
python -c 'import torch_neuronx'

echo "Installing Tensorboard Plugin for Neuron"
pip install --upgrade --no-cache-dir \
    "tensorboard-plugin-neuronx"
