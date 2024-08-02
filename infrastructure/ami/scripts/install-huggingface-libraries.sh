#!/bin/bash

# Activate the neuron virtual environment
source /opt/aws_neuron_venv_pytorch/bin/activate

echo "Step: install-hugging-face-libraries"

echo "TRANSFORMERS_VERSION: $TRANSFORMERS_VERSION"
echo "OPTIMUM_VERSION: $OPTIMUM_VERSION"

pip install --upgrade --no-cache-dir \
    "evaluate==0.4.1" \
    "requests==2.31.0" \
    "notebook==7.0.6" \
    "markupsafe==2.1.1" \
    "jinja2==3.1.2" \
    "attrs==23.1.0"

# Temporary fix for the issue: https://github.com/huggingface/optimum-neuron/issues/142
pip install -U optimum
echo 'export PATH="${HOME}/.local/bin:$PATH"' >> "${HOME}/.bashrc"

echo "Step: install-and-copy-optimum-neuron-examples"
git clone -b $OPTIMUM_VERSION https://github.com/huggingface/optimum-neuron.git

cd optimum-neuron
pip install ".[neuronx, diffusers, sentence-transformers]"
cd ..

mkdir /home/ubuntu/huggingface-neuron-samples/ /home/ubuntu/huggingface-neuron-notebooks/ /home/ubuntu/aws-examples/
mv optimum-neuron/examples/* /home/ubuntu/huggingface-neuron-samples/
mv optimum-neuron/notebooks/* /home/ubuntu/huggingface-neuron-notebooks/
mv optimum-neuron/aws-examples/* /home/ubuntu/aws-examples/
rm -rf optimum-neuron
chmod -R 777 /home/ubuntu/huggingface-neuron-samples /home/ubuntu/huggingface-neuron-notebooks

echo "Step: validate-imports-of-huggingface-libraries"
bash -c 'python -c "import transformers;import datasets;import accelerate;import evaluate;import tensorboard; import torch;from optimum.neuron import pipeline"'