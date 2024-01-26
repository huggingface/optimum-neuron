#!/bin/bash

echo "Step: install-hugging-face-libraries"

sudo -H -u ubuntu bash -c 'pip install --upgrade --no-cache-dir \
    "transformers[sklearn,sentencepiece,vision]==4.36.2" \
    "optimum-neuron==0.0.17" \
    "datasets==2.16.1" \
    "accelerate==0.23.0" \
    "diffusers==0.25.0" \
    "evaluate==0.4.1" \
    "requests==2.31.0" \
    "notebook==7.0.6" \
    "markupsafe==2.1.1" \
    "jinja2==3.1.2" \
    "attrs==23.1.0"'

echo 'export PATH="${HOME}/.local/bin:$PATH"' >> "${HOME}/.bashrc"


echo "Step: copy-optimum-neuron-examples"
git clone -b "v0.0.17" https://github.com/huggingface/optimum-neuron.git
mkdir /home/ubuntu/huggingface-neuron-samples/ /home/ubuntu/huggingface-neuron-notebooks/
mv optimum-neuron/examples/* /home/ubuntu/huggingface-neuron-samples/
mv optimum-neuron/notebooks/* /home/ubuntu/huggingface-neuron-notebooks/
rm -rf optimum-neuron
chown -R ubuntu:ubuntu /home/ubuntu/huggingface-neuron-samples /home/ubuntu/huggingface-neuron-notebooks


echo "Step: clean-apt-cache"
sudo apt autoremove -y
sudo apt clean -y

echo "Step: validate-imports-of-huggingface-libraries"
sudo -H -u ubuntu bash -c 'python -c "import transformers;import datasets;import accelerate;import evaluate;import tensorboard; import torch;"'