#!/bin/bash

echo "Step: install-hugging-face-libraries"

echo "Activating the virtual-env"
source /opt/aws_neuron_venv_pytorch/bin/activate

pip install --upgrade --no-cache-dir \
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
    "attrs==23.1.0"

echo 'export PATH="${HOME}/.local/bin:$PATH"' >> "${HOME}/.bashrc"


echo "Step: copy-optimum-neuron-examples"
git clone -b "v0.0.17" https://github.com/huggingface/optimum-neuron.git
mkdir /home/ubuntu/huggingface-neuron-samples/
mv optimum-neuron/examples/* /home/ubuntu/huggingface-neuron-samples/
rm -rf optimum-neuron
chown -R ubuntu:ubuntu /home/ubuntu/huggingface-neuron-samples


echo "Step: clean-apt-cache"
sudo apt autoremove -y
sudo apt clean -y

echo "Step: validate-imports-of-huggingface-libraries"
python -c 'import transformers;import datasets;import accelerate;import evaluate;import tensorboard;'