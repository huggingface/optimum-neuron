FROM 763104351884.dkr.ecr.us-east-1.amazonaws.com/huggingface-pytorch-training-neuronx:2.1.2-transformers4.43.2-neuronx-py310-sdk2.20.0-ubuntu20.04-v1.0

RUN apt-get update && apt-get install -y wget gnupg2 \
    && VERSION_CODENAME=$(cat /etc/os-release | grep VERSION_CODENAME | cut -d= -f2) \
    && echo "deb https://apt.repos.neuron.amazonaws.com ${VERSION_CODENAME} main" | tee /etc/apt/sources.list.d/neuron.list \
    && wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | apt-key add - \
    && apt-get update

RUN apt-get install aws-neuronx-dkms=2.* -y

# Upgrade neuron runtime
RUN apt-get install aws-neuronx-collectives=2.* -y
RUN apt-get install aws-neuronx-runtime-lib=2.* -y

# Install Neuron Tools
RUN apt-get install aws-neuronx-tools=2.* -y

# Set pip repository pointing to the Neuron repository
RUN python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com

# Install wget, awscli
RUN python -m pip install wget
RUN python -m pip install awscli

# Update Neuron Compiler and Framework
RUN python -m pip install --upgrade neuronx-cc==2.* torch-neuronx==2.1.* torchvision
RUN python -m pip install --upgrade neuronx-distributed neuronx-distributed-training


RUN mkdir workspace

RUN pip install git+https://github.com/huggingface/optimum-neuron.git

RUN git clone https://github.com/huggingface/optimum-neuron.git
RUN cp -r optimum-neuron/examples/language-modeling/requirements.txt workspace/requirements.txt
RUN cp -r optimum-neuron/examples/language-modeling/run_clm.py workspace/run_clm.py

WORKDIR /workspace

RUN pip install -r requirements.txt
