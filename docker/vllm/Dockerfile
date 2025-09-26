FROM ubuntu:22.04 AS base

RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-setuptools \
    python-is-python3 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
RUN pip3 --no-cache-dir install --upgrade pip

# Install system prerequisites
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
    gnupg2 \
    wget \
    libexpat1 \
    libpython3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN echo "deb https://apt.repos.neuron.amazonaws.com jammy main" > /etc/apt/sources.list.d/neuron.list
RUN wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | apt-key add -

# Install neuronx packages
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
    aws-neuronx-dkms=2.24.7.0 \
    aws-neuronx-collectives=2.28.27.0-bc30ece58 \
    aws-neuronx-runtime-lib=2.28.23.0-dd5879008 \
    aws-neuronx-tools=2.26.14.0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

ENV PATH="/opt/bin/:/opt/aws/neuron/bin:${PATH}"

# Install HuggingFace packages
RUN pip3 install \
    hf_transfer huggingface_hub

# Install manually torch CPU version to avoid pulling CUDA
RUN pip3 install \
    torch==2.8.0 \
    torchvision==0.23.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Install optimum-neuron
RUN mkdir optimum-neuron
COPY optimum optimum-neuron/optimum
COPY pyproject.toml optimum-neuron/pyproject.toml
RUN ls optimum-neuron
RUN cd optimum-neuron && pip3 install .[neuronx,vllm] \
    --extra-index-url=https://pip.repos.neuron.amazonaws.com

# HF base env
ENV HUGGINGFACE_HUB_CACHE=/tmp \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    HF_HUB_USER_AGENT_ORIGIN=aws:sagemaker:neuron:inference:vllm-optimum-neuron

# Create the entrypoint script
# We use a 'heredoc' pattern to keep the Dockerfile self-contained
# All content within the 'EOF' marker is written to the entrypoint.sh file
######## Start of entry-point script ########
RUN cat <<'EOF' > entrypoint.sh
#!/bin/bash

# Define the prefix for environment variables to look for
PREFIX="SM_VLLM_"
ARG_PREFIX="--"

# Initialize an array for storing the arguments
# port 8080 required by sagemaker, https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html#your-algorithms-inference-code-container-response
ARGS=(--port 8080)

# Loop through all environment variables
while IFS='=' read -r key value; do
    # Remove the prefix from the key, convert to lowercase, and replace underscores with dashes
    arg_name=$(echo "${key#"${PREFIX}"}" | tr '[:upper:]' '[:lower:]' | tr '_' '-')

    # Add the argument name and value to the ARGS array
    ARGS+=("${ARG_PREFIX}${arg_name}")
    if [ -n "$value" ]; then
        ARGS+=("$value")
    fi
done < <(env | grep "^${PREFIX}")

# Pass the collected arguments to the main entrypoint
exec python3 -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
EOF
######## End of entry-point script ########

RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
