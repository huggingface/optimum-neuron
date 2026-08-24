FROM ubuntu:24.04 AS base

# Install system prerequisites
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
    python3-setuptools \
    python-is-python3 \
    ca-certificates \
    curl \
    gnupg2 \
    wget \
    libexpat1 \
    libpython3-dev \
    libarchive13 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Install uv at a specific version on a given path
RUN curl -LsSf https://astral.sh/uv/0.9.27/install.sh | XDG_BIN_HOME=/usr/local/bin sh

# Provision a standalone Python 3.12 interpreter via uv.
# Use 3.12, not 3.11: neuronx-cc pins numpy<2 for python_full_version < '3.12', which
# conflicts with vllm's numpy>=2 requirement.
RUN uv venv --python 3.12 /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"
# torch_xla's compiled extension dynamically links against libpython, which uv's
# standalone interpreter ships but doesn't register with the dynamic linker.
RUN dirname "$(find /root/.local/share/uv/python -name 'libpython3.12.so.1.0')" \
    > /etc/ld.so.conf.d/uv-python.conf \
    && ldconfig

# Setup neuronx repository
RUN echo "deb https://apt.repos.neuron.amazonaws.com noble main" > /etc/apt/sources.list.d/neuron.list
RUN wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | apt-key add -

# Install neuronx packages
RUN apt-get update -y \
    && apt-get install -y --no-install-recommends \
    aws-neuronx-dkms=2.29.0.0 \
    aws-neuronx-collectives=2.33.10.0-068180c7a \
    aws-neuronx-runtime-lib=2.33.10.0-3dcef56f0 \
    aws-neuronx-tools=2.31.13.0-a9e473f33 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

ENV PATH="/opt/bin/:/opt/aws/neuron/bin:${PATH}"

# Install optimum-neuron
RUN mkdir optimum-neuron
COPY optimum optimum-neuron/optimum
COPY pyproject.toml optimum-neuron/pyproject.toml
RUN ls optimum-neuron
RUN cd optimum-neuron && uv pip install .[neuronx,vllm]

# HF base env
ENV HUGGINGFACE_HUB_CACHE=/tmp \
    HF_XET_HIGH_PERFORMANCE=1 \
    HF_HUB_USER_AGENT_ORIGIN=aws:sagemaker:neuron:inference:vllm-optimum-neuron

# Create the entrypoint script
# We use a 'heredoc' pattern to keep the Dockerfile self-contained
# All content within the 'EOF' marker is written to the entrypoint.sh file
######## Start of entry-point script ########
RUN cat <<'EOF' > entrypoint.sh
#!/bin/bash

# Save parameters, they will be passed over to the optimum-cli serve command
export ARGS_CLI="${@}"

# Define the prefix for environment variables to look for
PREFIX="SM_ON_"
ARG_PREFIX="--"

# Initialize an array for storing the arguments
# port 8080 required by sagemaker, https://docs.aws.amazon.com/sagemaker/latest/dg/your-algorithms-inference-code.html#your-algorithms-inference-code-container-response
ARGS=(--port 8080)

# Loop through all environment variables
while IFS='=' read -r key value; do
    # Remove the prefix from the key, convert to lowercase
    arg_name=$(echo "${key#"${PREFIX}"}" | tr '[:upper:]' '[:lower:]')

    # Add the argument name and value to the ARGS array
    ARGS+=("${ARG_PREFIX}${arg_name}")
    if [ -n "$value" ]; then
        ARGS+=("$value")
    fi
done < <(env | grep "^${PREFIX}")

# Pass the collected arguments to the optimum-cli serve command
exec optimum-cli neuron serve "${ARGS[@]}" ${ARGS_CLI}
EOF
######## End of entry-point script ########

RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
