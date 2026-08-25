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

# Copy the entrypoint script (a COPY, not a heredoc, so it works on the legacy
# docker builder as well as BuildKit — heredoc-in-RUN yields an empty file on
# the legacy builder, causing "exec format error" at container start).
COPY docker/vllm/entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
