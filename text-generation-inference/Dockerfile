# Fetch and extract the TGI sources (TGI_VERSION is mandatory)
FROM alpine AS tgi
ARG TGI_VERSION
RUN test -n ${TGI_VERSION:?}
RUN mkdir -p /tgi
ADD https://github.com/huggingface/text-generation-inference/archive/refs/tags/v${TGI_VERSION}.tar.gz /tgi/sources.tar.gz
RUN tar -C /tgi -xf /tgi/sources.tar.gz --strip-components=1

# Build cargo components (adapted from TGI original Dockerfile)
# Note that the build image is aligned on the same Linux version as the base image (Debian bookworm/ Ubuntu 22.04)
FROM lukemathwalker/cargo-chef:latest-rust-1.75-bookworm AS chef
WORKDIR /usr/src

ARG CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse

FROM chef as planner
COPY --from=tgi /tgi/Cargo.toml Cargo.toml
COPY --from=tgi /tgi/rust-toolchain.toml rust-toolchain.toml
COPY --from=tgi /tgi/proto proto
COPY --from=tgi /tgi/benchmark benchmark
COPY --from=tgi /tgi/router router
COPY --from=tgi /tgi/launcher launcher
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder

RUN PROTOC_ZIP=protoc-21.12-linux-x86_64.zip && \
    curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP && \
    unzip -o $PROTOC_ZIP -d /usr/local bin/protoc && \
    unzip -o $PROTOC_ZIP -d /usr/local 'include/*' && \
    rm -f $PROTOC_ZIP

COPY --from=planner /usr/src/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json

COPY --from=tgi /tgi/Cargo.toml Cargo.toml
COPY --from=tgi /tgi/rust-toolchain.toml rust-toolchain.toml
COPY --from=tgi /tgi/proto proto
COPY --from=tgi /tgi/benchmark benchmark
COPY --from=tgi /tgi/router router
COPY --from=tgi /tgi/launcher launcher
RUN cargo build --release --workspace --exclude benchmark

# Python base image
FROM ubuntu:22.04 AS base

RUN apt-get update -y \
 && apt-get install -y --no-install-recommends \
    python3-pip \
    python3-setuptools \
    python-is-python3 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
RUN pip3 --no-cache-dir install --upgrade pip

# Python server build image
FROM base AS pyserver

ARG VERSION

RUN test -n ${VERSION:?} && apt-get update -y \
 && apt-get install -y --no-install-recommends \
    make \
    python3-venv \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN install -d /pyserver
WORKDIR /pyserver
COPY text-generation-inference/server server
COPY --from=tgi /tgi/proto proto
RUN pip3 install -r server/build-requirements.txt
RUN VERBOSE=1 BUILDDIR=/pyserver/build PROTODIR=/pyserver/proto VERSION=${VERSION} make -C server gen-server

# Neuron base image (used for deployment)
FROM base AS neuron

ARG VERSION

# Install system prerequisites
RUN test -n ${VERSION:?} && apt-get update -y \
 && apt-get install -y --no-install-recommends \
    gnupg2 \
    wget \
    python3-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN echo "deb https://apt.repos.neuron.amazonaws.com jammy main" > /etc/apt/sources.list.d/neuron.list
RUN wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | apt-key add -

# Install neuronx packages
RUN apt-get update -y \
 && apt-get install -y --no-install-recommends \
    aws-neuronx-dkms=2.16.7.0 \
    aws-neuronx-collectives=2.20.22.0-c101c322e \
    aws-neuronx-runtime-lib=2.20.22.0-1b3ca6425 \
    aws-neuronx-tools=2.17.1.0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

ENV PATH="/opt/bin/:/opt/aws/neuron/bin:${PATH}"

RUN pip3 install \
    neuronx-cc==2.13.66.0 \
    torch-neuronx==2.1.2.2.1.0 \
    transformers-neuronx==0.10.0.21 \
    --extra-index-url=https://pip.repos.neuron.amazonaws.com

# Install HuggingFace packages
RUN pip3 install \
    hf_transfer huggingface_hub

# Install optimum-neuron
COPY dist/optimum_neuron-${VERSION}.tar.gz optimum-neuron.tar.gz
RUN pip3 install optimum-neuron.tar.gz

# TGI base env
ENV HUGGINGFACE_HUB_CACHE=/data \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PORT=80

# Install router
COPY --from=builder /usr/src/target/release/text-generation-router /usr/local/bin/text-generation-router
# Install launcher
COPY --from=builder /usr/src/target/release/text-generation-launcher /usr/local/bin/text-generation-launcher
# Install python server
COPY --from=pyserver /pyserver/build/dist dist
RUN pip install dist/text_generation_server*.tar.gz

# AWS Sagemaker compatible image
FROM neuron as sagemaker

COPY text-generation-inference/sagemaker-entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]

# Final image
FROM neuron
COPY text-generation-inference/tgi-entrypoint.sh text-generation-inference/tgi_env.py /
ENTRYPOINT ["/tgi-entrypoint.sh"]
CMD ["--json-output"]
