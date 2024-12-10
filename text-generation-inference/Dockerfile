# Fetch and extract the TGI sources
FROM alpine AS tgi
ARG TGI_VERSION=3.0.0
RUN mkdir -p /tgi
ADD https://github.com/huggingface/text-generation-inference/archive/refs/tags/v${TGI_VERSION}.tar.gz /tgi/sources.tar.gz
RUN tar -C /tgi -xf /tgi/sources.tar.gz --strip-components=1

# Build cargo components (adapted from TGI original Dockerfile)
# Note: we cannot use the cargo-chef base image as it uses python 3.11
FROM ubuntu:22.04 AS chef

RUN apt-get update -y \
 && apt-get install -y --no-install-recommends \
    curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- --default-toolchain 1.80.1 --profile minimal -y
ENV PATH="/root/.cargo/bin:${PATH}"
RUN cargo install cargo-chef --locked

WORKDIR /usr/src

ARG CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse

FROM chef AS planner
COPY text-generation-inference/Cargo.toml Cargo.toml
COPY --from=tgi /tgi/Cargo.lock Cargo.lock
COPY --from=tgi /tgi/rust-toolchain.toml rust-toolchain.toml
COPY --from=tgi /tgi/proto proto
COPY --from=tgi /tgi/router router
COPY --from=tgi /tgi/backends backends
COPY --from=tgi /tgi/launcher launcher
RUN cargo chef prepare --recipe-path recipe.json

FROM chef AS builder

RUN apt-get update -y \
 && apt-get install -y --no-install-recommends \
    unzip python3-dev libssl-dev pkg-config \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN PROTOC_ZIP=protoc-21.12-linux-x86_64.zip && \
    curl -OL https://github.com/protocolbuffers/protobuf/releases/download/v21.12/$PROTOC_ZIP && \
    unzip -o $PROTOC_ZIP -d /usr/local bin/protoc && \
    unzip -o $PROTOC_ZIP -d /usr/local 'include/*' && \
    rm -f $PROTOC_ZIP

COPY text-generation-inference/Cargo.toml Cargo.toml
COPY --from=planner /usr/src/recipe.json recipe.json
RUN cargo chef cook --release --recipe-path recipe.json

COPY text-generation-inference/Cargo.toml Cargo.toml
COPY --from=tgi /tgi/Cargo.lock Cargo.lock
COPY --from=tgi /tgi/rust-toolchain.toml rust-toolchain.toml
COPY --from=tgi /tgi/proto proto
COPY --from=tgi /tgi/router router
COPY --from=tgi /tgi/backends backends
COPY --from=tgi /tgi/launcher launcher
# Remove this line once TGI has fixed the conflict
RUN cargo update ureq --precise 2.9.7
RUN cargo build --release

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
    aws-neuronx-dkms=2.18.20.0 \
    aws-neuronx-collectives=2.22.33.0-d2128d1aa \
    aws-neuronx-runtime-lib=2.22.19.0-5856c0b42 \
    aws-neuronx-tools=2.19.0.0 \
    libxml2 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

ENV PATH="/opt/bin/:/opt/aws/neuron/bin:${PATH}"

RUN pip3 install \
    neuronx-cc==2.15.143.0 \
    torch-neuronx==2.1.2.2.3.2 \
    transformers-neuronx==0.12.313 \
    libneuronxla==2.0.5347.0 \
    --extra-index-url=https://pip.repos.neuron.amazonaws.com

# Install HuggingFace packages
RUN pip3 install \
    hf_transfer huggingface_hub

# Install optimum-neuron
COPY ./dist/optimum_neuron-${VERSION}.tar.gz optimum-neuron.tar.gz
RUN pip3 install optimum-neuron.tar.gz

# TGI base env
ENV HUGGINGFACE_HUB_CACHE=/data \
    HF_HUB_ENABLE_HF_TRANSFER=1 \
    PORT=80

# Install router
COPY --from=builder /usr/src/target/release/text-generation-router-v2 /usr/local/bin/text-generation-router
# Install launcher
COPY --from=builder /usr/src/target/release/text-generation-launcher /usr/local/bin/text-generation-launcher
# Install python server
COPY --from=pyserver /pyserver/build/dist dist
RUN pip install dist/text_generation_server*.tar.gz

# AWS Sagemaker compatible image
FROM neuron AS sagemaker

COPY text-generation-inference/sagemaker-entrypoint.sh entrypoint.sh
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]

# Final image
FROM neuron
COPY text-generation-inference/tgi-entrypoint.sh text-generation-inference/tgi_env.py /
ENTRYPOINT ["/tgi-entrypoint.sh"]
CMD ["--json-output"]
