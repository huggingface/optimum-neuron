# NeuronX TGI benchmark using multiple replicas

## Local environment setup

These configurations are tested and run on an inf2.48xlarge with the Hugging Face Deep Learning AMI from the AWS Marketplace.  

Copy the configurations down using

```shell
$ git clone https://github.com/huggingface/optimum-neuron.git
$ cd optimum-neuron/benchmark/text-generation-inference/
```


## Select model and configuration

Edit the `.env` file to select the model to use for the benchmark and its configuration.

The following instructions assume that you are testing a locally built image, so docker would have stored image neuronx-tgi:latest.

You can confirm this by running:

```shell
$ docker image ls
```

If you have not built it locally, you can download it and retag it using the following commands

```shell
$ docker pull ghcr.io/huggingface/neuronx-tgi:latest
$ docker tag ghcr.io/huggingface/neuronx-tgi:latest neuronx-tgi:latest
```
You should then see the single IMAGE ID with two different sets of tags:

```shell
$ docker image ls
REPOSITORY                        TAG       IMAGE ID       CREATED        SIZE
neuronx-tgi                       latest    f5ba57f8517b   12 hours ago   11.3GB
ghcr.io/huggingface/neuronx-tgi   latest    f5ba57f8517b   12 hours ago   11.3GB
```


Alternatively, you can edit the appropriate docker-compose.yaml to supply the fully path by changing ```neuronx-tgi:latest``` to ```ghcr.io/huggingface/neuronx-tgi:latest```

## Start the servers

For smaller models, you can use the multi-server configuration with a load balancer:

```shell
$ docker compose --env-file llama-7b/.env up
```

For larger models, use their specific docker files:

```shell
$ docker compose -f llama3-70b/docker-compose.yaml --env-file llama3-70b/.env up
```

Note: replace the .env file to change the model configuration

## Run the benchmark

### Install `llmperf`

Follow instalation [instructions](https://github.com/ray-project/llmperf/tree/main?tab=readme-ov-file#installation) for `llmperf`.

### Setup test environment

```shell
$ export LLMPerf=<path-to-llmperf>
```

### Launch benchmark run

The benchmark script takes the `model_id` and number of concurrent users as parameters.
The `model_id` must match the one corresponding to the selected `.env` file.

```
$ ./benchmark.sh NousResearch/Llama-2-7b-chat-hf 128
```


