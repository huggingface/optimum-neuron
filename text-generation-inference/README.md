# Text-generation-inference docker image

This docker image integrates into a base image:

- the AWS Neuron SDK for Inferentia2,
- the [Text Generation Inference](https://github.com/huggingface/text-generation-inference) launcher and scheduling front-end,
- a neuron specific inference server for text-generation.

## Features

The basic features of the [Text Generation Inference](https://github.com/huggingface/text-generation-inference) product are supported:

- continuous batching,
- token streaming,
- greedy search and multinomial sampling using [transformers](https://huggingface.co/docs/transformers/generation_strategies#customize-text-generation).

The main differences with the standard service for CUDA and CPU backends are that:

- the service uses a single internal static batch,
- new requests are inserted in the static batch during prefill,
- the static KV cache is rebuilt entirely during prefill (which makes it even more costly).

## Build image

The image must be built from the top directory

```
make neuronx-tgi
```

## Deploy the service

The service is launched simply by running the neuronx-tgi container with two sets of parameters:

```
docker run <system_parameters> neuronx-tgi:<version> <service_parameters>
```

- system parameters are used to map ports, volumes and devices between the host and the service,
- service parameters are forwarded to the `text-generation-launcher`.

The snippet below shows how you can deploy a service from a hub neuron model:

```
docker run -p 8080:80 \
       --device=/dev/neuron0 \
       neuronx-tgi:<version> \
       --model-id optimum/gpt2-neuronx-bs16 \
       --max-concurrent-requests 16 \
       --max-input-length 512 \
       --max-total-tokens 1024 \
       --max-batch-prefill-tokens 8192 \
       --max-batch-total-tokens 16384
```

Alternatively, you can first compile the model locally, and deploy the service using a shared volume:

```
docker run -p 8080:80 \
       -v $(pwd)/data:/data \
       --device=/dev/neuron0 \
       neuronx-tgi:0.0.11.dev0 \
       --model-id /data/neuron_gpt2_bs16 \
       --max-concurrent-requests 16 \
       --max-input-length 512 \
       --max-total-tokens 1024 \
       --max-batch-prefill-tokens 8192 \
       --max-batch-total-tokens 16384
```

### Choosing service parameters

Use the following command to list the available service parameters:

```
docker run neuronx-tgi --help
```

The configuration of an inference endpoint is always a compromise between throughput and latency: serving more requests in parallel will allow a higher throughput, but it will increase the latency.

The neuron models have static input dimensions `[batch_size, max_length]`.

It leads to a maximum number of tokens of `max_tokens = batch_size * max_length`.

This adds several restrictions to the following parameters:

- `--max-concurrent-requests` must be set to `batch size`,
- `--max-input-length` must be lower than `max_length`,
- `--max-total-tokens` must be set to `max_length` (it is per-request),
- `--max-batch-prefill-tokens` must be lower than `max_tokens`,
- `--max-batch-total-tokens` must be set to `max_tokens`.

### Choosing the correct batch size

As seen in the previous paragraph, neuron model static batch size has a direct influence on the endpoint latency and throughput.

For GPT2, a good compromise is a batch size of 16. If you need to absorb more load, then you can try a model compiled with a batch size of 128, but be aware
that the latency will increase a lot.

## Query the service

You can query the model using either the `/generate` or `/generate_stream` routes:

```
curl 127.0.0.1:8080/generate \
    -X POST \
    -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' \
    -H 'Content-Type: application/json'
```

```
curl 127.0.0.1:8080/generate_stream \
    -X POST \
    -d '{"inputs":"What is Deep Learning?","parameters":{"max_new_tokens":20}}' \
    -H 'Content-Type: application/json'
```
