# Text-generation-inference docker image for AWS inferentia2

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
- the static KV cache is rebuilt entirely during prefill.

## License

This docker image is released under [HFOIL 1.0](https://github.com/huggingface/text-generation-inference/blob/bde25e62b33b05113519e5dbf75abda06a03328e/LICENSE).

HFOIL stands for Hugging Face Optimized Inference License, and it has been specifically designed for our optimized inference solutions. While the source code remains accessible, HFOIL is not a true open source license because we added a restriction: to sell a hosted or managed service built on top of TGI, we require a separate agreement.

Please refer to [this reference documentation](https://github.com/huggingface/text-generation-inference/issues/726) to see if the HFOIL 1.0 restrictions apply to your deployment.

## Deploy the service

The service is launched simply by running the neuronx-tgi container with two sets of parameters:

```
docker run <system_parameters> ghcr.io/huggingface/neuronx-tgi:latest <service_parameters>
```

- system parameters are used to map ports, volumes and devices between the host and the service,
- service parameters are forwarded to the `text-generation-launcher`.

When deploying a service, you will need a working Neuron model. The NeuronX TGI service supports two main modes of operation:

- you can either deploy the service on a model that has already been exported to Neuron,
- or alternatively you can take advantage of the Neuron Model Cache to export your own model.

### Common system parameters

Whenever you launch a TGI service, we highly recommend you to mount a shared volume mounted as `/data` in the container: this is where
the models will be cached to speed up further instantiations of the service.

Note also that all neuron devices have to be explicitly made visible to the container.

Finally, you might want to export the `HF_TOKEN` if you want to access gated repository.

Here is an example of a service instantiation:

```
docker run -p 8080:80 \
       -v $(pwd)/data:/data \
       --device=/dev/neuron0 \
       -e HF_TOKEN=${HF_TOKEN} \
       ghcr.io/huggingface/neuronx-tgi:latest \
       <service_parameters>
```

If your instance has 12 neuron devices, the launch command becomes:

```
docker run -p 8080:80 \
       -v $(pwd)/data:/data \
       --device=/dev/neuron0 \
       --device=/dev/neuron1 \
       --device=/dev/neuron2 \
       --device=/dev/neuron3 \
       --device=/dev/neuron4 \
       --device=/dev/neuron5 \
       --device=/dev/neuron6 \
       --device=/dev/neuron7 \
       --device=/dev/neuron8 \
       --device=/dev/neuron9 \
       --device=/dev/neuron10 \
       --device=/dev/neuron11 \
       -e HF_TOKEN=${HF_TOKEN} \
       ghcr.io/huggingface/neuronx-tgi:latest \
       <service_parameters>
```


### Using a neuron model from the 🤗 [HuggingFace Hub](https://huggingface.co/aws-neuron) (recommended)

There are plenty of already exported neuron models on the hub, under the [aws-neuron](https://huggingface.co/aws-neuron) organization.

The snippet below shows how you can deploy a service from a hub neuron model:

```
docker run -p 8080:80 \
       -v $(pwd)/data:/data \
       --device=/dev/neuron0 \
       -e HF_TOKEN=${HF_TOKEN} \
       ghcr.io/huggingface/neuronx-tgi:latest \
       --model-id aws-neuron/Llama-2-7b-hf-neuron-budget \
       --max-batch-size 1 \
       --max-input-length 1024 \
       --max-total-tokens 2048
```

### Using a standard model from the 🤗 [HuggingFace Hub](https://huggingface.co/aws-neuron)


We maintain a Neuron Model Cache of the most popular architecture and deployment parameters under [aws-neuron/optimum-neuron-cache](https://huggingface.co/aws-neuron/optimum-neuron-cache).

If you just want to try the service quickly using a model that has not bee exported yet, it is thus still
possible to export it dynamically, pending some conditions:
- you must specify the export parameters when launching the service (or use default parameters),
- the model configuration must be cached.

The snippet below shows how you can deploy a service from a hub standard model:

```
docker run -p 8080:80 \
       -v $(pwd)/data:/data \
       --device=/dev/neuron0 \
       -e HF_TOKEN=${HF_TOKEN} \
       -e HF_AUTO_CAST_TYPE="fp16" \
       -e HF_NUM_CORES=2 \
       ghcr.io/huggingface/neuronx-tgi:latest \
       --model-id NousResearch/Llama-2-7b-chat-hf \
       --max-batch-size 1 \
       --max-input-length 3164 \
       --max-total-tokens 4096
```

### Using a model exported to a local path

Alternatively, you can first [export the model to neuron format](https://huggingface.co/docs/optimum-neuron/main/en/guides/models#configuring-the-export-of-a-generative-model) locally, and deploy the service inside the shared volume:

```
docker run -p 8080:80 \
       -v $(pwd)/data:/data \
       --device=/dev/neuron0 \
       ghcr.io/huggingface/neuronx-tgi:latest \
       --model-id /data/<neuron_model_path> \
       ...
```

### Choosing service parameters

Use the following command to list the available service parameters:

```
docker run ghcr.io/huggingface/neuronx-tgi --help
```

The configuration of an inference endpoint is always a compromise between throughput and latency: serving more requests in parallel will allow a higher throughput, but it will increase the latency.

The neuron models have static input dimensions `[batch_size, max_length]`.

This adds several restrictions to the following parameters:

- `--max-batch-size` must be set to `batch size`,
- `--max-input-length` must be lower than `max_length`,
- `--max-total-tokens` must be set to `max_length` (it is per-request).

Although not strictly necessary, but important for efficient prefilling:

- `--max-batch-prefill-tokens` should be set to `batch_size` * `max-input-length`.

### Choosing the correct batch size

As seen in the previous paragraph, neuron model static batch size has a direct influence on the endpoint latency and throughput.

Please refer to [text-generation-inference](https://github.com/huggingface/text-generation-inference) for optimization hints.

Note that the main constraint is to be able to fit the model for the specified `batch_size` within the total device memory available
on your instance (16GB per neuron core, with 2 cores per device).

All neuron models on the 🤗 [HuggingFace Hub](https://huggingface.co/aws-neuron) include the number of cores required to run them.

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

## Build your own image

The image must be built from the top directory

```
make neuronx-tgi
```
