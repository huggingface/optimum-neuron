# NeuronX TGI benchmark using multiple replicas

## Select model and configuration

Edit the `.env` file to select the model to use for the benchmark and its configuration.

## Start the servers

```shell
$ docker compose --env-file llama-7b/.env up
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


