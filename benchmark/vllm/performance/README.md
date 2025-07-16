# NeuronX vLLM benchmark using multiple replicas

## Local environment setup

These configurations are tested and run on an inf2.48xlarge with the Hugging Face Deep Learning AMI from the AWS Marketplace.

Copy the configurations down using

```shell
$ git clone https://github.com/huggingface/optimum-neuron.git
$ cd optimum-neuron/benchmark/vllm/
```

## Build the optimum-neuron vLLM image


```shell
$ make optimum-neuron-vllm
```

## Start the servers

If you are using a gated model, first export your credentials token.

```shell
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
```

Start the container servers with a single command:

```shell
$ docker compose -f llama3.1-8b/docker-compose.yaml --env-file llama3.1-8b/.env up
```

Note: you can edit the .env file to use a different model configuration

## Run the benchmark

### Install `guidellm`

```shell
$ pip install guidellm==0.1.0
```

### Launch benchmark run

The benchmark script takes the `model_id` and number of concurrent users as parameters.
The `model_id` must match the one corresponding to the selected `.env` file.

```shell
$ ./benchmark.sh "meta-llama/Meta-Llama-3.1-8B-Instruct"
```

Note that the evaluated model **must** be an `Instruct` model.

At the end of the benchmark, the results will be saved in a `.json` file and a
summary will be displayed on the console.

To obtain a summary from the raw benchmark output files, use the following command:

```shell
$ python generate_csv.py --dir <path_to_result_files>
```
