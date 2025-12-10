# optimum-neuron vLLM benchmark

## Local environment setup

These configurations are tested and run on with the Hugging Face Deep Learning AMI from the AWS Marketplace.

Copy the configurations down using

```shell
$ git clone https://github.com/huggingface/optimum-neuron.git
```

## Choose a deployment scenario and start the servers

The `data-parallel` directory contains instructions to deploy model configurations that use data parallelism.

The `single-instance` directory contains instructions to deploy model configurations that use a single server instance.

## Run the performance benchmark

### Install `guidellm`

```shell
$ pip install guidellm==0.1.0
```

### Launch benchmark run

The benchmark script takes the `model_id` and number of concurrent users as parameters.
The `model_id` must match the one corresponding to the selected model configuration.

```shell
$ cd optimum-neuron/benchmark/vllm/
$ ./performance.sh "meta-llama/Meta-Llama-3.1-8B-Instruct" 128
```

Note that the evaluated model **must** be an `Instruct` model.

At the end of the benchmark, the results will be saved in a `.json` file and a
summary will be displayed on the console.

To obtain a summary from the raw benchmark output files, use the following command:

```shell
$ python generate_csv.py --dir <path_to_result_files>
```

## Run the accuracy benchmark

### Install lm_eval

```shell
$ pip install lm_eval[api]
```

### Run the benchmark

```shell
$ cd optimum-neuron/benchmark/vllm/
$ ./accuracy.sh meta-llama/Meta-Llama-3.1-8B-Instruct
```
