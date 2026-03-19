# optimum-neuron vLLM benchmark

## Setup

This step can be omitted if you are using the [Hugging Face Neuron DLAMI](https://aws.amazon.com/marketplace/pp/prodview-gr3e6yiscria2).

```shell
pip install .[neuronx,vllm]
```

## Serve a model

Each model directory contains one or more `serve-dpX-tpY.env` files specifying
data-parallel size and tensor-parallel size.

If you are using a gated model, first export your credentials token.

```shell
export HF_TOKEN=$(cat ~/.cache/huggingface/token)
```

Start a server:

```shell
./serve.sh llama-3.1-8b/serve-dp1-tp8.env    # single replica
./serve.sh llama-3.1-8b/serve-dp3-tp8.env    # 3 DP replicas
```

Configurations with `DATA_PARALLEL_SIZE > 1` automatically launch multiple
replicas with a round-robin load balancer.

## Run the performance benchmark

```shell
pip install guidellm==0.1.0
./performance.sh "<MODEL_ID>" 128
```

The model ID can be omitted — it will be auto-detected from the running server.

Note that the evaluated model **must** be an `Instruct` model.

At the end of the benchmark, the results will be saved in a `.json` file and a
summary will be displayed on the console.

To obtain a summary from the raw benchmark output files, use the following command:

```shell
python generate_csv.py --dir <path_to_result_files>
```

## Run the accuracy benchmark

```shell
pip install lm_eval[api]
./accuracy.sh <MODEL_ID>
```
