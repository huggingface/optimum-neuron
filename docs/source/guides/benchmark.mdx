## Introduction

In todays world, mostly every AI Engineer is familiar with running inference by simply making an API call, but how is that request served optimally by the backend? How does the model provider or service you are using ensure latency and throughput requirements are met?

In this blog we will cover how to serve a model using Optimum Neuron on AWS Inferentia2 with the HuggingFace vLLM container. I'll also delve into how to optimize for latency and throughput and what decisions we can make to influence our priorities.

## Understanding the Tools

* Inferentia2 chips: Inferentia2 is the second generation AWS purpose-built Machine Learning inference accelerator.
* Optimum Neuron: The interface between the 🤗 Transformers library and AWS Accelerators including AWS Trainium and AWS Inferentia.
* vLLM container: vLLM is a toolkit for deploying and serving Large Language Models (LLMs).
* GuideLLM: A tool for evaluating and optimizing the deployment of large language models (LLMs).

The instance I am using for this experiment will be `inf2.48xlarge`. I can check instance type as well as see each device by running `neuron-ls` which gives the following output:

```
instance-type: inf2.48xlarge
+--------+--------+--------+-----------+---------+
| NEURON | NEURON | NEURON | CONNECTED |   PCI   |
| DEVICE | CORES  | MEMORY |  DEVICES  |   BDF   |
+--------+--------+--------+-----------+---------+
| 0      | 2      | 32 GB  | 11, 1     | 80:1e.0 |
| 1      | 2      | 32 GB  | 0, 2      | 90:1e.0 |
| 2      | 2      | 32 GB  | 1, 3      | 80:1d.0 |
| 3      | 2      | 32 GB  | 2, 4      | 90:1f.0 |
| 4      | 2      | 32 GB  | 3, 5      | 80:1f.0 |
| 5      | 2      | 32 GB  | 4, 6      | 90:1d.0 |
| 6      | 2      | 32 GB  | 5, 7      | 20:1e.0 |
| 7      | 2      | 32 GB  | 6, 8      | 20:1f.0 |
| 8      | 2      | 32 GB  | 7, 9      | 10:1e.0 |
| 9      | 2      | 32 GB  | 8, 10     | 10:1f.0 |
| 10     | 2      | 32 GB  | 9, 11     | 10:1d.0 |
| 11     | 2      | 32 GB  | 10, 0     | 20:1d.0 |
+--------+--------+--------+-----------+---------+
```

## Setup and Installation

First, I ran the following commands to install the necessary dependencies, and pull the container needed to compile the model, as well as serve the compiled model for benchmarking.

`!pip install hftransfer guidellm==0.1.0`
`!git clone https://github.com/huggingface/optimum-neuron.git`
`!docker pull ghcr.io/huggingface/text-generation-inference:latest-neuron`

Depending on the model, optionally configure your HF_TOKEN like so:

`!export HF_TOKEN=YOUR_HF_TOKEN`

## Model Compilation and Deployment

For my use case, I needed to compile my model with specific parameters that were unique. It is important to mention that compilation is not always needed. For example, in the event that the already cached configuration would have worked for me, optimum would use that by default.

From the docs: "The Neuron Model Cache is a remote cache for compiled Neuron models in the `neff` format. It is integrated into the `NeuronTrainer` and `NeuronModelForCausalLM` classes to enable loading pretrained models from the cache instead of compiling them locally."

Now I compile the model I have selected, `meta-llama-3.1-8b-instruct` with the following command:

```bash
!docker run -p 8080:80 -e HF_TOKEN=YOUR_TOKEN \
-v $(pwd):/data \
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
-ti \
--entrypoint "optimum-cli" ghcr.io/huggingface/text-generation-inference:latest-neuron \
export neuron --model "meta-llama/Meta-Llama-3.1-8B-Instruct" \
--sequence_length 16512 \
--batch_size 8 \
--num_cores 8 \
/data/exportedmodel/
```

Take note that for my use case, I have decided to use a batch size of 8, with a tensor parallel degree of 8. Since an inf2.48xlarge has 24 cores, I can use a data parallel of 3, which means I will have 3 copies of my model across the instance.`

## Optimizing Batch Size for Maximum Throughput

When optimizing hardware utilization for cost-efficiency, particularly for the inf2.48xlarge instance at $12.98 per hour on-demand, the roofline model is a valuable framework.

The roofline model defines theoretical performance bounds. On one extreme, memory-bound workloads are limited by memory capacity, necessitating frequent read/write operations. On the other, compute-bound workloads fully utilize the accelerator's compute capabilities, maximizing on-device data processing.
Batch size is a key lever for controlling this balance. Larger batch sizes tend to shift workloads towards being compute-bound, while smaller batch sizes may result in more memory-bound operations.
With that stated, maximizing batch size is not always viable. Keeping in mind max batch size for the specified latency budget (the time we want to take to return a response) is paramount.
This is most directly controlled with batch size. For more information on this topic, check out this resource:
https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-features/neuroncore-batching.html

## Creating Files for Serving

Several files are needed to ensure our configuration is setup properly, and that the model I compiled is used rather than the cached configuration.

First I'll need to create my .env file, which specifies my batch size, precision, etc. It is important to note, that since I compiled my model, I needed to change the model_id from the usual huggingface repo designation, to the container volume location I specified within the compilation command.

```
MODEL_ID='/data/exportedmodel'
MAX_BATCH_SIZE=8
MAX_TOTAL_TOKENS=16512
```

Next, I create the benchmark.sh script with my desired settings:

```bash
#!/bin/bash

model=${1:-meta-llama/Meta-Llama-3.1-8B-Instruct}

date_str=$(date '+%Y-%m-%d-%H-%M-%S')
output_path="${model//\//_}#${date_str}_guidellm_report.json"

export HF_TOKEN=YOUR_TOKEN

export GUIDELLM__NUM_SWEEP_PROFILES=1
export GUIDELLM__MAX_CONCURRENCY=128
export GUIDELLM__REQUEST_TIMEOUT=60

guidellm \
 --target "http://localhost:8080/v1" \
 --model ${model} \
 --data-type emulated \
 --data "prompt_tokens=15900,prompt_tokens_variance=100,generated_tokens=450,generated_tokens_variance=50" \
 --output-path ${output_path} \
 ```

Take note of the parameters passed via the `--data` flag. As my use case is for long prompts and long generation, I have set `prompt_tokens` and `generated_tokens accordingly. Remember to set these according to your use case and the input / output token load you expect.
Based on these numbers, GuideLLM will generate prompts of random sizes in a normal distribution of around 15900 tokens, and ask for a random number of generated tokens in a normal distribution of around 450 tokens.

The docker compose file is important for defining your data parallel, by specifying the number of devices I wish to allocate to each container. This is also where I specify the load balancer.

```
version: '3.7'

services:
 vllm-1:
 image: ghcr.io/huggingface/optimum-neuron-vllm:latest
 ports:
 - "8081:8081"
 volumes:
 - $PWD:/data
 environment:
 - PORT=8081
 - SM_ON_MODEL=${MODEL_ID}
 - SM_ON_TENSOR_PARALLEL_SIZE=8
 - SM_ON_BATCH_SIZE=${MAX_BATCH_SIZE}
 - HF_TOKEN=YOUR_TOKEN
 - SM_ON_SEQUENCE_LENGTH=${MAX_TOTAL_TOKENS}
 devices:
 - "/dev/neuron0"
 - "/dev/neuron1"
 - "/dev/neuron2"
 - "/dev/neuron3"

 vllm-2:
 image: ghcr.io/huggingface/optimum-neuron-vllm:latest
 ports:
 - "8082:8082"
 volumes:
 - $PWD:/data
 environment:
 - PORT=8082
 - SM_ON_MODEL=${MODEL_ID}
 - SM_ON_TENSOR_PARALLEL_SIZE=8
 - SM_ON_BATCH_SIZE=${MAX_BATCH_SIZE}
 - HF_TOKEN=YOUR_TOKEN
 - SM_ON_SEQUENCE_LENGTH=${MAX_TOTAL_TOKENS}
 devices:
 - "/dev/neuron4"
 - "/dev/neuron5"
 - "/dev/neuron6"
 - "/dev/neuron7"

 vllm-3:
 image: ghcr.io/huggingface/optimum-neuron-vllm:latest
 ports:
 - "8083:8083"
 volumes:
 - $PWD:/data
 environment:
 - PORT=8083
 - SM_ON_MODEL=${MODEL_ID}
 - SM_ON_TENSOR_PARALLEL_SIZE=8
 - SM_ON_BATCH_SIZE=${MAX_BATCH_SIZE}
 - HF_TOKEN=YOUR_TOKEN
 - SM_ON_SEQUENCE_LENGTH=${MAX_TOTAL_TOKENS}
 devices:
 - "/dev/neuron8"
 - "/dev/neuron9"
 - "/dev/neuron10"
 - "/dev/neuron11"

 loadbalancer:
 image: nginx:alpine
 ports:
 - "8080:80"
 volumes:
 - ./nginx.conf:/etc/nginx/nginx.conf:ro
 depends_on:
 - vllm-1
 - vllm-2
 - vllm-3
 deploy:
 placement:
 constraints: [node.role == manager]
```

Lastly, I define the nginx.conf for the load balancer:

```
### Nginx vllm Load Balancer
events {}
http {
 upstream vllmcluster {
    server vllm-1:8081;
    server vllm-2:8082;
    server vllm-3:8083;
 }
 server {
    listen 80;
    location / {
    proxy_pass http://vllmcluster;
    }
 }
}
```

## Benchmarking with GuideLLM

Now that I have defined the necessary files, I start serving my optimum-neuron model with vllm backend.

`!docker compose -f docker-compose.yaml --env-file .env up`

As a sanity check, I can watch the output of the above command to ensure that each container starts properly as well as the load balancer.
Once I have started the containers successfully, I can begin benchmarking using the previously defined benchmarking script.

`!benchmark.sh "meta-llama/Meta-Llama-3.1-8B-Instruct"`

A colorful stdout will begin to populate the terminal as guidellm begins to test your model serving setup.

## Performance Analysis

In approximately 15-20 minutes, benchmarking is completed and displays the following detailed breakdown in the terminal:

```
╭─ Benchmarks ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ [15:02:17] 100% synchronous (0.10 req/sec avg)│
│ [15:04:17] 100% throughput (0.85 req/sec avg)│
│ [15:05:25] 100% constant@0.85 req/s (0.77 req/sec avg) │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
 Generating report... ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ (3/3) [ 0:05:04 < 0:00:00 ]
╭─ GuideLLM Benchmarks Report (meta-llama_Meta-Llama-3.1-8B-Instruct#2025-05-27-15-02-11_guidellm_report.json) ──────────────────────────────────────────────────────────────────────────────────╮
│ ╭─ Benchmark Report 1 ───────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮ │
│ │ Backend(type=openai_server, target=http://localhost:8080/v1, model=meta-llama/Meta-Llama-3.1-8B-Instruct) │ │
│ │ Data(type=emulated, source=prompt_tokens=15900,prompt_tokens_variance=100,generated_tokens=450,generated_tokens_variance=50, tokenizer=meta-llama/Meta-Llama-3.1-8B-Instruct) │ │
│ │ Rate(type=sweep, rate=None) │ │
│ │ Limits(max_number=None requests, max_duration=120 sec) │ │
│ │ │ │
│ │ │ │
│ │ Requests Data by Benchmark │ │
│ │ ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━━━┳━━━━━━━━━━┓ │ │
│ │ ┃ Benchmark ┃ Requests Completed ┃ Request Failed ┃ Duration ┃ Start Time ┃ End Time ┃ │ │
│ │ ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━━━╇━━━━━━━━━━┩ │ │
│ │ │ synchronous │ 11/11 │ 0/11 │ 113.56 sec │ 15:02:17 │ 15:04:11 │ │ │
│ │ │ asynchronous@0.85 req/sec │ 88/88 │ 0/88 │ 114.59 sec │ 15:05:25 │ 15:07:19 │ │ │
│ │ │ throughput │ 55/55 │ 0/55 │ 64.83 sec │ 15:04:17 │ 15:05:22 │ │ │
│ │ └───────────────────────────┴────────────────────┴────────────────┴────────────┴────────────┴──────────┘ │ │
│ │ │ │
│ │ Tokens Data by Benchmark │ │
│ │ ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓ │ │
│ │ ┃ Benchmark ┃ Prompt ┃ Prompt (1%, 5%, 50%, 95%, 99%) ┃ Output ┃ Output (1%, 5%, 50%, 95%, 99%) ┃ │ │
│ │ ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩ │ │
│ │ │ synchronous │ 15902.82 │ 15896.0, 15896.0, 15902.0, 15913.0, 15914.6 │ 293.09 │ 70.3, 119.5, 315.0, 423.5, 443.1 │ │ │
│ │ │ asynchronous@0.85 req/sec │ 15899.06 │ 15877.4, 15879.4, 15898.5, 15918.0, 15919.8 │ 288.75 │ 24.6, 74.1, 298.5, 452.6, 459.1 │ │ │
│ │ │ throughput │ 15899.22 │ 15879.5, 15883.7, 15898.0, 15914.6, 15920.5 │ 294.24 │ 59.1, 114.9, 285.0, 452.9, 456.4 │ │ │
│ │ └───────────────────────────┴──────────┴─────────────────────────────────────────────┴────────┴──────────────────────────────────┘ │ │
│ │ │ │
│ │ Performance Stats by Benchmark │ │
│ │ ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓ │ │
│ │ ┃ ┃ Request Latency [1%, 5%, 10%, 50%, 90%, 95%, 99%] ┃ Time to First Token [1%, 5%, 10%, 50%, 90%, 95%, ┃ Inter Token Latency [1%, 5%, 10%, 50%, 90% 95%, ┃ │ │
│ │ ┃ Benchmark ┃ (sec) ┃ 99%] (ms) ┃ 99%] (ms) ┃ │ │
│ │ ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩ │ │
│ │ │ synchronous │ 3.68, 5.13, 6.94, 10.91, 13.51, 14.26, 14.87 │ 1563.3, 1569.2, 1576.5, 1589.4, 1594.0, 1595.3, │ 23.2, 28.2, 29.4, 29.8, 30.3, 31.7, 36.5 │ │ │
│ │ │ │ │ 1596.4 │ │ │ │
│ │ │ asynchronous@0.85 req/sec │ 2.62, 6.55, 9.40, 20.66, 30.60, 32.78, 35.07 │ 1594.1, 1602.5, 1605.7, 1629.7, 4650.1, 4924.1, │ 0.2, 0.2, 0.2, 34.3, 44.9, 54.5, 1613.9 │ │ │
│ │ │ │ │ 5345.6 │ │ │ │
│ │ │ throughput │ 18.29, 21.24, 23.81, 44.60, 61.50, 62.80, 63.72 │ 2157.6, 9185.1, 12220.5, 23333.5, 44214.1, │ 28.2, 31.5, 33.1, 39.1, 59.0, 65.2, 1604.6 │ │ │
│ │ │ │ │ 45329.8, 51276.9 │ │ │ │
│ │ └───────────────────────────┴───────────────────────────────────────────────────┴───────────────────────────────────────────────────┴────────────────────────────────────────────────────┘ │ │
│ │ │ │
│ │ Performance Summary by Benchmark │ │
│ │ ┏━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━┓ │ │
│ │ ┃ Benchmark ┃ Requests per Second ┃ Request Latency ┃ Time to First Token ┃ Inter Token Latency ┃ Output Token Throughput ┃ │ │
│ │ ┡━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━┩ │ │
│ │ │ synchronous │ 0.10 req/sec │ 10.32 sec │ 1585.08 ms │ 29.81 ms │ 28.39 tokens/sec │ │ │
│ │ │ asynchronous@0.85 req/sec │ 0.77 req/sec │ 20.77 sec │ 2401.32 ms │ 63.69 ms │ 221.75 tokens/sec │ │ │
│ │ │ throughput │ 0.85 req/sec │ 43.78 sec │ 24624.46 ms │ 65.18 ms │ 249.64 tokens/sec │ │ │
│ │ └───────────────────────────┴─────────────────────┴─────────────────┴─────────────────────┴─────────────────────┴─────────────────────────┘ │ │
│ ╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯ │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

Unpacking the results, we get quite a few useful data points for us to use. Under the hood, guidellm runs three separate "loads" with which to benchmark the system against.

1. Synchronous - Serving one request at a time
2. Asynchronous - Serving multiple requests at once at a locked in req/sec (0.85 in this case)
3. Throughput - Serving the maximum number of requests that the system can sustain

From these tests we are given several metrics for each like how many requests were successfully performed vs how many failed. The time to first token, prompt input and output sizes and more.
For my experiment, I can see that under max load, I can serve up to 0.85 requests per second at a maximum latency of just under 44 seconds per request. Depending on my latency budget, the next step would be to increase my batch size if I can tolerate longer response times and desire more throughput. Alternatively, I could lower my batch size to decrease the latency, at the cost of potentially reducing throughput.

Lastly, the large input and output tokens required for my workload directly effect the benchmark results, specifically the time needed to encode my input context contributing to most of the benchmark time.

## Conclusion

In this blog post, I took you through how to compile and load an Optimum Neuron model, how to serve it with the HuggingFace Text Generation Inference container, and how to benchmark your settings to optimize for your workload.

## References

https://huggingface.co/docs/optimum-neuron/en/guides/cache_system
https://github.com/huggingface/optimum-neuron/tree/main/benchmark/vllm/performance
https://github.com/vllm-project/guidellm
