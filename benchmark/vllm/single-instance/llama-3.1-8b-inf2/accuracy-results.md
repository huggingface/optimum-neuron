# Accuracy results — meta-llama/Llama-3.1-8B-Instruct (inf2, TP=2 / BS=4 / SL=4096)

Date: 2026-08-24
Instance: inf2.24xlarge (2 devices / 4 NeuronCores)

## Command

```shell
./accuracy.sh meta-llama/Llama-3.1-8B-Instruct 4 gsm8k
```

i.e. `lm_eval --model local-completions --tasks gsm8k --batch_size 4` against the
vLLM server started with `single-instance/serve.sh` for this configuration.

## Results

| Task | Filter | n-shot | Metric | Value | Stderr |
|------|--------|--------|--------|-------|--------|
| gsm8k | flexible-extract | 5 | exact_match | 0.7756 | ± 0.0115 |
| gsm8k | strict-match | 5 | exact_match | 0.7111 | ± 0.0125 |

## Environment

- optimum-neuron 0.4.7.dev0
- vllm 0.16.0
- torch-neuronx 2.9.0.2.15.32035+de43f57c
- neuronx-cc 2.26.6360.0+6f180f47
- neuronx-distributed 0.19.28492+435aae2b
- aws-neuronx-runtime-lib 2.33.10.0
- lm-eval 0.4.12
