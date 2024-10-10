# Evaluate LLM on several benchmarks

NeuronX TGI supports the OpenAI API, which allows to evaluate neuron models using [lm-harness](https://github.com/EleutherAI/lm-evaluation-harness).

Please refer to [lm-harness](https://github.com/EleutherAI/lm-evaluation-harness) documentation for installation instructions and benchmark comnfiguration.

## Some results

### meta-llama/Meta-Llama-3.1-8B-Instruct

Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.7885|±  |0.0112|
|     |       |strict-match    |     5|exact_match|↑  |0.0425|±  |0.0056|
