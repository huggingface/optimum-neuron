# Neuron LLM models accuracy

EleutherAI [Language Model Evaluation Harness](https://github.com/EleutherAI/lm-evaluation-harness) is compatible with neuron models
exported using `optimum-neuron`.

Please refer to [lm-harness](https://github.com/EleutherAI/lm-evaluation-harness) documentation for installation instructions and benchmark comnfiguration.

You can evaluate:
- a vanilla model (it will be exported before evaluation),
- a pre-exported neuron model (either from the hub or local).

## Some results

### meta-llama/Meta-Llama-3-8B-Instruct

|    Tasks     |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|--------------|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k         |      3|flexible-extract|     5|exact_match|↑  |0.7604|±  |0.0118|
|              |       |strict-match    |     5|exact_match|↑  |0.7635|±  |0.0117|
|hellaswag     |      1|none            |     0|acc        |↑  |0.5775|±  |0.0049|
|              |       |none            |     0|acc_norm   |↑  |0.7581|±  |0.0043|
|lambada_openai|      1|none            |     0|acc        |↑  |0.7173|±  |0.0063|
|              |       |none            |     0|perplexity |↓  |3.1102|±  |0.0769|
