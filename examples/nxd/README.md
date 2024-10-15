# Llama NxD PoC

This is a Proof-of-Concept for the integration of Llama on top of `neuronx-distributed`.

The code tree replicates the internal organization of `optimum-neuron` in a separate tree, because today:

- Llama for training is integrated on top of `neuronx-distributed` with its own dynamic modeling code,
- Llama for inference is integrated on top of `transformers-neuronx`.

The code tree contains the following items:

- exporters: a folder containing the Llama `optimum.exporter.base.ExportConfig` that provides all parameters and methods required to export a vanilla `transformers.models.llama.LlamaModel` to a `NeuronDecoderModel`. The `NeuronDecoderModel` is a wrapper around a traced Llama model based on NxD,
- modeling_decoder.py: contains the `NeuronModelForCausalLM` class implemented on top of NxD (as opposed to the existing `optimum-neuron` class implemented on top of `transformers-neuronx`). This class uses two underlying `NeuronDecoderModel` for prefill and decode following the pattern in the AWS Neuron SDK Llama example,
- models: contains AWS Neuron SDK specific Llama modeling code, which is an alternative to `optimum-neuron` Llama modeling code used for training on top of NxD,
- modules: contains a few optional AWS Neuron SDK helpers for bucketing and on-device sampling.
