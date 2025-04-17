# HLO Neuronx backend

This backend is a backport of features first implemented in the [transformers-neuronx](https://github.com/aws-neuron/transformers-neuronx) package from the
[AWS Neuron SDK](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/).

As the original [transformers-neuronx](https://github.com/aws-neuron/transformers-neuronx) implementation, it relies on XLA High Level Operations (HLO)
as the compiled language for implementing Neuron optimized transformer decoder classes.
More specifically, it uses a syntax called “PyHLO”, name of a Neuron internal tool for writing/compiling the HLO language in Python.

Each LLM model implementation using that backend should inherit from the `NeuronHloDecoderModel` class.

This class expects four parameters to be passed at initialization:

- a base transformers `PretrainedConfig`,
- a `HloNeuronConfig`,
- an instance of a "CPU" model that inherits from `PretrainedModel`,
- an instance of a HLO `DecoderGraphBuilder` that provides implementations for "hooks" in the forward pass of the model (typically before and inside the layers loop).

In addition, the child model class must implement a `load_weights` method that transfers the weights from the "CPU" model to the internal HLO graphs.

Plese refer to the implementaiton for the Llama architecture for reference.
