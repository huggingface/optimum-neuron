# NeuronX TGI: Text-generation-inference for AWS inferentia2

NeuronX TGI is distributed as docker images for [EC2](https://github.com/huggingface/optimum-neuron/pkgs/container/neuronx-tgi) and SageMaker.

These docker images integrate:

- the AWS Neuron SDK for Inferentia2,
- the [Text Generation Inference](https://github.com/huggingface/text-generation-inference) launcher and scheduling front-end,
- a neuron specific inference server for text-generation.

## Usage

Please refer to the official [documentation](https://huggingface.co/docs/optimum-neuron/main/en/guides/neuronx_tgi).

## Build your own image

The image must be built from the top directory

```
make neuronx-tgi
```
