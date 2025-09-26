# Serve a neuron model using a single instance

## Setup the serving environment

This step can be omitted if you are using the [Hugging Face Neuron DLAMI](https://aws.amazon.com/marketplace/pp/prodview-gr3e6yiscria2).

```shell
pip install .[neuronx,vllm]
```

## Serve the model

```shell
serve.sh <MODEL_CONFIGURATION_DIRECTORY>
```

Note: configurations that use a tensor parallel size of 32 require a trn1.32xlarge instance
