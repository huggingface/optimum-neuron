# Optimal export configurations

This file contains detailed instructions to find the optimal export configurations for a given decoder model.

## Prerequisites

**IMPORTANT: Before executing any command, you MUST activate the virtual environment.**

All instructions in this file assume that the commands are executed in a virtual environment containing
`optimum-neuron` with the AWS Neuron SDK installed.

### Create the virtual environment

```bash
# Create .venv only if it doesn't exist, then install dependencies if needed
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate

# Install optimum-neuron with neuronx extras only if not already installed
python -c "import optimum_neuron; import neuronx_distributed" 2>/dev/null || pip install -e ".[neuronx]"
```

### Activate the virtual environment:
```bash
source .venv/bin/activate
```

## Validation workflow

The workflow to find a valid export configurations is as follows:

- export the model to a local directory,
- instantiate it to verify it can be loaded on Neuron devices.

Note: the export and loading processes can last up to one hour and cannot be parallelized, so it is
better to avoid running them in the background. In addition, the console output will display meaningful
information about any errors that might occur.

### Causal LM models

For generative models, this workflow comprises the following commands:

Export:
```shell
optimum-cli export neuron -m <model_id> \
                          --batch_size <batch_size> \
                          --sequence_length <sequence_length> \
                          --tensor_parallel_size <tensor_parallel_size> \
                          --task text-generation \
                          ./data/<path_to_model>
```

Load:
```shell
python -c "from optimum.neuron import NeuronModelForCausalLM;model = NeuronModelForCausalLM.from_pretrained('./data<path_to_model>')"
```

**It is important to use separate processes for the tests, otherwise the neuron runtime will not properly release the devices**

### Embedding models

For embeddings models, this workflow comprises the following commands:

Export:
```shell
optimum-cli export neuron -m <model_id> \
                          --batch_size <batch_size> \
                          --sequence_length <sequence_length> \
                          --tensor_parallel_size <tensor_parallel_size> \
                          --task feature-extraction \
                          ./data/<path_to_model>
```

Load:
```shell
python -c "from optimum.neuron import NeuronModelForEmbedding;model = NeuronModelForEmbedding.from_pretrained('./data<path_to_model>')"
```

**It is important to use separate processes for the tests, otherwise the neuron runtime will not properly release the devices**


## Optimal configurations workflow

The workflow to find optimal configurations is to iterate over model configurations and store those that
produces neuron models that were successfully loaded onto neuron devices.

This workflow comprises the following steps:

- define a list of candidate tensor_parallel_size,
- define a list of batch size,
- define a list fo sequence length,
- iterate over:
    - **increasing** tensor_parallel_size,
    - **decreasing** batch_size,
    - **decreasing** sequence_length,
- if a configuration is valid, append it to a local <model_name>.json file, following this example for [llama4](https://huggingface.co/aws-neuron/optimum-neuron-cache/raw/main/inference-cache-config/trn1/llama4.json). Save the file every time it is updated to avoid losing the results if the script aborts,
- when starting iterating configurations, check if the <model_name>.json file is present and check for configurations that have already been tested,

## Defining a list of candidate tensor_parallel_size

The list of candidate tensor_parallel_size depends on the following:

- must be at least two,
- must be a multiple of the number of cores per device,
- must be less than the total number of neuron cores available on the system,
- must divide the model number of attention heads.

**You can get the total number of cores using the following command**

```python
from optimum.neuron.utils.system import get_available_cores

print(get_available_cores())
```

**You can get the number of cores per device using the following command**

```python
from optimum.neuron.utils.system import cores_per_device

print(cores_per_device())
```

**You can get the current instance type using the following command**

```python
from optimum.neuron.utils.instance import current_instance_type

print(current_instance_type())
```

## Defining a list of candidate batch sizes

The list of candidate batch sizes is always the following:

[1, 4, 8, 16, 32, 64, 128]

## Defining a list of candidate sequence length

- must not be less than 1024,
- must be a power-of-two,
- must be lower than the model maximum number of positions.

## Optimizations (**important**)

Once a configuration (tensor_parallel_size, batch_size, sequence_length) successfully validates, skip testing ALL configurations with the same tensor_parallel_size and batch_size but with SMALLER sequence_length values, without testing them. This is because if a model compiles and loads with a larger sequence length, it will definitely work with shorter sequence lengths for the same parallelism and batch size settings.

In the same way, when reloading an existing configuration file to resume testing:
- If a valid configuration exists for a given (tensor_parallel_size, batch_size) pair, skip testing all sequence_length values for that pair. Since sequence lengths are tested in decreasing order, any larger sequence lengths have already been tested and failed, while smaller ones will also work.
- If a valid configuration exists for a given tensor_parallel_size, skip testing all batch_size values larger than the maximum valid batch_size for that tensor_parallel_size. Since batch sizes are tested in decreasing order, higher batch sizes have already been tested and failed.
