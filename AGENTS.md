# Optimum Neuron Agent Guide (Root)

This repository bridges Hugging Face libraries with AWS Trainium/Inferentia. Use this file for project-wide guidance and see the model-specific guides:
- Inference models guide: [optimum/neuron/models/inference/AGENTS.md](optimum/neuron/models/inference/AGENTS.md)
- vLLM guide: [optimum/neuron/vllm/AGENTS.md](optimum/neuron/vllm/AGENTS.md)

## Core Architecture

### Three-Layer Model System
1. Export/Config layer: model export + configs in [optimum/exporters/neuron](optimum/exporters/neuron)
2. Inference layer: runtime models in [optimum/neuron/models/inference](optimum/neuron/models/inference)
3. Training layer: Trainium wrappers in [optimum/neuron/models/training](optimum/neuron/models/training)

### Backend Distinction: NxD vs Legacy
- NxD backend uses `neuronx_distributed` for tensor parallelism and static graphs.
- Legacy backend uses traced models in [optimum/neuron/modeling_traced.py](optimum/neuron/modeling_traced.py).
- vLLM integration only supports NxD backend.

## Essential Developer Workflows

### Testing (Neuron hardware required)
```bash
pytest tests/decoder/
pytest tests/training/
pytest tests/vllm/
```

### Code Quality
```bash
make style
make style_check
```

### Model Export (Compilation)
```bash
optimum-cli export neuron \
  --model meta-llama/Llama-3.1-8B \
  --batch_size 1 \
  --sequence_length 4096 \
  --tensor_parallel_size 8 \
  --torch_dtype bfloat16 \
  llama-neuron/
```

### Training Invocation
```bash
NEURON_CC_FLAGS="--model-type transformer" torchrun \
  --nproc_per_node 32 \
  examples/training/qwen3/finetune_qwen3.py
```

## Project-Specific Conventions

### Parallelism Terminology
- `tensor_parallel_size` (export/inference) ≡ `tp_degree` in `neuron_config.json`.
- Training uses `tensor_parallel_size`, `pipeline_parallel_size`, `sequence_parallel_enabled`.
- Single Trainium/Inferentia2 chip = 2 NeuronCores.

### Import Patterns
- Training: `from optimum.neuron.models.training import NeuronModelForCausalLM`
- Inference: `from optimum.neuron import NeuronModelForCausalLM`
- Avoid importing `neuronx_distributed` at module level in CLI code (see [optimum/commands/neuron/subcommands.py](optimum/commands/neuron/subcommands.py)).

## Porting Models from [NxDI](https://github.com/aws-neuron/neuronx-distributed-inference)
Use [NxDI](https://github.com/aws-neuron/neuronx-distributed-inference) for neuron-specific graph changes and [HF Transformers](https://github.com/huggingface/transformers) for base architecture.
- [NxDI](https://github.com/aws-neuron/neuronx-distributed-inference) reference: https://github.com/aws-neuron/neuronx-distributed-inference/tree/main/src/neuronx_distributed_inference/models
- HF reference: https://github.com/huggingface/transformers/tree/main/src/transformers/models
- Optimum target: [optimum/neuron/models/inference](optimum/neuron/models/inference)

For the full porting checklist and test guidance, see [optimum/neuron/models/inference/AGENTS.md](optimum/neuron/models/inference/AGENTS.md).

## Cache Management
Compiled models are cached to the HF Hub. Test helpers live in [tests/conftest.py](tests/conftest.py). Relevant env vars: `NEURON_CC_FLAGS`, `NEURON_COMPILE_CACHE_URL`, `NEURON_RT_VISIBLE_CORES`.
