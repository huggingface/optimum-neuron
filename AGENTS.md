# Optimum Neuron Agent Guide (Root)

Optimum Neuron bridges Hugging Face libraries (Transformers, Diffusers, PEFT) with AWS Trainium/Inferentia accelerators. Use this file for project-wide guidance and the model-specific guides below:
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

### Virtual Environment (Required)

Always activate the venv before any command; commands fail without it.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[neuronx,tests]"
```

### Testing (Neuron hardware required)

Most tests require real Neuron hardware and will skip or fail on CPU-only machines.

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

Use separate processes for export and load to avoid Neuron device conflicts.

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

## CI/CD Workflows (Summary)

All test workflows follow the same pattern:
1. Checkout code
2. Install Neuronx runtime (via `.github/actions/install_neuronx_runtime`)
3. Prepare venv `aws_neuron_venv_pytorch` (via `.github/actions/prepare_venv`)
4. Install `optimum-neuron[neuronx,tests]` (via `.github/actions/install_optimum_neuron`)
5. Run pytest in the venv

## Runtime Pitfalls

- Static shapes: runtime input shapes must match compiled shapes.
- Export and load in separate processes to avoid device conflicts.
- Neuron runtime does not release devices reliably within the same process.
- Decoder graph changes require cache prune when using the fixtures defined under `tests/fixtures/export_models.py`: `python tools/prune_test_models.py`.

## Environment Variables

- `HF_TOKEN`: Required for hub access in tests.
- `NEURON_CC_FLAGS="--model-type transformer"`: Required for training compilation.
- `NEURON_RT_VISIBLE_CORES`: Control visible NeuronCores.

## Validation Checklist (Before PR)

1. Activate venv: `source .venv/bin/activate`.
2. Style check: `make style_check` (or `make style`).
3. Run relevant tests:
  - CPU export logic: `pytest tests/exporters/`
  - INF2 decoder: `pytest tests/decoder/`
  - TRN1 training: `pytest -m "is_trainium_test" tests/training/`
4. Check model-specific AGENTS.md if you touched a model directory.

## Troubleshooting

- `ruff: command not found`: activate venv first.
- `No module named 'neuronx_distributed'`: install extras with `pip install -e ".[neuronx]"`.
- Tests failing on CPU: expected for most Neuron tests.
- Compilation timeout: large models take 30-60 min, use `--timeout 0`.

Trust these instructions and only search for more context if something is missing or incorrect.
