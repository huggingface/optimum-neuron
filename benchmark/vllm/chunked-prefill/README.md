# Chunked Prefill Benchmark Results

Benchmark comparing standard context encoding vs chunked prefill on Llama 3.1 8B Instruct.

## Setup

| Parameter | Value |
|-----------|-------|
| Model | `meta-llama/Llama-3.1-8B-Instruct` |
| Batch size | 32 |
| Sequence length | 4096 |
| Tensor parallel | 8 |
| Chunk size | 1024 (for chunked configs) |
| Benchmark tool | [guidellm](https://github.com/neuralmagic/guidellm) |
| Concurrent users | 32 |

Two measurement runs are recorded. Both use TP=8 on Inferentia2, but they differ in
host size and software stack, so the comparison below is indicative rather than a
controlled A/B.

| Run | Date | Instance | Stack |
|-----|------|----------|-------|
| Original | 2026-03-03 | `inf2.48xlarge` (12 devices / 24 cores, 192 vCPU) | vllm 0.11.0, neuronx-cc 2.23 |
| Current | 2026-08-25 | `inf2.24xlarge` (4 devices / 8 cores, 96 vCPU) | vllm 0.16.0, neuronx-cc 2.26.6360.0, torch-neuronx 2.9.0.2.15, neuronx_distributed 0.19.28492 |

TP=8 occupies 8 NeuronCores in both cases, so the accelerator configuration is identical;
the current host has half the vCPUs of the original, which matters mainly for the
CPU-sampling configs.

## Results

Original run (2026-03-03):

| Config | Prefill | Sampling | Sync ITL (ms) | Throughput ITL (ms) | Throughput (tok/s) | CSV |
|--------|---------|----------|---------------|--------------------|--------------------|-----|
| A | Standard CE | On-device (ODS) | 24.5 | 59.1 | 426.2 | `std-ods.csv` |
| B | Standard CE | CPU | 48.2 | 123.8 | 177.0 | `std-cpu-sampling.csv` |
| C | Chunked | CPU | 47.3 | 119.2 | 189.9 | `chunked-cpu-sampling.csv` |
| D | Chunked | Hybrid ODS | 24.6 | 55.9 | 440.4 | `hybrid-ods.csv` |

Current run (2026-08-25):

| Config | Prefill | Sampling | Sync ITL (ms) | Throughput ITL (ms) | Throughput (tok/s) | CSV |
|--------|---------|----------|---------------|--------------------|--------------------|-----|
| A | Standard CE | On-device (ODS) | 23.9 | 59.7 | 428.5 | `std-ods-2026-08-25.csv` |
| B | Standard CE | CPU | 26.2 | 80.8 | 315.2 | `std-cpu-sampling-2026-08-25.csv` |
| C | Chunked | CPU | 26.2 | 87.4 | 291.6 | `chunked-cpu-sampling-2026-08-25.csv` |
| D | Chunked | Hybrid ODS | 24.0 | 57.2 | 427.2 | `hybrid-ods-2026-08-25.csv` |

Change in output throughput, current vs original:

| Config | Original (tok/s) | Current (tok/s) | Δ |
|--------|------------------|-----------------|-----|
| A | 426.2 | 428.5 | +0.5% |
| B | 177.0 | 315.2 | +78.1% |
| C | 189.9 | 291.6 | +53.5% |
| D | 440.4 | 427.2 | −3.0% |

## Analysis

**The ODS configurations are unchanged.** A and D land within ±3% of their original
numbers on every metric (A: +0.5% throughput, D: −3.0%). Whatever moved in the stack
did not move the on-device sampling path.

**The CPU-sampling configurations improved substantially.** B gained +78.1% throughput
(177.0 → 315.2 tok/s) and halved its synchronous ITL (48.2 → 26.2 ms); C gained +53.5%.
This happened on a host with *half* the vCPUs of the original run, so it is not
explained by more CPU being available. The specific cause has not been isolated —
several things changed together (vllm 0.11 → 0.16, neuronx-cc 2.23 → 2.26, and the
plugin rewrite) and no single-variable experiment has been run to attribute it.

**Two conclusions from the original run no longer hold.** They were correct for the
stack measured in March; they are not correct for the current one:

- *"ODS gives a 2.4x throughput improvement"* — the A/B ratio is now **1.36x**, not
  2.41x. ODS still wins, but by far less, because CPU sampling got much faster while
  ODS stayed flat.
- *"Chunked prefill is faster than standard CE at equal sampling (+7.3%)"* — now
  **reversed**: C is **−7.5%** against B (291.6 vs 315.2 tok/s), with higher throughput
  ITL (87.4 vs 80.8 ms).
- *"Hybrid ODS exceeds the production baseline by +3.3%"* — D and A are now effectively
  tied (**−0.3%**, 427.2 vs 428.5 tok/s), within run-to-run noise.

D remains the default configuration when `sequence_length > 1024`
(`modeling_utils.py:151`). These results do not argue against that default — D is still
at the top of the range — but they no longer show it beating standard CE with ODS.

**Caveat on the cross-run comparison.** The two runs differ in host size as well as
software, and each cell is a single measurement with no repeat runs, so small deltas
(the ±3% seen on A and D) should be read as noise. The large CPU-sampling deltas are
well outside that band.

## Reproducing

Config D is served directly, since chunked prefill with on-device sampling is the
default for `sequence_length > 1024`:

```shell
optimum-cli neuron serve -m meta-llama/Llama-3.1-8B-Instruct \
  --batch_size 32 --sequence_length 4096 --tensor_parallel_size 8 --port 8080
```

Configs A, B and C are not reachable from the CLI — it exposes no `--prefill_chunk_size`
or sampling flags, and `NeuronLlamaForCausalLM._get_neuron_config` hardcodes
`on_device_sampling=True`. They must be exported through the Python API first, then
served from the resulting directory:

```python
from transformers import AutoTokenizer
from optimum.neuron import NeuronModelForCausalLM
from optimum.neuron.models.inference.backend.config import NxDNeuronConfig

# A: prefill_chunk_size=0,    on_device_sampling=True
# B: prefill_chunk_size=0,    on_device_sampling=False
# C: prefill_chunk_size=1024, on_device_sampling=False
neuron_config = NxDNeuronConfig(
    checkpoint_id="meta-llama/Llama-3.1-8B-Instruct",
    checkpoint_revision=revision,
    batch_size=32,
    sequence_length=4096,
    tp_degree=8,
    torch_dtype=torch.bfloat16,
    target="trn1",
    on_device_sampling=...,
    fused_qkv=True,
    continuous_batching=True,
    prefill_chunk_size=...,
)
model = NeuronModelForCausalLM.export(model_id, neuron_config, revision=revision)
model.save_pretrained(out_dir)
# Required: save_pretrained does not write the tokenizer, and `neuron serve` needs it.
AutoTokenizer.from_pretrained(model_id, revision=revision).save_pretrained(out_dir)
```

Serve an exported directory with `--served_model_name` so the API reports the model id
the benchmark expects rather than the local path:

```shell
optimum-cli neuron serve -m <out_dir> \
  --served_model_name meta-llama/Llama-3.1-8B-Instruct --port 8080
```

Then, for each config:

```shell
../performance.sh meta-llama/Llama-3.1-8B-Instruct 32
python ../generate_csv.py --dir .
```

## CSV Columns

All CSVs share the same schema with two rows: `synchronous` (single-user latency) and
`throughput` (32 concurrent users):

- `model_id` — HuggingFace model identifier
- `Date` — benchmark timestamp
- `Input type` — `synchronous` or `throughput`
- `Requests per Second` — sustained request rate
- `Request Latency (s)` — end-to-end request latency
- `Time-to-first-token (ms)` — time to first generated token
- `Inter Token Latency (ms)` — average time between consecutive tokens
- `Output Token Throughput (t/s)` — total tokens generated per second
