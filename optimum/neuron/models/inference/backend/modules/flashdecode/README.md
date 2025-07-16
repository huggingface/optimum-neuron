# Flash Decoding

Flash decoding supports long context inference by reducing KV cache memory. This is done by sharding and distributing
cache storage instead of replicating it on multiple devices (cores).

Flash decoding lives in the context of GQA (group query attention). This means it is a feature on top of GQA and not
traditional MHA (multi-head attention). In GQA, we replicate the KV cache in the devices within the same KV group.
Now, instead of replicating, we shard the KV and distribute them to each device in the group. To accommodate this setup, we modify the attention computation as follows:
1) Gather all query heads in the group,
2) Compute partial softmax on each device,
3) Reduce-scatter in the end to get the complete result.

## User guide
Simply add `flash_decoding_enabled` to be True.
- `generation_demo`: set it inside `examples/generation_demo.py` when constructing `NeuronConfig`.
- `inference_demo`: add the flag `--flash-decoding-enabled`.

## Development

We onboarded LLAMA onto this feature and use it for reference. To enable flash decoding for a new model (dense LLM for now), you need modify below:
- Override `add_derived_config` to set `num_cores_per_group` based on model configuration. See LLAMA example in `src/neuronx_distributed_inference/models/llama/modeling_llama.py`
- Modify `convert_hf_to_neuron_state_dict` to facilitate rank usage in base model. See LLAMA example in `src/neuronx_distributed_inference/models/llama/modeling_llama.py`
