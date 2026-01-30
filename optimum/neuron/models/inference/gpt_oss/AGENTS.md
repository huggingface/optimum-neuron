# GPT-OSS Porting Guide for NxD Inference

For project-wide guidance, see [../AGENTS.md](../AGENTS.md) and [../../../../AGENTS.md](../../../../AGENTS.md).

## Development Workflow

1. **Analysis** read reference implementations from HF and NxDI
2. **Creation** create `modeling_gpt_oss.py` (see porting check list)
3. **Test module**: Verify individual components work
4. **Recompile model**: `source .venv/bin/activate && optimum-cli export neuron ...`
5. **Test loading**: `NeuronModelForCausalLM.from_pretrained("...")`
6. **Verify generation**: `source .venv/bin/activate && python examples/inference/text-generation/generation.py ...`

## References

**Optimum-Neuron Guidelines**:
- Project: [../../../../AGENTS.md](../../../../AGENTS.md)
- Inference models: [../AGENTS.md](../AGENTS.md)
- Mixtral reference: [../mixtral/AGENTS.md](../mixtral/AGENTS.md)

**GPT-OSS Model**:
- HF transformers modeling: https://github.com/huggingface/transformers/tree/main/src/transformers/models/gpt_oss
- HF Hub model configuration and weights (for tests): https://huggingface.co/openai/gpt-oss-20b
  - Config: 24 layers, 2880 hidden, 32 experts, top-4 routing

**NxD Inference Framework neuron port**:
- NxDI: https://github.com/aws-neuron/neuronx-distributed-inference/tree/main/src/neuronx_distributed_inference/models/gpt_oss
- neuronx_distributed: Tensor parallelism, expert sharding

**YaRN Scaling**:
- Paper: https://arxiv.org/abs/2309.00071
- Implementation: YaRN concentration + NTK interpolation/extrapolation

## Known Considerations

### 1. MXFP4 Dequantization

**Observation**: Expert weights are 4-bit quantized in HF checkpoint.

**Handling**: Automatic dequantization in `convert_moe_packed_tensors()` during load.

Try to use directly the conversion method imported from `transformers`. Be careful as the
weights last dimensions must be transposed for some weights (check dimensions and NxDI code).

### 2. Head Dimension Mismatch Prevention

**Pattern**: Always use explicit `config.head_dim` when available.

```python
head_dim = getattr(config, "head_dim", None) or \
           (config.hidden_size // config.num_attention_heads)
```

Without this fallback, division `2880 / 64` would give 45 (incorrect).

### 3. Hidden and intermediate size padding

The NxDI port supports hidden and intermediate size padding: do not include
that when porting the model to the current repository.


## Porting Checklist

- Follow NxDI reference implementation
- Use HF Transformers as base architecture
- Replace nn.Linear/Embedding with TP-aware parallel layers
- Replace HF attention with NeuronAttentionBase
- Integrate KV cache management
- Implement state dict conversion for model-specific formats
- Create per-module implementations (Rotary, Attention, MoE, Decoder)
- Register model in auto_models.py
- Add proper package initialization
- Document porting decisions and known considerations
- Verify successful model loading and weight sharding

## Questions?

For issues or questions:
1. Review AGENTS.md files at each level (root, inference, gpt_oss)
2. Check NxDI reference implementation for architecture details
3. Consult HF Transformers documentation for model configuration
4. If a similar class of method is defined in both HF and NxDI, import it from HF and not NxDI
5. Run module-level tests to isolate issues
6. Enable debug logging for state dict conversion
