import os
from dataclasses import dataclass
from tempfile import TemporaryDirectory

import pytest
import torch
from nxd_testing import build_module, validate_accuracy
from transformers import AutoConfig, set_seed
from transformers.models.llama.modeling_llama import (
    LlamaAttention,
    LlamaRotaryEmbedding,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralAttention,
    MixtralRotaryEmbedding,
)

from optimum.neuron.models.inference.backend.config import NxDNeuronConfig
from optimum.neuron.models.inference.llama.modeling_llama import (
    NeuronLlamaAttention,
)
from optimum.neuron.models.inference.mixtral.modeling_mixtral import (
    NeuronMixtralAttention,
)
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


def _set_weights(module):
    """Set the weights of the module to random values"""
    state_dict = module.state_dict()
    for key in list(state_dict.keys()):
        state_dict[key] = torch.nn.Parameter(torch.rand_like(state_dict[key]) * 0.1)
    module.load_state_dict(state_dict)


# These wrappers are necessary to start the attention layer, because when building the traced model,
# The GroupQueryAttention_QKV class has a preshard_hook that expects the weights names in the state
# dict to be like "layers.0.self_attn.qkv_proj.q_proj.weight", otherwise it will do odd stuff.
class AttentionWrapper(torch.nn.Module):
    def __init__(self, attention_cls, config, neuron_config):
        super().__init__()
        self.self_attn = attention_cls(config, neuron_config)

    def forward(self, hidden_states, attention_mask, position_ids):
        return self.self_attn(hidden_states, attention_mask, position_ids)


class AttentionModelWrapper(torch.nn.Module):
    def __init__(self, attention_cls, config, neuron_config):
        super().__init__()
        # we add only one layer
        self.layers = torch.nn.ModuleList([AttentionWrapper(attention_cls, config, neuron_config)])

    def forward(self, hidden_states, attention_mask, position_ids):
        layer = self.layers[0]
        attn_output, _past_key_value, _cos_cache, _sin_cache = layer(hidden_states, attention_mask, position_ids)
        return attn_output


@dataclass
class AttentionTestConfig:
    name: str
    config_id: str
    attention_cls: torch.nn.Module
    rotary_embedding_cls: torch.nn.Module
    neuron_attention_cls: torch.nn.Module


CONFIGS = [
    AttentionTestConfig(
        name="llama",
        config_id="llamafactory/tiny-random-Llama-3",
        attention_cls=LlamaAttention,
        rotary_embedding_cls=LlamaRotaryEmbedding,
        neuron_attention_cls=NeuronLlamaAttention,
    ),
    AttentionTestConfig(
        name="mixtral",
        config_id="dacorvo/Mixtral-tiny",
        attention_cls=MixtralAttention,
        rotary_embedding_cls=MixtralRotaryEmbedding,
        neuron_attention_cls=NeuronMixtralAttention,
    ),
]


@is_inferentia_test
@requires_neuronx
@pytest.mark.parametrize("test_config", CONFIGS, ids=[test_config.name for test_config in CONFIGS])
def test_attention_prefill(test_config: AttentionTestConfig):
    set_seed(42)
    checkpoint = test_config.config_id
    config = AutoConfig.from_pretrained(checkpoint)
    hidden_size = config.hidden_size
    dtype = torch.float32
    seq_len = 2048
    hidden_states = torch.rand(1, seq_len, hidden_size, dtype=dtype) * 6 - 3
    position_ids = torch.arange(seq_len, dtype=torch.int64).view(1, -1)
    attention_mask = torch.ones(1, 1, seq_len, seq_len).to(torch.bool).tril(diagonal=0)
    inputs = [(hidden_states, attention_mask, position_ids)]
    example_inputs = [
        tuple([torch.zeros_like(input_element, dtype=input_element.dtype) for input_element in input_elements])
        for input_elements in inputs
    ]
    cpu_module = test_config.attention_cls(config, layer_idx=0)
    cpu_module = cpu_module.to(dtype=dtype)
    cpu_module.config._attn_implementation = "eager"  # Force eager attention in cpu
    _set_weights(cpu_module)

    rotary_embedding = test_config.rotary_embedding_cls(config)

    def cpu_module_wrapper(hidden_states, attention_mask, position_ids):
        # This is done outside the attention in transformers
        position_embeddings = rotary_embedding(hidden_states, position_ids)
        # CPU version requires the mask to be in float32
        attention_mask_cpu = torch.where(attention_mask, 0.0, torch.finfo(torch.float32).min)

        attn_output, _attn_weights = cpu_module(
            hidden_states, position_embeddings=position_embeddings, attention_mask=attention_mask_cpu
        )
        # attn_output = attn_output.to(target_dtype)
        return attn_output

    # Make the state dict compatible with the AttentionModelWrapper
    state_dict = cpu_module.state_dict()
    keys = list(state_dict.keys())
    for key in keys:
        state_dict[f"layers.0.self_attn.{key}"] = state_dict.pop(key)

    # Create a function to run the model
    neuron_config = NxDNeuronConfig(
        batch_size=1,
        sequence_length=seq_len,
    )
    with TemporaryDirectory() as tmpdir:
        # There are many quirks in the neuron attention implementation, so we will just save the state dict and load it
        # again to build the module.
        checkpoint_path = os.path.join(tmpdir, "checkpoint.pt")
        torch.save(state_dict, checkpoint_path)

        neuron_module = build_module(
            AttentionModelWrapper,
            example_inputs,
            module_init_kwargs={
                "attention_cls": test_config.neuron_attention_cls,
                "config": config,
                "neuron_config": neuron_config,
            },
            checkpoint_path=checkpoint_path,
        )

    # Validate the accuracy of the model
    validate_accuracy(
        neuron_module,
        inputs,
        cpu_callable=cpu_module_wrapper,
        assert_close_kwargs={"atol": torch.finfo(torch.bfloat16).resolution, "rtol": 1e-1},
    )
