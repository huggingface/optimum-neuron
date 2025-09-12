import os
from dataclasses import dataclass
from tempfile import TemporaryDirectory
from typing import Callable

import pytest
import torch
from nxd_testing import build_module, validate_accuracy
from transformers import AutoConfig, set_seed
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaMLP,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from transformers.models.mixtral.modeling_mixtral import (
    MixtralDecoderLayer,
    MixtralRotaryEmbedding,
)

from optimum.neuron.models.inference.backend.config import NxDNeuronConfig
from optimum.neuron.models.inference.backend.modules.rms_norm import NeuronRMSNorm
from optimum.neuron.models.inference.llama.modeling_llama import Llama3RotaryEmbedding as NeuronLlama3RotaryEmbedding
from optimum.neuron.models.inference.llama.modeling_llama import NeuronLlamaDecoderLayer, NeuronLlamaMLP
from optimum.neuron.models.inference.mixtral.modeling_mixtral import (
    NeuronMixtralDecoderLayer,
    convert_mixtral_to_neuron_state_dict,
)
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


@is_inferentia_test
@requires_neuronx
def test_rms_norm():
    set_seed(42)
    config_id = "unsloth/Llama-3.2-1B-Instruct"
    config = AutoConfig.from_pretrained(config_id)
    hidden_size = config.hidden_size
    seq_len = 4096
    # Initialize hidden states to random values between -3 and 3
    hidden_states = torch.rand(seq_len, hidden_size, dtype=torch.bfloat16) * 6 - 3
    inputs = [(hidden_states,)]

    example_inputs = [
        tuple([torch.zeros_like(input_element) for input_element in input_elements]) for input_elements in inputs
    ]

    # Create a cpu layer
    cpu_module = LlamaRMSNorm(hidden_size).to(torch.bfloat16)

    # Create a function to run the model
    neuron_module = build_module(NeuronRMSNorm, example_inputs, module_init_kwargs={"hidden_size": hidden_size})

    # Validate the accuracy of the model
    validate_accuracy(neuron_module, inputs, cpu_callable=cpu_module)


def _set_weights(module):
    """Set the weights of the module to random values"""
    state_dict = module.state_dict()
    for key in list(state_dict.keys()):
        state_dict[key] = torch.nn.Parameter(torch.rand_like(state_dict[key]) * 0.1)
    module.load_state_dict(state_dict)


@is_inferentia_test
@requires_neuronx
def test_llama_mlp():
    set_seed(42)
    config_id = "unsloth/Llama-3.2-1B-Instruct"
    config = AutoConfig.from_pretrained(config_id)
    hidden_size = config.hidden_size
    dtype = torch.float32
    seq_len = 2
    # Initialize hidden states to random values between -3 and 3
    hidden_states = torch.rand(seq_len, hidden_size, dtype=dtype) * 6 - 3
    inputs = [(hidden_states,)]

    example_inputs = [
        tuple([torch.zeros_like(input_element) for input_element in input_elements]) for input_elements in inputs
    ]

    cpu_module = LlamaMLP(config)
    _set_weights(cpu_module)

    # Create a function to run the model
    neuron_config = NxDNeuronConfig(
        batch_size=1,
        sequence_length=seq_len,
    )
    neuron_module = build_module(
        NeuronLlamaMLP, example_inputs, module_init_kwargs={"config": config, "neuron_config": neuron_config}
    )
    state_dict = cpu_module.state_dict()
    for key in state_dict.keys():
        state_dict[key] = state_dict[key].to(torch.bfloat16)

    start_rank_id = 0
    start_rank_tensor = torch.tensor([start_rank_id], dtype=torch.int32, device="cpu")

    weights = [state_dict]
    neuron_module.nxd_model.initialize(weights, start_rank_tensor)

    # Validate the accuracy of the model. Note that atol and rtol are relaxed considering calculations in the neuron
    # model are in bfloat16 and errors accumulate.
    validate_accuracy(
        neuron_module,
        inputs,
        cpu_callable=cpu_module,
        assert_close_kwargs={"atol": torch.finfo(torch.bfloat16).resolution, "rtol": 1e-1},
    )


@is_inferentia_test
@requires_neuronx
def test_llama_rotary_embedding():
    set_seed(42)
    config_id = "unsloth/Llama-3.2-1B-Instruct"
    config = AutoConfig.from_pretrained(config_id)
    hidden_size = config.hidden_size
    seq_len = 4096
    dtype = torch.float32

    hidden_size = config.hidden_size
    x = torch.rand(1, seq_len, hidden_size, dtype=dtype) - 0.5
    position_ids = torch.arange(seq_len, dtype=torch.int64).view(1, -1)
    inputs = [(x, position_ids)]

    example_inputs = [
        tuple([torch.zeros_like(input_element, dtype=input_element.dtype) for input_element in input_elements])
        for input_elements in inputs
    ]

    # Create a cpu layer
    cpu_module = LlamaRotaryEmbedding(config)

    def cpu_module_wrapper(x, position_ids):
        cos, sin = cpu_module(x, position_ids)
        return cos, sin

    # Create a neuron equivalent of the cpu module
    head_dim = config.hidden_size // config.num_attention_heads
    neuron_module = build_module(
        NeuronLlama3RotaryEmbedding,
        example_inputs,
        module_init_kwargs={
            "dim": head_dim,
            "max_position_embeddings": config.max_position_embeddings,
            "base": config.rope_theta,
            "factor": config.rope_scaling["factor"],
            "low_freq_factor": config.rope_scaling["low_freq_factor"],
            "high_freq_factor": config.rope_scaling["high_freq_factor"],
            "original_max_position_embeddings": config.rope_scaling["original_max_position_embeddings"],
        },
    )

    # Validate the accuracy of the model. Note that atol and rtol are relaxed considering calculations in the neuron
    # model are in bfloat16 and errors accumulate.
    validate_accuracy(
        neuron_module,
        inputs,
        cpu_callable=cpu_module_wrapper,
        assert_close_kwargs={"atol": torch.finfo(torch.bfloat16).resolution, "rtol": 1e-1},
    )


def _convert_state_dict_for_mixtral(state_dict, config, neuron_config):
    # Add "layers.0" prefix to the state dict keys to use "convert_mixtral_to_neuron_state_dict"
    keys = list(state_dict.keys())
    for key in keys:
        state_dict[f"layers.0.{key}"] = state_dict.pop(key)
    state_dict = convert_mixtral_to_neuron_state_dict(state_dict, config, neuron_config)
    # Remove "layers.0" prefix from the state dict keys
    keys = list(state_dict.keys())
    for key in keys:
        state_dict[key.replace("layers.0.", "")] = state_dict.pop(key)
    return state_dict


@dataclass
class DecoderLayerTestConfig:
    name: str
    config_id: str
    decoder_layer_cls: torch.nn.Module
    neuron_decoder_layer_cls: torch.nn.Module
    rotary_embedding_cls: torch.nn.Module
    neuron_init_kwargs: Callable[[AutoConfig], dict]
    state_dict_conversion_fn: Callable[[dict, AutoConfig, NxDNeuronConfig], dict]


DECODER_TESTS_CONFIGS = [
    DecoderLayerTestConfig(
        name="llama",
        config_id="unsloth/Llama-3.2-1B-Instruct",
        decoder_layer_cls=LlamaDecoderLayer,
        neuron_decoder_layer_cls=NeuronLlamaDecoderLayer,
        rotary_embedding_cls=LlamaRotaryEmbedding,
        neuron_init_kwargs=lambda config, neuron_config: {"config": config, "neuron_config": neuron_config},
        state_dict_conversion_fn=lambda state_dict, config, neuron_config: state_dict,
    ),
    DecoderLayerTestConfig(
        name="mixtral",
        config_id="dacorvo/Mixtral-tiny",
        decoder_layer_cls=MixtralDecoderLayer,
        neuron_decoder_layer_cls=NeuronMixtralDecoderLayer,
        rotary_embedding_cls=MixtralRotaryEmbedding,
        neuron_init_kwargs=lambda config, neuron_config: {
            "config": config,
            "neuron_config": neuron_config,
            "layer_idx": 0,
        },
        state_dict_conversion_fn=_convert_state_dict_for_mixtral,
    ),
]


@is_inferentia_test
@requires_neuronx
@pytest.mark.parametrize(
    "test_config", DECODER_TESTS_CONFIGS, ids=[test_config.name for test_config in DECODER_TESTS_CONFIGS]
)
def test_decoder_layer(test_config: DecoderLayerTestConfig):
    set_seed(42)
    config_id = test_config.config_id
    config = AutoConfig.from_pretrained(config_id)
    hidden_size = config.hidden_size
    dtype = torch.float32
    seq_len = 128
    # Initialize hidden states to random values between -3 and 3
    hidden_states = torch.rand(1, seq_len, hidden_size, dtype=dtype) * 6 - 3
    position_ids = torch.arange(seq_len, dtype=torch.int64).view(1, -1)
    attention_mask = torch.ones(1, 1, seq_len, seq_len).to(torch.bool).tril(diagonal=0)
    inputs = [(hidden_states, attention_mask, position_ids)]
    example_inputs = [
        tuple([torch.zeros_like(input_element, dtype=input_element.dtype) for input_element in input_elements])
        for input_elements in inputs
    ]

    example_inputs = [
        tuple([torch.zeros_like(input_element) for input_element in input_elements]) for input_elements in inputs
    ]

    config.num_hidden_layers = 1  # This is set to 1 to match the number of decoder layers tested here.
    config._attn_implementation = "eager"  # Force eager attention in cpu
    cpu_module = test_config.decoder_layer_cls(config, 0)
    _set_weights(cpu_module)

    rotary_embedding = test_config.rotary_embedding_cls(config)

    def cpu_module_wrapper(hidden_states, attention_mask, position_ids):
        # Position embeddings are computed outside the layer in transformers
        position_embeddings = rotary_embedding(hidden_states, position_ids)
        # CPU version requires the mask to be in float32
        attention_mask_cpu = torch.where(attention_mask, 0.0, torch.finfo(torch.float32).min)

        return cpu_module(
            hidden_states,
            attention_mask=attention_mask_cpu,
            position_ids=position_ids,
            use_cache=False,
            position_embeddings=position_embeddings,
        )

    # Make a wrapper to consider only hidden states and ignore other inputs
    class DecoderLayerWrapper(test_config.neuron_decoder_layer_cls):
        def forward(self, hidden_states, attention_mask, position_ids):
            hidden_states, _present_key_value, _cos_cache, _sin_cache = super().forward(
                hidden_states, attention_mask, position_ids
            )
            return hidden_states

    # Create a function to run the model
    neuron_config = NxDNeuronConfig(
        batch_size=1,
        sequence_length=seq_len,
    )
    state_dict = cpu_module.state_dict()
    state_dict = test_config.state_dict_conversion_fn(state_dict, config, neuron_config)

    with TemporaryDirectory() as tmpdir:
        # There are many quirks in the neuron attention implementation, so we will just save the state dict and load it
        # again to build the module.
        checkpoint_path = os.path.join(tmpdir, "checkpoint.pt")
        torch.save(state_dict, checkpoint_path)

        neuron_module = build_module(
            DecoderLayerWrapper,
            example_inputs,
            module_init_kwargs=test_config.neuron_init_kwargs(config, neuron_config),
            checkpoint_path=checkpoint_path,
        )

    # Validate the accuracy of the model
    validate_accuracy(
        neuron_module,
        inputs,
        cpu_callable=cpu_module_wrapper,
        assert_close_kwargs={"atol": torch.finfo(torch.bfloat16).resolution, "rtol": 1e-1},
    )
