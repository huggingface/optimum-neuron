# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Adapted from https://github.com/aws-neuron/neuronx-distributed-inference/blob/9993358ce052fd7a1bb4a7497a6318aac36ed95c/src/neuronx_distributed_inference/models/llama/modeling_llama.py
"""PyTorch LLaMA model for NXD inference."""

import gc
import logging
import math
from typing import Type

import torch
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from torch import nn
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaRotaryEmbedding

from ..backend.config import NxDNeuronConfig  # noqa: E402
from ..backend.modules.attention.attention_base import NeuronAttentionBase
from ..backend.modules.attention.utils import (
    RotaryEmbedding,
)
from ..backend.modules.custom_calls import CustomRMSNorm
from ..backend.modules.decoder import NxDDecoderModel, NxDModelForCausalLM


logger = logging.getLogger("Neuron")


def convert_state_dict_to_fused_qkv(llama_state_dict, cfg: LlamaConfig):
    """
    This function concats the qkv weights to a Wqkv weight for fusedqkv, and deletes the qkv weights.
    """
    for l in range(cfg.num_hidden_layers):  # noqa: E741
        llama_state_dict[f"layers.{l}.self_attn.Wqkv.weight"] = torch.cat(
            [
                llama_state_dict[f"layers.{l}.self_attn.q_proj.weight"],
                llama_state_dict[f"layers.{l}.self_attn.k_proj.weight"],
                llama_state_dict[f"layers.{l}.self_attn.v_proj.weight"],
            ],
        )
        del llama_state_dict[f"layers.{l}.self_attn.q_proj.weight"]
        del llama_state_dict[f"layers.{l}.self_attn.k_proj.weight"]
        del llama_state_dict[f"layers.{l}.self_attn.v_proj.weight"]

    gc.collect()

    return llama_state_dict


class NeuronLlamaMLP(nn.Module):
    """
    This class just replace the linear layers (gate_proj, up_proj and down_proj) with column and row parallel layers
    """

    def __init__(self, config: LlamaConfig, neuron_config: NxDNeuronConfig):
        super().__init__()
        self.tp_degree = neuron_config.tp_degree
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]

        self.rms_norm_eps = config.rms_norm_eps
        self.logical_nc_config = neuron_config.logical_nc_config
        mlp_bias = getattr(config, "mlp_bias", False)
        self.gate_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=mlp_bias,
            gather_output=False,
            dtype=neuron_config.torch_dtype,
            pad=True,
        )
        self.up_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=mlp_bias,
            gather_output=False,
            dtype=neuron_config.torch_dtype,
            pad=True,
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=mlp_bias,
            input_is_parallel=True,
            dtype=neuron_config.torch_dtype,
            pad=True,
            reduce_dtype=neuron_config.torch_dtype,
        )

    def forward(self, x, rmsnorm=None, residual=None):
        gate_proj_output = self.gate_proj(x)
        up_proj_output = self.up_proj(x)
        down_proj_input = self.act_fn(gate_proj_output) * up_proj_output
        output = self.down_proj(down_proj_input)
        return output


class NeuronLlamaAttention(NeuronAttentionBase):
    """
    The only difference with the NeuronAttentionBase is the definition of the Llama rotary embedding
    """

    def __init__(
        self,
        config: LlamaConfig,
        neuron_config: NxDNeuronConfig,
        qkv_proj_bias: bool | None = False,
        o_proj_bias: bool | None = False,
        qk_scale: float | None = None,
    ):
        super().__init__(
            config, neuron_config, qkv_proj_bias=qkv_proj_bias, o_proj_bias=o_proj_bias, qk_scale=qk_scale
        )
        head_dim = config.hidden_size // config.num_attention_heads
        if not hasattr(config, "rope_scaling") or config.rope_scaling is None:
            self.rotary_emb = RotaryEmbedding(
                head_dim,
                max_position_embeddings=config.max_position_embeddings,
                base=config.rope_theta,
            )
        else:
            rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type", None))
            if rope_type == "llama3":
                self.rotary_emb = Llama3RotaryEmbedding(
                    dim=head_dim,
                    max_position_embeddings=config.max_position_embeddings,
                    base=config.rope_theta,
                    factor=config.rope_scaling["factor"],
                    low_freq_factor=config.rope_scaling["low_freq_factor"],
                    high_freq_factor=config.rope_scaling["high_freq_factor"],
                    original_max_position_embeddings=config.rope_scaling["original_max_position_embeddings"],
                )
            else:
                # LlamaRotaryEmbedding automatically chooses the correct scaling type from config.
                # Warning: The HF implementation may have precision issues when run on Neuron.
                # We include it here for compatibility with other scaling types.
                self.rotary_emb = LlamaRotaryEmbedding(config)


# TODO: Modularize RotaryEmbedding. See how HF transformers does it in 4.43.
class Llama3RotaryEmbedding(nn.Module):
    """
    Adapted from Llama 4.43 impl
    * https://github.com/huggingface/transformers/blob/v4.43.4/src/transformers/models/llama/modeling_llama.py#L78
    * https://github.com/huggingface/transformers/blob/v4.43.4/src/transformers/modeling_rope_utils.py#L345

    This implementation ensures inv_freq is calculated and stored in fp32.
    """

    def __init__(
        self,
        dim,
        max_position_embeddings=131072,
        base=500000.0,
        factor=8.0,
        low_freq_factor=1.0,
        high_freq_factor=4.0,
        original_max_position_embeddings=8192,
    ):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.factor = factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.old_context_len = original_max_position_embeddings
        self.register_buffer("inv_freq", None, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.inv_freq is None:
            inv_freq = 1.0 / (
                self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(x.device) / self.dim)
            )

            low_freq_wavelen = self.old_context_len / self.low_freq_factor
            high_freq_wavelen = self.old_context_len / self.high_freq_factor
            new_freqs = []
            for freq in inv_freq:
                wavelen = 2 * math.pi / freq
                if wavelen < high_freq_wavelen:
                    new_freqs.append(freq)
                elif wavelen > low_freq_wavelen:
                    new_freqs.append(freq / self.factor)
                else:
                    assert low_freq_wavelen != high_freq_wavelen
                    smooth = (self.old_context_len / wavelen - self.low_freq_factor) / (
                        self.high_freq_factor - self.low_freq_factor
                    )
                    new_freqs.append((1 - smooth) * freq / self.factor + smooth * freq)
            self.inv_freq = torch.tensor(new_freqs, dtype=inv_freq.dtype, device=inv_freq.device)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        with torch.autocast(device_type=x.device.type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class NeuronLlamaDecoderLayer(nn.Module):
    """
    Just replace the attention with the NXD version, and MLP with the NXD version
    """

    def __init__(self, config: LlamaConfig, neuron_config: NxDNeuronConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronLlamaAttention(config, neuron_config)
        self.mlp = NeuronLlamaMLP(config, neuron_config)
        logger.debug(
            f"Instantiating RMSNorm modules with hidden size {config.hidden_size} and EPS {config.rms_norm_eps}"
        )
        self.input_layernorm = CustomRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = CustomRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.config = config

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            rmsnorm=self.input_layernorm,
            **kwargs,
        )

        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(
            hidden_states,
            rmsnorm=self.post_attention_layernorm,
        )

        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache)
        return outputs


class NxDLlamaModel(NxDDecoderModel):
    """
    The neuron version of the LlamaModel
    """

    def __init__(self, config: LlamaConfig, neuron_config: NxDNeuronConfig):
        super().__init__(config, neuron_config)

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            dtype=neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
        )

        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=not neuron_config.on_device_sampling,
            bias=False,
            pad=True,
        )

        self.layers = nn.ModuleList(
            [NeuronLlamaDecoderLayer(config, neuron_config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = CustomRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class LlamaNxDModelForCausalLM(NxDModelForCausalLM):
    """
    This class extends LlamaForCausalLM create traceable
    blocks for Neuron.

    Args:
        LlamaForCausalLM (_type_): _description_
    """

    _model_cls = NxDLlamaModel

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: LlamaConfig, neuron_config: NxDNeuronConfig) -> dict:
        """This function should be over-ridden in child classes as needed"""
        if neuron_config.fused_qkv:
            state_dict = convert_state_dict_to_fused_qkv(state_dict, config)

        # to facilitate rank usage in attention
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        # to facilitate rank usage in base model
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)
        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()

    @classmethod
    def get_neuron_config_cls(cls) -> Type[NxDNeuronConfig]:
        return NxDNeuronConfig

    @classmethod
    def _get_neuron_config(
        cls,
        checkpoint_id: str,
        checkpoint_revision: str,
        batch_size: int,
        sequence_length: int,
        tensor_parallel_size: int,
        auto_cast_type: str,
    ):
        continuous_batching = (batch_size > 1) if batch_size else False
        return NxDNeuronConfig(
            checkpoint_id=checkpoint_id,
            checkpoint_revision=checkpoint_revision,
            batch_size=batch_size,
            sequence_length=sequence_length,
            tp_degree=tensor_parallel_size,
            torch_dtype=auto_cast_type,
            on_device_sampling=True,
            fused_qkv=True,
            continuous_batching=continuous_batching,
        )
