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
# Adapted from https://github.com/aws-neuron/neuronx-distributed-inference/blob/main/src/neuronx_distributed_inference/models/gpt_oss/modeling_gpt_oss.py
"""PyTorch GPT-OSS model for NXD inference."""

import gc
import logging
import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)

# Adapted imports for optimum-neuron (following Mixtral pattern)
from ..backend.config import NxDNeuronConfig
from ..backend.modules.attention.attention_base import NeuronAttentionBase
from ..backend.modules.decoder import NxDDecoderModelForCausalLM, NxDModelForCausalLM
from ..backend.modules.moe import initialize_moe_module
from ..backend.modules.rms_norm import NeuronRMSNorm


logger = logging.getLogger(__name__)

# Copied from GPT_OSS repo
FP4_VALUES = [
    +0.0,
    +0.5,
    +1.0,
    +1.5,
    +2.0,
    +3.0,
    +4.0,
    +6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def convert_moe_packed_tensors(
    blocks,
    scales,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 32768 * 1024,
) -> torch.Tensor:
    """Convert MXFP4 packed tensors to full precision (from original GPT-OSS)."""
    import math

    scales = scales.to(torch.int32) - 127
    assert tuple(blocks.shape[:-1]) == tuple(scales.shape), (
        f"{tuple(blocks.shape[:-1])} does not match {tuple(scales.shape)}"
    )

    lut = torch.tensor(FP4_VALUES, dtype=dtype, device=blocks.device)
    *prefix_shape, G, B = blocks.shape
    rows_total = math.prod(prefix_shape) * G

    blocks = blocks.reshape(rows_total, B)
    scales = scales.reshape(rows_total, 1)

    out = torch.empty(rows_total, B * 2, dtype=dtype, device=blocks.device)

    for r0 in range(0, rows_total, rows_per_chunk):
        r1 = min(r0 + rows_per_chunk, rows_total)
        blk = blocks[r0:r1]
        exp = scales[r0:r1]

        # nibble indices -> int64
        idx_lo = (blk & 0x0F).to(torch.long)
        idx_hi = (blk >> 4).to(torch.long)

        sub = out[r0:r1]
        sub[:, 0::2] = lut[idx_lo]
        sub[:, 1::2] = lut[idx_hi]

        torch.ldexp(sub, exp, out=sub)
        del idx_lo, idx_hi, blk, exp, sub

    out = out.reshape(*prefix_shape, G, B * 2).view(*prefix_shape, G * B * 2)
    del blocks, scales, lut
    return out


def convert_gate_up_proj(tensor: torch.Tensor, is_bias: bool = False) -> torch.Tensor:
    """
    Convert the gate_up_proj tensor from GptOss reference format to NxDI format.
    Reference format: E, 2xI, H with interleaved gate and up projection
    NxDI format: E, H, 2xI with chunked gate and up project
    """
    gate, up_proj = tensor[:, ::2, ...], tensor[:, 1::2, ...]
    gate_up_proj = torch.cat((gate, up_proj), dim=1)
    return gate_up_proj if is_bias else gate_up_proj.transpose(1, 2)


class GptOssRotaryEmbedding(nn.Module):
    """GPT-OSS RoPE with YaRN scaling (from original implementation)."""

    def __init__(
        self,
        dim: int,
        base: int = 10000,
        initial_context_length: int = 4096,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.register_buffer("inv_freq", None, persistent=False)
        self.concentration = None

    def get_inv_freqs_and_concentration(self, device: Optional[torch.device] = None) -> torch.Tensor:
        """See YaRN paper: https://arxiv.org/abs/2309.00071"""
        freq = self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float, device=device) / self.dim)
        if self.scaling_factor > 1.0:
            concentration = 0.1 * math.log(self.scaling_factor) + 1.0  # YaRN concentration

            d_half = self.dim / 2
            # NTK by parts
            low = d_half * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi)) / math.log(self.base)
            high = (
                d_half * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi)) / math.log(self.base)
            )
            assert 0 < low < high < d_half - 1

            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            ramp = (torch.arange(d_half, dtype=torch.float32, device=freq.device) - low) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return inv_freq, concentration

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.inv_freq is None or self.concentration is None:
            self.inv_freq, self.concentration = self.get_inv_freqs_and_concentration(x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.concentration
        sin = emb.sin() * self.concentration
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class NeuronGptOssAttention(NeuronAttentionBase):
    """GPT-OSS attention (from original, using NeuronAttentionBase from optimum-neuron)."""

    def __init__(self, config, neuron_config: NxDNeuronConfig):
        super().__init__(config=config, neuron_config=neuron_config)

        # Get rope_scaling attributes with defaults
        rope_scaling_factor = 1.0
        ntk_alpha = 1.0
        ntk_beta = 32.0
        initial_context_length = getattr(config, "max_position_embeddings", 4096)

        if hasattr(config, "rope_scaling") and config.rope_scaling:
            rope_scaling_factor = config.rope_scaling.get("factor", 1.0)
            ntk_alpha = config.rope_scaling.get("beta_slow", 1.0)
            ntk_beta = config.rope_scaling.get("beta_fast", 32.0)

        head_dim = getattr(config, "head_dim", None) or (config.hidden_size // config.num_attention_heads)
        self.rotary_emb = GptOssRotaryEmbedding(
            dim=head_dim,
            base=config.rope_theta,
            initial_context_length=initial_context_length,
            scaling_factor=rope_scaling_factor,
            ntk_alpha=ntk_alpha,
            ntk_beta=ntk_beta,
        )


class NeuronGptOssDecoderLayer(nn.Module):
    """GPT-OSS decoder layer (from original, adapted for optimum-neuron)."""

    def __init__(self, config, neuron_config: NxDNeuronConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size

        self.self_attn = NeuronGptOssAttention(config, neuron_config)

        self.post_attention_layernorm = NeuronRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.feed_forward = initialize_moe_module(
            neuron_config=neuron_config,
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            normalize_top_k_affinities=False,
        )

        self.input_layernorm = NeuronRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        cos_cache: Optional[torch.Tensor] = None,
        sin_cache: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        residual = hidden_states.clone()

        # RMSNorm
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states.clone()

        # MoE
        hidden_states = self.feed_forward(hidden_states)[0]
        hidden_states = residual + hidden_states

        outputs = (hidden_states, present_key_value, cos_cache, sin_cache)
        return outputs


class NxDGptOssModel(NxDDecoderModelForCausalLM):
    """Base GPT-OSS model (adapted for optimum-neuron following Mixtral pattern)."""

    def __init__(self, config, neuron_config: NxDNeuronConfig):
        super().__init__(config, neuron_config)

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            dtype=neuron_config.torch_dtype,
            shard_across_embedding=True,
        )
        self.layers = nn.ModuleList(
            [
                NeuronGptOssDecoderLayer(config, neuron_config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = NeuronRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=not neuron_config.on_device_sampling,
            bias=False,
        )


class GptOssNxDModelForCausalLM(NxDModelForCausalLM):
    """GPT-OSS model for causal LM (adapted for optimum-neuron following Mixtral pattern)."""

    _model_cls = NxDGptOssModel

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config, neuron_config: NxDNeuronConfig) -> dict:
        """Convert HuggingFace format state dict to Neuron format (from original)."""
        num_layers = config.num_hidden_layers

        # Remove "model." prefix
        state_dict_no_prefix = {}
        for key in list(state_dict.keys()):
            if key.startswith("model."):
                new_key = key[6:]  # Remove "model."
                state_dict_no_prefix[new_key] = state_dict.pop(key)
            else:
                state_dict_no_prefix[key] = state_dict.pop(key)

        state_dict = state_dict_no_prefix

        # Process each layer
        for layer in range(num_layers):
            # Router
            if f"layers.{layer}.mlp.router.weight" in state_dict:
                state_dict[f"layers.{layer}.feed_forward.router.linear_router.weight"] = state_dict[
                    f"layers.{layer}.mlp.router.weight"
                ]
                del state_dict[f"layers.{layer}.mlp.router.weight"]

            if f"layers.{layer}.mlp.router.bias" in state_dict:
                state_dict[f"layers.{layer}.feed_forward.router.linear_router.bias"] = state_dict[
                    f"layers.{layer}.mlp.router.bias"
                ]
                del state_dict[f"layers.{layer}.mlp.router.bias"]

            # Process MoE expert weights (dequantize MXFP4 and convert format)
            for proj in ["down_proj", "gate_up_proj"]:
                blocks_key = f"layers.{layer}.mlp.experts.{proj}_blocks"
                scales_key = f"layers.{layer}.mlp.experts.{proj}_scales"
                bias_key = f"layers.{layer}.mlp.experts.{proj}_bias"

                if blocks_key in state_dict and scales_key in state_dict:
                    # Dequantize MXFP4
                    dequantized_weights = convert_moe_packed_tensors(state_dict[blocks_key], state_dict[scales_key])

                    if proj == "gate_up_proj":
                        state_dict[f"layers.{layer}.feed_forward.expert_mlps.mlp_op.{proj}.weight"] = (
                            convert_gate_up_proj(dequantized_weights)
                        )
                        if bias_key in state_dict:
                            state_dict[f"layers.{layer}.feed_forward.expert_mlps.mlp_op.{proj}.bias"] = (
                                convert_gate_up_proj(state_dict[bias_key], is_bias=True)
                            )
                    else:
                        state_dict[f"layers.{layer}.feed_forward.expert_mlps.mlp_op.{proj}.weight"] = (
                            dequantized_weights.transpose(1, 2)
                        )
                        if bias_key in state_dict:
                            state_dict[f"layers.{layer}.feed_forward.expert_mlps.mlp_op.{proj}.bias"] = state_dict[
                                bias_key
                            ]

                    del state_dict[blocks_key]
                    del state_dict[scales_key]
                    if bias_key in state_dict:
                        del state_dict[bias_key]

        # Cleanup any remaining .mlp.* keys not processed
        keys_to_delete = [k for k in state_dict.keys() if ".mlp.experts." in k or ".mlp.router." in k]
        for key in keys_to_delete:
            del state_dict[key]

        gc.collect()
        return state_dict

    @staticmethod
    def _get_neuron_config(
        checkpoint_id: str,
        checkpoint_revision: str,
        instance_type: str,
        batch_size: int,
        sequence_length: int,
        tensor_parallel_size: int,
        dtype: str,
    ) -> NxDNeuronConfig:
        """Factory method to create NxDNeuronConfig (following Mixtral pattern)."""
        return NxDNeuronConfig(
            checkpoint_id=checkpoint_id,
            checkpoint_revision=checkpoint_revision,
            batch_size=batch_size,
            sequence_length=sequence_length,
            tp_degree=tensor_parallel_size,
            torch_dtype=dtype,
            target=instance_type,
        )

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """Update state dict to handle tied weights (copy embed_tokens to lm_head)."""
        if "embed_tokens.weight" in state_dict and "lm_head.weight" not in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.weight"].clone()
