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
from neuronx_distributed.parallel_layers.mappings import (
    gather_from_sequence_parallel_region,
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from neuronxcc.nki._private_kernels.mlp import (
    mlp_fused_add_isa_kernel,
    mlp_isa_kernel,
)
from neuronxcc.nki.language import nc
from torch import nn
from torch_neuronx.xla_impl.ops import nki_jit
from transformers.activations import ACT2FN
from transformers.models.llama.modeling_llama import LlamaConfig, LlamaRotaryEmbedding

from ..backend.config import NxDNeuronConfig  # noqa: E402
from ..backend.modules.attention.attention_base import NeuronAttentionBase
from ..backend.modules.attention.utils import (
    RotaryEmbedding,
    transpose_parallel_linear_layer,
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

        self.sequence_parallel_enabled = getattr(neuron_config, "sequence_parallel_enabled", False)
        self.sequence_dimension = 1 if self.sequence_parallel_enabled else None
        self.rms_norm_eps = config.rms_norm_eps
        self.mlp_kernel_enabled = neuron_config.mlp_kernel_enabled
        self.logical_nc_config = neuron_config.logical_nc_config
        mlp_bias = getattr(config, "mlp_bias", False)
        self.gate_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=mlp_bias,
            gather_output=False,
            dtype=neuron_config.torch_dtype,
            pad=True,
            sequence_parallel_enabled=False,
            sequence_dimension=None,
        )
        self.up_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=mlp_bias,
            gather_output=False,
            dtype=neuron_config.torch_dtype,
            pad=True,
            sequence_parallel_enabled=False,
            sequence_dimension=None,
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=mlp_bias,
            input_is_parallel=True,
            dtype=neuron_config.torch_dtype,
            pad=True,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            sequence_dimension=self.sequence_dimension,
            reduce_dtype=neuron_config.torch_dtype,
        )

        if self.mlp_kernel_enabled:
            # Transpose the weights to the layout expected by kernels
            self.gate_proj.weight = transpose_parallel_linear_layer(self.gate_proj.weight)
            self.up_proj.weight = transpose_parallel_linear_layer(self.up_proj.weight)
            self.down_proj.weight = transpose_parallel_linear_layer(self.down_proj.weight)

    def _kernel_enabled_mlp(self, x, fused_rmsnorm, rmsnorm, residual):
        fused_residual = residual is not None
        logger.debug(
            f"MLP: kernel, fused_residual={fused_residual}, fused_rmsnorm={fused_rmsnorm}, logical_nc_config={self.logical_nc_config}"
        )

        # Choose which kernel to call
        if fused_residual:
            assert not self.sequence_parallel_enabled, (
                "MLP kernel cannot have both fused residual add and sequence parallel RMSnorm!"
            )
            # Using fused residual add
            _mlp_fwd_call = nki_jit()(mlp_fused_add_isa_kernel)
        else:
            _mlp_fwd_call = nki_jit()(mlp_isa_kernel)

        if self.sequence_parallel_enabled:
            x = gather_from_sequence_parallel_region(x, self.sequence_dimension)

        # Build output tensor
        output_tensor_seqlen = x.shape[1]
        if fused_residual:
            # seqlen dim is doubled to store the residual add output
            output_tensor_seqlen *= 2

        output_tensor = torch.zeros(
            size=(
                x.shape[0],  # batch size
                output_tensor_seqlen,
                self.hidden_size,  # hidden size
            ),
            dtype=x.dtype,
            device=x.device,
        )

        # Grab weights
        # all weights of the layers are stored in (out, in) shape
        # unsqueeze so that shape of RMS gamma weight is [1, hidden] instead of [hidden]
        ln_w = rmsnorm.weight.unsqueeze(0)
        gate_w = self.gate_proj.weight.data
        up_w = self.up_proj.weight.data
        down_w = self.down_proj.weight.data

        grid = (nc(self.logical_nc_config),)

        if fused_residual:
            _mlp_fwd_call[grid](
                x,  # attn_output
                residual,  # hidden
                ln_w,  # ln_w
                gate_w,  # gate_w
                up_w,  # up_w
                down_w,  # down_w
                output_tensor,  # out
                fused_rmsnorm=fused_rmsnorm,
                eps=self.rms_norm_eps,
                kernel_name="MLP",
                store_add=True,
            )
            original_seqlen = x.shape[1]
            residual = output_tensor[:, original_seqlen:, :]
            output_tensor = output_tensor[:, :original_seqlen, :]
        else:
            _mlp_fwd_call[grid](
                x,  # hidden
                # should be fine to pass gamma is as a dummy even if not using fused rmsnorm
                ln_w,
                gate_w,
                up_w,
                down_w,
                output_tensor,  # out
                # Run RMSNorm inside the kernel if NOT using SP rmsnorm
                fused_rmsnorm=fused_rmsnorm,
                eps=self.rms_norm_eps,
                kernel_name="MLP",
            )
            residual = None

        # All-reduce or reduce-scatter, depending on whether SP is enabled
        if self.sequence_parallel_enabled:
            output_tensor = reduce_scatter_to_sequence_parallel_region(output_tensor, self.sequence_dimension)
        else:
            output_tensor = reduce_from_tensor_model_parallel_region(output_tensor)

        logger.debug(f"MLP output shape {output_tensor.shape}")
        return (output_tensor, residual)

    def _native_mlp(self, x, rmsnorm):
        logger.debug("MLP: native compiler")
        # all-gather is done here instead of CPL layers to
        # avoid 2 all-gathers from up and gate projections
        if self.sequence_parallel_enabled:
            x = gather_from_sequence_parallel_region(x, self.sequence_dimension)

        gate_proj_output = self.gate_proj(x)
        up_proj_output = self.up_proj(x)
        down_proj_input = self.act_fn(gate_proj_output) * up_proj_output
        output = self.down_proj(down_proj_input)
        logger.debug(f"MLP output shape {output.shape}")
        return output

    def forward(self, x, rmsnorm=None, residual=None):
        """
        If residual is passed in, will fuse its add into the MLP kernel

        Returns a tuple of (output, residual), where residual is the output of the residual add
        """
        if self.mlp_kernel_enabled:
            fused_rmsnorm = not self.sequence_parallel_enabled
            # MLP kernel
            return self._kernel_enabled_mlp(x, fused_rmsnorm, rmsnorm, residual)
        else:
            # No kernel
            return (self._native_mlp(x, rmsnorm), None)


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
        self.qkv_kernel_enabled = neuron_config.qkv_kernel_enabled
        self.mlp_kernel_enabled = neuron_config.mlp_kernel_enabled
        self.mlp_kernel_fuse_residual_add = neuron_config.mlp_kernel_fuse_residual_add
        self.sequence_parallel_enabled = neuron_config.sequence_parallel_enabled
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

        # RMSNorm (fused with QKV kernel when SP is disabled)
        if (not self.qkv_kernel_enabled or self.sequence_parallel_enabled) and self.input_layernorm:
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

        if self.mlp_kernel_enabled and self.mlp_kernel_fuse_residual_add:
            assert not self.sequence_parallel_enabled, (
                "mlp_kernel_fuse_residual_add should be off when sequence parallelism is enabled"
            )
            # First residual add handled in the MLP kernel
            hidden_states, residual = self.mlp(
                hidden_states,
                rmsnorm=self.post_attention_layernorm,
                residual=residual,
            )
        else:
            hidden_states = residual + hidden_states
            residual = hidden_states
            # RMSNorm (fused with QKV kernel when SP is disabled)
            if not self.mlp_kernel_enabled or self.sequence_parallel_enabled:
                hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states, _ = self.mlp(
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
            shard_across_embedding=not neuron_config.vocab_parallel,
            sequence_parallel_enabled=False,
            pad=True,
            use_spmd_rank=neuron_config.vocab_parallel,
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

        if neuron_config.vocab_parallel:
            # TODO: this hack can be removed after replication_id is ready to use
            state_dict["embed_tokens.rank_util.rank"] = torch.arange(0, neuron_config.local_ranks_size)

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
