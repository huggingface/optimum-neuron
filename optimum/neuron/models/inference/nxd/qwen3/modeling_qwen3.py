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
# Adapted from https://github.com/aws-neuron/neuronx-distributed-inference/blob/9993358ce052fd7a1bb4a7497a6318aac36ed95c/src/neuronx_distributed_inference/models/Qwen3/modeling_Qwen3.py
"""PyTorch Qwen3 model for NXD inference."""

import gc
import logging
from typing import Optional, Tuple, Type

import torch
from neuronx_distributed.parallel_layers import parallel_state
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
from transformers.models.qwen3.configuration_qwen3 import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm, Qwen3RotaryEmbedding

from ..backend.config import NxDNeuronConfig  # noqa: E402
from ..backend.modules.attention.attention_base import NeuronAttentionBase
from ..backend.modules.attention.utils import (
    apply_rotary_pos_emb,
    move_heads_front,
    transpose_parallel_linear_layer,
)
from ..backend.modules.custom_calls import CustomRMSNorm
from ..backend.modules.decoder import NxDDecoderModel, NxDModelForCausalLM


logger = logging.getLogger("Neuron")


def get_rmsnorm_cls():
    # Initialize to the appropriate implementation of RMSNorm
    # If infer on NXD -> CustomRMSNorm
    # If infer on CPU -> HF_RMSNorm (CustomRMSNorm does not work on CPU)
    return CustomRMSNorm if parallel_state.model_parallel_is_initialized() else Qwen3RMSNorm


def convert_state_dict_to_fused_qkv(Qwen3_state_dict, cfg: Qwen3Config):
    """
    This function concats the qkv weights to a Wqkv weight for fusedqkv, and deletes the qkv weights.
    """
    for l in range(cfg.num_hidden_layers):  # noqa: E741
        Qwen3_state_dict[f"layers.{l}.self_attn.Wqkv.weight"] = torch.cat(
            [
                Qwen3_state_dict[f"layers.{l}.self_attn.q_proj.weight"],
                Qwen3_state_dict[f"layers.{l}.self_attn.k_proj.weight"],
                Qwen3_state_dict[f"layers.{l}.self_attn.v_proj.weight"],
            ],
        )
        del Qwen3_state_dict[f"layers.{l}.self_attn.q_proj.weight"]
        del Qwen3_state_dict[f"layers.{l}.self_attn.k_proj.weight"]
        del Qwen3_state_dict[f"layers.{l}.self_attn.v_proj.weight"]

    gc.collect()

    return Qwen3_state_dict


class NeuronQwen3MLP(nn.Module):
    """
    This class just replace the linear layers (gate_proj, up_proj and down_proj) with column and row parallel layers
    """

    def __init__(self, config: Qwen3Config, neuron_config: NxDNeuronConfig):
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
        if parallel_state.model_parallel_is_initialized():
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
                reduce_dtype=neuron_config.rpl_reduce_dtype,
            )

            if self.mlp_kernel_enabled:
                # Transpose the weights to the layout expected by kernels
                self.gate_proj.weight = transpose_parallel_linear_layer(self.gate_proj.weight)
                self.up_proj.weight = transpose_parallel_linear_layer(self.up_proj.weight)
                self.down_proj.weight = transpose_parallel_linear_layer(self.down_proj.weight)

        else:
            self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
            self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=mlp_bias)
            self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=mlp_bias)

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


class NeuronQwen3Attention(NeuronAttentionBase):
    """
    Compared with Qwen3Attention, this class just
    1. replaces the q_proj, k_proj, v_proj with column parallel layer
    2. replaces the o_proj with row parallel layer
    3. update self.num_head to be self.num_head / tp_degree
    4. update self.num_key_value_heads to be self.num_key_value_heads / tp_degree
    5. update forward() method to adjust to changes from self.num_head
    """

    def __init__(self, config: Qwen3Config, neuron_config: NxDNeuronConfig):
        super().__init__(config, neuron_config)
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.rotary_emb = Qwen3RotaryEmbedding(config)

    def prep_qkv_tensors(
        self,
        position_ids,
        hidden_states,
        past_key_value,
        adapter_ids=None,
        cos_cache=None,
        sin_cache=None,
        rmsnorm=None,
    ):
        """take care of the shape, layout, group query, custom position encoding, etc."""
        Q, K, V = self.qkv_proj(hidden_states=hidden_states, rmsnorm=rmsnorm, adapter_ids=adapter_ids)

        # Divide hidden_dim across heads for MHA
        # Change layout: BSHD -> BHSD
        bsz, q_len, _ = hidden_states.size()
        if self.sequence_parallel_enabled:
            q_len *= self.tensor_model_parallel_group.size()

        Q = move_heads_front(Q, bsz, q_len, self.num_heads, self.head_dim, layernorm=self.q_norm)
        K = move_heads_front(
            K,
            bsz,
            q_len,
            self.num_key_value_heads,
            self.head_dim,
            layernorm=self.k_norm,
        )
        V = move_heads_front(V, bsz, q_len, self.num_key_value_heads, self.head_dim, layernorm=None)

        # Rotate Q and K
        if self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, position_ids)

            Q, K = apply_rotary_pos_emb(Q, K, cos_cache, sin_cache)

        return Q, K, V, cos_cache, sin_cache


class NeuronQwen3DecoderLayer(nn.Module):
    """
    Just replace the attention with the NXD version, and MLP with the NXD version
    """

    def __init__(self, config: Qwen3Config, neuron_config: NxDNeuronConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = NeuronQwen3Attention(config, neuron_config)
        self.mlp = NeuronQwen3MLP(config, neuron_config)
        logger.debug(
            f"Instantiating RMSNorm modules with hidden size {config.hidden_size} and EPS {config.rms_norm_eps}"
        )
        self.input_layernorm = get_rmsnorm_cls()(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = get_rmsnorm_cls()(
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
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
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


class NxDQwen3Model(NxDDecoderModel):
    """
    The neuron version of the Qwen3Model
    """

    def __init__(self, config: Qwen3Config, neuron_config: NxDNeuronConfig):
        super().__init__(config, neuron_config)

        if parallel_state.model_parallel_is_initialized():
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
        else:
            self.embed_tokens = nn.Embedding(
                config.vocab_size,
                config.hidden_size,
                config.pad_token_id,
            )
            self.lm_head = nn.Linear(
                config.hidden_size,
                config.vocab_size,
                bias=False,
            )

        self.layers = nn.ModuleList(
            [NeuronQwen3DecoderLayer(config, neuron_config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = get_rmsnorm_cls()(config.hidden_size, eps=config.rms_norm_eps)


class Qwen3NxDModelForCausalLM(NxDModelForCausalLM):
    """
    This class extends Qwen3ForCausalLM create traceable
    blocks for Neuron.

    Args:
        Qwen3ForCausalLM (_type_): _description_
    """

    _model_cls = NxDQwen3Model

    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict, config: Qwen3Config, neuron_config: NxDNeuronConfig) -> dict:
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
        return NxDNeuronConfig(
            checkpoint_id=checkpoint_id,
            checkpoint_revision=checkpoint_revision,
            batch_size=batch_size,
            sequence_length=sequence_length,
            tp_degree=tensor_parallel_size,
            torch_dtype=auto_cast_type,
            on_device_sampling=True,
            fused_qkv=True,
        )
