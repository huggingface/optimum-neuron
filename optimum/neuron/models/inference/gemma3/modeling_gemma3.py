# coding=utf-8
# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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
"""PyTorch Gemma3 model for NXD inference."""

import logging

import torch
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.mappings import _gather_along_dim
from torch import nn
from torch_neuronx.xla_impl.ops import RmsNorm
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

from ..backend.config import NxDNeuronConfig
from ..backend.modules.attention.attention_base import NeuronAttentionBase
from ..backend.modules.attention.utils import RotaryEmbedding
from ..backend.modules.decoder import NxDDecoderModelForCausalLM, NxDModelForCausalLM
from ..backend.modules.generation.sampling import mask_padded_logits
from ..llama.modeling_llama import convert_state_dict_to_fused_qkv


logger = logging.getLogger("Neuron")


class NeuronGemma3RMSNorm(nn.Module):
    """
    Gemma3-specific RMSNorm using the hardware-optimized AwsNeuronRmsNorm custom op.

    Gemma3 uses (1.0 + weight) scaling instead of just weight, with weights
    initialized to zeros. This class keeps that convention in the forward pass
    so no weight adjustment is needed during state dict conversion.

    Formula: output = RmsNorm(x) * (1 + weight)

    Reference: transformers/models/gemma3/modeling_gemma3.py:Gemma3RMSNorm
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        original_dtype = x.dtype
        x = x.to(torch.float32)
        result = RmsNorm.apply(x, 1.0 + self.weight, self.eps, len(x.shape) - 1)
        return result.to(original_dtype)


class NeuronGemma3TextScaledWordEmbedding(nn.Module):
    """
    Gemma3-specific scaled embeddings.

    Embeddings are multiplied by sqrt(hidden_size) as per Gemma3 architecture.
    This wrapper adds the scaling on top of ParallelEmbedding.

    Reference: transformers/models/gemma3/modeling_gemma3.py:Gemma3TextScaledWordEmbedding
    """

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int,
        dtype: torch.dtype,
        shard_across_embedding: bool = True,
        pad: bool = True,
    ):
        super().__init__()
        # Store the scaling factor
        self.embed_scale = embedding_dim**0.5
        # Use standard ParallelEmbedding
        self.embedding = ParallelEmbedding(
            num_embeddings,
            embedding_dim,
            padding_idx,
            dtype=dtype,
            shard_across_embedding=shard_across_embedding,
            pad=pad,
        )

    def forward(self, input_ids: torch.Tensor):
        """Get embeddings and scale by sqrt(hidden_size)"""
        embeds = self.embedding(input_ids)
        return embeds * self.embed_scale


class NeuronGemma3Attention(NeuronAttentionBase):
    """
    Gemma3 attention mechanism with Q-K normalization.

    Key features:
    - Q-K normalization after projection (similar to Qwen3)
    - MQA configuration (num_kv_heads=1 in Gemma3-270M/1B models)
    - Layer-specific RoPE: sliding_attention layers use rope_local_base_freq,
      full_attention layers use rope_theta
    - Custom NKI flash attention kernel for head_dim=256 (d-tiling approach)
    - Sliding window attention support in the flash attention
    """

    def __init__(self, config: Gemma3TextConfig, neuron_config: NxDNeuronConfig, layer_idx: int = 0):
        # Initialize base attention without Q-K norm (we'll add them manually)
        super().__init__(config, neuron_config)

        # Select RoPE theta based on layer type:
        # - sliding_attention layers use rope_local_base_freq (e.g. 10000)
        # - full_attention layers use rope_theta (e.g. 1000000)
        is_sliding = config.layer_types[layer_idx] == "sliding_attention"
        rope_theta = config.rope_local_base_freq if is_sliding else config.rope_theta

        # Set sliding window size for the flash attention kernel
        self.sliding_window_size = config.sliding_window if is_sliding else 0

        self.rotary_emb = RotaryEmbedding(
            dim=self.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=rope_theta,
        )

        # Q-K normalization (Gemma3-specific, similar to Qwen3)
        # IMPORTANT: Use Gemma3RMSNorm which applies (1 + weight) scaling, not standard RMSNorm!
        self.q_layernorm = NeuronGemma3RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_layernorm = NeuronGemma3RMSNorm(self.head_dim, eps=config.rms_norm_eps)


class NeuronGemma3MLP(nn.Module):
    """
    Gemma3 MLP (feed-forward network).

    Architecture: gate_proj, up_proj, down_proj with GELU activation.
    Similar to LLaMA but uses GELU with tanh approximation instead of SiLU.

    Reference: transformers/models/gemma3/modeling_gemma3.py:Gemma3MLP
    """

    def __init__(self, config: Gemma3TextConfig, neuron_config: NxDNeuronConfig):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size

        # Gate and up projections (column parallel)
        self.gate_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=neuron_config.torch_dtype,
            pad=True,
        )
        self.up_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=neuron_config.torch_dtype,
            pad=True,
        )

        # Down projection (row parallel)
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=neuron_config.torch_dtype,
            pad=True,
            reduce_dtype=neuron_config.torch_dtype,
        )

        # GELU activation with tanh approximation (Gemma3-specific)
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """MLP forward: down_proj(act(gate_proj(x)) * up_proj(x))"""
        gate_proj_output = self.gate_proj(x)
        up_proj_output = self.up_proj(x)
        down_proj_input = self.act_fn(gate_proj_output) * up_proj_output
        output = self.down_proj(down_proj_input)
        return output


class NeuronGemma3DecoderLayer(nn.Module):
    """
    Gemma3 decoder layer.

    Key architectural features:
    - Four normalization layers: input, post_attention, pre_feedforward, post_feedforward
    - Pre-norm architecture with residual connections
    - Pattern: residual + post_norm(module(pre_norm(hidden)))
    """

    def __init__(self, config: Gemma3TextConfig, neuron_config: NxDNeuronConfig, layer_idx: int = 0):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx
        self.attention_type = config.layer_types[layer_idx]

        # Attention and MLP
        self.self_attn = NeuronGemma3Attention(config, neuron_config, layer_idx=layer_idx)
        self.mlp = NeuronGemma3MLP(config, neuron_config)

        # Four normalization layers (Gemma3-specific)
        # Use custom NeuronGemma3RMSNorm with (1.0 + weight) scaling
        self.input_layernorm = NeuronGemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = NeuronGemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_layernorm = NeuronGemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_layernorm = NeuronGemma3RMSNorm(self.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[
        torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor], torch.Tensor | None, torch.Tensor | None
    ]:
        """
        Gemma3 decoder layer forward pass.

        Architecture:
        1. Attention block: residual + post_attn_norm(attn(input_norm(hidden)))
        2. MLP block: residual + post_ffn_norm(mlp(pre_ffn_norm(hidden)))
        """
        # Attention block with pre and post normalization
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self attention
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )

        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        # MLP block with pre and post normalization
        residual = hidden_states
        hidden_states = self.pre_feedforward_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, present_key_value, cos_cache, sin_cache


class NxDGemma3Model(NxDDecoderModelForCausalLM):
    """
    The neuron version of the Gemma3 model.

    Key Gemma3 features:
    - Scaled embeddings (sqrt(hidden_size) multiplier)
    - Four normalization layers per decoder block
    - Q-K normalization in attention
    - GELU activation in MLP

    """

    def __init__(self, config: Gemma3TextConfig, neuron_config: NxDNeuronConfig):
        super().__init__(config, neuron_config)
        self.sliding_window = config.sliding_window

        # Use scaled embeddings (Gemma3-specific)
        self.embed_tokens = NeuronGemma3TextScaledWordEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            dtype=neuron_config.torch_dtype,
            shard_across_embedding=True,
            pad=True,
        )

        # Language modeling head
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=not neuron_config.on_device_sampling,
            bias=False,
            pad=True,
        )

        # Decoder layers
        self.layers = nn.ModuleList(
            [NeuronGemma3DecoderLayer(config, neuron_config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )

        # Final normalization (use custom Gemma3RMSNorm)
        self.norm = NeuronGemma3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids,
        position_ids,
        seq_ids,
        sampling_params,
    ):
        """Forward pass with per-layer sliding window / full attention masks.

        This overrides the base NxDDecoderModelForCausalLM.forward() to create
        two attention masks (full causal and sliding window) and select the
        appropriate one for each decoder layer based on its attention_type.
        """
        is_for_context_encoding = self._is_context_encoding(input_ids.shape[-1])
        if self._is_for_speculation(input_ids.shape[-1]):
            raise ValueError("Speculation is not supported for Gemma3 model")

        cache_size = self.n_positions
        device = input_ids.device

        if is_for_context_encoding:
            past_key_values = None
            # Full causal mask (lower triangle)
            full_attention_mask = torch.full((self.n_positions, self.n_positions), True, device=device).tril(
                diagonal=0
            )
            full_attention_mask = full_attention_mask[None, None, :, :].expand(
                self.batch_size, 1, self.n_positions, self.n_positions
            )
            # Sliding window mask: banded causal matrix
            sliding_attention_mask = (
                full_attention_mask
                & torch.full((self.n_positions, self.n_positions), True, device=device).triu(
                    diagonal=-(self.sliding_window - 1)
                )[None, None, :, :]
            )
            active_mask = None
        else:
            past_key_values = self.kv_mgr.get_cache(cache_size)
            max_cached_positions = position_ids.expand(self.batch_size, self.n_positions) - 1
            all_positions = (
                torch.arange(self.n_positions, device=device).view(1, -1).expand(self.batch_size, self.n_positions)
            )
            # Full attention mask for cached tokens
            full_attention_mask = (max_cached_positions >= all_positions).view(self.batch_size, 1, 1, self.n_positions)
            # Sliding window mask: only attend to positions within the window
            current_position = position_ids.expand(self.batch_size, self.n_positions)
            sliding_attention_mask = full_attention_mask & (
                (current_position - all_positions) < self.sliding_window
            ).view(self.batch_size, 1, 1, self.n_positions)
            active_mask = None

        # Map attention types to masks
        causal_mask_mapping = {
            "full_attention": full_attention_mask,
            "sliding_attention": sliding_attention_mask,
        }

        batch_size, seq_length = input_ids.shape[:2]
        position_ids = position_ids.view(-1, seq_length).long()

        inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        new_key_values = []
        # Gemma3 uses different RoPE bases for sliding (rope_local_base_freq) vs
        # full (rope_theta) attention layers, so we must maintain separate cos/sin
        # caches per attention type to avoid reusing the wrong embeddings.
        rope_cache = {
            "sliding_attention": (None, None),
            "full_attention": (None, None),
        }
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            attention_mask = causal_mask_mapping[decoder_layer.attention_type]
            cos_cache, sin_cache = rope_cache[decoder_layer.attention_type]

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                active_mask=active_mask,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
            )

            hidden_states = layer_outputs[0]
            new_key_values.append(layer_outputs[1])
            rope_cache[decoder_layer.attention_type] = layer_outputs[2:]

        hidden_states = self.norm(hidden_states)

        updated_kv_cache = self.kv_mgr.update_cache(
            is_for_context_encoding=is_for_context_encoding,
            seq_ids=seq_ids,
            position_ids=position_ids,
            new_key_values=new_key_values,
            seq_len=cache_size,
        )

        hidden_size = hidden_states.shape[-1]
        if is_for_context_encoding:
            index = torch.max(position_ids, dim=1, keepdim=True).indices
            index = index.unsqueeze(1).expand(batch_size, 1, hidden_size)
            hidden_states = torch.gather(hidden_states, dim=1, index=index)

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        if self.lm_head.gather_output:
            rank_id = torch.tensor(0, device=logits.device, dtype=torch.int32)
            world_size = 1
        else:
            rank_id = self.rank_util.get_rank()
            world_size = torch.distributed.get_world_size(group=self.lm_head.tensor_parallel_group)

        if hasattr(self.lm_head, "pad_size"):
            logits = mask_padded_logits(logits, rank_id, world_size, pad_size=self.lm_head.pad_size)

        res = logits
        if self.neuron_config.on_device_sampling:
            res = self.sampler(logits[:, -1, :], sampling_params, rank_id=rank_id)

        outputs = [res]
        if self.neuron_config.output_logits:
            logits = _gather_along_dim(
                logits,
                partition_dim=2,
            )
            outputs += [logits]
        outputs += updated_kv_cache

        return outputs


class Gemma3NxDModelForCausalLM(NxDModelForCausalLM):
    """
    Gemma3 model for NxD inference.

    This class wraps NxDGemma3Model and provides the interface for
    compilation, inference, and weight loading with Optimum Neuron.

    Supports per-layer sliding window attention: layers with attention_type
    "sliding_attention" use a banded causal mask of size sliding_window,
    while "full_attention" layers use the standard full causal mask.
    """

    _model_cls = NxDGemma3Model
    # Gemma3's custom mixed sliding window / full attention masks don't support chunked prefill yet
    _supports_chunked_prefill = False

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: Gemma3TextConfig, neuron_config: NxDNeuronConfig
    ) -> dict:
        """
        Convert HuggingFace Gemma3 state dict to NeuronX format.

        Key mappings:
        - embed_tokens.weight -> embed_tokens.embedding.weight (due to scaling wrapper)
        - layers.*.self_attn.q_norm.weight -> layers.*.self_attn.q_layernorm.weight
        - layers.*.self_attn.k_norm.weight -> layers.*.self_attn.k_layernorm.weight
        - Four decoder layer norms already match HF format (no transformation needed)
        """
        # Handle embeddings with scaling wrapper
        if "embed_tokens.weight" in state_dict:
            state_dict["embed_tokens.embedding.weight"] = state_dict.pop("embed_tokens.weight")

        # Rename Q-K normalization layers to match NeuronAttentionBase expectations
        for l in range(config.num_hidden_layers):
            attn_prefix = f"layers.{l}.self_attn"

            # Q-K norm renaming (these go directly to NeuronRMSNorm, no extra .norm wrapper)
            if f"{attn_prefix}.q_norm.weight" in state_dict:
                state_dict[f"{attn_prefix}.q_layernorm.weight"] = state_dict.pop(f"{attn_prefix}.q_norm.weight")
            if f"{attn_prefix}.k_norm.weight" in state_dict:
                state_dict[f"{attn_prefix}.k_layernorm.weight"] = state_dict.pop(f"{attn_prefix}.k_norm.weight")

            # Four decoder layer norms use NeuronGemma3RMSNorm (no nested structure)
            # Keys already match HF format - no transformation needed

        # Final model norm also uses NeuronGemma3RMSNorm (no nested structure)
        # Key already matches HF format - no transformation needed

        # Apply fused QKV if enabled
        if neuron_config.fused_qkv:
            state_dict = convert_state_dict_to_fused_qkv(state_dict, config)

        # Add rank tensors for tensor parallelism
        num_layers = config.num_hidden_layers
        tp_degree = neuron_config.tp_degree
        for i in range(num_layers):
            state_dict[f"layers.{i}.self_attn.rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        # Add rank tensor for base model
        state_dict["rank_util.rank"] = torch.arange(0, tp_degree, dtype=torch.int32)

        return state_dict

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """
        Handle tied weights between embeddings and lm_head.

        Gemma3 ties embeddings by default (tie_word_embeddings=True in config).
        Note: The embedding is nested as embed_tokens.embedding.weight due to scaling wrapper.
        """
        if "embed_tokens.embedding.weight" in state_dict:
            state_dict["lm_head.weight"] = state_dict["embed_tokens.embedding.weight"].clone()

    @classmethod
    def _get_neuron_config(
        cls,
        checkpoint_id: str,
        checkpoint_revision: str,
        instance_type: str,
        batch_size: int,
        sequence_length: int,
        tensor_parallel_size: int,
        dtype: torch.dtype,
        prefill_chunk_size: int = 0,
    ):
        """
        Get the neuron configuration for Gemma3 model.

        Default settings:
        - fused_qkv=True (optimization)
        - continuous_batching enabled when batch_size > 1
        """
        continuous_batching = (batch_size > 1) if batch_size else False
        return NxDNeuronConfig(
            checkpoint_id=checkpoint_id,
            checkpoint_revision=checkpoint_revision,
            batch_size=batch_size,
            sequence_length=sequence_length,
            tp_degree=tensor_parallel_size,
            torch_dtype=dtype,
            target=instance_type,
            on_device_sampling=True,
            fused_qkv=True,
            continuous_batching=continuous_batching,
            prefill_chunk_size=prefill_chunk_size,
        )
