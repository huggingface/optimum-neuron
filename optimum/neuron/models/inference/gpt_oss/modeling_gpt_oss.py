# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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
import gc
from typing import List, Optional, Tuple

import torch
import torch_xla.core.xla_model as xm
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, ParallelEmbedding
from neuronx_distributed.parallel_layers.mappings import (
    _reduce_scatter_along_dim,
    gather_from_sequence_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
    scatter_to_tensor_model_parallel_region,
)
from torch import nn
from transformers.integrations.bitsandbytes import dequantize_and_replace
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.utils import logging

from optimum.neuron.models.inference.backend.modules.moe.expert_mlps import ExpertMLPs

from ..backend.config import NxDNeuronConfig
from ..backend.modules.attention.attention_base import (
    FlashAttentionStrategy,
    NeuronAttentionBase,
    manual_softmax,
    repeat_kv,
)
from ..backend.modules.attention.utils import move_heads_front
from ..backend.modules.custom_calls import CustomRMSNorm
from ..backend.modules.decoder import NxDDecoderModel, NxDModelForCausalLM
from ..mixtral.modeling_mixtral import NeuronMixtralDecoderLayer
from .configuration_gpt_oss import GptOssConfig


logger = logging.get_logger(__name__)

# NOTE: GptOssRMSNorm is the same as LlamaRMSNorm, but in the original Llama code, there was a cast to the
# original_dtype only of hidden_states, while in the GptOss code, there was a cast to the original_dtype of
# hidden_states and the weight multiplied together. I think CustomRMSNorm does exactly that, so there is no need to add
# a different implementation.


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


def _convert_moe_packed_tensors(
    blocks,
    scales,
    *,
    dtype: torch.dtype = torch.bfloat16,
    rows_per_chunk: int = 32768 * 1024,
) -> torch.Tensor:
    import math

    # Check if blocks and scales are on CPU, and move to GPU if so
    if not blocks.is_cuda and torch.cuda.is_available():
        blocks = blocks.cuda()
        scales = scales.cuda()

    scales = scales.to(torch.int32) - 127

    assert blocks.shape[:-1] == scales.shape, f"{blocks.shape=} does not match {scales.shape=}"

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


def _convert_weight(neuron_state_dict, weight_name, new_weight_name):
    """In this model, weights can be quantized using MXFP4. To support this, we dequantize weights on the fly.
    """
    # If weight is not quantized, we can just copy it with the new name
    if weight_name in neuron_state_dict:
        neuron_state_dict[new_weight_name] = neuron_state_dict.pop(weight_name)
        return

    # Search the blocks and scales in the state dict
    blocks_name = weight_name + "_blocks"
    scales_name = weight_name + "_scales"
    if blocks_name in neuron_state_dict and scales_name in neuron_state_dict:
        blocks = neuron_state_dict[blocks_name]
        scales = neuron_state_dict[scales_name]
        dequantized = _convert_moe_packed_tensors(blocks, scales)
        # Dimensions are transposed when quantized, so we need to transpose them back
        neuron_state_dict[new_weight_name] = dequantized.transpose(1, 2).contiguous()
        neuron_state_dict.pop(blocks_name)
        neuron_state_dict.pop(scales_name)
        return

    raise ValueError(f"Weight {weight_name} not found in neuron_state_dict")


def convert_gptoss_to_neuron_state_dict(neuron_state_dict, config, neuron_config):
    """
    Helper function which returns the model weights from the GptOss model in a state dictionary compatible with the stucture of the neuron MoE model.
    """
    assert neuron_config.glu_mlp is True, "Only GLU MLP is supported for GptOss Top-K model"

    for l in range(config.num_hidden_layers):  # noqa: E741
        # Using NxD ExpertMLPs layer changes weights
        _convert_weight(
            neuron_state_dict,
            f"layers.{l}.mlp.experts.gate_up_proj",
            f"layers.{l}.mlp.experts.mlp_op.gate_up_proj.weight",
        )
        _convert_weight(
            neuron_state_dict,
            f"layers.{l}.mlp.experts.gate_up_proj_bias",
            f"layers.{l}.mlp.experts.mlp_op.gate_up_proj.bias",
        )
        _convert_weight(
            neuron_state_dict, f"layers.{l}.mlp.experts.down_proj", f"layers.{l}.mlp.experts.mlp_op.down_proj.weight"
        )
        _convert_weight(
            neuron_state_dict,
            f"layers.{l}.mlp.experts.down_proj_bias",
            f"layers.{l}.mlp.experts.mlp_op.down_proj.bias",
        )
        gc.collect()

    return neuron_state_dict


def _experts_swiglu_activation(x: torch.Tensor) -> torch.Tensor:
    ALPHA = 1.702
    LIMIT = 7.0
    # gate, up = torch.chunk(x, chunks=2, dim=-1)
    gate, up = x[..., ::2], x[..., 1::2]
    gate = gate.clamp(min=None, max=LIMIT)
    up = up.clamp(min=-LIMIT, max=LIMIT)
    glu = gate * torch.sigmoid(gate * ALPHA)
    gated_output = (up + 1) * glu
    return gated_output


class GptOssMLP(nn.Module):
    def __init__(self, config: GptOssConfig, neuron_config: NxDNeuronConfig):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.hidden_dim = config.hidden_size
        self.num_local_experts = config.num_local_experts
        self.experts = ExpertMLPs(
            num_experts=config.num_local_experts,
            top_k=config.num_experts_per_tok,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            capacity_factor=neuron_config.capacity_factor,
            glu_mlp=True,
            expert_bias=True,
            normalize_top_k_affinities=True,
            glu_activation_fn=_experts_swiglu_activation,
        )
        self.router = nn.Linear(config.hidden_size, self.num_local_experts, bias=True)

        self.sequence_parallel_enabled = neuron_config.sequence_parallel_enabled
        setattr(self.router.weight, "sequence_parallel_enabled", self.sequence_parallel_enabled)

        self.batch_size = neuron_config.batch_size
        self.sequence_dimension = 1
        self.tensor_parallel_group = parallel_state.get_tensor_model_parallel_group()
        self.ep_enabled = parallel_state.get_expert_model_parallel_size() > 1

    def forward(self, hidden_states):
        if self.sequence_parallel_enabled:
            # All-Gather the hidden_states to exit sequence parallel
            # full_hidden_states: (S', B, H) -> (S, B, H)
            full_hidden_states = gather_from_sequence_parallel_region(
                hidden_states,
                sequence_dimension=self.sequence_dimension,
                to_model_parallel=False,
                process_group=self.tensor_parallel_group,
            )
        else:
            full_hidden_states = hidden_states
        # full_hidden_states: (S, B, H) or (B, S, H)
        full_hidden_states_shape = full_hidden_states.shape
        seq_len = full_hidden_states_shape[self.sequence_dimension]

        # Get the router_logits, expert_affinities and expert_index from the router
        # router_logits: (T, E), expert_affinities: (T, E), expert_index: (T, top_k)
        if self.sequence_parallel_enabled:
            router_logits = self.router(hidden_states)
        else:
            router_logits = self.router(full_hidden_states)

        if self.sequence_parallel_enabled:
            # Gather the router_logits across ranks
            router_logits = gather_from_sequence_parallel_region(
                router_logits,
                sequence_dimension=self.sequence_dimension,
                to_model_parallel=False,
                process_group=self.tensor_parallel_group,
            )

        # Flatten S and B to T dimension
        router_logits = router_logits.reshape(-1, self.num_local_experts)
        # expert_index: (T, top_k)
        router_top_value, router_indices = torch.topk(router_logits, self.top_k, dim=-1)
        router_indices = router_indices.detach().to(dtype=torch.long)
        # Perform activation in fp64 to prevent auto-downcasting of operation to bf16, for numerical accuracy
        # expert_affinities: (T, E)
        router_top_value = torch.nn.functional.softmax(router_top_value, dim=1, dtype=router_logits.dtype)
        # Cast to required dtype
        # router_top_value = router_top_value.to(dtype=router_logits.dtype)
        router_scores = torch.zeros_like(router_logits).scatter_(1, router_indices, router_top_value)

        # full_hidden_states: (S, B, H) or (B, S, H) -> (T, H)
        full_hidden_states = full_hidden_states.reshape(-1, self.hidden_dim)
        output_states = self.experts(
            hidden_states=full_hidden_states,
            expert_affinities=router_scores,
            expert_index=router_indices,
            seq_len=seq_len,
        )

        if self.sequence_parallel_enabled:
            # Delayed reduce-scatter back to sequence parallel (as the hidden_states were in SP)
            output_states = reduce_scatter_to_sequence_parallel_region(
                output_states,
                self.sequence_dimension,
                process_group=self.tensor_parallel_group,
            )

        # TODO: check if this is correct when sequence_parallel is enabled
        batch_size = hidden_states.shape[0]
        output_states = output_states.reshape(batch_size, -1, self.hidden_dim)
        return (output_states,)


class GptOssRotaryEmbedding(nn.Module):
    def __init__(self, config: GptOssConfig, device=None):
        super().__init__()
        # NOTE: the rope type used by default in the GptOss code is "yarn", this is the configuration that has been
        # tested.
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]
        self.register_buffer("inv_freq", None, persistent=False)
        self.attention_scaling = 1.0

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if self.inv_freq is None:
            self.inv_freq, self.attention_scaling = self.rope_init_fn(self.config, x.device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = freqs
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    first_half, second_half = torch.chunk(x, 2, dim=-1)
    first_ = first_half * cos - second_half * sin
    second_ = second_half * cos + first_half * sin
    return torch.cat((first_, second_), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = _apply_rotary_emb(q, cos, sin)
    k_embed = _apply_rotary_emb(k, cos, sin)
    return q_embed, k_embed


class GptOssAttention(NeuronAttentionBase):
    def __init__(self, config: GptOssConfig, neuron_config: NxDNeuronConfig):
        super().__init__(
            config,
            neuron_config,
            qkv_proj_bias=config.attention_bias,
            o_proj_bias=config.attention_bias,
        )
        self.tp_degree = parallel_state.get_tensor_model_parallel_size()
        self.rotary_emb = GptOssRotaryEmbedding(config=config)
        # Main difference with Mixtral is the usage of sinks. These are sharded across the tensor parallel group.
        # TODO: check how to load sharded sinks correctly
        self.sinks = nn.Parameter(torch.empty(config.num_attention_heads))

    # Flash attention is not supported for GptOssAttention because of the sinks
    def get_flash_attention_strategy(self, q_len) -> FlashAttentionStrategy:
        return FlashAttentionStrategy.NONE

    def _get_sinks(self, Q_shape: torch.Size) -> torch.Tensor:
        # TODO: instead of sharding here, it would be better to do it in the init function
        sharded_sinks = scatter_to_tensor_model_parallel_region(self.sinks)
        sinks = sharded_sinks.reshape(1, -1, 1, 1).expand(
            Q_shape[0], -1, Q_shape[-2], -1
        )  # TODO make sure the sink is like a new token
        return sinks

    def compute_for_token_gen(
        self, Q, K, V, position_ids, past_key_value, attention_mask, active_mask
    ) -> torch.Tensor:
        """attention computation at token generation phase"""

        is_speculation = position_ids.shape[-1] > 1

        # Attention computation: softmax((Q.K/√dkv) + mask).V
        # i. prior (cached) KV
        K_prior = past_key_value[0]
        V_prior = past_key_value[1]
        K_prior = repeat_kv(K_prior, self.num_key_value_groups)
        V_prior = repeat_kv(V_prior, self.num_key_value_groups)
        prior_scores = torch.matmul(Q, K_prior.transpose(2, 3)) * self.qk_scale
        prior_scores = torch.where(attention_mask, prior_scores, torch.finfo(prior_scores.dtype).min)
        prior_scores = prior_scores.to(torch.float32)

        # ii. active (current/new) KV
        K_active = repeat_kv(K, self.num_key_value_groups)
        V_active = repeat_kv(V, self.num_key_value_groups)
        active_scores = torch.matmul(Q, K_active.transpose(2, 3)) * self.qk_scale
        if is_speculation:
            active_scores = torch.where(active_mask, active_scores, torch.finfo(active_scores.dtype).min)
        sinks = self._get_sinks(Q.shape)
        active_scores = torch.cat([active_scores, sinks], dim=-1)
        active_scores = active_scores.to(torch.float32)

        # iii. attention scores
        softmax_prior, softmax_active = manual_softmax(prior_scores, active_scores)
        # Drop the sinks after softmax
        softmax_active = softmax_active[..., :-1]
        softmax_prior, softmax_active = softmax_prior.to(Q.dtype), softmax_active.to(Q.dtype)
        attn_prior = torch.matmul(softmax_prior, V_prior)
        attn_active = torch.matmul(softmax_active, V_active)
        attn_output = attn_prior + attn_active

        return attn_output

    def perform_prefill(self, Q, K, V, q_len, bsz, attention_mask) -> Tuple[torch.Tensor, FlashAttentionStrategy]:
        """attention computation at prefilling (context encoding) phase"""
        K_active = repeat_kv(K, self.num_key_value_groups)
        V_active = repeat_kv(V, self.num_key_value_groups)

        # NOTE: flash attention is NONE for GptOssAttention
        flash_attn_strategy = FlashAttentionStrategy.NONE
        logger.debug("ATTN: native compiler")
        logger.debug(f"Not using flash_fwd for Q.shape={Q.shape}")
        active_scores = self.scaled_qk(Q, K_active, attention_mask)
        sinks = self._get_sinks(Q.shape)
        active_scores = torch.cat([active_scores, sinks], dim=-1)

        # Softmax is applied to the sinks as well
        active_scores = nn.functional.softmax(active_scores, dim=-1, dtype=torch.float32).to(Q.dtype)
        # But the sinks are not used in the output
        attn_output = torch.matmul(active_scores[..., :-1], V_active)
        return attn_output, flash_attn_strategy

    def prep_qkv_tensors(
        self,
        position_ids,
        hidden_states,
        past_key_value,
        cos_cache=None,
        sin_cache=None,
        rmsnorm=None,
    ):
        """
        This function is copied here so that local apply_rotary_pos_emb is used instead of the one in the
        attention_base.py file
        """
        Q, K, V = self.qkv_proj(hidden_states=hidden_states, rmsnorm=rmsnorm)

        # Divide hidden_dim across heads for MHA
        # Change layout: BSHD -> BHSD
        bsz, q_len, _ = hidden_states.size()
        if self.sequence_parallel_enabled:
            q_len *= self.tensor_model_parallel_group.size()

        Q = move_heads_front(Q, bsz, q_len, self.num_heads, self.head_dim, layernorm=self.q_layernorm)
        K = move_heads_front(K, bsz, q_len, self.num_key_value_heads, self.head_dim, layernorm=self.k_layernorm)
        V = move_heads_front(V, bsz, q_len, self.num_key_value_heads, self.head_dim, layernorm=None)

        # Rotate Q and K
        if cos_cache is None or sin_cache is None:
            cos_cache, sin_cache = self.rotary_emb(V, position_ids)

        # Use local apply_rotary_pos_emb function
        Q, K = apply_rotary_pos_emb(Q, K, cos_cache, sin_cache)

        return Q, K, V, cos_cache, sin_cache


class NeuronGptOssDecoderLayer(NeuronMixtralDecoderLayer):
    def __init__(self, config: GptOssConfig, neuron_config: NxDNeuronConfig):
        nn.Module.__init__(self)
        self.hidden_size = config.hidden_size
        self.self_attn = GptOssAttention(config, neuron_config)

        self.mlp = GptOssMLP(config, neuron_config)

        self.input_layernorm = CustomRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )
        self.post_attention_layernorm = CustomRMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
        )


class NeuronGptOssModel(NxDDecoderModel):
    def __init__(self, config: GptOssConfig, neuron_config: NxDNeuronConfig):
        super().__init__(config, neuron_config)
        if neuron_config.padding_side != "right":
            raise ValueError("Padding side must be right for GptOss models")
        if neuron_config.speculation_length != 0:
            raise ValueError("Speculation decoding is not supported for GptOss models")
        if neuron_config.flash_decoding_enabled:
            raise ValueError("Flash decoding is not supported for GptOss models")

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            dtype=neuron_config.torch_dtype,
            shard_across_embedding=True,
        )
        self.layers = nn.ModuleList(
            [NeuronGptOssDecoderLayer(config, neuron_config) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = CustomRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=not neuron_config.on_device_sampling,
            bias=False,
        )

        self.sliding_window = config.sliding_window
        self.attention_mask_functions = []
        for layer_idx in range(config.num_hidden_layers):
            if config.layer_types[layer_idx] == "sliding_attention":
                self.attention_mask_functions.append(self._create_sliding_window_attn_mask)
            elif config.layer_types[layer_idx] == "full_attention":
                self.attention_mask_functions.append(self.full_attention_mask)
            else:
                raise ValueError(f"Unsupported layer type: {config.layer_types[layer_idx]}")

    def full_attention_mask(self, attention_mask: torch.Tensor, input_ids: torch.LongTensor):
        return attention_mask

    def get_model_output(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        active_mask: Optional[List[torch.FloatTensor]] = None,
        # In llava context encoding model, input_embeds is precomputed
        inputs_embeds: Optional[torch.FloatTensor] = None,
    ):
        batch_size, seq_length = input_ids.shape[:2]

        past_key_values_length = 0
        if past_key_values is not None:
            past_key_values_length = past_key_values[0][0].shape[2]

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device  # noqa
            position_ids = torch.arange(
                past_key_values_length,
                seq_length + past_key_values_length,
                dtype=torch.long,
                device=device,
            )
            position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        else:
            position_ids = position_ids.view(-1, seq_length).long()

        # embed positions
        if self.sequence_parallel_enabled:
            hidden_states = _reduce_scatter_along_dim(
                inputs_embeds,
                self.sequence_dimension,
                xm.REDUCE_MAX,
            )
        else:
            hidden_states = inputs_embeds

        next_decoder_cache = ()
        cos_cache = None
        sin_cache = None
        for idx, decoder_layer in enumerate(self.layers):
            past_key_value = past_key_values[idx] if past_key_values is not None else None
            # TODO: attention_mask function could be called only once for each kind of layer
            mask = self.attention_mask_functions[idx](attention_mask, input_ids)
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=mask,
                position_ids=position_ids,
                past_key_value=past_key_value,
                active_mask=active_mask,
                cos_cache=cos_cache,
                sin_cache=sin_cache,
            )

            hidden_states = layer_outputs[0]
            next_decoder_cache += (layer_outputs[1],)
            cos_cache, sin_cache = layer_outputs[2:]

        hidden_states = self.norm(hidden_states)

        if self.sequence_parallel_enabled:
            hidden_states = gather_from_sequence_parallel_region(
                hidden_states,
                self.sequence_dimension,
            )

        return (hidden_states, next_decoder_cache)

    def _create_sliding_window_attn_mask(self, attention_mask: torch.Tensor, input_ids: torch.LongTensor):
        is_for_context_encoding = self._is_context_encoding(input_ids)
        sliding_window = self.sliding_window
        if is_for_context_encoding:
            sliding_window_overlay = (
                torch.ones(self.n_positions, self.n_positions).to(torch.bool).triu(diagonal=1 - sliding_window)
            ).to(attention_mask.device)
            sliding_window_overlay = sliding_window_overlay[None, None, :, :].expand(
                attention_mask.shape[0], 1, self.n_positions, self.n_positions
            )
            mask = torch.logical_and(attention_mask, sliding_window_overlay)
        else:
            mask = attention_mask.clone()
            sliding_window = min(sliding_window, self.n_positions)
            mask[:, :, :, :-sliding_window] = False
        return mask


class GptOssNxdForCausalLM(NxDModelForCausalLM):
    """
    GptOss model for NxD inference.
    """

    _model_cls = NeuronGptOssModel

    @classmethod
    def get_neuron_config_cls(cls):
        return NxDNeuronConfig

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: GptOssConfig, neuron_config: NxDNeuronConfig
    ) -> dict:
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
        return convert_gptoss_to_neuron_state_dict(state_dict, config, neuron_config)

    @classmethod
    def get_compiler_args(cls, neuron_config: NxDNeuronConfig) -> str:
        compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer -O1"
        # Add flags for cc-overlap
        compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
        compiler_args += " --auto-cast=none"
        # Enable vector-offset DGE
        compiler_args += " --internal-enable-dge-levels vector_dynamic_offsets"
        return compiler_args

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
        )
