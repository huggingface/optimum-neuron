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
"""PyTorch T5 model for NXD inference."""

import torch
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    BaseParallelLinear,
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.utils import divide
from torch import nn
from transformers import T5Config
from transformers.activations import ACT2FN
from transformers.cache_utils import EncoderDecoderCache
from transformers.models.t5.modeling_t5 import (
    T5Attention,
    T5DenseActDense,
    T5DenseGatedActDense,
    T5LayerCrossAttention,
    T5LayerFF,
    T5LayerNorm,
    T5LayerSelfAttention,
)
from transformers.pytorch_utils import find_pruneable_heads_and_indices


"""
T5 NxD custom modeling, copied from: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/neuronx_distributed/t5-inference/t5-inference-tutorial.html.
"""


def prune_linear_layer(layer: BaseParallelLinear, index: torch.LongTensor, dim: int = 0) -> BaseParallelLinear:
    """
    Prune a linear layer to keep only entries in index.

    Used to remove heads.

    Args:
        layer (`BaseParallelLinear`): The layer to prune.
        index (`torch.LongTensor`): The indices to keep in the layer.
        dim (`int`, *optional*, defaults to 0): The dimension on which to keep the indices.

    Returns:
        `BaseParallelLinear`: The pruned layer as a new layer with `requires_grad=True`.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = ColumnParallelLinear(new_size[1], new_size[0], bias=layer.bias is not None, gather_output=False).to(
        layer.weight.device
    )
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


class NeuronT5Attention(T5Attention):
    def __init__(
        self,
        config: T5Config,
        has_relative_attention_bias=False,
        layer_idx: int | None = None,
    ):
        super().__init__(config, has_relative_attention_bias, layer_idx)
        # Per attention head and per partition values
        world_size = parallel_state.get_tensor_model_parallel_size()
        self.num_attention_heads_per_partition = divide(self.n_heads, world_size)
        self.hidden_size_per_partition = self.num_attention_heads_per_partition * self.key_value_proj_dim

        # Mesh TensorFlow initialization to avoid scaling before softmax
        self.q = ColumnParallelLinear(self.d_model, self.inner_dim, bias=False, gather_output=False)
        self.k = ColumnParallelLinear(self.d_model, self.inner_dim, bias=False, gather_output=False)
        self.v = ColumnParallelLinear(self.d_model, self.inner_dim, bias=False, gather_output=False)
        self.o = RowParallelLinear(self.inner_dim, self.d_model, bias=False, input_is_parallel=True)

        if self.has_relative_attention_bias:
            self.relative_attention_bias = ParallelEmbedding(self.relative_attention_num_buckets, self.n_heads)
        self.n_heads = self.num_attention_heads_per_partition

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.num_attention_heads_per_partition, self.key_value_proj_dim, self.pruned_heads
        )
        # Prune linear layers
        self.q = prune_linear_layer(self.q, index)
        self.k = prune_linear_layer(self.k, index)
        self.v = prune_linear_layer(self.v, index)
        self.o = prune_linear_layer(self.o, index, dim=1)
        # Update hyper params
        self.num_attention_heads_per_partition = self.num_attention_heads_per_partition - len(heads)
        self.hidden_size_per_partition = self.key_value_proj_dim * self.num_attention_heads_per_partition
        self.pruned_heads = self.pruned_heads.union(heads)

    def compute_bias(self, query_length, key_length, device=None, cache_position=None):
        """Compute binned relative position bias"""
        if device is None:
            device = self.relative_attention_bias.weight.device
        if cache_position is None:
            context_position = torch.arange(query_length, dtype=torch.long, device=device)[:, None]
        else:
            context_position = cache_position[:, None].to(device)
        memory_position = torch.arange(key_length, dtype=torch.long, device=device)[None, :]
        relative_position = memory_position - context_position  # shape (query_length, key_length)
        relative_position_bucket = self._relative_position_bucket(
            relative_position,  # shape (query_length, key_length)
            bidirectional=(not self.is_decoder),
            num_buckets=self.relative_attention_num_buckets,
            max_distance=self.relative_attention_max_distance,
        )
        values = self.relative_attention_bias(relative_position_bucket)

        # TP
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        values = values[
            :,
            :,
            tp_rank * self.num_attention_heads_per_partition : (tp_rank + 1) * self.num_attention_heads_per_partition,
        ]

        values = values.permute([2, 0, 1]).unsqueeze(0)  # shape (1, num_heads, query_length, key_length)
        return values

    def forward(
        self,
        hidden_states,
        mask=None,
        key_value_states=None,
        position_bias=None,
        past_key_values=None,
        layer_head_mask=None,
        query_length=None,
        use_cache=False,
        output_attentions=False,
        cache_position=None,
    ):
        """
        Self-attention (if key_value_states is None) or attention over source sentence (provided by key_value_states).
        """
        # Input is (batch_size, seq_length, dim)
        # Mask is (batch_size, 1, 1, key_length) (non-causal encoder) or (batch_size, 1, seq_length, key_length) (causal decoder)
        self.is_decoder = True
        batch_size, seq_length = hidden_states.shape[:2]

        # if key_value_states are provided this layer is used as a cross-attention layer for the decoder
        is_cross_attention = key_value_states is not None

        query_states = self.q(hidden_states)
        query_states = query_states.view(
            batch_size, -1, self.num_attention_heads_per_partition, self.key_value_proj_dim
        ).transpose(1, 2)

        # Check is encoder-decoder model is being used. Otherwise we'll get `DynamicCache`
        is_updated = False
        if isinstance(past_key_values, EncoderDecoderCache):
            is_updated = past_key_values.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                curr_past_key_values = past_key_values.cross_attention_cache
            else:
                curr_past_key_values = past_key_values.self_attention_cache
        else:
            curr_past_key_values = past_key_values

        current_states = key_value_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_values is not None and is_updated:
            # reuse k,v, cross_attentions
            key_states = curr_past_key_values.layers[self.layer_idx].keys
            value_states = curr_past_key_values.layers[self.layer_idx].values
        else:
            key_states = self.k(current_states)
            value_states = self.v(current_states)
            key_states = key_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)
            value_states = value_states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

            if past_key_values is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = curr_past_key_values.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                if is_cross_attention and isinstance(past_key_values, EncoderDecoderCache):
                    past_key_values.is_updated[self.layer_idx] = True

        # compute scores, equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9
        scores = torch.matmul(query_states, key_states.transpose(3, 2))

        if position_bias is None:
            key_length = key_states.shape[-2]
            # cache position is 0-indexed so we add 1 to get the real length of queries (aka with past)
            real_seq_length = query_length if query_length is not None else cache_position[-1] + 1
            if not self.has_relative_attention_bias:
                position_bias = torch.zeros(
                    (1, self.num_attention_heads_per_partition, seq_length, key_length),
                    device=scores.device,
                    dtype=scores.dtype,
                )
                if self.gradient_checkpointing and self.training:
                    position_bias.requires_grad = True
            else:
                position_bias = self.compute_bias(
                    real_seq_length, key_length, device=scores.device, cache_position=cache_position
                )
                position_bias = position_bias[:, :, -seq_length:, :]

            if mask is not None:
                causal_mask = mask[:, :, :, : key_states.shape[-2]]
                position_bias = position_bias + causal_mask

        position_bias_masked = position_bias
        scores += position_bias_masked

        # (batch_size, n_heads, seq_length, key_length)
        attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(scores)
        attn_weights = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        # Mask heads if we want to
        if layer_head_mask is not None:
            attn_weights = attn_weights * layer_head_mask

        attn_output = torch.matmul(attn_weights, value_states)

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, -1, self.hidden_size_per_partition)
        attn_output = self.o(attn_output)

        outputs = (attn_output, position_bias)

        if output_attentions:
            outputs = outputs + (attn_weights,)
        return outputs


class NeuronT5LayerSelfAttention(T5LayerSelfAttention):
    def __init__(self, config, has_relative_attention_bias=False, layer_idx: int | None = None):
        super().__init__(config, has_relative_attention_bias=False, layer_idx=layer_idx)
        self.SelfAttention = NeuronT5Attention(
            config,
            has_relative_attention_bias=has_relative_attention_bias,
            layer_idx=layer_idx,
        )
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)


class NeuronT5LayerCrossAttention(T5LayerCrossAttention):
    def __init__(self, config, layer_idx: int | None = None):
        super().__init__(config)
        self.EncDecAttention = NeuronT5Attention(config, has_relative_attention_bias=False, layer_idx=layer_idx)
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)


class NeuronT5DenseActDense(T5DenseActDense):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.wi = ColumnParallelLinear(config.d_model, config.d_ff, gather_output=False, bias=False)
        self.wo = RowParallelLinear(config.d_ff, config.d_model, input_is_parallel=True, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]


class NeuronT5DenseGatedActDense(T5DenseGatedActDense):
    def __init__(self, config: T5Config):
        super().__init__(config)
        self.wi_0 = ColumnParallelLinear(config.d_model, config.d_ff, gather_output=False, bias=False)
        self.wi_1 = ColumnParallelLinear(config.d_model, config.d_ff, gather_output=False, bias=False)
        self.wo = RowParallelLinear(config.d_ff, config.d_model, input_is_parallel=True, bias=False)
        self.dropout = nn.Dropout(config.dropout_rate)
        self.act = ACT2FN[config.dense_act_fn]


class NeuronT5LayerFF(T5LayerFF):
    def __init__(self, config: T5Config):
        super().__init__(config)
        if config.is_gated_act:
            self.DenseReluDense = NeuronT5DenseGatedActDense(config)
        else:
            self.DenseReluDense = NeuronT5DenseActDense(config)

        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)


def parallelize(model):
    for index, block in enumerate(model.decoder.block):
        block.layer[0] = NeuronT5LayerSelfAttention(
            model.config,
            has_relative_attention_bias=bool(index == 0),
            layer_idx=index,
        )
        block.layer[1] = NeuronT5LayerCrossAttention(model.config, layer_idx=index)
        block.layer[2] = NeuronT5LayerFF(model.config)

    return model
