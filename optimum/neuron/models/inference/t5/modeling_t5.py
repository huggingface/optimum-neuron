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

from typing import TYPE_CHECKING

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

from optimum.neuron.utils import f32Wrapper


if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel


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


# Adapted from https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/inference/hf_pretrained_pixart_alpha_inference_on_inf2.ipynb
# For text encoding
class T5EncoderWrapper(torch.nn.Module):
    def __init__(
        self,
        model: "PreTrainedModel",
        sequence_length: int,
        batch_size: int | None = None,
        device: str = "xla",
        tensor_parallel_size: int = 1,
    ):
        super().__init__()
        self.model = model
        self.config = model.config
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.device = device
        self.tensor_parallel_size = tensor_parallel_size

        for block in self.model.encoder.block:
            block.layer[1].DenseReluDense.act = torch.nn.GELU(approximate="tanh")
        precomputed_bias = (
            self.model.encoder.block[0].layer[0].SelfAttention.compute_bias(self.sequence_length, self.sequence_length)
        )
        if self.tensor_parallel_size > 1:
            self.model = self.parallelize(self.model)
            precomputed_bias_tp = self.shard_weights(precomputed_bias, 1)
            self.model.encoder.block[0].layer[0].SelfAttention.compute_bias = (
                lambda *args, **kwargs: precomputed_bias_tp
            )
        else:
            self.model.encoder.block[0].layer[0].SelfAttention.compute_bias = lambda *args, **kwargs: precomputed_bias

    def parallelize(self, model):
        for _, block in enumerate(model.encoder.block):
            selfAttention = block.layer[0].SelfAttention
            ff = block.layer[1]
            layer_norm_0 = block.layer[0].layer_norm.to(torch.float32)
            layer_norm_1 = block.layer[1].layer_norm.to(torch.float32)
            block.layer[1] = self.shard_ff(ff)
            block.layer[0].SelfAttention = self.shard_self_attention(selfAttention, self.tensor_parallel_size)
            block.layer[0].layer_norm = f32Wrapper(layer_norm_0)
            block.layer[1].layer_norm = f32Wrapper(layer_norm_1)
        final_layer_norm = model.encoder.final_layer_norm.to(torch.float32)
        model.encoder.final_layer_norm = f32Wrapper(final_layer_norm)

        return model

    @staticmethod
    def shard_weights(data, dim):
        tp_rank = parallel_state.get_tensor_model_parallel_rank()
        s = data.shape[dim] // parallel_state.get_tensor_model_parallel_size()
        if dim == 0:
            return data[s * tp_rank : s * (tp_rank + 1)].clone()
        elif dim == 1:
            return data[:, s * tp_rank : s * (tp_rank + 1)].clone()

    @staticmethod
    def shard_ff(ff: "T5LayerFF"):
        orig_wi_0 = ff.DenseReluDense.wi_0  # only applicable for T5 with gated silu`
        ff.DenseReluDense.wi_0 = ColumnParallelLinear(
            orig_wi_0.in_features, orig_wi_0.out_features, bias=False, gather_output=False
        )
        ff.DenseReluDense.wi_0.weight.data = T5EncoderWrapper.shard_weights(orig_wi_0.weight.data, 0)
        orig_wi_1 = ff.DenseReluDense.wi_1
        ff.DenseReluDense.wi_1 = ColumnParallelLinear(
            orig_wi_1.in_features, orig_wi_1.out_features, bias=False, gather_output=False
        )
        ff.DenseReluDense.wi_1.weight.data = T5EncoderWrapper.shard_weights(orig_wi_1.weight.data, 0)
        orig_wo = ff.DenseReluDense.wo
        ff.DenseReluDense.wo = RowParallelLinear(
            orig_wo.in_features, orig_wo.out_features, bias=False, input_is_parallel=True
        )
        ff.DenseReluDense.wo.weight.data = T5EncoderWrapper.shard_weights(orig_wo.weight.data, 1)
        ff.DenseReluDense.act = torch.nn.GELU(approximate="tanh")
        return ff

    @staticmethod
    def shard_self_attention(selfAttention: "T5Attention", tensor_parallel_size: int):
        orig_inner_dim = selfAttention.q.out_features
        dim_head = orig_inner_dim // selfAttention.n_heads
        selfAttention.n_heads = selfAttention.n_heads // tensor_parallel_size
        selfAttention.inner_dim = dim_head * selfAttention.n_heads
        orig_q = selfAttention.q
        selfAttention.q = ColumnParallelLinear(
            selfAttention.q.in_features, selfAttention.q.out_features, bias=False, gather_output=False
        )
        selfAttention.q.weight.data = T5EncoderWrapper.shard_weights(orig_q.weight.data, 0)
        del orig_q
        orig_k = selfAttention.k
        selfAttention.k = ColumnParallelLinear(
            selfAttention.k.in_features,
            selfAttention.k.out_features,
            bias=(selfAttention.k.bias is not None),
            gather_output=False,
        )
        selfAttention.k.weight.data = T5EncoderWrapper.shard_weights(orig_k.weight.data, 0)
        del orig_k
        orig_v = selfAttention.v
        selfAttention.v = ColumnParallelLinear(
            selfAttention.v.in_features,
            selfAttention.v.out_features,
            bias=(selfAttention.v.bias is not None),
            gather_output=False,
        )
        selfAttention.v.weight.data = T5EncoderWrapper.shard_weights(orig_v.weight.data, 0)
        del orig_v
        orig_out = selfAttention.o
        selfAttention.o = RowParallelLinear(
            selfAttention.o.in_features,
            selfAttention.o.out_features,
            bias=(selfAttention.o.bias is not None),
            input_is_parallel=True,
        )
        selfAttention.o.weight.data = T5EncoderWrapper.shard_weights(orig_out.weight.data, 1)
        del orig_out
        return selfAttention

    def forward(self, input_ids):
        return self.model(input_ids, output_hidden_states=False)


# Adapted from https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/torch-neuronx/t5-inference-tutorial.html
# For text encoding + KV cache initialization
class T5EncoderForSeq2SeqLMWrapper(torch.nn.Module):
    """Wrapper to trace the encoder and the kv cache initialization in the decoder."""

    def __init__(
        self,
        model: "PreTrainedModel",
        sequence_length: int | None = None,
        batch_size: int | None = None,
        num_beams: int = 1,
        device: str = "xla",
        tensor_parallel_size: int = 1,
    ):
        super().__init__()
        self.model = model
        self.config = model.config
        self.num_beams = num_beams
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.device = device
        self.tensor_parallel_size = tensor_parallel_size
        self.num_attention_heads_per_partition = self.config.num_heads  # when tensor_parallel_size=1

        if self.tensor_parallel_size > 1:
            self.num_attention_heads_per_partition = (
                self.num_attention_heads_per_partition // parallel_state.get_tensor_model_parallel_size()
            )
            self.past_key_values_sa = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(
                        torch.ones(
                            (
                                self.num_beams * batch_size,
                                self.num_attention_heads_per_partition,
                                self.sequence_length - 1,
                                self.config.d_kv,
                            ),
                            dtype=torch.float32,
                        ),
                        requires_grad=False,
                    )
                    for _ in range(self.config.num_decoder_layers * 2)
                ]
            )
            self.past_key_values_ca = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(
                        torch.ones(
                            (
                                self.num_beams * batch_size,
                                self.num_attention_heads_per_partition,
                                self.sequence_length,
                                self.config.d_kv,
                            ),
                            dtype=torch.float32,
                        ),
                        requires_grad=False,
                    )
                    for _ in range(self.config.num_decoder_layers * 2)
                ]
            )

    def forward(self, input_ids, attention_mask):
        # Infer shapes of dummy inputs used for tracing
        batch_size = input_ids.shape[0]
        sequence_length = input_ids.shape[1]
        if self.sequence_length is not None:
            assert self.sequence_length, (
                f"Different sequence length for the parallel partition({self.sequence_length}) and for dummy inputs({sequence_length}). Make sure that they have the same value."
            )
        if self.batch_size is not None:
            assert self.batch_size, (
                f"Different batch size for the parallel partition({self.batch_size}) and for dummy inputs({batch_size}). Make sure that they have the same value."
            )

        encoder_output = self.model.encoder(
            input_ids=input_ids, attention_mask=attention_mask, output_attentions=False, output_hidden_states=False
        )

        last_hidden_state = encoder_output["last_hidden_state"]
        encoder_hidden_states = torch.concat(
            [tensor.unsqueeze(0).repeat(self.num_beams, 1, 1) for tensor in last_hidden_state]
        )

        decoder_blocks = self.model.decoder.block
        present_key_value_states_sa = []
        present_key_value_states_ca = []

        for i, block in enumerate(decoder_blocks):
            # Cross attention has to be initialized with the encoder hidden state
            cross_attention: T5LayerCrossAttention = block.layer[1]
            attention = cross_attention.EncDecAttention

            def shape(states):
                """projection"""
                return states.view(
                    self.num_beams * batch_size,
                    -1,
                    self.num_attention_heads_per_partition,
                    attention.key_value_proj_dim,
                ).transpose(1, 2)

            key_states = shape(attention.k(encoder_hidden_states))
            value_states = shape(attention.v(encoder_hidden_states))

            if not self.tensor_parallel_size > 1:
                # cross_attn_kv_state
                present_key_value_states_ca.append(key_states)
                present_key_value_states_ca.append(value_states)

                # Self attention kv states are initialized to zeros. This is done to keep the size of the kv cache tensor constant.
                # The kv cache is padded here to keep a fixed shape.
                # [key states]
                present_key_value_states_sa.append(
                    torch.zeros(
                        (self.num_beams * batch_size, self.config.num_heads, sequence_length - 1, self.config.d_kv),
                        dtype=torch.float32,
                        device=self.device,
                    )
                )
                # [value states]
                present_key_value_states_sa.append(
                    torch.zeros(
                        (self.num_beams * batch_size, self.config.num_heads, sequence_length - 1, self.config.d_kv),
                        dtype=torch.float32,
                        device=self.device,
                    )
                )
            else:
                present_key_value_states_ca.append((self.past_key_values_ca[i * 2] * 0) + key_states)
                present_key_value_states_ca.append((self.past_key_values_ca[i * 2 + 1] * 0) + value_states)
                present_key_value_states_sa.append(
                    self.past_key_values_sa[i * 2]
                    * torch.zeros(
                        (
                            self.num_beams * self.batch_size,
                            self.num_attention_heads_per_partition,
                            self.sequence_length - 1,
                            self.config.d_kv,
                        ),
                        dtype=torch.float32,
                        device=self.device,
                    )
                )
                present_key_value_states_sa.append(
                    self.past_key_values_sa[i * 2 + 1]
                    * torch.zeros(
                        (
                            self.num_beams * self.batch_size,
                            self.num_attention_heads_per_partition,
                            self.sequence_length - 1,
                            self.config.d_kv,
                        ),
                        dtype=torch.float32,
                        device=self.device,
                    )
                )

        return present_key_value_states_sa + present_key_value_states_ca


# Adapted from https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/torch-neuronx/t5-inference-tutorial.html
class T5DecoderWrapper(torch.nn.Module):
    """Wrapper to trace the decoder with past keys values with a language head."""

    def __init__(
        self,
        model: "PreTrainedModel",
        batch_size: int,
        sequence_length: int,
        num_beams: int = 1,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        device: str = "xla",
        tensor_parallel_size: int = 1,
    ):
        super().__init__()
        self.model = model
        self.config = model.config
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_beams = num_beams
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions
        self.device = device
        self.tensor_parallel_size = tensor_parallel_size

        self.num_attention_heads_per_partition = self.config.num_heads
        if tensor_parallel_size > 1:
            self.num_attention_heads_per_partition = (
                self.num_attention_heads_per_partition // parallel_state.get_tensor_model_parallel_size()
            )

        # Initialize KV cache (num_beams, n_heads, seq_length, dim_per_head)
        if device == "cpu":
            self.past_key_values_sa = [
                torch.ones(
                    (num_beams, self.config.num_heads, self.sequence_length - 1, self.config.d_kv), dtype=torch.float32
                )
                for _ in range(self.config.num_decoder_layers * 2)
            ]
            self.past_key_values_ca = [
                torch.ones(
                    (num_beams, self.config.num_heads, self.sequence_length, self.config.d_kv), dtype=torch.float32
                )
                for _ in range(self.config.num_decoder_layers * 2)
            ]
        elif device == "xla":
            self.past_key_values_sa = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(
                        torch.ones(
                            (
                                self.batch_size * self.num_beams,
                                self.num_attention_heads_per_partition,
                                sequence_length - 1,
                                self.config.d_kv,
                            ),
                            dtype=torch.float32,
                        ),
                        requires_grad=False,
                    )
                    for _ in range(self.config.num_decoder_layers * 2)
                ]
            )
            self.past_key_values_ca = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(
                        torch.ones(
                            (
                                self.batch_size * self.num_beams,
                                self.num_attention_heads_per_partition,
                                sequence_length,
                                self.config.d_kv,
                            ),
                            dtype=torch.float32,
                        ),
                        requires_grad=False,
                    )
                    for _ in range(self.config.num_decoder_layers * 2)
                ]
            )

    def update_past(self, past_key_values):
        new_past_sa = []
        new_past_ca = []
        for past_layer in past_key_values:
            new_past_layer = list(past_layer)
            for i in range(len(new_past_layer[:2])):
                new_past_layer[i] = past_layer[i][:, :, 1:]
            new_past_sa += [
                new_past_layer[:2],
            ]
            new_past_ca += [
                new_past_layer[2:],
            ]
        return new_past_sa, new_past_ca

    def reorder_cache(self, past_key_values, beam_idx):
        for i in range(len(past_key_values)):
            gather_index = beam_idx.view([beam_idx.shape[0], 1, 1, 1]).expand_as(past_key_values[i])
            past_key_values[i] = torch.gather(past_key_values[i], dim=0, index=gather_index)
        return past_key_values

    def forward(
        self,
        input_ids,
        decoder_attention_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        beam_idx,
        beam_scores,
        **kwargs,
    ):
        if self.num_beams > 1:
            # We reorder the cache based on the beams selected in each iteration. Required step for beam search.
            past_key_values_sa = self.reorder_cache(self.past_key_values_sa, beam_idx)
            past_key_values_ca = self.reorder_cache(self.past_key_values_ca, beam_idx)
        else:
            # We do not need to reorder for greedy sampling
            past_key_values_sa = self.past_key_values_sa
            past_key_values_ca = self.past_key_values_ca

        # The cache is stored in a flatten form. We order the cache per layer before passing it to the decoder.
        # Each layer has 4 tensors, so we group by 4.
        past_key_values = [
            [*past_key_values_sa[i * 2 : i * 2 + 2], *past_key_values_ca[i * 2 : i * 2 + 2]]
            for i in range(0, int(len(past_key_values_ca) / 2))
        ]

        decoder_output = self.model.decoder(
            input_ids=input_ids,
            attention_mask=decoder_attention_mask,
            past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values),
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
        )

        last_hidden_state = decoder_output["last_hidden_state"]
        past_key_values = decoder_output["past_key_values"]
        if self.output_hidden_states:
            decoder_hidden_states = list(
                decoder_output["hidden_states"]
            )  # flatten `hidden_states` which is a tuple of tensors

        if self.output_attentions:
            decoder_attentions = list(
                decoder_output["attentions"]
            )  # flatten `decoder_attentions` which is a tuple of tensors
            cross_attentions = list(
                decoder_output["cross_attentions"]
            )  # flatten `cross_attentions` which is a tuple of tensors

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            last_hidden_state = last_hidden_state * (self.model.config.d_model**-0.5)

        lm_logits = self.model.lm_head(last_hidden_state)

        past_key_values_sa, past_key_values_ca = self.update_past(past_key_values)

        # We flatten the cache to a single array. This is required for the input output aliasing to work
        past_key_values_sa = [vec for kv_per_layer in past_key_values_sa for vec in kv_per_layer]
        past_key_values_ca = [vec for kv_per_layer in past_key_values_ca for vec in kv_per_layer]

        if self.device == "cpu":
            self.past_key_values_sa = past_key_values_sa
            self.past_key_values_ca = past_key_values_ca

        # We calculate topk inside the wrapper
        next_token_logits = lm_logits[:, -1, :]

        if self.num_beams > 1:
            # This section of beam search is run outside the decoder in the huggingface t5 implementation.
            # To maximize the computation within the neuron device, we move this within the wrapper
            logit_max, _ = torch.max(next_token_logits, dim=-1, keepdim=True)
            logsumexp = torch.log(torch.exp(next_token_logits - logit_max).sum(dim=-1, keepdim=True))
            next_token_scores = next_token_logits - logit_max - logsumexp
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(self.batch_size, self.num_beams * vocab_size)
            next_token_scores = next_token_scores * 1

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * self.num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            neuron_outputs = [next_token_scores, next_tokens, next_indices] + past_key_values_sa + past_key_values_ca

        else:
            # Greedy
            next_tokens = torch.argmax(next_token_logits, dim=-1)

            neuron_outputs = [next_tokens] + past_key_values_sa + past_key_values_ca

        if self.output_hidden_states:
            neuron_outputs += decoder_hidden_states

        if self.output_attentions:
            neuron_outputs += decoder_attentions
            neuron_outputs += cross_attentions

        return neuron_outputs


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
        past_key_value=None,
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

        if past_key_value is not None:
            is_updated = past_key_value.is_updated.get(self.layer_idx)
            if is_cross_attention:
                # after the first generated id, we can subsequently re-use all key/value_states from cache
                curr_past_key_value = past_key_value.cross_attention_cache
            else:
                curr_past_key_value = past_key_value.self_attention_cache

        current_states = key_value_states if is_cross_attention else hidden_states
        if is_cross_attention and past_key_value is not None and is_updated:
            # reuse k,v, cross_attentions
            key_states = curr_past_key_value.key_cache[self.layer_idx]
            value_states = curr_past_key_value.value_cache[self.layer_idx]
        else:
            key_states = self.k(current_states)
            value_states = self.v(current_states)
            key_states = key_states.view(
                batch_size, -1, self.num_attention_heads_per_partition, self.key_value_proj_dim
            ).transpose(1, 2)
            value_states = value_states.view(
                batch_size, -1, self.num_attention_heads_per_partition, self.key_value_proj_dim
            ).transpose(1, 2)

            if past_key_value is not None:
                # save all key/value_states to cache to be re-used for fast auto-regressive generation
                cache_position = cache_position if not is_cross_attention else None
                key_states, value_states = curr_past_key_value.update(
                    key_states, value_states, self.layer_idx, {"cache_position": cache_position}
                )
                # set flag that curr layer for cross-attn is already updated so we can re-use in subsequent calls
                if is_cross_attention:
                    past_key_value.is_updated[self.layer_idx] = True

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

        if self.pruned_heads:
            mask = torch.ones(position_bias.shape[1])
            mask[list(self.pruned_heads)] = 0
            position_bias_masked = position_bias[:, mask.bool()]
        else:
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
