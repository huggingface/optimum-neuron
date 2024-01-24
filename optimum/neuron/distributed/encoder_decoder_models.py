# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Classes related to `neuronx-distributed` to perform parallelism."""

from typing import TYPE_CHECKING, Optional

import torch
from transformers.models.t5.modeling_t5 import T5Attention, T5ForSequenceClassification, T5LayerNorm

from ...utils import NormalizedConfigManager
from .base import Parallelizer, SequenceParallelismSpecs
from .parallel_layers import (
    LayerNormType,
    ParallelCrossEntropy,
    ParallelEmbedding,
    ParallelMLP,
    ParallelSelfAttention,
    SequenceCollectiveOpInfo,
)
from .utils import linear_to_parallel_linear


if TYPE_CHECKING:
    from transformers import PreTrainedModel


class T5ParallelEmbedding(ParallelEmbedding):
    EMBEDDING_NAME = "shared"
    LM_HEAD_NAME = "lm_head"


class T5ParallelSelfAttention(ParallelSelfAttention):
    QUERIES_NAME = "q"
    KEYS_NAME = "k"
    VALUES_NAME = "v"
    OUTPUT_PROJECTION_NAME = "o"
    NUM_ATTENTION_HEADS_NAME = "n_heads"
    ALL_HEAD_SIZE_NAME = "inner_dim"

    @classmethod
    def transform(
        cls,
        model: "PreTrainedModel",
        layer: "torch.nn.Module",
        device: Optional["torch.device"] = None,
        sequence_parallel_enabled: bool = False,
    ) -> "torch.nn.Module":
        from neuronx_distributed.parallel_layers.parallel_state import (
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_size,
        )
        from neuronx_distributed.parallel_layers.utils import set_tensor_model_parallel_attributes

        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_size()

        config = model.config
        normalized_config = NormalizedConfigManager.get_normalized_config_class(config.model_type)(config)
        if cls.NUM_ATTENTION_HEADS_NAME is None:
            num_attention_heads_name = normalized_config.NUM_ATTENTION_HEADS
        else:
            num_attention_heads_name = cls.NUM_ATTENTION_HEADS_NAME

        num_attention_heads = getattr(layer, num_attention_heads_name)
        num_attention_heads_per_rank = num_attention_heads // tp_size

        if layer.has_relative_attention_bias:
            with torch.no_grad():
                layer.relative_attention_bias.weight.data = layer.relative_attention_bias.weight.data[
                    :, num_attention_heads_per_rank * tp_rank : num_attention_heads_per_rank * (tp_rank + 1)
                ]
                layer.relative_attention_bias.num_embeddings = num_attention_heads_per_rank
                set_tensor_model_parallel_attributes(layer.relative_attention_bias.weight, True, 1, stride=1)

        layer = super().transform(model, layer, sequence_parallel_enabled=sequence_parallel_enabled, device=device)

        return layer


class T5ParallelMLP(ParallelMLP):
    FIRST_LINEAR_NAME = "wi"
    SECOND_LINEAR_NAME = "wo"

    @classmethod
    def transform(
        cls,
        model: "PreTrainedModel",
        layer: "torch.nn.Module",
        sequence_parallel_enabled: bool = False,
        device: Optional["torch.device"] = None,
    ) -> "torch.nn.Module":
        from transformers.models.t5.modeling_t5 import T5DenseGatedActDense

        if cls.FIRST_LINEAR_NAME is None or cls.SECOND_LINEAR_NAME is None:
            raise ValueError("Both `FIRST_LINEAR_NAME` and `SECOND_LINEAR_NAME` class attributes must be set.")

        orig_first_linear_name = cls.FIRST_LINEAR_NAME

        if isinstance(layer, T5DenseGatedActDense):
            # Changing the name of the first linear layer to match wi_0.
            cls.FIRST_LINEAR_NAME = f"{orig_first_linear_name}_0"

        # This will parallelize both wi_0 and wo.
        layer = super().transform(model, layer, sequence_parallel_enabled=sequence_parallel_enabled, device=device)

        if isinstance(layer, T5DenseGatedActDense):
            # In this case, only wi_1 remains to be parallelized, we do it here.
            cls.FIRST_LINEAR_NAME = f"{orig_first_linear_name}_1"
            layer_to_fully_qualified_name = {id(module): name for name, module in model.named_modules()}
            weight_map = getattr(model, "_weight_map", None)

            linear_layer_weight_info, linear_layer_bias_weight_info = None, None
            module, attribute_name = cls._get_module_and_attribute_name(layer, cls.FIRST_LINEAR_NAME)
            if weight_map is not None:
                layer_qualified_name = layer_to_fully_qualified_name[id(module)]
                linear_layer_weight_info, linear_layer_bias_weight_info = cls._get_linear_weight_info(
                    weight_map,
                    f"{layer_qualified_name}.{attribute_name}",
                    device=device,
                )

            setattr(
                module,
                attribute_name,
                linear_to_parallel_linear(
                    getattr(module, attribute_name),
                    "column",
                    gather_output=False,
                    linear_layer_weight_info=linear_layer_weight_info,
                    linear_layer_bias_weight_info=linear_layer_bias_weight_info,
                    sequence_parallel_enabled=sequence_parallel_enabled,
                    device=device,
                ),
            )

        cls.FIRST_LINEAR_NAME = orig_first_linear_name

        return layer


class T5ParallelCrossEntropy(ParallelCrossEntropy):
    LAST_LINEAR_PROJECTION_NAME = {"T5ForConditionalGeneration": "lm_head"}


class T5SequenceParallelismSpecs(SequenceParallelismSpecs):
    SEQUENCE_PARALLEL_LAYERNORM_PATTERNS = [
        "encoder.block.[0-9]+.layer.[0-9]+.layer_norm",
        "encoder.final_layer_norm",
        "decoder.block.[0-9]+.layer.[0-9]+.layer_norm",
        "decoder.final_layer_norm",
    ]

    LAYERNORM_TYPE = LayerNormType.RMS_NORM
    SEQUENCE_COLLECTIVE_OPS_INFOS = [
        SequenceCollectiveOpInfo("scatter", "shared", "output", "first"),
        SequenceCollectiveOpInfo("gather", T5LayerNorm, "output", "last"),
    ]

    @classmethod
    def patch_for_sequence_parallelism(cls, model: "PreTrainedModel", sequence_parallel_enabled: bool):
        if not sequence_parallel_enabled:
            return

        from torch import nn

        def sequence_parallel_forward(
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
        ):
            # Without sequence parallelism:
            # Input is (batch_size, seq_length, dim)
            # Mask is (batch_size, key_length) (non-causal) or (batch_size, key_length, key_length)
            # past_key_value[0] is (batch_size, n_heads, q_len - 1, dim_per_head)
            if sequence_parallel_enabled:
                batch_size = hidden_states.shape[1]
            else:
                batch_size = hidden_states.shape[0]

            def shape(states):
                """projection"""
                if sequence_parallel_enabled:
                    return states.view(-1, batch_size, self.n_heads, self.key_value_proj_dim).permute(1, 2, 0, 3)
                return states.view(batch_size, -1, self.n_heads, self.key_value_proj_dim).transpose(1, 2)

            def unshape(states):
                """reshape"""
                if sequence_parallel_enabled:
                    return states.permute(2, 0, 1, 3).view(-1, batch_size, self.inner_dim)
                return states.transpose(1, 2).contiguous().view(batch_size, -1, self.inner_dim)

            def project(hidden_states, proj_layer, key_value_states, past_key_value):
                """projects hidden states correctly to key/query states"""
                if key_value_states is None:
                    # self-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(hidden_states))
                elif past_key_value is None:
                    # cross-attn
                    # (batch_size, n_heads, seq_length, dim_per_head)
                    hidden_states = shape(proj_layer(key_value_states))

                if past_key_value is not None:
                    if key_value_states is None:
                        # self-attn
                        # (batch_size, n_heads, key_length, dim_per_head)
                        if sequence_parallel_enabled:
                            hidden_states = torch.cat([past_key_value, hidden_states.transpose(0, 1)], dim=2)
                        else:
                            hidden_states = torch.cat([past_key_value, hidden_states], dim=2)
                    elif past_key_value.shape[2] != key_value_states.shape[1]:
                        # checking that the `sequence_length` of the `past_key_value` is the same as
                        # the provided `key_value_states` to support prefix tuning
                        # cross-attn
                        # (batch_size, n_heads, seq_length, dim_per_head)
                        hidden_states = shape(proj_layer(key_value_states))
                    else:
                        # cross-attn
                        hidden_states = past_key_value
                return hidden_states

            # get query states
            query_states = shape(self.q(hidden_states))  # (batch_size, n_heads, seq_length, dim_per_head)

            # get key/value states
            key_states = project(
                hidden_states, self.k, key_value_states, past_key_value[0] if past_key_value is not None else None
            )
            value_states = project(
                hidden_states, self.v, key_value_states, past_key_value[1] if past_key_value is not None else None
            )

            real_seq_length = key_states.shape[2]

            if past_key_value is not None:
                if len(past_key_value) != 2:
                    raise ValueError(
                        f"past_key_value should have 2 past states: keys and values. Got { len(past_key_value)} past states"
                    )
                real_seq_length += past_key_value[0].shape[2] if query_length is None else query_length

            key_length = real_seq_length if key_value_states is None else key_value_states.shape[1]

            # compute scores
            scores = torch.matmul(
                query_states, key_states.transpose(3, 2)
            )  # equivalent of torch.einsum("bnqd,bnkd->bnqk", query_states, key_states), compatible with onnx op>9

            if position_bias is None:
                if not self.has_relative_attention_bias:
                    position_bias = torch.zeros(
                        (1, self.n_heads, real_seq_length, key_length), device=scores.device, dtype=scores.dtype
                    )
                    if self.gradient_checkpointing and self.training:
                        position_bias.requires_grad = True
                else:
                    position_bias = self.compute_bias(real_seq_length, key_length, device=scores.device)

                # if key and values are already calculated
                # we want only the last query position bias
                if past_key_value is not None:
                    position_bias = position_bias[:, :, -hidden_states.size(1) :, :]

                if mask is not None:
                    position_bias = position_bias + mask  # (batch_size, n_heads, seq_length, key_length)

            if self.pruned_heads:
                mask = torch.ones(position_bias.shape[1])
                mask[list(self.pruned_heads)] = 0
                position_bias_masked = position_bias[:, mask.bool()]
            else:
                position_bias_masked = position_bias

            scores += position_bias_masked
            attn_weights = nn.functional.softmax(scores.float(), dim=-1).type_as(
                scores
            )  # (batch_size, n_heads, seq_length, key_length)
            attn_weights = nn.functional.dropout(
                attn_weights, p=self.dropout, training=self.training
            )  # (batch_size, n_heads, seq_length, key_length)

            # Mask heads if we want to
            if layer_head_mask is not None:
                attn_weights = attn_weights * layer_head_mask

            attn_output = unshape(torch.matmul(attn_weights, value_states))  # (batch_size, seq_length, dim)
            attn_output = self.o(attn_output)

            present_key_value_state = (key_states, value_states) if (self.is_decoder and use_cache) else None
            outputs = (attn_output,) + (present_key_value_state,) + (position_bias,)

            if output_attentions:
                outputs = outputs + (attn_weights,)
            return outputs

        for module in model.modules():
            if isinstance(module, T5Attention):
                module.forward = sequence_parallel_forward.__get__(module)


class T5Parallelizer(Parallelizer):
    @classmethod
    def _parallelize(
        cls,
        model: "PreTrainedModel",
        device: Optional["torch.device"] = None,
        parallelize_embeddings: bool = True,
        sequence_parallel_enabled: bool = False,
    ) -> "PreTrainedModel":
        if isinstance(model, T5ForSequenceClassification):
            raise NotImplementedError(
                "Model parallelism is currently not supported for T5ForSequenceClassification. Please open an issue to "
                "request support or submit a PR to implement it in the optimum-neuron repo "
                "(https://github.com/huggingface/optimum-neuron)."
            )
        if parallelize_embeddings:
            model = T5ParallelEmbedding.transform(
                model, model, sequence_parallel_enabled=sequence_parallel_enabled, device=device
            )
        if parallelize_embeddings and model.encoder.embed_tokens is not None:
            model.encoder.embed_tokens = model.shared
        if parallelize_embeddings and model.decoder.embed_tokens is not None:
            model.decoder.embed_tokens = model.shared
        for block in model.encoder.block:
            block.layer[0].SelfAttention = T5ParallelSelfAttention.transform(
                model,
                block.layer[0].SelfAttention,
                sequence_parallel_enabled=sequence_parallel_enabled,
                device=device,
            )
            block.layer[1].DenseReluDense = T5ParallelMLP.transform(
                model,
                block.layer[1].DenseReluDense,
                sequence_parallel_enabled=sequence_parallel_enabled,
                device=device,
            )
        for block in model.decoder.block:
            block.layer[0].SelfAttention = T5ParallelSelfAttention.transform(
                model,
                block.layer[0].SelfAttention,
                sequence_parallel_enabled=sequence_parallel_enabled,
                device=device,
            )
            block.layer[1].EncDecAttention = T5ParallelSelfAttention.transform(
                model,
                block.layer[1].EncDecAttention,
                sequence_parallel_enabled=sequence_parallel_enabled,
                device=device,
            )
            block.layer[2].DenseReluDense = T5ParallelMLP.transform(
                model,
                block.layer[2].DenseReluDense,
                sequence_parallel_enabled=sequence_parallel_enabled,
                device=device,
            )
        if parallelize_embeddings:
            model = T5ParallelCrossEntropy.transform(
                model, model, sequence_parallel_enabled=sequence_parallel_enabled, device=device
            )
        return model
