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

from typing import TYPE_CHECKING, Dict, Optional

from .base import Parallelizer
from .parallel_layers import ParallelEmbedding, ParallelMLP, ParallelSelfAttention
from .utils import linear_to_parallel_linear


if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel


class GPTNeoParallelEmbedding(ParallelEmbedding):
    EMBEDDING_NAME = "transformer.wte"
    LM_HEAD_NAME = "lm_head"


class GPTNeoParallelSelfAttention(ParallelSelfAttention):
    QUERIES_NAME = "q_proj"
    KEYS_NAME = "k_proj"
    VALUES_NAME = "v_proj"
    OUTPUT_PROJECTION_NAME = "out_proj"
    ALL_HEAD_SIZE_NAME = "embed_dim"


class GPTNeoParallelMLP(ParallelMLP):
    FIRST_LINEAR_NAME = "c_fc"
    SECOND_LINEAR_NAME = "c_proj"


class GPTNeoParallelizer(Parallelizer):
    @classmethod
    def _parallelize(
        cls,
        model: "PreTrainedModel",
        orig_to_parallel: Optional[Dict[int, "torch.nn.Parameter"]],
        device: Optional["torch.device"] = None,
        parallelize_embeddings: bool = True,
    ) -> "PreTrainedModel":
        if parallelize_embeddings:
            model = GPTNeoParallelEmbedding.transform(model, model, device=device)
        for block in model.transformer.h:
            block.attn.attention = GPTNeoParallelSelfAttention.transform(
                model,
                block.attn.attention,
                device=device,
            )
            block.mlp = GPTNeoParallelMLP.transform(model, block.mlp, device=device)
        return model


class LlamaParallelEmbedding(ParallelEmbedding):
    EMBEDDING_NAME = "model.embed_tokens"
    LM_HEAD_NAME = "lm_head"


class LlamaParallelSelfAttention(ParallelSelfAttention):
    QUERIES_NAME = "q_proj"
    KEYS_NAME = "k_proj"
    VALUES_NAME = "v_proj"
    OUTPUT_PROJECTION_NAME = "o_proj"
    NUM_ATTENTION_HEADS_NAME = "num_heads"
    NUM_KEY_VALUE_HEADS_NAME = "num_key_value_heads"
    NUM_KEY_VALUE_GROUPS_NAME = "num_key_value_groups"
    ALL_HEAD_SIZE_NAME = "hidden_size"


class LLamaParallelMLP(ParallelMLP):
    FIRST_LINEAR_NAME = "up_proj"
    SECOND_LINEAR_NAME = "down_proj"

    @classmethod
    def transform(
        cls,
        model: "PreTrainedModel",
        layer: "torch.nn.Module",
        orig_to_parallel: Optional[Dict[int, "torch.nn.Parameter"]] = None,
        device: Optional["torch.device"] = None,
    ) -> "torch.nn.Module":
        # TODO: Make it smart by merging the gate and the up_proj.
        # WARNING: be careful of the interleaved outputs when doing TP!
        layer = super().transform(model, layer, orig_to_parallel=orig_to_parallel, device=device)

        weight_map = getattr(model, "_weight_map", None)

        module, attribute_name = cls._get_module_and_attribute_name(layer, "gate_proj")
        linear_layer_weight_info, linear_layer_bias_weight_info = None, None
        layer_qualified_name = ""
        if weight_map is not None:
            layer_to_fully_qualified_name = {id(module): name for name, module in model.named_modules()}
            layer_qualified_name = layer_to_fully_qualified_name[id(layer)]
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
                orig_to_parallel=orig_to_parallel,
                device=device,
            ),
        )
        return layer


class LlamaParallelizer(Parallelizer):
    @classmethod
    def _parallelize(
        cls,
        model: "PreTrainedModel",
        orig_to_parallel: Optional[Dict[int, "torch.nn.Parameter"]],
        device: Optional["torch.device"] = None,
        parallelize_embeddings: bool = True,
    ) -> "PreTrainedModel":
        if parallelize_embeddings:
            model = LlamaParallelEmbedding.transform(model, model, device=device)
        for layer in model.model.layers:
            layer.self_attn = LlamaParallelSelfAttention.transform(model, layer.self_attn, device=device)
            layer.mlp = LLamaParallelMLP.transform(model, layer.mlp, device=device)
        return model
