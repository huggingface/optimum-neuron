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
from .parallel_layers import ParallelMLP, ParallelSelfAttention
from .utils import WeightInformation, embedding_to_parallel_embedding, linear_to_parallel_linear


if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedModel


class GPTNeoParallelSelfAttention(ParallelSelfAttention):
    QUERIES_NAME = "q_proj"
    KEYS_NAME = "k_proj"
    VALUES_NAME = "v_proj"
    OUTPUT_PROJECTION_NAME = "out_proj"
    ALL_HEAD_SIZE_NAME = "embed_dim"


class GPTNeoParallelizer(Parallelizer):
    @classmethod
    def _parallelize(
        cls,
        model: "PreTrainedModel",
        orig_to_parallel: Optional[Dict[int, "torch.nn.Parameter"]],
        device: Optional["torch.device"] = None,
    ) -> "PreTrainedModel":
        for block in model.transformer.h:
            block.attn.attention = GPTNeoParallelSelfAttention.transform(
                model,
                block.attn.attention,
                device=device,
            )
        return model


class LlamaParallelSelfAttention(ParallelSelfAttention):
    QUERIES_NAME = "q_proj"
    KEYS_NAME = "k_proj"
    VALUES_NAME = "v_proj"
    OUTPUT_PROJECTION_NAME = "o_proj"
    NUM_ATTENTION_HEADS_NAME = "num_heads"
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
    ) -> "PreTrainedModel":
        embedding_weight_info = None
        lm_head_weight_info = None
        lm_head_bias_weight_info = None
        weight_map = getattr(model, "_weight_map", None)
        if weight_map is not None:
            embedding_weight_info = WeightInformation(
                weight_map["model.embed_tokens.weight"],
                "model.embed_tokens.weight",
                device=device,
            )
            lm_head_weight_info = WeightInformation(weight_map["lm_head.weight"], "lm_head.weight", device=device)
            if "lm_head.bias" in weight_map:
                lm_head_bias_weight_info = WeightInformation(weight_map["lm_head.bias"], "lm_head.bias", device=device)

        model.model.embed_tokens, model.lm_head = embedding_to_parallel_embedding(
            model.model.embed_tokens,
            lm_head_layer=model.lm_head,
            embedding_weight_info=embedding_weight_info,
            lm_head_weight_info=lm_head_weight_info,
            lm_head_bias_weight_info=lm_head_bias_weight_info,
            orig_to_parallel=orig_to_parallel,
            device=device,
        )
        for layer in model.model.layers:
            layer.self_attn = LlamaParallelSelfAttention.transform(model, layer.self_attn, device=device)
            layer.mlp = LLamaParallelMLP.transform(model, layer.mlp, device=device)
        return model
