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


if TYPE_CHECKING:
    import torch
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
        orig_to_parallel: Optional[Dict[int, "torch.nn.Parameter"]] = None,
        device: Optional["torch.device"] = None,
    ) -> "torch.nn.Module":
        layer = super().transform(model, layer, orig_to_parallel=orig_to_parallel, device=device)

        num_heads = layer.n_heads
        from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_rank

        tp_rank = get_tensor_model_parallel_rank()

        orig_compute_bias = layer.compute_bias

        def compute_bias(self, query_length, key_length, device=None):
            """Compute binned relative position bias"""
            values = orig_compute_bias(query_length, key_length, device=device)
            return values[:, tp_rank * num_heads : (tp_rank + 1) * num_heads, :]

        layer.compute_bias = compute_bias.__get__(layer)
        return layer


class T5ParallelMLP(ParallelMLP):
    FIRST_LINEAR_NAME = "wi"
    SECOND_LINEAR_NAME = "wo"

    @classmethod
    def transform(
        cls,
        model: "PreTrainedModel",
        layer: "torch.nn.Module",
        orig_to_parallel: Optional[Dict[int, "torch.nn.Parameter"]] = None,
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
        layer = super().transform(model, layer, orig_to_parallel=orig_to_parallel, device=device)

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
                    orig_to_parallel=orig_to_parallel,
                    device=device,
                ),
            )

        cls.FIRST_LINEAR_NAME = orig_first_linear_name


class T5Parallelizer(Parallelizer):
    @classmethod
    def _parallelize(
        cls,
        model: "PreTrainedModel",
        orig_to_parallel: Optional[Dict[int, "torch.nn.Parameter"]],
        device: Optional["torch.device"] = None,
    ) -> "PreTrainedModel":
        model = T5ParallelEmbedding.transform(model, model, device=device)
        if model.encoder.embed_tokens is not None:
            model.encoder.embed_tokens = model.shared
        if model.decoder.embed_tokens is not None:
            model.decoder.embed_tokens = model.shared
        for block in model.encoder.block:
            block.layer[0].SelfAttention = T5ParallelSelfAttention.transform(
                model,
                block.layer[0].SelfAttention,
                device=device,
            )
            block.layer[1].DenseReluDense = T5ParallelMLP.transform(
                model, block.layer[1].DenseReluDense, device=device
            )
        for block in model.decoder.block:
            block.layer[0].SelfAttention = T5ParallelSelfAttention.transform(
                model,
                block.layer[0].SelfAttention,
                device=device,
            )
            block.layer[2].DenseReluDense = T5ParallelMLP.transform(
                model, block.layer[2].DenseReluDense, device=device
            )
        return model
