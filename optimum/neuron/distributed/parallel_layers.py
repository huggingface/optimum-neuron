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
"""Classes related to parallel versions of common blocks in Transformers models."""

from abc import ABC, abstractclassmethod
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple

from ...utils import NormalizedConfigManager
from ..utils import is_neuronx_distributed_available
from .utils import WeightInformation, linear_to_parallel_linear


if is_neuronx_distributed_available():
    from neuronx_distributed.parallel_layers import parallel_state

if TYPE_CHECKING:
    import torch
    from transformers import PretrainedConfig, PreTrainedModel


class ParallelLayer(ABC):
    @classmethod
    def _get_linear_weight_info(
        cls, weight_map: Dict[str, Path], linear_layer_qualified_name: str
    ) -> Tuple[WeightInformation, Optional[WeightInformation]]:
        linear_layer_weight_qualified_name = f"{linear_layer_qualified_name}.weight"
        linear_layer_weight_info = WeightInformation(
            weight_map[linear_layer_weight_qualified_name], linear_layer_weight_qualified_name
        )

        linear_layer_bias_qualified_name = f"{linear_layer_qualified_name}.bias"
        linear_layer_bias_filename = weight_map.get(linear_layer_bias_qualified_name, None)
        if linear_layer_bias_filename is not None:
            linear_layer_bias_weight_info = WeightInformation(
                linear_layer_bias_filename, linear_layer_bias_qualified_name
            )
        else:
            linear_layer_bias_weight_info = None

        return linear_layer_weight_info, linear_layer_bias_weight_info

    @abstractclassmethod
    def transform(
        cls,
        layer: "torch.nn.Module",
        config: "PretrainedConfig",
        orig_to_parallel: Optional[Dict[int, "torch.nn.Parameter"]] = None,
    ) -> "torch.nn.Module":
        pass


class ParallelSelfAttention(ParallelLayer):
    QUERIES_NAME = "query"
    KEYS_NAME = "key"
    VALUES_NAME = "value"
    OUTPUT_PROJECTION_NAME: Optional[str] = None
    NUM_ATTENTION_HEADS_NAME: Optional[str] = None
    # TODO: add this in NormalizedConfig
    ALL_HEAD_SIZE_NAME: Optional[str] = None  # "all_head_size"

    @classmethod
    def transform(
        cls,
        model: "PreTrainedModel",
        layer: "torch.nn.Module",
        orig_to_parallel: Optional[Dict[int, "torch.nn.Parameter"]] = None,
    ) -> "torch.nn.Module":
        weight_map = getattr(model, "_weight_map", None)
        config = model.config
        normalized_config = NormalizedConfigManager.get_normalized_config_class(config.model_type)(config)

        if weight_map is not None:
            layer_to_fully_qualified_name = {id(module): name for name, module in model.named_modules()}
            layer_qualified_name = layer_to_fully_qualified_name[id(layer)]
        else:
            layer_qualified_name = ""

        for name in [cls.QUERIES_NAME, cls.KEYS_NAME, cls.VALUES_NAME]:
            linear_layer_weight_info, linear_layer_bias_weight_info = None, None
            if weight_map is not None:
                linear_layer_weight_info, linear_layer_bias_weight_info = cls._get_linear_weight_info(
                    weight_map, f"{layer_qualified_name}.{name}"
                )
            parallel_linear = linear_to_parallel_linear(
                getattr(layer, name),
                "column",
                gather_output=False,
                linear_layer_weight_info=linear_layer_weight_info,
                linear_layer_bias_weight_info=linear_layer_bias_weight_info,
                orig_to_parallel=orig_to_parallel,
            )
            setattr(layer, name, parallel_linear)

        if cls.OUTPUT_PROJECTION_NAME is not None:
            linear_layer_weight_info, linear_layer_bias_weight_info = None, None
            if weight_map is not None:
                linear_layer_weight_info, linear_layer_bias_weight_info = cls._get_linear_weight_info(
                    weight_map, f"{layer_qualified_name}.{cls.OUTPUT_PROJECTION_NAME}"
                )
            setattr(
                layer,
                cls.OUTPUT_PROJECTION_NAME,
                linear_to_parallel_linear(
                    getattr(layer, cls.OUTPUT_PROJECTION_NAME),
                    "row",
                    input_is_parallel=True,
                    linear_layer_weight_info=linear_layer_weight_info,
                    linear_layer_bias_weight_info=linear_layer_bias_weight_info,
                    orig_to_parallel=orig_to_parallel,
                ),
            )

        if cls.NUM_ATTENTION_HEADS_NAME is None:
            num_attention_heads_name = normalized_config.NUM_ATTENTION_HEADS
        else:
            num_attention_heads_name = cls.NUM_ATTENTION_HEADS_NAME

        if not hasattr(layer, num_attention_heads_name):
            raise AttributeError(f"The {type(layer)} layer has not attribute {num_attention_heads_name}.")

        if cls.ALL_HEAD_SIZE_NAME is None:
            all_head_size_name = normalized_config.ALL_HEAD_SIZE_NAME
        else:
            all_head_size_name = cls.ALL_HEAD_SIZE_NAME

        if not hasattr(layer, all_head_size_name):
            raise AttributeError(f"The {type(layer)} layer has not attribute {all_head_size_name}.")

        setattr(
            layer,
            num_attention_heads_name,
            normalized_config.num_attention_heads // parallel_state.get_tensor_model_parallel_size(),
        )
        setattr(
            layer,
            all_head_size_name,
            getattr(layer, all_head_size_name) // parallel_state.get_tensor_model_parallel_size(),
        )
        return layer


class ParallelSelfOutput(ParallelLayer):
    OUTPUT_PROJECTION_NAME = "dense"

    @classmethod
    def transform(
        cls,
        model: "PreTrainedModel",
        layer: "torch.nn.Module",
        orig_to_parallel: Optional[Dict[int, "torch.nn.Parameter"]] = None,
    ) -> "torch.nn.Module":
        weight_map = getattr(model, "_weight_map", None)

        linear_layer_weight_info, linear_layer_bias_weight_info = None, None
        if weight_map is not None:
            layer_to_fully_qualified_name = {id(module): name for name, module in model.named_modules()}
            layer_qualified_name = layer_to_fully_qualified_name[id(layer)]
            linear_layer_weight_info, linear_layer_bias_weight_info = cls._get_linear_weight_info(
                weight_map, f"{layer_qualified_name}.{cls.OUTPUT_PROJECTION_NAME}"
            )

        setattr(
            layer,
            cls.OUTPUT_PROJECTION_NAME,
            linear_to_parallel_linear(
                getattr(layer, cls.OUTPUT_PROJECTION_NAME),
                "row",
                input_is_parallel=True,
                linear_layer_weight_info=linear_layer_weight_info,
                linear_layer_bias_weight_info=linear_layer_bias_weight_info,
                orig_to_parallel=orig_to_parallel,
            ),
        )
        return layer


class ParallelMLP(ParallelLayer):
    FIRST_LINEAR_NAME: Optional[str] = None
    SECOND_LINEAR_NAME: Optional[str] = None

    @classmethod
    def _get_module_and_attribute_name(
        cls, layer: "torch.nn.Module", fully_qualified_linear_name: str
    ) -> Tuple["torch.nn.Module", str]:
        split = fully_qualified_linear_name.rsplit(".", maxsplit=1)
        if len(split) == 1:
            module = layer
            attribute_name = split[0]
        else:
            module = layer.get_submodule(split[0])
            attribute_name = split[1]
        return module, attribute_name

    @classmethod
    def transform(
        cls,
        model: "PreTrainedModel",
        layer: "torch.nn.Module",
        orig_to_parallel: Optional[Dict[int, "torch.nn.Parameter"]] = None,
    ) -> "torch.nn.Module":
        if cls.FIRST_LINEAR_NAME is None or cls.SECOND_LINEAR_NAME is None:
            raise ValueError("Both `FIRST_LINEAR_NAME` and `SECOND_LINEAR_NAME` class attributes must be set.")

        weight_map = getattr(model, "_weight_map", None)

        linear_layer_weight_info, linear_layer_bias_weight_info = None, None
        layer_qualified_name = ""
        if weight_map is not None:
            layer_to_fully_qualified_name = {id(module): name for name, module in model.named_modules()}
            layer_qualified_name = layer_to_fully_qualified_name[id(layer)]
            linear_layer_weight_info, linear_layer_bias_weight_info = cls._get_linear_weight_info(
                weight_map, f"{layer_qualified_name}.{cls.FIRST_LINEAR_NAME}"
            )

        module, attribute_name = cls._get_module_and_attribute_name(layer, cls.FIRST_LINEAR_NAME)
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
            ),
        )

        module, attribute_name = cls._get_module_and_attribute_name(layer, cls.SECOND_LINEAR_NAME)

        linear_layer_weight_info, linear_layer_bias_weight_info = None, None
        if weight_map is not None:
            linear_layer_weight_info, linear_layer_bias_weight_info = cls._get_linear_weight_info(
                model, f"{layer_qualified_name}.{cls.SECOND_LINEAR_NAME}"
            )

        setattr(
            module,
            attribute_name,
            linear_to_parallel_linear(
                getattr(module, attribute_name),
                "row",
                input_is_parallel=True,
                linear_layer_weight_info=linear_layer_weight_info,
                linear_layer_bias_weight_info=linear_layer_bias_weight_info,
                orig_to_parallel=orig_to_parallel,
            ),
        )

        return layer
