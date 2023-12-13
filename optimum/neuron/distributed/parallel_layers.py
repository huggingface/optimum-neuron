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

import functools
import re
from abc import ABC, abstractclassmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Literal, Optional, Tuple, Type, Union

import torch
from torch.nn.modules.loss import _WeightedLoss

from ...utils import NormalizedConfigManager, logging
from ..utils import patch_everywhere, patch_within_function
from ..utils.require_utils import requires_neuronx_distributed
from .utils import (
    GroupedQueryAttentionInfo,
    WeightInformation,
    embedding_to_parallel_embedding,
    gqa_key_value_slicing_when_tp_size_greater_than_num_key_value_heads,
    linear_to_parallel_linear,
)


if TYPE_CHECKING:
    from transformers import PreTrainedModel


logger = logging.get_logger()


# Just for testing purposes, setting that to True will feed a copy of the  input to `parallel_cross_entropy` which
# changes inputs inplace. This way the original input is not transformed and can be used in tests for comparison.
_PARALLEL_CROSS_ENTROPY_SHOULD_PRESERVE_INPUT: bool = False


class ParallelLayer(ABC):
    @classmethod
    def _get_module_and_attribute_name(
        cls,
        module: "torch.nn.Module",
        fully_qualified_name: str,
    ) -> Tuple["torch.nn.Module", str]:
        split = fully_qualified_name.rsplit(".", maxsplit=1)
        if len(split) == 1:
            leaf_module = module
            attribute_name = split[0]
        else:
            leaf_module = module.get_submodule(split[0])
            attribute_name = split[1]
        return leaf_module, attribute_name

    @classmethod
    def _get_linear_weight_info(
        cls,
        weight_map: Dict[str, Union[Path, str]],
        linear_layer_qualified_name: str,
        device: Optional["torch.device"] = None,
        fail_if_not_found: bool = True,
    ) -> Tuple[Optional[WeightInformation], Optional[WeightInformation]]:
        linear_layer_weight_qualified_name = f"{linear_layer_qualified_name}.weight"
        if linear_layer_weight_qualified_name not in weight_map:
            if fail_if_not_found:
                raise ValueError(
                    f"Could not find the linear weight called {linear_layer_weight_qualified_name} in the weight map."
                )
            else:
                linear_layer_weight_info = None
        else:
            linear_layer_weight_info = WeightInformation(
                weight_map[linear_layer_weight_qualified_name],
                linear_layer_weight_qualified_name,
                weight_map=weight_map,
                device=device,
            )

        linear_layer_bias_qualified_name = f"{linear_layer_qualified_name}.bias"
        linear_layer_bias_filename = weight_map.get(linear_layer_bias_qualified_name, None)
        if linear_layer_bias_filename is not None:
            linear_layer_bias_weight_info = WeightInformation(
                linear_layer_bias_filename,
                linear_layer_bias_qualified_name,
                weight_map=weight_map,
                device=device,
            )
        else:
            linear_layer_bias_weight_info = None

        return linear_layer_weight_info, linear_layer_bias_weight_info

    @abstractclassmethod
    def transform(
        cls,
        model: "PreTrainedModel",
        layer: "torch.nn.Module",
        sequence_parallel_enabled: bool = False,
        device: Optional["torch.device"] = None,
    ) -> "torch.nn.Module":
        """
        Transforms a layer to its parallel counterpart.

        Args:
            model (`PreTrainedModel`):
                The model to parallelize.
            layer (`torch.nn.Module`):
                The layer to transform.
            sequence_parallel_enabled (`bool`, defaults to `False`):
                Whether or not sequence parallelism is enabled.
            device (`Optional[torch.device]`, defaults to `None`):
                The device where the new parallel layer should be put.
        """


class ParallelEmbedding(ParallelLayer):
    """
    Transforms an Embedding layer into a ParallelEmbedding layer, also takes care of parallelizing a potential tied LM
    head.

    Attributes:
        EMBEDDING_NAME (`str`, defaults to `"embedding"`):
            The qualified name of the embedding layer.
        VOCAB_SIZE_NAME (`Optional[str]`, defaults to `"config.vocab_size"`):
            The name of the attribute holding the value of the vocabulary size.
            If specified, it will overwrite the value to account for embedding parallelization.
        LM_HEAD_NAME (`Optional[Union[str, Dict[str, str]]]`, defaults to `None`):
            The qualified name of the LM head tied to the embedding layer (if any). It can be also a dictionary mapping
            a class name to LM head qualified name.
    """

    EMBEDDING_NAME: str
    VOCAB_SIZE_NAME: Optional[str] = "config.vocab_size"
    LM_HEAD_NAME: Optional[Union[str, Dict[str, str]]] = None

    @classmethod
    @requires_neuronx_distributed
    def overwrite_vocab_size_value_for_cross_entropy_computation(cls, layer: "torch.nn.Module"):
        from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_size

        tp_size = get_tensor_model_parallel_size()
        if cls.VOCAB_SIZE_NAME is not None:
            attribute_names = cls.VOCAB_SIZE_NAME.split(".")
            obj = layer
            for name in attribute_names[:-1]:
                obj = getattr(obj, name)
            vocab_size_attribute_name = attribute_names[-1]
            new_vocab_size = getattr(obj, vocab_size_attribute_name) // tp_size
            setattr(obj, vocab_size_attribute_name, new_vocab_size)

    @classmethod
    @requires_neuronx_distributed
    def transform(
        cls,
        model: "PreTrainedModel",
        layer: "torch.nn.Module",
        sequence_parallel_enabled: bool = False,
        device: Optional["torch.device"] = None,
    ) -> "torch.nn.Module":
        from neuronx_distributed.parallel_layers import parallel_state

        if cls.LM_HEAD_NAME is not None:
            if isinstance(cls.LM_HEAD_NAME, dict):
                lm_head_name = cls.LM_HEAD_NAME.get(model.__class__.__name__, None)
            else:
                lm_head_name = cls.LM_HEAD_NAME
            model_has_lm_head = False
            if lm_head_name is not None:
                parent_lm_head_module, parent_lm_head_attribute_name = cls._get_module_and_attribute_name(
                    layer, lm_head_name
                )
                model_has_lm_head = hasattr(parent_lm_head_module, parent_lm_head_attribute_name)
        else:
            model_has_lm_head = False

        embedding_weight_info = None
        lm_head_weight_info = None
        lm_head_bias_weight_info = None
        weight_map = getattr(model, "_weight_map", None)
        if weight_map is not None:
            layer_to_fully_qualified_name = {id(module): name for name, module in model.named_modules()}
            layer_qualified_name = layer_to_fully_qualified_name[id(layer)]
            if layer_qualified_name:
                embedding_weight_name = f"{layer_qualified_name}.{cls.EMBEDDING_NAME}.weight"
            else:
                embedding_weight_name = f"{cls.EMBEDDING_NAME}.weight"
            embedding_weight_info = WeightInformation(
                weight_map[embedding_weight_name],
                embedding_weight_name,
                weight_map=weight_map,
                device=device,
            )
            if model_has_lm_head:
                if layer_qualified_name:
                    lm_head_weight_name = f"{layer_qualified_name}.{lm_head_name}.weight"
                    lm_head_bias_weight_name = f"{layer_qualified_name}.{lm_head_name}.bias"
                else:
                    lm_head_weight_name = f"{lm_head_name}.weight"
                    lm_head_bias_weight_name = f"{lm_head_name}.bias"
                if lm_head_weight_name in weight_map:
                    lm_head_weight_info = WeightInformation(
                        weight_map[lm_head_weight_name], lm_head_weight_name, weight_map=weight_map, device=device
                    )
                if lm_head_bias_weight_name in weight_map:
                    lm_head_bias_weight_info = WeightInformation(
                        weight_map[lm_head_bias_weight_name],
                        lm_head_bias_weight_name,
                        weight_map=weight_map,
                        device=device,
                    )

        embedding_layer = layer.get_submodule(cls.EMBEDDING_NAME)
        tp_size = parallel_state.get_tensor_model_parallel_size()
        if embedding_layer.num_embeddings % tp_size != 0:
            import torch_xla.core.xla_model as xm

            if xm.get_ordinal() == 0:
                logger.warning(
                    f"Embedding parallelization for TP was skipped because the tensor parallel size ({tp_size}) does not "
                    f"divide the number of embeddings ({embedding_layer.num_embeddings})"
                )
            return layer

        parallel_layers = embedding_to_parallel_embedding(
            layer.get_submodule(cls.EMBEDDING_NAME),
            lm_head_layer=layer.get_submodule(lm_head_name) if model_has_lm_head else None,
            embedding_weight_info=embedding_weight_info,
            lm_head_weight_info=lm_head_weight_info,
            lm_head_bias_weight_info=lm_head_bias_weight_info,
            device=device,
        )
        parent_embedding_module, embedding_attribute_name = cls._get_module_and_attribute_name(
            layer, cls.EMBEDDING_NAME
        )
        if model_has_lm_head:
            setattr(parent_embedding_module, embedding_attribute_name, parallel_layers[0])
            setattr(parent_lm_head_module, parent_lm_head_attribute_name, parallel_layers[1])
        else:
            setattr(parent_embedding_module, embedding_attribute_name, parallel_layers)

        return layer


class ParallelSelfAttention(ParallelLayer):
    """
    Transforms a Self-Attention layer into a Parallel Self-Attention layer.

    Attributes:
        QUERIES_NAME (`str`, defaults to `"query"`):
            The qualified name of the queries layer in the Self-Attention module.
        KEYS_NAME (`str`, defaults to `"key"`):
            The qualified name of the keys layer in the Self-Attention module.
        VALUES_NAME (`str`, defaults to `"value"`):
            The qualified name of the values layer in the Self-Attention module.
        OUTPUT_PROJECTION_NAME (`Optional[str]`, defaults to `None`):
            The qualified name of the output projection layer in the Self-Attention module.
        NUM_ATTENTION_HEADS_NAME (`Optional[str]`, defaults to `None`):
            The name of the attribute in the layer specifying the number of attention heads.
            If left unspecified, the attribute will be fetched by using the NormalizedConfig associated to the model.
        NUM_KEY_VALUE_HEADS_NAME (`Optional[str]`, defaults to `None`):
            The name of the attribute in the layer specifying the number of key value heads (when using Grouped Query
            Attention). If left unspecified, it is interpreted as the model using the regular Multi Head Attention
            mechanism.
        NUM_KEY_VALUE_GROUPS_NAME (`Optional[str]`, defaults to `None`):
            The name of the attribute in the layer specifying the number of query groups (when using Grouped Query
            Attention). If left unspecified, it is interpreted as the model using the regular Multi Head Attention
            mechnism.
        ALL_HEAD_SIZE_NAME (`Optional[str]`, defaults to `None`):
            The name of the attribute in the layer specifying the hidden dimension of each attention head.
            If left unspecified, the attribute will be fetched by using the NormalizedConfig associated to the model.
    """

    QUERIES_NAME = "query"
    KEYS_NAME = "key"
    VALUES_NAME = "value"
    OUTPUT_PROJECTION_NAME: Optional[str] = None
    NUM_ATTENTION_HEADS_NAME: Optional[str] = None
    NUM_KEY_VALUE_HEADS_NAME: Optional[str] = None
    NUM_KEY_VALUE_GROUPS_NAME: Optional[str] = None
    ALL_HEAD_SIZE_NAME: Optional[str] = None

    @classmethod
    @requires_neuronx_distributed
    def transform(
        cls,
        model: "PreTrainedModel",
        layer: "torch.nn.Module",
        sequence_parallel_enabled: bool = False,
        device: Optional["torch.device"] = None,
    ) -> "torch.nn.Module":
        if (cls.NUM_KEY_VALUE_HEADS_NAME is not None and cls.NUM_KEY_VALUE_GROUPS_NAME is None) or (
            cls.NUM_KEY_VALUE_HEADS_NAME is None and cls.NUM_KEY_VALUE_GROUPS_NAME is not None
        ):
            raise AttributeError("Both NUM_KEY_VALUE_HEADS_NAME and NUM_KEY_VALUE_GROUPS_NAME must be specified.")

        from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_size

        tp_size = get_tensor_model_parallel_size()

        weight_map = getattr(model, "_weight_map", None)
        config = model.config
        normalized_config = NormalizedConfigManager.get_normalized_config_class(config.model_type)(config)

        if weight_map is not None:
            layer_to_fully_qualified_name = {id(module): name for name, module in model.named_modules()}
            layer_qualified_name = layer_to_fully_qualified_name[id(layer)]
        else:
            layer_qualified_name = ""

        if cls.NUM_ATTENTION_HEADS_NAME is None:
            num_attention_heads_name = normalized_config.NUM_ATTENTION_HEADS
        else:
            num_attention_heads_name = cls.NUM_ATTENTION_HEADS_NAME

        if not hasattr(layer, num_attention_heads_name):
            raise AttributeError(f"The {type(layer)} layer has not attribute {num_attention_heads_name}.")

        num_attention_heads = getattr(layer, num_attention_heads_name)

        if cls.ALL_HEAD_SIZE_NAME is None:
            all_head_size_name = normalized_config.ALL_HEAD_SIZE_NAME
        else:
            all_head_size_name = cls.ALL_HEAD_SIZE_NAME

        if not hasattr(layer, all_head_size_name):
            raise AttributeError(f"The {type(layer)} layer has not attribute {all_head_size_name}.")

        if cls.NUM_KEY_VALUE_HEADS_NAME is not None:
            num_key_value_heads = getattr(layer, cls.NUM_KEY_VALUE_HEADS_NAME)
            if num_key_value_heads % tp_size != 0 and tp_size % num_key_value_heads != 0:
                raise ValueError(
                    "Only the cases where the number of key value heads is divisible by the TP size, or the other way around are supported."
                )
            elif num_key_value_heads < tp_size:
                logger.warning(
                    f"The TP size ({tp_size}) is bigger than the number of key value heads ({num_key_value_heads}). "
                    "This is not ideal because the key and value projections will not be sharded accross the TP ranks. "
                    "For better performance choose the number of key value heads to be divisible by the TP size."
                )
            kv_heads_are_parallelized = num_key_value_heads >= tp_size
        else:
            num_key_value_heads = getattr(layer, num_attention_heads_name)
            kv_heads_are_parallelized = True

        for name in [cls.QUERIES_NAME, cls.KEYS_NAME, cls.VALUES_NAME]:
            linear_layer_weight_info, linear_layer_bias_weight_info = None, None
            if weight_map is not None:
                linear_layer_weight_info, linear_layer_bias_weight_info = cls._get_linear_weight_info(
                    weight_map,
                    f"{layer_qualified_name}.{name}",
                    device=device,
                )
            # Under GQA setting with num_key_value_heads < tp_size, the key and value projections are replicated accross
            # workers.
            if not kv_heads_are_parallelized and name in [cls.KEYS_NAME, cls.VALUES_NAME]:
                gqa_info = GroupedQueryAttentionInfo(num_attention_heads, num_key_value_heads)
                parallel_linear = gqa_key_value_slicing_when_tp_size_greater_than_num_key_value_heads(
                    gqa_info,
                    getattr(layer, name),
                    linear_layer_weight_info=linear_layer_weight_info,
                    linear_layer_bias_weight_info=linear_layer_bias_weight_info,
                    device=device,
                )
            else:
                parallel_linear = linear_to_parallel_linear(
                    getattr(layer, name),
                    "column",
                    gather_output=False,
                    linear_layer_weight_info=linear_layer_weight_info,
                    linear_layer_bias_weight_info=linear_layer_bias_weight_info,
                    sequence_parallel_enabled=sequence_parallel_enabled,
                    device=device,
                )
            setattr(layer, name, parallel_linear)

        if cls.OUTPUT_PROJECTION_NAME is not None:
            linear_layer_weight_info, linear_layer_bias_weight_info = None, None
            if weight_map is not None:
                linear_layer_weight_info, linear_layer_bias_weight_info = cls._get_linear_weight_info(
                    weight_map,
                    f"{layer_qualified_name}.{cls.OUTPUT_PROJECTION_NAME}",
                    device=device,
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
                    sequence_parallel_enabled=sequence_parallel_enabled,
                    device=device,
                ),
            )

        setattr(
            layer,
            num_attention_heads_name,
            num_attention_heads // tp_size,
        )

        if cls.NUM_KEY_VALUE_HEADS_NAME is not None:
            # This happens when Grouped Query Attention is used and the number of kv heads is bigger than the TP size.
            # Since those heads end-up sharded accross TP ranks just as the query heads, only the number of kv heads
            # needs to be updated. The number of query groups remains the same here because it is the ratio between the
            # number of query heads and the number of kv heads.
            if kv_heads_are_parallelized:
                setattr(
                    layer,
                    cls.NUM_KEY_VALUE_HEADS_NAME,
                    num_key_value_heads // tp_size,
                )
            # This happens when Grouped Query Attention (or Multi Query Attention) is used and the number of kv heads is
            # smaller than the TP size.
            # In this case, multiple ranks will end-up with the same kv head, and each rank will only have one kv head
            # and query group.
            else:
                setattr(
                    layer,
                    cls.NUM_KEY_VALUE_HEADS_NAME,
                    1,
                )
                setattr(
                    layer,
                    cls.NUM_KEY_VALUE_GROUPS_NAME,
                    1,
                )

        setattr(
            layer,
            all_head_size_name,
            getattr(layer, all_head_size_name) // tp_size,
        )
        return layer


class ParallelSelfAttentionWithFusedQKV(ParallelLayer):
    """
    Transforms a Self-Attention layer into a Parallel Self-Attention layer.

    Attributes:
        QUERY_KEY_VALUE_NAME (`str`, defaults to `"query_key_value"`):
            The qualified name of the fused query, key and value layer in the Self-Attention module.
        OUTPUT_PROJECTION_NAME (`Optional[str]`, defaults to `None`):
            The qualified name of the output projection layer in the Self-Attention module.
        NUM_ATTENTION_HEADS_NAME (`Optional[str]`, defaults to `None`):
            The name of the attribute in the layer specifying the number of attention heads.
            If left unspecified, the attribute will be fetched by using the NormalizedConfig associated to the model.
        ALL_HEAD_SIZE_NAME (`Optional[str]`, defaults to `None`):
            The name of the attribute in the layer specifying the hidden dimension of each attention head.
            If left unspecified, the attribute will be fetched by using the NormalizedConfig associated to the model.
    """

    QUERY_KEY_VALUE_NAME = "query_key_value"
    OUTPUT_PROJECTION_NAME: Optional[str] = None
    NUM_ATTENTION_HEADS_NAME: Optional[str] = None
    # TODO: add this in NormalizedConfig
    ALL_HEAD_SIZE_NAME: Optional[str] = None  # "all_head_size"

    @classmethod
    @requires_neuronx_distributed
    def transform(
        cls,
        model: "PreTrainedModel",
        layer: "torch.nn.Module",
        sequence_parallel_enabled: bool = False,
        device: Optional["torch.device"] = None,
    ) -> "torch.nn.Module":
        from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_size

        tp_size = get_tensor_model_parallel_size()

        weight_map = getattr(model, "_weight_map", None)
        config = model.config
        normalized_config = NormalizedConfigManager.get_normalized_config_class(config.model_type)(config)

        if weight_map is not None:
            layer_to_fully_qualified_name = {id(module): name for name, module in model.named_modules()}
            layer_qualified_name = layer_to_fully_qualified_name[id(layer)]
        else:
            layer_qualified_name = ""

        if cls.NUM_ATTENTION_HEADS_NAME is None:
            num_attention_heads_name = normalized_config.NUM_ATTENTION_HEADS
        else:
            num_attention_heads_name = cls.NUM_ATTENTION_HEADS_NAME

        if not hasattr(layer, num_attention_heads_name):
            raise AttributeError(f"The {type(layer)} layer has not attribute {num_attention_heads_name}.")

        num_attention_heads = getattr(layer, num_attention_heads_name)

        if cls.ALL_HEAD_SIZE_NAME is None:
            all_head_size_name = normalized_config.ALL_HEAD_SIZE_NAME
        else:
            all_head_size_name = cls.ALL_HEAD_SIZE_NAME

        if not hasattr(layer, all_head_size_name):
            raise AttributeError(f"The {type(layer)} layer has not attribute {all_head_size_name}.")

        linear_layer_weight_info, linear_layer_bias_weight_info = None, None
        if weight_map is not None:
            linear_layer_weight_info, linear_layer_bias_weight_info = cls._get_linear_weight_info(
                weight_map,
                f"{layer_qualified_name}.{cls.QUERY_KEY_VALUE_NAME}",
                device=device,
            )

        parallel_linear = linear_to_parallel_linear(
            getattr(layer, cls.QUERY_KEY_VALUE_NAME),
            "column",
            gather_output=False,
            stride=3,
            linear_layer_weight_info=linear_layer_weight_info,
            linear_layer_bias_weight_info=linear_layer_bias_weight_info,
            sequence_parallel_enabled=sequence_parallel_enabled,
            device=device,
        )

        setattr(layer, cls.QUERY_KEY_VALUE_NAME, parallel_linear)

        if cls.OUTPUT_PROJECTION_NAME is not None:
            linear_layer_weight_info, linear_layer_bias_weight_info = None, None
            if weight_map is not None:
                linear_layer_weight_info, linear_layer_bias_weight_info = cls._get_linear_weight_info(
                    weight_map,
                    f"{layer_qualified_name}.{cls.OUTPUT_PROJECTION_NAME}",
                    device=device,
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
                    sequence_parallel_enabled=sequence_parallel_enabled,
                    device=device,
                ),
            )

        setattr(
            layer,
            num_attention_heads_name,
            num_attention_heads // tp_size,
        )
        setattr(
            layer,
            all_head_size_name,
            getattr(layer, all_head_size_name) // tp_size,
        )
        return layer


class ParallelSelfOutput(ParallelLayer):
    """
    Transforms the output projection of the Self-Attention mechanism into a parallel version of it.

    Attributes:
        OUTPUT_PROJECTION_NAME (`str`, defaults to `"dense"`):
            The name of the projection layer in the module containing it.
    """

    OUTPUT_PROJECTION_NAME = "dense"

    @classmethod
    def transform(
        cls,
        model: "PreTrainedModel",
        layer: "torch.nn.Module",
        sequence_parallel_enabled: bool = False,
        device: Optional["torch.device"] = None,
    ) -> "torch.nn.Module":
        weight_map = getattr(model, "_weight_map", None)

        linear_layer_weight_info, linear_layer_bias_weight_info = None, None
        if weight_map is not None:
            layer_to_fully_qualified_name = {id(module): name for name, module in model.named_modules()}
            layer_qualified_name = layer_to_fully_qualified_name[id(layer)]
            linear_layer_weight_info, linear_layer_bias_weight_info = cls._get_linear_weight_info(
                weight_map,
                f"{layer_qualified_name}.{cls.OUTPUT_PROJECTION_NAME}",
                device=device,
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
                sequence_parallel_enabled=sequence_parallel_enabled,
                device=device,
            ),
        )
        return layer


class ParallelMLP(ParallelLayer):
    """
    Transforms a MLP into a Parallel MLP.

    Attributes:
        FIRST_LINEAR_NAME (`str`):
            The qualified name of the first linear projection in the module.
        SECOND_LINEAR_NAME (`str`):
            The qualified name of the second linear projection in the module.
    """

    FIRST_LINEAR_NAME: str
    SECOND_LINEAR_NAME: str

    @classmethod
    def transform(
        cls,
        model: "PreTrainedModel",
        layer: "torch.nn.Module",
        sequence_parallel_enabled: bool = False,
        device: Optional["torch.device"] = None,
    ) -> "torch.nn.Module":
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

        module, attribute_name = cls._get_module_and_attribute_name(layer, cls.SECOND_LINEAR_NAME)
        linear_layer_weight_info, linear_layer_bias_weight_info = None, None
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
                "row",
                input_is_parallel=True,
                linear_layer_weight_info=linear_layer_weight_info,
                linear_layer_bias_weight_info=linear_layer_bias_weight_info,
                sequence_parallel_enabled=sequence_parallel_enabled,
                device=device,
            ),
        )

        return layer


@requires_neuronx_distributed
@torch.fx.wrap
def safe_parallel_cross_entropy(*args, **kwargs):
    if kwargs.pop("weight", None) is not None:
        raise ValueError("The weight keyword argument is not supported when using parallel cross entropy")
    if kwargs.pop("size_average", None) is not None:
        raise ValueError("The size_average keyword argument is not supported when using parallel cross entropy")
    if kwargs.pop("ignore_index", -100) != -100:
        raise ValueError("The ignore_index keyword argument is not supported when using parallel cross entropy")
    if kwargs.pop("reduce", None) is not None:
        raise ValueError("The reduce keyword argument is not supported when using parallel cross entropy")
    reduction = kwargs.pop("reduction", "mean")
    if reduction not in ["mean", "sum", "none"]:
        raise ValueError(
            f'The reduction parameter only accepts 3 values: "mean", "sum" and "none", but {reduction} was provided '
            "here."
        )

    from neuronx_distributed.parallel_layers.loss_functions import parallel_cross_entropy

    input_ = args[0]
    if _PARALLEL_CROSS_ENTROPY_SHOULD_PRESERVE_INPUT:
        input_ = input_.clone()

    loss = parallel_cross_entropy(input_, *args[1:], **kwargs)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


class ParallelCrossEntropyLoss(_WeightedLoss):
    """
    Same as `torch.nn.CrossEntropyLoss` except that it uses
    `neuronx_distributed.parallel_layers.loss_functions.parallel_cross_entropy`.
    """

    __constants__ = ["ignore_index", "reduction", "label_smoothing"]
    ignore_index: int
    label_smoothing: float

    def __init__(
        self,
        weight: Optional[torch.Tensor] = None,
        size_average=None,
        ignore_index: int = -100,
        reduce=None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ) -> None:
        super().__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index
        self.label_smoothing = label_smoothing

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Original way of computing the cross-entropy in `torch_neuronx`:
        # from torch_neuronx.xla_impl.ops import SimpleCrossEntropyLoss
        # output = SimpleCrossEntropyLoss.gen_override().forward(self, input, target)
        output = safe_parallel_cross_entropy(
            input,
            target,
            weight=self.weight,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
            label_smoothing=self.label_smoothing,
        )
        return output


class ParallelCrossEntropy(ParallelLayer):
    LAST_LINEAR_PROJECTION_NAME: Union[str, Dict[str, str]]

    @classmethod
    def is_eligible_for_cross_entropy_parallelization(cls, model: "PreTrainedModel") -> bool:
        """
        Specifies whether a given model is eligible for cross entropy loss parallelization.
        """
        return getattr(model.config, "problem_type", "") not in ["regression", "multi_label_classification"]

    @classmethod
    def patch_cross_entropy(cls, model: "PreTrainedModel"):
        patch_everywhere("CrossEntropyLoss", ParallelCrossEntropyLoss)
        orig_forward = model.forward
        patcher = patch_within_function(
            [
                ("torch.nn.functional.cross_entropy", safe_parallel_cross_entropy),
                ("torch.nn.modules.loss.F.cross_entropy", safe_parallel_cross_entropy),
            ]
        )
        model.forward = patcher(orig_forward)

    @classmethod
    @requires_neuronx_distributed
    def transform(
        cls,
        model: "PreTrainedModel",
        layer: "torch.nn.Module",
        sequence_parallel_enabled: bool = False,
        device: Optional["torch.device"] = None,
    ) -> "torch.nn.Module":
        from neuronx_distributed import parallel_layers

        linear_projection_name = None
        if cls.LAST_LINEAR_PROJECTION_NAME is not None:
            if isinstance(cls.LAST_LINEAR_PROJECTION_NAME, dict):
                linear_projection_name = cls.LAST_LINEAR_PROJECTION_NAME.get(model.__class__.__name__, None)
            else:
                linear_projection_name = cls.LAST_LINEAR_PROJECTION_NAME

        if linear_projection_name is None or not cls.is_eligible_for_cross_entropy_parallelization(model):
            return layer

        linear_projection_parent, linear_projection_attr_name = cls._get_module_and_attribute_name(
            layer, linear_projection_name
        )
        linear_projection = getattr(linear_projection_parent, linear_projection_attr_name)

        # If the layer was already parallelized, which is the case most of the time with tied embeddings and LM heads
        # because it is handled in ParallelEmbedding, we only patch the cross entropy loss object.
        if isinstance(linear_projection, parallel_layers.ColumnParallelLinear):
            cls.patch_cross_entropy(model)
            return layer

        if isinstance(linear_projection, parallel_layers.RowParallelLinear):
            raise ValueError(
                "Cannot parallelize the cross entropy loss if the last linear projection is a RowParallelLinear "
                "instance."
            )

        linear_projection_weight_info, linear_projection_bias_weight_info = None, None
        weight_map = getattr(model, "_weight_map", None)
        if weight_map is not None:
            layer_to_fully_qualified_name = {id(module): name for name, module in model.named_modules()}
            linear_projection_qualified_name = layer_to_fully_qualified_name[id(linear_projection)]
            linear_projection_weight_info, linear_projection_bias_weight_info = cls._get_linear_weight_info(
                weight_map,
                linear_projection_qualified_name,
                device=device,
            )

        parallel_linear_projection = linear_to_parallel_linear(
            getattr(linear_projection_parent, linear_projection_attr_name),
            axis="column",
            linear_layer_weight_info=linear_projection_weight_info,
            linear_layer_bias_weight_info=linear_projection_bias_weight_info,
            # The output will be kept parallel.
            gather_output=False,
            # Since it is the last linear projection, we do not want the output to be sequence parallel.
            sequence_parallel_enabled=False,
            device=device,
        )
        setattr(linear_projection_parent, linear_projection_attr_name, parallel_linear_projection)

        cls.patch_cross_entropy(model)

        return layer


@dataclass(frozen=True)
class SequenceCollectiveOpInfo:
    """
    Represents a collective op to perform.

    Attributes:
        - collective_op (`Union[Literal["scatter"], Literal["gather"]]`) -- The collective operation that should be
        performed.
        - layer (`Union[Type[torch.nn.Module], str]`) -- The layer to consider. It can either be a type or a qualified
        name pattern to match.
        - io (`Union[Literal["input"], Literal["output"]]`) -- Whether the collective op should be applied to the input
        or to the output of the matched layer.
        - first_or_last (`Union[Literal["first"], Literal["last"]]`) -- Whether to consider the first matching layer or
        the last matching layer.

    Example:
        ```python
        info = SequenceCollectiveOpInfo("gather", torch.nn.LayerNorm, "output", "last")
        ```
        Here `info` indicates that the output of the last `torch.nn.LayerNorm` must be gathered on the sequence axis.

    """

    collective_op: Union[Literal["scatter"], Literal["gather"]]
    layer: Union[Type["torch.nn.Module"], str]
    io: Union[Literal["input"], Literal["output"]]
    first_or_last: Union[Literal["first"], Literal["last"]]

    def __post_init__(self):
        if self.collective_op not in ["scatter", "gather"]:
            raise ValueError(f'Authorized values are "scatter" and "gather", but {self.collective_op} was given here')
        if self.io not in ["input", "output"]:
            raise ValueError(f'Authorized values are "input" and "output", but {self.io} was given here')


class IOSequenceParallelizer:
    """
    Handles the input and output parallelization for sequence parallelism.
    """

    def __init__(
        self,
        sequence_parallel_enabled: bool,
        sequence_collective_op_infos: Optional[List[SequenceCollectiveOpInfo]] = None,
    ):
        self.sequence_parallel_enabled = sequence_parallel_enabled
        self.sequence_collective_op_infos = sequence_collective_op_infos

    def get_first_and_last_layers_matching_pattern(
        self, model: "torch.nn.Module", pattern: str
    ) -> Tuple["torch.nn.Module", "torch.nn.Module"]:
        first_layer = None
        last_layer = None
        for name, module in model.named_modules():
            if re.match(pattern, name):
                if first_layer is None:
                    first_layer = module
                last_layer = module
        if first_layer is None:
            raise ValueError(f"Could not find layer of with pattern {pattern} in {model}.")
        return [first_layer, last_layer]

    def get_first_and_last_layers_of_type(
        self, model: "torch.nn.Module", type_: Type["torch.nn.Module"]
    ) -> Tuple["torch.nn.Module", "torch.nn.Module"]:
        first_layer = None
        last_layer = None
        for module in model.modules():
            if isinstance(module, type_):
                if first_layer is None:
                    first_layer = module
                last_layer = module
        if first_layer is None:
            raise ValueError(f"Could not find layer of type {type_} in {model}.")
        return [first_layer, last_layer]

    @requires_neuronx_distributed
    def _sequence_parallelize(self, model: "torch.nn.Module", sequence_collective_op_info: SequenceCollectiveOpInfo):
        from neuronx_distributed.parallel_layers.mappings import (
            gather_from_sequence_parallel_region,
            scatter_to_sequence_parallel_region,
        )

        if sequence_collective_op_info.collective_op == "scatter":
            if isinstance(sequence_collective_op_info.layer, str):
                first_layer, last_layer = self.get_first_and_last_layers_matching_pattern(
                    model, sequence_collective_op_info.layer
                )
            else:
                first_layer, last_layer = self.get_first_and_last_layers_of_type(
                    model, sequence_collective_op_info.layer
                )
            scatter_layer = first_layer if sequence_collective_op_info.first_or_last == "first" else last_layer
            orig_scatter_layer_forward = scatter_layer.forward

            if sequence_collective_op_info.io == "input":

                @functools.wraps(orig_scatter_layer_forward)
                def sequence_parallel_forward(*args, **kwargs):
                    to_scatter = args[1]
                    to_scatter = to_scatter.transpose(0, 1).contiguous()
                    scattered = scatter_to_sequence_parallel_region(to_scatter)
                    return orig_scatter_layer_forward(scattered, *args[2:], **kwargs)

            else:

                @functools.wraps(orig_scatter_layer_forward)
                def sequence_parallel_forward(*args, **kwargs):
                    output = orig_scatter_layer_forward(*args[1:], **kwargs)
                    to_scatter = output if isinstance(output, torch.Tensor) else output[0]
                    to_scatter = to_scatter.transpose(0, 1).contiguous()
                    scattered = scatter_to_sequence_parallel_region(to_scatter)
                    return scattered if isinstance(output, torch.Tensor) else (scattered,) + output[1:]

            scatter_layer.forward = sequence_parallel_forward.__get__(scatter_layer)

        else:
            if isinstance(sequence_collective_op_info.layer, str):
                first_layer, last_layer = self.get_first_and_last_layers_matching_pattern(
                    model, sequence_collective_op_info.layer
                )
            else:
                first_layer, last_layer = self.get_first_and_last_layers_of_type(
                    model, sequence_collective_op_info.layer
                )
            gather_layer = first_layer if sequence_collective_op_info.first_or_last == "first" else last_layer
            orig_gather_layer_forward = gather_layer.forward

            if sequence_collective_op_info.io == "output":

                @functools.wraps(orig_gather_layer_forward)
                def sequence_parallel_forward(*args, **kwargs):
                    output = orig_gather_layer_forward(*args[1:], **kwargs)
                    to_gather = output if isinstance(output, torch.Tensor) else output[0]
                    gathered = gather_from_sequence_parallel_region(to_gather, to_model_parallel=False)
                    gathered = gathered.transpose(0, 1).contiguous()
                    return gathered if isinstance(output, torch.Tensor) else (gathered,) + output[1:]

            else:

                @functools.wraps(orig_gather_layer_forward)
                def sequence_parallel_forward(*args, **kwargs):
                    to_gather = args[1]
                    gathered = gather_from_sequence_parallel_region(to_gather, to_model_parallel=False)
                    gathered = gathered.transpose(0, 1).contiguous()
                    output = orig_gather_layer_forward(gathered, *args[2:], **kwargs)
                    return output

            gather_layer.forward = sequence_parallel_forward.__get__(gather_layer)

    def sequence_parallelize(self, model: "torch.nn.Module"):
        if not self.sequence_parallel_enabled or not self.sequence_collective_op_infos:
            return
        for sequence_collective_op_info in self.sequence_collective_op_infos:
            self._sequence_parallelize(model, sequence_collective_op_info)


class LayerNormType(str, Enum):
    REGULAR = "regular"
    RMS_NORM = "rms_norm"


class LayerNormSequenceParallelizer:
    """
    Handles the parallelization of LayerNorm layers for sequence parallelism.
    """

    def __init__(self, sequence_parallel_enabled: bool, layer_norm_qualified_name_patterns: List[str]):
        self.sequence_parallel_enabled = sequence_parallel_enabled
        self.layer_norm_qualified_name_patterns = [
            re.compile(pattern) for pattern in layer_norm_qualified_name_patterns
        ]

    @requires_neuronx_distributed
    def parallelize_layernorm(self, module: torch.nn.LayerNorm):
        from neuronx_distributed.parallel_layers.layer_norm import LayerNorm, _set_sequence_parallel_enabled

        if not isinstance(module, torch.nn.LayerNorm):
            raise TypeError(f"Expected torch.nn.LayerNorm but got {type(module)}.")

        module.__class__ = LayerNorm
        module.sequence_parallel_enabled = self.sequence_parallel_enabled
        if module.elementwise_affine:
            _set_sequence_parallel_enabled(module.weight, self.sequence_parallel_enabled)
            _set_sequence_parallel_enabled(module.bias, self.sequence_parallel_enabled)

    @requires_neuronx_distributed
    def parallelize_rmsnorm(self, module: "torch.nn.Module"):
        from neuronx_distributed.parallel_layers.layer_norm import _set_sequence_parallel_enabled

        _set_sequence_parallel_enabled(module.weight, self.sequence_parallel_enabled)

    def sequence_parallelize(self, model: "PreTrainedModel", layernorm_type: Union[str, LayerNormType]):
        if type(layernorm_type) is str:  # noqa: E721
            layernorm_type = LayerNormType(layernorm_type)

        layernorm_type_to_parallelize_method = {
            LayerNormType.REGULAR: self.parallelize_layernorm,
            LayerNormType.RMS_NORM: self.parallelize_rmsnorm,
        }
        parallelize_method = layernorm_type_to_parallelize_method[layernorm_type]

        for name, module in model.named_modules():
            for pattern in self.layer_norm_qualified_name_patterns:
                if re.match(pattern, name):
                    parallelize_method(module)
