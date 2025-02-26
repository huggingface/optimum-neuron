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
"""Utilities for performing parallelism with `neuronx_distributed`"""

import contextlib
import copy
import functools
import inspect
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Callable, Dict, List, Literal, Optional, Set, Tuple, Type, Union

import torch
from transformers import PretrainedConfig
from transformers.utils.fx import HFTracer

from ...utils import logging
from ..utils import DynamicPatch, Patcher
from ..utils.import_utils import is_neuronx_distributed_available, is_peft_available
from ..utils.misc import download_checkpoints_in_cache, is_precompilation
from ..utils.peft_utils import NeuronPeftModel
from ..utils.require_utils import requires_neuronx_distributed, requires_peft, requires_safetensors, requires_torch_xla


if is_neuronx_distributed_available():
    from neuronx_distributed.modules.qkv_linear import GQAQKVColumnParallelLinear
    from neuronx_distributed.parallel_layers import layers
    from neuronx_distributed.parallel_layers.parallel_state import (
        get_tensor_model_parallel_group,
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_size,
    )
    from neuronx_distributed.parallel_layers.utils import EmbeddingUtility
    from neuronx_distributed.pipeline.trace import HFTracerWrapper, NxDTracer
else:

    class GQAQKVColumnParallelLinear(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    from transformers.utils.fx import HFTracer

    HFTracerWrapper = HFTracer


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    if is_peft_available():
        from peft.tuners.tuners_utils import BaseTunerLayer


logger = logging.get_logger()


MODEL_PARALLEL_SHARDS_DIR_NAME = "shards"


def get_base_model_and_peft_prefix(model: torch.nn.Module) -> Tuple[torch.nn.Module, str]:
    if is_peft_available() and isinstance(model, NeuronPeftModel):
        from peft.tuners.tuners_utils import BaseTunerLayer

        if model.active_peft_config.is_prompt_learning or str(model.peft_type) == "poly":
            peft_prefix = "base_model"
            orig_model = model.base_model
        else:
            peft_prefix = "base_model.model"
            orig_model = model.base_model.model

        # We need to attach this information to enable initialization of tuner layers during parallelization.
        for mod in model.modules():
            if isinstance(mod, BaseTunerLayer):
                mod._peft_config = model.peft_config
    else:
        peft_prefix = ""
        orig_model = model
    return orig_model, peft_prefix


@dataclass
class WeightInformation:
    """
    Describes the information about the weight of a parameter.

    Attributes:
        - filename (`Union[str, Path]`) -- The name of the `safetensors` checkpoint file containing the weights of the
        parameter.
        - qualified_name (`str`) -- The fully qualified name of the parameter in the model hierarchy.
        - weight_map (`Optional[Dict[str, Union[Path, str]]]`, defaults to `None`) -- The weight map use to get the
        filename and the qualified name. It is useful to specify it because then `WeightInformation` will take into
        account potential prefixes that were artficially added to the qualified names.
        - device (`Optional[torch.device]`, defaults to `None`) -- The device to put the weight on, initialized to
        `torch.device("cpu")` if left unspecified.
    """

    filename: Union[str, Path]
    qualified_name: str
    weight_map: Optional[Dict[str, Union[Path, str]]] = None
    device: Optional[torch.device] = None

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cpu")

        prefix = None
        peft_prefix = None
        if self.weight_map is not None:
            prefix = self.weight_map.get("lazy_load_used_prefix", None)
            peft_prefix = self.weight_map.get("peft_prefix", None)
        if peft_prefix is not None and self.qualified_name.startswith(peft_prefix):
            # `peft_prefix` does not contain the `"."` character, that is why we skip the first len(peft_prefix) + 1
            # characters.
            self.qualified_name = self.qualified_name[len(peft_prefix) + 1 :].replace(".base_layer", "")
        if prefix is not None and self.qualified_name.startswith(prefix):
            self.qualified_name = self.qualified_name[len(prefix) :]


class FakeProj(torch.nn.Module):
    """
    Dummy layer that replaces a Linear projection by gathering the result from its associated merged
    QGAQKVColumnParallelLinear.
    """

    def __init__(
        self,
        fully_qualified_name: str,
        proj_name: str,
        output_index: int,
        get_parent_module: Callable[[], torch.nn.Module],
        parent_module_fully_qualified_name: str,
        gqa_qkv_proj_name: str,
    ):
        super().__init__()
        self.fully_qualified_name = fully_qualified_name
        self.proj_name = proj_name
        self.output_index = output_index
        self.get_parent_module = get_parent_module
        self.parent_module_fully_qualified_name = parent_module_fully_qualified_name
        self.gqa_qkv_proj_name = gqa_qkv_proj_name

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        parent_module = self.get_parent_module()
        gqa_qkv_column_parallel_linear = getattr(parent_module, self.gqa_qkv_proj_name)
        if not hasattr(parent_module, "_gqa_qkv_output"):
            parent_module._gqa_qkv_output = gqa_qkv_column_parallel_linear(hidden_states)
            parent_module._gqa_qkv_output_fetch_counter = 0
        parent_module._gqa_qkv_output_fetch_counter += 1
        output = parent_module._gqa_qkv_output[self.output_index]
        if parent_module._gqa_qkv_output_fetch_counter == 3:
            del parent_module._gqa_qkv_output
        return output


class OptimumGQAQKVColumnParallelLinear(GQAQKVColumnParallelLinear):
    """
    Same as GQAQKVColumnParallelLinear with the needed metadata for `optimum-neuron`.
    """

    @requires_neuronx_distributed
    def __init__(
        self,
        query_proj_name: str,
        key_proj_name: str,
        value_proj_name: str,
        output_proj_name: str,
        num_attention_heads: int,
        num_key_value_heads: int,
        input_size: int,
        output_sizes: int,
        bias: bool = True,
        gather_output: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[torch.device] = None,
        init_method: Optional[Callable] = None,
        sequence_parallel_enabled: bool = False,
        keep_master_weight: bool = False,
        kv_size_multiplier: int = 1,
    ):
        from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_size
        from neuronx_distributed.parallel_layers.utils import set_tensor_model_parallel_attributes

        super().__init__(
            input_size,
            output_sizes,
            bias=bias,
            gather_output=gather_output,
            dtype=dtype,
            device=device,
            init_method=init_method,
            sequence_parallel_enabled=sequence_parallel_enabled,
            keep_master_weight=keep_master_weight,
            kv_size_multiplier=kv_size_multiplier,
        )

        if self.fuse_qkv:
            set_tensor_model_parallel_attributes(
                tensor=self.weight_qkv,
                is_parallel=True,
                dim=0,
                stride=1,
                num_partitions=get_tensor_model_parallel_size(),
            )

        self.query_proj_name = query_proj_name
        self.key_proj_name = key_proj_name
        self.value_proj_name = value_proj_name
        self.output_proj_name = output_proj_name

        self._qkv_proj_name_to_proj_name = {"q": query_proj_name, "k": key_proj_name, "v": value_proj_name}
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads

    def get_parameter_names_mapping(
        self, module_to_name: Dict[torch.nn.Module, str], reversed: bool = False
    ) -> Dict[str, str]:
        fully_qualified_name = module_to_name[self]
        parent_module_name, _ = fully_qualified_name.rsplit(".", maxsplit=1)
        mapping = {}
        for qkv_proj_name, proj_name in self._qkv_proj_name_to_proj_name.items():
            if self.fuse_qkv:
                mapping[f"{parent_module_name}.{proj_name}.weight"] = f"{fully_qualified_name}.weight_qkv"
            else:
                mapping[f"{parent_module_name}.{proj_name}.weight"] = f"{fully_qualified_name}.weight_{qkv_proj_name}"
            if self.use_bias:
                if self.fuse_qkv:
                    mapping[f"{parent_module_name}.{proj_name}.bias"] = f"{fully_qualified_name}.bias_qkv"
                else:
                    mapping[f"{parent_module_name}.{proj_name}.bias"] = f"{fully_qualified_name}.bias_{qkv_proj_name}"
        if reversed:
            mapping = {v: k for k, v in mapping.items()}
        return mapping


@requires_neuronx_distributed
def get_parameter_names_mapping_after_gqa_qkv_replacement(
    model: torch.nn.Module, reversed: bool = False
) -> Dict[str, str]:
    """
    Returns the mapping between the original projection names and their names after replacing them with
    GQAQKVColumnParallelLinear.
    """
    from neuronx_distributed.pipeline import NxDPPModel

    mapping = {}
    if isinstance(model, NxDPPModel):
        named_modules = dict(model.local_named_modules())
    else:
        named_modules = dict(model.named_modules())
    module_to_name = {v: k for k, v in named_modules.items()}
    for _, mod in named_modules.items():
        if isinstance(mod, OptimumGQAQKVColumnParallelLinear):
            mapping.update(**mod.get_parameter_names_mapping(module_to_name, reversed=reversed))
    return mapping


@requires_neuronx_distributed
def get_output_projection_qualified_names_after_qga_qkv_replacement(model: torch.nn.Module) -> Set[str]:
    """
    Returns the names of the output projections inside the attention layer, these are needed when using
    GQAQKVColumnParallelLinear.
    """
    from neuronx_distributed.pipeline import NxDPPModel

    qualified_names = set()
    if isinstance(model, NxDPPModel):
        named_modules = dict(model.local_named_modules())
    else:
        named_modules = dict(model.named_modules())
    for name, mod in named_modules.items():
        if isinstance(mod, OptimumGQAQKVColumnParallelLinear):
            parent_name = name.rsplit(".", maxsplit=1)[0]
            output_projection_name = f"{parent_name}.{mod.output_proj_name}"
            qualified_names.add(f"{output_projection_name}.weight")
            if model.get_submodule(output_projection_name).bias is not None:
                qualified_names.add(f"{output_projection_name}.bias")
    return qualified_names


@requires_safetensors
def load_tensor_for_weight(
    weight_info: WeightInformation, tensor_slices: Optional[Tuple[Optional[Tuple[int, ...]], ...]] = None
) -> torch.Tensor:
    """
    Loads potentially sliced weights from a `safetensors` checkpoint.

    Args:
        weight_info (`WeightInformation`):
            The information about the weight, this is used to know where to load the weight from.
        tensor_slices (`Optional[Tuple[Optional[Tuple[int, ...]]]]`, defaults to `None`):
            If specified it will be interpreted as the slices to load from the saved weight, it must be a tuple of either:
                1. A tuple of up to three ints: they will be used to compute a slice.
                2. None: it will be interpreted as taking the whole dimension.
            Example:
                t = torch.rand(4, 3, 2) with tensor_slices = ((2, 3), None, (1,)) will result in t[2:3, :, :1].

    Returns:
        `torch.Tensor`: The loaded tensor.
    """
    from safetensors import safe_open

    # TODO: for now `safetensors` does not support loading directly to the `xla` device.
    # device = str(weight_info.device)
    device = "cpu"
    with safe_open(weight_info.filename, framework="pt", device=device) as fp:
        if tensor_slices is not None:
            slices = [slice(*slice_) if slice_ is not None else slice(None, None, None) for slice_ in tensor_slices]
        else:
            slices = None
        if is_precompilation():
            # During precompilation the actual value of the weights is not important so we skip the loading to make
            # things faster.
            tensor_slice = fp.get_slice(weight_info.qualified_name)
            shape = tuple(tensor_slice.get_shape())
            # Commented entries are supported by later versions of torch. Will uncomment when relevant.
            dtype_str_to_torch_dtype = {
                "BOOL": torch.bool,
                "U8": torch.uint8,
                "F8_E4M3": torch.float8_e4m3fn,
                "F8_E5M2": torch.float8_e5m2,
                "I16": torch.int16,
                # "U16": torch.uint16,
                "F16": torch.float16,
                "BF16": torch.bfloat16,
                "I32": torch.int32,
                # "U32": torch.uint32,
                "F32": torch.float32,
                "F64": torch.float64,
                "I64": torch.int64,
                # "U64": torch.uint64,
            }
            dtype = dtype_str_to_torch_dtype[tensor_slice.get_dtype()]
            tensor = torch.empty(shape, dtype=dtype)
            if tensor_slices is not None:
                tensor = tensor[slices]
        elif tensor_slices is None:
            tensor = fp.get_tensor(weight_info.qualified_name)
        else:
            tensor_slice = fp.get_slice(weight_info.qualified_name)
            tensor = tensor_slice[slices].contiguous()
            # This is needed to make sure tensor.numel() == tensor.storage().size().
            tensor = torch.empty_like(tensor).copy_(tensor)
    return tensor


def _validate_weight_info_device_matches_specified_device(device: "torch.device", weight_info: WeightInformation):
    if device != weight_info.device:
        raise ValueError(
            f"The specfified device must match the device in the `WeightInformation` but got {device} and "
            f"{weight_info.device}, the `WeightInformation` object is: {weight_info}."
        )


def mark_parameter_init_status_during_parallelization(parameter: "torch.nn.Parameter", initialized: bool):
    setattr(parameter, "_was_initialized_during_parallelization", initialized)


def was_already_initialized_during_parallelization(parameter: "torch.nn.Parameter") -> bool:
    return getattr(parameter, "_was_initialized_during_parallelization", False)


@requires_peft
@requires_neuronx_distributed
def _peft_tuner_embedding_to_parallel_embedding(
    tuner_layer: "BaseTunerLayer",
    lm_head_layer: Optional[Union["torch.nn.Linear", "BaseTunerLayer"]] = None,
    embedding_weight_info: Optional[WeightInformation] = None,
    lm_head_weight_info: Optional[WeightInformation] = None,
    lm_head_bias_weight_info: Optional[WeightInformation] = None,
    sequence_parallel_enabled: bool = False,
    device: Optional["torch.device"] = None,
):
    from peft.tuners.lora import Embedding as LoraEmbedding
    from peft.tuners.tuners_utils import BaseTunerLayer

    # This is necessary for the case that the tuner layer wraps another tuner layer.
    parent = tuner_layer
    base_layer = tuner_layer
    while hasattr(base_layer, "base_layer"):
        parent = base_layer
        base_layer = base_layer.base_layer

    parallel_layers = embedding_to_parallel_embedding(
        base_layer,
        lm_head_layer=lm_head_layer,
        embedding_weight_info=embedding_weight_info,
        lm_head_weight_info=lm_head_weight_info,
        lm_head_bias_weight_info=lm_head_bias_weight_info,
        sequence_parallel_enabled=sequence_parallel_enabled,
        device=device,
    )
    if lm_head_layer is None:
        parallel_embedding = parallel_layers
    else:
        parallel_embedding, parallel_linear = parallel_layers

    if isinstance(base_layer, BaseTunerLayer):
        tuner_layer = parallel_embedding
    else:
        parent.base_layer = parallel_embedding

    if isinstance(parent, LoraEmbedding):
        base_layer_is_on_meta_device = parallel_embedding.weight.device == torch.device("meta")
        if base_layer_is_on_meta_device:
            parallel_embedding.weight.data = torch.empty_like(parallel_embedding.weight, device="cpu")
        try:
            peft_config = parent._peft_config
        except AttributeError:
            raise AttributeError(
                f'It seems that {parent} does not have a "_peft_config" attribute. Please use the `parallelize` method '
                "to attach this information to each tuner that needs to be parallelized."
            )

        with torch.no_grad():
            for adapter_name in parent.active_adapters:
                config = peft_config[adapter_name]
                parent.update_layer(
                    adapter_name,
                    parent.r[adapter_name],
                    parent.lora_alpha[adapter_name],
                    config.lora_dropout,
                    config.init_lora_weights,
                    config.use_rslora,
                    config.use_dora,
                    config.lora_bias,
                )
                mark_parameter_init_status_during_parallelization(parent.lora_embedding_A[adapter_name], True)
                mark_parameter_init_status_during_parallelization(parent.lora_embedding_B[adapter_name], True)

        if base_layer_is_on_meta_device:
            parallel_embedding.weight.data = parallel_embedding.weight.to("meta")
    else:
        raise NotImplementedError(f"{parent.__class__.__name__} is not supported yet for model parallelism.")

    if lm_head_layer is None:
        return parent
    return parent, parallel_linear


def embedding_to_parallel_embedding(
    embedding_layer: Union["torch.nn.Embedding", "BaseTunerLayer"],
    lm_head_layer: Optional[Union["torch.nn.Linear", "BaseTunerLayer"]] = None,
    embedding_weight_info: Optional[WeightInformation] = None,
    lm_head_weight_info: Optional[WeightInformation] = None,
    lm_head_bias_weight_info: Optional[WeightInformation] = None,
    sequence_parallel_enabled: bool = False,
    device: Optional["torch.device"] = None,
) -> Union["layers.ParallelEmbedding", Tuple["layers.ParallelEmbedding", "layers.ColumnParallelLinear"]]:
    """
    Helper function that creates a `neuronx_distributed.parallel_layers.layers.ParallelEmbedding` from a regular
    `torch.nn.Embedding`.

    It can also handle the case where the embedding layer is tied to a linear layer.

    Args:
        embedding_layer (`torch.nn.Embedding`):
            The embedding layer to parallelize.
        lm_head_layer (`Optional[torch.nn.Linear]`, defaults to `None`):
            If specified, the linear layer tied to the embedding.
        embedding_weight_info (`Optional[WeightInformation]`, defaults to `None`):
            Information about which checkpoint file the embedding weights are stored in.
        lm_head_weight_info (`Optional[WeightInformation]`, defaults to `None`):
            Information about which checkpoint file the tied linear projection weights are stored in.
        lm_head_bias_weight_info (`Optional[WeightInformation]`, defaults to `None`):
            Information about which checkpoint file the tied linear projection bias is stored in.
        sequence_parallel_enabled (`bool`, defaults to `False`):
            Whether or not sequence parallelism is enabled.
        device (`Optional[torch.device]`, defaults to `None`):
            The device where the new parallel layer should be put.

    Returns:
        `Union[ParallelEmbedding, Tuple[ParallelEmbedding", layers.ColumnParallelLinear]]`: The parallel embedding and the
        parallel linear projection if specified.
    """

    device = device if device is not None else torch.device("cpu")

    for weight_info in [embedding_weight_info, lm_head_weight_info, lm_head_bias_weight_info]:
        if weight_info is None:
            continue
        _validate_weight_info_device_matches_specified_device(device, weight_info)

    if is_peft_available():
        from peft.tuners.tuners_utils import BaseTunerLayer

        if isinstance(embedding_layer, BaseTunerLayer):
            return _peft_tuner_embedding_to_parallel_embedding(
                embedding_layer,
                lm_head_layer=lm_head_layer,
                embedding_weight_info=embedding_weight_info,
                lm_head_weight_info=lm_head_weight_info,
                lm_head_bias_weight_info=lm_head_bias_weight_info,
                sequence_parallel_enabled=sequence_parallel_enabled,
                device=device,
            )

    parallel_embedding_layer = layers.ParallelEmbedding(
        embedding_layer.num_embeddings,
        embedding_layer.embedding_dim,
        padding_idx=embedding_layer.padding_idx,
        max_norm=embedding_layer.max_norm,
        norm_type=embedding_layer.norm_type,
        scale_grad_by_freq=embedding_layer.scale_grad_by_freq,
        sparse=embedding_layer.sparse,
        device=device,
        dtype=embedding_layer.weight.dtype,
    )

    parallel_embedding_layer.weight.requires_grad = embedding_layer.weight.requires_grad

    tp_rank = get_tensor_model_parallel_rank()
    row_size, _ = parallel_embedding_layer.weight.shape

    is_tied = False
    if lm_head_layer is not None:
        is_tied = id(embedding_layer.weight) == id(lm_head_layer.weight)

    embedding_weight_to_tie = parallel_embedding_layer.weight if is_tied else None

    with torch.no_grad():
        if embedding_weight_info is not None:
            weight_data = load_tensor_for_weight(
                embedding_weight_info,
                tensor_slices=(
                    (tp_rank * row_size, (tp_rank + 1) * row_size),
                    None,
                ),
            )
            parallel_embedding_layer.weight.copy_(weight_data)
            mark_parameter_init_status_during_parallelization(parallel_embedding_layer.weight, True)
        elif embedding_layer.weight.device != torch.device("meta"):
            parallel_embedding_layer.weight.copy_(
                embedding_layer.weight[tp_rank * row_size : (tp_rank + 1) * row_size, :]
            )
            mark_parameter_init_status_during_parallelization(parallel_embedding_layer.weight, True)
        else:
            mark_parameter_init_status_during_parallelization(parallel_embedding_layer.weight, False)

        if lm_head_layer is not None:
            parallel_lm_head_layer = linear_to_parallel_linear(
                lm_head_layer,
                "column",
                linear_layer_weight_info=lm_head_weight_info,
                linear_layer_bias_weight_info=lm_head_bias_weight_info,
                embedding_weight_to_tie=embedding_weight_to_tie,
                gather_output=False,
                sequence_parallel_enabled=False,
                device=device,
            )

    del embedding_layer.weight

    if lm_head_layer is None:
        return parallel_embedding_layer

    return parallel_embedding_layer, parallel_lm_head_layer


def get_linear_weight_info(
    weight_map: Optional[Dict[str, Union[Path, str]]],
    linear_layer_qualified_name: str,
    device: Optional[torch.device] = None,
    fail_if_not_found: bool = True,
) -> Tuple[Optional[WeightInformation], Optional[WeightInformation]]:
    linear_layer_weight_qualified_name = f"{linear_layer_qualified_name}.weight"
    if weight_map is None:
        weight_map = {}
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


@requires_neuronx_distributed
def create_kv_proj_local_weight_from_regular_weight(
    weight_data: torch.Tensor, kv_size_multiplier: int, output_size_per_partition: int
) -> torch.Tensor:
    """
    Creates the local version of the key or value projections weight for the given TP rank when using
    GQAQKVColumnParallelLinear.
    """
    assert not isinstance(weight_data, torch.nn.Parameter)

    tp_size = get_tensor_model_parallel_size()
    tp_rank = get_tensor_model_parallel_rank()
    repeated_weight = weight_data.repeat(kv_size_multiplier, 1)
    split = torch.split(repeated_weight, output_size_per_partition, dim=0)
    return torch.cat(split[tp_rank::tp_size], dim=0)


def compute_query_indices_for_rank(
    tp_size: int, tp_rank: int, num_attention_heads: int, num_key_value_heads: int, kv_size_multiplier: int
):
    """
    Computes the permutation for the query weight wheun using GQAQKVColumnParallelLinear.
    """
    num_attention_heads_per_rank = num_attention_heads // tp_size
    num_key_value_heads_per_rank = (num_key_value_heads * kv_size_multiplier) // tp_size
    query_group_size = num_attention_heads // num_key_value_heads
    query_group_size_per_rank = num_attention_heads_per_rank // num_key_value_heads_per_rank

    queries_indices = [torch.arange(query_group_size_per_rank) for _ in range(num_key_value_heads_per_rank)]

    keys_indices = torch.arange(num_key_value_heads).repeat(kv_size_multiplier)
    keys_indices = torch.repeat_interleave(keys_indices, num_attention_heads_per_rank // num_key_value_heads_per_rank)
    keys_indices = torch.chunk(keys_indices, tp_size)

    shift_per_key = torch.arange(0, num_attention_heads, query_group_size)

    shift_within_query_group = torch.arange(0, query_group_size, query_group_size_per_rank)
    num_ranks_to_fit_all_key_value_heads = num_key_value_heads // num_key_value_heads_per_rank
    num_query_heads_before_next_head_of_same_group = (
        num_ranks_to_fit_all_key_value_heads * num_attention_heads_per_rank
    )
    shift_within_query_group = torch.repeat_interleave(
        shift_within_query_group, num_query_heads_before_next_head_of_same_group
    )
    shift_within_query_group = torch.chunk(shift_within_query_group, tp_size)

    indices = []
    for idx, q_indices in enumerate(queries_indices):
        s = slice(idx * query_group_size_per_rank, (idx + 1) * query_group_size_per_rank)
        k_indices = keys_indices[tp_rank][s]
        k_shift = shift_per_key[k_indices]
        group_shift = shift_within_query_group[tp_rank][s]
        indices.append(q_indices + k_shift + group_shift)

    indices = torch.cat(indices, dim=0)
    return indices


@requires_neuronx_distributed
def create_query_or_output_projection_local_weight_from_regular_weight(
    weight_data: torch.Tensor,
    num_attention_heads: int,
    num_key_value_heads: int,
    kv_size_multiplier: int,
    query_or_output_proj: Union[Literal["query"], Literal["output"]],
) -> torch.Tensor:
    """
    Creates the local version of the query or output projections weight for the given TP rank when using
    GQAQKVColumnParallelLinear.
    """
    assert query_or_output_proj in ["query", "output"]
    assert not isinstance(weight_data, torch.nn.Parameter)

    tp_size = get_tensor_model_parallel_size()
    tp_rank = get_tensor_model_parallel_rank()

    if query_or_output_proj == "query":
        hidden_size = weight_data.size(1)
        head_dim = weight_data.size(0) // num_attention_heads
    else:
        hidden_size = weight_data.size(0)
        head_dim = weight_data.size(1) // num_attention_heads
        weight_data = weight_data.transpose(0, 1)

    indices = compute_query_indices_for_rank(
        tp_size, tp_rank, num_attention_heads, num_key_value_heads, kv_size_multiplier
    )
    reshaped_weight = weight_data.view(num_attention_heads, head_dim, hidden_size)
    shuffled_weight = reshaped_weight[indices]
    shuffled_weight = shuffled_weight.reshape(-1, hidden_size)

    if query_or_output_proj == "output":
        shuffled_weight = shuffled_weight.transpose(0, 1)

    return shuffled_weight


def create_local_bias_from_regular_bias(
    bias_weigth_data: torch.Tensor,
    num_attention_heads: int,
    num_key_value_heads: int,
    kv_size_multiplier: int,
    query_or_key_value_bias: Union[Literal["query"], Literal["key_value"]],
    gather_output: bool,
) -> torch.Tensor:
    """
    Creates the local version of the query, key and value projections bias for the given TP rank when using
    GQAQKVColumnParallelLinear.
    """
    assert query_or_key_value_bias in ["query", "key_value"]
    assert not isinstance(bias_weigth_data, torch.nn.Parameter)

    tp_size = get_tensor_model_parallel_size()
    tp_rank = get_tensor_model_parallel_rank()

    if query_or_key_value_bias == "key_value":
        local_bias_weight = bias_weigth_data.repeat(kv_size_multiplier)
        if not gather_output:
            local_bias_weight = local_bias_weight.chunk(tp_size)[tp_rank]

    else:
        if gather_output:
            indices = torch.cat(
                [
                    compute_query_indices_for_rank(
                        tp_size, tp_rank, num_attention_heads, num_key_value_heads, kv_size_multiplier
                    )
                    for tp_rank in range(tp_size)
                ],
                dim=0,
            )
        else:
            indices = compute_query_indices_for_rank(
                tp_size, tp_rank, num_attention_heads, num_key_value_heads, kv_size_multiplier
            )
        reshaped_bias_weight = bias_weigth_data.view(num_attention_heads, -1)
        shuffled_bias_weight = reshaped_bias_weight[indices]
        local_bias_weight = shuffled_bias_weight.reshape(-1)
    return local_bias_weight


@requires_neuronx_distributed
def maybe_load_linear_weight_to_gqa_qkv_column_parallel_linear(
    layer: OptimumGQAQKVColumnParallelLinear,
    proj_name: str,
    weight_name: str,
    linear_layer_weight_info: Optional[WeightInformation] = None,
    linear_layer_bias_weight_info: Optional[WeightInformation] = None,
    linear_layer: Optional["torch.nn.Linear"] = None,
):
    if (
        linear_layer_weight_info is not None or linear_layer_bias_weight_info is not None
    ) and linear_layer is not None:
        raise ValueError(
            "Specify either a linear layer's WeightInformation, or a linear layer to copy the weights from, but not both."
        )
    if linear_layer_weight_info is None and linear_layer_bias_weight_info is None and linear_layer is None:
        raise ValueError(
            "A linear's layer WeightInformation or a linear layer to copy the weights from need to specified."
        )

    if layer.fuse_qkv:
        weight = getattr(layer, "weight_qkv")
        bias = getattr(layer, "bias_qkv")
    else:
        weight = getattr(layer, weight_name)
        bias = getattr(layer, f"bias_{proj_name}")

    num_attention_heads = layer.num_attention_heads
    num_key_value_heads = layer.num_key_value_heads
    kv_size_multiplier = layer.kv_size_multiplier

    with torch.no_grad():
        if layer.fuse_qkv or not was_already_initialized_during_parallelization(weight):
            weight_data = None
            if linear_layer_weight_info is not None:
                weight_data = load_tensor_for_weight(linear_layer_weight_info)
            elif linear_layer is not None and linear_layer.weight.device != torch.device("meta"):
                weight_data = linear_layer.weight.data
            if weight_data is not None:
                if proj_name in "kv":
                    output_size = layer.kv_output_size_per_partition
                    weight_data = create_kv_proj_local_weight_from_regular_weight(
                        weight_data, kv_size_multiplier, output_size
                    )
                else:
                    weight_data = create_query_or_output_projection_local_weight_from_regular_weight(
                        weight_data, num_attention_heads, num_key_value_heads, kv_size_multiplier, "query"
                    )
                if layer.fuse_qkv:
                    if proj_name == "q":
                        s = slice(0, layer.q_output_size_per_partition)
                    elif proj_name == "k":
                        s = slice(
                            layer.q_output_size_per_partition,
                            layer.q_output_size_per_partition + layer.kv_output_size_per_partition,
                        )
                    else:
                        s = slice(layer.q_output_size_per_partition + layer.kv_output_size_per_partition, None)
                    weight[s, :] = weight_data
                else:
                    weight.copy_(weight_data)
                mark_parameter_init_status_during_parallelization(weight, True)
            else:
                mark_parameter_init_status_during_parallelization(weight, False)

            if bias is not None:
                if not was_already_initialized_during_parallelization(bias):
                    bias_weight_data = None
                    if linear_layer_bias_weight_info is not None:
                        bias_weight_data = load_tensor_for_weight(linear_layer_bias_weight_info)
                    elif linear_layer is not None and linear_layer.bias.device != torch.device("meta"):
                        bias_weight_data = linear_layer.bias.data
                    if bias_weight_data is not None:
                        local_bias_weight_data = create_local_bias_from_regular_bias(
                            bias_weight_data,
                            num_attention_heads,
                            num_key_value_heads,
                            kv_size_multiplier,
                            "key_value" if proj_name in "kv" else "query",
                            layer.gather_output,
                        )
                        bias.copy_(local_bias_weight_data)
                        mark_parameter_init_status_during_parallelization(bias, True)
                    else:
                        mark_parameter_init_status_during_parallelization(bias, False)


def maybe_load_weights_to_gqa_qkv_column_parallel_linear(
    model: torch.nn.Module,
    layer: OptimumGQAQKVColumnParallelLinear,
    try_from_checkpoint: bool = True,
    try_from_original_layer: bool = False,
):
    weight_map = getattr(model, "_weight_map", {})
    named_modules = {v: k for k, v in model.named_modules()}
    original_to_gqa = layer.get_parameter_names_mapping(named_modules)

    for orig_name, gqa_name in original_to_gqa.items():
        if layer.query_proj_name in orig_name:
            proj_name = "q"
        elif layer.key_proj_name in orig_name:
            proj_name = "k"
        else:
            proj_name = "v"
        linear_layer_qualified_name, _ = orig_name.rsplit(".", maxsplit=1)
        linear_weight_info, linear_bias_weight_info = get_linear_weight_info(
            weight_map, linear_layer_qualified_name, fail_if_not_found=False
        )
        weight_name = gqa_name.split(".")[-1]
        if try_from_checkpoint and linear_weight_info is not None:
            maybe_load_linear_weight_to_gqa_qkv_column_parallel_linear(
                layer,
                proj_name,
                weight_name,
                linear_layer_weight_info=linear_weight_info,
                linear_layer_bias_weight_info=linear_bias_weight_info,
            )
        elif try_from_original_layer:
            orig_layer_name, _ = orig_name.rsplit(".", maxsplit=1)
            maybe_load_linear_weight_to_gqa_qkv_column_parallel_linear(
                layer,
                proj_name,
                weight_name,
                linear_layer=model.get_submodule(orig_layer_name),
            )


def maybe_load_weights_to_output_projection_when_using_gqa_qkv_column_parallel_linear(
    output_projection: "layers.RowParallelLinear",
    num_attention_heads: int,
    num_key_value_heads: int,
    kv_size_multiplier: int,
    original_output_projection: Optional[torch.nn.Linear] = None,
    linear_layer_weight_info: Optional[WeightInformation] = None,
    linear_layer_bias_weight_info: Optional[WeightInformation] = None,
    try_from_checkpoint: bool = True,
    try_from_original_layer: bool = False,
):
    weight = output_projection.weight
    bias = output_projection.bias
    with torch.no_grad():
        if not was_already_initialized_during_parallelization(weight):
            weight_data = None
            if try_from_checkpoint and linear_layer_weight_info is not None:
                weight_data = load_tensor_for_weight(linear_layer_weight_info)
            elif (
                try_from_original_layer
                and original_output_projection is not None
                and original_output_projection.weight.device != torch.device("meta")
            ):
                weight_data = original_output_projection.weight.data
            if weight_data is not None:
                weight_data = create_query_or_output_projection_local_weight_from_regular_weight(
                    weight_data, num_attention_heads, num_key_value_heads, kv_size_multiplier, "output"
                )
                weight.copy_(weight_data)
                mark_parameter_init_status_during_parallelization(weight, True)
            else:
                mark_parameter_init_status_during_parallelization(weight, False)
        if bias is not None and not was_already_initialized_during_parallelization(bias):
            bias_weight_data = None
            if linear_layer_bias_weight_info is not None:
                bias_weight_data = load_tensor_for_weight(linear_layer_bias_weight_info)
            elif original_output_projection is not None and original_output_projection.bias.device != torch.device(
                "meta"
            ):
                bias_weight_data = original_output_projection.bias.data
            if bias_weight_data is not None:
                output_projection.bias.copy_(bias_weight_data)
                mark_parameter_init_status_during_parallelization(output_projection.bias, True)
            else:
                mark_parameter_init_status_during_parallelization(output_projection.bias, False)


@requires_neuronx_distributed
def maybe_load_linear_weight_to_parallel_linear(
    parallel_linear_layer: "layers.BaseParallelLinear",
    linear_layer_weight_info: Optional[WeightInformation] = None,
    linear_layer_bias_weight_info: Optional[WeightInformation] = None,
    linear_layer: Optional["torch.nn.Linear"] = None,
):
    if (
        linear_layer_weight_info is not None or linear_layer_bias_weight_info is not None
    ) and linear_layer is not None:
        raise ValueError(
            "Specify either a linear layer's WeightInformation, or a linear layer to copy the weights from, but not both."
        )
    if linear_layer_weight_info is None and linear_layer_bias_weight_info is None and linear_layer is None:
        raise ValueError(
            "A linear's layer WeightInformation or a linear layer to copy the weight from need to specified."
        )

    from neuronx_distributed.parallel_layers.layers import RowParallelLinear

    tp_rank = get_tensor_model_parallel_rank()
    row_size, col_size = parallel_linear_layer.weight.shape

    with torch.no_grad():
        if isinstance(parallel_linear_layer, RowParallelLinear):
            if not was_already_initialized_during_parallelization(parallel_linear_layer.weight):
                if linear_layer_weight_info is not None:
                    weight_data = load_tensor_for_weight(
                        linear_layer_weight_info,
                        tensor_slices=(
                            None,
                            (tp_rank * col_size, (tp_rank + 1) * col_size),
                        ),
                    )
                    parallel_linear_layer.weight.copy_(weight_data)
                    mark_parameter_init_status_during_parallelization(parallel_linear_layer.weight, True)
                elif linear_layer.weight.device != torch.device("meta"):
                    parallel_linear_layer.weight.copy_(
                        linear_layer.weight.data[:, tp_rank * col_size : (tp_rank + 1) * col_size]
                    )
                    mark_parameter_init_status_during_parallelization(parallel_linear_layer.weight, True)
                else:
                    mark_parameter_init_status_during_parallelization(parallel_linear_layer.weight, False)

            if parallel_linear_layer.bias is not None:
                if not was_already_initialized_during_parallelization(parallel_linear_layer.bias):
                    if linear_layer_bias_weight_info is not None:
                        bias_weight_data = load_tensor_for_weight(linear_layer_bias_weight_info)
                        parallel_linear_layer.bias.copy_(bias_weight_data)
                        mark_parameter_init_status_during_parallelization(parallel_linear_layer.bias, True)
                    elif linear_layer.bias.device != torch.device("meta"):
                        parallel_linear_layer.bias.copy_(linear_layer.bias.data)
                        mark_parameter_init_status_during_parallelization(parallel_linear_layer.bias, True)
                    else:
                        mark_parameter_init_status_during_parallelization(parallel_linear_layer.bias, False)

        else:
            if not was_already_initialized_during_parallelization(parallel_linear_layer.weight):
                if linear_layer_weight_info is not None:
                    weight_data = load_tensor_for_weight(
                        linear_layer_weight_info,
                        tensor_slices=(
                            (tp_rank * row_size, (tp_rank + 1) * row_size),
                            None,
                        ),
                    )
                    parallel_linear_layer.weight.copy_(weight_data)
                    mark_parameter_init_status_during_parallelization(parallel_linear_layer.weight, True)
                    del weight_data
                elif linear_layer.weight.device != torch.device("meta"):
                    parallel_linear_layer.weight.copy_(
                        linear_layer.weight.data[tp_rank * row_size : (tp_rank + 1) * row_size, :]
                    )
                    mark_parameter_init_status_during_parallelization(parallel_linear_layer.weight, True)
                else:
                    mark_parameter_init_status_during_parallelization(parallel_linear_layer.weight, False)

            if parallel_linear_layer.bias is not None:
                if not was_already_initialized_during_parallelization(parallel_linear_layer.bias):
                    if linear_layer_bias_weight_info is not None:
                        if parallel_linear_layer.gather_output:
                            tensor_slices = (None,)
                        else:
                            tensor_slices = (
                                (
                                    tp_rank * row_size,
                                    (tp_rank + 1) * row_size,
                                ),
                            )
                        bias_weight_data = load_tensor_for_weight(
                            linear_layer_bias_weight_info,
                            tensor_slices=tensor_slices,
                        )
                        parallel_linear_layer.bias.copy_(bias_weight_data)
                        mark_parameter_init_status_during_parallelization(parallel_linear_layer.bias, True)
                        del bias_weight_data
                    elif linear_layer.bias.device != torch.device("meta"):
                        if parallel_linear_layer.gather_output:
                            parallel_linear_layer.bias.copy_(linear_layer.bias.data)
                        else:
                            parallel_linear_layer.bias.copy_(
                                linear_layer.bias.data[tp_rank * row_size : (tp_rank + 1) * row_size]
                            )
                        mark_parameter_init_status_during_parallelization(parallel_linear_layer.bias, True)
                    else:
                        mark_parameter_init_status_during_parallelization(parallel_linear_layer.bias, False)


@requires_peft
@requires_neuronx_distributed
def _peft_tuner_linear_to_parallel_linear(
    tuner_layer: "BaseTunerLayer",
    axis: Union[Literal["row"], Literal["column"]],
    input_is_parallel: bool = False,
    gather_output: bool = True,
    stride: int = 1,
    linear_layer_weight_info: Optional[WeightInformation] = None,
    linear_layer_bias_weight_info: Optional[WeightInformation] = None,
    embedding_weight_to_tie: Optional["torch.nn.Parameter"] = None,
    sequence_parallel_enabled: bool = False,
    skip_weight_load: bool = False,
    device: Optional["torch.device"] = None,
) -> "BaseTunerLayer":
    from neuronx_distributed.parallel_layers.layers import BaseParallelLinear
    from peft.tuners.lora import Linear as LoraLinear
    from peft.tuners.tuners_utils import BaseTunerLayer

    # This is necessary for the case that the tuner layer wraps another tuner layer.
    parent = tuner_layer
    base_layer = tuner_layer
    while hasattr(base_layer, "base_layer"):
        parent = base_layer
        base_layer = base_layer.base_layer

    if isinstance(base_layer, BaseParallelLinear):
        # It can be the case for instance if the embeddings were parallelized and are tied to the LM head.
        # If we apply LoRA to the LM head, it will actually already be a `ColumnParallelLinear`.
        parallel_base_layer = base_layer
    else:
        parallel_base_layer = linear_to_parallel_linear(
            base_layer,
            axis,
            input_is_parallel=input_is_parallel,
            gather_output=gather_output,
            stride=stride,
            linear_layer_weight_info=linear_layer_weight_info,
            linear_layer_bias_weight_info=linear_layer_bias_weight_info,
            embedding_weight_to_tie=embedding_weight_to_tie,
            sequence_parallel_enabled=sequence_parallel_enabled,
            skip_weight_load=skip_weight_load,
            device=device,
        )

    if isinstance(base_layer, BaseTunerLayer):
        tuner_layer = parallel_base_layer
    else:
        parent.base_layer = parallel_base_layer

    if isinstance(parent, LoraLinear):
        # Cases to handle:
        #   1. The base linear layer is a RowParallelLinear, then:
        #       - The lora A matrix needs to be a RowParallelLinear as well,
        #       - The lora B matrix does not need to be parallelized.
        #   2. The base linear layer is a ColumnParallelLinear, then:
        #       - The lora A matrix does not need to be parallelized,
        #       - The lora B matrix needs to be a ColumnParallelLinear as well.
        base_layer_is_on_meta_device = parallel_base_layer.weight.device == torch.device("meta")
        if base_layer_is_on_meta_device:
            parallel_base_layer.weight.data = torch.empty_like(parallel_base_layer.weight, device="cpu")
        try:
            peft_config = parent._peft_config
        except AttributeError:
            raise AttributeError(
                f'It seems that {parent} does not have a "_peft_config" attribute. Please use the `parallelize` method '
                "to attach this information to each tuner that needs to be parallelized."
            )

        for adapter_name in parent.active_adapters:
            config = peft_config[adapter_name]
            parent.update_layer(
                adapter_name,
                parent.r[adapter_name],
                parent.lora_alpha[adapter_name],
                config.lora_dropout,
                config.init_lora_weights,
                config.use_rslora,
                config.use_dora,
                config.lora_bias,
            )
            if axis == "row":
                layer_to_parallelize = parent.lora_A[adapter_name]
            else:
                layer_to_parallelize = parent.lora_B[adapter_name]

            # TODO: handle the case were weights already exist for this adapter.
            parallel_layer = linear_to_parallel_linear(
                layer_to_parallelize,
                axis,
                input_is_parallel=input_is_parallel,
                gather_output=gather_output,
                stride=stride,
                sequence_parallel_enabled=sequence_parallel_enabled,
                skip_weight_load=skip_weight_load,
                device=device,
            )
            if axis == "row":
                parent.lora_A[adapter_name] = parallel_layer
            else:
                parent.lora_B[adapter_name] = parallel_layer

            mark_parameter_init_status_during_parallelization(parent.lora_A[adapter_name].weight, True)
            mark_parameter_init_status_during_parallelization(parent.lora_B[adapter_name].weight, True)

        if base_layer_is_on_meta_device:
            parallel_base_layer.weight.data = parallel_base_layer.weight.to("meta")
    else:
        raise NotImplementedError(f"{parent.__class__.__name__} is not supported yet for model parallelism.")

    return tuner_layer


@requires_neuronx_distributed
def linear_to_parallel_linear(
    linear_layer: Union["torch.nn.Linear", "BaseTunerLayer"],
    axis: Union[Literal["row"], Literal["column"]],
    input_is_parallel: bool = False,
    gather_output: bool = True,
    stride: int = 1,
    linear_layer_weight_info: Optional[WeightInformation] = None,
    linear_layer_bias_weight_info: Optional[WeightInformation] = None,
    embedding_weight_to_tie: Optional["torch.nn.Parameter"] = None,
    sequence_parallel_enabled: bool = False,
    skip_weight_load: bool = False,
    device: Optional["torch.device"] = None,
) -> Union["layers.RowParallelLinear", "layers.ColumnParallelLinear"]:
    """
    Helper function that creates a `neuronx_distributed.parallel_layers.layers.RowParallelLinear` or a
    `neuronx_distributed.parallel_layers.layers.ColumnParallelLinear` from a regular `torch.nn.Linear`.

    Args:
        linear_layer (`torch.nn.Linear`):
            The linear layer to parallelize.
        axis (`Union[Literal["row"], Literal["column"]]`):
            Either to create a `RowParallelLinear` or a `ColumnParallelLinear`.
        input_is_parallel (`bool`, defaults to `False`):
            Only relevant when `axis="row"`. It means that resulting `RowParallelLinear` must expect a parallelized
            input.
        gather_output (`bool`, defaults to `True`):
            Only relevant when `axis="column"`. It means that the resulting `ColumnParallelLinear` will gather the
            output after its forward. It allows to get a non-parallelized output from a `ColumnParallelLinear` layer.
        stride (`int`, defaults to 1):
            The stride of the new parallel layer weights.
        linear_layer_weight_info (`Optional[torch.nn.Linear]`, defaults to `None`):
            Information about which checkpoint file the linear layer weights are stored in.
        linear_layer_bias_weight_info (`Optional[WeightInformation]`, defaults to `None`):
            Information about which checkpoint file the linear layer bias is stored in.
        embedding_weight_to_tie (`Optional[torch.nn.Parameter]`, defaults to `None`):
            If specified, will tie the linear layer weights to it.
        sequence_parallel_enabled (`bool`, defaults to `False`):
            Whether or not sequence parallelism is enabled.
        skip_weight_load (`bool`, defaults to `False`):
            Whether or not to skip the loading of the weights in the newly created parallel linear layer.
        device (`Optional[torch.device]`, defaults to `None`):
            The device where the new parallel layer should be put.

    Returns:
        `Union[RowParallelLinear, ColumnParallelLinear]`: The parallel linear layer.
    """
    from neuronx_distributed.parallel_layers import layers

    if is_peft_available():
        from peft.tuners.tuners_utils import BaseTunerLayer

        if isinstance(linear_layer, BaseTunerLayer):
            return _peft_tuner_linear_to_parallel_linear(
                linear_layer,
                axis,
                input_is_parallel=input_is_parallel,
                gather_output=gather_output,
                stride=stride,
                linear_layer_weight_info=linear_layer_weight_info,
                linear_layer_bias_weight_info=linear_layer_bias_weight_info,
                embedding_weight_to_tie=embedding_weight_to_tie,
                sequence_parallel_enabled=sequence_parallel_enabled,
                skip_weight_load=skip_weight_load,
                device=device,
            )

    if axis not in ["row", "column"]:
        raise ValueError(f'axis must either be "row" or "column", but {axis} was given here.')

    device = device if device is not None else torch.device("cpu")
    for weight_info in [linear_layer_weight_info, linear_layer_bias_weight_info]:
        if weight_info is None:
            continue
        _validate_weight_info_device_matches_specified_device(device, weight_info)

    kwargs = {}
    if axis == "row":
        parallel_linear_class = layers.RowParallelLinear
        kwargs["input_is_parallel"] = input_is_parallel
    else:
        parallel_linear_class = layers.ColumnParallelLinear
        kwargs["gather_output"] = gather_output

    kwargs["dtype"] = linear_layer.weight.dtype
    kwargs["bias"] = linear_layer.bias is not None
    kwargs["device"] = device

    parallel_linear_layer = parallel_linear_class(
        linear_layer.in_features,
        linear_layer.out_features,
        sequence_parallel_enabled=sequence_parallel_enabled,
        stride=stride,
        **kwargs,
    )
    # Not skipping when we tie an embedding layer to make things easier.
    # Should not produce a big overhead.
    skip_weight_load = skip_weight_load and embedding_weight_to_tie is None
    if linear_layer_weight_info is not None and not skip_weight_load:
        maybe_load_linear_weight_to_parallel_linear(
            parallel_linear_layer,
            linear_layer_weight_info=linear_layer_weight_info,
            linear_layer_bias_weight_info=linear_layer_bias_weight_info,
        )
    else:
        maybe_load_linear_weight_to_parallel_linear(parallel_linear_layer, linear_layer=linear_layer)

    if embedding_weight_to_tie is not None:
        parallel_linear_layer.weight = embedding_weight_to_tie

    parallel_linear_layer.weight.requires_grad = linear_layer.weight.requires_grad
    if linear_layer.bias is not None:
        parallel_linear_layer.bias.requires_grad = linear_layer.bias.requires_grad

    return parallel_linear_layer


@requires_neuronx_distributed
def delete_tensor_model_parallel_attributes(tensor: torch.Tensor):
    from neuronx_distributed.parallel_layers.utils import _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS

    for attr_name in _MODEL_PARALLEL_ATTRIBUTE_DEFAULTS:
        if hasattr(tensor, attr_name):
            delattr(tensor, attr_name)


def try_to_hf_initialize(
    model: "PreTrainedModel",
    mod: torch.nn.Module,
    parameter_names: List[str],
    parameter_names_mapping: Optional[Dict[str, str]] = None,
) -> List[str]:
    """
    Tries to initialize the parameters in `parameter_names` that belong to the module `mod` by using the
    `model._init_weights` method. It returns the names of the parameters that were left uninitialized.

    """
    device = torch.device("cpu")
    for name in parameter_names:
        param_device = getattr(mod, name).device
        if param_device != torch.device("meta"):
            device = param_device

    mod.to("cpu")

    cached_params_data = {name: param.data.detach().clone() for name, param in mod.named_parameters()}

    # We initialize on cpu to have the same RNG state (mostly useful for tests).
    model._init_weights(mod)

    if parameter_names_mapping is None:
        parameter_names_mapping = {}

    reverse_parameter_names_mapping = {v: k for k, v in parameter_names_mapping.items()}

    def name_in_mod(name: str):
        return parameter_names_mapping.get(name, name)

    dummy_mod = copy.deepcopy(mod)
    for name in parameter_names:
        getattr(dummy_mod, name_in_mod(name)).random_()
    model._init_weights(dummy_mod)

    left_uninitialized = []
    with torch.no_grad():
        for param_name in parameter_names:
            name = name_in_mod(param_name)
            # The parameter was left unchanged.
            param = getattr(mod, name).data
            if torch.all(param == cached_params_data[name]):
                # There are two possible reasons:
                #   1. The model cannot initialize the module that owns the parameter.
                #   2. The parameter already had the proper value.

                # We check if a dummy copy of the module, filled with random values is modified to know if the model
                # can initialize the module.
                dummy_param_was_changed = torch.all(getattr(dummy_mod, name).data == param)
                if not dummy_param_was_changed:
                    left_uninitialized.append(param_name)

        for name, cached_data in cached_params_data.items():
            param_name = reverse_parameter_names_mapping.get(name, name)
            if param_name not in parameter_names:
                param = getattr(mod, name)
                param.data = cached_data

    # We restore the module back to its original device.
    mod.to(device)

    return left_uninitialized


def initialize_torch_nn_module(mod: torch.nn.Module, parameter_names: List[str]):
    """
    Initializes the parameters in `parameter_names` of a `torch.nn.Linear` module.
    """
    if not hasattr(mod, "reset_parameters"):
        raise ValueError(f"{mod} does not have a `reset_parameters` method.")
    cached_parameters = {name: param.data.detach().clone() for name, param in mod.named_parameters()}
    mod.reset_parameters()
    with torch.no_grad():
        for name, param in mod.named_parameters():
            if param is not None and name not in parameter_names:
                param.data = cached_parameters[name]


@requires_neuronx_distributed
def initialize_parallel_linear(mod: "layers.BaseParallelLinear", parameter_names: List[str]):
    """
    Initializes the parameters in `parameter_names` of a parallel linear module.
    """
    from neuronx_distributed.modules.qkv_linear import GQAQKVColumnParallelLinear
    from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear

    if isinstance(mod, (RowParallelLinear, ColumnParallelLinear)):
        if "weight" in parameter_names:
            delete_tensor_model_parallel_attributes(mod.weight)
            # It is needed to use `init_weight_cpu` instead of `_init_weights` because the initialization
            # needs to happen on the full parameter and then scatter it accross TP ranks otherwise it will
            # not be equivalent to the non-parallel case.
            mod.init_weight_cpu()
        if mod.bias is not None and "bias" in parameter_names:
            mod._init_bias()
    elif isinstance(mod, GQAQKVColumnParallelLinear):
        # It ignores parameter_names, so it might initialize some parameters that should be left unchanged.
        # To improve if it becomes neeeded.
        mod.initialize_weight_biases()
    else:
        raise RuntimeError(f"This kind of parallel linear is not supported yet: {mod.__class__.__name__}")


def duplicate_module_with_random_weights_on_cpu(module: torch.nn.Module) -> torch.nn.Module:
    """
    Create a clone of `module` on CPU without moving any tensor from the XLA device to CPU.
    This has the advantage to not accumulate any graph / trigger any compilation.
    """
    clone = torch.nn.Module()

    children_names = {n for n, _ in module.named_children()}
    buffer_names = {n for n, _ in module.named_buffers()}
    parameter_names = {n for n, _ in module.named_parameters()}

    for name in dir(module):
        attr = getattr(module, name)
        if inspect.ismethod(attr):
            continue
        if name in (children_names | buffer_names | parameter_names) or name.startswith("__"):
            continue
        try:
            cloned_attr = copy.deepcopy(attr)
        except Exception:
            # Attribute is not pickable or cannot be copied
            continue
        setattr(clone, name, cloned_attr)

    for name, mod in module.named_children():
        clone.add_module(name, duplicate_module_with_random_weights_on_cpu(mod))

    for name, buffer in module.named_buffers():
        if "." in name:
            continue
        clone.register_buffer(name, torch.empty_like(buffer, device="cpu"))

    for name, param in module.named_parameters():
        if "." in name:
            continue
        clone.register_parameter(name, torch.nn.Parameter(torch.empty_like(param, device="cpu")))

    clone.__class__ = module.__class__
    return clone


def parameter_can_be_initialized(model: torch.nn.Module, parent_module: torch.nn.Module, parameter_name: str) -> bool:
    # TODO: cannot always print the duplicated clone because it does not have all the same attributes.
    # Might be worth spending some time to fix if printing is needed at some point.
    clone = duplicate_module_with_random_weights_on_cpu(parent_module)
    left_uninitialized = try_to_hf_initialize(model, clone, [parameter_name])
    is_parallel_linear = isinstance(parent_module, layers.BaseParallelLinear)
    return (
        hasattr(parent_module, "reset_parameters") or is_parallel_linear or (parameter_name not in left_uninitialized)
    )


def create_wrapper_for_resize_token_embedding(orig_resize_token_embeddings):
    @functools.wraps(orig_resize_token_embeddings)
    def wrapper(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> torch.nn.Embedding:
        embeddings = self.get_input_embeddings()
        lm_head = self.get_output_embeddings()
        param2name = {param: name for name, param in self.named_parameters()}
        if embeddings.weight.device == torch.device("meta"):
            embeddings_qualified_name = param2name[embeddings.weight]
            if embeddings_qualified_name in self._weight_map:
                filename = self._weight_map[embeddings_qualified_name]
                embeddings_weight_info = WeightInformation(
                    filename=filename,
                    qualified_name=embeddings_qualified_name,
                    weight_map=self._weight_map,
                )
                setattr(embeddings, "weight", torch.nn.Parameter(load_tensor_for_weight(embeddings_weight_info)))
                self._weight_map.pop(embeddings_qualified_name)
            else:
                self._init_weights(embeddings)

        if lm_head is not None and lm_head.weight.device == torch.device("meta"):
            lm_head_qualified_name = param2name[lm_head.weight]
            if lm_head_qualified_name in self._weight_map:
                lm_head_weight_filename = self._weight_map[lm_head_qualified_name]
                lm_head_weight_info = WeightInformation(
                    filename=lm_head_weight_filename,
                    qualified_name=lm_head_qualified_name,
                    weight_map=self._weight_map,
                )
                setattr(lm_head, "weight", torch.nn.Parameter(load_tensor_for_weight(lm_head_weight_info)))
                self._weight_map.pop(lm_head_qualified_name)

                if lm_head.bias is not None:
                    lm_head_bias_qualified_name = param2name[lm_head.bias]
                    lm_head_bias_filename = self._weight_map[lm_head_bias_qualified_name]
                    lm_head_bias_weight_info = WeightInformation(
                        filename=lm_head_bias_filename,
                        qualified_name=lm_head_bias_qualified_name,
                        weight_map=self._weight_map,
                    )
                    setattr(lm_head, "bias", torch.nn.Parameter(load_tensor_for_weight(lm_head_bias_weight_info)))
                    self._weight_map.pop(lm_head_bias_qualified_name)
            else:
                self._init_weights(lm_head)

        return orig_resize_token_embeddings(new_num_tokens=new_num_tokens, pad_to_multiple_of=pad_to_multiple_of)

    bound_wrapper = wrapper.__get__(orig_resize_token_embeddings.__self__)
    return bound_wrapper


@classmethod
@requires_torch_xla
def from_pretrained_for_mp(
    cls,
    pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
    *model_args,
    config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    # TODO: `ignore_mismatched_sizes` is not used in the function, figure out if it leads to a bug.
    ignore_mismatched_sizes: bool = False,
    force_download: bool = False,
    local_files_only: bool = False,
    token: Optional[Union[str, bool]] = None,
    revision: str = "main",
    use_safetensors: bool = False,
    **kwargs,
):
    """
    Custom `from_pretrained()` method for tensor parallelism.
    It will download the weights and create the model but not load the weights in the model. Instead it will create a
    weight map, which maps each parameter name to the `safetensors` checkpoint file storing its weights and attach
    this map to the model instance under the `_weight_map` attribute.
    """
    kwargs.pop("state_dict", None)
    kwargs.pop("from_tf", False)
    kwargs.pop("from_flax", False)
    resume_download = kwargs.pop("resume_download", None)
    proxies = kwargs.pop("proxies", None)
    kwargs.pop("output_loading_info", False)
    kwargs.pop("use_auth_token", None)
    kwargs.pop("trust_remote_code", None)
    kwargs.pop("mirror", None)
    from_pipeline = kwargs.pop("_from_pipeline", None)
    from_auto_class = kwargs.pop("_from_auto", False)
    kwargs.pop("_fast_init", True)
    kwargs.pop("torch_dtype", None)
    kwargs.pop("low_cpu_mem_usage", None)
    kwargs.pop("device_map", None)
    kwargs.pop("max_memory", None)
    kwargs.pop("offload_folder", None)
    kwargs.pop("offload_state_dict", False)
    kwargs.pop("load_in_8bit", False)
    kwargs.pop("load_in_4bit", False)
    kwargs.pop("quantization_config", None)
    subfolder = kwargs.pop("subfolder", "")
    commit_hash = kwargs.pop("_commit_hash", None)
    kwargs.pop("variant", None)
    adapter_kwargs = kwargs.pop("adapter_kwargs", {})
    adapter_name = kwargs.pop("adapter_name", "default")
    kwargs.pop("use_flash_attention_2", False)

    if token is not None and adapter_kwargs is not None and "token" not in adapter_kwargs:
        adapter_kwargs["token"] = token

    filenames, sharded_metadata = download_checkpoints_in_cache(
        pretrained_model_name_or_path,
        cache_dir=cache_dir,
        force_download=force_download,
        local_files_only=local_files_only,
        token=token,
        revision=revision,
        use_safetensors=use_safetensors,
        use_safetensors_in_priority=True,
        convert_to_safetensors=True,
        **kwargs,
    )

    if not isinstance(config, PretrainedConfig):
        config_path = config if config is not None else pretrained_model_name_or_path
        config, model_kwargs = cls.config_class.from_pretrained(
            config_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            _from_auto=from_auto_class,
            _from_pipeline=from_pipeline,
            **kwargs,
        )
    else:
        model_kwargs = kwargs

    if is_peft_available():
        from transformers.utils import find_adapter_config_file

        _adapter_model_path = adapter_kwargs.pop("_adapter_model_path", None)

        if _adapter_model_path is None:
            _adapter_model_path = find_adapter_config_file(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                _commit_hash=commit_hash,
                **adapter_kwargs,
            )
        if _adapter_model_path is not None and os.path.isfile(_adapter_model_path):
            with open(_adapter_model_path, "r", encoding="utf-8") as f:
                _adapter_model_path = pretrained_model_name_or_path
                pretrained_model_name_or_path = json.load(f)["base_model_name_or_path"]
    else:
        _adapter_model_path = None

    model = cls(config, *model_args, **model_kwargs)

    if sharded_metadata:
        weight_map = sharded_metadata["weight_map"]
    else:
        filename = Path(filenames)
        # TODO: manage the safetensor check dependency.
        from safetensors import safe_open

        with safe_open(filename, framework="pt", device="cpu") as fp:
            weight_map = {weight_name: filename for weight_name in fp.keys()}

        # If the model checkpoint used is from a base model but our model is "task-specific", for instance a checkpoint
        # from `GPTNeoModel` when using `GPTNeoForCausalLM`, then our model weight names might not match the names in
        # `weight_map`.
        weight_map_for_model = {}
        model_parameter_and_buffer_names = {
            n for n, _ in itertools.chain(model.named_parameters(), model.named_buffers())
        }
        names_of_weights_not_in_model = set()
        prefixes = set()
        for name, filename in weight_map.items():
            if name not in model_parameter_and_buffer_names:
                sharing_same_suffix_as_name = [n for n in model_parameter_and_buffer_names if n.endswith(name)]
                if not sharing_same_suffix_as_name:
                    continue
                names_of_weights_not_in_model.add(name)
                shortest_sharing_parameter_name = min(sharing_same_suffix_as_name, key=lambda s: len(s))
                prefixes.add(shortest_sharing_parameter_name.replace(name, ""))
            else:
                weight_map_for_model[name] = filename
        if names_of_weights_not_in_model:
            if len(prefixes) == 1:
                prefix = prefixes.pop()
                weight_map_for_model["lazy_load_used_prefix"] = prefix
                for name in names_of_weights_not_in_model:
                    weight_map_for_model[f"{prefix}{name}"] = weight_map[name]
            else:
                raise ValueError(
                    "Some weights in weight_map do not match any model parameters or buffers: "
                    f"{', '.join(names_of_weights_not_in_model)}."
                )
        weight_map = weight_map_for_model

    if _adapter_model_path is not None:
        model.load_adapter(
            _adapter_model_path,
            adapter_name=adapter_name,
            token=token,
            adapter_kwargs=adapter_kwargs,
        )

    model._weight_map = weight_map

    resize_token_embeddings = create_wrapper_for_resize_token_embedding(model.resize_token_embeddings)
    model.resize_token_embeddings = resize_token_embeddings

    return model


@contextlib.contextmanager
def lazy_load_for_parallelism(tensor_parallel_size: int = 1, pipeline_parallel_size: int = 1):
    """
    Context manager that makes the loading of a model lazy for model parallelism:

        - Every `torch.nn.Linear` is put on the `torch.device("meta")` device, meaning that it takes no memory to
        instantiate.
        - Every `torch.nn.Embedding` is also put on the `torch.device("meta")` device.
        - No state dict is actually loaded, instead a weight map is created and attached to the model. For more
        information, read the [`optimum.neuron.distributed.utils.from_pretrained_for_mp`] docstring.

    If both `tensor_parallel_size` and `pipeline_parallel_size` are set to 1, no lazy loading is performed.

    Args:
        tensor_parallel_size (`int`, defaults to 1):
            The tensor parallel size considered.
        pipeline_parallel_size (`int`, defaults to 1):
            The pipeline parallel size considered.
    """

    def meta_init(init_fn):
        @functools.wraps(init_fn)
        def wrapper(*args, **kwargs):
            kwargs["device"] = kwargs.pop("device", torch.device("meta"))
            return init_fn(*args, **kwargs)

        return wrapper

    meta_init_patch = DynamicPatch(meta_init)

    patching_specs = [
        ("torch.nn.Embedding.__init__", meta_init_patch),
        ("torch.nn.Linear.__init__", meta_init_patch),
        ("transformers.modeling_utils.PreTrainedModel.from_pretrained", from_pretrained_for_mp),
    ]
    if tensor_parallel_size > 1 or pipeline_parallel_size > 1:
        patcher = Patcher(patching_specs=patching_specs)
    else:
        patcher = contextlib.nullcontext()
    try:
        patcher.__enter__()
        yield
    finally:
        patcher.__exit__(None, None, None)
        pass


def make_optimizer_constructor_lazy(optimizer_cls: Type["torch.optim.Optimizer"]):
    """
    Transforms an optimizer constructor (optimizer class) to make it lazy by not initializing the parameters.
    This makes the optimizer lightweight and usable to create a "real" optimizer once the model has been
    parallelized.
    """

    def optimizer_constructor(*args, **kwargs):
        optimizer_with_no_parameters = optimizer_cls([torch.nn.Parameter(torch.empty(1))], *args[1:], **kwargs)
        # It is necessary to make sure that what's holding the parameters is not an iterator, otherwise it can lead to
        # unexpected behaviour since each entry will be evaluated at iteration time. There are 2 possibilities:
        #   1. args[0] holds the parameters
        #   2. args[0] holds a list of parameter groups
        parameters_or_parameter_groups = args[0]
        if not isinstance(parameters_or_parameter_groups, list):
            parameters_or_parameter_groups = list(parameters_or_parameter_groups)
        if isinstance(parameters_or_parameter_groups[0], dict):
            # It means that parameter groups were provided. We iterate over each group and make sure that the
            # `"params"` entry is not an iterator.
            for group in parameters_or_parameter_groups:
                if not isinstance(group["params"], list):
                    group["params"] = list(group["params"])

        args = (parameters_or_parameter_groups,) + args[1:]
        optimizer_with_no_parameters._args_to_recreate = (args, kwargs)
        return optimizer_with_no_parameters

    return optimizer_constructor


@dataclass
class ParameterMetadata:
    kind: Literal["tied", "sharded"]
    partition_dim: Optional[int] = None

    def __post_init__(self):
        if self.kind == "sharded":
            if self.partition_dim is None:
                raise ValueError("ParameterMetadata.partion_dim must be specified when the parameter is sharded.")

    @property
    def is_tied(self):
        return self.kind == "tied"

    @property
    def is_sharded(self):
        return self.kind == "sharded"


def get_parameters_tp_metadata(named_parameters: Dict[str, "torch.nn.Parameter"]):
    tp_metadata = {}
    for name, param in named_parameters.items():
        if getattr(param, "tensor_model_parallel", False):
            param_metadata = ParameterMetadata(
                "sharded",
                partition_dim=param.partition_dim,
            )
        else:
            param_metadata = ParameterMetadata("tied")
        tp_metadata[name] = param_metadata
    return tp_metadata


class OptimumNeuronFXTracer(HFTracerWrapper):
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        return (
            NxDTracer.is_leaf_module(self, m, module_qualified_name)
            or HFTracer.is_leaf_module(self, m, module_qualified_name)
            or isinstance(m, FakeProj)
        )


class SavedModelInTemporaryDirectory:
    def __init__(self, model: "PreTrainedModel"):
        self.tmpdir = TemporaryDirectory()
        self.model = model

    def __enter__(self):
        self.model.save_pretrained(self.tmpdir.name)
        return self.tmpdir.name

    def __exit__(self, *exc):
        self.tmpdir.cleanup()


class _ParallelCrossEntropy(torch.autograd.Function):
    @staticmethod
    def forward(ctx, vocab_parallel_logits, target, ignore_index=-100, reduction="mean", label_smoothing=0.0):
        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
        torch.distributed.all_reduce(
            logits_max,
            op=torch.distributed.ReduceOp.MAX,
            group=get_tensor_model_parallel_group(),
        )
        # Subtract the maximum value.
        vocab_parallel_logits.sub_(logits_max.unsqueeze(dim=-1))

        # Get the partition's vocab indecies
        get_vocab_range = EmbeddingUtility.range_from_per_partition_vocab_size
        partition_vocab_size = vocab_parallel_logits.size()[-1]
        rank = get_tensor_model_parallel_rank()
        world_size = get_tensor_model_parallel_size()
        vocab_start_index, vocab_end_index = get_vocab_range(partition_vocab_size, rank, world_size)

        # Original implementation:
        # masked_target = target.clone() - vocab_start_index
        # masked_target[target_mask] = 0
        # New xla friendly implementation:
        is_not_ignore_index_mask = (target != ignore_index).to(vocab_parallel_logits.dtype)
        target_mask = (target >= vocab_start_index) & (target < vocab_end_index)
        masked_target = target.clone() - vocab_start_index
        masked_target = torch.mul(masked_target, target_mask.long())

        # Get predicted-logits = logits[target].
        # For Simplicity, we convert logits to a 2-D tensor with size
        # [*, partition-vocab-size] and target to a 1-D tensor of size [*].
        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)
        masked_target_1d = masked_target.view(-1)
        arange_1d = torch.arange(start=0, end=logits_2d.size()[0], device=logits_2d.device, dtype=torch.long)
        predicted_logits_1d = logits_2d[arange_1d, masked_target_1d]
        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
        predicted_logits = predicted_logits_1d.view_as(target)

        # Original implementation:
        # predicted_logits[target_mask] = 0.0
        # New xla friendly implementation:
        predicted_logits = torch.mul(predicted_logits, target_mask.float())

        # All reduce is needed to get the chunks from other devices.
        torch.distributed.all_reduce(
            predicted_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group(),
        )

        # Sum of exponential of logits along vocab dimension across all devices.
        # Original implementation:
        # exp_logits = vocab_parallel_logits
        # torch.exp(vocab_parallel_logits, out=exp_logits)
        # New xla friendly implementation:
        exp_logits = torch.exp(vocab_parallel_logits)
        sum_exp_logits = exp_logits.sum(dim=-1)

        torch.distributed.all_reduce(
            sum_exp_logits,
            op=torch.distributed.ReduceOp.SUM,
            group=get_tensor_model_parallel_group(),
        )

        # Loss = log(sum(exp(logits))) - predicted-logit.
        loss = torch.log(sum_exp_logits) - predicted_logits

        # Store softmax, target-mask and masked-target for backward pass.
        exp_logits.div_(sum_exp_logits.unsqueeze(dim=-1))

        vocab_size = exp_logits.size(-1)
        if label_smoothing > 0:
            """
            We'd like to assign 1 / (K - 1) probability mass to every index that is not the ground truth.
            = (1 - alpha) * y_gt + alpha * mean(y_{i for i != gt})
            = (1 - alpha) * y_gt + (alpha / (K - 1)) * sum_{i != gt} y_i
            = ((K - 1) * (1 - alpha) / (K - 1)) * y_gt + (alpha / (K - 1)) * sum_{i != gt} y_i
            = (K * (1 - alpha) - 1) / (K - 1)) * y_gt  + (alpha / (K - 1)) * sum_{i} y_i
            = (1 - (alpha * K) / (K - 1)) * y_gt + ( (alpha * K) / (K - 1) ) * sum_{i} y_i / K
            From: https://github.com/NVIDIA/NeMo/blob/main/nemo/collections/common/losses/smoothed_cross_entropy.py
            """
            assert 1.0 > label_smoothing > 0.0
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)

            # Exp logits at this point are normalized probabilities. So we can just take the log to get log-probs.
            log_probs = torch.log(exp_logits)
            mean_log_probs = log_probs.mean(dim=-1)
            loss = (1.0 - smoothing) * loss - smoothing * mean_log_probs

        # Zerooing the loss for the ignored tokens.
        loss = loss * is_not_ignore_index_mask

        # Apply the reduction, to respect the torch.nn.functional.cross_entropy_loss API
        # the reduction happens only on the non-ignored tokens.
        if reduction == "sum":
            loss = loss.sum()
        elif reduction == "mean":
            num_non_ignored_tokens = is_not_ignore_index_mask.sum()
            loss = loss.sum() / num_non_ignored_tokens

        ctx.reduction = reduction
        ctx.label_smoothing, ctx.vocab_size = label_smoothing, vocab_size
        ctx.save_for_backward(exp_logits, target_mask, masked_target_1d, is_not_ignore_index_mask)

        return loss

    @staticmethod
    def backward(ctx, grad_output):
        # Retreive tensors from the forward path.
        softmax, target_mask, masked_target_1d, is_non_ignore_index_mask = ctx.saved_tensors
        label_smoothing, vocab_size = ctx.label_smoothing, ctx.vocab_size

        reduction = ctx.reduction

        # All the inputs have softmax as their gradient.
        grad_input = softmax

        # For simplicity, work with the 2D gradient.
        partition_vocab_size = softmax.size()[-1]
        grad_2d = grad_input.view(-1, partition_vocab_size)

        # Add the gradient from matching classes.
        arange_1d = torch.arange(start=0, end=grad_2d.size()[0], device=grad_2d.device, dtype=torch.long)

        if label_smoothing > 0:
            softmax_update = 1.0 - target_mask.view(-1).float()
            smoothing = label_smoothing * vocab_size / (vocab_size - 1)
            grad_2d[arange_1d.long(), masked_target_1d.long()] -= (1.0 - smoothing) * softmax_update
            average_grad = 1 / vocab_size
            grad_2d[arange_1d.long(), :] -= smoothing * average_grad
        else:
            grad_2d[arange_1d, masked_target_1d] -= target_mask.view(-1).float()

        grad_input *= is_non_ignore_index_mask.unsqueeze(dim=-1)

        if reduction == "mean":
            num_non_ignored_tokens = is_non_ignore_index_mask.sum()
            grad_input *= grad_output / num_non_ignored_tokens
        elif reduction == "sum":
            grad_input *= grad_output
        else:
            grad_input.mul_(grad_output.unsqueeze(dim=-1))

        return grad_input, None, None, None, None


# Just for testing purposes, setting that to True will feed a copy of the  input to `parallel_cross_entropy` which
# changes inputs inplace. This way the original input is not transformed and can be used in tests for comparison.
_PARALLEL_CROSS_ENTROPY_SHOULD_PRESERVE_INPUT: bool = False


def parallel_cross_entropy(vocab_parallel_logits, target, ignore_index=-100, reduction="mean", label_smoothing=0.0):
    """Helper function for the cross entropy."""
    if _PARALLEL_CROSS_ENTROPY_SHOULD_PRESERVE_INPUT:
        vocab_parallel_logits = vocab_parallel_logits.clone()
    return _ParallelCrossEntropy.apply(vocab_parallel_logits, target, ignore_index, reduction, label_smoothing)
