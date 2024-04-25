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
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Callable, Dict, List, Literal, Optional, Set, Tuple, Type, Union

import torch
from transformers import PretrainedConfig
from transformers.utils import is_peft_available

from ...utils import logging
from ..utils import DynamicPatch, Patcher
from ..utils.deprecate_utils import deprecate
from ..utils.import_utils import is_neuronx_distributed_available
from ..utils.misc import download_checkpoints_in_cache
from ..utils.require_utils import (
    requires_neuronx_distributed,
    requires_safetensors,
    requires_torch_xla,
)


if is_neuronx_distributed_available():
    from neuronx_distributed.modules.qkv_linear import GQAQKVColumnParallelLinear
    from neuronx_distributed.parallel_layers import layers
    from neuronx_distributed.pipeline.trace import HFTracerWrapper
else:

    class GQAQKVColumnParallelLinear(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    from transformers.utils.fx import HFTracer

    HFTracerWrapper = HFTracer


if TYPE_CHECKING:
    from transformers import PreTrainedModel


logger = logging.get_logger()


MODEL_PARALLEL_SHARDS_DIR_NAME = "shards"


@deprecate(
    "2.0.0",
    package_name="torch",
    reason="torch.nn.Module._named_members takes a `remove_duplicate` parameter starting from 2.0.0",
)
def _named_members(module, get_members_fn, prefix="", recurse=True, remove_duplicate: bool = True):
    r"""Helper method for yielding various names + members of modules."""
    memo = set()
    modules = module.named_modules(prefix=prefix, remove_duplicate=remove_duplicate) if recurse else [(prefix, module)]
    for module_prefix, mod in modules:
        members = get_members_fn(mod)
        for k, v in members:
            if v is None or v in memo:
                continue
            if remove_duplicate:
                memo.add(v)
            name = module_prefix + ("." if module_prefix else "") + k
            yield name, v


def named_parameters(module: "torch.nn.Module", prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
    gen = _named_members(
        module, lambda mod: mod._parameters.items(), prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
    )
    yield from gen


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
        if self.weight_map is not None:
            prefix = self.weight_map.get("lazy_load_used_prefix", None)
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
            mapping[f"{parent_module_name}.{proj_name}.weight"] = f"{fully_qualified_name}.weight_{qkv_proj_name}"
            if self.use_bias:
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
        if tensor_slices is None:
            tensor = fp.get_tensor(weight_info.qualified_name)
        else:
            tensor_slice = fp.get_slice(weight_info.qualified_name)
            slices = [slice(*slice_) if slice_ is not None else slice(None, None, None) for slice_ in tensor_slices]
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


@requires_neuronx_distributed
def embedding_to_parallel_embedding(
    embedding_layer: "torch.nn.Embedding",
    lm_head_layer: Optional["torch.nn.Linear"] = None,
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
    from neuronx_distributed.parallel_layers import layers
    from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_rank

    device = device if device is not None else torch.device("cpu")

    for weight_info in [embedding_weight_info, lm_head_weight_info, lm_head_bias_weight_info]:
        if weight_info is None:
            continue
        _validate_weight_info_device_matches_specified_device(device, weight_info)

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
    from neuronx_distributed.parallel_layers.parallel_state import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_size,
    )

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

    from neuronx_distributed.parallel_layers.parallel_state import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_size,
    )

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
    from neuronx_distributed.parallel_layers.parallel_state import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_size,
    )

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

    proj_name = weight_name[-1]
    weight = getattr(layer, weight_name)
    bias = getattr(layer, f"bias_{proj_name}")

    num_attention_heads = layer.num_attention_heads
    num_key_value_heads = layer.num_key_value_heads
    kv_size_multiplier = layer.kv_size_multiplier

    with torch.no_grad():
        if not was_already_initialized_during_parallelization(weight):
            weight_data = None
            if linear_layer_weight_info is not None:
                weight_data = load_tensor_for_weight(linear_layer_weight_info)
            elif linear_layer is not None and linear_layer.weight.device != torch.device("meta"):
                weight_data = linear_layer.weight.data
            if weight_data is not None:
                if proj_name in "kv":
                    weight_data = create_kv_proj_local_weight_from_regular_weight(
                        weight_data, kv_size_multiplier, weight.size(0)
                    )
                else:
                    weight_data = create_query_or_output_projection_local_weight_from_regular_weight(
                        weight_data, num_attention_heads, num_key_value_heads, kv_size_multiplier, "query"
                    )
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
        linear_layer_qualified_name, _ = orig_name.rsplit(".", maxsplit=1)
        linear_weight_info, linear_bias_weight_info = get_linear_weight_info(
            weight_map, linear_layer_qualified_name, fail_if_not_found=False
        )
        weight_name = gqa_name.split(".")[-1]
        if try_from_checkpoint and linear_weight_info is not None:
            maybe_load_linear_weight_to_gqa_qkv_column_parallel_linear(
                layer,
                weight_name,
                linear_layer_weight_info=linear_weight_info,
                linear_layer_bias_weight_info=linear_bias_weight_info,
            )
        elif try_from_original_layer:
            orig_layer_name, _ = orig_name.rsplit(".", maxsplit=1)
            maybe_load_linear_weight_to_gqa_qkv_column_parallel_linear(
                layer,
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
    from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_rank

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


@requires_neuronx_distributed
def linear_to_parallel_linear(
    linear_layer: "torch.nn.Linear",
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


def parameter_can_be_initialized(model: torch.nn.Module, parent_module: torch.nn.Module, parameter_name: str) -> bool:
    clone = copy.deepcopy(parent_module)
    left_uninitialized = try_to_hf_initialize(model, clone, [parameter_name])
    is_parallel_linear = isinstance(parent_module, layers.BaseParallelLinear)
    return (
        hasattr(parent_module, "reset_parameters") or is_parallel_linear or (parameter_name not in left_uninitialized)
    )


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
    resume_download = kwargs.pop("resume_download", False)
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
    with patcher:
        try:
            yield
        finally:
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


class OptimumNeuronFXTracer(HFTracerWrapper):
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        return super().is_leaf_module(m, module_qualified_name) or isinstance(m, FakeProj)


class SavedModelInTemporaryDirectory:
    def __init__(self, model: "PreTrainedModel"):
        self.tmpdir = TemporaryDirectory()
        self.model = model

    def __enter__(self):
        self.model.save_pretrained(self.tmpdir.name)
        return self.tmpdir.name

    def __exit__(self, *exc):
        self.tmpdir.cleanup()
