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

import copy
import gc
import json
import math
import os
import sys
import warnings
from abc import abstractmethod
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Type, Union

import torch
from safetensors import safe_open
from torch_xla.utils.checkpoint import checkpoint
from transformers import PretrainedConfig
from transformers.modeling_utils import (
    SpecificPreTrainedModelType,
    get_parameter_dtype,
    get_state_dict_dtype,
    load_state_dict,
    no_init_weights,
    set_initialized_submodules,
)
from transformers.utils import (
    CONFIG_NAME,
    ContextManagers,
    cached_file,
    extract_commit_hash,
    find_adapter_config_file,
    is_offline_mode,
    is_peft_available,
    is_safetensors_available,
    logging,
)

from ...utils.import_utils import is_neuronx_distributed_available, is_torch_xla_available
from ...utils.misc import download_checkpoints_in_cache, is_main_worker, is_precompilation


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

if is_neuronx_distributed_available():
    import neuronx_distributed
    from neuronx_distributed.kernels.flash_attn import nki_flash_attn_func
    from neuronx_distributed.parallel_layers.layers import BaseParallelLinear, create_local_weight
    from neuronx_distributed.parallel_layers.parallel_state import (
        get_data_parallel_rank,
        get_pipeline_model_parallel_rank,
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_size,
    )
    from neuronx_distributed.parallel_layers.utils import get_local_world_size, move_model_to_device


logger = logging.get_logger(__name__)

MODEL_PARALLEL_SHARDS_DIR_NAME = "shards"

ALL_ATTENTION_FUNCTIONS: Dict[str, Dict[str, Callable]] = {
    "flash_attention_2": nki_flash_attn_func,
}

def create_local_fused_weight(tp_rank, tp_size, individual_weights, partition_dim, fuse_axis, out_weight=None):
    weight_lists = []
    for weight in individual_weights:
        weight_list = torch.split(weight, weight.size(partition_dim) //  tp_size, dim=partition_dim)[tp_rank::tp_size]
        weight_lists.append(weight_list)

    with torch.no_grad():
        return torch.cat(
            [torch.cat(weight_list, dim=partition_dim) for weight_list in weight_lists],
            dim=fuse_axis,
            out=out_weight,
        )

class ModelWeightTransformationSpec:
    @abstractmethod
    def adapt_state_dict(self, module_fully_qualified_name: str, named_parameters: Dict[str, torch.nn.Parameter], orig_state_dict: Dict[str, torch.Tensor], inplace: bool = False):
        """
        Adapt the state dict of the model to the custom model.
        """

    @abstractmethod
    def to_original_weights(self, sharded_state_dicts: Dict[str, list[torch.Tensor]], parameters_metadata: Dict[str, Dict[str, Any]]) -> tuple[Dict[str, torch.Tensor], list[str]]:
        """
        Transform the weights back to the original weights.
        """

@dataclass
class FusedLinearsSpec(ModelWeightTransformationSpec):
    fused_linear_name: str
    linear_names: list[str]
    bias: bool
    fuse_axis: int
    original_dims: list[int]
    tp_size: int = field(default_factory=get_tensor_model_parallel_size)

    def adapt_state_dict(self, module_fully_qualified_name: str, named_parameters: Dict[str, torch.nn.Parameter], orig_state_dict: Dict[str, torch.Tensor], inplace: bool = False):
        tp_size = get_tensor_model_parallel_size()
        tp_rank = get_tensor_model_parallel_rank()

        if inplace:
            state_dict = orig_state_dict
        else:
            state_dict = dict(orig_state_dict)

        # To go from original weights to fused weights we need to:
        # 1. Gather all the weights of the linear layers
        # 2. Shard them across the tensor model parallel size if TP is enabled
        # 3. Concatenate them along the fuse axis
        fused_linear_fully_qualified_name = f"{module_fully_qualified_name}.{self.fused_linear_name}"
        param_names = ["weight", "bias"] if self.bias else ["weight"]
        for param_name in param_names:
            new_name = f"{fused_linear_fully_qualified_name}.{param_name}"
            full_weight_names = [f"{module_fully_qualified_name}.{linear_name}.{param_name}" for linear_name in self.linear_names]
            full_weights = [state_dict.pop(key) for key in full_weight_names]
            param = named_parameters[new_name]
            state_dict[new_name] = create_local_fused_weight(tp_rank, tp_size, full_weights, param.partition_dim, self.fuse_axis)

        return state_dict

    def to_original_weights(self, module_fully_qualified_name: str, sharded_state_dicts: Dict[str, list[torch.Tensor]], parameters_metadata: Dict[str, Dict[str, Any]]) -> tuple[Dict[str, torch.Tensor], list[str]]:
        # To recreate original weights from the fused weights we need to:
        # 1. Unfuse the sharded weights
        # 2. Concat each unsharded local weight accross the partion_dim if TP is enabled
        original_weights = {}
        keys_to_remove = []
        for param_name in ["weight", "bias"] if self.bias else ["weight"]:
            unfused_local_weights = []
            fused_linear_weight_name = f"{module_fully_qualified_name}.{self.fused_linear_name}.{param_name}"
            fused_linear_sharded_weights = sharded_state_dicts[fused_linear_weight_name]
            for fused_local_weight in fused_linear_sharded_weights:
                unfused_local_weights.append(
                    torch.split(
                        fused_local_weight,
                        [dim // self.tp_size for dim in self.original_dims],
                        dim=self.fuse_axis,
                    )
                )

            for idx, linear_name in enumerate(self.linear_names):
                original_weight_name = f"{module_fully_qualified_name}.{linear_name}.{param_name}"
                partition_dim = parameters_metadata[fused_linear_weight_name]["partition_dim"]
                original_weight = torch.cat(
                    [unfused_local_weights[tp_rank][idx] for tp_rank in range(len(unfused_local_weights))],
                    dim=partition_dim,
                )
                original_weights[original_weight_name] = original_weight

            keys_to_remove.append(fused_linear_weight_name)

        return original_weights, keys_to_remove

@dataclass
class GQAQKVColumnParallelLinearSpecs(ModelWeightTransformationSpec):
    gqa_qkv_projection_name: str
    query_projection_name: str
    key_projection_name: str
    value_projection_name: str
    output_projection_name: str
    num_attention_heads: int
    num_key_value_heads: int
    kv_size_multiplier: int
    q_output_size_per_partition: int
    kv_output_size_per_partition: int
    fuse_qkv: bool
    bias: bool
    tp_size: int = field(default_factory=get_tensor_model_parallel_size)

    @staticmethod
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

    @staticmethod
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

    @staticmethod
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

        indices = GQAQKVColumnParallelLinearSpecs.compute_query_indices_for_rank(
            tp_size, tp_rank, num_attention_heads, num_key_value_heads, kv_size_multiplier
        )
        reshaped_weight = weight_data.view(num_attention_heads, head_dim, hidden_size)
        shuffled_weight = reshaped_weight[indices]
        shuffled_weight = shuffled_weight.reshape(-1, hidden_size)

        if query_or_output_proj == "output":
            shuffled_weight = shuffled_weight.transpose(0, 1)

        return shuffled_weight

    @staticmethod
    def create_gqa_query_or_output_projection_weight_from_full_weight(
        full_weight: torch.Tensor,
        tp_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        kv_size_multiplier: int,
        query_or_output: Union[Literal["query"], Literal["output"]],
    ):
        assert query_or_output in ["query", "output"]
        assert full_weight.device == torch.device("cpu")
        if query_or_output == "query":
            hidden_size = full_weight.size(1)
        else:
            hidden_size = full_weight.size(0)
            full_weight = full_weight.transpose(0, 1)

        indices = [
            GQAQKVColumnParallelLinearSpecs.compute_query_indices_for_rank(tp_size, tp_rank, num_attention_heads, num_key_value_heads, kv_size_multiplier)
            for tp_rank in range(tp_size)
        ]
        indices = torch.cat(indices, dim=0)
        reversed_indices = torch.sort(indices, dim=0).indices

        full_weight = full_weight.reshape(num_attention_heads, -1, hidden_size)
        full_weight = full_weight[reversed_indices]
        full_weight = full_weight.reshape(-1, hidden_size)

        if query_or_output == "output":
            full_weight = full_weight.transpose(0, 1)

        return full_weight


    def adapt_state_dict(self, module_fully_qualified_name: str, named_parameters: Dict[str, torch.nn.Parameter], orig_state_dict: Dict[str, torch.Tensor], inplace: bool = False):
        if inplace:
            state_dict = orig_state_dict
        else:
            state_dict = dict(orig_state_dict)

        param_names = ["weight", "bias"] if self.bias else ["weight"]
        for param_name in param_names:
            q_name = f"{module_fully_qualified_name}.{self.query_projection_name}.{param_name}"
            k_name = f"{module_fully_qualified_name}.{self.key_projection_name}.{param_name}"
            v_name = f"{module_fully_qualified_name}.{self.value_projection_name}.{param_name}"
            if self.fuse_qkv:
                new_name = f"{module_fully_qualified_name}.{self.gqa_qkv_projection_name}.{param_name}_qkv"

                full_weights = [
                    GQAQKVColumnParallelLinearSpecs.create_query_or_output_projection_local_weight_from_regular_weight(
                        state_dict.pop(q_name),
                        self.num_attention_heads,
                        self.num_key_value_heads,
                        self.kv_size_multiplier,
                        "query",
                    ),
                    GQAQKVColumnParallelLinearSpecs.create_kv_proj_local_weight_from_regular_weight(
                        state_dict.pop(k_name),
                        self.kv_size_multiplier,
                        self.kv_output_size_per_partition,
                    ),
                    GQAQKVColumnParallelLinearSpecs.create_kv_proj_local_weight_from_regular_weight(
                        state_dict.pop(v_name),
                        self.kv_size_multiplier,
                        self.kv_output_size_per_partition,
                    ),
                ]
                state_dict[new_name] = torch.cat(full_weights, dim=0)
            else:
                new_name_weight_q = f"{module_fully_qualified_name}.{self.gqa_qkv_projection_name}.{param_name}_q"
                new_name_weight_k = f"{module_fully_qualified_name}.{self.gqa_qkv_projection_name}.{param_name}_k"
                new_name_weight_v = f"{module_fully_qualified_name}.{self.gqa_qkv_projection_name}.{param_name}_v"

                state_dict[new_name_weight_q] = GQAQKVColumnParallelLinearSpecs.create_query_or_output_projection_local_weight_from_regular_weight(state_dict[q_name], self.num_attention_heads, self.num_key_value_heads, self.kv_size_multiplier, "query")

                state_dict[new_name_weight_k] = GQAQKVColumnParallelLinearSpecs.create_kv_proj_local_weight_from_regular_weight(state_dict[k_name], self.kv_size_multiplier, self.kv_output_size_per_partition)

                state_dict[new_name_weight_v] = GQAQKVColumnParallelLinearSpecs.create_kv_proj_local_weight_from_regular_weight(state_dict[v_name], self.kv_size_multiplier, self.kv_output_size_per_partition)

            output_projection_name = f"{module_fully_qualified_name}.{self.output_projection_name}.{param_name}"
            state_dict[output_projection_name] = GQAQKVColumnParallelLinearSpecs.create_query_or_output_projection_local_weight_from_regular_weight(state_dict[output_projection_name], self.num_attention_heads, self.num_key_value_heads, self.kv_size_multiplier, "output")

        return state_dict

    def to_original_weights(self, module_fully_qualified_name: str, sharded_state_dicts: Dict[str, list[torch.Tensor]], parameters_metadata: Dict[str, Dict[str, Any]]) -> tuple[Dict[str, torch.Tensor], list[str]]:
        param_names = ["weight", "bias"] if self.bias else ["weight"]
        state_dict = {}
        keys_to_remove = []
        for param_name in param_names:
            if self.fuse_qkv:
                fuse_qkv_weight_name = f"{module_fully_qualified_name}.{self.gqa_qkv_projection_name}.{param_name}_qkv"
                fused_qkv_local_weights = sharded_state_dicts[fuse_qkv_weight_name]

                slice_q = slice(0, self.q_output_size_per_partition)
                weights_q = [fused_qkv_local_weights[tp_rank][slice_q].contiguous() for tp_rank in range(self.tp_size)]

                slice_k = slice(
                    self.q_output_size_per_partition,
                    self.q_output_size_per_partition + self.kv_output_size_per_partition,
                )
                weights_k = [fused_qkv_local_weights[tp_rank][slice_k].contiguous() for tp_rank in range(self.tp_size)]

                slice_v = slice(
                    self.q_output_size_per_partition + self.kv_output_size_per_partition,
                    None,
                )
                weights_v = [fused_qkv_local_weights[tp_rank][slice_v].contiguous() for tp_rank in range(self.tp_size)]

                qkv_partition_dim = parameters_metadata[fuse_qkv_weight_name]["partition_dim"]
                keys_to_remove += [f"{module_fully_qualified_name}.{self.gqa_qkv_projection_name}.{param_name}_qkv"]
            else:
                weights_q = sharded_state_dicts[f"{module_fully_qualified_name}.{self.gqa_qkv_projection_name}.{param_name}_q"]
                weights_k = sharded_state_dicts[f"{module_fully_qualified_name}.{self.gqa_qkv_projection_name}.{param_name}_k"]
                weights_v = sharded_state_dicts[f"{module_fully_qualified_name}.{self.gqa_qkv_projection_name}.{param_name}_v"]
                # The query, key and value share the same partition dim.
                qkv_partition_dim = parameters_metadata[f"{module_fully_qualified_name}.{self.gqa_qkv_projection_name}.{param_name}_q"]["partition_dim"]
                keys_to_remove += [
                    f"{module_fully_qualified_name}.{self.gqa_qkv_projection_name}.{param_name}_q",
                    f"{module_fully_qualified_name}.{self.gqa_qkv_projection_name}.{param_name}_k",
                    f"{module_fully_qualified_name}.{self.gqa_qkv_projection_name}.{param_name}_v",
                ]

            weights_o = sharded_state_dicts[f"{module_fully_qualified_name}.{self.output_projection_name}.{param_name}"]


            full_weight_q = torch.cat(weights_q, dim=qkv_partition_dim).contiguous()
            full_weight_k = torch.cat(weights_k, dim=qkv_partition_dim).contiguous()
            full_weight_v = torch.cat(weights_v, dim=qkv_partition_dim).contiguous()

            o_partition_dim = parameters_metadata[f"{module_fully_qualified_name}.{self.output_projection_name}.{param_name}"]["partition_dim"]
            full_weight_o = torch.cat(weights_o, dim=o_partition_dim).contiguous()

            full_weight_q = GQAQKVColumnParallelLinearSpecs.create_gqa_query_or_output_projection_weight_from_full_weight(
                full_weight_q,
                self.tp_size,
                self.num_attention_heads,
                self.num_key_value_heads,
                self.kv_size_multiplier,
                "query",
            )
            full_weight_o = GQAQKVColumnParallelLinearSpecs.create_gqa_query_or_output_projection_weight_from_full_weight(
                full_weight_o,
                self.tp_size,
                self.num_attention_heads,
                self.num_key_value_heads,
                self.kv_size_multiplier,
                "output",
            )

            full_weight_k = torch.chunk(full_weight_k, self.kv_size_multiplier, dim=0)[0].detach().clone()
            full_weight_v = torch.chunk(full_weight_v, self.kv_size_multiplier, dim=0)[0].detach().clone()

            state_dict.update({
                f"{module_fully_qualified_name}.{self.query_projection_name}.{param_name}": full_weight_q,
                f"{module_fully_qualified_name}.{self.key_projection_name}.{param_name}": full_weight_k,
                f"{module_fully_qualified_name}.{self.value_projection_name}.{param_name}": full_weight_v,
                f"{module_fully_qualified_name}.{self.output_projection_name}.{param_name}": full_weight_o,
            })
        return state_dict, keys_to_remove
            
@dataclass
class ModelWeightTransformationSpecs:
    module_fully_qualified_name: Optional[str] = None
    specs: Union[ModelWeightTransformationSpec, list[ModelWeightTransformationSpec]] = field(default_factory=list)

    @classmethod
    def from_metadata(cls, specs_metadata: Dict[str, Any]):
        specs = cls(module_fully_qualified_name=specs_metadata["module_fully_qualified_name"])
        for spec_metadata in specs_metadata["specs"]:
            cls_name, metadata = spec_metadata
            # We dynamically import the class from the module.
            # We could use a dictionary as it is cleaner.
            cls_ = getattr(sys.modules[__name__], cls_name)
            spec = cls_(**metadata)
            specs.add_spec(spec)
        return specs

    def to_metadata(self) -> Dict[str, Any]:
        if self.module_fully_qualified_name is None:
            raise ValueError("`module_fully_qualified_name` must be set to serialize the specs")
        serialized_specs = []
        for spec in self.specs:
            serialized_specs.append((spec.__class__.__name__, asdict(spec)))

        return {
            "module_fully_qualified_name": self.module_fully_qualified_name,
            "specs": serialized_specs,
        }

    def __post_init__(self):
        if not isinstance(self.specs, list):
            self.specs = [self.specs]

    def add_spec(self, spec: ModelWeightTransformationSpec):
        if not isinstance(spec, ModelWeightTransformationSpec):
            raise TypeError(f"spec must be of type ModelWeightTransformationSpec, but got {type(spec)}")
        self.specs.append(spec)


    def adapt_state_dict(self, named_parameters: Dict[str, torch.nn.Parameter], orig_state_dict: Dict[str, torch.Tensor], inplace: bool = False):
        if self.module_fully_qualified_name is None:
            raise ValueError("`module_fully_qualified_name` must be set to adapt the state dict")
        for spec in self.specs:
            if not isinstance(spec, ModelWeightTransformationSpec):
                raise TypeError(f"spec must be of type ModelWeightTransformationSpec, but got {type(spec)}")
            orig_state_dict = spec.adapt_state_dict(self.module_fully_qualified_name, named_parameters, orig_state_dict, inplace=inplace)
        return orig_state_dict

    def to_original_weights(self, sharded_state_dicts: Dict[str, list[torch.Tensor]], parameters_metadata: Dict[str, Dict[str, Any]]) -> tuple[Dict[str, torch.Tensor], list[str]]:
        if self.module_fully_qualified_name is None:
            raise ValueError("`module_fully_qualified_name` must be set to adapt the state dict")
        original_weights = {}
        keys_to_remove = []
        for spec in self.specs:
            if not isinstance(spec, ModelWeightTransformationSpec):
                raise TypeError(f"spec must be of type ModelWeightTransformationSpec, but got {type(spec)}")
            spec_weights, spec_keys_to_remove = spec.to_original_weights(self.module_fully_qualified_name, sharded_state_dicts, parameters_metadata)
            original_weights.update(spec_weights)
            keys_to_remove.extend(spec_keys_to_remove)
        return original_weights, keys_to_remove

    def __iter__(self):
        return iter(self.specs)


def set_module_names_in_transformation_specs(model: torch.nn.Module):
    for name, mod in model.named_modules():
        specs = getattr(mod, "specs", None)
        if isinstance(specs, ModelWeightTransformationSpecs):
            specs.module_fully_qualified_name = name


def adapt_state_dict(model: torch.nn.Module, state_dict: Dict[str, torch.Tensor], inplace: bool = False):
    tp_size = get_tensor_model_parallel_size()
    named_parameters = dict(model.named_parameters())
    original_data_ptrs = {n: p.data_ptr() for n, p in state_dict.items()}
    original_state_dict_keys = set(state_dict.keys())
    for name, module in model.named_modules():
        model_weight_transformation_specs = getattr(module, "specs", None)
        if model_weight_transformation_specs is not None:
            state_dict = model_weight_transformation_specs.adapt_state_dict(named_parameters, state_dict, inplace=inplace)

    # There are 2 cases:
    # 1. A new key was inserted by the adapt_state_dict function
    # 2. A key was mutated by the adapt_state_dict function
    new_keys = set(state_dict.keys()) - original_state_dict_keys
    mutated_keys = {n for n, p in state_dict.items() if p.data_ptr() != original_data_ptrs.get(n, p.data_ptr())}

    for name, param in model.named_parameters():
        if name in new_keys | mutated_keys:
            continue
        if hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel:
            if param.partition_dim not in [0, 1]:
                raise Exception(f"Partiton value of 0,1 are supported, found {param.partition_dim}.")

            # It means there are no weights in the state dict for the current parameter.
            if name not in state_dict:
                continue

            full_weight = state_dict[name]
            per_partition_size = full_weight.shape[param.partition_dim] // tp_size
            state_dict[name] = create_local_weight(
                full_weight, param.partition_dim, per_partition_size, param.partition_stride
            )
    return state_dict

def to_original_weights(transformations_specs: list[ModelWeightTransformationSpecs], sharded_state_dicts: Dict[str, list[torch.Tensor]], parameters_metadata: Dict[str, Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    consolidated_state_dict = {}
    parameters_to_remove = set()
    for specs in transformations_specs:
        original_weights, keys_to_remove = specs.to_original_weights(
            sharded_state_dicts, parameters_metadata
        )
        consolidated_state_dict.update(original_weights)
        parameters_to_remove.update(keys_to_remove)

    for name, metadata in parameters_metadata.items():
        # It means it was already processed by the transformation specs.
        if name in consolidated_state_dict or name in parameters_to_remove:
            continue

        is_tensor_model_parallel = metadata["tensor_model_parallel"]
        if is_tensor_model_parallel:
            consolidated_weight = torch.cat(
                sharded_state_dicts[name], dim=metadata["partition_dim"]
            )
            consolidated_state_dict[name] = consolidated_weight
        else:
            consolidated_state_dict[name] = sharded_state_dicts[name][0]
    return consolidated_state_dict


def create_parameter_metadata(model):
    metadata = {"parameters": {}, "model_weight_transformation_specs": []}
    for name, param in model.named_parameters():
        tensor_model_parallel = getattr(param, "tensor_model_parallel", False)
        if tensor_model_parallel:
            metadata["parameters"][name] = {
                "tensor_model_parallel": tensor_model_parallel,
                "partition_dim": param.partition_dim,
                "partition_stride": param.partition_stride,
                "num_partitions": param.num_partitions,
            }
        else:
            metadata["parameters"][name] = {
                "tensor_model_parallel": tensor_model_parallel,
            }
    for name, module in model.named_modules():
        model_weight_transformation_specs = getattr(module, "specs", None)
        if model_weight_transformation_specs is not None:
            serialized_specs = model_weight_transformation_specs.to_metadata()
            metadata["model_weight_transformation_specs"].append(serialized_specs)
    return metadata



class NeuronModelMixin:
    @classmethod
    def _check_and_enable_flash_attn_2(
        cls,
        config,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        check_device_map: bool = True,
        hard_check_only: bool = False,
    ) -> PretrainedConfig:
        """
        Checks the availability of Flash Attention 2 and compatibility with the current model.

        If all checks pass and `hard_check_only` is False, the method will set the config attribute `attn_implementation` to "flash_attention_2" so that the model can initialize the correct attention module.
        """
        if not cls._supports_flash_attn_2:
            raise ValueError(
                f"{cls.__name__} does not support Flash Attention 2.0 yet. Please request to add support where"
                f" the model is hosted, on its model hub page: https://huggingface.co/{config._name_or_path}/discussions/new"
                " or in the Transformers GitHub repo: https://github.com/huggingface/transformers/issues/new"
            )

        if torch_dtype is None:
            logger.warning_once(
                "You are attempting to use Flash Attention 2.0 without specifying a torch dtype. This might lead to unexpected behaviour"
            )
        elif torch_dtype is not None and torch_dtype not in [torch.float16, torch.bfloat16]:
            logger.warning_once(
                "Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes, but"
                f" the current dype in {cls.__name__} is {torch_dtype}. You should run training or inference using Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):` decorator,"
                ' or load the model with the `torch_dtype` argument. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="flash_attention_2", torch_dtype=torch.float16)`'
            )

        # The check `torch.empty(0).device.type != "xla"` is needed as the model may be initialized after `torch.set_default_device` has been called,
        # or the model may be initialized under the context manager `with torch.device("cuda"):`.
        if check_device_map and device_map is None and torch.empty(0).device.type != "xla":
            logger.warning_once(
                "You are attempting to use Flash Attention 2.0 with a model not initialized on XLA. Make sure to move the model to XLA"
                " after initializing it on CPU with `model.to('xla')`."
            )
        elif (
            check_device_map
            and device_map is not None
            and isinstance(device_map, dict)
            and ("cpu" in device_map.values() or "disk" in device_map.values())
        ):
            raise ValueError(
                "You are attempting to use Flash Attention 2.0 with a model dispatched on CPU or disk. This is not supported. Please make sure to "
                "initialise the model on XLA by passing a device_map that contains only GPU devices as keys."
            )
        if not hard_check_only:
            config._attn_implementation = "flash_attention_2"
        return config

    @classmethod
    def _autoset_attn_implementation(
        cls,
        config,
        use_flash_attention_2: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        check_device_map: bool = True,
    ):
        """
        Automatically checks and dispatches to a default attention implementation. In order of priority:
            1. An implementation specified in `config._attn_implementation` (due for example to the argument attn_implementation="sdpa" in from_pretrained).
            2. DEPRECATED: if use_flash_attention_2 is set to `True` and `flash_attn` is available, flash attention. (`LlamaFlashAttention` for example)
            3. SDPA implementation, if available and supported by the model type. (`LlamaSdpaAttention` for example)
            4. The default model's implementation otherwise (`LlamaAttention` for example) .
        """
        # Here we use config._attn_implementation_internal to check whether the attention implementation was explicitely set by the user.
        # The property `PretrainedConfig._attn_implementation` is never `None`, for backward compatibility (always fall back on "eager").
        # The `hasattr` here is used as some Transformers tests for some reason do not call PretrainedConfig __init__ (e.g. test_no_super_init_config_and_model)
        requested_attn_implementation = None
        if hasattr(config, "_attn_implementation_internal") and config._attn_implementation_internal is not None:
            if config._attn_implementation != "flash_attention_2" and use_flash_attention_2:
                raise ValueError(
                    f'Both attn_implementation="{config._attn_implementation}" and `use_flash_attention_2=True` were used when loading the model, which are not compatible.'
                    ' We recommend to just use `attn_implementation="flash_attention_2"` when loading the model.'
                )

            if not isinstance(config._attn_implementation, dict) and config._attn_implementation not in [
                "eager"
            ] + list(ALL_ATTENTION_FUNCTIONS.keys()):
                message = f'Specified `attn_implementation="{config._attn_implementation}"` is not supported. The only possible arguments are `attn_implementation="eager"` (manual attention implementation)'
                if cls._supports_flash_attn_2:
                    message += ', `"attn_implementation=flash_attention_2"` (implementation using nki flash attention 2)'
                # Keeping this if supported one day.
                # if cls._supports_flex_attn:
                #     message += (
                #         ', `"attn_implementation=flex_attention"` (implementation using torch\'s flex_attention)'
                #     )
                raise ValueError(message + ".")

            # If a config is passed with a preset attn_implementation, we skip the automatic dispatch and use the user-provided config, with hard checks that the requested attention implementation is available.
            requested_attn_implementation = config._attn_implementation_internal

        # Composite models consisting of several PretrainedModels have to specify attention impl as a dict
        # where keys are sub-config names. But most people will specify one `str` which means that should dispatch it
        # for all sub-models.
        # Below we check if a config is composite and manually prepare a dict of attn impl if not already passed as a dict.
        # Later each sub-module will dispatch with its own attn impl, by calling `XXXModel._from_config(config.text_config)`
        # If any of sub-modules doesn't support requested attn, an error will be raised. See https://github.com/huggingface/transformers/pull/32238
        for key in config.sub_configs.keys():
            sub_config = getattr(config, key)
            curr_attn_implementation = (
                requested_attn_implementation
                if not isinstance(requested_attn_implementation, dict)
                else requested_attn_implementation.get(key, None)
            )
            sub_config._attn_implementation_internal = curr_attn_implementation

        if use_flash_attention_2:
            logger.warning_once(
                'The model was loaded with use_flash_attention_2=True, which is deprecated and may be removed in a future release. Please use `attn_implementation="flash_attention_2"` instead.'
            )
            config._attn_implementation = "flash_attention_2"

        if config._attn_implementation == "flash_attention_2":
            cls._check_and_enable_flash_attn_2(
                config,
                torch_dtype=torch_dtype,
                device_map=device_map,
                hard_check_only=False,
                check_device_map=check_device_map,
            )
        # Keeping it if supported one day.
        # elif requested_attn_implementation == "flex_attention":
        #     config = cls._check_and_enable_flex_attn(config, hard_check_only=True)
        elif requested_attn_implementation in list(ALL_ATTENTION_FUNCTIONS.keys()):
            config._attn_implementation = requested_attn_implementation
        elif isinstance(requested_attn_implementation, dict):
            config._attn_implementation = None
        else:
            config._attn_implementation = "eager"

        config._attn_implementation_autoset = True
        return config

    # This method uses `torch.xla.utils.checkpoint.checkpoint` instead of the torch one.
    def _set_gradient_checkpointing(self, enable: bool = True, gradient_checkpointing_func: Callable = checkpoint):
        return super()._set_gradient_checkpointing(enable=enable, gradient_checkpointing_func=gradient_checkpointing_func)

    @classmethod
    def from_pretrained(
        cls: Type[SpecificPreTrainedModelType],
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: Optional[bool] = None,
        weights_only: bool = True,
        **kwargs,
    ) -> SpecificPreTrainedModelType:
        state_dict = kwargs.pop("state_dict", None)
        kwargs.pop("from_tf", False)
        kwargs.pop("from_flax", False)
        resume_download = kwargs.pop("resume_download", None)
        proxies = kwargs.pop("proxies", None)
        kwargs.pop("output_loading_info", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        _ = kwargs.pop("mirror", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        _fast_init = kwargs.pop("_fast_init", True)
        torch_dtype = kwargs.pop("torch_dtype", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", None)
        device_map = kwargs.pop("device_map", None)
        kwargs.pop("max_memory", None)
        kwargs.pop("offload_folder", None)
        kwargs.pop("offload_state_dict", False)
        kwargs.pop("offload_buffers", False)
        load_in_8bit = kwargs.pop("load_in_8bit", False)
        load_in_4bit = kwargs.pop("load_in_4bit", False)
        kwargs.pop("quantization_config", None)
        subfolder = kwargs.pop("subfolder", "")
        commit_hash = kwargs.pop("_commit_hash", None)
        kwargs.pop("variant", None)
        adapter_kwargs = kwargs.pop("adapter_kwargs", {})
        kwargs.pop("adapter_name", "default")
        use_flash_attention_2 = kwargs.pop("use_flash_attention_2", False)
        kwargs.pop("generation_config", None)

        kwargs.pop("gguf_file", None)
        # Cache path to the GGUF file

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if token is not None and adapter_kwargs is not None and "token" not in adapter_kwargs:
            adapter_kwargs["token"] = token

        if use_safetensors is None and not is_safetensors_available():
            use_safetensors = False
        if trust_remote_code is True:
            logger.warning(
                "The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is"
                " ignored."
            )

        if commit_hash is None:
            if not isinstance(config, PretrainedConfig):
                # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
                resolved_config_file = cached_file(
                    pretrained_model_name_or_path,
                    CONFIG_NAME,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_gated_repo=False,
                    _raise_exceptions_for_missing_entries=False,
                    _raise_exceptions_for_connection_errors=False,
                )
                commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
            else:
                commit_hash = getattr(config, "_commit_hash", None)

        if is_peft_available():
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

        if device_map not in [None, "xla", "cpu"]:
            raise RuntimeError('The only device map values supported are: `None`, "cpu" or "xla".')

        if low_cpu_mem_usage is not None:
            raise RuntimeError("Low cpu memory usage is not supported for optimum-neuron.")

        if load_in_4bit or load_in_8bit:
            raise RuntimeError("Quantization is not supported yet.")

        user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True


        # Load config if we don't provide a configuration
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
            # In case one passes a config to `from_pretrained` + "attn_implementation"
            # override the `_attn_implementation` attribute to `attn_implementation` of the kwargs
            # Please see: https://github.com/huggingface/transformers/issues/28038

            # Overwrite `config._attn_implementation` by the one from the kwargs --> in auto-factory
            # we pop attn_implementation from the kwargs but this handles the case where users
            # passes manually the config to `from_pretrained`.
            config = copy.deepcopy(config)

            kwarg_attn_imp = kwargs.pop("attn_implementation", None)
            if kwarg_attn_imp is not None:
                config._attn_implementation = kwarg_attn_imp

            model_kwargs = kwargs

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

        is_sharded = sharded_metadata is not None

        if torch_dtype is not None:
            if isinstance(torch_dtype, str):
                if torch_dtype == "auto":
                    if hasattr(config, "torch_dtype") and config.torch_dtype is not None:
                        torch_dtype = config.torch_dtype
                        logger.info(f"Will use torch_dtype={torch_dtype} as defined in model's config object")
                    else:
                        if is_sharded and "dtype" in sharded_metadata:
                            torch_dtype = sharded_metadata["dtype"]
                        elif not is_sharded:
                            torch_dtype = get_state_dict_dtype(state_dict)
                        else:
                            one_state_dict = load_state_dict(filenames[0], weights_only=weights_only)
                            torch_dtype = get_state_dict_dtype(one_state_dict)
                            del one_state_dict  # free CPU memory
                        logger.info(
                            "Since the `torch_dtype` attribute can't be found in model's config object, "
                            "will use torch_dtype={torch_dtype} as derived from model's weights"
                        )
                elif hasattr(torch, torch_dtype):
                    torch_dtype = getattr(torch, torch_dtype)
                else:
                    raise ValueError(
                        f'`torch_dtype` can be one of: `torch.dtype`, `"auto"` or a string of a valid `torch.dtype`, but received {torch_dtype}'
                    )
            config.torch_dtype = torch_dtype

        config.name_or_path = pretrained_model_name_or_path

        # Instantiate model.
        init_contexts = [no_init_weights(_enable=_fast_init)]

        config = copy.deepcopy(config)  # We do not want to modify the config inplace in from_pretrained.
        # This is due to a Neuron compiler bug, and it should be removed when the bug is fixed.
        should_fake_tie = config.tie_word_embeddings
        config.tie_word_embeddings = False
        if not getattr(config, "_attn_implementation_autoset", False):
            # We do not check for the device_map because we are going to move the model to XLA anyway on our own.
            config = cls._autoset_attn_implementation(
                config, use_flash_attention_2=use_flash_attention_2, torch_dtype=torch_dtype, device_map=device_map, check_device_map=False,
            )

        with ContextManagers(init_contexts):
            # Let's make sure we don't run the init function of buffer modules
            model = cls(config, *model_args, **model_kwargs)

        # make sure we use the model's config since the __init__ call might have copied it
        config = model.config

        state_dict  = {}

        mp_config = model.mp_config
        num_local_ranks_per_step = mp_config.num_local_ranks_per_step
        local_world_size = get_local_world_size()
        local_rank = xm.get_local_ordinal()
        if num_local_ranks_per_step <= 0:
            num_local_ranks_per_step = local_world_size

        for worker in range(math.ceil(local_world_size / num_local_ranks_per_step)):
            model_to_load = model
            if local_rank // num_local_ranks_per_step == worker:
                if sharded_metadata:
                    weight_map = sharded_metadata["weight_map"]
                else:
                    filename = Path(filenames)
                    # TODO: manage the safetensor check dependency.
                    with safe_open(filename, framework="pt", device="cpu") as fp:
                        weight_map = {weight_name: filename for weight_name in fp.keys()}

                state_dict = {}
                for weight_name, filename in weight_map.items():
                    with safe_open(filename, framework="pt", device="cpu") as fp:
                        state_dict[weight_name] = fp.get_tensor(weight_name)

                prefix = model.base_model_prefix
                loaded_keys = state_dict.keys()
                expected_keys = model.state_dict().keys()
                if len(prefix) > 0:
                    has_prefix_module = any(s.startswith(prefix) for s in loaded_keys)
                    expects_prefix_module = any(s.startswith(prefix) for s in expected_keys)
                else:
                    has_prefix_module = False
                    expects_prefix_module = False

                # key re-naming operations are never done on the keys
                # that are loaded, but always on the keys of the newly initialized model
                remove_prefix_from_model = not has_prefix_module and expects_prefix_module
                add_prefix_to_model = has_prefix_module and not expects_prefix_module

                if remove_prefix_from_model:
                    model_to_load = getattr(model, prefix)
                    # expected_keys_not_prefixed = [s for s in expected_keys if not s.startswith(_prefix)]
                    # expected_keys = [s[len(_prefix) :] if s.startswith(_prefix) else s for s in expected_keys]
                elif add_prefix_to_model:
                    state_dict = {".".join([prefix, key]): value for key, value in state_dict.items()}
                    # expected_keys = [".".join([prefix, s]) for s in expected_keys]

                # This is required to have the specs properly defined.
                set_module_names_in_transformation_specs(model_to_load)

                # Adapts the state dict to the custom model.
                state_dict = adapt_state_dict(model_to_load, state_dict, inplace=False)
                model_to_load.load_state_dict(state_dict, strict=False)

                if torch_dtype is not None:
                    model = model.to(torch_dtype)
                if device_map == "xla":
                    move_model_to_device(model, xm.xla_device())

                gc.collect()
                model.tie_weights()

                # Now we set the modules names using the full model regardless of prefixes.
                # This is this name that will be saved and used when re-loading the model.
                set_module_names_in_transformation_specs(model)

            # It is important to initialize modules that are not in the state dict.
            if _fast_init:
                # We call "set_initialized_submodules" twice:
                # One with `model_to_load` to handle the base submodules from the state dict
                # And one with `model`, which should contain anything that is not in the base model
                not_initialized_submodules = set_initialized_submodules(model_to_load, state_dict.keys())
                if model is not model_to_load:
                    not_initialized_submodules.update(set_initialized_submodules(model, state_dict.keys()))
                for name, mod in not_initialized_submodules.items():
                    if getattr(mod, "_is_hf_initialized", False):
                        # It means that it was set as initialized by the first `set_initialized_submodules`, we can
                        # skip.
                        continue
                    elif isinstance(mod, BaseParallelLinear):
                        mod.initialize_weight_and_bias()
                    print(f"Initializing {name} with default weights")
                    model._initialize_weights(mod)

            xm.rendezvous(f"load_state_dict_{worker}")

        # Currently tie_word_embeddings leads to a compiler bug.
        # If weights are initially tied, we still copy the value but we do not tie them.
        if should_fake_tie:
            with torch.no_grad():
                model.get_output_embeddings().weight.data.copy_(model.get_input_embeddings().weight)

        xm.mark_step()

        return model

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: Union[bool, Literal["auto"]] = "auto",
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = False,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs,
    ):
        if is_precompilation():
            return
        if is_main_process == "auto":
            is_main_process = is_main_worker()

        use_auth_token = kwargs.pop("use_auth_token", None)
        kwargs.pop("ignore_metadata_errors", False)

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if token is not None:
            kwargs["token"] = token

        _hf_peft_config_loaded = getattr(self, "_hf_peft_config_loaded", False)

        if "save_config" in kwargs:
            warnings.warn(
                "`save_config` is deprecated and will be removed in v5 of Transformers. Use `is_main_process` instead."
            )
            is_main_process = kwargs.pop("save_config")
        if safe_serialization:
            raise logger.error("`safe_serialization` is not supported when saving the sharded checkpoints. It is possible to consolidate the model weights into `safetensors` format.")

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        save_directory = Path(save_directory)

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            raise RuntimeError("`push_to_hub` is not supported because checkpoints are sharded. Consolidate them then push to hub.")

        model_to_save = self

        # save the string version of dtype to the config, e.g. convert torch.float32 => "float32"
        # we currently don't use this setting automatically, but may start to use with v5
        dtype = get_parameter_dtype(model_to_save)
        model_to_save.config.torch_dtype = str(dtype).split(".")[1]

        # Attach architecture to the config
        model_to_save.config.architectures = [model_to_save.__class__.__name__]

        # Unset attn implementation so it can be set to another one when loading back
        model_to_save.config._attn_implementation_autoset = False

        # Save the config
        if is_main_process:
            if not _hf_peft_config_loaded:
                # If the model config has set attributes that should be in the generation config, move them there.
                misplaced_generation_parameters = model_to_save.config._get_non_default_generation_parameters()
                if self.can_generate() and len(misplaced_generation_parameters) > 0:
                    warnings.warn(
                        "Moving the following attributes in the config to the generation config: "
                        f"{misplaced_generation_parameters}. You are seeing this warning because you've set "
                        "generation parameters in the model config, as opposed to in the generation config.",
                        UserWarning,
                    )
                    for param_name, param_value in misplaced_generation_parameters.items():
                        setattr(model_to_save.generation_config, param_name, param_value)
                        setattr(model_to_save.config, param_name, None)

                model_to_save.config.save_pretrained(save_directory)
            if self.can_generate():
                model_to_save.generation_config.save_pretrained(save_directory)

            if _hf_peft_config_loaded:
                logger.info(
                    "Detected adapters on the model, saving the model in the PEFT format, only adapter weights will be saved."
                )
                state_dict = model_to_save.get_adapter_state_dict()

                if save_peft_format:
                    logger.info(
                        "To match the expected format of the PEFT library, all keys of the state dict of adapters will be pre-pended with `base_model.model`."
                    )
                    peft_state_dict = {}
                    for key, value in state_dict.items():
                        peft_state_dict[f"base_model.model.{key}"] = value
                    state_dict = peft_state_dict

                active_adapter = self.active_adapters()

                if len(active_adapter) > 1:
                    raise ValueError(
                        "Multiple active adapters detected, saving multiple active adapters is not supported yet. You can save adapters separately one by one "
                        "by iteratively calling `model.set_adapter(adapter_name)` then `model.save_pretrained(...)`"
                    )
                active_adapter = active_adapter[0]

                current_peft_config = self.peft_config[active_adapter]
                current_peft_config.save_pretrained(save_directory)

            with open(save_directory / "mp_config.json", "w") as f:
                mp_config_data = asdict(self.mp_config)
                if isinstance(mp_config_data["checkpoint_dir"], Path):
                    mp_config_data["checkpoint_dir"] = mp_config_data["checkpoint_dir"].as_posix()
                f.write(json.dumps(mp_config_data, indent=4))


        # Saving the metadata required to consolidate the checkpoints properly.
        if get_data_parallel_rank() == 0 and get_tensor_model_parallel_rank() == 0:
            metadata = create_parameter_metadata(model_to_save)
            pp_rank = get_pipeline_model_parallel_rank()
            metadata_path = save_directory / MODEL_PARALLEL_SHARDS_DIR_NAME / f"mp_metadata_pp_rank_{pp_rank}.json"
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metadata_path, "w") as f:
                f.write(json.dumps(metadata, indent=4))

        neuronx_distributed.trainer.save_checkpoint(
            save_directory.as_posix(),
            tag=MODEL_PARALLEL_SHARDS_DIR_NAME,
            model=self,
            optimizer=optimizer,
            use_xser=self.mp_config.use_xser,
            async_save=self.mp_config.async_save,
            num_workers=self.mp_config.num_local_ranks_per_step,
        )
