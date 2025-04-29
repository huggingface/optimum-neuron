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

import sys
from abc import abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional, Union

import torch

from ...utils.import_utils import is_neuronx_distributed_available


if is_neuronx_distributed_available():
    from neuronx_distributed.parallel_layers.layers import create_local_weight
    from neuronx_distributed.parallel_layers.parallel_state import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_size,
    )
else:
    # We define this dummy function in case we do not have neuronx_distributed, for instance when building the docs.
    def get_tensor_model_parallel_size():
        return 0


def create_local_fused_weight(tp_rank, tp_size, individual_weights, partition_dim, fuse_axis, out_weight=None):
    """
    Shards individual weights across the tensor parallel ranks and fuses them into a single weight.
    """
    weight_lists = []
    for weight in individual_weights:
        weight_list = torch.split(weight, weight.size(partition_dim) // tp_size, dim=partition_dim)[tp_rank::tp_size]
        weight_lists.append(weight_list)

    with torch.no_grad():
        return torch.cat(
            [torch.cat(weight_list, dim=partition_dim) for weight_list in weight_lists],
            dim=fuse_axis,
            out=out_weight,
        )


class ModelWeightTransformationSpec:
    """
    This class defines the interface for transforming model weights between the original Transformers implementation
    and the custom implementation for Neuron.
    """

    @abstractmethod
    def adapt_state_dict(
        self,
        module_fully_qualified_name: str,
        named_parameters: Dict[str, torch.nn.Parameter],
        orig_state_dict: Dict[str, torch.Tensor],
        inplace: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Transforms the state dict from the original Transformers model to match the custom modeling implementation.
        """
        pass

    @abstractmethod
    def to_original_weights(
        self, sharded_state_dicts: Dict[str, list[torch.Tensor]], parameters_metadata: Dict[str, Dict[str, Any]]
    ) -> tuple[Dict[str, torch.Tensor], list[str]]:
        """
        Produces the weights associated to this transformation spec from the custom model to match the original
        Transformers weights.

        Args:
            sharded_state_dicts (Dict[str, list[torch.Tensor]]): The sharded state dicts from the custom modeling
                implementation.
            parameters_metadata (Dict[str, Dict[str, Any]]): Metadata about the parameters in the original model.

        Returns:
            tuple[Dict[str, torch.Tensor], list[str]]: A tuple containing the transformed weights and a list of the
            names of the parameters to remove from the final state dict.
        """
        pass


@dataclass
class ModelWeightTransformationSpecs:
    """
    Defines a list of transformation specs for a given module of the model.
    """

    module_fully_qualified_name: Optional[str] = None
    specs: Union[ModelWeightTransformationSpec, list[ModelWeightTransformationSpec]] = field(default_factory=list)

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

    def __post_init__(self):
        if not isinstance(self.specs, list):
            self.specs = [self.specs]

    def __iter__(self):
        return iter(self.specs)

    def add_spec(self, spec: ModelWeightTransformationSpec):
        if not isinstance(spec, ModelWeightTransformationSpec):
            raise TypeError(f"spec must be of type ModelWeightTransformationSpec, but got {type(spec)}")
        self.specs.append(spec)

    def adapt_state_dict(
        self,
        named_parameters: Dict[str, torch.nn.Parameter],
        orig_state_dict: Dict[str, torch.Tensor],
        inplace: bool = False,
    ):
        if self.module_fully_qualified_name is None:
            raise ValueError("`module_fully_qualified_name` must be set to adapt the state dict")
        for spec in self.specs:
            if not isinstance(spec, ModelWeightTransformationSpec):
                raise TypeError(f"spec must be of type ModelWeightTransformationSpec, but got {type(spec)}")
            orig_state_dict = spec.adapt_state_dict(
                self.module_fully_qualified_name, named_parameters, orig_state_dict, inplace=inplace
            )
        return orig_state_dict

    def to_original_weights(
        self, sharded_state_dicts: Dict[str, list[torch.Tensor]], parameters_metadata: Dict[str, Dict[str, Any]]
    ) -> tuple[Dict[str, torch.Tensor], list[str]]:
        if self.module_fully_qualified_name is None:
            raise ValueError("`module_fully_qualified_name` must be set to adapt the state dict")
        original_weights = {}
        keys_to_remove = []
        for spec in self.specs:
            if not isinstance(spec, ModelWeightTransformationSpec):
                raise TypeError(f"spec must be of type ModelWeightTransformationSpec, but got {type(spec)}")
            spec_weights, spec_keys_to_remove = spec.to_original_weights(
                self.module_fully_qualified_name, sharded_state_dicts, parameters_metadata
            )
            original_weights.update(spec_weights)
            keys_to_remove.extend(spec_keys_to_remove)
        return original_weights, keys_to_remove


class CustomModule:
    """
    This class is used to mark a module as a custom module. It is used to identify the modules that contain weights
    that need to transformed when loading and saving the model.
    """
    @property
    def specs(self) -> ModelWeightTransformationSpecs:
        if not hasattr(self, "_specs"):
            self._specs = ModelWeightTransformationSpecs()
        return self._specs

    @specs.setter
    def specs(self, specs: ModelWeightTransformationSpecs):
        if not isinstance(specs, ModelWeightTransformationSpecs):
            raise TypeError(f"specs must be of type ModelWeightTransformationSpecs, but got {type(specs)}")
        self._specs = specs

    def __repr__(self):
        return f"CustomModule(specs={self.specs})"

@dataclass
class FusedLinearsSpec(ModelWeightTransformationSpec):
    """
    Represents a transformation where multiple linear layers are fused into a single linear layer.
    It can handle the case where the fused linear layer is sharded across multiple tensor parallel ranks.
    """

    fused_linear_name: str
    linear_names: list[str]
    bias: bool
    fuse_axis: Union[Literal[0], Literal[1], Literal["column"], Literal["row"]]
    original_dims: list[int]
    tp_size: int = field(default_factory=get_tensor_model_parallel_size)

    def __post_init__(self):
        if self.fuse_axis == "column":
            self.fuse_axis = 0
        elif self.fuse_axis == "row":
            self.fuse_axis = 1

    def adapt_state_dict(
        self,
        module_fully_qualified_name: str,
        named_parameters: Dict[str, torch.nn.Parameter],
        orig_state_dict: Dict[str, torch.Tensor],
        inplace: bool = False,
    ) -> Dict[str, torch.Tensor]:
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
            full_weight_names = [
                f"{module_fully_qualified_name}.{linear_name}.{param_name}" for linear_name in self.linear_names
            ]
            full_weights = [state_dict.pop(key) for key in full_weight_names]
            param = named_parameters[new_name]
            state_dict[new_name] = create_local_fused_weight(
                tp_rank, tp_size, full_weights, param.partition_dim, self.fuse_axis
            )

        return state_dict

    def to_original_weights(
        self,
        module_fully_qualified_name: str,
        sharded_state_dicts: Dict[str, list[torch.Tensor]],
        parameters_metadata: Dict[str, Dict[str, Any]],
    ) -> tuple[Dict[str, torch.Tensor], list[str]]:
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
class GQAQKVColumnParallelLinearSpec(ModelWeightTransformationSpec):
    """
    Represents the transformation of separate query, key, and value projections into a single GQAQKVColumnParalleLinear
    projection.
    """

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
        Computes the permutation for the query weight for a given TP rank.
        """
        num_attention_heads_per_rank = num_attention_heads // tp_size
        num_key_value_heads_per_rank = (num_key_value_heads * kv_size_multiplier) // tp_size
        query_group_size = num_attention_heads // num_key_value_heads
        query_group_size_per_rank = num_attention_heads_per_rank // num_key_value_heads_per_rank

        queries_indices = [torch.arange(query_group_size_per_rank) for _ in range(num_key_value_heads_per_rank)]

        keys_indices = torch.arange(num_key_value_heads).repeat(kv_size_multiplier)
        keys_indices = torch.repeat_interleave(
            keys_indices, num_attention_heads_per_rank // num_key_value_heads_per_rank
        )
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
        Creates the local version of the key or value projections weight for the given TP rank.
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
        Creates the local version of the query or output projections weight for the given TP rank.
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

        indices = GQAQKVColumnParallelLinearSpec.compute_query_indices_for_rank(
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
            GQAQKVColumnParallelLinearSpec.compute_query_indices_for_rank(
                tp_size, tp_rank, num_attention_heads, num_key_value_heads, kv_size_multiplier
            )
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

    def adapt_state_dict(
        self,
        module_fully_qualified_name: str,
        named_parameters: Dict[str, torch.nn.Parameter],
        orig_state_dict: Dict[str, torch.Tensor],
        inplace: bool = False,
    ) -> Dict[str, torch.Tensor]:
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
                    GQAQKVColumnParallelLinearSpec.create_query_or_output_projection_local_weight_from_regular_weight(
                        state_dict.pop(q_name),
                        self.num_attention_heads,
                        self.num_key_value_heads,
                        self.kv_size_multiplier,
                        "query",
                    ),
                    GQAQKVColumnParallelLinearSpec.create_kv_proj_local_weight_from_regular_weight(
                        state_dict.pop(k_name),
                        self.kv_size_multiplier,
                        self.kv_output_size_per_partition,
                    ),
                    GQAQKVColumnParallelLinearSpec.create_kv_proj_local_weight_from_regular_weight(
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

                state_dict[new_name_weight_q] = (
                    GQAQKVColumnParallelLinearSpec.create_query_or_output_projection_local_weight_from_regular_weight(
                        state_dict[q_name],
                        self.num_attention_heads,
                        self.num_key_value_heads,
                        self.kv_size_multiplier,
                        "query",
                    )
                )

                state_dict[new_name_weight_k] = (
                    GQAQKVColumnParallelLinearSpec.create_kv_proj_local_weight_from_regular_weight(
                        state_dict[k_name], self.kv_size_multiplier, self.kv_output_size_per_partition
                    )
                )

                state_dict[new_name_weight_v] = (
                    GQAQKVColumnParallelLinearSpec.create_kv_proj_local_weight_from_regular_weight(
                        state_dict[v_name], self.kv_size_multiplier, self.kv_output_size_per_partition
                    )
                )

            output_projection_name = f"{module_fully_qualified_name}.{self.output_projection_name}.{param_name}"
            state_dict[output_projection_name] = (
                GQAQKVColumnParallelLinearSpec.create_query_or_output_projection_local_weight_from_regular_weight(
                    state_dict[output_projection_name],
                    self.num_attention_heads,
                    self.num_key_value_heads,
                    self.kv_size_multiplier,
                    "output",
                )
            )

        return state_dict

    def to_original_weights(
        self,
        module_fully_qualified_name: str,
        sharded_state_dicts: Dict[str, list[torch.Tensor]],
        parameters_metadata: Dict[str, Dict[str, Any]],
    ) -> tuple[Dict[str, torch.Tensor], list[str]]:
        state_dict = {}
        keys_to_remove = []
        param_names = ["weight", "bias"] if self.bias else ["weight"]
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
                query_name = f"{module_fully_qualified_name}.{self.gqa_qkv_projection_name}.{param_name}_q"
                weights_q = sharded_state_dicts[query_name]

                key_name = f"{module_fully_qualified_name}.{self.gqa_qkv_projection_name}.{param_name}_k"
                weights_k = sharded_state_dicts[key_name]

                value_name = f"{module_fully_qualified_name}.{self.gqa_qkv_projection_name}.{param_name}_v"
                weights_v = sharded_state_dicts[value_name]

                # The query, key and value share the same partition dim.
                qkv_partition_dim = parameters_metadata[query_name]["partition_dim"]
                keys_to_remove += [query_name, key_name, value_name]

            full_weight_q = torch.cat(weights_q, dim=qkv_partition_dim).contiguous()
            full_weight_k = torch.cat(weights_k, dim=qkv_partition_dim).contiguous()
            full_weight_v = torch.cat(weights_v, dim=qkv_partition_dim).contiguous()

            output_projection_name = f"{module_fully_qualified_name}.{self.output_projection_name}.{param_name}"
            weights_o = sharded_state_dicts[output_projection_name]
            o_partition_dim = parameters_metadata[output_projection_name]["partition_dim"]
            full_weight_o = torch.cat(weights_o, dim=o_partition_dim).contiguous()

            full_weight_q = (
                GQAQKVColumnParallelLinearSpec.create_gqa_query_or_output_projection_weight_from_full_weight(
                    full_weight_q,
                    self.tp_size,
                    self.num_attention_heads,
                    self.num_key_value_heads,
                    self.kv_size_multiplier,
                    "query",
                )
            )
            full_weight_o = (
                GQAQKVColumnParallelLinearSpec.create_gqa_query_or_output_projection_weight_from_full_weight(
                    full_weight_o,
                    self.tp_size,
                    self.num_attention_heads,
                    self.num_key_value_heads,
                    self.kv_size_multiplier,
                    "output",
                )
            )

            full_weight_k = torch.chunk(full_weight_k, self.kv_size_multiplier, dim=0)[0].detach().clone()
            full_weight_v = torch.chunk(full_weight_v, self.kv_size_multiplier, dim=0)[0].detach().clone()

            state_dict.update(
                {
                    f"{module_fully_qualified_name}.{self.query_projection_name}.{param_name}": full_weight_q,
                    f"{module_fully_qualified_name}.{self.key_projection_name}.{param_name}": full_weight_k,
                    f"{module_fully_qualified_name}.{self.value_projection_name}.{param_name}": full_weight_v,
                    f"{module_fully_qualified_name}.{self.output_projection_name}.{param_name}": full_weight_o,
                }
            )
        return state_dict, keys_to_remove


def set_module_names_in_transformation_specs(model: torch.nn.Module):
    for name, mod in model.named_modules():
        if not isinstance(mod, CustomModule):
            continue
        mod.specs.module_fully_qualified_name = name


def adapt_state_dict(
    model: torch.nn.Module, state_dict: Dict[str, torch.Tensor], inplace: bool = False
) -> Dict[str, torch.Tensor]:
    """
    Transforms the state dict from the original Transformers model to match the custom modeling implementation.
    """
    tp_size = get_tensor_model_parallel_size()

    named_parameters = dict(model.named_parameters())
    original_data_ptrs = {n: p.data_ptr() for n, p in state_dict.items()}
    original_state_dict_keys = set(state_dict.keys())
    for name, module in model.named_modules():
        if not isinstance(module, CustomModule):
            continue
        # If a submodule is a CustomModule, it has transformation specs and we use them to transform the associated
        # weights from the original state dict.
        state_dict = module.specs.adapt_state_dict(named_parameters, state_dict, inplace=inplace)

    # There are 2 cases:
    # 1. A new key was inserted by the adapt_state_dict function
    # 2. A key was mutated by the adapt_state_dict function
    new_keys = set(state_dict.keys()) - original_state_dict_keys
    mutated_keys = {n for n, p in state_dict.items() if p.data_ptr() != original_data_ptrs.get(n, p.data_ptr())}

    for name, param in model.named_parameters():
        if name in new_keys | mutated_keys:
            # In this case, we don't need to do anything, it was handled by the transformation specs.
            continue
        if hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel:
            if param.partition_dim not in [0, 1]:
                raise Exception(f"Partiton value of 0,1 are supported, found {param.partition_dim}.")

            # It means there are no weights in the state dict for the current parameter.
            if name not in state_dict:
                continue

            # If the parameter associated to the weight is parallel, we shard the weight.
            full_weight = state_dict[name]
            per_partition_size = full_weight.shape[param.partition_dim] // tp_size
            state_dict[name] = create_local_weight(
                full_weight, param.partition_dim, per_partition_size, param.partition_stride
            )
    return state_dict


def to_original_weights(
    transformations_specs: list[ModelWeightTransformationSpecs],
    sharded_state_dicts: Dict[str, list[torch.Tensor]],
    parameters_metadata: Dict[str, Dict[str, Any]],
) -> Dict[str, torch.Tensor]:
    """
    Consolidates the sharded state dicts produced by saving the custom model into a single state dict that matches the
    original Transformers model weights.
    """
    consolidated_state_dict = {}
    parameters_to_remove = set()
    for specs in transformations_specs:
        original_weights, keys_to_remove = specs.to_original_weights(sharded_state_dicts, parameters_metadata)
        consolidated_state_dict.update(original_weights)
        parameters_to_remove.update(keys_to_remove)

    for name, metadata in parameters_metadata.items():
        # It means it was already processed by the transformation specs.
        if name in consolidated_state_dict or name in parameters_to_remove:
            continue

        is_tensor_model_parallel = metadata["tensor_model_parallel"]
        if is_tensor_model_parallel:
            consolidated_weight = torch.cat(sharded_state_dicts[name], dim=metadata["partition_dim"])
            consolidated_state_dict[name] = consolidated_weight
        else:
            consolidated_state_dict[name] = sharded_state_dicts[name][0]
    return consolidated_state_dict


def get_tensor_model_parallel_attributes(tensor: torch.Tensor) -> Dict[str, Any]:
    """
    Returns the tensor model parallel attributes of a tensor.
    """
    if not hasattr(tensor, "tensor_model_parallel"):
        return {}
    return {
        "tensor_model_parallel": tensor.tensor_model_parallel,
        "partition_dim": tensor.partition_dim,
        "partition_stride": tensor.partition_stride,
        "num_partitions": tensor.num_partitions,
    }

def create_parameter_metadata(model) -> Dict[str, Dict[str, Any]]:
    """
    Creates the metadata to be saved with the model weights to be able to reconstruct the original weights when
    consolidating the sharded state dicts.
    """
    metadata = {"parameters": {}, "model_weight_transformation_specs": []}
    for name, param in model.named_parameters():
        tensor_model_parallel = getattr(param, "tensor_model_parallel", False)
        if tensor_model_parallel:
            metadata["parameters"][name] = get_tensor_model_parallel_attributes(param)
        else:
            metadata["parameters"][name] = {
                "tensor_model_parallel": tensor_model_parallel,
            }
    for name, module in model.named_modules():
        if not isinstance(module, CustomModule):
            continue
        serialized_specs = module.specs.to_metadata()
        metadata["model_weight_transformation_specs"].append(serialized_specs)
    return metadata
