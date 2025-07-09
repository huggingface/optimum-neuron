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
"""
Defines the API to represent model weights transformations that happen between the original Transformers
implementation and the custom implementation for Neuron.
"""

import copy
import re
import sys
from abc import abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Literal

import torch
from neuronx_distributed.parallel_layers.layers import create_local_weight
from neuronx_distributed.parallel_layers.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_size,
)

from optimum.utils import logging

from ...utils.import_utils import is_peft_available


if is_peft_available():
    from peft import PeftConfig
else:

    class PeftConfig:
        pass


logger = logging.get_logger()


LORA_PATTERN = re.compile(
    r"(?P<qualified_name>(\w+\.)+?lora_(embedding_)?(A|B))\.(?P<adapter_name>\w+)(?P<remaining>(\.{0,1}\w+\.?)+)?"
)


def create_local_weight_with_padding(
    full_weight: torch.Tensor,
    partition_dim: int,
    stride: int,
    out_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Shards a tensor along a given axis and return a slice corresponding to the rank.
    This will round up the layer to the next multiple if there is need to pad the tensor.
    """
    tp_size = get_tensor_model_parallel_size()
    axis_len = full_weight.shape[partition_dim]

    # Round up to the next multiple of tp_size
    split_len = (axis_len + tp_size - 1) // tp_size

    return create_local_weight(full_weight, partition_dim, split_len, stride, out_weight=out_weight)


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


@dataclass
class ModelWeightTransformationSpec:
    """
    This class defines the interface for transforming model weights between the original Transformers implementation
    and the custom implementation for Neuron.
    """

    @property
    def peft_type(self) -> str | None:
        if not hasattr(self, "_peft_type"):
            self._peft_type = None
        return self._peft_type

    @peft_type.setter
    def peft_type(self, value: str):
        self._peft_type = value

    @abstractmethod
    def get_relevant_parameter_names(self, module_fully_qualified_name: str) -> set[str]:
        """
        Returns the set of parameter names that this spec would affect.
        """
        pass

    @abstractmethod
    def guess_peft_type(self, model: torch.nn.Module, module_fully_qualified_name: str) -> str | None:
        """
        Guesses the PEFT type of the module associated to the spec.
        """
        pass

    @abstractmethod
    def adapt_peft_config(self, peft_config: PeftConfig, inplace: bool = False) -> PeftConfig:
        """
        Adapts the PEFT config to match the custom modeling implementation.
        """
        pass

    @abstractmethod
    def to_original_peft_config(self, peft_config: PeftConfig, inplace: bool = False) -> PeftConfig:
        """
        Restores the PEFT config to the original one that matches the original Transformers implementation.
        """
        pass

    @abstractmethod
    def _adapt_state_dict(
        self,
        module_fully_qualified_name: str,
        named_parameters: dict[str, torch.nn.Parameter],
        orig_state_dict: dict[str, torch.Tensor],
        upstanding_sharded_params: dict[str, torch.Tensor],
        inplace: bool = False,
    ) -> dict[str, torch.Tensor]:
        pass

    @abstractmethod
    def _lora_adapt_state_dict(
        self,
        module_fully_qualified_name: str,
        named_parameters: dict[str, torch.nn.Parameter],
        orig_state_dict: dict[str, torch.Tensor],
        upstanding_sharded_params: dict[str, torch.Tensor],
        inplace: bool = False,
    ) -> dict[str, torch.Tensor]:
        pass

    def adapt_state_dict(
        self,
        module_fully_qualified_name: str,
        named_parameters: dict[str, torch.nn.Parameter],
        orig_state_dict: dict[str, torch.Tensor],
        upstanding_sharded_params: dict[str, torch.Tensor],
        inplace: bool = False,
    ) -> dict[str, torch.Tensor]:
        """
        Transforms the state dict from the original Transformers model to match the custom modeling implementation.
        """
        if self.peft_type is None:
            return self._adapt_state_dict(
                module_fully_qualified_name,
                named_parameters,
                orig_state_dict,
                upstanding_sharded_params,
                inplace=inplace,
            )
        elif self.peft_type == "lora":
            return self._lora_adapt_state_dict(
                module_fully_qualified_name,
                named_parameters,
                orig_state_dict,
                upstanding_sharded_params,
                inplace=inplace,
            )
        else:
            raise NotImplementedError(
                f"PEFT type {self.peft_type} is not supported for the transformation spec {self.__class__.__name__}."
            )

    @abstractmethod
    def _to_original_weights(
        self,
        module_fully_qualified_name: str,
        sharded_state_dicts: dict[str, list[torch.Tensor]],
        parameters_metadata: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        pass

    @abstractmethod
    def _lora_to_original_weights(
        self,
        module_fully_qualified_name: str,
        sharded_state_dicts: dict[str, list[torch.Tensor]],
        parameters_metadata: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        pass

    def to_original_weights(
        self,
        module_fully_qualified_name: str,
        sharded_state_dicts: dict[str, list[torch.Tensor]],
        parameters_metadata: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        """
        Produces the weights associated to this transformation spec from the custom model to match the original
        Transformers weights.

        Args:
            sharded_state_dicts (dict[str, list[torch.Tensor]]): The sharded state dicts from the custom modeling
                implementation.
            parameters_metadata (dict[str, dict[str, Any]]): Metadata about the parameters in the original model.

        Returns:
            tuple[dict[str, torch.Tensor], list[str]]: A tuple containing the transformed weights and a list of the
            names of the parameters to remove from the final state dict.
        """
        if self.peft_type is None:
            return self._to_original_weights(module_fully_qualified_name, sharded_state_dicts, parameters_metadata)
        elif self.peft_type == "lora":
            return self._lora_to_original_weights(
                module_fully_qualified_name, sharded_state_dicts, parameters_metadata
            )
        else:
            raise NotImplementedError(
                f"PEFT type {self.peft_type} is not supported for the transformation spec {self.__class__.__name__}."
            )


@dataclass
class ModelWeightTransformationSpecs:
    """
    Defines a list of transformation specs for a given module of the model.
    """

    module_fully_qualified_name: str | None = None
    specs: ModelWeightTransformationSpec | list[ModelWeightTransformationSpec] = field(default_factory=list)

    def to_metadata(self, parameters_for_current_stage: set[str] | None = None) -> dict[str, Any]:
        if self.module_fully_qualified_name is None:
            raise ValueError("`module_fully_qualified_name` must be set to serialize the specs")
        serialized_specs = []
        for spec in self.specs:
            if parameters_for_current_stage is not None and not self.is_transformation_spec_relevant(
                spec, parameters_for_current_stage
            ):
                continue
            spec_data = asdict(spec)
            spec_data["peft_type"] = spec.peft_type
            serialized_specs.append((spec.__class__.__name__, spec_data))
        return {
            "module_fully_qualified_name": self.module_fully_qualified_name,
            "specs": serialized_specs,
        }

    @classmethod
    def from_metadata(cls, specs_metadata: dict[str, Any]):
        specs = cls(module_fully_qualified_name=specs_metadata["module_fully_qualified_name"])
        for spec_metadata in specs_metadata["specs"]:
            cls_name, metadata = spec_metadata
            # We dynamically import the class from the module.
            # We could use a dictionary as it is cleaner.
            cls_ = getattr(sys.modules[__name__], cls_name)
            peft_type = metadata.pop("peft_type")
            spec = cls_(**metadata)
            spec.peft_type = peft_type
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

    def is_transformation_spec_relevant(
        self, spec: ModelWeightTransformationSpec, parameters_for_current_stage: set[str]
    ) -> bool:
        if self.module_fully_qualified_name is None:
            raise ValueError("`module_fully_qualified_name` must be set to check relevance of the spec")
        relevant_param_names = spec.get_relevant_parameter_names(self.module_fully_qualified_name)
        return any(name in parameters_for_current_stage for name in relevant_param_names)

    def adapt_state_dict(
        self,
        named_parameters: dict[str, torch.nn.Parameter],
        orig_state_dict: dict[str, torch.Tensor],
        upstanding_sharded_params: dict[str, torch.Tensor],
        parameters_for_current_stage: set[str],
        inplace: bool = False,
    ):
        if self.module_fully_qualified_name is None:
            raise ValueError("`module_fully_qualified_name` must be set to adapt the state dict")
        for spec in self.specs:
            if not isinstance(spec, ModelWeightTransformationSpec):
                raise TypeError(f"spec must be of type ModelWeightTransformationSpec, but got {type(spec)}")

            if not self.is_transformation_spec_relevant(spec, parameters_for_current_stage):
                continue

            orig_state_dict = spec.adapt_state_dict(
                self.module_fully_qualified_name,
                named_parameters,
                orig_state_dict,
                upstanding_sharded_params=upstanding_sharded_params,
                inplace=inplace,
            )
        return orig_state_dict

    def to_original_weights(
        self, sharded_state_dicts: dict[str, list[torch.Tensor]], parameters_metadata: dict[str, dict[str, Any]]
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
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
    fuse_axis: Literal[0] | Literal[1] | Literal["column"] | Literal["row"]
    original_dims: list[int]
    tp_size: int = field(default_factory=get_tensor_model_parallel_size)

    def __post_init__(self):
        if self.fuse_axis == "column":
            self.fuse_axis = 0
        elif self.fuse_axis == "row":
            self.fuse_axis = 1

    def get_relevant_parameter_names(self, module_fully_qualified_name: str) -> set[str]:
        param_names = ["weight", "bias"] if self.bias else ["weight"]
        fused_names = {
            f"{module_fully_qualified_name}.{self.fused_linear_name}.{param_name}" for param_name in param_names
        }
        original_names = {
            f"{module_fully_qualified_name}.{linear_name}.{param_name}"
            for linear_name in self.linear_names
            for param_name in param_names
        }

        lora_param_names = set()
        for name in fused_names:
            lora_param_names.add(name.replace(self.fused_linear_name, f"{self.fused_linear_name}.lora_A"))
            lora_param_names.add(name.replace(self.fused_linear_name, f"{self.fused_linear_name}.lora_B"))
        for name in original_names:
            for linear_name in self.linear_names:
                lora_param_names.add(name.replace(linear_name, f"{linear_name}.lora_A"))
                lora_param_names.add(name.replace(linear_name, f"{linear_name}.lora_B"))

        return fused_names | original_names | lora_param_names

    def guess_peft_type(self, model: torch.nn.Module, module_fully_qualified_name: str) -> str | None:
        # Importing here to avoid circular imports
        from ...peft.tuners.lora.layer import ParallelLinear

        fused_linear_qualified_name = f"{module_fully_qualified_name}.{self.fused_linear_name}"
        fused_linear = model.get_submodule(fused_linear_qualified_name)
        if isinstance(fused_linear, ParallelLinear):
            return "lora"
        return None

    def adapt_peft_config(self, peft_config: PeftConfig, inplace: bool = False) -> PeftConfig:
        if not inplace:
            peft_config = copy.deepcopy(peft_config)
        if peft_config.peft_type == "LORA":
            target_modules = peft_config.target_modules
            at_least_one_linear_in_target_modules = any(name in target_modules for name in self.linear_names)
            all_linears_in_target_modules = (
                all(name in target_modules for name in self.linear_names) or target_modules == "all-linear"
            )
            if at_least_one_linear_in_target_modules and not all_linears_in_target_modules:
                missing_modules = [name for name in self.linear_names if name not in target_modules]
                raise ValueError(
                    "If you use FusedLinearsSpec, either all linear layers must be in the target modules of the PEFT "
                    f"config or none at all. The following linear layers are missing: {', '.join(missing_modules)}."
                )
            if all_linears_in_target_modules:
                for name in self.linear_names:
                    target_modules.remove(name)
                target_modules.add(self.fused_linear_name)
        else:
            raise NotImplementedError(
                f"PEFT type {peft_config.peft_type} is not supported for the transformation spec {self.__class__.__name__}."
            )
        return peft_config

    def to_original_peft_config(self, peft_config: PeftConfig, inplace: bool = False) -> PeftConfig:
        peft_config = copy.deepcopy(peft_config) if not inplace else peft_config
        if peft_config.peft_type == "LORA":
            target_modules = peft_config.target_modules
            if self.fused_linear_name in target_modules:
                target_modules.remove(self.fused_linear_name)
                for name in self.linear_names:
                    target_modules.add(name)
        else:
            raise NotImplementedError(
                f"PEFT type {peft_config.peft_type} is not supported for the transformation spec {self.__class__.__name__}."
            )
        return peft_config

    def _adapt_state_dict(
        self,
        module_fully_qualified_name: str,
        named_parameters: dict[str, torch.nn.Parameter],
        orig_state_dict: dict[str, torch.Tensor],
        upstanding_sharded_params: dict[str, torch.Tensor],
        inplace: bool = False,
    ) -> dict[str, torch.Tensor]:
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
            # Move the weights to the upstanding_sharded_params dict, so that if they are not found in the
            # state_dict, they will be added next time the function is called.
            for full_k in full_weight_names:
                if full_k in state_dict:
                    upstanding_sharded_params[full_k] = state_dict.pop(full_k)

            if len(full_weight_names) != len(upstanding_sharded_params):
                # This means state_dict is not fully loaded for this parameter, move on
                continue
            full_weights = [upstanding_sharded_params.pop(key) for key in full_weight_names]
            param = named_parameters[new_name]
            state_dict[new_name] = create_local_fused_weight(
                tp_rank, tp_size, full_weights, param.partition_dim, self.fuse_axis
            )
            if len(upstanding_sharded_params) != 0:
                raise ValueError(
                    f"It appears that several parameters have sharded weights, this is not supported: {upstanding_sharded_params.keys()}"
                )
        return state_dict

    def _lora_adapt_state_dict(
        self,
        module_fully_qualified_name: str,
        named_parameters: dict[str, torch.nn.Parameter],
        orig_state_dict: dict[str, torch.Tensor],
        upstanding_sharded_params: dict[str, torch.Tensor],
        inplace: bool = False,
    ) -> dict[str, torch.Tensor]:
        tp_size = get_tensor_model_parallel_size()
        tp_rank = get_tensor_model_parallel_rank()

        if inplace:
            state_dict = orig_state_dict
        else:
            state_dict = dict(orig_state_dict)

        # There are two cases for LoRA:
        # Case 1: the base layer is a ColumnParallelLinear
        #   1. "Mix" the LoRA A weights of the several linear layers into a single weight, maybe by doing the mean.
        #   2. Gather all the LoRA B weights of the linear layers
        #   3. Shard them across the tensor model parallel size if TP is enabled
        #   4. Fuse them along the fuse axis
        # Case 2: the base layer is a RowParallelLinear
        #   1. "Mix" the LoRA A weights of the several linear layers into a single weight, maybe by doing the mean.
        #   2. Shard them across the tensor model parallel size if TP is enabled
        #   3. Gather all the LoRA B weights of the linear layers
        #   4. Fuse them along the fuse axis
        fused_linear_fully_qualified_name = f"{module_fully_qualified_name}.{self.fused_linear_name}"
        param_names = ["weight", "bias"] if self.bias else ["weight"]
        for param_name in param_names:
            lora_A_weight_names = [
                f"{module_fully_qualified_name}.{name}.lora_A.{param_name}" for name in self.linear_names
            ]

            logger.warning("Taking the mean of the LoRA A weights since there is only one LoRA A weight after fusing.")
            lora_A_weight = torch.mean(
                torch.stack([state_dict.pop(name) for name in lora_A_weight_names], dim=0),
                dim=0,
            )
            lora_A_weight_name = f"{fused_linear_fully_qualified_name}.lora_A.{param_name}"

            # Case 1: the base layer is a ColumnParallelLinear
            if self.fuse_axis == 0:
                state_dict[lora_A_weight_name] = lora_A_weight
                lora_B_weight_names = [
                    f"{module_fully_qualified_name}.{name}.lora_B.{param_name}" for name in self.linear_names
                ]
                lora_B_weights = [state_dict.pop(name) for name in lora_B_weight_names]

                lora_B_fused_local_weight = create_local_fused_weight(
                    tp_rank, tp_size, lora_B_weights, 0, self.fuse_axis
                )
                lora_B_weight_name = f"{fused_linear_fully_qualified_name}.lora_B.{param_name}"
                state_dict[lora_B_weight_name] = lora_B_fused_local_weight

            # Case 2: the base layer is a RowParallelLinear
            else:
                lora_A_local_weight = create_local_weight_with_padding(lora_A_weight, 1, 1)
                state_dict[lora_A_weight_name] = lora_A_local_weight

                lora_B_weight_names = [
                    f"{module_fully_qualified_name}.{name}.lora_B.{param_name}" for name in self.linear_names
                ]
                lora_B_weights = [state_dict.pop(name) for name in lora_B_weight_names]

                lora_B_fused_weight = torch.cat(
                    lora_B_weights,
                    dim=self.fuse_axis,
                )
                local_B_weight_name = f"{fused_linear_fully_qualified_name}.lora_B.{param_name}"
                state_dict[local_B_weight_name] = lora_B_fused_weight

        return state_dict

    def _to_original_weights(
        self,
        module_fully_qualified_name: str,
        sharded_state_dicts: dict[str, list[torch.Tensor]],
        parameters_metadata: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        # To recreate original weights from the fused weights we need to:
        # 1. Unfuse the sharded weights
        # 2. Concat each unsharded local weight across the partion_dim if TP is enabled
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

    def _lora_to_original_weights(
        self,
        module_fully_qualified_name: str,
        sharded_state_dicts: dict[str, list[torch.Tensor]],
        parameters_metadata: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        # To recreate original weights from the fused weights there are two cases:
        #   - Case 1: the base layer is a ColumnParallelLinear
        #   Steps:
        #       1. Unfuse the fused weight for LoRA B
        #       2. Concat the local unfused weights for LoRA B accross the partition_dim if TP is enabled
        #       3. Duplicate the weight for LoRA A for each unfused linear
        #   - Case 2: the bae layer is a RowParallelLinear
        #   Steps:
        #       1. Concat the local weights for LoRA A accross the partition_dim if TP is enabled
        #       2. Unfuse the fused weight for LoRA B
        #       3. Duplicate the weight for LoRA A for each unfused linear
        original_weights = {}
        keys_to_remove = []
        for param_name in ["weight", "bias"] if self.bias else ["weight"]:
            unfused_local_weights = []
            lora_A_prefix = f"{module_fully_qualified_name}.{self.fused_linear_name}.lora_A"
            lora_B_prefix = f"{module_fully_qualified_name}.{self.fused_linear_name}.lora_B"

            # The base layer is a ColumnParallelLinear
            if self.fuse_axis == 0:
                # We get the names of the LoRA weights that are fused and sharded.
                # There should be only one adapter left in parameters_metadata because we filter for it in the main
                # `to_original_weights` function.
                weight_name = None
                to_duplicate_name = None
                for name in parameters_metadata:
                    if lora_B_prefix in name and name.endswith(param_name):
                        weight_name = name
                    if lora_A_prefix in name and name.endswith(param_name):
                        to_duplicate_name = name
                    if weight_name is not None and to_duplicate_name is not None:
                        break

                if weight_name is None or to_duplicate_name is None:
                    raise ValueError(
                        f"Could not find LoRA weights for {module_fully_qualified_name} with param name {param_name}."
                    )

                # When saved, the name of the adapter is removed in the weight qualified name since weights for each
                # adapter are saved separately.
                weight_name_without_adapter_name = remove_adapter_name(weight_name)
                fused_linear_sharded_weights = sharded_state_dicts[weight_name_without_adapter_name]
                for fused_local_weight in fused_linear_sharded_weights:
                    unfused_local_weights.append(
                        torch.split(
                            fused_local_weight,
                            [dim // self.tp_size for dim in self.original_dims],
                            dim=self.fuse_axis,
                        )
                    )
                for idx, linear_name in enumerate(self.linear_names):
                    original_weight_name = weight_name_without_adapter_name.replace(
                        self.fused_linear_name, linear_name
                    )
                    partition_dim = parameters_metadata[weight_name]["partition_dim"]
                    original_weight = torch.cat(
                        [unfused_local_weights[tp_rank][idx] for tp_rank in range(len(unfused_local_weights))],
                        dim=partition_dim,
                    )
                    original_weights[original_weight_name] = original_weight

                keys_to_remove.append(weight_name)

                # We duplicate LoRA A weight for each unfused linear
                name_without_adapter_name = remove_adapter_name(to_duplicate_name)
                weight = sharded_state_dicts[name_without_adapter_name][0]
                for linear_name in self.linear_names:
                    original_name = name_without_adapter_name.replace(self.fused_linear_name, linear_name)
                    original_weights[original_name] = weight.clone()

                keys_to_remove.append(to_duplicate_name)

            # Otherwise the base layer is a RowParallelLinear
            else:
                to_concat_and_duplicate_name = None
                to_unfuse_name = None
                for name in parameters_metadata:
                    if lora_A_prefix in name and name.endswith(param_name):
                        to_concat_and_duplicate_name = name
                    if lora_B_prefix in name and name.endswith(param_name):
                        to_duplicate_name = name
                    if to_concat_and_duplicate_name is not None and to_unfuse_name is not None:
                        break
                if to_concat_and_duplicate_name is None or to_unfuse_name is None:
                    raise ValueError(
                        f"Could not find LoRA weights for {module_fully_qualified_name} with param name {param_name}."
                    )

                weight_name_without_adapter_name = remove_adapter_name(to_concat_and_duplicate_name)
                linear_sharded_weights = sharded_state_dicts[weight_name_without_adapter_name]
                partition_dim = parameters_metadata[to_concat_and_duplicate_name]["partition_dim"]
                linear_weight = torch.cat(linear_sharded_weights, dim=partition_dim)
                for linear_name in self.linear_names:
                    original_weight_name = weight_name_without_adapter_name.replace(
                        self.fused_linear_name, linear_name
                    )
                    original_weights[original_weight_name] = linear_weight.clone()

                keys_to_remove.append(to_concat_and_duplicate_name)

                weight_name_without_adapter_name = remove_adapter_name(to_unfuse_name)
                fused_linear_weight = sharded_state_dicts[weight_name_without_adapter_name][0]
                unfused_linear_weights = torch.split(
                    fused_linear_weight, self.original_dims[0] // self.tp_size, dim=self.fuse_axis
                )
                for idx, linear_name in enumerate(self.linear_names):
                    original_weight_name = weight_name_without_adapter_name.replace(
                        self.fused_linear_name, linear_name
                    )
                    original_weights[original_weight_name] = unfused_linear_weights[idx]

                keys_to_remove.append(to_unfuse_name)

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

    def get_relevant_parameter_names(self, module_fully_qualified_name: str) -> set[str]:
        param_names = ["weight", "bias"] if self.bias else ["weight"]
        original_proj_names = [
            self.query_projection_name,
            self.key_projection_name,
            self.value_projection_name,
            self.output_projection_name,
        ]
        original_names = {
            f"{module_fully_qualified_name}.{proj_name}.{param_name}"
            for proj_name in original_proj_names
            for param_name in param_names
        }

        if self.fuse_qkv:
            gqa_names = {
                f"{module_fully_qualified_name}.{self.gqa_qkv_projection_name}.{param_name}_qkv"
                for param_name in param_names
            }
        else:
            gqa_names = {
                f"{module_fully_qualified_name}.{self.gqa_qkv_projection_name}.{param_name}_{suffix}"
                for param_name in param_names
                for suffix in ["q", "k", "v"]
            }

        lora_param_names = set()
        # LoRA for original layers
        for param_name in param_names:
            for proj_name in original_proj_names:
                lora_param_names.add(f"{module_fully_qualified_name}.{proj_name}.lora_A.{param_name}")
                lora_param_names.add(f"{module_fully_qualified_name}.{proj_name}.lora_B.{param_name}")

        # LoRA for GQA layer
        for param_name in param_names:
            lora_param_names.add(f"{module_fully_qualified_name}.{self.gqa_qkv_projection_name}.lora_A.{param_name}")
            if self.fuse_qkv:
                lora_param_names.add(
                    f"{module_fully_qualified_name}.{self.gqa_qkv_projection_name}.lora_B.{param_name}"
                )
            else:
                lora_param_names.add(
                    f"{module_fully_qualified_name}.{self.gqa_qkv_projection_name}.lora_B.{param_name}_q"
                )
                lora_param_names.add(
                    f"{module_fully_qualified_name}.{self.gqa_qkv_projection_name}.lora_B.{param_name}_k"
                )
                lora_param_names.add(
                    f"{module_fully_qualified_name}.{self.gqa_qkv_projection_name}.lora_B.{param_name}_v"
                )
        return original_names | gqa_names | lora_param_names

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
        query_or_output_proj: Literal["query"] | Literal["output"],
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
        query_or_output: Literal["query", Literal["output"]],
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

    def guess_peft_type(self, model: torch.nn.Module, module_fully_qualified_name: str) -> str | None:
        # Importing here to avoid circular imports
        from ...peft.tuners.lora.layer import GQAQKVColumnParallelLinear

        gqa_qkv_projection_qualified_name = f"{module_fully_qualified_name}.{self.gqa_qkv_projection_name}"
        qkv_linear = model.get_submodule(gqa_qkv_projection_qualified_name)
        if isinstance(qkv_linear, GQAQKVColumnParallelLinear):
            return "lora"
        return None

    def adapt_peft_config(self, peft_config: PeftConfig, inplace: bool = False) -> PeftConfig:
        if not inplace:
            peft_config = copy.deepcopy(peft_config)
        if peft_config.peft_type == "LORA":
            linear_names = [self.query_projection_name, self.key_projection_name, self.value_projection_name]
            target_modules = peft_config.target_modules
            at_least_one_linear_in_target_modules = any(name in target_modules for name in linear_names)
            all_linears_in_target_modules = (
                all(name in target_modules for name in linear_names) or target_modules == "all-linear"
            )
            if at_least_one_linear_in_target_modules and not all_linears_in_target_modules:
                missing_modules = [name for name in linear_names if name not in target_modules]
                raise ValueError(
                    "If you use GQAQKVColumnParallelLinearSpec, either all linear layers must be in the target modules "
                    "of the PEFT config or none at all. The following linear layers are missing: "
                    f"{', '.join(missing_modules)}."
                )
            if all_linears_in_target_modules:
                for name in linear_names:
                    target_modules.remove(name)
                target_modules.add(self.gqa_qkv_projection_name)
        else:
            raise NotImplementedError(
                f"PEFT type {peft_config.peft_type} is not supported for the transformation spec {self.__class__.__name__}."
            )
        return peft_config

    def to_original_peft_config(self, peft_config: PeftConfig, inplace: bool = False) -> PeftConfig:
        peft_config = copy.deepcopy(peft_config) if not inplace else peft_config
        if peft_config.peft_type == "LORA":
            target_modules = peft_config.target_modules
            if self.gqa_qkv_projection_name in target_modules:
                target_modules.remove(self.gqa_qkv_projection_name)
                target_modules.add(self.query_projection_name)
                target_modules.add(self.key_projection_name)
                target_modules.add(self.value_projection_name)
        else:
            raise NotImplementedError(
                f"PEFT type {peft_config.peft_type} is not supported for the transformation spec {self.__class__.__name__}."
            )
        return peft_config

    def _adapt_state_dict(
        self,
        module_fully_qualified_name: str,
        named_parameters: dict[str, torch.nn.Parameter],
        orig_state_dict: dict[str, torch.Tensor],
        upstanding_sharded_params: dict[str, torch.Tensor],
        inplace: bool = False,
    ) -> dict[str, torch.Tensor]:
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
                        state_dict.pop(q_name),
                        self.num_attention_heads,
                        self.num_key_value_heads,
                        self.kv_size_multiplier,
                        "query",
                    )
                )

                state_dict[new_name_weight_k] = (
                    GQAQKVColumnParallelLinearSpec.create_kv_proj_local_weight_from_regular_weight(
                        state_dict.pop(k_name), self.kv_size_multiplier, self.kv_output_size_per_partition
                    )
                )

                state_dict[new_name_weight_v] = (
                    GQAQKVColumnParallelLinearSpec.create_kv_proj_local_weight_from_regular_weight(
                        state_dict.pop(v_name), self.kv_size_multiplier, self.kv_output_size_per_partition
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

    def _lora_adapt_state_dict(
        self,
        module_fully_qualified_name: str,
        named_parameters: dict[str, torch.nn.Parameter],
        orig_state_dict: dict[str, torch.Tensor],
        upstanding_sharded_params: dict[str, torch.Tensor],
        inplace: bool = False,
    ) -> dict[str, torch.Tensor]:
        if inplace:
            state_dict = orig_state_dict
        else:
            state_dict = dict(orig_state_dict)

        # To adapt the state dict for LoRA, we need to:
        #   1. "Mix" the LoRA A weights of the query, key and value projections
        #   2. Gather all the LoRA B weights of the query, key and value projections
        #   3. Create the local version of the weights for the given TP rank
        #   4. Fuse them if `fuse_qkv` is True
        #   5. If there is LoRA weights for the output projection, we need to adapt it as well.
        #       - The LoRA A weight needs to be processed
        #       - The LoRA B weight stays as it is
        param_names = ["weight", "bias"] if self.bias else ["weight"]
        for param_name in param_names:
            lora_A_q_name = f"{module_fully_qualified_name}.{self.query_projection_name}.lora_A.{param_name}"
            lora_A_k_name = f"{module_fully_qualified_name}.{self.key_projection_name}.lora_A.{param_name}"
            lora_A_v_name = f"{module_fully_qualified_name}.{self.value_projection_name}.lora_A.{param_name}"
            lora_B_q_name = f"{module_fully_qualified_name}.{self.query_projection_name}.lora_B.{param_name}"
            lora_B_k_name = f"{module_fully_qualified_name}.{self.key_projection_name}.lora_B.{param_name}"
            lora_B_v_name = f"{module_fully_qualified_name}.{self.value_projection_name}.lora_B.{param_name}"

            lora_A_weight_names = [lora_A_q_name, lora_A_k_name, lora_A_v_name]

            logger.warning("Taking the mean of the LoRA A weights since there is only one LoRA A weight after fusing.")
            lora_A_weight = torch.mean(
                torch.stack([state_dict.pop(name) for name in lora_A_weight_names], dim=0),
                dim=0,
            )
            lora_A_weight_name = f"{module_fully_qualified_name}.{self.gqa_qkv_projection_name}.lora_A.{param_name}"
            state_dict[lora_A_weight_name] = lora_A_weight

            if self.fuse_qkv:
                lora_B_weight_name = (
                    f"{module_fully_qualified_name}.{self.gqa_qkv_projection_name}.lora_B.{param_name}"
                )

                full_weights = [
                    GQAQKVColumnParallelLinearSpec.create_query_or_output_projection_local_weight_from_regular_weight(
                        state_dict.pop(lora_B_q_name),
                        self.num_attention_heads,
                        self.num_key_value_heads,
                        self.kv_size_multiplier,
                        "query",
                    ),
                    GQAQKVColumnParallelLinearSpec.create_kv_proj_local_weight_from_regular_weight(
                        state_dict.pop(lora_B_k_name),
                        self.kv_size_multiplier,
                        self.kv_output_size_per_partition,
                    ),
                    GQAQKVColumnParallelLinearSpec.create_kv_proj_local_weight_from_regular_weight(
                        state_dict.pop(lora_B_v_name),
                        self.kv_size_multiplier,
                        self.kv_output_size_per_partition,
                    ),
                ]
                state_dict[lora_B_weight_name] = torch.cat(full_weights, dim=0)
            else:
                new_lora_B_weight_q_name = (
                    f"{module_fully_qualified_name}.{self.gqa_qkv_projection_name}.lora_B.{param_name}_q"
                )
                new_lora_B_weight_k_name = (
                    f"{module_fully_qualified_name}.{self.gqa_qkv_projection_name}.lora_B.{param_name}_k"
                )
                new_lora_B_weight_v_name = (
                    f"{module_fully_qualified_name}.{self.gqa_qkv_projection_name}.lora_B.{param_name}_v"
                )

                state_dict[new_lora_B_weight_q_name] = (
                    GQAQKVColumnParallelLinearSpec.create_query_or_output_projection_local_weight_from_regular_weight(
                        state_dict.pop(lora_B_q_name),
                        self.num_attention_heads,
                        self.num_key_value_heads,
                        self.kv_size_multiplier,
                        "query",
                    )
                )

                state_dict[new_lora_B_weight_k_name] = (
                    GQAQKVColumnParallelLinearSpec.create_kv_proj_local_weight_from_regular_weight(
                        state_dict.pop(lora_B_k_name), self.kv_size_multiplier, self.kv_output_size_per_partition
                    )
                )

                state_dict[new_lora_B_weight_v_name] = (
                    GQAQKVColumnParallelLinearSpec.create_kv_proj_local_weight_from_regular_weight(
                        state_dict.pop(lora_B_v_name), self.kv_size_multiplier, self.kv_output_size_per_partition
                    )
                )

            # If there are LoRA weights for the output projection, we need to adapt them as well
            lora_A_output_projection_name = (
                f"{module_fully_qualified_name}.{self.output_projection_name}.lora_A.{param_name}"
            )
            if lora_A_output_projection_name in state_dict:
                state_dict[lora_A_output_projection_name] = (
                    GQAQKVColumnParallelLinearSpec.create_query_or_output_projection_local_weight_from_regular_weight(
                        state_dict[lora_A_output_projection_name],
                        self.num_attention_heads,
                        self.num_key_value_heads,
                        self.kv_size_multiplier,
                        "output",
                    )
                )

        return state_dict

    def _to_original_weights(
        self,
        module_fully_qualified_name: str,
        sharded_state_dicts: dict[str, list[torch.Tensor]],
        parameters_metadata: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
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

                # TODO: fix once it is fixed in neuronx_distributed
                # We should to as follows:
                # qkv_partition_dim = parameters_metadata[fuse_qkv_weight_name]["partition_dim"]
                # But it seems not tensor model attributes are set to `weight_qkv`.
                qkv_partition_dim = 0  # Since it is a ColumnParallelLinear, the partition dim is always 0.
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

    def _lora_to_original_weights(
        self,
        module_fully_qualified_name: str,
        sharded_state_dicts: dict[str, list[torch.Tensor]],
        parameters_metadata: dict[str, dict[str, Any]],
    ) -> tuple[dict[str, torch.Tensor], list[str]]:
        state_dict = {}
        keys_to_remove = []
        param_names = ["weight", "bias"] if self.bias else ["weight"]
        for param_name in param_names:
            lora_A_prefix = f"{module_fully_qualified_name}.{self.gqa_qkv_projection_name}.lora_A"
            lora_B_prefix = f"{module_fully_qualified_name}.{self.gqa_qkv_projection_name}.lora_B"
            lora_B_qkv_seperate_weights = []
            if self.fuse_qkv:
                weight_suffix = f"{param_name}_qkv"
                lora_B_weight_name = None
                for name in parameters_metadata:
                    if lora_B_prefix in name and name.endswith(weight_suffix):
                        lora_B_weight_name = name
                        break

                if lora_B_weight_name is not None:
                    weight_name_without_adapter_name = remove_adapter_name(lora_B_weight_name)
                    fused_qkv_local_weights = sharded_state_dicts[weight_name_without_adapter_name]

                    slice_q = slice(0, self.q_output_size_per_partition)
                    weights_q = [
                        fused_qkv_local_weights[tp_rank][slice_q].contiguous() for tp_rank in range(self.tp_size)
                    ]

                    slice_k = slice(
                        self.q_output_size_per_partition,
                        self.q_output_size_per_partition + self.kv_output_size_per_partition,
                    )
                    weights_k = [
                        fused_qkv_local_weights[tp_rank][slice_k].contiguous() for tp_rank in range(self.tp_size)
                    ]

                    slice_v = slice(
                        self.q_output_size_per_partition + self.kv_output_size_per_partition,
                        None,
                    )
                    weights_v = [
                        fused_qkv_local_weights[tp_rank][slice_v].contiguous() for tp_rank in range(self.tp_size)
                    ]

                    # TODO: fix once it is fixed in neuronx_distributed
                    # We should to as follows:
                    # qkv_partition_dim = parameters_metadata[lora_B_weight_name]["partition_dim"]
                    # But it seems not tensor model attributes are set to `weight_qkv`.
                    qkv_partition_dim = 0  # Since it is a ColumnParallelLinear, the partition dim is always 0.
                    keys_to_remove += [lora_B_weight_name]

                    lora_B_qkv_seperate_weights += [
                        ["query", weights_q, qkv_partition_dim, weight_name_without_adapter_name],
                        ["key", weights_k, qkv_partition_dim, weight_name_without_adapter_name],
                        ["value", weights_v, qkv_partition_dim, weight_name_without_adapter_name],
                    ]
            else:
                weight_suffixes = [f"{param_name}_q", f"{param_name}_k", f"{param_name}_v"]

                lora_B_weight_names = []
                for name in parameters_metadata:
                    if name.startswith(lora_B_prefix) and any(name.endswith(suffix) for suffix in weight_suffixes):
                        lora_B_weight_names.append(name)

                for weight_name in lora_B_weight_names:
                    weight_name_without_adapter_name = remove_adapter_name(weight_name)

                    weights = sharded_state_dicts[weight_name_without_adapter_name]

                    if f"{param_name}_q" in weight_name_without_adapter_name:
                        query_key_or_value = "query"
                    elif f"{param_name}_k" in weight_name_without_adapter_name:
                        query_key_or_value = "key"
                    else:
                        query_key_or_value = "value"

                    # The query, key and value share the same partition dim.
                    qkv_partition_dim = parameters_metadata[weight_name]["partition_dim"]
                    keys_to_remove += [weight_name]

                    lora_B_qkv_seperate_weights += [
                        [query_key_or_value, weights, qkv_partition_dim, weight_name_without_adapter_name],
                    ]

            # First we handle LoRA A weights.
            # We need to duplicate the LoRA A weight for the query, key and value projections.
            lora_A_weight_name = None
            for name in parameters_metadata:
                if lora_A_prefix in name and name.endswith(param_name):
                    lora_A_weight_name = name
                    break

            if lora_A_weight_name is not None:
                lora_A_weight_name_without_adapter = remove_adapter_name(lora_A_weight_name)
                query_weight_name = lora_A_weight_name_without_adapter.replace(
                    self.gqa_qkv_projection_name, self.query_projection_name
                )
                key_weight_name = lora_A_weight_name_without_adapter.replace(
                    self.gqa_qkv_projection_name, self.key_projection_name
                )
                value_weight_name = lora_A_weight_name_without_adapter.replace(
                    self.gqa_qkv_projection_name, self.value_projection_name
                )
                state_dict[query_weight_name] = sharded_state_dicts[lora_A_weight_name_without_adapter][0].clone()
                state_dict[key_weight_name] = sharded_state_dicts[lora_A_weight_name_without_adapter][0].clone()
                state_dict[value_weight_name] = sharded_state_dicts[lora_A_weight_name_without_adapter][0].clone()
                keys_to_remove.append(lora_A_weight_name)

            # Then we handle LoRA B weights.
            # There is a little bit more work, it mostly consists in doing the same as for the
            # GQAQKVColumnParallelLinear weights.
            for query_key_or_value, weights, qkv_partition_dim, weight_name in lora_B_qkv_seperate_weights:
                if query_key_or_value == "query":
                    full_weight_q = torch.cat(weights, dim=qkv_partition_dim).contiguous()
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
                    query_weight_name = (
                        weight_name.replace(f"{param_name}_qkv", param_name)
                        .replace(f"{param_name}_q", param_name)
                        .replace(self.gqa_qkv_projection_name, self.query_projection_name)
                    )
                    state_dict[query_weight_name] = full_weight_q
                elif query_key_or_value == "key":
                    full_weight_k = torch.cat(weights, dim=qkv_partition_dim).contiguous()
                    full_weight_k = torch.chunk(full_weight_k, self.kv_size_multiplier, dim=0)[0].detach().clone()
                    key_weight_name = (
                        weight_name.replace(f"{param_name}_qkv", param_name)
                        .replace(f"{param_name}_k", param_name)
                        .replace(self.gqa_qkv_projection_name, self.key_projection_name)
                    )
                    state_dict[key_weight_name] = full_weight_k
                else:
                    full_weight_v = torch.cat(weights, dim=qkv_partition_dim).contiguous()
                    full_weight_v = torch.chunk(full_weight_v, self.kv_size_multiplier, dim=0)[0].detach().clone()
                    value_weight_name = (
                        weight_name.replace(f"{param_name}_qkv", param_name)
                        .replace(f"{param_name}_v", param_name)
                        .replace(self.gqa_qkv_projection_name, self.value_projection_name)
                    )
                    state_dict[value_weight_name] = full_weight_v

            # Now we handle the output projection.
            # There is only work for the LoRA A weights, the LoRA B weights are already regular linear weights.
            lora_A_prefix = f"{module_fully_qualified_name}.{self.output_projection_name}.lora_A"
            lora_A_weight_name = None
            for name in parameters_metadata:
                if name.startswith(lora_A_prefix) and name.endswith(param_name):
                    lora_A_weight_name = name
                    break
            if lora_A_weight_name is not None:
                weight_name_without_adapter_name = remove_adapter_name(lora_A_weight_name)
                weights_o = sharded_state_dicts[weight_name_without_adapter_name]
                o_partition_dim = parameters_metadata[lora_A_weight_name]["partition_dim"]
                full_weight_o = torch.cat(weights_o, dim=o_partition_dim).contiguous()
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
                keys_to_remove.append(lora_A_weight_name)
                state_dict[weight_name_without_adapter_name] = full_weight_o

        return state_dict, keys_to_remove


def specialize_transformation_specs_for_model(model: torch.nn.Module):
    for name, mod in model.named_modules():
        if not isinstance(mod, CustomModule):
            continue
        mod.specs.module_fully_qualified_name = name
        for spec in mod.specs:
            spec.peft_type = spec.guess_peft_type(model, name)


def adapt_peft_config_for_model(
    model: torch.nn.Module, peft_config: PeftConfig | dict[str, PeftConfig], inplace: bool = False
) -> PeftConfig | dict[str, PeftConfig]:
    adapted_peft_config = copy.deepcopy(peft_config) if not inplace else peft_config
    for _, mod in model.named_modules():
        if not isinstance(mod, CustomModule):
            continue
        for spec in mod.specs:
            # inplace=True because we already do a deepcopy if needed once at the beginning of this function.
            if isinstance(adapted_peft_config, dict):
                for _, config in adapted_peft_config.items():
                    spec.adapt_peft_config(config, inplace=True)
            else:
                spec.adapt_peft_config(adapted_peft_config, inplace=True)
    return adapted_peft_config


def to_original_peft_config_for_model(
    model: torch.nn.Module, peft_config: PeftConfig, inplace: bool = False
) -> PeftConfig:
    adapted_peft_config = copy.deepcopy(peft_config) if not inplace else peft_config
    for _, mod in model.named_modules():
        if not isinstance(mod, CustomModule):
            continue
        for spec in mod.specs:
            # inplace=True because we already do a deepcopy if needed once at the beginning of this function.
            if isinstance(adapted_peft_config, dict):
                for _, config in adapted_peft_config.items():
                    spec.to_original_peft_config(config, inplace=True)
            else:
                spec.to_original_peft_config(adapted_peft_config, inplace=True)
    return adapted_peft_config


def remove_adapter_name(name: str) -> str:
    return re.sub(LORA_PATTERN, r"\g<qualified_name>\g<remaining>", name)


def is_base_layer(name: str) -> bool:
    return "base_layer" in name


def get_adapter_name(parameter_fully_qualified_name: str) -> str | None:
    match = re.match(LORA_PATTERN, parameter_fully_qualified_name)
    if match:
        return match.group("adapter_name")
    return None


def adapt_state_dict(
    model: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    upstanding_sharded_params: dict[str, torch.Tensor],
    inplace: bool = False,
    **peft_kwargs: Any,
) -> dict[str, torch.Tensor]:
    """
    Transforms the state dict from the original Transformers model to match the custom modeling implementation.
    """
    named_parameters = dict(model.named_parameters())
    named_parameters = {n: p for n, p in named_parameters.items() if n in model.parameters_for_current_stage}

    original_data_ptrs = {n: p.data_ptr() for n, p in state_dict.items()}
    original_state_dict_keys = set(state_dict.keys())
    parameters_for_current_stage_without_adapter_name = {
        remove_adapter_name(name) for name in model.parameters_for_current_stage
    }
    for name, module in model.named_modules():
        if not isinstance(module, CustomModule):
            continue
        # If a submodule is a CustomModule, it has transformation specs and we use them to transform the associated
        # weights from the original state dict.
        state_dict = module.specs.adapt_state_dict(
            named_parameters,
            state_dict,
            upstanding_sharded_params=upstanding_sharded_params,
            parameters_for_current_stage=parameters_for_current_stage_without_adapter_name,
            inplace=inplace,
        )

    # There are 2 cases:
    # 1. A new key was inserted by the adapt_state_dict function
    # 2. A key was mutated by the adapt_state_dict function
    new_keys = set(state_dict.keys()) - original_state_dict_keys
    mutated_keys = {n for n, p in state_dict.items() if p.data_ptr() != original_data_ptrs.get(n, p.data_ptr())}

    adapter_name = peft_kwargs.pop("adapter_name", None)

    for name, param in model.named_parameters():
        name_without_adapter_name = remove_adapter_name(name)
        if name not in model.parameters_for_current_stage:
            continue
        if is_base_layer(name_without_adapter_name):
            # In this case, it was already handled when loading the base layer weights in
            # `NeuronModelMixin.from_pretrained`.
            state_dict.pop(
                name_without_adapter_name, None
            )  # We remove it to avoid confusion, maybe it is actually needed?
            continue
        if adapter_name is not None and adapter_name not in name:
            # If the parameter is not associated to the current adapter, we skip it.
            continue
        if name_without_adapter_name in new_keys | mutated_keys:
            # In this case, we don't need to do anything, it was handled by the transformation specs.
            continue

        if hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel:
            if param.partition_dim not in [0, 1]:
                raise Exception(f"Partiton value of 0,1 are supported, found {param.partition_dim}.")

            # It means there are no weights in the state dict for the current parameter.
            if name_without_adapter_name not in state_dict:
                continue

            # If the parameter associated to the weight is parallel, we shard the weight.
            full_weight = state_dict.pop(name_without_adapter_name)
            state_dict[name_without_adapter_name] = create_local_weight_with_padding(
                full_weight, param.partition_dim, param.partition_stride
            )
    return state_dict


def to_original_weights(
    transformations_specs: list[ModelWeightTransformationSpecs],
    sharded_state_dicts: dict[str, list[torch.Tensor]],
    parameters_metadata: dict[str, dict[str, Any]],
    **peft_kwargs: Any,
) -> dict[str, torch.Tensor]:
    """
    Consolidates the sharded state dicts produced by saving the custom model into a single state dict that matches the
    original Transformers model weights.
    """
    consolidated_state_dict = {}
    parameters_to_remove = set()

    adapter_name = peft_kwargs.pop("adapter_name", None)
    if adapter_name is not None:

        def should_keep_parameter(name: str) -> bool:
            return is_base_layer(name) or get_adapter_name(name) == adapter_name

        parameters_metadata = {
            name: metadata for name, metadata in parameters_metadata.items() if should_keep_parameter(name)
        }

    for specs in transformations_specs:
        original_weights, keys_to_remove = specs.to_original_weights(sharded_state_dicts, parameters_metadata)
        consolidated_state_dict.update(original_weights)
        parameters_to_remove.update(keys_to_remove)

    for name, metadata in parameters_metadata.items():
        name_without_adapter_name = remove_adapter_name(name)

        # It means it was already processed by the transformation specs.
        if name_without_adapter_name in consolidated_state_dict:
            continue

        # `parameters_to_remove` contains the names with the adapter name so we need to use the full name to check.
        if name in parameters_to_remove:
            continue

        # It means that it was a parameter of the model but not saved. It can be the case when this parameter did not
        # required gradient computation.
        if name_without_adapter_name not in sharded_state_dicts:
            continue

        is_tensor_model_parallel = metadata["tensor_model_parallel"]
        if is_tensor_model_parallel:
            consolidated_weight = torch.cat(
                sharded_state_dicts[name_without_adapter_name], dim=metadata["partition_dim"]
            )
            consolidated_state_dict[name_without_adapter_name] = consolidated_weight
        else:
            consolidated_state_dict[name_without_adapter_name] = sharded_state_dicts[name_without_adapter_name][0]
    return consolidated_state_dict


def get_tensor_model_parallel_attributes(tensor: torch.Tensor) -> dict[str, Any]:
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


def create_parameter_metadata(model) -> dict[str, dict[str, Any]]:
    """
    Creates the metadata to be saved with the model weights to be able to reconstruct the original weights when
    consolidating the sharded state dicts.
    """
    metadata = {"parameters": {}, "model_weight_transformation_specs": []}
    for name, param in model.named_parameters():
        if name not in model.parameters_for_current_stage:
            continue
        tensor_model_parallel = getattr(param, "tensor_model_parallel", False)
        if tensor_model_parallel:
            metadata["parameters"][name] = get_tensor_model_parallel_attributes(param)
        else:
            metadata["parameters"][name] = {
                "tensor_model_parallel": False,
            }
    for name, module in model.named_modules():
        if not isinstance(module, CustomModule):
            continue
        parameters_for_current_stage_without_adapter_name = {
            remove_adapter_name(n) for n in model.parameters_for_current_stage
        }
        serialized_specs = module.specs.to_metadata(
            parameters_for_current_stage=parameters_for_current_stage_without_adapter_name
        )
        metadata["model_weight_transformation_specs"].append(serialized_specs)

    return metadata
