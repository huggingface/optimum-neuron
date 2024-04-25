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
"""Utilities of various sorts related to accelerate with Neuron."""

from typing import TYPE_CHECKING, Dict, Union

import torch
from transformers.modeling_utils import get_parameter_dtype

from ...distributed.utils import named_parameters
from ...utils import is_torch_neuronx_available, is_torch_xla_available, patch_everywhere
from ...utils.require_utils import requires_neuronx_distributed


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    if is_torch_neuronx_available():
        from neuronx_distributed.pipeline import NxDPPModel


def is_tpu_available(check_device=True):
    """
    Fake `is_tpu_available` that returns `is_torch_xla_available` to patch `accelerate`.
    """
    return is_torch_xla_available()


def patch_accelerate_is_tpu_available():
    patch_everywhere("is_tpu_available", is_tpu_available, module_name_prefix="accelerate")


_ORIG_TORCH_FINFO = torch.finfo


def create_patched_finfo(xla_downcast_bf16: bool = False, use_amp: bool = False, xla_use_bf16: bool = False):
    def patched_finfo(dtype):
        if xla_downcast_bf16 or use_amp or xla_use_bf16:
            return _ORIG_TORCH_FINFO(torch.bfloat16)
        return _ORIG_TORCH_FINFO(dtype)

    return patched_finfo


def create_patched_get_parameter_dtype(
    xla_downcast_bf16: bool = False, use_amp: bool = False, xla_use_bf16: bool = False
):
    def patched_get_parameter_dtype(module):
        dtype = get_parameter_dtype(module)
        if xla_downcast_bf16 or use_amp or xla_use_bf16:
            return torch.bfloat16
        return dtype

    return patched_get_parameter_dtype


@requires_neuronx_distributed
def get_tied_parameters_dict(model: Union["torch.nn.Module", "NxDPPModel"]) -> Dict[str, str]:
    from neuronx_distributed.pipeline import NxDPPModel

    unique_parameters = {}
    tied_parameters = {}
    if isinstance(model, NxDPPModel):
        module = model.local_module
    else:
        module = model
    for name, param in named_parameters(module, remove_duplicate=False):
        if param in unique_parameters:
            tied_parameter_name = unique_parameters[param]
            tied_parameters[name] = tied_parameter_name
        else:
            unique_parameters[param] = name
    return tied_parameters


@requires_neuronx_distributed
def tie_parameters(model: Union["torch.nn.Module", "NxDPPModel"], tied_parameters_dict: Dict[str, str]):
    from neuronx_distributed.pipeline import NxDPPModel

    if isinstance(model, NxDPPModel):
        module = model.local_module
    else:
        module = model

    for param_to_tie_name, param_name in tied_parameters_dict.items():
        param_to_tie_name = param_to_tie_name.rsplit(".", maxsplit=1)

        param_to_tie_parent_module = (
            module if len(param_to_tie_name) == 1 else module.get_submodule(param_to_tie_name[0])
        )
        param_to_tie = getattr(param_to_tie_parent_module, param_to_tie_name[1])

        param_name = param_name.rsplit(".", maxsplit=1)
        parent_module = module if len(param_name) == 1 else module.get_submodule(param_name[0])
        param = getattr(parent_module, param_name[1])

        if param_to_tie is not param:
            del param_to_tie
            setattr(param_to_tie_parent_module, param_to_tie_name[1], param)


@requires_neuronx_distributed
def apply_activation_checkpointing(model: Union["PreTrainedModel", "NxDPPModel"]):
    from neuronx_distributed.pipeline import NxDPPModel
    from neuronx_distributed.utils.activation_checkpoint import (
        apply_activation_checkpointing as nxd_apply_activation_checkpointing,
    )

    if isinstance(model, NxDPPModel):
        modules = model.local_module.modules()
    else:
        modules = model.modules()

    gradient_checkpointing_modules = set()
    for module in modules:
        if getattr(module, "gradient_checkpointing", False):
            module.gradient_checkpointing = False
            gradient_checkpointing_modules.add(module)

    def check_fn(m: torch.nn.Module) -> bool:
        return m in gradient_checkpointing_modules

    if gradient_checkpointing_modules:
        nxd_apply_activation_checkpointing(model, check_fn=check_fn)
