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
"""Utilities related to the model."""

from typing import TYPE_CHECKING, Dict, Tuple, Union

import torch

from .import_utils import is_torch_neuronx_available
from .require_utils import requires_neuronx_distributed


if TYPE_CHECKING:

    if is_torch_neuronx_available():
        from neuronx_distributed.pipeline import NxDPPModel


@requires_neuronx_distributed
def get_tied_parameters_dict(model: Union["torch.nn.Module", "NxDPPModel"]) -> Dict[str, str]:
    from neuronx_distributed.pipeline import NxDPPModel

    if isinstance(model, NxDPPModel):
        tied_parameters = {}
        for module in model.local_stage_modules:
            tied_parameters.update(get_tied_parameters_dict(module))
        return tied_parameters

    unique_parameters = {}
    tied_parameters = {}
    for name, param in model.named_parameters(remove_duplicate=False):
        if param in unique_parameters:
            tied_parameter_name = unique_parameters[param]
            tied_parameters[name] = tied_parameter_name
        else:
            unique_parameters[param] = name
    return tied_parameters


def get_parent_module_and_param_name_from_fully_qualified_name(
    module: "torch.nn.Module", fully_qualified_name: str
) -> Tuple["torch.nn.Module", str]:
    fully_qualified_name = fully_qualified_name.rsplit(".", maxsplit=1)
    parent_module = module if len(fully_qualified_name) == 1 else module.get_submodule(fully_qualified_name[0])
    param_name = fully_qualified_name[0] if len(fully_qualified_name) == 1 else fully_qualified_name[1]
    return parent_module, param_name


@requires_neuronx_distributed
def tie_parameters(model: Union["torch.nn.Module", "NxDPPModel"], tied_parameters_dict: Dict[str, str]):
    from neuronx_distributed.pipeline import NxDPPModel

    if isinstance(model, NxDPPModel):
        for module in model.local_stage_modules:
            tie_parameters(module, tied_parameters_dict)
    else:
        for param_to_tie_name, param_name in tied_parameters_dict.items():
            param_to_tie_parent_module, param_to_tie_name = get_parent_module_and_param_name_from_fully_qualified_name(
                model, param_to_tie_name
            )
            param_to_tie = getattr(param_to_tie_parent_module, param_to_tie_name)

            parent_module, param_name = get_parent_module_and_param_name_from_fully_qualified_name(model, param_name)
            param = getattr(parent_module, param_name)

            if param_to_tie is not param:
                del param_to_tie
                setattr(param_to_tie_parent_module, param_to_tie_name, param)
