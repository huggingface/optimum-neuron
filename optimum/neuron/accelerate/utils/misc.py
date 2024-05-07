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

import functools
import gc
from typing import TYPE_CHECKING, Callable, Dict, Optional, Union

import torch
from transformers.modeling_utils import get_parameter_dtype

from ...distributed.utils import named_parameters
from ...utils import is_torch_neuronx_available, is_torch_xla_available, patch_everywhere
from ...utils.patching import Patcher
from ...utils.require_utils import requires_neuronx_distributed, requires_safetensors


if TYPE_CHECKING:
    import os

    from transformers import PreTrainedModel

    if is_torch_neuronx_available():
        from neuronx_distributed.pipeline import NxDPPModel


def patched_accelerate_is_torch_xla_available(check_is_tpu=False, check_is_gpu=False):
    """
    Fake `is_tpu_available` that returns `is_torch_xla_available` to patch `accelerate`.
    """
    return is_torch_xla_available()


def patch_accelerate_is_torch_xla_available():
    if is_torch_xla_available():
        import accelerate
        import torch_xla.core.xla_model as xm

        # Since `is_torch_xla_available` does not work properly for us, it does not import `xm`, which causes failure.
        # We set it manually.
        accelerate.accelerator.xm = xm
        accelerate.state.xm = xm
        accelerate.checkpointing.xm = xm

    patch_everywhere(
        "is_torch_xla_available", patched_accelerate_is_torch_xla_available, module_name_prefix="accelerate"
    )


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
@requires_safetensors
def torch_xla_safe_save_file(
    tensors: Dict[str, torch.Tensor],
    filename: Union[str, "os.PathLike"],
    metadata: Optional[Dict[str, str]] = None,
    master_only: bool = True,
    global_master: bool = False,
):
    """
    Torch XLA compatible implementation of `safetensors.torch.save_file`.
    """
    from neuronx_distributed.parallel_layers.utils import move_all_tensor_to_cpu
    from safetensors.torch import save_file
    from torch_xla.core.xla_model import is_master_ordinal

    should_write_data = not master_only or is_master_ordinal(local=not global_master)
    cpu_data = move_all_tensor_to_cpu(tensors, convert=should_write_data)
    if should_write_data:
        save_file(cpu_data, filename, metadata=metadata)


@requires_neuronx_distributed
def create_patched_save_pretrained(orig_save_pretrained_function: Callable[["PreTrainedModel"], None]):
    """
    Creates a wrapper around the `transformers.modeling_utils.PreTrainedModel.save_pretrained` method.
    This methods calls `tensor.data_ptr()` on the model parameters, which causes segmentation fault when the tensors
    are on the XLA device. To prevent that, this wrapper calls `save_pretrained` with the model on the CPU device.
    """
    import torch_xla.core.xla_model as xm
    from neuronx_distributed.parallel_layers.parallel_state import (
        get_data_parallel_rank,
        model_parallel_is_initialized,
    )
    from neuronx_distributed.parallel_layers.utils import move_all_tensor_to_cpu

    orig_self = orig_save_pretrained_function.__self__
    orig_func = orig_save_pretrained_function.__func__

    patcher = Patcher([("transformers.modeling_utils.safe_save_file", torch_xla_safe_save_file)])

    @functools.wraps(orig_func)
    def wrapper(*args, **kwargs):
        self = args[0]
        if model_parallel_is_initialized():
            should_write_data = get_data_parallel_rank() == 0
        else:
            should_write_data = xm.is_master_ordinal(local=True)
        orig_state_dict = self.state_dict()
        if should_write_data:
            cpu_state_dict = move_all_tensor_to_cpu(self.state_dict(), convert=True)
            self.load_state_dict(cpu_state_dict, assign=True)
            with patcher:
                output = orig_func(*args, **kwargs)
            self.load_state_dict(orig_state_dict, assign=True)
            del cpu_state_dict
            gc.collect()

    return wrapper.__get__(orig_self)


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
