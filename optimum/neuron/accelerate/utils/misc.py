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
import inspect
import itertools
from typing import TYPE_CHECKING, Callable, Dict, Optional, Union

import accelerate
import torch
import torch_xla.core.xla_model as xm
from neuronx_distributed.parallel_layers.parallel_state import (
    get_data_parallel_rank,
    model_parallel_is_initialized,
)
from neuronx_distributed.parallel_layers.utils import move_all_tensor_to_cpu
from torch_xla.core.xla_model import is_master_ordinal
from torch_xla.utils.checkpoint import checkpoint

from ....utils import logging
from ...utils import is_torch_neuronx_available, patch_everywhere
from ...utils.patching import Patcher


logger = logging.get_logger(__name__)

if TYPE_CHECKING:
    import os

    # Dummy class to avoid import errors in type checking.
    class NeuronPeftModel:
        def __init__(self, *args, **kwargs):
            pass

    from transformers import PreTrainedModel

    if is_torch_neuronx_available():
        from neuronx_distributed.pipeline import NxDPPModel


def patched_accelerate_is_torch_xla_available(check_is_tpu=False, check_is_gpu=False):
    """
    Fake `is_tpu_available` that returns `is_torch_xla_available` to patch `accelerate`.
    """
    return True


def patch_accelerate_is_torch_xla_available():
    # Since `is_torch_xla_available` does not work properly for us, it does not import `xm`, which causes failure.
    # We set it manually.
    accelerate.accelerator.xm = xm
    accelerate.state.xm = xm
    accelerate.checkpointing.xm = xm

    patch_everywhere(
        "is_torch_xla_available", patched_accelerate_is_torch_xla_available, module_name_prefix="accelerate"
    )


_ORIG_TORCH_FINFO = torch.finfo


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
    from safetensors.torch import save_file

    should_write_data = not master_only or is_master_ordinal(local=not global_master)
    cpu_data = move_all_tensor_to_cpu(tensors, convert=should_write_data)
    if should_write_data:
        save_file(cpu_data, filename, metadata=metadata)


def create_patched_save_pretrained(orig_save_pretrained_function: Callable[["PreTrainedModel"], None]):
    """
    Creates a wrapper around the `transformers.modeling_utils.PreTrainedModel.save_pretrained` method.
    This methods calls `tensor.data_ptr()` on the model parameters, which causes segmentation fault when the tensors
    are on the XLA device. To prevent that, this wrapper calls `save_pretrained` with the model on the CPU device.
    """
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
        cpu_state_dict = move_all_tensor_to_cpu(self.state_dict(), convert=should_write_data)
        self.load_state_dict(cpu_state_dict, assign=True)
        output = None
        if should_write_data:
            with patcher:
                output = orig_func(*args, **kwargs)
        self.load_state_dict(orig_state_dict, assign=True)
        xm.mark_step()
        del cpu_state_dict
        gc.collect()
        return output

    return wrapper.__get__(orig_self)


# TODO: @michaelbenayoun
# Needs to make it work in the general case or be deleted and only use `apply_activation_checkpointing`.
def patched_gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
    if not self.supports_gradient_checkpointing:
        raise ValueError(f"{self.__class__.__name__} does not support gradient checkpointing.")

    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {"use_reentrant": True}

    gradient_checkpointing_func = functools.partial(checkpoint, **gradient_checkpointing_kwargs)

    # For old GC format (transformers < 4.35.0) for models that live on the Hub
    # we will fall back to the overwritten `_set_gradient_checkpointing` method
    _is_using_old_format = "value" in inspect.signature(self._set_gradient_checkpointing).parameters

    if not _is_using_old_format:
        self._set_gradient_checkpointing(enable=True, gradient_checkpointing_func=gradient_checkpointing_func)
    else:
        self.apply(functools.partial(self._set_gradient_checkpointing, value=True))
        logger.warning(
            "You are using an old version of the checkpointing format that is deprecated (We will also silently ignore `gradient_checkpointing_kwargs` in case you passed it)."
            "Please update to the new format on your modeling file. To use the new format, you need to completely remove the definition of the method `_set_gradient_checkpointing` in your model."
        )

    if getattr(self, "_hf_peft_config_loaded", False):
        # When using PEFT + gradient checkpointing + Trainer we need to make sure the input has requires_grad=True
        # we do it also on PEFT: https://github.com/huggingface/peft/blob/85013987aa82aa1af3da1236b6902556ce3e483e/src/peft/peft_model.py#L334
        # When training with PEFT, only LoRA layers will have requires grad set to True, but the output of frozen layers need to propagate
        # the gradients to make sure the gradient flows.
        self.enable_input_require_grads()


def apply_activation_checkpointing(model: Union["PreTrainedModel", "NxDPPModel", "NeuronPeftModel"]):
    from neuronx_distributed.pipeline import NxDPPModel
    from neuronx_distributed.utils.activation_checkpoint import (
        apply_activation_checkpointing as nxd_apply_activation_checkpointing,
    )

    from ...peft.peft_model import NeuronPeftModel

    if isinstance(model, NeuronPeftModel):
        model._prepare_model_for_gradient_checkpointing(model.get_base_model())

    if isinstance(model, NxDPPModel):
        modules = itertools.chain(module.modules() for module in model.local_stage_modules)
    else:
        modules = model.modules()

    gradient_checkpointing_modules = set()
    for module in modules:
        if isinstance(module, torch.nn.ModuleList):
            for mod in module:
                # TODO: @michaelbenayoun. Need to find a better way to identify the blocks to apply gradient
                # checkpointing to.
                if "Layer" in mod.__class__.__name__ or "Block" in mod.__class__.__name__:
                    gradient_checkpointing_modules.add(mod)

    def check_fn(m: torch.nn.Module) -> bool:
        return m in gradient_checkpointing_modules

    if gradient_checkpointing_modules:
        nxd_apply_activation_checkpointing(model, check_fn=check_fn)
