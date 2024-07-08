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
import inspect
from typing import TYPE_CHECKING, Union

import torch

from ....utils import logging
from ...utils import is_torch_neuronx_available, is_torch_xla_available, patch_everywhere
from ...utils.peft_utils import NeuronPeftModel
from ...utils.require_utils import requires_neuronx_distributed, requires_torch_xla


logger = logging.get_logger(__name__)

if TYPE_CHECKING:

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


# TODO: @michaelbenayoun
# Needs to make it work in the general case or be deleted and only use `apply_activation_checkpointing`.
@requires_torch_xla
def patched_gradient_checkpointing_enable(self, gradient_checkpointing_kwargs=None):
    from torch_xla.utils.checkpoint import checkpoint

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


@requires_neuronx_distributed
def apply_activation_checkpointing(model: Union["PreTrainedModel", "NxDPPModel", NeuronPeftModel]):
    from neuronx_distributed.pipeline import NxDPPModel
    from neuronx_distributed.utils.activation_checkpoint import (
        apply_activation_checkpointing as nxd_apply_activation_checkpointing,
    )

    if isinstance(model, NeuronPeftModel):
        model._prepare_model_for_gradient_checkpointing(model.get_base_model())

    if isinstance(model, NxDPPModel):
        modules = model.local_module.modules()
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
