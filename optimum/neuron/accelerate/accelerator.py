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
"""Custom Accelerator class for Neuron."""

import inspect
import os
from typing import TYPE_CHECKING, List, Optional, Union
from optimum.neuron.accelerate.state import NeuronAcceleratorState
from optimum.neuron.accelerate.utils.dataclasses import NeuronDistributedType

import torch
import accelerate
from accelerate import Accelerator
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.scheduler import AcceleratedScheduler
# from accelerate.state import AcceleratorState
from accelerate.tracking import GeneralTracker
from accelerate.utils import (
    DeepSpeedPlugin,
    DistributedType,
    DynamoBackend,
    FullyShardedDataParallelPlugin,
    GradientAccumulationPlugin,
    KwargsHandler,
    LoggerType,
    MegatronLMPlugin,
    PrecisionType,
    ProjectConfiguration,
    RNGType,
    convert_model,
    convert_outputs_to_fp32,
    has_transformer_engine_layers,
    is_fp8_available,
    is_torch_version,
)

from ...utils import logging
from ..utils import is_neuronx_available
from ..utils.misc import args_and_kwargs_to_kwargs_only, patch_within_function
from .state import NeuronAcceleratorState
from .scheduler import NeuronAcceleratedScheduler
from .optimizer import NeuronAcceleratedOptimizer


if TYPE_CHECKING:
    try:
        from torch.optim.lr_scheduler import LRScheduler
    except ImportError:
        from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

if is_neuronx_available():
    import torch_xla.distributed.xla_multiprocessing as xmp

if is_fp8_available():
    import transformer_engine.common.recipe as te_recipe
    from transformer_engine.pytorch import fp8_autocast


logger = logging.get_logger(__name__)


class NeuronAcceleratedOptimizer(AcceleratedOptimizer):
    def step(self, closure=None):
        if self.gradient_state.sync_gradients:
            if self.accelerator_state.distributed_type == DistributedType.TPU:
                # optimizer_args = {"closure": closure} if closure is not None else {}
                # xm.optimizer_step(self.optimizer, optimizer_args=optimizer_args)
                self.optimizer.step(closure)
            elif self.scaler is not None:
                scale_before = self.scaler.get_scale()
                self.scaler.step(self.optimizer, closure)
                self.scaler.update()
                scale_after = self.scaler.get_scale()
                # If we reduced the loss scale, it means the optimizer step was skipped because of gradient overflow.
                self._is_overflow = scale_after < scale_before
            else:
                self.optimizer.step(closure)





class NeuronAccelerator(Accelerator):

    @patch_within_function(("accelerate.accelerator.AcceleratorState", NeuronAcceleratorState))
    def __init__(self, *args, **kwargs):
        full_kwargs = args_and_kwargs_to_kwargs_only(super().__init__, args=args, kwargs=kwargs, include_default_values=True)

        # There is a check for gradient_accumulation_steps to be equal to 1 when
        # DistributedType == DistributedType.TPU, so we change that for initialization
        # and restore it back afterwards.
        num_steps = 1
        gradient_accumulation_plugin = full_kwargs["gradient_accumulation_plugin"]
        gradient_accumulation_steps = full_kwargs["gradient_accumulation_steps"]
        if gradient_accumulation_plugin is not None:
            num_steps = gradient_accumulation_plugin.num_steps
            gradient_accumulation_plugin.num_steps = 1
        elif gradient_accumulation_steps != 1:
            num_steps = gradient_accumulation_steps
            gradient_accumulation_steps = 1
        full_kwargs["gradient_accumulation_plugin"] = gradient_accumulation_plugin
        full_kwargs["gradient_accumulation_steps"] = gradient_accumulation_steps

        super().__init__(**full_kwargs)

        if num_steps != 1:
            self.gradient_accumulation_steps = num_steps

    @patch_within_function(("accelerate.accelerator.AcceleratedOptimizer", NeuronAcceleratedOptimizer))
    def prepare_optimizer(self, optimizer: torch.optim.Optimizer, device_placement=None):
        return super().prepare_optimizer(optimizer, device_placement=device_placement)

    @patch_within_function(("accelerate.accelerator.AcceleratedScheduler", NeuronAcceleratedScheduler))
    def prepare_scheduler(self, scheduler: "LRScheduler"):
        return super().prepare_scheduler(scheduler)

    def prepare_model_for_xla_fsdp(self, model: torch.nn.Module, device_placement=None):
        if device_placement is None:
            device_placement = self.device_placement
        self._models.append(model)
        # We check only for models loaded with `accelerate`

        # Checks if any of the child module has the attribute `hf_device_map`.
        has_hf_device_map = False
        for m in model.modules():
            if hasattr(m, "hf_device_map"):
                has_hf_device_map = True
                break

        if getattr(model, "is_loaded_in_8bit", False) and getattr(model, "hf_device_map", False):
            model_devices = set(model.hf_device_map.values())
            if len(model_devices) > 1:
                raise ValueError(
                    "You can't train a model that has been loaded in 8-bit precision on multiple devices."
                )

            current_device_index = list(model_devices)[0]
            if torch.device(current_device_index) != self.device:
                # if on the first device (GPU 0) we don't care
                if (self.device.index is not None) or (current_device_index != 0):
                    raise ValueError(
                        "You can't train a model that has been loaded in 8-bit precision on a different device than the one "
                        "you're training on. Make sure you loaded the model on the correct device using for example `device_map={'':torch.cuda.current_device()}"
                        "you're training on. Make sure you loaded the model on the correct device using for example `device_map={'':torch.cuda.current_device() or device_map={'':torch.xpu.current_device()}"
                    )

            if "cpu" in model_devices or "disk" in model_devices:
                raise ValueError(
                    "You can't train a model that has been loaded in 8-bit precision with CPU or disk offload."
                )
        elif device_placement and not has_hf_device_map:
            model = model.to(self.device)

        try:
            from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
        except ImportError:
            raise ImportError("Missing XLA FSDP related module; please make sure to use torch-xla >= 2.0.")
        
        # Check if the model is already a FSDP model due to `Manual Wrapping` and if so,
        # don't wrap it again
        # TODO: validate which arguments work for XLA FSDP.
        if type(model) != FSDP:
            self.state.fsdp_plugin.set_auto_wrap_policy(model)
            fsdp_plugin = self.state.fsdp_plugin
            kwargs = {
                "sharding_strategy": fsdp_plugin.sharding_strategy,
                "cpu_offload": fsdp_plugin.cpu_offload,
                "auto_wrap_policy": fsdp_plugin.auto_wrap_policy,
                "backward_prefetch": fsdp_plugin.backward_prefetch,
                "mixed_precision": fsdp_plugin.mixed_precision_policy,
                "ignored_modules": fsdp_plugin.ignored_modules,
                "device_id": self.device,
            }
            signature = inspect.signature(FSDP.__init__).parameters.keys()
            if "limit_all_gathers" in signature:
                kwargs["limit_all_gathers"] = fsdp_plugin.limit_all_gathers
            if "use_orig_params" in signature:
                kwargs["use_orig_params"] = fsdp_plugin.use_orig_params
            model = FSDP(model, **kwargs)
        self._models[-1] = model

        return model

    def prepare_model(self, model: torch.nn.Module, device_placement=None):
        if self.distributed_type is NeuronDistributedType.XLA_FSDP:
            return self.prepare_model_for_xla_fsdp(model, device_placement=device_placement)
        return super().prepare_model(model, device_placement=device_placement)


    def backward_for_xla_fsdp(self, loss, **kwargs):
        if self.scaler is not None:
            self.scaler.scale(loss).backward(**kwargs)
        else:
            loss.backward(**kwargs)

    def backward(self, loss, **kwargs):
        if self.distributed_type is NeuronDistributedType.XLA_FSDP:
            return self.prepare_model_for_xla_fsdp(loss, **kwargs)
        return super().backward(loss, **kwargs)
