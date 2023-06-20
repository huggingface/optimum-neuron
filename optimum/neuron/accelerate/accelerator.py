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
import re
import shutil
from typing import TYPE_CHECKING, Optional

import torch
from accelerate import Accelerator
from accelerate.checkpointing import save_accelerator_state, save_custom_state

# from accelerate.state import AcceleratorState
from accelerate.utils import (
    DistributedType,
    is_fp8_available,
)

from ...utils import logging
from ..utils import is_torch_xla_available, patch_within_function
from ..utils.misc import args_and_kwargs_to_kwargs_only
from .optimizer import NeuronAcceleratedOptimizer
from .scheduler import NeuronAcceleratedScheduler
from .state import NeuronAcceleratorState
from .utils import NeuronDistributedType, NeuronFullyShardedDataParallelPlugin, patch_accelerate_is_tpu_available


if TYPE_CHECKING:
    try:
        from torch.optim.lr_scheduler import LRScheduler
    except ImportError:
        from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

if is_fp8_available():
    pass


logger = logging.get_logger(__name__)


# TODO: should we do a XLAFSDPNeuronAccelerator instead?
class NeuronAccelerator(Accelerator):
    @patch_within_function(("accelerate.accelerator.AcceleratorState", NeuronAcceleratorState))
    def __init__(self, *args, **kwargs):
        # Patches accelerate.utils.imports.is_tpu_available to match `is_torch_xla_available`
        patch_accelerate_is_tpu_available()

        full_kwargs = args_and_kwargs_to_kwargs_only(
            super().__init__, args=args, kwargs=kwargs, include_default_values=True
        )

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

        fsdp_plugin = full_kwargs["fsdp_plugin"]
        if os.environ.get("ACCELERATE_USE_FSDP", "false") == "true":
            if fsdp_plugin is None:
                fsdp_plugin = NeuronFullyShardedDataParallelPlugin()

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
        if self.distributed_type != DistributedType.DEEPSPEED:
            loss = loss / self.gradient_accumulation_steps
        if self.distributed_type is NeuronDistributedType.XLA_FSDP:
            self.backward_for_xla_fsdp(loss, **kwargs)
        elif self.scaler is not None:
            self.scaler.scale(loss).backward(**kwargs)
        else:
            # Providing **kwargs causes "Unsupported XLA type 10"
            loss.backward(**kwargs)

    def clip_grad_norm_for_xla_fsdp(self, parameters, max_norm, norm_type=2):
        self.unscale_gradients()
        parameters = list(parameters)
        for model in self._models:
            if parameters == list(model.parameters()):
                return model.clip_grad_norm_(max_norm, norm_type)

    def clip_grad_norm_(self, parameters, max_norm, norm_type=2):
        if self.distributed_type is NeuronDistributedType.XLA_FSDP:
            return self.clip_grad_norm_for_xla_fsdp(parameters, max_norm, norm_type=norm_type)
        return super().clip_grad_norm_(parameters, max_norm, norm_type=norm_type)

    def clip_grad_value_(self, parameters, clip_value):
        if self.distributed_type is NeuronDistributedType.XLA_FSDP:
            raise Exception("XLA FSDP  does not support `clip_grad_value_`. Use `clip_grad_norm_` instead.")
        return super().clip_grad_value_(parameters, clip_value)

    def save_state_for_xla_fsdp(self, output_dir: Optional[str] = None, **save_model_func_kwargs):
        if self.project_configuration.automatic_checkpoint_naming:
            output_dir = os.path.join(self.project_dir, "checkpoints")

        if output_dir is None:
            raise ValueError("An `output_dir` must be specified.")

        os.makedirs(output_dir, exist_ok=True)
        if self.project_configuration.automatic_checkpoint_naming:
            folders = [os.path.join(output_dir, folder) for folder in os.listdir(output_dir)]
            if self.project_configuration.total_limit is not None and (
                len(folders) + 1 > self.project_configuration.total_limit
            ):

                def _inner(folder):
                    return list(map(int, re.findall(r"[\/]?([0-9]+)(?=[^\/]*$)", folder)))[0]

                folders.sort(key=_inner)
                logger.warning(
                    f"Deleting {len(folders) + 1 - self.project_configuration.total_limit} checkpoints to make room for new checkpoint."
                )
                for folder in folders[: len(folders) + 1 - self.project_configuration.total_limit]:
                    shutil.rmtree(folder)
            output_dir = os.path.join(output_dir, f"checkpoint_{self.save_iteration}")
            if os.path.exists(output_dir):
                raise ValueError(
                    f"Checkpoint directory {output_dir} ({self.save_iteration}) already exists. Please manually override `self.save_iteration` with what iteration to start with."
                )
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving current state to {output_dir}")

        # Finish running the previous step before checkpointing
        xm.mark_step()

        # Save the models taking care of FSDP and DeepSpeed nuances
        weights = []
        for i, model in enumerate(self._models):
            logger.info("Saving FSDP model")
            self.state.fsdp_plugin.save_model(self, model, output_dir, i)
            logger.info(f"FSDP Model saved to output dir {output_dir}")

        # Save the optimizers taking care of FSDP and DeepSpeed nuances
        optimizers = []
        for i, opt in enumerate(self._optimizers):
            logger.info("Saving FSDP Optimizer")
            self.state.fsdp_plugin.save_optimizer(self, opt, self._models[i], output_dir, i)
            logger.info(f"FSDP Optimizer saved to output dir {output_dir}")

        # Save the lr schedulers taking care of DeepSpeed nuances
        schedulers = self._schedulers

        # Call model loading hooks that might have been registered with
        # accelerator.register_model_state_hook
        for hook in self._save_model_state_pre_hook.values():
            hook(self._models, weights, output_dir)

        save_location = save_accelerator_state(
            output_dir, weights, optimizers, schedulers, self.state.process_index, self.scaler
        )
        for i, obj in enumerate(self._custom_objects):
            save_custom_state(obj, output_dir, i)
        self.project_configuration.iteration += 1
        return save_location

    def save_state(self, output_dir: Optional[str] = None, **save_model_func_kwargs):
        if self.distributed_type is NeuronDistributedType.XLA_FSDP:
            return self.save_state_for_xla_fsdp(output_dir=output_dir, **save_model_func_kwargs)
        return super().save_state(output_dir=output_dir, **save_model_func_kwargs)
