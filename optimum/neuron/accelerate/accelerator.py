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
"""Custom Accelerator classes for Neuron."""

import inspect
import os
from typing import TYPE_CHECKING, List, Optional, Union
from optimum.neuron.accelerate.state import NeuronAcceleratorState

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
from .state import NeuronAcceleratorState


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


class NeuronAcceleratedScheduler(AcceleratedScheduler):
    def step(self, *args, **kwargs):
        if not self.step_with_optimizer:
            # No link between scheduler and optimizer -> just step
            self.scheduler.step(*args, **kwargs)
            return

        # Otherwise, first make sure the optimizer was stepped.
        if not self.gradient_state.sync_gradients:
            if self.gradient_state.adjust_scheduler:
                self.scheduler._step_count += 1
            return

        for opt in self.optimizers:
            if opt.step_was_skipped:
                return
        if self.split_batches:
            # Split batches -> the training dataloader batch size is not changed so one step per training step
            self.scheduler.step(*args, **kwargs)
        else:
            # Otherwise the training dataloader batch size was multiplied by `num_processes`, so we need to do
            # num_processes steps per training step
            num_processes = NeuronAcceleratorState().num_processes
            for _ in range(num_processes):
                # Special case when using OneCycle and `drop_last` was not used
                if hasattr(self.scheduler, "total_steps"):
                    if self.scheduler._step_count <= self.scheduler.total_steps:
                        self.scheduler.step(*args, **kwargs)
                else:
                    self.scheduler.step(*args, **kwargs)



class NeuronAccelerator(Accelerator):
    def __init__(
        self,
        device_placement: bool = True,
        split_batches: bool = False,
        mixed_precision: Optional[Union[PrecisionType, str]] = None,
        gradient_accumulation_steps: int = 1,
        cpu: bool = False,
        deepspeed_plugin: Optional[DeepSpeedPlugin] = None,
        fsdp_plugin: Optional[FullyShardedDataParallelPlugin] = None,
        megatron_lm_plugin: Optional[MegatronLMPlugin] = None,
        ipex_plugin=None,
        rng_types: Optional[List[Union[str, RNGType]]] = None,
        log_with: Optional[
            Union[str, LoggerType, GeneralTracker, List[Union[str, LoggerType, GeneralTracker]]]
        ] = None,
        project_dir: Optional[Union[str, os.PathLike]] = None,
        project_config: Optional[ProjectConfiguration] = None,
        logging_dir: Optional[Union[str, os.PathLike]] = None,
        gradient_accumulation_plugin: Optional[GradientAccumulationPlugin] = None,
        dispatch_batches: Optional[bool] = None,
        even_batches: bool = True,
        step_scheduler_with_optimizer: bool = True,
        kwargs_handlers: Optional[List[KwargsHandler]] = None,
        dynamo_backend: Optional[Union[DynamoBackend, str]] = None,
    ):
        accelerate.accelerator.AcceleratorState = NeuronAcceleratorState
        # TODO: restore it back afterwards.

        # There is a check for gradient_accumulation_steps to be equal to 1 when
        # DistributedType == DistributedType.TPU, so we change that for initialization
        # and restore it back afterwards.
        num_steps = 1
        if gradient_accumulation_plugin is not None:
            num_steps = gradient_accumulation_plugin.num_steps
            gradient_accumulation_plugin.num_steps = 1
        elif gradient_accumulation_steps != 1:
            num_steps = gradient_accumulation_steps
            gradient_accumulation_steps = 1

        super().__init__(
            device_placement=device_placement,
            split_batches=split_batches,
            mixed_precision=mixed_precision,
            gradient_accumulation_steps=gradient_accumulation_steps,
            cpu=cpu,
            deepspeed_plugin=deepspeed_plugin,
            fsdp_plugin=fsdp_plugin,
            megatron_lm_plugin=megatron_lm_plugin,
            ipex_plugin=ipex_plugin,
            rng_types=rng_types,
            log_with=log_with,
            project_dir=project_dir,
            project_config=project_config,
            logging_dir=logging_dir,
            gradient_accumulation_plugin=gradient_accumulation_plugin,
            dispatch_batches=dispatch_batches,
            even_batches=even_batches,
            step_scheduler_with_optimizer=step_scheduler_with_optimizer,
            kwargs_handlers=kwargs_handlers,
            dynamo_backend=dynamo_backend,
        )
        if num_steps != 1:
            self.gradient_accumulation_steps = num_steps

        print("Distributed type", self.distributed_type)
        from accelerate.accelerator import AcceleratorState
        print("AcceleratorState", AcceleratorState)


    def prepare_optimizer(self, optimizer: torch.optim.Optimizer, device_placement=None):
        if device_placement is None:
            device_placement = self.device_placement
        optimizer = NeuronAcceleratedOptimizer(optimizer, device_placement=device_placement, scaler=self.scaler)
        self._optimizers.append(optimizer)
        return optimizer

    def prepare_scheduler(self, scheduler: "LRScheduler"):
        # We try to find the optimizer associated with `scheduler`, the default is the full list.
        optimizer = self._optimizers
        for opt in self._optimizers:
            if getattr(scheduler, "optimizer", None) == opt.optimizer:
                optimizer = opt
                break
        scheduler = NeuronAcceleratedScheduler(
            scheduler,
            optimizer,
            step_with_optimizer=self.step_scheduler_with_optimizer,
            split_batches=self.split_batches,
        )
        self._schedulers.append(scheduler)
        return scheduler

    def prepare_model(self, model: torch.nn.Module, device_placement=None):
        if device_placement is None:
            device_placement = self.device_placement and self.distributed_type != DistributedType.FSDP
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

        distributed_types = [DistributedType.MULTI_GPU]
        if hasattr(DistributedType, "MULTI_XPU"):
            distributed_types.append(DistributedType.MULTI_XPU)

        if self.distributed_type in distributed_types:
            if any(p.requires_grad for p in model.parameters()):
                kwargs = self.ddp_handler.to_kwargs() if self.ddp_handler is not None else {}
                model = torch.nn.parallel.DistributedDataParallel(
                    model, device_ids=[self.local_process_index], output_device=self.local_process_index, **kwargs
                )
        elif self.distributed_type == DistributedType.FSDP:
            try:
                from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
            except ImportError:
                raise ImportError("Missing XLA FSDP related module; please make sure to use torch-xla >= 2.0.")

            import pdb; pdb.set_trace()

            # Check if the model is already a FSDP model due to `Manual Wrapping` and if so,
            # don't wrap it again
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
        elif self.distributed_type == DistributedType.MULTI_CPU:
            kwargs = self.ddp_handler.to_kwargs() if self.ddp_handler is not None else {}
            model = torch.nn.parallel.DistributedDataParallel(model, **kwargs)
        if self.native_amp:
            model._original_forward = model.forward
            if self.mixed_precision == "fp16" and is_torch_version(">=", "1.10"):
                model.forward = torch.cuda.amp.autocast(dtype=torch.float16)(model.forward)
            elif self.mixed_precision == "bf16" and self.distributed_type != DistributedType.TPU:
                model.forward = torch.autocast(device_type=self.device.type, dtype=torch.bfloat16)(model.forward)
            else:
                model.forward = torch.cuda.amp.autocast()(model.forward)
            model.forward = convert_outputs_to_fp32(model.forward)
        elif self.mixed_precision == "fp8":
            if not has_transformer_engine_layers(model):
                with torch.no_grad():
                    convert_model(model)
                model._converted_to_transformer_engine = True
            model._original_forward = model.forward

            kwargs = self.fp8_recipe_handler.to_kwargs() if self.fp8_recipe_handler is not None else {}
            if "fp8_format" in kwargs:
                kwargs["fp8_format"] = getattr(te_recipe.Format, kwargs["fp8_format"])
            fp8_recipe = te_recipe.DelayedScaling(**kwargs)
            cuda_device_capacity = torch.cuda.get_device_capability()
            fp8_enabled = cuda_device_capacity[0] >= 9 or (
                cuda_device_capacity[0] == 8 and cuda_device_capacity[1] >= 9
            )
            if not fp8_enabled:
                logger.warn(
                    f"The current device has compute capability of {cuda_device_capacity} which is "
                    "insufficient for FP8 mixed precision training (requires a GPU Hopper/Ada Lovelace "
                    "or higher, compute capability of 8.9 or higher). Will use FP16 instead."
                )
            model.forward = fp8_autocast(enabled=fp8_enabled, fp8_recipe=fp8_recipe)(model.forward)
        if self.distributed_type == DistributedType.TPU and self.state.fork_launched:
            model = xmp.MpModelWrapper(model).to(self.device)
        # torch.compile should be called last.
        if self.state.dynamo_plugin.backend != DynamoBackend.NO:
            if not is_torch_version(">=", "2.0"):
                raise ValueError("Using `torch.compile` requires PyTorch 2.0 or higher.")
            model = torch.compile(model, **self.state.dynamo_plugin.to_kwargs())
        return model

    def backward(self, loss, **kwargs):
        if self.distributed_type != DistributedType.DEEPSPEED:
            # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
            loss = loss / self.gradient_accumulation_steps
        if self.distributed_type == DistributedType.DEEPSPEED:
            self.deepspeed_engine_wrapped.backward(loss, **kwargs)
        elif self.distributed_type == DistributedType.MEGATRON_LM:
            return
        elif self.distributed_type == DistributedType.TPU:
            loss.backward(**kwargs)
        elif self.scaler is not None:
            self.scaler.scale(loss).backward(**kwargs)
        else:
            loss.backward(**kwargs)
