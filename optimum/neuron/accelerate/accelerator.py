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

import contextlib
import os
import re
import shutil
import sys
import warnings
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch
from accelerate import Accelerator
from accelerate.checkpointing import save_accelerator_state, save_custom_state
from accelerate.utils import AutocastKwargs, DistributedType
from accelerate.utils.operations import gather_object, recursively_apply
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import PreTrainedModel

from ...utils import logging
from ..models.neuron_config import TrainingNeuronConfig
from ..models.training.pipeline_utils import create_nxdpp_model
from ..utils import (
    DynamicPatch,
    ModelPatcher,
    Patcher,
    patch_within_function,
)
from ..utils.misc import args_and_kwargs_to_kwargs_only, is_main_worker
from ..utils.torch_xla_and_neuronx_initialization import check_neuron_cc_flags_for_model
from ..utils.training_utils import is_custom_modeling_model
from .optimizer import NeuronAcceleratedOptimizer
from .scheduler import NeuronAcceleratedScheduler
from .state import NeuronAcceleratorState
from .utils import (
    AutocastBackend,
    NeuronDistributedType,
    patch_accelerate_is_torch_xla_available,
)
from .utils.misc import (
    apply_activation_checkpointing,
    create_patched_save_pretrained,
)
from .utils.operations import _xla_gather


if TYPE_CHECKING:
    try:
        from torch.optim.lr_scheduler import LRScheduler
    except ImportError:
        from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from neuronx_distributed.utils.model_utils import move_model_to_device
from torch_xla.distributed.parallel_loader import MpDeviceLoader


logger = logging.get_logger(__name__)


class NeuronAccelerator(Accelerator):
    def __init__(
        self,
        *args,
        trn_config: TrainingNeuronConfig | None = None,
        zero_1: bool = False,
        autocast_backend: str | AutocastBackend = "xla",
        **kwargs,
    ):
        # Patches accelerate.utils.imports.is_tpu_available to match `is_torch_xla_available`
        # TODO: check that removing it does not break anything.
        patch_accelerate_is_torch_xla_available()

        full_kwargs = args_and_kwargs_to_kwargs_only(
            super().__init__, args=args, kwargs=kwargs, include_default_values=True
        )

        # There is a check for gradient_accumulation_steps to be equal to 1 when
        # DistributedType == DistributedType.XLA, so we change that for initialization
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
        if fsdp_plugin is not None:
            raise ValueError("FSDP is not supported.")
        self.fsdp_plugin = fsdp_plugin

        self._model_cpu_parameters_to_xla = {}

        if not isinstance(autocast_backend, AutocastBackend):
            autocast_backend = AutocastBackend(autocast_backend)

        # The original `is_torch_xla_available` function is checking for TPU or GPU in `accelerate`.
        # Here, we patch it to return True for Neuron cores as well.
        def patched_is_torch_xla_available(check_is_tpu: bool = False, check_is_gpu: bool = False) -> bool:
            return True

        import accelerate

        accelerate.state.is_torch_xla_available = patched_is_torch_xla_available

        patched_accelerator_state = partial(
            NeuronAcceleratorState, trn_config=trn_config, autocast_backend=autocast_backend
        )
        with Patcher([("accelerate.accelerator.AcceleratorState", patched_accelerator_state)]):
            super().__init__(**full_kwargs)

        self.zero_1 = zero_1

        if self.autocast_handler is None:
            enabled = self.state.mixed_precision == "bf16" and autocast_backend is AutocastBackend.AMP
            self.autocast_handler = AutocastKwargs(enabled=enabled)

        if self.process_index == -1 and self.zero_1:
            raise ValueError("XLA ZeRO Stage 1 can only be enabled in a distributed training setting.")

        if num_steps != 1:
            self.gradient_accumulation_steps = num_steps

    def _prepare_data_loader_for_distributed(
        self,
        data_loader: DataLoader,
        num_replicas: int,
        rank: int,
        force_drop_last: bool,
    ) -> DataLoader:
        # TODO: make it more robust, similar to the prepare_data_loader function in `accelerate`.
        if isinstance(data_loader.sampler, DistributedSampler):
            return data_loader

        orig_sampler = data_loader.sampler
        if hasattr(orig_sampler, "shuffle"):
            shuffle = orig_sampler.shuffle
        elif isinstance(orig_sampler, torch.utils.data.SequentialSampler):
            shuffle = False
        else:
            shuffle = True
            if not isinstance(orig_sampler, torch.utils.data.RandomSampler):
                logger.warning(
                    f"The sampler {orig_sampler} is going to be replaced by a torch.utils.data.DistributedSampler. This "
                    "new sampler will shuffle the dataset, it might not be the expected behaviour."
                )

        sampler = DistributedSampler(data_loader.dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle)

        distributed_dataloader = DataLoader(
            data_loader.dataset,
            batch_size=data_loader.batch_size,
            sampler=sampler,
            num_workers=data_loader.num_workers,
            collate_fn=data_loader.collate_fn,
            pin_memory=data_loader.pin_memory,
            drop_last=data_loader.drop_last or force_drop_last,
        )

        distributed_dataloader._is_accelerate_prepared = True
        return distributed_dataloader

    def prepare_data_loader(
        self, data_loader: DataLoader, device_placement: bool | None = None, use_mp_device_loader: bool = False
    ):
        force_drop_last = False
        if self.state.distributed_type is NeuronDistributedType.MODEL_PARALLELISM:
            from neuronx_distributed import parallel_layers

            num_replicas = parallel_layers.parallel_state.get_data_parallel_size()
            rank = parallel_layers.parallel_state.get_data_parallel_rank()
            force_drop_last = parallel_layers.parallel_state.get_pipeline_model_parallel_size() > 1
            if is_main_worker() and force_drop_last:
                logger.warning(
                    "Pipeline parallelsim: forcing the dataloader to drop the last incomplete batch because it can "
                    "cause failure if the last batch size is not divisible by the number of microbatches for the pipeline."
                )
        else:
            num_replicas = xr.world_size()
            rank = xr.global_ordinal()
        if self.state.num_processes > 1:
            data_loader = self._prepare_data_loader_for_distributed(
                data_loader, num_replicas=num_replicas, rank=rank, force_drop_last=force_drop_last
            )
            # No need to wrap the dataloader if we are using pipeline parallelism.
            if use_mp_device_loader and self.state.trn_config.pipeline_parallel_size == 1:
                data_loader = MpDeviceLoader(data_loader, self.device)
        return data_loader

    def _prepare_optimizer_for_zero_1(self, optimizer: torch.optim.Optimizer, device_placement=None):
        mixed_precision_to_dtype = {
            "no": torch.float32,
            "bf16": torch.bfloat16,
        }
        optimizer_dtype = mixed_precision_to_dtype.get(self.state.mixed_precision, None)
        if optimizer_dtype is None:
            raise ValueError(f"The precision {self.state.mixed_precision} is not supported for ZeRO Stage 1")

        from neuronx_distributed.optimizer import NeuronZero1Optimizer
        from neuronx_distributed.parallel_layers.parallel_state import (
            get_data_parallel_group,
            get_tensor_model_parallel_group,
            model_parallel_is_initialized,
        )

        if not model_parallel_is_initialized():
            sharding_groups = None
            grad_norm_groups = None
        else:
            sharding_groups = get_data_parallel_group(as_list=True)
            grad_norm_groups = get_tensor_model_parallel_group(as_list=True)

        zero_1_optimizer = NeuronZero1Optimizer(
            optimizer.param_groups,
            optimizer.__class__,
            optimizer_dtype=optimizer_dtype,
            pin_layout=False,
            sharding_groups=sharding_groups,
            grad_norm_groups=grad_norm_groups,
        )

        return zero_1_optimizer

    @patch_within_function(("accelerate.accelerator.AcceleratedOptimizer", NeuronAcceleratedOptimizer))
    def prepare_optimizer(self, optimizer: torch.optim.Optimizer, device_placement: bool | None = None):
        if self.zero_1:
            optimizer = self._prepare_optimizer_for_zero_1(optimizer, device_placement=device_placement)
        return super().prepare_optimizer(optimizer, device_placement=device_placement)

    @patch_within_function(("accelerate.accelerator.AcceleratedScheduler", NeuronAcceleratedScheduler))
    def prepare_scheduler(self, scheduler: "LRScheduler"):
        return super().prepare_scheduler(scheduler)

    def patch_model_for_neuron(
        self,
        model: "torch.nn.Module",
    ) -> "torch.nn.Module":
        patching_specs = [
            ("config.layerdrop", 0),
            ("no_sync", lambda: contextlib.nullcontext()),
        ]

        if not is_custom_modeling_model(model):
            patching_specs.append(
                (
                    "save_pretrained",
                    DynamicPatch(create_patched_save_pretrained),
                ),
            )

        prepared_patching_specs = []
        for spec in patching_specs:
            prepared_patching_specs.append((model,) + spec)

        model_patcher = ModelPatcher(prepared_patching_specs, ignore_missing_attributes=True)
        model_patcher.patch()

        return model

    def prepare_model(
        self, model: torch.nn.Module, device_placement: bool | None = None, evaluation_mode: bool = False
    ):
        # If the model was already prepared, we skip.
        if model in self._models:
            return model

        if self.distributed_type is NeuronDistributedType.MODEL_PARALLELISM and not is_custom_modeling_model(model):
            raise NotImplementedError(
                "Model parallelism is only supported for models with a custom modeling implementation."
            )

        # Since it is not possible to set the best compiler flags for a given model because XLA is initialized before
        # we get access to the model, we simply check if the flags are the best and notify the user otherwise.
        check_neuron_cc_flags_for_model(model)

        model = self.patch_model_for_neuron(model)

        # We do not want to use the cache, or output unused tensors as it would imply more communication that we do not
        # need.
        model.config.use_cache = False
        model.config.output_attentions = False
        model.config.output_hidden_states = False

        if is_custom_modeling_model(model):
            if model.trn_config.pipeline_parallel_size > 1:
                model = create_nxdpp_model(model)
                model.move_model_to_device()
            else:
                move_model_to_device(model, self.device)
                model.tie_weights()
            model = super().prepare_model(model, device_placement=False, evaluation_mode=evaluation_mode)
        else:
            # Question: should do the same for custom models?
            if self.state.mixed_precision == "bf16":
                model.to(torch.bfloat16)

            should_apply_activation_checkpointing = False
            for mod in model.modules():
                if getattr(mod, "gradient_checkpointing", False):
                    should_apply_activation_checkpointing = True
                    model.gradient_checkpointing_disable()

            # It is needed for now otherwise sdpa is used since PT > 2.* is available.
            for module in model.modules():
                if getattr(module, "_use_sdpa", False):
                    module._use_sdpa = False
                if getattr(module, "_use_flash_attention_2", False):
                    module._use_flash_attention_2 = False

            if should_apply_activation_checkpointing:
                apply_activation_checkpointing(model)
            move_model_to_device(model, xm.xla_device())
            model.tie_weights()
            device_placement = False
            model = super().prepare_model(model, device_placement=device_placement, evaluation_mode=evaluation_mode)

        xm.mark_step()
        return model

    def backward(self, loss, **kwargs):
        if self.distributed_type != DistributedType.DEEPSPEED:
            loss = loss / self.gradient_accumulation_steps
        if self.scaler is not None:
            self.scaler.scale(loss).backward(**kwargs)
        else:
            loss.backward(**kwargs)

    @contextlib.contextmanager
    def autocast(self, cache_enabled: bool = False, autocast_handler: AutocastKwargs | None = None):
        if cache_enabled:
            warnings.warn(
                "Passing `cache_enabled=True` to `accelerator.autocast` is deprecated and will be removed in v0.23.0. "
                "Please use the `AutocastKwargs` class instead and pass it to the `Accelerator` as a `kwarg_handler`.",
                FutureWarning,
            )
            if self.autocast_handler is not None:
                self.autocast_handler.cache_enabled = True
            else:
                self.autocast_handler = AutocastKwargs(cache_enabled=True)
        if autocast_handler is None:
            # By default `self.autocast_handler` enables autocast if:
            #   - `self.state.mixed_precision == "bf16"`
            #   - `self.state.autocast_backend is AutocastBackend.AMP`
            autocast_handler = self.autocast_handler

        if autocast_handler.enabled:
            autocast_kwargs = autocast_handler.to_kwargs()
            autocast_context = torch.autocast(dtype=torch.bfloat16, device_type="cuda", **autocast_kwargs)
        else:
            autocast_context = contextlib.nullcontext()

        autocast_context.__enter__()
        yield
        autocast_context.__exit__(*sys.exc_info())

    def clip_grad_norm_(self, parameters, max_norm, norm_type=2, postpone_clipping_to_optimizer_step: bool = False):
        if postpone_clipping_to_optimizer_step:
            parameters = list(parameters)
            if len(self._optimizers) > 1:
                raise RuntimeError(
                    "Postponing gradient clipping to the optimizer step is not possible when multiple optimizer were "
                    "prepared by the NeuronAccelerator."
                )
            self._optimizers[0].prepare_clip_grad_norm(parameters, max_norm, norm_type=norm_type)
        else:
            return super().clip_grad_norm_(parameters, max_norm, norm_type=norm_type)

    def _custom_save_state(
        self,
        save_model_func: Callable[["Accelerator", "PreTrainedModel", str | Path | None | int, Any], None],
        save_optimizer_func: Callable[
            ["Accelerator", "torch.optim.Optimizer", "PreTrainedModel", str | Path, int], Any
        ],
        output_dir: str | None = None,
        safe_serialization: bool = True,
        **save_model_func_kwargs: Any,
    ) -> str:
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

        # Save the models
        if save_model_func is not None:
            for i, model in enumerate(self._models):
                save_model_func(self, model, output_dir, i)

        # Save the optimizers
        if not self._optimizers and save_model_func is None:
            optimizers = [None] * len(self._models)
        else:
            optimizers = self._optimizers
        for i, opt in enumerate(optimizers):
            save_optimizer_func(self, opt, self._models[i], output_dir, i)

        # Save the lr schedulers taking care of DeepSpeed nuances
        schedulers = self._schedulers

        # Save the samplers of the dataloaders
        dataloaders = self._dataloaders

        # Setting those to be empty list so that `save_accelerator_state` does not redo the job.
        weights = []
        optimizers = []

        # Call model loading hooks that might have been registered with
        # accelerator.register_model_state_hook
        for hook in self._save_model_state_pre_hook.values():
            hook(self._models, weights, output_dir)

        save_location = save_accelerator_state(
            output_dir,
            weights,
            optimizers,
            schedulers,
            dataloaders,
            self.state.process_index,
            self.scaler,
            save_on_each_node=self.project_configuration.save_on_each_node,
            safe_serialization=safe_serialization,
        )
        for i, obj in enumerate(self._custom_objects):
            save_custom_state(obj, output_dir, i, save_on_each_node=self.project_configuration.save_on_each_node)
        self.project_configuration.iteration += 1
        return save_location

    def save_state_for_mp(self, output_dir: str | None = None, **save_model_func_kwargs):
        # The model is saved at the same time as the optimizer.
        save_model_func = None

        def save_optimizer_func(accelerator, optimizer, model, output_dir, i):
            # TODO: can it be cleaned?
            logger.info("Saving parallel model and optimizer")
            model.save_pretrained(output_dir, optimizer=optimizer)
            logger.info(f"Parallel model and optimizer saved to the directory {output_dir}")

        return self._custom_save_state(
            save_model_func,
            save_optimizer_func,
            output_dir=output_dir,
            safe_serialization=False,
            **save_model_func_kwargs,
        )

    def save_state(
        self, output_dir: str | None = None, safe_serialization: bool = True, **save_model_func_kwargs
    ) -> str:
        if self.distributed_type is NeuronDistributedType.MODEL_PARALLELISM:
            return self.save_state_for_mp(output_dir=output_dir, **save_model_func_kwargs)
        return super().save_state(
            output_dir=output_dir, safe_serialization=safe_serialization, **save_model_func_kwargs
        )

    def gather(self, tensor, out_of_graph: bool = False):
        return _xla_gather(tensor, out_of_graph=out_of_graph)

    def gather_for_metrics(self, input_data):
        try:
            recursively_apply(lambda x: x, input_data, error_on_other_type=True)
            all_tensors = True
        except TypeError:
            all_tensors = False

        if not all_tensors:
            data = gather_object(input_data)
        else:
            # It is needed to perform out-of-graph gather otherwise re-compilation happens at every evaluation step.
            data = self.gather(input_data, out_of_graph=True)

        try:
            if self.gradient_state.end_of_dataloader:
                # at the end of a dataloader, `gather_for_metrics` regresses to
                # `gather` unless the dataset has a remainder so log.
                if self.gradient_state.remainder == -1:
                    logger.info(
                        "The used dataset had no length, returning gathered tensors. You should drop the remainder yourself."
                    )
                    return data
                elif self.gradient_state.remainder > 0:
                    # Last batch needs to be truncated on distributed systems as it contains additional samples
                    def _adjust_samples(tensor):
                        return tensor[: self.gradient_state.remainder]

                    return recursively_apply(_adjust_samples, data)
                else:  # remainder is 0
                    # no remainder even though at end of dataloader, so nothing to do.
                    return data
            else:
                # Not at the end of the dataloader, no need to adjust the tensors
                return data
        except Exception:
            # Dataset had no length or raised an error
            return data
