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

import collections
import contextlib
import inspect
import os
import re
import shutil
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple, Union

import torch
from accelerate import Accelerator
from accelerate.checkpointing import save_accelerator_state, save_custom_state
from accelerate.utils import DistributedType
from accelerate.utils.operations import gather_object, recursively_apply
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from ...utils import logging
from ..distributed import Parallelizer, ParallelizersManager
from ..utils import (
    DynamicPatch,
    ModelPatcher,
    Patcher,
    is_neuronx_distributed_available,
    is_torch_xla_available,
    patch_within_function,
    patched_finfo,
)
from ..utils.misc import args_and_kwargs_to_kwargs_only
from ..utils.require_utils import requires_neuronx_distributed, requires_torch_xla
from .optimizer import NeuronAcceleratedOptimizer
from .scheduler import NeuronAcceleratedScheduler
from .state import NeuronAcceleratorState
from .utils import (
    ModelParallelismPlugin,
    NeuronDistributedType,
    NeuronFullyShardedDataParallelPlugin,
    get_tied_parameters_dict,
    patch_accelerate_is_tpu_available,
    tie_parameters,
)
from .utils.operations import _xla_gather


if TYPE_CHECKING:
    from transformers import PreTrainedModel

    try:
        from torch.optim.lr_scheduler import LRScheduler
    except ImportError:
        from torch.optim.lr_scheduler import _LRScheduler as LRScheduler

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    from torch_xla.distributed.parallel_loader import MpDeviceLoader
else:
    xm = None

if is_neuronx_distributed_available():
    from neuronx_distributed.utils.model_utils import move_model_to_device


logger = logging.get_logger(__name__)


MODEL_PATCHING_SPECS = [
    ("config.layerdrop", 0),
    ("no_sync", lambda: contextlib.nullcontext()),
    (
        "forward",
        DynamicPatch(patch_within_function(("torch.finfo", patched_finfo))),
    ),
]

NxDPPMODEL_PATCHING_SPECS = [
    (
        "forward",
        DynamicPatch(patch_within_function(("torch.finfo", patched_finfo))),
    ),
]


class NeuronAccelerator(Accelerator):
    def __init__(self, *args, mp_plugin: Optional[ModelParallelismPlugin] = None, zero_1: bool = False, **kwargs):
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
        if fsdp_plugin is None:
            if os.environ.get("ACCELERATE_USE_FSDP", "false") == "true":
                fsdp_plugin = NeuronFullyShardedDataParallelPlugin()
        elif not isinstance(fsdp_plugin, NeuronFullyShardedDataParallelPlugin):
            raise ValueError(
                "The fsdp_plugin must be an instance of NeuronFullyShardedDataParallelPlugin to use XLA FSDP with "
                f"the NeuronAccelerator, but an instance of {type(fsdp_plugin)} was given here."
            )
        self.fsdp_plugin = fsdp_plugin

        use_neuronx_distributed_tp = os.environ.get("ACCELERATE_USE_NEURONX_DISTRIBUTED_TP", "false")
        use_neuronx_distributed_pp = os.environ.get("ACCELERATE_USE_NEURONX_DISTRIBUTED_PP", "false")
        if mp_plugin is None:
            if use_neuronx_distributed_tp == "false":
                tp_size = 1
            else:
                tp_size = int(use_neuronx_distributed_tp)
            if use_neuronx_distributed_pp == "false":
                pp_size = 1
            else:
                pp_size = int(use_neuronx_distributed_pp)
            mp_plugin = ModelParallelismPlugin(
                tensor_parallel_size=tp_size, parallelize_embeddings=True, pipeline_parallel_size=pp_size
            )
        self._model_cpu_parameters_to_xla = {}

        if mp_plugin.tensor_parallel_size > 1:
            os.environ["ACCELERATE_USE_NEURONX_DISTRIBUTED_TP"] = "true"

        if mp_plugin.pipeline_parallel_size > 1:
            os.environ["ACCELERATE_USE_NEURONX_DISTRIBUTED_PP"] = "true"

        patched_accelerator_state = partial(NeuronAcceleratorState, mp_plugin=mp_plugin)
        with Patcher([("accelerate.accelerator.AcceleratorState", patched_accelerator_state)]):
            super().__init__(**full_kwargs)

        self.zero_1 = zero_1

        if self.fsdp_plugin is not None and self.zero_1:
            raise ValueError("Either enable XLA ZeRO Stage 1 or XLA FSDP but not both.")

        if self.process_index == -1 and self.zero_1:
            raise ValueError("XLA ZeRO Stage 1 can only be enabled in a distributed training setting.")

        if fsdp_plugin is not None and mp_plugin is not None:
            raise ValueError("It is not possible to both use neuronx_distributed Tensor Parallelism and XLA FSDP.")

        if num_steps != 1:
            self.gradient_accumulation_steps = num_steps

    def _prepare_data_loader_for_distributed(
        self, data_loader: DataLoader, num_replicas: int, rank: int
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
            drop_last=data_loader.drop_last,
        )
        distributed_dataloader._is_accelerate_prepared = True
        return distributed_dataloader

    def prepare_data_loader(self, data_loader: DataLoader, device_placement: Optional[bool] = None):
        if self.state.distributed_type is NeuronDistributedType.MODEL_PARALLELISM:
            from neuronx_distributed import parallel_layers

            num_replicas = parallel_layers.parallel_state.get_data_parallel_size()
            rank = parallel_layers.parallel_state.get_data_parallel_rank()
        else:
            num_replicas = xm.xrt_world_size()
            rank = xm.get_ordinal()
        if self.state.num_processes > 1:
            data_loader = self._prepare_data_loader_for_distributed(data_loader, num_replicas=num_replicas, rank=rank)
            # No need to wrap the dataloader if we are using pipeline parallelism.
            if self.state.mp_plugin.pipeline_parallel_size == 1:
                data_loader = MpDeviceLoader(data_loader, self.device)
        return data_loader
        # TODO: fix that.
        # return super().prepare_data_loader(data_loader, device_placement=device_placement)

    def _prepare_optimizer_for_mp(self, optimizer: torch.optim.Optimizer, device_placement=None):
        cpu_parameters_to_xla = collections.ChainMap(*self._model_cpu_parameters_to_xla.values())
        if not self.zero_1:
            optimizer = Parallelizer.optimizer_for_mp(optimizer, cpu_parameters_to_xla)
        else:
            xla_parameters, _ = Parallelizer.optimizer_cpu_params_to_xla_params(optimizer, cpu_parameters_to_xla)
            if hasattr(optimizer, "_args_to_recreate"):
                args, kwargs = optimizer._args_to_recreate
                args = (xla_parameters,) + args[1:]
                optimizer._args_to_recreate = (args, kwargs)
            else:
                optimizer.param_groups = xla_parameters
        return optimizer

    @requires_neuronx_distributed
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

        if hasattr(optimizer, "_args_to_recreate"):
            args, kwargs = optimizer._args_to_recreate
            params = args[0]
            defaults = args_and_kwargs_to_kwargs_only(optimizer.__class__, args[1:], kwargs)

            zero_1_optimizer = NeuronZero1Optimizer(
                params,
                optimizer.__class__,
                optimizer_dtype=optimizer_dtype,
                pin_layout=False,
                sharding_groups=sharding_groups,
                grad_norm_groups=grad_norm_groups,
                **defaults,
            )
            del optimizer
        else:
            logger.warning(
                f"Creating a NeuronZero1Optimizer from {optimizer}, this might change some default values. When "
                "using ZeRO 1 it is recommended to create the ZeroRedundancyOptimizer yourself to avoid this kind of "
                "issues."
            )
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
    def prepare_optimizer(self, optimizer: torch.optim.Optimizer, device_placement: Optional[bool] = None):
        if self.distributed_type is NeuronDistributedType.MODEL_PARALLELISM:
            optimizer = self._prepare_optimizer_for_mp(optimizer, device_placement=device_placement)
        if self.zero_1:
            optimizer = self._prepare_optimizer_for_zero_1(optimizer, device_placement=device_placement)
        # Edge case: if the optimizer was created lazily outside of the Model Parallelism and/or ZeRO-1 setting, we make
        # sure to actually load the proper parameters.
        if hasattr(optimizer, "_args_to_recreate"):
            args, kwargs = optimizer._args_to_recreate
            optimizer = optimizer.__class__(*args, **kwargs)

        return super().prepare_optimizer(optimizer, device_placement=device_placement)

    @patch_within_function(("accelerate.accelerator.AcceleratedScheduler", NeuronAcceleratedScheduler))
    def prepare_scheduler(self, scheduler: "LRScheduler"):
        return super().prepare_scheduler(scheduler)

    @staticmethod
    def patch_model_for_neuron(
        model: "torch.nn.Module", patching_specs: Optional[List[Tuple[str, Any]]] = None
    ) -> "torch.nn.Module":
        if patching_specs is None:
            patching_specs = MODEL_PATCHING_SPECS
        prepared_patching_specs = []
        for spec in patching_specs:
            prepared_patching_specs.append((model,) + spec)

        model_patcher = ModelPatcher(prepared_patching_specs, ignore_missing_attributes=True)
        model_patcher.patch()
        return model

    def prepare_model_for_xla_fsdp(
        self, model: torch.nn.Module, device_placement: Optional[bool] = None, evaluation_mode: bool = False
    ):
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

        if not evaluation_mode:
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

    @requires_neuronx_distributed
    def _prepare_model_for_mp(
        self, model: torch.nn.Module, device_placement: Optional[bool] = None, evaluation_mode: bool = False
    ):
        from neuronx_distributed.pipeline import NxDPPModel

        if model in self._models or Parallelizer.was_parallelized(model):
            return model

        cpu_ids = {name: id(param) for name, param in model.named_parameters()}
        tied_parameters_dict = get_tied_parameters_dict(model)
        model_main_input_name = getattr(model, "main_input_name", None)
        # TODO: enable self.device (if needed).
        model = self.state.mp_plugin.parallelize_model(model, device=None)

        if model_main_input_name is not None:
            setattr(model, "main_input_name", model_main_input_name)

        if isinstance(model, NxDPPModel):
            model.local_module = self.patch_model_for_neuron(
                model.local_module, patching_specs=NxDPPMODEL_PATCHING_SPECS
            )
            model_to_cast = model.local_module
        else:
            model_to_cast = model

        model_to_cast = model.local_module if isinstance(model, NxDPPModel) else model
        if os.environ.get("XLA_USE_BF16", "0") == "1" or os.environ.get("XLA_DOWNCAST_BF16", "0") == "1":
            model_to_cast.to(torch.bfloat16)
        else:
            model_to_cast.to(torch.float32)

        def _tie_or_clone_weights_for_mp(self, output_embeddings, input_embeddings):
            """Tie or clone module weights depending of whether we are using TorchScript or not"""
            output_embeddings.weight = input_embeddings.weight
            if hasattr(output_embeddings, "out_features") and hasattr(input_embeddings, "num_embeddings"):
                output_embeddings.out_features = input_embeddings.num_embeddings

        if isinstance(model, NxDPPModel):
            with ModelPatcher(patching_specs=[(model, "_tie_or_clone_weights", _tie_or_clone_weights_for_mp)]):
                model.move_model_to_device()
                tie_parameters(model, tied_parameters_dict)
            xla_params = dict(model.local_named_parameters())
            self._model_cpu_parameters_to_xla[id(model)] = {
                cpu_ids[name]: xla_params[name] for name, _ in model.local_named_parameters()
            }
        else:
            with ModelPatcher(patching_specs=[(model, "_tie_or_clone_weights", _tie_or_clone_weights_for_mp)]):
                move_model_to_device(model, self.device)
                tie_parameters(model, tied_parameters_dict)
            xla_params = dict(model.named_parameters())
            symmetric_diff = set(cpu_ids.keys()).symmetric_difference((xla_params.keys()))
            if symmetric_diff:
                raise ValueError(
                    f"The parameters on CPU do not match the parameters on the XLA device: {', '.join(symmetric_diff)}."
                )

            self._model_cpu_parameters_to_xla[id(model)] = {
                cpu_ids[name]: xla_params[name] for name, _ in model.named_parameters()
            }

        device_placement = False

        return super().prepare_model(model, device_placement=device_placement, evaluation_mode=evaluation_mode)

    @requires_torch_xla
    @requires_neuronx_distributed
    def prepare_model(
        self, model: torch.nn.Module, device_placement: Optional[bool] = None, evaluation_mode: bool = False
    ):
        # If the model was already prepared, we skip.
        if model in self._models:
            return model

        model = self.patch_model_for_neuron(model)

        if self.distributed_type is NeuronDistributedType.XLA_FSDP:
            return self.prepare_model_for_xla_fsdp(
                model, device_placement=device_placement, evaluation_mode=evaluation_mode
            )
        elif self.distributed_type is NeuronDistributedType.MODEL_PARALLELISM:
            return self._prepare_model_for_mp(
                model, device_placement=device_placement, evaluation_mode=evaluation_mode
            )
        move_model_to_device(model, xm.xla_device())
        device_placement = False
        return super().prepare_model(model, device_placement=device_placement, evaluation_mode=evaluation_mode)

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
            loss.backward(**kwargs)

    def clip_grad_norm_for_xla_fsdp(self, parameters, max_norm, norm_type: int = 2):
        self.unscale_gradients()
        parameters = list(parameters)
        for model in self._models:
            if parameters == list(model.parameters()):
                return model.clip_grad_norm_(max_norm, norm_type)

    @requires_neuronx_distributed
    def _prepare_clip_grad_norm(self, parameters, max_norm, norm_type: int = 2):
        from neuronx_distributed.pipeline import NxDPPModel

        self.unscale_gradients()
        parameters = list(parameters)
        for model in self._models:
            model_parameters = model.local_parameters() if isinstance(model, NxDPPModel) else model.parameters()
            if parameters == list(model_parameters) or self.zero_1:
                for opt in self._optimizers:
                    # Under this setting, the gradient clipping will be deferred to the optimizer step.
                    # It will happen after the gradients have been reduced and before the optimizer step.
                    return opt.prepare_clip_grad_norm(parameters, max_norm, norm_type=norm_type)

    def clip_grad_norm_(self, parameters, max_norm, norm_type=2):
        if self.distributed_type is NeuronDistributedType.XLA_FSDP:
            return self.clip_grad_norm_for_xla_fsdp(parameters, max_norm, norm_type=norm_type)
        elif self.distributed_type is NeuronDistributedType.MODEL_PARALLELISM or self.zero_1:
            return self._prepare_clip_grad_norm(parameters, max_norm, norm_type=norm_type)
        return super().clip_grad_norm_(parameters, max_norm, norm_type=norm_type)

    def clip_grad_value_(self, parameters, clip_value):
        if self.distributed_type is NeuronDistributedType.XLA_FSDP:
            raise Exception("XLA FSDP  does not support `clip_grad_value_`. Use `clip_grad_norm_` instead.")
        return super().clip_grad_value_(parameters, clip_value)

    def _custom_save_state(
        self,
        save_model_func: Optional[Callable[["Accelerator", "PreTrainedModel", Union[str, Path], int], Any]],
        save_optimizer_func: Callable[
            ["Accelerator", "torch.optim.Optimizer", "PreTrainedModel", Union[str, Path], int], Any
        ],
        output_dir: Optional[str] = None,
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

        # Setting those to be empty list so that `save_accelerator_state` does not redo the job.
        weights = []
        optimizers = []

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

    def save_state_for_xla_fsdp(self, output_dir: Optional[str] = None, **save_model_func_kwargs):
        def save_model_func(accelelerator, model, output_dir, i):
            logger.info("Saving FSDP model")
            self.state.fsdp_plugin.save_model(accelelerator, model, output_dir, i)
            logger.info(f"FSDP Model saved to the directory {output_dir}")

        def save_optimizer_func(accelerator, optimizer, model, output_dir, i):
            logger.info("Saving FSDP Optimizer")
            self.state.fsdp_plugin.save_optimizer(accelerator, optimizer, model, output_dir, i)
            logger.info(f"FSDP Optimizer saved to the directory {output_dir}")

        return self._custom_save_state(
            save_model_func, save_optimizer_func, output_dir=output_dir, **save_model_func_kwargs
        )

    def save_state_for_mp(self, output_dir: Optional[str] = None, **save_model_func_kwargs):
        # The model is saved at the same time as the optimizer.
        save_model_func = None

        def save_optimizer_func(accelerator, optimizer, model, output_dir, i):
            logger.info("Saving parallel model and optimizer")
            parallelizer = ParallelizersManager.parallelizer_for_model(model)
            parallelizer.save_model_checkpoint(model, output_dir, as_regular=False, optimizer=optimizer)
            logger.info(f"Parallel model and optimizer saved to the directory {output_dir}")

        return self._custom_save_state(
            save_model_func, save_optimizer_func, output_dir=output_dir, **save_model_func_kwargs
        )

    @patch_within_function(("accelerate.checkpointing.xm", xm), ignore_missing_attributes=True)
    def save_state(self, output_dir: Optional[str] = None, **save_model_func_kwargs) -> str:
        if self.distributed_type is NeuronDistributedType.XLA_FSDP:
            return self.save_state_for_xla_fsdp(output_dir=output_dir, **save_model_func_kwargs)
        elif self.distributed_type is NeuronDistributedType.MODEL_PARALLELISM:
            return self.save_state_for_mp(output_dir=output_dir, **save_model_func_kwargs)
        return super().save_state(output_dir=output_dir, **save_model_func_kwargs)

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
