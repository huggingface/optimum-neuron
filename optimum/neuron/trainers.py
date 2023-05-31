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
"""Defines Trainer subclasses to perform training on AWS Trainium instances."""

import functools
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import GenerationMixin, Seq2SeqTrainer, Trainer
from transformers.dependency_versions_check import dep_version_check
from transformers.integrations import is_fairscale_available
from transformers.modeling_utils import unwrap_model
from transformers.trainer_pt_utils import get_module_class_from_name
from transformers.trainer_utils import ShardedDDPOption
from transformers.training_args import ParallelMode
from transformers.utils import is_apex_available, is_sagemaker_dp_enabled, is_sagemaker_mp_enabled

from ..utils import logging
from .accelerator import TrainiumAccelerator
from .generation import NeuronGenerationMixin
from .trainer_callback import NeuronCacheCallaback
from .utils import is_neuronx_available
from .utils.argument_utils import validate_arg
from .utils.cache_utils import get_neuron_cache_path
from .utils.training_utils import (
    FirstAndLastDataset,
    is_model_officially_supported,
    is_precompilation,
    prepare_environment_for_neuron,
)


if TYPE_CHECKING:
    from transformers import PreTrainedModel, TrainingArguments

if is_apex_available():
    from apex import amp

if is_neuronx_available():
    import torch_xla.core.xla_model as xm

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_fairscale_available():
    dep_version_check("fairscale")
    from fairscale.nn.data_parallel import FullyShardedDataParallel as FullyShardedDDP
    from fairscale.nn.data_parallel import ShardedDataParallel as ShardedDDP
    from fairscale.nn.wrap import auto_wrap


logger = logging.get_logger("transformers.trainer")

KEEP_HF_HUB_PROGRESS_BARS = os.environ.get("KEEP_HF_HUB_PROGRESS_BARS")
if KEEP_HF_HUB_PROGRESS_BARS is None:
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Used for torch.distributed.
_ORIGINAL_NEURON_CACHE_PATH: Optional[Path] = None
_TMP_NEURON_CACHE_DIR: Optional[TemporaryDirectory] = None

if os.environ.get("TORCHELASTIC_RUN_ID"):
    import torch_xla.distributed.xla_backend as xbn

    if not isinstance(torch.distributed.group.WORLD, xbn.ProcessGroupXla):
        _ORIGINAL_NEURON_CACHE_PATH = get_neuron_cache_path()
        _TMP_NEURON_CACHE_DIR = NeuronCacheCallaback.create_temporary_neuron_cache(get_neuron_cache_path())
        torch.distributed.init_process_group(backend="xla")
        if not isinstance(torch.distributed.group.WORLD, xbn.ProcessGroupXla):
            raise AssertionError("Failed to initialize torch.distributed process group using XLA backend.")


class AugmentTrainerForTrainiumMixin:
    def __init__(self, *args, **kwargs):
        if not isinstance(self, Trainer):
            raise TypeError(f"{self.__class__.__name__} can only be mixed with Trainer subclasses.")

        training_args = kwargs.get("args", None)
        if training_args is None and len(args) >= 2:
            training_args = args[1]

        if training_args is not None:
            if training_args.bf16:
                training_args.bf16 = False
                os.environ["XLA_USE_BF16"] = "1"

        self.validate_args(training_args)
        if is_precompilation():
            self.prepare_args_for_precompilation(training_args)

        prepare_environment_for_neuron()
        super().__init__(*args, **kwargs)

        self.accelerator = TrainiumAccelerator(
            deepspeed_plugin=self.args.deepspeed_plugin,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
        )

        if self.args.local_rank <= 0:
            logger.setLevel(logging.INFO)

        if not is_precompilation():
            callback = NeuronCacheCallaback(
                tmp_neuron_cache=_TMP_NEURON_CACHE_DIR,
                original_neuron_cache_path=_ORIGINAL_NEURON_CACHE_PATH,
                only_do_fetching=self.args.local_rank > 0,
            )
            self.add_callback(callback)

        # Make the model Neuron-compatible for generation.
        self.patch_generation_mixin_to_neuron_generation_mixin(self.model)

    def prepare_args_for_precompilation(self, args: "TrainingArguments"):
        if args.num_train_epochs != 1:
            logger.info("Setting the number of epochs for precompilation to 1.")
            args.num_train_epochs = 1
        if args.max_steps is not None:
            logger.info("Disabling max_steps for precompilation.")
            args.nax_steps = None
        if args.do_eval is True:
            logger.info("Disabling evaluation during precompilation as this is not well supported yet.")
            args.do_eval = False
        if args.do_predict is True:
            logger.info("Disabling prediction during precompilation as this is not well supported yet.")
            args.do_predict = False

    def validate_args(self, args: "TrainingArguments"):
        if isinstance(self, Seq2SeqTrainer):
            validate_arg(
                args,
                "prediction_loss_only",
                "prediction_loss_only=False is not supported for now because it requires generation.",
                expected_value=True,
            )

    def patch_generation_mixin_to_neuron_generation_mixin(self, model: "PreTrainedModel"):
        """
        Changes the vanilla `GenerationMixin` class from Transformers to `NeuronGenerationMixin` in the model's
        inheritance. This allows to make the model Neuron-compatible for generation without much hassle.
        """
        to_visit = [model.__class__]
        should_stop = False
        while to_visit and not should_stop:
            cls = to_visit.pop(0)
            bases = cls.__bases__
            new_bases = []
            for base in bases:
                to_visit.append(base)
                if base == GenerationMixin:
                    new_bases.append(NeuronGenerationMixin)
                    should_stop = True
                elif base == NeuronGenerationMixin:
                    should_stop = True
                    new_bases.append(base)
                else:
                    new_bases.append(base)
            cls.__bases__ = tuple(new_bases)

    def _wrap_model(self, model, training=True, dataloader=None):
        if not is_model_officially_supported(model):
            logger.warning(
                f"{model.__class__.__name__} is not officially supported by optimum-neuron. Training might not work as  "
                "expected."
            )

        if self.args.use_ipex:
            dtype = torch.bfloat16 if self.use_cpu_amp else torch.float32
            model = self.ipex_optimize_model(model, training, dtype=dtype)

        if is_sagemaker_mp_enabled():
            # Wrapping the base model twice in a DistributedModel will raise an error.
            if isinstance(self.model_wrapped, smp.model.DistributedModel):
                return self.model_wrapped
            return smp.DistributedModel(model, backward_passes_per_step=self.args.gradient_accumulation_steps)

        # train/eval could be run multiple-times - if already wrapped, don't re-wrap it again
        if unwrap_model(model) is not model:
            return model

        # Mixed precision training with apex (torch < 1.6)
        if self.use_apex and training:
            model, self.optimizer = amp.initialize(model, self.optimizer, opt_level=self.args.fp16_opt_level)

        # Multi-gpu training (should be after apex fp16 initialization) / 8bit models does not support DDP
        # if self.args.n_gpu > 1 and not getattr(model, "is_loaded_in_8bit", False):
        #     model = nn.DataParallel(model)

        if self.args.jit_mode_eval:
            start_time = time.time()
            model = self.torch_jit_model_eval(model, dataloader, training)
            self.jit_compilation_time = round(time.time() - start_time, 4)

        # Note: in torch.distributed mode, there's no point in wrapping the model
        # inside a DistributedDataParallel as we'll be under `no_grad` anyways.
        if not training:
            return model

        # Distributed training (should be after apex fp16 initialization)
        if self.sharded_ddp is not None:
            # Sharded DDP!
            if self.sharded_ddp == ShardedDDPOption.SIMPLE:
                model = ShardedDDP(model, self.optimizer)
            else:
                mixed_precision = self.args.fp16 or self.args.bf16
                cpu_offload = ShardedDDPOption.OFFLOAD in self.args.sharded_ddp
                zero_3 = self.sharded_ddp == ShardedDDPOption.ZERO_DP_3
                # XXX: Breaking the self.model convention but I see no way around it for now.
                if ShardedDDPOption.AUTO_WRAP in self.args.sharded_ddp:
                    model = auto_wrap(model)
                self.model = model = FullyShardedDDP(
                    model,
                    mixed_precision=mixed_precision,
                    reshard_after_forward=zero_3,
                    cpu_offload=cpu_offload,
                ).to(self.args.device)
        # Distributed training using PyTorch FSDP
        # TODO: should we try for self.args.fsdp["xla"] or just do it?
        elif self.fsdp is not None and self.args.fsdp_config["xla"]:
            try:
                from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
                from torch_xla.distributed.fsdp import checkpoint_module
                from torch_xla.distributed.fsdp.wrap import (
                    size_based_auto_wrap_policy,
                    transformer_auto_wrap_policy,
                )
            except ImportError:
                raise ImportError("Missing XLA FSDP related module; please make sure to use torch-xla >= 2.0.")
            auto_wrap_policy = None
            auto_wrapper_callable = None
            if self.args.fsdp_config["fsdp_min_num_params"] > 0:
                auto_wrap_policy = functools.partial(
                    size_based_auto_wrap_policy, min_num_params=self.args.fsdp_config["fsdp_min_num_params"]
                )
            elif self.args.fsdp_config.get("fsdp_transformer_layer_cls_to_wrap", None) is not None:
                transformer_cls_to_wrap = set()
                for layer_class in self.args.fsdp_config["fsdp_transformer_layer_cls_to_wrap"]:
                    transformer_cls = get_module_class_from_name(model, layer_class)
                    if transformer_cls is None:
                        raise Exception("Could not find the transformer layer class to wrap in the model.")
                    else:
                        transformer_cls_to_wrap.add(transformer_cls)
                auto_wrap_policy = functools.partial(
                    transformer_auto_wrap_policy,
                    # Transformer layer class to wrap
                    transformer_layer_cls=transformer_cls_to_wrap,
                )
            fsdp_kwargs = self.args.xla_fsdp_config
            if self.args.fsdp_config["xla_fsdp_grad_ckpt"]:
                # Apply gradient checkpointing to auto-wrapped sub-modules if specified
                def auto_wrapper_callable(m, *args, **kwargs):
                    return FSDP(checkpoint_module(m), *args, **kwargs)

            # Wrap the base model with an outer FSDP wrapper
            self.model = model = FSDP(
                model,
                auto_wrap_policy=auto_wrap_policy,
                auto_wrapper_callable=auto_wrapper_callable,
                **fsdp_kwargs,
            )

            # Patch `xm.optimizer_step` should not reduce gradients in this case,
            # as FSDP does not need gradient reduction over sharded parameters.
            def patched_optimizer_step(optimizer, barrier=False, optimizer_args={}):
                loss = optimizer.step(**optimizer_args)
                if barrier:
                    xm.mark_step()
                return loss

            xm.optimizer_step = patched_optimizer_step
        elif is_sagemaker_dp_enabled():
            model = nn.parallel.DistributedDataParallel(
                model, device_ids=[int(os.getenv("SMDATAPARALLEL_LOCAL_RANK"))]
            )
        elif self.args.parallel_mode == ParallelMode.DISTRIBUTED:
            return model
            # TODO: not supported for now?
            # kwargs = {}
            # if self.args.ddp_find_unused_parameters is not None:
            #     kwargs["find_unused_parameters"] = self.args.ddp_find_unused_parameters
            # elif isinstance(model, PreTrainedModel):
            #     # find_unused_parameters breaks checkpointing as per
            #     # https://github.com/huggingface/transformers/pull/4659#issuecomment-643356021
            #     kwargs["find_unused_parameters"] = not model.is_gradient_checkpointing
            # else:
            #     kwargs["find_unused_parameters"] = True

            # if self.args.ddp_bucket_cap_mb is not None:
            #     kwargs["bucket_cap_mb"] = self.args.ddp_bucket_cap_mb

            # self.accelerator.ddp_handler = DistributedDataParallelKwargs(**kwargs)

        return model

    def get_train_dataloader(self) -> DataLoader:
        if is_precompilation():
            return DataLoader(
                FirstAndLastDataset(
                    super().get_train_dataloader(),
                    gradient_accumulation_steps=self.args.gradient_accumulation_steps,
                    world_size=self.args.world_size,
                ),
                batch_size=None,
            )
        return super().get_train_dataloader()

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:
        if is_precompilation():
            return DataLoader(
                FirstAndLastDataset(
                    super().get_eval_dataloader(eval_dataset=eval_dataset), world_size=self.args.world_size
                ),
                batch_size=None,
            )
        return super().get_eval_dataloader(eval_dataset=eval_dataset)

    def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:
        if is_precompilation():
            return DataLoader(
                FirstAndLastDataset(
                    super().get_test_dataloader(test_dataset),
                    world_size=self.args.world_size,
                ),
                batch_size=None,
            )
        return super().get_test_dataloader(test_dataset)

    # TODO: make this cleaner.
    def trigger_on_step_middle_for_neuron_cache_callback(self, model: "PreTrainedModel"):
        for callback in self.callback_handler.callbacks:
            if isinstance(callback, NeuronCacheCallaback):
                # kwargs might not have everything expected (like metrics) but all we need is here.
                kwargs = {
                    "model": model,
                    "tokenizer": self.tokenizer,
                    "optimizer": self.optimizer,
                    "lr_scheduler": self.lr_scheduler,
                    "train_dataloader": self.callback_handler.train_dataloader,
                    "eval_dataloader": self.callback_handler.eval_dataloader,
                }
                callback.on_step_middle(self.args, self.state, self.control, **kwargs)

    def compute_loss(self, model, inputs, return_outputs: bool = False):
        self.state.last_inputs = inputs
        self.trigger_on_step_middle_for_neuron_cache_callback(model)
        return super().compute_loss(model, inputs, return_outputs=return_outputs)

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        self.state.last_inputs = inputs
        self.trigger_on_step_middle_for_neuron_cache_callback(model)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)


class TrainiumTrainer(AugmentTrainerForTrainiumMixin, Trainer):
    """
    Trainer that is suited for performing training on AWS Tranium instances.
    """


class Seq2SeqTrainiumTrainer(AugmentTrainerForTrainiumMixin, Seq2SeqTrainer):
    """
    Seq2SeqTrainer that is suited for performing training on AWS Tranium instances.
    """
