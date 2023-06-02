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
import glob
import sys
import math
import os
import random
import shutil
import time
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from accelerate import skip_first_batches as accelerate_skip_first_batches
from packaging import version
from torch import nn
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import GenerationMixin, Seq2SeqTrainer, Trainer
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init, deepspeed_load_checkpoint
from transformers.dependency_versions_check import dep_version_check
from transformers.integrations import hp_params, is_fairscale_available
from transformers.modeling_utils import unwrap_model
from transformers.pytorch_utils import is_torch_less_than_1_11
from transformers.trainer import (
    OPTIMIZER_NAME,
    SCALER_NAME,
    SCHEDULER_NAME,
    TRAINER_STATE_NAME,
    TRAINING_ARGS_NAME,
    WEIGHTS_NAME,
)
from transformers.trainer_callback import TrainerState
from transformers.trainer_pt_utils import get_model_param_count, get_module_class_from_name, reissue_pt_warnings
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, ShardedDDPOption, TrainOutput, has_length, HPSearchBackend
from transformers.training_args import ParallelMode
from transformers.utils import (
    is_apex_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_torch_tpu_available,
)

from ..utils import check_if_transformers_greater, logging
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
    import torch_xla.distributed.parallel_loader as pl
    from torch_xla.distributed.fsdp.state_dict_utils import consolidate_sharded_model_checkpoints

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


# Used for FSDP
RANK_PREFIX = "rank"
SHARD_METADATA_NAME = "shard_metadata.bin"

if os.environ.get("TORCHELASTIC_RUN_ID"):
    import torch_xla.distributed.xla_backend as xbn

    if not isinstance(torch.distributed.group.WORLD, xbn.ProcessGroupXla):
        _ORIGINAL_NEURON_CACHE_PATH = get_neuron_cache_path()
        _TMP_NEURON_CACHE_DIR = NeuronCacheCallaback.create_temporary_neuron_cache(get_neuron_cache_path())
        torch.distributed.init_process_group(backend="xla")
        if not isinstance(torch.distributed.group.WORLD, xbn.ProcessGroupXla):
            raise AssertionError("Failed to initialize torch.distributed process group using XLA backend.")


def skip_first_batches(dataloader, num_batches=0):
    if isinstance(dataloader, (pl.ParallelLoader, pl.PerDeviceLoader)):
        dataloader._loader = skip_first_batches(dataloader._loader, num_batches=num_batches)
    else:
        dataloader = accelerate_skip_first_batches(dataloader, num_batches=num_batches)
    return dataloader


def get_fsdp_checkpoint_path(output_dir):
    return os.path.join(output_dir, f"{RANK_PREFIX}-{xm.get_ordinal()}-of-{xm.xrt_world_size()}.pth")


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

        if check_if_transformers_greater("4.30"):
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

    def _save_tpu(self, output_dir: Optional[str] = None):
        # With FSDP we need to save everything together.
        if self.fsdp is not None:
            return
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        logger.info(f"Saving model checkpoint to {output_dir}")

        if xm.is_master_ordinal():
            os.makedirs(output_dir, exist_ok=True)
            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        xm.rendezvous("saving_checkpoint")
        if not isinstance(self.model, PreTrainedModel):
            if isinstance(unwrap_model(self.model), PreTrainedModel):
                unwrap_model(self.model).save_pretrained(
                    output_dir,
                    is_main_process=self.args.should_save,
                    state_dict=self.model.state_dict(),
                    save_function=xm.save,
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                state_dict = self.model.state_dict()
                xm.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            self.model.save_pretrained(output_dir, is_main_process=self.args.should_save, save_function=xm.save)
        if self.tokenizer is not None and self.args.should_save:
            self.tokenizer.save_pretrained(output_dir)

    def _save_checkpoint(self, model, trial, metrics=None):
        # In all cases, including ddp/dp/deepspeed, self.model is always a reference to the model we
        # want to save except FullyShardedDDP.
        # assert unwrap_model(model) is self.model, "internal model should be a reference to self.model"

        # Save model checkpoint
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        Path(output_dir).mkdir(exist_ok=True)
        if self.is_deepspeed_enabled:
            # under zero3 model file itself doesn't get saved since it's bogus! Unless deepspeed
            # config `stage3_gather_16bit_weights_on_model_save` is True
            self.model_wrapped.save_checkpoint(output_dir)

        # Save optimizer and scheduler
        if self.sharded_ddp == ShardedDDPOption.SIMPLE:
            self.optimizer.consolidate_state_dict()

        if self.fsdp is not None:
            if self.args.parallel_mode == ParallelMode.TPU:
                ckpt = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "shard_metadata": self.model.get_shard_metadata(),
                }
                ckpt_path = get_fsdp_checkpoint_path(output_dir)
                xm.save(ckpt, ckpt_path, master_only=False)
                xm.rendezvous("saved_sharded_checkpoint")
                if self.args.process_index == 0:
                    full_state_dict, _ = consolidate_sharded_model_checkpoints(
                        f"{output_dir}/rank-",
                        save_model=False,
                    )
                    torch.save(full_state_dict, os.path.join(output_dir, WEIGHTS_NAME))
                    with warnings.catch_warnings(record=True) as caught_warnings:
                        torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            else:
                # FSDP has a different interface for saving optimizer states.
                # Needs to be called on all ranks to gather all states.
                # full_optim_state_dict will be deprecated after Pytorch 2.2!
                full_osd = self.model.__class__.full_optim_state_dict(self.model, self.optimizer)

        if is_torch_tpu_available():
            if self.fsdp is None:
                xm.rendezvous("saving_optimizer_states")
                xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))
                with warnings.catch_warnings(record=True) as caught_warnings:
                    xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                    reissue_pt_warnings(caught_warnings)
        elif is_sagemaker_mp_enabled():
            opt_state_dict = self.optimizer.local_state_dict(gather_if_shard=False)
            smp.barrier()
            if smp.rdp_rank() == 0 or smp.state.cfg.shard_optimizer_state:
                smp.save(
                    opt_state_dict,
                    os.path.join(output_dir, OPTIMIZER_NAME),
                    partial=True,
                    v3=smp.state.cfg.shard_optimizer_state,
                )
            if self.args.should_save:
                with warnings.catch_warnings(record=True) as caught_warnings:
                    torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
                reissue_pt_warnings(caught_warnings)
                if self.do_grad_scaling:
                    torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))
        elif self.args.should_save and not self.is_deepspeed_enabled:
            # deepspeed.save_checkpoint above saves model/optim/sched
            if self.fsdp:
                torch.save(full_osd, os.path.join(output_dir, OPTIMIZER_NAME))
            else:
                torch.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))

            with warnings.catch_warnings(record=True) as caught_warnings:
                torch.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)
            if self.do_grad_scaling:
                torch.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))

        # Determine the new best metric / best model checkpoint
        if metrics is not None and self.args.metric_for_best_model is not None:
            metric_to_check = self.args.metric_for_best_model
            if not metric_to_check.startswith("eval_"):
                metric_to_check = f"eval_{metric_to_check}"
            metric_value = metrics[metric_to_check]

            operator = np.greater if self.args.greater_is_better else np.less
            if (
                self.state.best_metric is None
                or self.state.best_model_checkpoint is None
                or operator(metric_value, self.state.best_metric)
            ):
                self.state.best_metric = metric_value
                self.state.best_model_checkpoint = output_dir

        # Save the Trainer state
        if self.args.should_save:
            self.state.save_to_json(os.path.join(output_dir, TRAINER_STATE_NAME))

        # Save RNG state in non-distributed training
        rng_states = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            if self.args.parallel_mode == ParallelMode.DISTRIBUTED:
                # In non distributed, we save the global CUDA RNG state (will take care of DataParallel)
                rng_states["cuda"] = torch.cuda.random.get_rng_state_all()
            else:
                rng_states["cuda"] = torch.cuda.random.get_rng_state()

        if is_torch_tpu_available():
            rng_states["xla"] = xm.get_rng_state()

        # A process can arrive here before the process 0 has a chance to save the model, in which case output_dir may
        # not yet exist.
        os.makedirs(output_dir, exist_ok=True)

        if self.args.world_size <= 1:
            torch.save(rng_states, os.path.join(output_dir, "rng_state.pth"))
        else:
            torch.save(rng_states, os.path.join(output_dir, f"rng_state_{self.args.process_index}.pth"))

        if self.args.push_to_hub:
            self._push_from_checkpoint(output_dir)

        # Maybe delete some older checkpoints.
        if self.args.should_save:
            self._rotate_checkpoints(use_mtime=True, output_dir=run_dir)

    def _load_optimizer_and_scheduler(self, checkpoint):
        if checkpoint is None:
            return

        if self.is_deepspeed_enabled:
            # deepspeed loads optimizer/lr_scheduler together with the model in deepspeed_init
            return

        checkpoint_file_exists = (
            glob.glob(os.path.join(checkpoint, OPTIMIZER_NAME) + "_*")
            if is_sagemaker_mp_enabled()
            else os.path.isfile(os.path.join(checkpoint, OPTIMIZER_NAME))
        )
        if is_torch_tpu_available() and self.fsdp is not None:
            checkpoint_file_exists = os.path.isfile(get_fsdp_checkpoint_path(checkpoint))

        if checkpoint_file_exists and os.path.isfile(os.path.join(checkpoint, SCHEDULER_NAME)):
            # Load in optimizer and scheduler states
            if is_torch_tpu_available():
                if self.fsdp is not None:
                    state = torch.load(get_fsdp_checkpoint_path(checkpoint))
                    optimizer_state = state["optimizer"]
                else:
                    # On TPU we have to take some extra precautions to properly load the states on the right device.
                    optimizer_state = torch.load(os.path.join(checkpoint, OPTIMIZER_NAME), map_location="cpu")

                with warnings.catch_warnings(record=True) as caught_warnings:
                    lr_scheduler_state = torch.load(os.path.join(checkpoint, SCHEDULER_NAME), map_location="cpu")
                reissue_pt_warnings(caught_warnings)

                xm.send_cpu_data_to_device(optimizer_state, self.args.device)
                xm.send_cpu_data_to_device(lr_scheduler_state, self.args.device)

                self.optimizer.load_state_dict(optimizer_state)
                self.lr_scheduler.load_state_dict(lr_scheduler_state)
            else:
                if is_sagemaker_mp_enabled():
                    if os.path.isfile(os.path.join(checkpoint, "user_content.pt")):
                        # Optimizer checkpoint was saved with smp >= 1.10
                        def opt_load_hook(mod, opt):
                            opt.load_state_dict(smp.load(os.path.join(checkpoint, OPTIMIZER_NAME), partial=True))

                    else:
                        # Optimizer checkpoint was saved with smp < 1.10
                        def opt_load_hook(mod, opt):
                            if IS_SAGEMAKER_MP_POST_1_10:
                                opt.load_state_dict(
                                    smp.load(os.path.join(checkpoint, OPTIMIZER_NAME), partial=True, back_compat=True)
                                )
                            else:
                                opt.load_state_dict(smp.load(os.path.join(checkpoint, OPTIMIZER_NAME), partial=True))

                    self.model_wrapped.register_post_step_hook(opt_load_hook)
                else:
                    # We use the CPU when training on one GPU to avoid OOM for GPU RAM when training big models.
                    # In distributed training however, we load directly on each GPU and risk the GPU OOM as it's more
                    # likely to get OOM on CPU (since we load num_gpu times the optimizer state
                    map_location = self.args.device if self.args.world_size > 1 else "cpu"
                    if self.fsdp:
                        full_osd = None
                        # In FSDP, we need to load the full optimizer state dict on rank 0 and then shard it
                        if self.args.process_index == 0:
                            full_osd = torch.load(os.path.join(checkpoint, OPTIMIZER_NAME))
                        # call scatter_full_optim_state_dict on all ranks
                        sharded_osd = self.model.__class__.scatter_full_optim_state_dict(full_osd, self.model)
                        self.optimizer.load_state_dict(sharded_osd)
                    else:
                        self.optimizer.load_state_dict(
                            torch.load(os.path.join(checkpoint, OPTIMIZER_NAME), map_location=map_location)
                        )
                with warnings.catch_warnings(record=True) as caught_warnings:
                    self.lr_scheduler.load_state_dict(torch.load(os.path.join(checkpoint, SCHEDULER_NAME)))
                reissue_pt_warnings(caught_warnings)
                if self.do_grad_scaling and os.path.isfile(os.path.join(checkpoint, SCALER_NAME)):
                    self.scaler.load_state_dict(torch.load(os.path.join(checkpoint, SCALER_NAME)))

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        self._train_batch_size = batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps and args.logging_steps < 1:
            args.logging_steps = math.ceil(max_steps * args.logging_steps)
        if args.eval_steps and args.eval_steps < 1:
            args.eval_steps = math.ceil(max_steps * args.eval_steps)
        if args.save_steps and args.save_steps < 1:
            args.save_steps = math.ceil(max_steps * args.save_steps)

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )

        if self.is_deepspeed_enabled:
            self.optimizer, self.lr_scheduler = deepspeed_init(self, num_training_steps=max_steps)

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # Fairscale Sharded DDP, FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.lr_scheduler
            )

        if self.is_fsdp_enabled:
            self.model = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # backward compatibility
        if self.is_deepspeed_enabled:
            self.deepspeed = self.model_wrapped

        # deepspeed ckpt loading
        if resume_from_checkpoint is not None and self.is_deepspeed_enabled:
            deepspeed_load_checkpoint(self.model_wrapped, resume_from_checkpoint)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                self.model, self.optimizer, self.lr_scheduler
            )

        if getattr(self.accelerator.state, "fsdp_plugin", None) is not None:
            self.model = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self._train_batch_size:,}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size:,}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps:,}")
        logger.info(f"  Number of trainable parameters = {get_model_param_count(model, trainable_only=True):,}")

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                if skip_first_batches is None:
                    logger.info(
                        f"  Will skip the first {epochs_trained} epochs then the first"
                        f" {steps_trained_in_current_epoch} batches in the first epoch. If this takes a lot of time,"
                        " you can install the latest version of Accelerate with `pip install -U accelerate`.You can"
                        " also add the `--ignore_data_skip` flag to your launch command, but you will resume the"
                        " training on data already seen by your model."
                    )
                else:
                    logger.info(
                        f"  Will skip the first {epochs_trained} epochs then the first"
                        f" {steps_trained_in_current_epoch} batches in the first epoch."
                    )
                if self.is_local_process_zero() and not args.disable_tqdm and skip_first_batches is None:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            rng_to_sync = False
            steps_skipped = 0
            if skip_first_batches is not None and steps_trained_in_current_epoch > 0:
                epoch_iterator = skip_first_batches(epoch_iterator, steps_trained_in_current_epoch)
                steps_skipped = steps_trained_in_current_epoch
                steps_trained_in_current_epoch = 0
                rng_to_sync = True

            step = -1
            for step, inputs in enumerate(epoch_iterator):
                total_batched_samples += 1
                if rng_to_sync:
                    self._load_rng_state(resume_from_checkpoint)
                    rng_to_sync = False

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                with self.accelerator.accumulate(model):
                    tr_loss_step = self.training_step(model, inputs)

                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                # should this be under the accumulate context manager?
                # the `or` condition of `steps_in_epoch <= args.gradient_accumulation_steps` is not covered
                # in accelerate
                if total_batched_samples % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # Reduce gradients first for XLA
                            if is_torch_tpu_available():
                                gradients = xm._fetch_gradients(self.optimizer)
                                xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if is_torch_tpu_available():
                        if self.do_grad_scaling:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            xm.optimizer_step(self.optimizer)
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1 + steps_skipped) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss

        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)

        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)


class TrainiumTrainer(AugmentTrainerForTrainiumMixin, Trainer):
    """
    Trainer that is suited for performing training on AWS Tranium instances.
    """


class Seq2SeqTrainiumTrainer(AugmentTrainerForTrainiumMixin, Seq2SeqTrainer):
    """
    Seq2SeqTrainer that is suited for performing training on AWS Tranium instances.
    """
