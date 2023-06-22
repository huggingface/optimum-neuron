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

import contextlib
import glob
import os
import random
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from packaging import version
from transformers import PreTrainedModel, Seq2SeqTrainer, Trainer, TrainingArguments
from transformers.dependency_versions_check import dep_version_check
from transformers.integrations import is_fairscale_available
from transformers.trainer import (
    OPTIMIZER_NAME,
    SCHEDULER_NAME,
    TRAINER_STATE_NAME,
)
from transformers.trainer_pt_utils import reissue_pt_warnings
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import is_sagemaker_mp_enabled

from ..utils import check_if_transformers_greater, logging
from .accelerate import NeuronAccelerator
from .trainer_callback import NeuronCacheCallaback
from .utils import DynamicPatch, ModelPatcher, is_torch_xla_available, patch_within_function
from .utils.cache_utils import get_neuron_cache_path
from .utils.training_utils import (
    TRANSFORMERS_MIN_VERSION_USE_ACCELERATE,
    is_precompilation,
    patch_generation_mixin_to_neuron_generation_mixin,
    patched_finfo,
    prepare_environment_for_neuron,
    skip_first_batches,
)


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

if is_sagemaker_mp_enabled():
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

else:
    IS_SAGEMAKER_MP_POST_1_10 = False

if is_fairscale_available():
    dep_version_check("fairscale")


logger = logging.get_logger("transformers.trainer")

KEEP_HF_HUB_PROGRESS_BARS = os.environ.get("KEEP_HF_HUB_PROGRESS_BARS")
if KEEP_HF_HUB_PROGRESS_BARS is None:
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

# Used for torch.distributed.
_ORIGINAL_NEURON_CACHE_PATH: Optional[Path] = None
_TMP_NEURON_CACHE_DIR: Optional[TemporaryDirectory] = None


MODEL_PATCHING_SPECS = [
    ("config.layerdrop", 0),
    ("no_sync", lambda: contextlib.nullcontext()),
    (
        "forward",
        DynamicPatch(patch_within_function(("torch.finfo", patched_finfo))),
    ),
]


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

        if check_if_transformers_greater(TRANSFORMERS_MIN_VERSION_USE_ACCELERATE):
            import transformers

            transformers.trainer.Accelerator = NeuronAccelerator

        prepare_environment_for_neuron()
        super().__init__(*args, **kwargs)

        # That's the case for Transformers < 4.30.0
        if not hasattr(self, "is_fsdp_enabled"):
            self.is_fsdp_enabled = False

        if self.is_fsdp_enabled and self.args.do_eval:
            raise ValueError("Evaluation is not supported with XLA FSDP yet.")

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
        patch_generation_mixin_to_neuron_generation_mixin(self.model)

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
        pass

    @patch_within_function(("transformers.trainer.Accelerator", NeuronAccelerator), ignore_missing_attributes=True)
    def create_accelerator_and_postprocess(self):
        return super().create_accelerator_and_postprocess()

    def _wrap_model(self, model, training=True, dataloader=None):
        patching_specs = []
        for spec in MODEL_PATCHING_SPECS:
            patching_specs.append((model,) + spec)

        with ModelPatcher(patching_specs, ignore_missing_attributes=True):
            return super()._wrap_model(model, training=training, dataloader=dataloader)

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

    # def _nested_gather_for_xla_fsdp(self, tensors, name=None):
    #     # if isinstance(tensors, (list, tuple)):
    #     #     return type(tensors)(self._nested_gather_for_xla_fsdp(t, f"{name}_{i}") for i, t in enumerate(tensors))
    #     # if isinstance(tensors, dict):
    #     #     return type(tensors)(
    #     #         {k: self._nested_gather_for_xla_fsdp(t, f"{name}_{i}") for i, (k, t) in enumerate(tensors.items())}
    #     #     )

    #     # tensors = atleast_1d(tensors)
    #     # return xm.mesh_reduce(name, tensors, torch.cat)
    #     if isinstance(tensors, (tuple, list)):
    #         return type(tensors)(self._nested_gather_for_xla_fsdp(t) for t in tensors)
    #     elif isinstance(tensors, dict):
    #         return type(tensors)({k: self._nested_gather_for_xla_fsdp(t) for k, t in tensors.items()})
    #     tensors = atleast_1d(tensors)
    #     # result = torch.empty((self.args.world_size,), device=self.args.device, dtype=tensors.dtype)
    #     # print("tensors", tensors)
    #     # print("result", result)
    #     result = xm.all_gather(tensors, dim=0)
    #     # print("gathered result", result)
    #     return result

    # def _nested_gather(self, tensors, name=None):
    #     if self.is_fsdp_enabled:
    #         return self._nested_gather_for_xla_fsdp(tensors, name="nested_gather_for_xla_fsdp")
    #     return super()._nested_gather(tensors, name=name)

    def _save_checkpoint_for_xla_fsdp(self, model, trial, metrics=None):
        if not self.is_fsdp_enabled:
            # TODO: handle this case better?
            # Do we want to fail here? Can we save anyway?
            raise RuntimeError("Cannot save checkpoint if FSDP is not enabled.")

        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)

        # Save model
        self.accelerator.state.fsdp_plugin.save_model(self.accelerator, self.model, output_dir)

        # Save optimizer
        self.accelerator.state.fsdp_plugin.save_optimizer(self.accelerator, self.optimizer, self.model, output_dir)

        # Save scheduler
        with warnings.catch_warnings(record=True):
            xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))

        # Save scaler
        # TODO: is grad scaling supported with TORCH XLA?
        # reissue_pt_warnings(caught_warnings)
        # if self.do_grad_scaling:
        #     xm.save(self.scaler.state_dict(), os.path.join(output_dir, SCALER_NAME))

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

        if is_torch_xla_available():
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

    def _save_checkpoint(self, model, trial, metrics=None):
        if self.fsdp or self.is_fsdp_enabled:
            return self._save_checkpoint_for_xla_fsdp(model, trial, metrics=metrics)
        return super()._save_checkpoint(model, trial, metrics=metrics)

    def _load_optimizer_and_scheduler_for_xla_fsdp(self, checkpoint):
        if checkpoint is None:
            return
        checkpoint_file_exists = (
            glob.glob(os.path.join(checkpoint, OPTIMIZER_NAME) + "_*")
            if is_sagemaker_mp_enabled()
            else os.path.isfile(os.path.join(checkpoint, OPTIMIZER_NAME))
        )
        if checkpoint_file_exists and os.path.isfile(os.path.join(checkpoint, SCHEDULER_NAME)):
            self.accelerator.state.fsdp_plugin.load_optimizer(self.accelerator, self.optimizer, self.model, checkpoint)

            with warnings.catch_warnings(record=True) as caught_warnings:
                lr_scheduler_state = torch.load(os.path.join(checkpoint, SCHEDULER_NAME), map_location="cpu")
            reissue_pt_warnings(caught_warnings)
            xm.send_cpu_data_to_device(lr_scheduler_state, self.args.device)
            self.lr_scheduler.load_state_dict(lr_scheduler_state)

        # TODO: load grad scaling?

    def _load_optimizer_and_scheduler(self, checkpoint):
        if self.fsdp or self.is_fsdp_enabled:
            return self._load_optimizer_and_scheduler_for_xla_fsdp(checkpoint)
        return super()._load_optimizer_and_scheduler(checkpoint)

    @patch_within_function(("transformers.trainer.skip_first_batches", skip_first_batches))
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        return super()._inner_training_loop(
            batch_size=batch_size,
            args=args,
            resume_from_checkpoint=resume_from_checkpoint,
            trial=trial,
            ignore_keys_for_eval=ignore_keys_for_eval,
        )


class TrainiumTrainer(AugmentTrainerForTrainiumMixin, Trainer):
    """
    Trainer that is suited for performing training on AWS Tranium instances.
    """


class Seq2SeqTrainiumTrainer(AugmentTrainerForTrainiumMixin, Seq2SeqTrainer):
    """
    Seq2SeqTrainer that is suited for performing training on AWS Tranium instances.
    """
