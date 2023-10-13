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
"""Defines Trainer subclasses to perform training on AWS Neuron instances."""

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
    TRAINING_ARGS_NAME,
)
from transformers.trainer_pt_utils import reissue_pt_warnings
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR, EvalLoopOutput
from transformers.utils import is_sagemaker_mp_enabled

from ..utils import check_if_transformers_greater, logging
from .accelerate import NeuronAccelerator, NeuronDistributedType
from .distributed import ParallelizersManager
from .distributed.utils import make_optimizer_constructor_lazy
from .trainer_callback import NeuronCacheCallback
from .utils import (
    DynamicPatch,
    ModelPatcher,
    is_torch_xla_available,
    patch_within_function,
)
from .utils.cache_utils import NEURON_COMPILE_CACHE_NAME, get_neuron_cache_path, set_neuron_cache_path
from .utils.training_utils import (
    TRANSFORMERS_MIN_VERSION_USE_ACCELERATE,
    get_model_param_count,
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
_TMP_NEURON_CACHE_PATH: Optional[Path] = None
_TCP_STORE_ADDRESS = "127.0.0.1"
_TCP_STORE_PORT = 5000


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

        if not is_precompilation():
            if os.environ["RANK"] == "0":
                _TMP_NEURON_CACHE_DIR = NeuronCacheCallback.create_temporary_neuron_cache(get_neuron_cache_path())
                store = torch.distributed.TCPStore(_TCP_STORE_ADDRESS, _TCP_STORE_PORT, is_master=True)
                store.set("tmp_neuron_cache_path", _TMP_NEURON_CACHE_DIR.name)
                _TMP_NEURON_CACHE_PATH = Path(_TMP_NEURON_CACHE_DIR.name)
            else:
                store = torch.distributed.TCPStore(_TCP_STORE_ADDRESS, _TCP_STORE_PORT, is_master=False)
                _TMP_NEURON_CACHE_PATH = Path(store.get("tmp_neuron_cache_path").decode("utf-8"))
            set_neuron_cache_path(_TMP_NEURON_CACHE_PATH / NEURON_COMPILE_CACHE_NAME)

        torch.distributed.init_process_group(backend="xla")
        if not isinstance(torch.distributed.group.WORLD, xbn.ProcessGroupXla):
            raise AssertionError("Failed to initialize torch.distributed process group using XLA backend.")

transformers_get_optimizer_cls_and_kwargs = Trainer.get_optimizer_cls_and_kwargs


class AugmentTrainerForNeuronMixin:
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

        push = self.args.local_rank <= 0 and not is_precompilation()
        fetch = self.args.local_rank <= 0 or self.args.tp_plugin.should_parallelize

        callback = NeuronCacheCallback(
            tmp_neuron_cache=_TMP_NEURON_CACHE_PATH,
            original_neuron_cache_path=_ORIGINAL_NEURON_CACHE_PATH,
            fetch=fetch,
            push=push,
            wait_for_everyone_on_fetch=False,
            wait_for_everyone_on_push=True,
        )
        self.add_callback(callback)

        # Make the model Neuron-compatible for generation.
        patch_generation_mixin_to_neuron_generation_mixin(self.model)

    @property
    def tp_enabled(self):
        return (
            check_if_transformers_greater("4.30.0")
            and self.accelerator.distributed_type is NeuronDistributedType.TENSOR_PARALLELISM
        )

    def prepare_args_for_precompilation(self, args: "TrainingArguments"):
        if args.num_train_epochs != 1:
            logger.info("Setting the number of epochs for precompilation to 1.")
            args.num_train_epochs = 1
        if args.do_eval is True:
            logger.info("Disabling evaluation during precompilation as this is not well supported yet.")
            args.do_eval = False
        if args.do_predict is True:
            logger.info("Disabling prediction during precompilation as this is not well supported yet.")
            args.do_predict = False

    def validate_args(self, args: "TrainingArguments"):
        pass

    def create_accelerator_and_postprocess(self):
        # create accelerator object
        self.accelerator = NeuronAccelerator(
            deepspeed_plugin=self.args.deepspeed_plugin,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            tp_plugin=self.args.tp_plugin,
            zero_1=self.args.zero_1,
        )

        # deepspeed and accelerate flags covering both trainer args and accelerate launcher
        self.is_deepspeed_enabled = getattr(self.accelerator.state, "deepspeed_plugin", None) is not None
        self.is_fsdp_enabled = getattr(self.accelerator.state, "fsdp_plugin", None) is not None

        # post accelerator creation setup
        if self.is_fsdp_enabled:
            fsdp_plugin = self.accelerator.state.fsdp_plugin
            fsdp_plugin.limit_all_gathers = self.args.fsdp_config.get("limit_all_gathers", False)
            fsdp_plugin.use_orig_params = self.args.fsdp_config.get("use_orig_params", False)

        if self.is_deepspeed_enabled:
            if getattr(self.args, "hf_deepspeed_config", None) is None:
                from transformers.deepspeed import HfTrainerDeepSpeedConfig

                ds_plugin = self.accelerator.state.deepspeed_plugin

                ds_plugin.hf_ds_config = HfTrainerDeepSpeedConfig(ds_plugin.hf_ds_config.config)
                ds_plugin.deepspeed_config = ds_plugin.hf_ds_config.config
                ds_plugin.hf_ds_config.trainer_config_process(self.args)

    def _wrap_model(self, model, training=True, dataloader=None):
        patching_specs = []
        for spec in MODEL_PATCHING_SPECS:
            patching_specs.append((model,) + spec)

        with ModelPatcher(patching_specs, ignore_missing_attributes=True):
            return super()._wrap_model(model, training=training, dataloader=dataloader)

    # TODO: make this cleaner.
    def trigger_on_step_middle_for_neuron_cache_callback(self, model: "PreTrainedModel"):
        for callback in self.callback_handler.callbacks:
            if isinstance(callback, NeuronCacheCallback):
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

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.tp_enabled:
            return None
        return super()._get_train_sampler()

    @staticmethod
    def get_optimizer_cls_and_kwargs(args: TrainingArguments) -> Tuple[Any, Any]:
        optimizer_cls, optimizer_kwargs = transformers_get_optimizer_cls_and_kwargs(args)
        lazy_load = args.tp_plugin.should_parallelize or args.zero_1
        if check_if_transformers_greater("4.30.0") and lazy_load:
            optimizer_cls = make_optimizer_constructor_lazy(optimizer_cls)
        return optimizer_cls, optimizer_kwargs

    @patch_within_function(("transformers.Trainer.get_optimizer_cls_and_kwargs", get_optimizer_cls_and_kwargs))
    def create_optimizer(self):
        return super().create_optimizer()

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

    @patch_within_function(("transformers.trainer.get_model_param_count", get_model_param_count))
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

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            if is_torch_xla_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            if self.args.tp_plugin.tensor_parallel_size > 1:
                from neuronx_distributed.parallel_layers.parallel_state import (
                    get_data_parallel_group,
                    get_data_parallel_size,
                )

                dp_size = get_data_parallel_size()
                tr_loss_div = tr_loss / dp_size
                tr_loss_scalar = xm.all_reduce(
                    xm.REDUCE_SUM,
                    tr_loss_div,
                    groups=get_data_parallel_group(as_list=True),
                )
                tr_loss_scalar = tr_loss_scalar.detach().item()
            else:
                # all_gather + mean() to get average loss over all processes
                tr_loss_scalar = self._nested_gather(tr_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss

            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)
            logs["learning_rate"] = self._get_learning_rate()

            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                metrics = {}
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    dataset_metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
                    metrics.update(dataset_metrics)
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

            # Run delayed LR scheduler now that metrics are populated
            if isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                metric_to_check = self.args.metric_for_best_model
                if not metric_to_check.startswith("eval_"):
                    metric_to_check = f"eval_{metric_to_check}"
                self.lr_scheduler.step(metrics[metric_to_check])

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)

    def _save_checkpoint_with_accelerator(self, model, trial, metrics=None):
        if self.accelerator.distributed_type is NeuronDistributedType.XLA_FSDP and not self.is_fsdp_enabled:
            # TODO: handle this case better?
            # Do we want to fail here? Can we save anyway?
            raise RuntimeError("Cannot save checkpoint if FSDP is not enabled.")

        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        os.makedirs(output_dir, exist_ok=True)

        if xm.is_master_ordinal():
            os.makedirs(output_dir, exist_ok=True)
            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        if self.tokenizer is not None and self.args.should_save:
            self.tokenizer.save_pretrained(output_dir)

        if isinstance(self.model, PreTrainedModel):
            self.model.config.save_pretrained(output_dir)

        self.accelerator.save_state(output_dir)

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
        if check_if_transformers_greater("4.30.0") and self.accelerator.distributed_type in [
            NeuronDistributedType.XLA_FSDP,
            NeuronDistributedType.TENSOR_PARALLELISM,
        ]:
            return self._save_checkpoint_with_accelerator(model, trial, metrics=metrics)
        return super()._save_checkpoint(model, trial, metrics=metrics)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if output_dir is None:
            output_dir = self.args.output_dir
        if self.accelerator.distributed_type is NeuronDistributedType.XLA_FSDP:
            self.accelerator.state.fsdp_plugin.save_model(self.accelerator, self.model, output_dir, 0)
        elif self.accelerator.distributed_type is NeuronDistributedType.TENSOR_PARALLELISM:
            parallelizer = ParallelizersManager.parallelizer_for_model(self.model)
            parallelizer.save_model_checkpoint(self.model, output_dir, as_regular=False)
        else:
            return super().save_model(output_dir=output_dir, _internal_call=_internal_call)

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

    def evaluation_loop(
        self,
        dataloader: torch.utils.data.DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        # This will prepare the model if it was not prepared before.
        # This is needed for example for TP when we performing only evaluation (no training):
        #   1. The model needs to be loaded if it was lazy loaded.
        #   2. The model needs to be parallelized.
        self.accelerator.prepare_model(self.model)

        return super().evaluation_loop(
            dataloader,
            description,
            prediction_loss_only=prediction_loss_only,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )


class NeuronTrainer(AugmentTrainerForNeuronMixin, Trainer):
    """
    Trainer that is suited for performing training on AWS Tranium instances.
    """


class Seq2SeqNeuronTrainer(AugmentTrainerForNeuronMixin, Seq2SeqTrainer):
    """
    Seq2SeqTrainer that is suited for performing training on AWS Tranium instances.
    """
