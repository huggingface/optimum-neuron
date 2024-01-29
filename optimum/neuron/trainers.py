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

import copy
import glob
import math
import os
import random
import shutil
import sys
import time
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from accelerate import __version__ as accelerate_version
from packaging import version
from transformers import PreTrainedModel, Seq2SeqTrainer, Trainer, TrainingArguments
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.integrations import hp_params
from transformers.modeling_utils import unwrap_model
from transformers.pytorch_utils import is_torch_less_than_1_11
from transformers.trainer import (
    OPTIMIZER_NAME,
    SCHEDULER_NAME,
    TRAINER_STATE_NAME,
    TRAINING_ARGS_NAME,
)
from transformers.trainer_callback import TrainerState
from transformers.trainer_pt_utils import (
    IterableDatasetShard,
    find_batch_size,
    get_dataloader_sampler,
    nested_concat,
    nested_numpify,
    reissue_pt_warnings,
)
from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    EvalLoopOutput,
    EvalPrediction,
    HPSearchBackend,
    TrainOutput,
    denumpify_detensorize,
    has_length,
    speed_metrics,
)
from transformers.training_args import ParallelMode
from transformers.utils import WEIGHTS_NAME, is_apex_available, is_sagemaker_mp_enabled

from ..utils import check_if_transformers_greater, logging
from .accelerate import NeuronAccelerator, NeuronDistributedType
from .distributed import Parallelizer, ParallelizersManager
from .distributed.utils import make_optimizer_constructor_lazy
from .trainer_callback import NeuronCacheCallback
from .utils import (
    Patcher,
    is_torch_xla_available,
    patch_within_function,
)
from .utils.cache_utils import get_neuron_cache_path, set_neuron_cache_path
from .utils.require_utils import requires_neuronx_distributed
from .utils.training_utils import (
    TRANSFORMERS_MIN_VERSION_USE_ACCELERATE,
    get_model_param_count,
    is_precompilation,
    is_topology_supported,
    patch_generation_mixin_to_neuron_generation_mixin,
    prepare_environment_for_neuron,
    set_neuron_cc_optlevel_for_model,
    skip_first_batches,
    torch_xla_safe_save_file,
)


if is_apex_available():
    from apex import amp

if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

if is_sagemaker_mp_enabled():
    from smdistributed.modelparallel import __version__ as SMP_VERSION

    IS_SAGEMAKER_MP_POST_1_10 = version.parse(SMP_VERSION) >= version.parse("1.10")

else:
    IS_SAGEMAKER_MP_POST_1_10 = False

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


if os.environ.get("TORCHELASTIC_RUN_ID"):
    import torch_xla.distributed.xla_backend as xbn

    if not isinstance(torch.distributed.group.WORLD, xbn.ProcessGroupXla):
        _ORIGINAL_NEURON_CACHE_PATH = get_neuron_cache_path()

        # _ORIGINAL_NEURON_CACHE_PATH is `None` when the `--no-cache` flag is set.
        if _ORIGINAL_NEURON_CACHE_PATH is not None:
            if is_precompilation():
                # During precompilation, we make sure to set the cache path to the defined compile cache path by the
                # user. If nothing is specified, it is set to the default compile cache used by the Neuron compiler:
                # /var/tmp/neuron-compile-cache
                set_neuron_cache_path(_ORIGINAL_NEURON_CACHE_PATH)
            else:
                if os.environ["RANK"] == "0":
                    _TMP_NEURON_CACHE_DIR = NeuronCacheCallback.create_temporary_neuron_cache(get_neuron_cache_path())
                    store = torch.distributed.TCPStore(_TCP_STORE_ADDRESS, _TCP_STORE_PORT, is_master=True)
                    store.set("tmp_neuron_cache_path", _TMP_NEURON_CACHE_DIR.name)
                    _TMP_NEURON_CACHE_PATH = Path(_TMP_NEURON_CACHE_DIR.name)
                else:
                    store = torch.distributed.TCPStore(_TCP_STORE_ADDRESS, _TCP_STORE_PORT, is_master=False)
                    _TMP_NEURON_CACHE_PATH = Path(store.get("tmp_neuron_cache_path").decode("utf-8"))
                set_neuron_cache_path(_TMP_NEURON_CACHE_PATH)

        torch.distributed.init_process_group(backend="xla")
        if not isinstance(torch.distributed.group.WORLD, xbn.ProcessGroupXla):
            raise AssertionError("Failed to initialize torch.distributed process group using XLA backend.")

transformers_get_optimizer_cls_and_kwargs = Trainer.get_optimizer_cls_and_kwargs


class AugmentTrainerForNeuronMixin:
    def __init__(self, *args, **kwargs):
        if not isinstance(self, Trainer):
            raise TypeError(f"{self.__class__.__name__} can only be mixed with Trainer subclasses.")

        if not is_topology_supported():
            num_devices = xm.xrt_world_size()
            raise ValueError(
                f"Topology not supported. Supported number of devices: 1, 2, 8 or a multiple of 32. Got: {num_devices}."
            )

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

        push = self.args.local_rank <= 0 and not is_precompilation() and not self.args.skip_cache_push
        fetch = self.args.local_rank <= 0 or self.args.mp_plugin.should_parallelize

        callback = NeuronCacheCallback(
            tmp_neuron_cache=_TMP_NEURON_CACHE_PATH,
            original_neuron_cache_path=_ORIGINAL_NEURON_CACHE_PATH,
            fetch=fetch,
            push=push,
            wait_for_everyone_on_fetch=True,
            wait_for_everyone_on_push=True,
        )
        self.add_callback(callback)

        # Make the model Neuron-compatible for generation.
        patch_generation_mixin_to_neuron_generation_mixin(self.model)

        set_neuron_cc_optlevel_for_model(self.model, optlevel=self.args.neuron_cc_optlevel)

    @property
    def mp_enabled(self):
        return self.accelerator.distributed_type is NeuronDistributedType.MODEL_PARALLELISM

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
            mp_plugin=self.args.mp_plugin,
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
        return super()._wrap_model(
            self.accelerator.patch_model_for_neuron(model), training=training, dataloader=dataloader
        )

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
        if self.mp_enabled:
            if self.train_dataset is None or not has_length(self.train_dataset):
                return None

            if self.args.group_by_length:
                raise ValueError("LengthGroupedSampler is currently not supported with model parallelism.")

            return torch.utils.data.RandomSampler(self.train_dataset)
        return super()._get_train_sampler()

    def _get_eval_sampler(self, eval_dataset: torch.utils.data.Dataset) -> Optional[torch.utils.data.Sampler]:
        return torch.utils.data.SequentialSampler(eval_dataset)

    @staticmethod
    def get_optimizer_cls_and_kwargs(args: TrainingArguments) -> Tuple[Any, Any]:
        optimizer_cls, optimizer_kwargs = transformers_get_optimizer_cls_and_kwargs(args)
        lazy_load = args.mp_plugin.should_parallelize or args.zero_1
        if check_if_transformers_greater("4.30.0") and lazy_load:
            optimizer_cls = make_optimizer_constructor_lazy(optimizer_cls)
        return optimizer_cls, optimizer_kwargs

    @patch_within_function(("transformers.Trainer.get_optimizer_cls_and_kwargs", get_optimizer_cls_and_kwargs))
    def create_optimizer(self):
        return super().create_optimizer()

    def _prepare_input(self, data: Union[torch.Tensor, Any]) -> Union[torch.Tensor, Any]:
        # When pipeline parallelism is enabled, we should not put any tensor on device.
        # It is handled by the NxDPPModel class.
        if self.args.mp_plugin.pipeline_parallel_size > 1:
            return data
        return super()._prepare_input(data)

    def compute_loss(self, model, inputs, return_outputs: bool = False):
        self.state.last_inputs = inputs
        self.trigger_on_step_middle_for_neuron_cache_callback(model)
        from neuronx_distributed.pipeline import NxDPPModel

        if isinstance(model, NxDPPModel):
            inputs = self._prepare_inputs(inputs)
            loss = model.run_train(**inputs)
            return loss

        return super().compute_loss(model, inputs, return_outputs=return_outputs)

    def training_step(self, model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        from neuronx_distributed.pipeline import NxDPPModel

        if isinstance(model, NxDPPModel):
            from neuronx_distributed.parallel_layers.parallel_state import (
                get_pipeline_model_parallel_rank,
                get_pipeline_model_parallel_size,
            )

            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)

            if get_pipeline_model_parallel_rank() != get_pipeline_model_parallel_size() - 1:
                use_bf16 = os.environ.get("XLA_USE_BF16", False) or os.environ.get("XLA_DOWNCAST_BF16", False)
                dtype = torch.bfloat16 if use_bf16 else torch.float32
                loss = torch.tensor(0, dtype=dtype).to(xm.xla_device())
            else:
                loss = loss.detach()
            return loss / self.args.gradient_accumulation_steps
        return super().training_step(model, inputs)

    @requires_neuronx_distributed
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        from neuronx_distributed.pipeline import NxDPPModel

        self.state.last_inputs = inputs
        self.trigger_on_step_middle_for_neuron_cache_callback(model)

        if isinstance(model, NxDPPModel):
            if not prediction_loss_only:
                raise ValueError("Only the prediction loss can be returned when doing pipeline parallelism.")
            loss = model.run_eval(**inputs)
            if loss is None:
                use_bf16 = os.environ.get("XLA_USE_BF16", False) or os.environ.get("XLA_DOWNCAST_BF16", False)
                dtype = torch.bfloat16 if use_bf16 else torch.float32
                loss = torch.tensor(0, dtype=dtype).to(xm.xla_device())
            return (loss, None, None)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

    def _maybe_log_save_evaluate(self, tr_loss, model, trial, epoch, ignore_keys_for_eval):
        if self.control.should_log:
            logs: Dict[str, float] = {}

            xm.mark_step()

            if self.args.mp_plugin.should_parallelize:
                from neuronx_distributed.parallel_layers.parallel_state import (
                    get_data_parallel_group,
                    get_data_parallel_size,
                    get_pipeline_model_parallel_group,
                    get_pipeline_model_parallel_size,
                )

                dp_size = get_data_parallel_size()
                pp_size = get_pipeline_model_parallel_size()
                tr_loss_div = tr_loss / dp_size

                if pp_size > 1:
                    tr_loss_div = xm.all_reduce(
                        xm.REDUCE_SUM, tr_loss_div, groups=get_data_parallel_group(as_list=True)
                    )
                    tr_loss_div = xm.all_reduce(
                        xm.REDUCE_SUM,
                        tr_loss_div,
                        groups=get_pipeline_model_parallel_group(as_list=True),
                    )
                    xm.mark_step()
                    tr_loss_scalar = tr_loss_div.item()
                else:
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

    def _save_xla(self, output_dir: Optional[str] = None):
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        logger.info(f"Saving model checkpoint to {output_dir}")

        if xm.is_master_ordinal():
            os.makedirs(output_dir, exist_ok=True)
            torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        xm.rendezvous("saving_checkpoint")
        if self.accelerator.distributed_type is NeuronDistributedType.MODEL_PARALLELISM:
            logger.info("Model parallelism is enabled, only saving the model sharded state dict.")
            # TODO: how to handle pp?
            if isinstance(self.model, PreTrainedModel):
                from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_size

                config = copy.deepcopy(self.model.config)
                if self.args.mp_plugin.parallelize_embeddings:
                    config.vocab_size = config.vocab_size * get_tensor_model_parallel_size()
                config.save_pretrained(output_dir)

            # This mark_step is needed to avoid hang issues.
            xm.mark_step()
            Parallelizer.save_model_checkpoint(self.model, output_dir, as_sharded=True, optimizer=self.optimizer)
        else:
            safe_save_function_patcher = Patcher(
                [("transformers.modeling_utils.safe_save_file", torch_xla_safe_save_file)]
            )
            if not isinstance(self.model, PreTrainedModel):
                if isinstance(unwrap_model(self.model), PreTrainedModel):
                    with safe_save_function_patcher:
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
                with safe_save_function_patcher:
                    self.model.save_pretrained(
                        output_dir, is_main_process=self.args.should_save, save_function=xm.save
                    )

        if self.tokenizer is not None and self.args.should_save:
            self.tokenizer.save_pretrained(output_dir)

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if not os.environ.get("NEURON_PARALLEL_COMPILE"):  # Avoid unnecessary model saving during precompilation
            if output_dir is None:
                output_dir = self.args.output_dir

            self._save_xla(output_dir)

            # Push to the Hub when `save_model` is called by the user.
            if self.args.push_to_hub and not _internal_call:
                self.push_to_hub(commit_message="Model save")
        else:
            logger.info("Skipping trainer.save_model() while running under neuron_parallel_compile")

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

        if xm.is_master_ordinal():
            os.makedirs(output_dir, exist_ok=True)

        self.save_model(output_dir, _internal_call=True)

        # The optimizer state is saved in the shard alongside with the model parameters when doing TP.
        if self.accelerator.distributed_type is not NeuronDistributedType.MODEL_PARALLELISM:
            xm.rendezvous("saving_optimizer_states")
            # TODO: how to handle pp?
            xm.save(self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME))

        with warnings.catch_warnings(record=True) as caught_warnings:
            xm.save(self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME))
            reissue_pt_warnings(caught_warnings)

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

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        # It has been handled during model parallelization.
        # TODO: how to handle pp?
        if self.accelerator.distributed_type is NeuronDistributedType.MODEL_PARALLELISM:
            return
        super()._load_from_checkpoint(resume_from_checkpoint, model=model)

    def _load_optimizer_and_scheduler_for_xla_fsdp(self, checkpoint):
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
        if checkpoint is None:
            return
        if self.accelerator.distributed_type is NeuronDistributedType.XLA_FSDP:
            return self._load_optimizer_and_scheduler_for_xla_fsdp(checkpoint)
        elif self.accelerator.distributed_type is NeuronDistributedType.MODEL_PARALLELISM:
            lr_scheduler_state = torch.load(os.path.join(checkpoint, SCHEDULER_NAME), map_location="cpu")
            xm.send_cpu_data_to_device(lr_scheduler_state, self.args.device)
            self.lr_scheduler.load_state_dict(lr_scheduler_state)

            parallelizer = ParallelizersManager.parallelizer_for_model(self.model)
            parallelizer.load_optimizer_sharded_checkpoint(self.optimizer, checkpoint)
        else:
            return super()._load_optimizer_and_scheduler(checkpoint)

    @requires_neuronx_distributed
    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        from neuronx_distributed.pipeline import NxDPPModel

        self.accelerator.free_memory()
        self._train_batch_size = batch_size
        logger.debug(f"Currently training with a batch size of: {self._train_batch_size}")
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = self._train_batch_size * args.gradient_accumulation_steps * args.world_size

        len_dataloader = None
        num_train_tokens = None
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
                if args.include_tokens_per_second:
                    num_train_tokens = (
                        self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
                    )
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
                if args.include_tokens_per_second:
                    num_train_tokens = self.num_tokens(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
            if args.include_tokens_per_second:
                num_train_tokens = self.num_tokens(train_dataloader, args.max_steps) * args.gradient_accumulation_steps
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

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

        delay_optimizer_creation = is_sagemaker_mp_enabled() or self.is_fsdp_xla_enabled or self.is_fsdp_enabled

        # We need to reset the scheduler, as its parameters may be different on subsequent calls
        if self._created_lr_scheduler:
            self.lr_scheduler = None
            self._created_lr_scheduler = False

        if not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Compute absolute values for logging, eval, and save if given as ratio
        if args.logging_steps is not None:
            if args.logging_steps < 1:
                self.state.logging_steps = math.ceil(max_steps * args.logging_steps)
            else:
                self.state.logging_steps = args.logging_steps
        if args.eval_steps is not None:
            if args.eval_steps < 1:
                self.state.eval_steps = math.ceil(max_steps * args.eval_steps)
            else:
                self.state.eval_steps = args.eval_steps
        if args.save_steps is not None:
            if args.save_steps < 1:
                self.state.save_steps = math.ceil(max_steps * args.save_steps)
            else:
                self.state.save_steps = args.save_steps

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            if args.gradient_checkpointing_kwargs is None:
                gradient_checkpointing_kwargs = {}
            else:
                gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs

            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs)

        model = self._wrap_model(self.model_wrapped)

        # as the model is wrapped, don't use `accelerator.prepare`
        # this is for unhandled cases such as
        # FSDP-XLA, SageMaker MP/DP, DataParallel, IPEX
        use_accelerator_prepare = True if model is self.model else False

        if delay_optimizer_creation:
            if use_accelerator_prepare:
                self.model = self.accelerator.prepare(self.model)
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # prepare using `accelerator` prepare
        if use_accelerator_prepare:
            self.model.train()
            if hasattr(self.lr_scheduler, "step"):
                if self.use_apex:
                    model = self.accelerator.prepare(self.model)
                else:
                    model, self.optimizer = self.accelerator.prepare(self.model, self.optimizer)
            else:
                # to handle cases wherein we pass "DummyScheduler" such as when it is specified in DeepSpeed config.
                model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(
                    self.model, self.optimizer, self.lr_scheduler
                )

        if isinstance(model, NxDPPModel):
            self.model = model

        if self.is_fsdp_enabled:
            self.model = self.model_wrapped = model

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model),
        # FSDP(Transformers Model), Dynamo Optimized Module(Transformers Model) etc.

        # Train!
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples:,}")
        logger.info(f"  Num Epochs = {num_train_epochs:,}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size:,}")
        if self.args.per_device_train_batch_size != self._train_batch_size:
            logger.info(f"  Training with DataParallel so batch size has been adjusted to: {self._train_batch_size:,}")
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
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first"
                    f" {steps_trained_in_current_epoch} batches in the first epoch."
                )

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

        # It should be equivalent but prefer to use the `zero_grad` method from the optimizer when doing pipeline
        # parallelism.
        if isinstance(model, NxDPPModel):
            self.optimizer.zero_grad()
        else:
            model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                sampler = get_dataloader_sampler(train_dataloader)
                sampler_kinds = [torch.utils.data.RandomSampler]
                if version.parse(accelerate_version) > version.parse("0.23.0"):
                    from accelerate.data_loader import SeedableRandomSampler

                    sampler_kinds.append(SeedableRandomSampler)
                is_random_sampler = isinstance(sampler, tuple(sampler_kinds))
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    sampler = sampler if sampler is not None else []
                    _ = list(sampler)

        total_batched_samples = 0
        for epoch in range(epochs_trained, num_train_epochs):
            epoch_iterator = train_dataloader
            if hasattr(epoch_iterator, "set_epoch"):
                epoch_iterator.set_epoch(epoch)

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
            if steps_trained_in_current_epoch > 0:
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
                    and not is_torch_xla_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step

                self.current_flos += float(self.floating_point_ops(inputs))

                is_last_step_and_steps_less_than_grad_acc = (
                    steps_in_epoch <= args.gradient_accumulation_steps and (step + 1) == steps_in_epoch
                )

                if (
                    total_batched_samples % args.gradient_accumulation_steps == 0
                    or
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    is_last_step_and_steps_less_than_grad_acc
                ):
                    # the `or` condition of `is_last_step_and_steps_less_than_grad_acc` is not covered
                    # in accelerate. So, explicitly enable sync gradients to True in that case.
                    if is_last_step_and_steps_less_than_grad_acc or (
                        version.parse(accelerate_version) <= version.parse("0.20.3")
                    ):
                        self.accelerator.gradient_state._set_sync_gradients(True)

                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0:
                        # deepspeed does its own clipping

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif self.use_apex:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            torch.nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer),
                                args.max_grad_norm,
                            )
                        else:
                            self.accelerator.clip_grad_norm_(
                                model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    self.optimizer.step()
                    optimizer_was_run = not self.accelerator.optimizer_step_was_skipped
                    if optimizer_was_run:
                        # Delay optimizer scheduling until metrics are generated
                        if not isinstance(self.lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                            self.lr_scheduler.step()

                    # It should be equivalent but prefer to use the `zero_grad` method from the optimizer when doing
                    # pipeline parallelism.
                    if isinstance(model, NxDPPModel):
                        self.optimizer.zero_grad()
                    else:
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
                if is_torch_xla_available():
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
            # Wait for everyone to get here so we are sure the model has been saved by process 0.
            if is_torch_xla_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.parallel_mode == ParallelMode.DISTRIBUTED:
                torch.distributed.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step

        metrics = speed_metrics(
            "train",
            start_time,
            num_samples=num_train_samples,
            num_steps=self.state.max_steps,
            num_tokens=num_train_tokens,
        )
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
                if not os.path.samefile(checkpoint, self.state.best_model_checkpoint):
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        # Wait for the checkpoint to be uploaded.
        self._finish_current_push()

        # After training we make sure to retrieve back the original forward pass method
        # for the embedding layer by removing the forward post hook.
        if self.neftune_noise_alpha is not None:
            self._deactivate_neftune(self.model)

        return TrainOutput(self.state.global_step, train_loss, metrics)

    @requires_neuronx_distributed
    def evaluation_loop(
        self,
        dataloader: torch.utils.data.DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """
        Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.

        Works both with or without labels.
        """
        from neuronx_distributed.parallel_layers.parallel_state import get_data_parallel_size
        from neuronx_distributed.pipeline import NxDPPModel

        # This will prepare the model if it was not prepared before.
        # This is needed for example for TP when we performing only evaluation (no training):
        #   1. The model needs to be loaded if it was lazy loaded.
        #   2. The model needs to be parallelized.
        model = self.accelerator.prepare_model(self.model)

        args = self.args

        prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

        is_nxdppmodel = isinstance(model, NxDPPModel)
        if not is_nxdppmodel:
            model = self._wrap_model(model, training=False, dataloader=dataloader)

        if len(self.accelerator._models) == 0 and model is self.model:
            model = (
                self.accelerator.prepare(model)
                if self.is_deepspeed_enabled
                else self.accelerator.prepare_model(model, evaluation_mode=True)
            )

            if self.is_fsdp_enabled:
                self.model = model

            # for the rest of this function `model` is the outside model, whether it was wrapped or not
            if model is not self.model:
                self.model_wrapped = model

            # backward compatibility
            if self.is_deepspeed_enabled:
                self.deepspeed = self.model_wrapped

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device
        if not self.is_in_train and not is_nxdppmodel:
            if args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=args.device)
            elif args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=args.device)

        batch_size = self.args.eval_batch_size

        logger.info(f"***** Running {description} *****")
        dp_size = get_data_parallel_size()
        logger.info(f"  Num data parallel workers = {dp_size}")
        if has_length(dataloader):
            num_examples = self.num_examples(dataloader)
            total_num_examples = num_examples * dp_size
            logger.info(f"  Per data parallel worker num examples = {num_examples}")
            logger.info(f"  Total num examples = {total_num_examples}")
        else:
            logger.info("  Num examples: Unknown")
        logger.info(f"  Batch size = {batch_size}")

        if not is_nxdppmodel:
            model.eval()

        self.callback_handler.eval_dataloader = dataloader
        # Do this before wrapping.
        eval_dataset = getattr(dataloader, "dataset", None)

        if args.past_index >= 0:
            self._past = None

        # Initialize containers
        # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
        losses_host = None
        preds_host = None
        labels_host = None
        inputs_host = None

        # losses/preds/labels on CPU (final containers)
        all_losses = None
        all_preds = None
        all_labels = None
        all_inputs = None
        # Will be useful when we have an iterable dataset so don't know its length.

        observed_num_examples = 0
        # Main evaluation loop
        for step, inputs in enumerate(dataloader):
            # Update the observed num examples
            observed_batch_size = find_batch_size(inputs)
            if observed_batch_size is not None:
                observed_num_examples += observed_batch_size
                # For batch samplers, batch_size is not known by the dataloader in advance.
                if batch_size is None:
                    batch_size = observed_batch_size

            if is_nxdppmodel and observed_batch_size % model.num_microbatches != 0:
                if xm.get_local_ordinal() == 0:
                    logger.warning(
                        "Skipping the evaluation step because the pipeline number of microbatches "
                        f"({model.num_microbatches}) does not divide the batch size ({observed_batch_size})."
                    )
                continue

            # Prediction step
            loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
            main_input_name = getattr(model, "main_input_name", "input_ids")
            inputs_decode = self._prepare_input(inputs[main_input_name]) if args.include_inputs_for_metrics else None

            xm.mark_step()

            # Update containers on host
            if loss is not None:
                losses = self.accelerator.gather_for_metrics((loss.repeat(batch_size)))
                losses_host = losses if losses_host is None else nested_concat(losses_host, losses, padding_index=-100)
            if labels is not None:
                labels = self.accelerator.pad_across_processes(labels, dim=1, pad_index=-100)
            if inputs_decode is not None:
                inputs_decode = self.accelerator.pad_across_processes(inputs_decode, dim=1, pad_index=-100)
                inputs_decode = self.accelerator.gather_for_metrics((inputs_decode))
                inputs_host = (
                    inputs_decode
                    if inputs_host is None
                    else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                )
            if logits is not None:
                logits = self.accelerator.pad_across_processes(logits, dim=1, pad_index=-100)
                if self.preprocess_logits_for_metrics is not None:
                    logits = self.preprocess_logits_for_metrics(logits, labels)
                logits = self.accelerator.gather_for_metrics((logits))
                preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)

            if labels is not None:
                labels = self.accelerator.gather_for_metrics((labels))
                labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)

            self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

            # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
            if (
                args.eval_accumulation_steps is not None
                and (step + 1) % args.eval_accumulation_steps == 0
                and (self.accelerator.sync_gradients or version.parse(accelerate_version) > version.parse("0.20.3"))
            ):
                if losses_host is not None:
                    losses = nested_numpify(losses_host)
                    all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                if preds_host is not None:
                    logits = nested_numpify(preds_host)
                    all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                if inputs_host is not None:
                    inputs_decode = nested_numpify(inputs_host)
                    all_inputs = (
                        inputs_decode
                        if all_inputs is None
                        else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                    )
                if labels_host is not None:
                    labels = nested_numpify(labels_host)
                    all_labels = (
                        labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                    )

                # Set back to None to begin a new accumulation
                losses_host, preds_host, inputs_host, labels_host = None, None, None, None

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of the evaluation loop
            delattr(self, "_past")

        # Gather all remaining tensors and put them back on the CPU
        if losses_host is not None:
            losses = nested_numpify(losses_host)
            all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
        if preds_host is not None:
            logits = nested_numpify(preds_host)
            all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
        if inputs_host is not None:
            inputs_decode = nested_numpify(inputs_host)
            all_inputs = (
                inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
            )
        if labels_host is not None:
            labels = nested_numpify(labels_host)
            all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

        # Number of samples
        if has_length(eval_dataset):
            num_samples = len(eval_dataset)
        # The instance check is weird and does not actually check for the type, but whether the dataset has the right
        # methods. Therefore we need to make sure it also has the attribute.
        elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
            num_samples = eval_dataset.num_examples
        else:
            if has_length(dataloader):
                num_samples = self.num_examples(dataloader)
            else:  # both len(dataloader.dataset) and len(dataloader) fail
                num_samples = observed_num_examples
        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        # Metrics!
        if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
            if args.include_inputs_for_metrics:
                metrics = self.compute_metrics(
                    EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                )
            else:
                metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
        else:
            metrics = {}

        # To be JSON-serializable, we need to remove numpy types or zero-d tensors
        metrics = denumpify_detensorize(metrics)

        if all_losses is not None:
            metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
        if hasattr(self, "jit_compilation_time"):
            metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)


class NeuronTrainer(AugmentTrainerForNeuronMixin, Trainer):
    """
    Trainer that is suited for performing training on AWS Tranium instances.
    """


class Seq2SeqNeuronTrainer(AugmentTrainerForNeuronMixin, Seq2SeqTrainer):
    """
    Seq2SeqTrainer that is suited for performing training on AWS Tranium instances.
    """
