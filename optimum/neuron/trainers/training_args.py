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

import json
import math
import os
from dataclasses import dataclass, field, fields
from enum import Enum
from functools import cached_property
from typing import Any

import torch
import torch_xla.core.xla_model as xm
from transformers.trainer_pt_utils import AcceleratorConfig
from transformers.trainer_utils import (
    IntervalStrategy,
    SaveStrategy,
    SchedulerType,
    get_last_checkpoint,
)
from transformers.training_args import OptimizerNames, _convert_str_dict, default_logdir, trainer_log_levels

from ...utils import logging
from ..accelerate import NeuronAcceleratorState, NeuronPartialState
from ..accelerate.utils import patch_accelerate_is_torch_xla_available
from ..models.training.config import TrainingNeuronConfig
from ..models.training.training_utils import is_logging_process
from ..utils import is_main_worker


logger = logging.get_logger(__name__)

log_levels = dict(**trainer_log_levels, silent=100)


@dataclass
class NeuronTrainingArguments:
    # Sometimes users will pass in a `str` repr of a dict in the CLI
    # We need to track what fields those can be. Each time a new arg
    # has a dict type, it must be added to this list.
    # Important: These should be typed with Optional[Union[dict,str,...]]
    _VALID_DICT_FIELDS = [
        "accelerator_config",
        "gradient_checkpointing_kwargs",
        "lr_scheduler_kwargs",
    ]
    framework = "pt"

    # Transformers specific arguments
    output_dir: str | None = field(
        default=None,
        metadata={
            "help": "The output directory where the model predictions and checkpoints will be written. Defaults to 'trainer_output' if not provided."
        },
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )

    do_train: bool = field(default=False, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=False, metadata={"help": "Whether to run eval on the dev set."})
    eval_strategy: IntervalStrategy | str = field(
        default="no",
        metadata={"help": "The evaluation strategy to use."},
    )
    per_device_train_batch_size: int = field(
        default=1, metadata={"help": "Batch size per device accelerator for training."}
    )
    per_device_eval_batch_size: int = field(
        default=1, metadata={"help": "Batch size per device accelerator for evaluation."}
    )
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "Number of updates steps to accumulate before performing a backward/update pass."},
    )

    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate for AdamW."})
    weight_decay: float = field(default=0.0, metadata={"help": "Weight decay for AdamW if we apply some."})
    adam_beta1: float = field(default=0.9, metadata={"help": "Beta1 for AdamW optimizer"})
    adam_beta2: float = field(default=0.999, metadata={"help": "Beta2 for AdamW optimizer"})
    adam_epsilon: float = field(default=1e-8, metadata={"help": "Epsilon for AdamW optimizer."})
    max_grad_norm: float = field(default=1.0, metadata={"help": "Max gradient norm."})

    num_train_epochs: float = field(default=3.0, metadata={"help": "Total number of training epochs to perform."})
    max_steps: int = field(
        default=-1,
        metadata={"help": "If > 0: set total number of training steps to perform. Override num_train_epochs."},
    )
    lr_scheduler_type: SchedulerType | str = field(
        default="linear",
        metadata={"help": "The scheduler type to use."},
    )
    lr_scheduler_kwargs: dict[str, Any] | str | None = field(
        default_factory=dict,
        metadata={
            "help": (
                "Extra parameters for the lr_scheduler such as {'num_cycles': 1} for the cosine with hard restarts."
            )
        },
    )
    warmup_ratio: float = field(
        default=0.0, metadata={"help": "Linear warmup over warmup_ratio fraction of total steps."}
    )
    warmup_steps: int = field(default=0, metadata={"help": "Linear warmup over warmup_steps."})

    log_level: str = field(
        default="info",
        metadata={
            "help": (
                "Logger log level to use on the main node. Possible choices are the log levels as strings: 'debug',"
                " 'info', 'warning', 'error' and 'critical', plus a 'passive' level which doesn't set anything and"
                " lets the application set the level. Defaults to 'info'."
            ),
            "choices": log_levels.keys(),
        },
    )
    log_level_replica: str = field(
        default="silent",
        metadata={
            "help": "Logger log level to use on replica nodes. Same choices as ``log_level``. Defaults to 'silent'.",
            "choices": log_levels.keys(),
        },
    )
    logging_dir: str | None = field(default=None, metadata={"help": "Tensorboard log dir."})
    logging_strategy: IntervalStrategy | str = field(
        default="steps",
        metadata={"help": "The logging strategy to use."},
    )
    logging_first_step: bool = field(default=False, metadata={"help": "Log the first global_step"})
    logging_steps: float = field(
        default=500,
        metadata={
            "help": (
                "Log every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    save_strategy: SaveStrategy | str = field(
        default="steps",
        metadata={"help": "The checkpoint save strategy to use."},
    )
    save_steps: float = field(
        default=500,
        metadata={
            "help": (
                "Save checkpoint every X updates steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    save_total_limit: int | None = field(
        default=None,
        metadata={
            "help": (
                "If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in"
                " `output_dir`. When `load_best_model_at_end` is enabled, the 'best' checkpoint according to"
                " `metric_for_best_model` will always be retained in addition to the most recent ones. For example,"
                " for `save_total_limit=5` and `load_best_model_at_end=True`, the four last checkpoints will always be"
                " retained alongside the best model. When `save_total_limit=1` and `load_best_model_at_end=True`,"
                " it is possible that two checkpoints are saved: the last one and the best one (if they are different)."
                " Default is unlimited checkpoints"
            )
        },
    )
    save_only_model: bool = field(
        default=False,
        metadata={
            "help": (
                "When checkpointing, whether to only save the model, or also the optimizer, scheduler & rng state."
                "Note that when this is true, you won't be able to resume training from checkpoint."
                "This enables you to save storage by not storing the optimizer, scheduler & rng state."
                "You can only load the model using from_pretrained with this option set to True."
            )
        },
    )
    restore_callback_states_from_checkpoint: bool = field(
        default=False,
        metadata={
            "help": "Whether to restore the callback states from the checkpoint. If `True`, will override callbacks passed to the `Trainer` if they exist in the checkpoint."
        },
    )
    seed: int = field(default=42, metadata={"help": "Random seed that will be set at the beginning of training."})
    bf16: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
                " architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change."
            )
        },
    )
    dataloader_drop_last: bool = field(
        default=False, metadata={"help": "Drop the last incomplete batch if it is not divisible by the batch size."}
    )
    eval_steps: float | None = field(
        default=None,
        metadata={
            "help": (
                "Run an evaluation every X steps. Should be an integer or a float in range `[0,1)`. "
                "If smaller than 1, will be interpreted as ratio of total training steps."
            )
        },
    )
    dataloader_num_workers: int = field(
        default=0,
        metadata={
            "help": (
                "Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded"
                " in the main process."
            )
        },
    )
    dataloader_prefetch_factor: int | None = field(
        default=None,
        metadata={
            "help": (
                "Number of batches loaded in advance by each worker. "
                "2 means there will be a total of 2 * num_workers batches prefetched across all workers. "
            )
        },
    )

    run_name: str | None = field(
        default=None,
        metadata={
            "help": "An optional descriptor for the run. Notably used for wandb, mlflow comet and swanlab logging."
        },
    )
    disable_tqdm: bool | None = field(
        default=None, metadata={"help": "Whether or not to disable the tqdm progress bars."}
    )

    remove_unused_columns: bool | None = field(
        default=True, metadata={"help": "Remove columns not required by the model when using an nlp.Dataset."}
    )
    label_names: list[str] | None = field(
        default=None, metadata={"help": "The list of keys in your dictionary of inputs that correspond to the labels."}
    )
    accelerator_config: dict | str | None = field(
        default=None,
        metadata={
            "help": (
                "Config to be used with the internal Accelerator object initialization. The value is either a "
                "accelerator json config file (e.g., `accelerator_config.json`) or an already loaded json file as `dict`."
            )
        },
    )
    label_smoothing_factor: float = field(
        default=0.0, metadata={"help": "The label smoothing epsilon to apply (zero means no label smoothing)."}
    )

    default_optim = "adamw_torch"
    optim: OptimizerNames | str = field(
        default=default_optim,
        metadata={"help": "The optimizer to use."},
    )
    optim_args: str | None = field(default=None, metadata={"help": "Optional arguments to supply to optimizer."})
    report_to: None | str | list[str] = field(
        default=None, metadata={"help": "The list of integrations to report the results and logs to."}
    )
    resume_from_checkpoint: str | None = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for your model."},
    )
    gradient_checkpointing: bool = field(
        default=False,
        metadata={
            "help": "If True, use gradient checkpointing to save memory at the expense of slower backward pass."
        },
    )
    gradient_checkpointing_kwargs: dict[str, Any] | str | None = field(
        default=None,
        metadata={
            "help": "Gradient checkpointing key word arguments such as `use_reentrant`. Will be passed to `torch.utils.checkpoint.checkpoint` through `model.gradient_checkpointing_enable`."
        },
    )

    use_liger_kernel: bool | None = field(
        default=False,
        metadata={"help": "Whether or not to enable the Liger Kernel for model training."},
    )

    average_tokens_across_devices: bool | None = field(
        default=False,
        metadata={
            "help": "Whether or not to average tokens across devices. If enabled, will use all_reduce to "
            "synchronize num_tokens_in_batch for precise loss calculation. Reference: "
            "https://github.com/huggingface/transformers/issues/34242"
        },
    )

    # Neuron-specific arguments
    dataloader_prefetch_size: int = field(
        default=None,
        metadata={"help": ("The number of batches to prefetch on device.")},
    )
    skip_cache_push: bool = field(
        default=False, metadata={"help": "Whether to skip pushing Neuron artifacts to hub cache"}
    )
    use_autocast: bool = field(
        default=False,
        metadata={
            "help": "Whether to use torch autocast to perform mixed precision training.",
        },
    )
    zero_1: bool = field(default=True, metadata={"help": "Whether to use  ZeRO Stage 1 Optimization."})
    stochastic_rounding_enabled: bool = field(
        default=True,
        metadata={"help": "Whether to enable stochastic rounding when using bf16 training."},
    )
    optimizer_use_master_weights: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to use FP32 master weights for the optimizer when using bf16 training. "
                "This is more stable than bf16 training without master weights, at the cost of using more memory. "
                "It is only supported with the ZeRO-1 optimizer, and enabled by default when using bf16 with ZeRO-1. "
                "Set it to false to disable."
            )
        },
    )
    optimizer_use_fp32_grad_acc: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to use FP32 gradient accumulation when using bf16 training. This is more stable than bf16 "
                "training without FP32 gradient accumulation, at the cost of using more memory. It is only supported "
                "with the ZeRO-1 optimizer, and enabled by default when using bf16 with ZeRO-1. Set it to false to "
                "disable."
            )
        },
    )
    optimizer_save_master_weights_in_ckpt: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to save the FP32 master weights in the checkpoint when using bf16 training with "
                "optimizer_use_master_weights. Set it to true to enable."
            )
        },
    )
    tensor_parallel_size: int = field(
        default=1, metadata={"help": "The number of replicas the model will be sharded on."}
    )
    disable_sequence_parallel: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable sequence parallelism."},
    )
    pipeline_parallel_size: int = field(
        default=1,
        metadata={"help": "The number of pipeline parallel replicas."},
    )
    pipeline_parallel_num_microbatches: int = field(
        default=-1,
        metadata={"help": "The number of microbatches used for pipeline execution."},
    )
    kv_size_multiplier: int | None = field(
        default=None,
        metadata={
            "help": (
                "The number of times to replicate the KV heads when the TP size is bigger than the number of KV heads."
                "If left unspecified, the smallest multiplier that makes the number of KV heads divisible by the TP size"
                "will be used."
            )
        },
    )
    num_local_ranks_per_step: int = field(
        default=8,
        metadata={
            "help": (
                "The number of local ranks to use concurrently during checkpoiting, weight initialization and loading "
                "when tensor parallelism is enabled. By default, it is set to 8."
            )
        },
    )
    use_xser: bool = field(
        default=True,
        metadata={
            "help": "Whether to use `torch-xla` serialization when saving checkpoints when doing model parallelism"
        },
    )
    async_save: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use asynchronous saving method when doing model parallelism. It can boost saving "
                "performance but will result in more host memory usage, increasing the risk of going OOM."
            )
        },
    )
    fuse_qkv: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to fuse the query, key, and value linear layers in the self-attention layers. Only works if "
                "there is the same number of query and key/value heads."
            ),
        },
    )
    recompute_causal_mask: bool = field(
        default=True,
        metadata={
            "help": (
                "Whether to recompute the causal mask in the forward pass. This is more efficient than passing the  "
                "causal mask computed from the attention mask to the attention layers but it does not support custom "
                "attention masks."
            ),
        },
    )

    def __post_init__(self):
        # Set the verbosity so that each process logs according to its rank.
        log_level = self.get_process_log_level()
        logging.set_verbosity(log_level)

        # Set default output_dir if not provided
        if self.output_dir is None:
            self.output_dir = "trainer_output"
            logger.info(
                "No output directory specified, defaulting to 'trainer_output'. "
                "To change this behavior, specify --output_dir when creating TrainingArguments."
            )

        # Parse in args that could be `dict` sent in from the CLI as a string
        for dict_field in self._VALID_DICT_FIELDS:
            passed_value = getattr(self, dict_field)
            # We only want to do this if the str starts with a bracket to indicate a `dict`
            # else its likely a filename if supported
            if isinstance(passed_value, str) and passed_value.startswith("{"):
                loaded_dict = json.loads(passed_value)
                # Convert str values to types if applicable
                loaded_dict = _convert_str_dict(loaded_dict)
                setattr(self, dict_field, loaded_dict)

        # expand paths, if not os.makedirs("~/bar") will make directory
        # in the current directory instead of the actual home
        # see https://github.com/huggingface/transformers/issues/10628
        if self.output_dir is not None:
            self.output_dir = os.path.expanduser(self.output_dir)
        if self.logging_dir is None and self.output_dir is not None:
            self.logging_dir = os.path.join(self.output_dir, default_logdir())
        if self.logging_dir is not None:
            self.logging_dir = os.path.expanduser(self.logging_dir)

        if self.disable_tqdm is None:
            self.disable_tqdm = logger.getEffectiveLevel() > logging.WARN

        self.eval_strategy = IntervalStrategy(self.eval_strategy)
        self.logging_strategy = IntervalStrategy(self.logging_strategy)
        self.save_strategy = SaveStrategy(self.save_strategy)

        self.lr_scheduler_type = SchedulerType(self.lr_scheduler_type)
        if self.do_eval is False and self.eval_strategy != IntervalStrategy.NO:
            self.do_eval = True

        # eval_steps has to be defined and non-zero, fallbacks to logging_steps if the latter is non-zero
        if self.eval_strategy == IntervalStrategy.STEPS and (self.eval_steps is None or self.eval_steps == 0):
            if self.logging_steps > 0:
                logger.info(f"using `logging_steps` to initialize `eval_steps` to {self.logging_steps}")
                self.eval_steps = self.logging_steps
            else:
                raise ValueError(
                    f"evaluation strategy {self.eval_strategy} requires either non-zero --eval_steps or"
                    " --logging_steps"
                )

        # logging_steps must be non-zero for logging_strategy that is other than 'no'
        if self.logging_strategy == IntervalStrategy.STEPS and self.logging_steps == 0:
            raise ValueError(f"logging strategy {self.logging_strategy} requires non-zero --logging_steps")

        if self.logging_strategy == IntervalStrategy.STEPS and self.logging_steps > 1:
            if self.logging_steps != int(self.logging_steps):
                raise ValueError(f"--logging_steps must be an integer if bigger than 1: {self.logging_steps}")
            self.logging_steps = int(self.logging_steps)
        if self.eval_strategy == IntervalStrategy.STEPS and self.eval_steps > 1:
            if self.eval_steps != int(self.eval_steps):
                raise ValueError(f"--eval_steps must be an integer if bigger than 1: {self.eval_steps}")
            self.eval_steps = int(self.eval_steps)
        if self.save_strategy == SaveStrategy.STEPS and self.save_steps > 1:
            if self.save_steps != int(self.save_steps):
                raise ValueError(f"--save_steps must be an integer if bigger than 1: {self.save_steps}")
            self.save_steps = int(self.save_steps)

        if self.run_name is None:
            self.run_name = self.output_dir

        if self.lr_scheduler_type == SchedulerType.REDUCE_ON_PLATEAU:
            if self.eval_strategy == IntervalStrategy.NO:
                raise ValueError("lr_scheduler_type reduce_lr_on_plateau requires an eval strategy")

        self.optim = OptimizerNames(self.optim)

        # We need to setup the accelerator config here *before* the first call to `self.device`
        if not isinstance(self.accelerator_config, AcceleratorConfig):
            if self.accelerator_config is None:
                self.accelerator_config = AcceleratorConfig()
            elif isinstance(self.accelerator_config, dict):
                self.accelerator_config = AcceleratorConfig(**self.accelerator_config)
            # Check that a user didn't pass in the class instantiator
            # such as `accelerator_config = AcceleratorConfig`
            elif isinstance(self.accelerator_config, type):
                raise NotImplementedError(
                    "Tried passing in a callable to `accelerator_config`, but this is not supported. "
                    "Please pass in a fully constructed `AcceleratorConfig` object instead."
                )
            else:
                self.accelerator_config = AcceleratorConfig.from_json_file(self.accelerator_config)

        # Disable average tokens when using single device
        if self.average_tokens_across_devices:
            try:
                if self.world_size == 1:
                    logger.warning(
                        "average_tokens_across_devices is set to True but it is invalid when world size is"
                        "1. Turn it to False automatically."
                    )
                    self.average_tokens_across_devices = False
            except ImportError as e:
                logger.warning(f"Can not specify world size due to {e}. Turn average_tokens_across_devices to False.")
                self.average_tokens_across_devices = False

        # if training args is specified, it will override the one specified in the accelerate config
        if self.report_to is None:
            logger.info(
                "The default value for the training argument `--report_to` will change in v5 (from all installed "
                "integrations to none). In v5, you will need to use `--report_to all` to get the same behavior as "
                "now. You should start updating your code and make this info disappear :-)."
            )
            self.report_to = "all"
        if self.report_to == "all" or self.report_to == ["all"]:
            # Import at runtime to avoid a circular import.
            from transformers.integrations import get_available_reporting_integrations

            self.report_to = get_available_reporting_integrations()

            if "codecarbon" in self.report_to and torch.version.hip:
                logger.warning(
                    "When using the Trainer, CodeCarbonCallback requires the `codecarbon` package, which is not compatible with AMD ROCm (https://github.com/mlco2/codecarbon/pull/490). Automatically disabling the codecarbon callback. Reference: https://huggingface.co/docs/transformers/v4.39.3/en/main_classes/trainer#transformers.TrainingArguments.report_to."
                )
                self.report_to.remove("codecarbon")

        elif self.report_to == "none" or self.report_to == ["none"]:
            self.report_to = []
        elif not isinstance(self.report_to, list):
            self.report_to = [self.report_to]

        if self.warmup_ratio < 0 or self.warmup_ratio > 1:
            raise ValueError("warmup_ratio must lie in range [0,1]")
        elif self.warmup_ratio > 0 and self.warmup_steps > 0:
            logger.info(
                "Both warmup_ratio and warmup_steps given, warmup_steps will override any effect of warmup_ratio"
                " during training"
            )

        if not isinstance(self.warmup_steps, int) or self.warmup_steps < 0:
            raise ValueError("warmup_steps must be of type int and must be 0 or a positive integer.")

        if self.dataloader_num_workers == 0 and self.dataloader_prefetch_factor is not None:
            raise ValueError(
                "--dataloader_prefetch_factor can only be set when data is loaded in a different process, i.e."
                " when --dataloader_num_workers > 1."
            )

        if self.do_eval:
            raise RuntimeError("Evaluation is not supported yet.")

        # Neuron-specific setup

        # Patches accelerate.utils.imports.is_tpu_available to match `is_torch_xla_available`
        patch_accelerate_is_torch_xla_available()

        resume_from_checkpoint = self.resume_from_checkpoint
        if resume_from_checkpoint is None and self.output_dir is not None and os.path.isdir(self.output_dir):
            # If checkpoint is None, then there was no checkpoint in output dir, otherwise we use it.
            checkpoint = get_last_checkpoint(self.output_dir)
            resume_from_checkpoint = checkpoint

        if self.pipeline_parallel_size > 1:
            if self.gradient_accumulation_steps > 1:
                if is_main_worker():
                    logger.info(
                        "Pipeline parallel used, setting gradient_accumulation_steps to 1 and scaling the pipeline batch size."
                    )
                self.per_device_train_batch_size *= self.gradient_accumulation_steps
                self.per_device_eval_batch_size *= self.gradient_accumulation_steps
                self.gradient_accumulation_steps = 1
            if self.pipeline_parallel_num_microbatches == -1:
                self.pipeline_parallel_num_microbatches = self.per_device_train_batch_size
            if self.per_device_train_batch_size % self.pipeline_parallel_num_microbatches != 0:
                raise ValueError(
                    f"The number of pipeline microbatches ({self.pipeline_parallel_num_microbatches}) divide the total "
                    f"per-device train batch size ({self.per_device_train_batch_size})."
                )
            if self.per_device_eval_batch_size % self.pipeline_parallel_num_microbatches != 0:
                raise ValueError(
                    f"The number of pipeline microbatches ({self.pipeline_parallel_num_microbatches}) divide the total "
                    f"per-device eval batch size ({self.per_device_eval_batch_size})."
                )

        # It is not supported so disabling it
        if not self.zero_1:
            self.optimizer_use_master_weights = False
            self.optimizer_use_fp32_grad_acc = False
            self.optimizer_save_master_weights_in_ckpt = False

        self.trn_config = TrainingNeuronConfig(
            self.tensor_parallel_size,
            sequence_parallel_enabled=not self.disable_sequence_parallel,
            kv_size_multiplier=self.kv_size_multiplier,
            pipeline_parallel_size=self.pipeline_parallel_size,
            pipeline_parallel_num_microbatches=self.pipeline_parallel_num_microbatches,
            pipeline_parallel_use_zero1_optimizer=self.zero_1,
            checkpoint_dir=resume_from_checkpoint,
            num_local_ranks_per_step=self.num_local_ranks_per_step,
            use_xser=self.use_xser,
            async_save=self.async_save,
            fuse_qkv=self.fuse_qkv,
            recompute_causal_mask=self.recompute_causal_mask,
            gradient_checkpointing=self.gradient_checkpointing,
        )

        if self.bf16 and self.use_autocast:
            os.environ["ACCELERATE_USE_AMP"] = "true"
        else:
            os.environ["ACCELERATE_USE_AMP"] = "false"

        # Setup the device here
        self._setup_devices

    @cached_property
    def _setup_devices(self) -> "torch.device":
        logger.info("PyTorch: setting up devices")

        # Initialize the accelerator state first
        accelerator_state_kwargs: dict[str, Any] = {"enabled": True, "use_configured_state": False}
        if isinstance(self.accelerator_config, AcceleratorConfig):
            accelerator_state_kwargs["use_configured_state"] = self.accelerator_config.pop(
                "use_configured_state", False
            )
        if accelerator_state_kwargs["use_configured_state"]:
            if NeuronPartialState._shared_state == {}:
                raise ValueError(
                    "Passing `'use_configured_state':True` to the AcceleratorConfig requires a pre-configured "
                    "`AcceleratorState` or `PartialState` to be defined before calling `TrainingArguments`. "
                )
            # We rely on `PartialState` to yell if there's issues here (which it will)
            self.distributed_state = NeuronPartialState(cpu=False)
        else:
            NeuronAcceleratorState._reset_state(reset_partial_state=True)
            self.distributed_state = None

        device = xm.xla_device()
        return device

    @property
    def device(self) -> "torch.device":
        """
        The device used by this process.
        """
        return self._setup_devices

    @property
    def world_size(self):
        return self.trn_config.data_parallel_size

    @property
    def process_index(self):
        """
        The index of the current process used.
        """
        if self.distributed_state is not None:
            return self.distributed_state.process_index
        return 0

    @property
    def local_process_index(self):
        """
        The index of the local process used.
        """
        if self.distributed_state is not None:
            return self.distributed_state.local_process_index
        return 0

    @property
    def train_batch_size(self) -> int:
        """
        The actual batch size for training.
        """
        return self.per_device_train_batch_size * self.trn_config.data_parallel_size

    @property
    def should_log(self):
        """
        Whether or not the current process should produce log.
        """
        return is_logging_process()

    @property
    def should_save(self):
        """
        Whether or not the current process should write to disk, e.g., to save models and checkpoints.
        """
        return self.process_index == 0

    def get_process_log_level(self):
        """
        Returns the log level to be used depending on whether this process is the main process of node 0, main process
        of node non-0, or a non-main process.

        For the main process the log level defaults to the logging level set (`logging.WARNING` if you didn't do
        anything) unless overridden by `log_level` argument.

        For the replica processes the log level defaults to `logging.WARNING` unless overridden by `log_level_replica`
        argument.

        The choice between the main and replica process settings is made according to the return value of `should_log`.
        """

        # convert to int
        log_level = log_levels[self.log_level]
        log_level_replica = log_levels[self.log_level_replica]

        log_level_main_node = logging.get_verbosity() if log_level == -1 else log_level
        log_level_replica_node = logging.get_verbosity() if log_level_replica == -1 else log_level_replica
        return log_level_main_node if self.should_log else log_level_replica_node

    def get_warmup_steps(self, num_training_steps: int):
        """
        Get number of steps used for a linear warmup.
        """
        warmup_steps = (
            self.warmup_steps if self.warmup_steps > 0 else math.ceil(num_training_steps * self.warmup_ratio)
        )
        return warmup_steps

    def _dict_torch_dtype_to_str(self, d: dict[str, Any]) -> None:
        """Checks whether the passed dictionary and its nested dicts have a *torch_dtype* key and if it's not None,
        converts torch.dtype to a string of just the type. For example, `torch.float32` get converted into *"float32"*
        string, which can then be stored in the json format.
        """
        if d.get("dtype", None) is not None and not isinstance(d["dtype"], str):
            d["dtype"] = str(d["dtype"]).split(".")[1]
        for value in d.values():
            if isinstance(value, dict):
                self._dict_torch_dtype_to_str(value)

    def to_dict(self):
        """
        Serializes this instance while replace `Enum` by their values (for JSON serialization support). It obfuscates
        the token values by removing their value.
        """
        # filter out fields that are defined as field(init=False)
        d = {field.name: getattr(self, field.name) for field in fields(self) if field.init}

        for k, v in d.items():
            if isinstance(v, Enum):
                d[k] = v.value
            if isinstance(v, list) and len(v) > 0 and isinstance(v[0], Enum):
                d[k] = [x.value for x in v]
            if k.endswith("_token"):
                d[k] = f"<{k.upper()}>"
            # Handle the accelerator_config if passed
            if isinstance(v, AcceleratorConfig):
                d[k] = v.to_dict()
            # Handle the quantization_config if passed
            if k == "model_init_kwargs" and isinstance(v, dict) and "quantization_config" in v:
                quantization_config = v.get("quantization_config")
                if quantization_config and not isinstance(quantization_config, dict):
                    d[k]["quantization_config"] = quantization_config.to_dict()
        self._dict_torch_dtype_to_str(d)

        return d

    def to_json_string(self):
        """
        Serializes this instance to a JSON string.
        """
        return json.dumps(self.to_dict(), indent=2)

    def to_sanitized_dict(self) -> dict[str, Any]:
        """
        Sanitized serialization to use with TensorBoard’s hparams
        """
        d = self.to_dict()
        d = {**d, **{"train_batch_size": self.train_batch_size}}

        valid_types = [bool, int, float, str, torch.Tensor]
        return {k: v if type(v) in valid_types else str(v) for k, v in d.items()}
