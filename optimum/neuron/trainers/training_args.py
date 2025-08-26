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
"""Defines a TrainingArguments class compatible with Neuron."""

import os
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
from transformers.trainer_utils import get_last_checkpoint
from transformers.training_args import TrainingArguments
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.utils import (
    cached_property,
    is_sagemaker_mp_enabled,
)

from ...utils import logging
from ..accelerate import NeuronAcceleratorState, NeuronPartialState
from ..accelerate.utils import patch_accelerate_is_torch_xla_available
from ..models.training.config import TrainingNeuronConfig
from ..utils import is_main_worker
from ..utils.patching import Patcher, patch_within_function
from ..utils.torch_xla_and_neuronx_initialization import set_neuron_cc_optlevel


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

    smp.init()


logger = logging.get_logger(__name__)


@dataclass
class NeuronTrainingArgumentsMixin:
    # Sometimes users will pass in a `str` repr of a dict in the CLI
    # We need to track what fields those can be. Each time a new arg
    # has a dict type, it must be added to this list.
    # Important: These should be typed with Optional[Union[dict,str,...]]
    _VALID_DICT_FIELDS = [
        "accelerator_config",
        "fsdp_config",
        "deepspeed",
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
        default="passive",
        metadata={
            "help": (
                "Logger log level to use on the main node. Possible choices are the log levels as strings: 'debug',"
                " 'info', 'warning', 'error' and 'critical', plus a 'passive' level which doesn't set anything and"
                " lets the application set the level. Defaults to 'passive'."
            ),
            "choices": trainer_log_levels.keys(),
        },
    )
    log_level_replica: str = field(
        default="warning",
        metadata={
            "help": "Logger log level to use on replica nodes. Same choices and defaults as ``log_level``",
            "choices": trainer_log_levels.keys(),
        },
    )
    log_on_each_node: bool = field(
        default=True,
        metadata={
            "help": (
                "When doing a multinode distributed training, whether to log once per node or just once on the main"
                " node."
            )
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
    data_seed: int | None = field(default=None, metadata={"help": "Random seed to be used with data samplers."})
    bf16: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use bf16 (mixed) precision instead of 32-bit. Requires Ampere or higher NVIDIA"
                " architecture or using CPU (use_cpu) or Ascend NPU. This is an experimental API and it may change."
            )
        },
    )
    local_rank: int = field(default=-1, metadata={"help": "For distributed training: local_rank"})

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
    ignore_data_skip: bool = field(
        default=False,
        metadata={
            "help": (
                "When resuming training, whether or not to skip the first epochs and batches to get to the same"
                " training data."
            )
        },
    )
    accelerator_config: dict | str | None= field(
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
    adafactor: bool = field(default=False, metadata={"help": "Whether or not to replace AdamW by Adafactor."})
    report_to: None | str | list[str] = field(
        default=None, metadata={"help": "The list of integrations to report the results and logs to."}
    )
    dataloader_pin_memory: bool = field(
        default=True, metadata={"help": "Whether or not to pin memory for DataLoader."}
    )
    dataloader_persistent_workers: bool = field(
        default=False,
        metadata={
            "help": "If True, the data loader will not shut down the worker processes after a dataset has been consumed once. This allows to maintain the workers Dataset instances alive. Can potentially speed up training, but will increase RAM usage."
        },
    )
    skip_memory_metrics: bool = field(
        default=True, metadata={"help": "Whether or not to skip adding of memory profiler reports to metrics."}
    )
    push_to_hub: bool = field(
        default=False, metadata={"help": "Whether or not to upload the trained model to the model hub after training."}
    )
    resume_from_checkpoint: str | None = field(
        default=None,
        metadata={"help": "The path to a folder with a valid checkpoint for your model."},
    )
    hub_model_id: str | None = field(
        default=None, metadata={"help": "The name of the repository to keep in sync with the local `output_dir`."}
    )
    hub_strategy: HubStrategy | str = field(
        default="every_save",
        metadata={"help": "The hub strategy to use when `--push_to_hub` is activated."},
    )
    hub_token: str | None = field(default=None, metadata={"help": "The token to use to push to the Model Hub."})
    hub_private_repo: bool | None = field(
        default=None,
        metadata={
            "help": "Whether to make the repo private. If `None` (default), the repo will be public unless the organization's default is private. This value is ignored if the repo already exists."
        },
    )
    hub_always_push: bool = field(
        default=False,
        metadata={"help": "Unless `True`, the Trainer will skip pushes if the previous one wasn't finished yet."},
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
    include_inputs_for_metrics: bool = field(
        default=False,
        metadata={
            "help": "This argument is deprecated and will be removed in version 5 of 🤗 Transformers. Use `include_for_metrics` instead."
        },
    )
    include_for_metrics: list[str] = field(
        default_factory=list,
        metadata={
            "help": "List of strings to specify additional data to include in the `compute_metrics` function."
            "Options: 'inputs', 'loss'."
        },
    )
    eval_do_concat_batches: bool = field(
        default=True,
        metadata={
            "help": "Whether to recursively concat inputs/losses/labels/predictions across batches. If `False`, will instead store them as lists, with each batch kept separate."
        },
    )

    include_tokens_per_second: bool | None = field(
        default=False,
        metadata={"help": "If set to `True`, the speed metrics will include `tgs` (tokens per second per device)."},
    )

    include_num_input_tokens_seen: bool | None = field(
        default=False,
        metadata={
            "help": "If set to `True`, will track the number of input tokens seen throughout training. (May be slower in distributed training)"
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
    skip_cache_push: bool = field(
        default=False, metadata={"help": "Whether to skip pushing Neuron artifacts to hub cache"}
    )
    half_precision_backend: str = field(
        default="xla",
        metadata={
            "help": "The backend to be used for half precision.",
            "choices": ["xla", "amp"],
        },
    )
    zero_1: bool = field(default=False, metadata={"help": "Whether to use  ZeRO Stage 1 Optimization."})
    tensor_parallel_size: int = field(
        default=1, metadata={"help": "The number of replicas the model will be sharded on."}
    )
    disable_sequence_parallel: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable sequence parallelism."},
    )
    neuron_cc_optlevel: int | None = field(
        default=None,
        metadata={
            "choices": [1, 2, 3],
            "help": "Specify the level of optimization the Neuron compiler should perform.",
        },
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
        if self.do_eval:
            raise RuntimeError("Evaluation is not supported yet.")

        if self.neuron_cc_flags_model_type is not None:
            os.environ["OPTIMUM_NEURON_COMMON_FLAGS_MODEL_TYPE"] = self.neuron_cc_flags_model_type

        # Patches accelerate.utils.imports.is_tpu_available to match `is_torch_xla_available`
        patch_accelerate_is_torch_xla_available()

        if self.fsdp not in ["", []]:
            raise RuntimeError("FSDP is not supported.")

        if self.fp16:
            raise ValueError("The fp16 data type is not supported in Neuron, please use bf16 instead.")

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

        if self.bf16 and self.half_precision_backend == "amp":
            os.environ["ACCELERATE_USE_AMP"] = "true"
        else:
            os.environ["ACCELERATE_USE_AMP"] = "false"

        if self.neuron_cc_optlevel is not None:
            set_neuron_cc_optlevel(self.neuron_cc_optlevel)

        self._world_size_should_behave_as_dp_size = False

        # This is required to be able to use bf16, otherwise a check in super().__post_init__() fails.
        with Patcher([("transformers.training_args.get_xla_device_type", lambda _: "GPU")]):
            super().__post_init__()

    @cached_property
    @patch_within_function(
        [
            ("transformers.training_args.PartialState", NeuronPartialState),
            ("transformers.training_args.AcceleratorState", NeuronAcceleratorState),
        ]
    )
    def _setup_devices(self) -> "torch.device":
        return super()._setup_devices

    @property
    def neuron_cc_flags_model_type(self) -> str | None:
        """Controls the value to provide to the Neuron Compiler for the model-type flag."""
        return "transformer"

    @property
    def place_model_on_device(self):
        return not self.trn_config.should_parallelize and super().place_model_on_device

    @property
    def world_size_should_behave_as_dp_size(self):
        return self._world_size_should_behave_as_dp_size

    @world_size_should_behave_as_dp_size.setter
    def world_size_should_behave_as_dp_size(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError(
                f"world_size_should_behave_as_dp_size should be a boolean, but a {type(value)} was provided here."
            )
        self._world_size_should_behave_as_dp_size = value

    @property
    def dp_size(self):
        divisor = 1
        if self.trn_config.should_parallelize:
            divisor = self.trn_config.tensor_parallel_size * self.trn_config.pipeline_parallel_size
        return super().world_size // divisor

    @property
    def world_size(self):
        if self.world_size_should_behave_as_dp_size:
            return self.dp_size
        return super().world_size

    @contextmanager
    def world_size_as_dp_size(self):
        orig_state = self.world_size_should_behave_as_dp_size
        self.world_size_should_behave_as_dp_size = True
        try:
            yield
        finally:
            self.world_size_should_behave_as_dp_size = orig_state


@dataclass
class NeuronTrainingArguments(NeuronTrainingArgumentsMixin, TrainingArguments):
    pass


@dataclass
class Seq2SeqNeuronTrainingArguments(NeuronTrainingArgumentsMixin, Seq2SeqTrainingArguments):
    pass
