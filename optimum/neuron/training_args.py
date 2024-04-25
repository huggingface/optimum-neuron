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
from dataclasses import dataclass, field
from typing import Optional

import torch
from transformers.trainer_utils import get_last_checkpoint
from transformers.training_args import TrainingArguments
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.utils import (
    cached_property,
    is_sagemaker_mp_enabled,
)

from ..utils import logging
from .accelerate import NeuronAcceleratorState, NeuronPartialState
from .accelerate.utils import ModelParallelismPlugin, patch_accelerate_is_tpu_available
from .utils import is_main_worker
from .utils.patching import Patcher, patch_within_function
from .utils.torch_xla_and_neuronx_initialization import set_neuron_cc_optlevel


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

    smp.init()


logger = logging.get_logger(__name__)


@dataclass
class NeuronTrainingArgumentsMixin:
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
    disable_embedding_parallelization: bool = field(
        default=False,
        metadata={
            "help": (
                "If set, the embeddings will not be parallelized when doing model parallelism. When embeddings are not "
                "parallelized in decoder and seq2seq models, the language modeling head cannot be parallelized either "
                "or need an all-gather, which can be costly."
            )
        },
    )
    disable_sequence_parallel: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable sequence parallelism."},
    )
    neuron_cc_optlevel: int = field(
        default=2,
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
    kv_size_multiplier: Optional[int] = field(
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

    def __post_init__(self):
        # Patches accelerate.utils.imports.is_tpu_available to match `is_torch_xla_available`
        patch_accelerate_is_tpu_available()

        if self.fsdp != "":
            raise RuntimeError("FSDP is not supported.")

        if self.fp16:
            raise ValueError("The fp16 data type is not supported in Neuron, please use bf16 instead.")

        resume_from_checkpoint = self.resume_from_checkpoint
        if resume_from_checkpoint is None and os.path.isdir(self.output_dir):
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

        self.mp_plugin = ModelParallelismPlugin(
            self.tensor_parallel_size,
            parallelize_embeddings=not self.disable_embedding_parallelization,
            sequence_parallel_enabled=not self.disable_sequence_parallel,
            kv_size_multiplier=self.kv_size_multiplier,
            pipeline_parallel_size=self.pipeline_parallel_size,
            pipeline_parallel_num_microbatches=self.pipeline_parallel_num_microbatches,
            pipeline_parallel_use_zero1_optimizer=self.zero_1,
            gradient_checkpointing=self.gradient_checkpointing,
            checkpoint_dir=resume_from_checkpoint,
            num_local_ranks_per_step=self.num_local_ranks_per_step,
            use_xser=self.use_xser,
            async_save=self.async_save,
        )

        if self.bf16 and self.half_precision_backend == "amp":
            os.environ["ACCELERATE_USE_AMP"] = "true"
        else:
            os.environ["ACCELERATE_USE_AMP"] = "false"

        set_neuron_cc_optlevel(self.neuron_cc_optlevel)

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
    def place_model_on_device(self):
        return not self.mp_plugin.should_parallelize and super().place_model_on_device

    @property
    def world_size(self):
        divisor = 1
        if self.mp_plugin.should_parallelize:
            divisor = self.mp_plugin.tensor_parallel_size * self.mp_plugin.pipeline_parallel_size
        return super().world_size // divisor


@dataclass
class NeuronTrainingArguments(NeuronTrainingArgumentsMixin, TrainingArguments):
    pass


@dataclass
class Seq2SeqNeuronTrainingArguments(NeuronTrainingArgumentsMixin, Seq2SeqTrainingArguments):
    pass
