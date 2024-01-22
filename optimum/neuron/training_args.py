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

import io
import json
import os
import warnings
from dataclasses import dataclass, field
from datetime import timedelta

import torch
from accelerate.utils import DistributedType
from packaging import version
from transformers.trainer_utils import get_last_checkpoint
from transformers.training_args import ParallelMode, TrainingArguments
from transformers.training_args_seq2seq import Seq2SeqTrainingArguments
from transformers.utils import (
    cached_property,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    requires_backends,
)

from ..utils import check_if_transformers_greater, logging
from .accelerate import NeuronAcceleratorState, NeuronPartialState
from .accelerate.utils import ModelParallelismPlugin, patch_accelerate_is_tpu_available
from .utils import is_accelerate_available, is_torch_xla_available
from .utils.training_utils import TRANSFORMERS_MIN_VERSION_FOR_XLA_FSDP


if is_sagemaker_mp_enabled():
    import smdistributed.modelparallel.torch as smp

    smp.init()


logger = logging.get_logger(__name__)


@dataclass
class NeuronTrainingArgumentsMixin:
    skip_cache_push: bool = field(
        default=False, metadata={"help": "Whether to skip pushing Neuron artifacts to hub cache"}
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
    neuron_cc_optlevel: str = field(
        default="auto",
        metadata={
            "choices": ["auto", "1", "2", "3"],
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

    def __post_init__(self):
        # Patches accelerate.utils.imports.is_tpu_available to match `is_torch_xla_available`
        patch_accelerate_is_tpu_available()

        if self.fsdp != "":
            # Disabling FSDP until next release because it is still very experimental and not validated.
            raise RuntimeError("FSDP is not supported yet.")
            if self.fsdp_config is None:
                self.fsdp_config = {"xla": True}
            elif isinstance(self.fsdp_config, str):
                with io.open(self.fsdp_config, "r", encoding="utf-8") as f:
                    self.fsdp_config = json.load(f)

            if "xla" in self.fsdp_config and not self.fsdp_config["xla"]:
                raise ValueError(
                    "XLA FSDP is the only supported FSDP implementation by `optimum-neuron` but the provided FSDP "
                    "config specified it should not be used."
                )
            else:
                self.fsdp_config["xla"] = True

            os.environ["ACCELERATE_USE_FSDP"] = "true"

            if not check_if_transformers_greater(TRANSFORMERS_MIN_VERSION_FOR_XLA_FSDP):
                import transformers

                raise RuntimeError(
                    "The minimal required Transformers version to perform XLA FSDP is "
                    f"{TRANSFORMERS_MIN_VERSION_FOR_XLA_FSDP} but {transformers.__version__} is installed."
                )
        if self.neuron_cc_optlevel != "auto":
            self.neuron_cc_optlevel = f"-O{self.neuron_cc_optlevel}"

        resume_from_checkpoint = self.resume_from_checkpoint
        if resume_from_checkpoint is None and os.path.isdir(self.output_dir):
            # If checkpoint is None, then there was no checkpoint in output dir, otherwise we use it.
            checkpoint = get_last_checkpoint(self.output_dir)
            resume_from_checkpoint = checkpoint

        if self.pipeline_parallel_size > 1:
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
            pipeline_parallel_size=self.pipeline_parallel_size,
            pipeline_parallel_num_microbatches=self.pipeline_parallel_num_microbatches,
            pipeline_parallel_use_zero1_optimizer=self.zero_1,
            checkpoint_dir=resume_from_checkpoint,
        )
        super().__post_init__()

    # Needed only to specialize the warning message for FSDP.
    @cached_property
    def _setup_devices(self) -> "torch.device":
        if not check_if_transformers_greater("4.30.0"):
            return super()._setup_devices

        requires_backends(self, ["torch"])
        logger.info("PyTorch: setting up devices")
        NeuronAcceleratorState._reset_state()
        NeuronPartialState._reset_state()
        if not is_sagemaker_mp_enabled() and not is_accelerate_available():
            raise ImportError(
                "Using the `Trainer` with `PyTorch` requires `accelerate>=0.20.1`: Please run `pip install "
                "transformers[torch]` or `pip install accelerate -U`"
            )
        self.distributed_state = None
        if self.no_cuda:
            self.distributed_state = NeuronPartialState(cpu=True, backend=self.ddp_backend)
            self._n_gpu = 0
        elif is_sagemaker_mp_enabled():
            local_rank = smp.local_rank()
            device = torch.device("cuda", local_rank)
            self._n_gpu = 1
            torch.cuda.set_device(device)
        elif is_sagemaker_dp_enabled():
            self.distributed_state = NeuronPartialState(_use_sagemaker_dp=True)
            self._n_gpu = 1
        elif self.deepspeed:
            # Need to do similar for Accelerator init
            os.environ["ACCELERATE_USE_DEEPSPEED"] = "true"
            self.distributed_state = NeuronPartialState(timeout=timedelta(seconds=self.ddp_timeout))
            del os.environ["ACCELERATE_USE_DEEPSPEED"]
            self._n_gpu = 1
        else:
            self.distributed_state = NeuronPartialState(backend=self.ddp_backend)
            self._n_gpu = 1
        if not is_sagemaker_mp_enabled():
            device = self.distributed_state.device
            self.local_rank = self.distributed_state.local_process_index
        if (
            torch.distributed.is_available()
            and torch.distributed.is_initialized()
            and self.parallel_mode not in [ParallelMode.DISTRIBUTED, ParallelMode.TPU]
        ):
            logger.warning(
                "torch.distributed process group is initialized, but parallel_mode != ParallelMode.DISTRIBUTED and "
                "parallel_mode != ParallelMode.TPU. "
                "In order to use Torch DDP / XLA FSDP, launch your script with `python -m torch.distributed.launch"
            )
        if is_torch_xla_available():
            device = self.distributed_state.device
            self._n_gpu = 0
        elif is_sagemaker_dp_enabled() or is_sagemaker_mp_enabled():
            # Already set _n_gpu
            pass
        elif self.distributed_state.distributed_type == DistributedType.NO:
            if self.use_mps_device:
                if not torch.backends.mps.is_available():
                    if not torch.backends.mps.is_built():
                        raise AssertionError(
                            "MPS not available because the current PyTorch install was not "
                            "built with MPS enabled. Please install torch version >=1.12.0 on "
                            "your Apple silicon Mac running macOS 12.3 or later with a native "
                            "version (arm64) of Python"
                        )
                    else:
                        raise AssertionError(
                            "MPS not available because the current MacOS version is not 12.3+ "
                            "and/or you do not have an MPS-enabled device on this machine."
                        )
                else:
                    if not version.parse(version.parse(torch.__version__).base_version) > version.parse("1.12.0"):
                        warnings.warn(
                            "We strongly recommend to install PyTorch >= 1.13 (nightly version at the time of writing)"
                            " on your MacOS machine. It has major fixes related to model correctness and performance"
                            " improvements for transformer based models. Please refer to"
                            " https://github.com/pytorch/pytorch/issues/82707 for more details."
                        )
                    device = torch.device("mps")
                    self._n_gpu = 1
            elif self.no_cuda:
                device = torch.device("cpu")
                self._n_gpu = 0
            else:
                # if n_gpu is > 1 we'll use nn.DataParallel.
                # If you only want to use a specific subset of GPUs use `CUDA_VISIBLE_DEVICES=0`
                # Explicitly set CUDA to the first (index 0) CUDA device, otherwise `set_device` will
                # trigger an error that a device index is missing. Index 0 takes into account the
                # GPUs available in the environment, so `CUDA_VISIBLE_DEVICES=1,2` with `cuda:0`
                # will use the first GPU in that env, i.e. GPU#1
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                # Sometimes the line in the postinit has not been run before we end up here, so just checking we're not at
                # the default value.
                self._n_gpu = torch.cuda.device_count()
                if device.type == "cuda":
                    torch.cuda.set_device(device)
        return device

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
