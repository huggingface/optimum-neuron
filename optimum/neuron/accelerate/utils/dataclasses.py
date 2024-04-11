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
"""Custom dataclasses for Neuron."""

import enum
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import torch

from ...distributed import ParallelizersManager


if TYPE_CHECKING:
    from transformers import PreTrainedModel


class NeuronDistributedType(str, enum.Enum):
    """
    Represents a type of distributed environment specific to Neuron.

    Values:
        - **MODEL_PARALLELISM** -- Tensor and Pipeline Parallelisms using `torch_xla` and `neuronx_distributed`.
    """

    MODEL_PARALLELISM = "MODEL_PARALLELISM"


class AutocastBackend(str, enum.Enum):
    """
    Represents the backend to use for mixed-precision training.
    """

    XLA = "xla"
    AMP = "amp"


@dataclass
class ModelParallelismPlugin:
    tensor_parallel_size: int = 1
    parallelize_embeddings: bool = True
    sequence_parallel_enabled: bool = False
    kv_size_multiplier: Optional[int] = None
    pipeline_parallel_size: int = 1
    pipeline_parallel_num_microbatches: int = 1
    pipeline_parallel_use_zero1_optimizer: bool = False
    gradient_checkpointing: bool = False
    checkpoint_dir: Optional[Union[str, Path]] = None
    num_local_ranks_per_step: int = 8
    use_xser: bool = True
    async_save: bool = False

    def __post_init__(self):
        if self.tensor_parallel_size < 1:
            raise ValueError(f"The tensor parallel size must be >= 1, but {self.tensor_parallel_size} was given here.")
        if self.pipeline_parallel_size < 1:
            raise ValueError(
                f"The pipeline parallel size must be >= 1, but {self.pipeline_parallel_size} was given here."
            )
        if isinstance(self.checkpoint_dir, str):
            self.checkpoint_dir = Path(self.checkpoint_dir)

    @property
    def should_parallelize(self):
        return self.tensor_parallel_size > 1 or self.pipeline_parallel_size > 1

    def parallelize_model(
        self,
        model: "PreTrainedModel",
        device: Optional["torch.device"] = None,
    ) -> Union["PreTrainedModel", Tuple["PreTrainedModel", Dict[int, "torch.nn.Parameter"]]]:
        parallelizer = ParallelizersManager.parallelizer_for_model(model)
        parallelized_model = parallelizer.parallelize(
            model,
            device=device,
            parallelize_embeddings=self.parallelize_embeddings,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            kv_size_multiplier=self.kv_size_multiplier,
            pipeline_parallel_num_microbatches=self.pipeline_parallel_num_microbatches,
            pipeline_parallel_use_zero1_optimizer=self.pipeline_parallel_use_zero1_optimizer,
            pipeline_parallel_gradient_checkpointing_enabled=self.gradient_checkpointing,
            checkpoint_dir=self.checkpoint_dir,
            num_local_ranks_per_step=self.num_local_ranks_per_step,
        )
        return parallelized_model
