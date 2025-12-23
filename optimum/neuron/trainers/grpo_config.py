# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass, field

from ..utils.import_utils import is_trl_available
from .training_args import NeuronTrainingArguments
from .trl_utils import TRL_VERSION


if is_trl_available():
    from trl import GRPOConfig
else:

    @dataclass
    class GRPOConfig:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(f"You need to install the `trl=={TRL_VERSION}` library to use the `NeuronGRPOConfig`.")


@dataclass
class NeuronGRPOConfig(NeuronTrainingArguments, GRPOConfig):
    """
    Configuration class for Neuron-optimized GRPO training.

    This class combines NeuronTrainingArguments for Trainium-specific settings
    with GRPOConfig for GRPO algorithm parameters.
    """

    use_vllm: bool = field(
        default=True,
        metadata={
            "help": "Whether to use vLLM for generating completions. If set to `True`, the trainer will use vLLM for "
            "generation instead of the default model.generate(). Requires `vllm` to be installed. Required for NeuronGRPOTrainer."
        },
    )

    def __post_init__(self):
        # For now, NeuronGRPOTrainer requires vLLM for generation, no other way is supported.
        if not self.use_vllm:
            raise ValueError("NeuronGRPOTrainer requires `use_vllm` to be set to `True`.")

        # Handle bf16 default (from GRPOConfig)
        self.bf16 = not (self.fp16) if self.bf16 is None else self.bf16

        # Call NeuronTrainingArguments.__post_init__ to initialize Neuron-specific settings
        NeuronTrainingArguments.__post_init__(self)

        # Convert scale_rewards boolean to string (from GRPOConfig)
        self.scale_rewards = {True: "group", False: "none"}.get(self.scale_rewards, self.scale_rewards)

        num_processes = self.world_size
        # The current default effective batch size
        if self.generation_batch_size is None and self.steps_per_generation is None:
            self.steps_per_generation = self.gradient_accumulation_steps
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
        elif self.generation_batch_size is not None and self.steps_per_generation is None:
            # Just ensure the value is divisible by the global batch size
            if self.generation_batch_size % (self.per_device_train_batch_size * num_processes) != 0:
                raise ValueError(
                    f"generation_batch_size ({self.generation_batch_size}) must be divisible by the global batch size "
                    f"({self.per_device_train_batch_size * num_processes})."
                )
            self.steps_per_generation = self.generation_batch_size // (
                self.per_device_train_batch_size * num_processes
            )
        elif self.generation_batch_size is None and self.steps_per_generation is not None:
            self.generation_batch_size = self.per_device_train_batch_size * num_processes * self.steps_per_generation
        else:
            raise ValueError(
                "'generation_batch_size' and 'steps_per_generation' can not be both configured at the same time"
            )

        if self.do_eval and self.eval_strategy != "no":
            # Just ensure the value is divisible by the global batch size
            if (self.per_device_eval_batch_size * num_processes) % self.num_generations != 0:
                raise ValueError(
                    f"The global eval batch size ({self.per_device_eval_batch_size} * {num_processes}) must be "
                    f"divisible by num_generations ({self.num_generations})."
                )

        # The generation batch must contain full prompt groups (no partials), so it must be divisible by
        # num_generations.
        if self.generation_batch_size % self.num_generations != 0:
            raise ValueError(
                f"generation_batch_size ({self.generation_batch_size}) must be divisible by num_generations "
                f"({self.num_generations})."
            )

        if self.num_generations < 2:
            raise ValueError(
                "GRPO requires at least 2 generations per prompt to calculate the advantages. You provided "
                f"{self.num_generations}, which is less than the minimum required."
            )

        if self.delta is not None and self.use_liger_loss:
            raise ValueError("Liger loss does not support two-sided GRPO loss yet.")
