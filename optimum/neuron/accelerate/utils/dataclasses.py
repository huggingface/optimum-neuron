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
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import torch
from accelerate.utils.constants import MODEL_NAME, OPTIMIZER_NAME
from accelerate.utils.dataclasses import FullyShardedDataParallelPlugin

from ...distributed import ParallelizersManager
from ...utils import is_torch_xla_available


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as FSDP
    from torch_xla.distributed.fsdp.state_dict_utils import consolidate_sharded_model_checkpoints

if TYPE_CHECKING:
    from transformers import PreTrainedModel


class NeuronDistributedType(str, enum.Enum):
    """
    Represents a type of distributed environment specific to Neuron.

    Values:
        - **XLA_FSDP** -- Fully Shareded Data Parallelism on Neuron cores using `torch_xla`.
    """

    XLA_FSDP = "XLA_FSDP"
    MODEL_PARALLELISM = "MODEL_PARALLELISM"


@dataclass
class NeuronFullyShardedDataParallelPlugin(FullyShardedDataParallelPlugin):
    # TODO: redefine the post init to do checks on which option is supported.
    def save_model(self, accelerator, model, output_dir, model_index=0):
        from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

        state_dict = {"model": model.state_dict(), "shard_metadata": model.get_shard_metadata()}
        weights_name = (
            f"{MODEL_NAME}_rank{accelerator.process_index}.pth"
            if model_index == 0
            else f"{MODEL_NAME}_{model_index}_rank{accelerator.process_index}.pth"
        )
        output_model_file = os.path.join(output_dir, weights_name)
        xm.save(state_dict, output_model_file, master_only=False)
        xm.rendezvous("saved sharded model checkpoint")

        if self.state_dict_type == StateDictType.FULL_STATE_DICT and accelerator.process_index == 0:
            weights_name = f"{MODEL_NAME}.bin" if model_index == 0 else f"{MODEL_NAME}_{model_index}.bin"
            output_model_file = os.path.join(output_dir, weights_name)
            if accelerator.process_index == 0:
                full_state_dict, _ = consolidate_sharded_model_checkpoints(
                    f"{output_dir}/{MODEL_NAME}_rank",
                    save_model=False,
                )
                torch.save(full_state_dict, output_model_file)
                print(f"Model saved to {output_model_file}")

    def load_model(self, accelerator, model, input_dir, model_index=0):
        from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

        accelerator.wait_for_everyone()
        if self.state_dict_type == StateDictType.FULL_STATE_DICT:
            if type(model) is FSDP:
                raise ValueError("Only sharded model weights can be loaded with XLA FSDP.")
            if accelerator.process_index == 0:
                weights_name = f"{MODEL_NAME}.bin" if model_index == 0 else f"{MODEL_NAME}_{model_index}.bin"
                input_model_file = os.path.join(input_dir, weights_name)
                accelerator.print(f"Loading model from {input_model_file}")
                state_dict = torch.load(input_model_file)
                accelerator.print(f"Model loaded from {input_model_file}")
                model.load_state_dict(state_dict, False)
        else:
            weights_name = (
                f"{MODEL_NAME}_rank{accelerator.process_index}.pth"
                if model_index == 0
                else f"{MODEL_NAME}_{model_index}_rank{accelerator.process_index}.pth"
            )
            input_model_file = os.path.join(input_dir, weights_name)
            state_dict = torch.load(input_model_file)
            model.load_state_dict(state_dict["model"], False)

    def save_optimizer(self, accelerator, optimizer, model, output_dir, optimizer_index=0, optim_input=None):
        # from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
        # optim_state = FSDP.full_optim_state_dict(model, optimizer, optim_input=optim_input)
        optim_state = {"optimizer": optimizer.state_dict(), "shard_metadata": model.get_shard_metadata()}
        optimizer_path = os.path.join(output_dir, f"{OPTIMIZER_NAME}_rank{accelerator.process_index}.bin")
        xm.save(optim_state, optimizer_path, master_only=False)
        xm.rendezvous("saved sharded optimizer checkpoint")

        # TODO: save the full optimizer state if possible.
        # if accelerator.process_index == 0:
        #     optim_state_name = (
        #         f"{OPTIMIZER_NAME}.bin" if optimizer_index == 0 else f"{OPTIMIZER_NAME}_{optimizer_index}.bin"
        #     )
        #     output_optimizer_file = os.path.join(output_dir, optim_state_name)
        #     print(f"Saving Optimizer state to {output_optimizer_file}")
        #     torch.save(optim_state, output_optimizer_file)
        #     print(f"Optimizer state saved in {output_optimizer_file}")

    def load_optimizer(self, accelerator, optimizer, model, input_dir, optimizer_index=0):
        accelerator.wait_for_everyone()
        # TODO: load full osd support.
        # full_osd = None
        # if accelerator.process_index == 0:
        #     optimizer_name = (
        #         f"{OPTIMIZER_NAME}.bin" if optimizer_index == 0 else f"{OPTIMIZER_NAME}_{optimizer_index}.bin"
        #     )
        #     input_optimizer_file = os.path.join(input_dir, optimizer_name)
        #     print(f"Loading Optimizer state from {input_optimizer_file}")
        #     full_osd = torch.load(input_optimizer_file)
        #     print(f"Optimizer state loaded from {input_optimizer_file}")
        # # called from all ranks, though only rank0 has a valid param for full_osd
        # sharded_osd = FSDP.scatter_full_optim_state_dict(full_osd, model)
        # optimizer.load_state_dict(sharded_osd)
        optimizer_path = os.path.join(input_dir, f"{OPTIMIZER_NAME}_rank{accelerator.process_index}.bin")
        optim_state = torch.load(optimizer_path)
        xm.send_cpu_data_to_device(optim_state, accelerator.device)
        optimizer.load_state_dict(optim_state["optimizer"])


@dataclass
class ModelParallelismPlugin:
    tensor_parallel_size: int = 1
    parallelize_embeddings: bool = True
    sequence_parallel_enabled: bool = False
    pipeline_parallel_size: int = 1
    pipeline_parallel_num_microbatches: int = 1
    pipeline_parallel_use_zero1_optimizer: bool = False
    checkpoint_dir: Optional[Union[str, Path]] = None

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
            pipeline_parallel_num_microbatches=self.pipeline_parallel_num_microbatches,
            pipeline_parallel_use_zero1_optimizer=self.pipeline_parallel_use_zero1_optimizer,
            checkpoint_dir=self.checkpoint_dir,
        )
        return parallelized_model
