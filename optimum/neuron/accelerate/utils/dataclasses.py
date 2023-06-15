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
""" """

import os
from dataclasses import dataclass
import enum

import torch

from accelerate.utils.dataclasses import FullyShardedDataParallelPlugin
from accelerate.utils.constants import MODEL_NAME, OPTIMIZER_NAME


from ...utils import is_torch_xla_available

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    from torch_xla.distributed.fsdp.state_dict_utils import consolidate_sharded_model_checkpoints


class NeuronDistributedType(str, enum.Enum):
    """
    Represents a type of distributed environment.

    Values:

        - **NO** -- Not a distributed environment, just a single process.
        - **MULTI_CPU** -- Distributed on multiple CPU nodes.
        - **MULTI_GPU** -- Distributed on multiple GPUs.
        - **MULTI_XPU** -- Distributed on multiple XPUs.
        - **DEEPSPEED** -- Using DeepSpeed.
        - **TPU** -- Distributed on TPUs.
    """

    # Subclassing str as well as Enum allows the `DistributedType` to be JSON-serializable out of the box.
    NO = "NO"
    MULTI_CPU = "MULTI_CPU"
    MULTI_GPU = "MULTI_GPU"
    MULTI_XPU = "MULTI_XPU"
    DEEPSPEED = "DEEPSPEED"
    FSDP = "FSDP"
    TPU = "TPU"
    MPS = "MPS"  # here for backward compatibility. Remove in v0.18.0
    MEGATRON_LM = "MEGATRON_LM"
    XLA_FSDP = "XLA_FSDP"


@dataclass
class NeuronFullyShardedDataParallelPlugin(FullyShardedDataParallelPlugin):
    # TODO: redefine the post init to do checks on which option is supported.

    def save_model(self, accelerator, model, output_dir, model_index=0):
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

        state_dict = {"model": model.state_dict(), "shard_metadata": model.get_shard_metadata()}
        weights_name = (
            f"{MODEL_NAME}_rank{accelerator.process_index}.bin"
            if model_index == 0
            else f"{MODEL_NAME}_{model_index}_rank{accelerator.process_index}.bin"
        )
        output_model_file = os.path.join(output_dir, weights_name)
        print(f"Saving model to {output_model_file}")
        xm.save(state_dict, output_model_file)
        xm.rendezvous("saved sharded model checkpoint")
        print(f"Model saved to {output_model_file}")

        if self.state_dict_type == StateDictType.FULL_STATE_DICT:
            weights_name = f"{MODEL_NAME}.bin" if model_index == 0 else f"{MODEL_NAME}_{model_index}.bin"
            output_model_file = os.path.join(output_dir, weights_name)
            if accelerator.process_index == 0:
                full_state_dict, _ = consolidate_sharded_model_checkpoints(
                    f"{output_dir}/{MODEL_NAME}_rank",
                    save_model=False,
                )
                print(f"Saving model to {output_model_file}")
                # TODO: test this.
                torch.save(full_state_dict["model"], output_model_file)
                print(f"Model saved to {output_model_file}")

    def load_model(self, accelerator, model, input_dir, model_index=0):
        from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
        accelerator.wait_for_everyone()
        if self.state_dict_type == StateDictType.FULL_STATE_DICT:
            raise ValueError("Only sharded model weights can be loaded with XLA FSDP.")
            # weights_name = f"{MODEL_NAME}.bin" if model_index == 0 else f"{MODEL_NAME}_{model_index}.bin"
            # input_model_file = os.path.join(input_dir, weights_name)
            # accelerator.print(f"Loading model from {input_model_file}")
            # state_dict = torch.load(input_model_file)
            # accelerator.print(f"Model loaded from {input_model_file}")
        else:
            weights_name = (
                f"{MODEL_NAME}_rank{accelerator.process_index}.bin"
                if model_index == 0
                else f"{MODEL_NAME}_{model_index}_rank{accelerator.process_index}.bin"
            )
            input_model_file = os.path.join(input_dir, weights_name)
            print(f"Loading model from {input_model_file}")
            state_dict = torch.load(input_model_file)
            print(f"Model loaded from {input_model_file}")
            model.load_state_dict(state_dict["model"])

    def save_optimizer(self, accelerator, optimizer, model, output_dir, optimizer_index=0, optim_input=None):
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP

        optim_state = FSDP.full_optim_state_dict(model, optimizer, optim_input=optim_input)
        optim_state = {"optimizer": optimizer.state_dict(), "shard_metadata": model.get_shard_metadata()}
        xm.save(optim_state, f"{OPTIMIZER_NAME}_rank{accelerator.process_index}.bin")
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
        optim_state = torch.load(f"{OPTIMIZER_NAME}_rank{accelerator.process_index}.bin")
        optimizer.load_state_dict(optim_state["optimizer"])
