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

import torch

from accelerate.state import AcceleratorState, PartialState, SharedDict
from accelerate.utils import (
    DynamoBackend,
    DistributedType,
    GradientAccumulationPlugin,
    get_ccl_version,
    get_int_from_env,
    is_ccl_available,
    is_deepspeed_available,
    is_fp8_available,
    is_mps_available,
    is_tpu_available,
    is_xpu_available,
    parse_choice_from_env,
    parse_flag_from_env,
)
from accelerate.utils.dataclasses import SageMakerDistributedType

from ..utils import is_torch_xla_available

from .utils import NeuronDistributedType


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

class NeuronPartialState(PartialState):
    pass

class NeuronAcceleratorState(AcceleratorState):
    """
    Singleton class that has information about the current training environment.

    **Available attributes:**

        - **device** (`torch.device`) -- The device to use.
        - **distributed_type** ([`~accelerate.state.DistributedType`]) -- The type of distributed environment currently
          in use.
        - **initialized** (`bool`) -- Whether or not the `AcceleratorState` has been initialized from `Accelerator`.
        - **local_process_index** (`int`) -- The index of the current process on the current server.
        - **mixed_precision** (`str`) -- Whether or not the current script will use mixed precision, and if so the type
          of mixed precision being performed.
        - **num_processes** (`int`) -- The number of processes currently launched in parallel.
        - **process_index** (`int`) -- The index of the current process.
        - **is_last_process** (`bool`) -- Whether or not the current process is the last one.
        - **is_main_process** (`bool`) -- Whether or not the current process is the main one.
        - **is_local_main_process** (`bool`) -- Whether or not the current process is the main one on the local node.
    """

    def __init__(
        self,
        mixed_precision: str = None,
        cpu: bool = False,
        dynamo_plugin=None,
        deepspeed_plugin=None,
        fsdp_plugin=None,
        megatron_lm_plugin=None,
        ipex_plugin=None,
        _from_accelerator: bool = False,
        **kwargs,
    ):
        self.__dict__ = self._shared_state
        if parse_flag_from_env("ACCELERATE_USE_CPU"):
            cpu = True
        if PartialState._shared_state == {}:
            PartialState(cpu, **kwargs)
        self.__dict__.update(PartialState._shared_state)
        self._check_initialized(mixed_precision, cpu)
        import pdb; pdb.set_trace()
        if not self.initialized:
            self.deepspeed_plugin = None
            self.ipex_plugin = None
            mixed_precision = (
                parse_choice_from_env("ACCELERATE_MIXED_PRECISION", "no")
                if mixed_precision is None
                else mixed_precision.lower()
            )
            if mixed_precision == "fp8" and not is_fp8_available():
                raise ValueError("Using `fp8` precision requires `transformer_engine` to be installed.")
            self.dynamo_plugin = dynamo_plugin
            if not _from_accelerator:
                raise ValueError(
                    "Please make sure to properly initialize your accelerator via `accelerator = Accelerator()` "
                    "before using any functionality from the `accelerate` library."
                )
            # deepspeed handles mixed_precision using deepspeed_config
            self._mixed_precision = "no" if self.distributed_type == DistributedType.DEEPSPEED else mixed_precision
            if self.distributed_type == DistributedType.TPU:
                if mixed_precision == "bf16":
                    if os.environ.get("ACCELERATE_DOWNCAST_BF16"):
                        os.environ["XLA_USE_BF16"] = str(0)
                        os.environ["XLA_DOWNCAST_BF16"] = str(1)
                        self.downcast_bfloat = True
                    else:
                        os.environ["XLA_USE_BF16"] = str(1)
                        os.environ["XLA_DOWNCAST_BF16"] = str(0)
                        self.downcast_bfloat = False
                print("LOL")
                if os.environ.get("ACCELERATE_USE_FSDP", "false") == "true":
                    self.distributed_type = NeuronDistributedType.XLA_FSDP
                    if self._mixed_precision != "no":
                        fsdp_plugin.set_mixed_precision(self._mixed_precision)
                    self.fsdp_plugin = fsdp_plugin
            elif os.environ.get("ACCELERATE_USE_DEEPSPEED", "false") == "true" and not cpu:
                self.deepspeed_plugin = deepspeed_plugin
            elif self.distributed_type == DistributedType.MULTI_GPU:
                if os.environ.get("ACCELERATE_USE_FSDP", "false") == "true":
                    self.distributed_type = DistributedType.FSDP
                    if self._mixed_precision != "no":
                        fsdp_plugin.set_mixed_precision(self._mixed_precision)
                    self.fsdp_plugin = fsdp_plugin
                if os.environ.get("ACCELERATE_USE_MEGATRON_LM", "false") == "true":
                    self.distributed_type = DistributedType.MEGATRON_LM
                    megatron_lm_plugin.set_mixed_precision(self._mixed_precision)
                    self.megatron_lm_plugin = megatron_lm_plugin
            elif self.distributed_type in [DistributedType.MULTI_CPU, DistributedType.MULTI_XPU, DistributedType.NO]:
                if (
                    self.device.type == "cpu"
                    or (self.device.type == "xpu" and is_xpu_available())
                    and ipex_plugin is not None
                ):
                    self.ipex_plugin = ipex_plugin
                    if self.ipex_plugin is not None:
                        self.ipex_plugin.set_mixed_precision(mixed_precision)
            if (
                self.dynamo_plugin.backend != DynamoBackend.NO
                and self._mixed_precision == "no"
                and self.device.type == "cuda"
            ):
                torch.backends.cuda.matmul.allow_tf32 = True
            PartialState._shared_state["distributed_type"] = self.distributed_type
