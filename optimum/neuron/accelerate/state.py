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

import os

import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from accelerate.state import AcceleratorState, PartialState, ThreadLocalSharedDict
from accelerate.utils import (
    DistributedType,
    parse_choice_from_env,
    parse_flag_from_env,
)
from accelerate.utils.dataclasses import SageMakerDistributedType
from neuronx_distributed.parallel_layers import parallel_state

from ...utils import logging
from ..models.neuron_config import TrainingNeuronConfig
from ..utils.torch_xla_and_neuronx_initialization import (
    init_process_group,
    set_common_flags,
    set_neuron_cc_flags_for_torch_amp,
)
from .utils import NeuronDistributedType


logger = logging.get_logger()

SharedDict = ThreadLocalSharedDict


class NeuronPartialState(PartialState):
    def __init__(self, cpu: bool = False, **kwargs):
        self.__dict__ = self._shared_state
        if not self.initialized:
            self._cpu = cpu
            self.backend = None
            env_device = os.environ.get("ACCELERATE_TORCH_DEVICE", None)
            self.device = torch.device(env_device) if env_device is not None else None
            self.debug = parse_flag_from_env("ACCELERATE_DEBUG_MODE")
            use_sagemaker_dp = kwargs.pop("_use_sagemaker_dp", None)
            if use_sagemaker_dp is not None:
                raise NotImplementedError("SageMaker distributed training is not supported by `optimum-neuron`.")
            os.environ["ACCELERATE_USE_SAGEMAKER"] = "false"
            os.environ["ACCELERATE_SAGEMAKER_DISTRIBUTED_TYPE"] = SageMakerDistributedType.NO.value
            backend, distributed_type = self._prepare_backend(cpu, use_sagemaker_dp, kwargs.pop("backend", None))
            self.backend = backend
            self.distributed_type = distributed_type
            # No backend == no distributed training
            if self.backend is None:
                self.distributed_type = DistributedType.NO
                self.num_processes = 1
                self.process_index = 0
                self.local_process_index = 0
            elif self.backend != "xla":
                raise ValueError("Only the XLA backend is supported by `optimum-neuron`.")
            else:
                # It is important to set the environment variables before initializing the process group otherwise they will be ignored by the Neuron compiler.
                set_common_flags()
                if os.environ.get("ACCELERATE_USE_AMP", "false") == "true":
                    set_neuron_cc_flags_for_torch_amp()
                if not torch.distributed.is_initialized():
                    init_process_group()
                self.num_processes = xr.world_size()
                self.process_index = xr.global_ordinal()
                self.local_process_index = xm.get_local_ordinal()
                self.device = xm.xla_device()

        # Important: This should be the *only* code outside of `self.initialized!`
        self.fork_launched = parse_flag_from_env("FORK_LAUNCHED", 0)

    def wait_for_everyone(self):
        xm.rendezvous("accelerate.utils.wait_for_everyone")


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
        mixed_precision: str | None = None,
        cpu: bool = False,
        dynamo_plugin=None,
        deepspeed_plugin=None,
        fsdp_plugin=None,
        torch_tp_plugin=None,
        megatron_lm_plugin=None,
        trn_config: TrainingNeuronConfig | None = None,
        _from_accelerator: bool = False,
        **kwargs,
    ):
        self.__dict__ = self._shared_state
        if parse_flag_from_env("ACCELERATE_USE_CPU"):
            cpu = True

        if NeuronPartialState._shared_state == {}:
            NeuronPartialState(cpu, **kwargs)
        self.__dict__.update(NeuronPartialState._shared_state)
        self._check_initialized(mixed_precision, cpu)
        if not self.initialized:
            self.deepspeed_plugin = None
            self.ipex_plugin = None
            mixed_precision = (
                parse_choice_from_env("ACCELERATE_MIXED_PRECISION", "no")
                if mixed_precision is None
                else mixed_precision.lower()
            )
            if mixed_precision == "fp8":
                raise NotImplementedError(
                    "FP8 mixed precision is not supported by `optimum-neuron`. Please use `bf16`, or `no` instead."
                )

            self.dynamo_plugin = dynamo_plugin
            if not _from_accelerator:
                raise ValueError(
                    "Please make sure to properly initialize your accelerator via `accelerator = Accelerator()` "
                    "before using any functionality from the `accelerate` library."
                )

            self._mixed_precision = mixed_precision
            if self.distributed_type == DistributedType.XLA:
                if mixed_precision == "bf16":
                    os.environ["NEURON_RT_STOCHASTIC_ROUNDING_EN"] = "1"

                if trn_config is None:
                    trn_config = TrainingNeuronConfig()

                if trn_config.should_parallelize:
                    self.distributed_type = NeuronDistributedType.MODEL_PARALLELISM

                self.trn_config = trn_config

                if torch.distributed.is_initialized() and not parallel_state.model_parallel_is_initialized():
                    parallel_state.initialize_model_parallel(
                        tensor_model_parallel_size=self.trn_config.tensor_parallel_size,
                        pipeline_model_parallel_size=self.trn_config.pipeline_parallel_size,
                    )

            elif self.distributed_type is DistributedType.NO:
                os.environ["ACCELERATE_USE_IPEX"] = "false"
                self.use_ipex = False

            PartialState._shared_state["distributed_type"] = self.distributed_type

    @property
    def deepspeed_plugin(self):
        # Returns `None` to maintain original behavior.
        return None

    @deepspeed_plugin.setter
    def deepspeed_plugin(self, value):
        if value is not None:
            raise ValueError(
                "Deepspeed plugin is not supported by `optimum-neuron`. Please use `deepspeed_plugin=None` instead."
            )
        self._deepspeed_plugin = value
