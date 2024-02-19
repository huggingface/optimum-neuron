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
"""Custom PartialState and AcceleratorState for Neuron."""

import os

import torch
from accelerate.state import AcceleratorState, PartialState, ThreadLocalSharedDict
from accelerate.utils import (
    DistributedType,
    DynamoBackend,
    get_ccl_version,
    get_int_from_env,
    is_ccl_available,
    is_deepspeed_available,
    is_fp8_available,
    is_ipex_available,
    is_xpu_available,
    parse_choice_from_env,
    parse_flag_from_env,
)
from accelerate.utils.dataclasses import FullyShardedDataParallelPlugin, SageMakerDistributedType

from ...utils import logging
from ..utils import is_neuronx_distributed_available, is_torch_xla_available
from .utils import NeuronDistributedType, NeuronFullyShardedDataParallelPlugin
from .utils.dataclasses import ModelParallelismPlugin


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

if is_neuronx_distributed_available():
    from neuronx_distributed.parallel_layers import parallel_state


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
            if use_sagemaker_dp is None:
                use_sagemaker_dp = (
                    os.environ.get("ACCELERATE_USE_SAGEMAKER", "false") == "true"
                    and os.environ.get("ACCELERATE_SAGEMAKER_DISTRIBUTED_TYPE") != SageMakerDistributedType.NO
                )

            if use_sagemaker_dp and not cpu:
                if (
                    os.environ.get("ACCELERATE_SAGEMAKER_DISTRIBUTED_TYPE") == SageMakerDistributedType.DATA_PARALLEL
                ) or use_sagemaker_dp:
                    self.distributed_type = DistributedType.MULTI_GPU
                    import smdistributed.dataparallel.torch.torch_smddp  # noqa

                    if not torch.distributed.is_initialized():
                        torch.distributed.init_process_group(backend="smddp")
                    self.backend = "smddp"
                    self.num_processes = torch.distributed.get_world_size()
                    self.process_index = torch.distributed.get_rank()
                    self.local_process_index = int(os.environ.get("LOCAL_RANK", -1))
                    if self.device is None:
                        self.device = torch.device("cuda", self.local_process_index)
                    torch.cuda.set_device(self.device)
            elif is_torch_xla_available() and not cpu:
                self.distributed_type = DistributedType.TPU
                self.num_processes = xm.xrt_world_size()
                self.process_index = xm.get_ordinal()
                self.local_process_index = xm.get_local_ordinal()
                self.device = xm.xla_device()
            elif (
                os.environ.get("ACCELERATE_USE_DEEPSPEED", "false") == "true"
                and int(os.environ.get("LOCAL_RANK", -1)) != -1
                and not cpu
            ):
                assert (
                    is_deepspeed_available()
                ), "DeepSpeed is not available => install it using `pip3 install deepspeed` or build it from source"
                self.distributed_type = DistributedType.DEEPSPEED
                if not torch.distributed.is_initialized():
                    from deepspeed import comm as dist

                    # DeepSpeed always uses nccl
                    kwargs.pop("backend", None)
                    self.backend = "nccl"
                    dist.init_distributed(dist_backend=self.backend, auto_mpi_discovery=False, **kwargs)

                self.num_processes = torch.distributed.get_world_size()
                self.process_index = torch.distributed.get_rank()
                self.local_process_index = int(os.environ.get("LOCAL_RANK", -1))
                if self.device is None:
                    if is_xpu_available():
                        self.device = torch.device("xpu", self.local_process_index)
                        if self.device is not None:
                            torch.xpu.set_device(self.device)
                    else:
                        self.device = torch.device("cuda", self.local_process_index)
                        if self.device is not None:
                            torch.cuda.set_device(self.device)
                self._mixed_precision = "no"  # deepspeed handles mixed_precision using deepspeed_config
            elif int(os.environ.get("LOCAL_RANK", -1)) != -1 and not cpu:
                self.distributed_type = DistributedType.MULTI_GPU
                if not torch.distributed.is_initialized():
                    self.backend = kwargs.pop("backend", "nccl")
                    # Special case for `TrainingArguments`, where `backend` will be `None`
                    if self.backend is None:
                        self.backend = "nccl"
                    torch.distributed.init_process_group(backend=self.backend, **kwargs)
                self.num_processes = torch.distributed.get_world_size()
                self.process_index = torch.distributed.get_rank()
                self.local_process_index = int(os.environ.get("LOCAL_RANK", -1))
                if self.device is None:
                    self.device = torch.device("cuda", self.local_process_index)
                torch.cuda.set_device(self.device)
            elif get_int_from_env(["PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE", "WORLD_SIZE"], 1) > 1:
                if is_xpu_available():
                    self.distributed_type = DistributedType.MULTI_XPU
                else:
                    self.distributed_type = DistributedType.MULTI_CPU
                if is_ccl_available() and get_int_from_env(["CCL_WORKER_COUNT"], 0) > 0:
                    if get_ccl_version() >= "1.12":
                        import oneccl_bindings_for_pytorch  # noqa: F401
                    else:
                        import torch_ccl  # noqa: F401
                    backend = "ccl"
                elif torch.distributed.is_mpi_available():
                    backend = "mpi"
                else:
                    backend = "gloo"
                # Try to get launch configuration from environment variables set by MPI launcher - works for Intel MPI, OpenMPI and MVAPICH
                rank = get_int_from_env(["RANK", "PMI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK"], 0)
                size = get_int_from_env(["WORLD_SIZE", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE"], 1)
                local_rank = get_int_from_env(
                    ["LOCAL_RANK", "MPI_LOCALRANKID", "OMPI_COMM_WORLD_LOCAL_RANK", "MV2_COMM_WORLD_LOCAL_RANK"], 0
                )
                local_size = get_int_from_env(
                    ["MPI_LOCALNRANKS", "OMPI_COMM_WORLD_LOCAL_SIZE", "MV2_COMM_WORLD_LOCAL_SIZE"], 1
                )
                self.local_process_index = local_rank
                os.environ["RANK"] = str(rank)
                os.environ["WORLD_SIZE"] = str(size)
                os.environ["LOCAL_RANK"] = str(local_rank)
                if not os.environ.get("MASTER_PORT", None):
                    os.environ["MASTER_PORT"] = "29500"
                if not os.environ.get("MASTER_ADDR", None):
                    if local_size != size and backend != "mpi":
                        raise ValueError(
                            "Looks like distributed multinode run but MASTER_ADDR env not set, "
                            "please try exporting rank 0's hostname as MASTER_ADDR"
                        )
                if not torch.distributed.is_initialized():
                    # Backend is not set by the user, we set it here
                    kwargs.pop("backend", None)
                    self.backend = backend
                    torch.distributed.init_process_group(self.backend, rank=rank, world_size=size, **kwargs)
                self.num_processes = torch.distributed.get_world_size()
                self.process_index = torch.distributed.get_rank()
                self.local_process_index = local_rank
                if self.device is None:
                    self.device = torch.device("cpu")
            else:
                self.distributed_type = DistributedType.NO
                self.num_processes = 1
                self.process_index = self.local_process_index = 0

                if self.device is None:
                    self.device = torch.device("cpu") if cpu else self.default_device

        self.fork_launched = parse_flag_from_env("FORK_LAUNCHED", 0)

    def wait_for_everyone(self):
        if self.distributed_type in [NeuronDistributedType.XLA_FSDP, NeuronDistributedType.MODEL_PARALLELISM]:
            xm.rendezvous("accelerate.utils.wait_for_everyone")
        else:
            super().wait_for_everyone()


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
        mp_plugin=None,
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
                if (
                    os.environ.get("ACCELERATE_USE_NEURONX_DISTRIBUTED_TP", "false") == "true"
                    or os.environ.get("ACCELERATE_USE_NEURONX_DISTRIBUTED_PP", "false") == "true"
                ):
                    if not is_neuronx_distributed_available():
                        raise RuntimeError(
                            "Model parallelism requires the neuronx_distributed package. You can install it by "
                            "running: python -m pip install neuronx_distributed --extra-index-url "
                            "https://pip.repos.neuron.amazonaws.com"
                        )
                    if mp_plugin is None:
                        raise ValueError(
                            "Could not initialize `neuronx_distributed` model parallelism because no "
                            "`ModelParallelismPlugin` was provided."
                        )
                    if mp_plugin.should_parallelize:
                        if not parallel_state.model_parallel_is_initialized():
                            parallel_state.initialize_model_parallel(
                                tensor_model_parallel_size=mp_plugin.tensor_parallel_size,
                                pipeline_model_parallel_size=mp_plugin.pipeline_parallel_size,
                            )
                        self.distributed_type = NeuronDistributedType.MODEL_PARALLELISM
                    else:
                        logger.warning(
                            "Model parallelism is requested but nothing is done because the tensor parallel size and "
                            "the pipeline parallel size are set to 1."
                        )
                    self.mp_plugin = mp_plugin
                else:
                    self.mp_plugin = ModelParallelismPlugin()
                if os.environ.get("ACCELERATE_USE_FSDP", "false") == "true":
                    self.distributed_type = NeuronDistributedType.XLA_FSDP
                    if self._mixed_precision != "no":
                        # TODO: do we need that?
                        fsdp_plugin.set_mixed_precision(self._mixed_precision)
                    if isinstance(fsdp_plugin, FullyShardedDataParallelPlugin) and not isinstance(
                        fsdp_plugin, NeuronFullyShardedDataParallelPlugin
                    ):
                        fsdp_plugin.__class__ = NeuronFullyShardedDataParallelPlugin
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
                if is_ipex_available():
                    "check if user disables it explicitly"
                    self.use_ipex = parse_flag_from_env("ACCELERATE_USE_IPEX", default=True)
                else:
                    self.use_ipex = False
            if (
                self.dynamo_plugin.backend != DynamoBackend.NO
                and self._mixed_precision == "no"
                and self.device.type == "cuda"
            ):
                torch.backends.cuda.matmul.allow_tf32 = True
            PartialState._shared_state["distributed_type"] = self.distributed_type
