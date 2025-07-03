# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""An optimum-neuron vLLM worker class."""

import logging
from typing import List, Optional, Set, Tuple

import torch
from vllm.config import VllmConfig
from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.sequence import ExecuteModelRequest
from vllm.worker.neuron_model_runner import NeuronModelRunner
from vllm.worker.worker_base import LocalOrDistributedWorkerBase, WorkerBase, WorkerInput

from .runner import OptimumNeuronModelRunner


logger = logging.getLogger("Neuron")


class OptimumNeuronWorker(LocalOrDistributedWorkerBase):
    """A worker class that executes the model on a group of neuron cores."""

    model_runner: NeuronModelRunner

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        WorkerBase.__init__(self, vllm_config=vllm_config)
        self.local_rank = local_rank
        self.rank = rank
        self.distributed_init_method = distributed_init_method
        self.is_driver_worker = is_driver_worker

        assert self.lora_config is None, "LoRA is not supported for optimum-neuron framework."
        assert self.speculative_config is None, "Speculative decoding is not supported for optimum-neuron framework."

        if self.model_config.trust_remote_code:
            # note: lazy import to avoid importing torch before initializing
            from vllm.utils import init_cached_hf_modules

            init_cached_hf_modules()

        self.model_runner = OptimumNeuronModelRunner(vllm_config=vllm_config)

    def init_device(self) -> None:
        self.init_distributed_environment()

        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    def determine_num_available_blocks(self) -> Tuple[int, int]:
        """Determine the number of available KV blocks.

        Swapping is not yet supported, so always return num_cpu_blocks=0.

        We configure num_gpu_blocks to be equal to max_num_seqs.
        """
        # Set the number of GPU blocks to be the same as the maximum number of
        # sequences that can be processed in a single batch. This is equivalent
        # to schedule without PagedAttention.
        num_gpu_blocks = self.scheduler_config.max_num_seqs + 1

        # Swap not yet supported with Neuron backend.
        num_cpu_blocks = 0

        return num_gpu_blocks, num_cpu_blocks

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        """Initialize the KV cache."""

        # Different values are not tested.
        assert num_cpu_blocks == 0
        assert num_gpu_blocks == self.scheduler_config.max_num_seqs + 1

        self.cache_config.num_gpu_blocks = num_gpu_blocks
        self.cache_config.num_cpu_blocks = num_cpu_blocks

    @property
    def do_metadata_broadcast(self) -> bool:
        return False

    @property
    def kv_cache(self) -> Optional[List[List[torch.Tensor]]]:
        return None

    @torch.inference_mode()
    def prepare_worker_input(self, execute_model_req: ExecuteModelRequest) -> WorkerInput:
        return WorkerInput(
            num_seq_groups=len(execute_model_req.seq_group_metadata_list),
        )

    def execute_worker(self, worker_input: WorkerInput) -> None:
        pass

    def get_cache_block_size_bytes(self) -> int:
        """Determine the size in bytes of a cache block.

        This is required for speculative decoding; it is not yet implemented.
        """
        raise NotImplementedError

    def init_distributed_environment(self):
        """Neuron uses transformers-neuronx for tensor parallelism.

        vLLM still needs the environment initialized when TP/PP > 1
        """
        init_distributed_environment(
            world_size=1,
            rank=self.rank,
            local_rank=self.local_rank,
            distributed_init_method=self.distributed_init_method,
            backend="gloo",
        )

        ensure_model_parallel_initialized(
            1,
            1,
        )

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError(f"{type(self)} does not support LoRA with Optimum Neuron platform")

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError(f"{type(self)} does not support LoRA with Optimum Neuron platform")

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError(f"{type(self)} does not support LoRA with Optimum Neuron platform")

    def list_loras(self) -> Set[int]:
        raise NotImplementedError(f"{type(self)} does not support LoRA with Optimum Neuron platform")
