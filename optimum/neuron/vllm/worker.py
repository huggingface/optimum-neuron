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
from collections.abc import Callable
from typing import TypeVar

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
from vllm.lora.request import LoRARequest
from vllm.model_executor import set_random_seed
from vllm.tasks import SupportedTask
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.worker.worker_base import WorkerBase

from .runner import OptimumNeuronModelRunner


logger = logging.getLogger("Neuron")

_R = TypeVar("_R")


class OptimumNeuronWorker(WorkerBase):
    """A worker class that executes the model on a group of neuron cores."""

    model_runner: OptimumNeuronModelRunner

    def __init__(
        self,
        vllm_config: VllmConfig,
        local_rank: int,
        rank: int,
        distributed_init_method: str,
        is_driver_worker: bool = False,
    ) -> None:
        super().__init__(
            vllm_config=vllm_config,
            local_rank=local_rank,
            rank=rank,
            distributed_init_method=distributed_init_method,
            is_driver_worker=is_driver_worker,
        )

        assert self.lora_config is None, "LoRA is not supported for optimum-neuron framework."
        assert self.speculative_config is None, "Speculative decoding is not supported for optimum-neuron framework."

        self.model_runner = OptimumNeuronModelRunner.create(vllm_config=vllm_config)

    # WorkerBase methods that are expected to be implemented
    # Note that some of the methods related to features we explicitly don't support
    # (e.g., prefix caching, lora) are omitted here because they won't be called

    def init_device(self) -> None:
        # Initialize distributed environment.
        dp_size = self.vllm_config.parallel_config.data_parallel_size
        if dp_size > 1:
            # With data parallelism, vLLM's init_distributed_environment adjusts
            # world_size to tp * pp * dp, expecting that many worker processes.
            # Since Neuron handles TP internally (uni executor, 1 worker per DP rank),
            # we pre-initialize torch.distributed with the correct topology:
            # dp_size participants, one per DP rank.
            parallel_config = self.vllm_config.parallel_config
            dp_rank = parallel_config.data_parallel_rank
            ip = parallel_config.data_parallel_master_ip
            port = parallel_config.get_next_dp_init_port()
            init_method = f"tcp://{ip}:{port}"
            torch.distributed.init_process_group(
                backend="gloo",
                init_method=init_method,
                world_size=dp_size,
                rank=dp_rank,
            )
        init_distributed_environment(
            world_size=1,
            rank=self.rank,
            local_rank=self.local_rank,
            distributed_init_method=self.distributed_init_method,
            backend="gloo",
        )

        ensure_model_parallel_initialized(1, 1)

        # Set random seed.
        set_random_seed(self.model_config.seed)

    def load_model(self):
        self.model_runner.load_model()

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        # Return empty dict since we disabled prefix caching.
        return {}

    def initialize_cache(self, num_gpu_blocks: int, num_cpu_blocks: int) -> None:
        # Nothing to do here as the KV cache is instantiated and managed internally
        # by the optimum-neuron model.
        assert num_cpu_blocks == 0
        assert num_gpu_blocks == 1

    def initialize_from_config(self, kv_cache_config: KVCacheConfig) -> None:
        # We don't need to do anything since we disabled prefix caching.
        pass

    def compile_or_warm_up_model(self) -> None:
        # Not required since the compilation happens implicitly when loading the model.
        pass

    def execute_dummy_batch(self) -> None:
        # No-op for Neuron. In DP mode, vLLM calls this on idle replicas to keep
        # them synchronized. Neuron models don't need dummy execution for sync.
        pass

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        return self.model_runner.get_supported_tasks()

    @torch.inference_mode()
    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput | None:
        # Main execution method called repeatedly by the vLLM scheduler.
        return self.model_runner.execute_model(scheduler_output)

    def sample_tokens(self, grammar_output) -> ModelRunnerOutput:
        raise NotImplementedError("Optimum Neuron worker does not support deferred token sampling.")

    def get_cache_block_size_bytes(self) -> int:
        # Prefix caching is disabled, so there is no meaningful cache block size.
        return 0

    def get_model(self) -> nn.Module:
        if self.model_runner.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() before get_model().")
        return self.model_runner.model

    def apply_model(self, fn: Callable[[nn.Module], _R]) -> _R:
        return fn(self.get_model())

    def add_lora(self, lora_request: LoRARequest) -> bool:
        raise NotImplementedError("LoRA is not supported for optimum-neuron framework.")

    def remove_lora(self, lora_id: int) -> bool:
        raise NotImplementedError("LoRA is not supported for optimum-neuron framework.")

    def pin_lora(self, lora_id: int) -> bool:
        raise NotImplementedError("LoRA is not supported for optimum-neuron framework.")

    def list_loras(self) -> set[int]:
        return set()
