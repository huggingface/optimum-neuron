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

import torch
from vllm.config import VllmConfig
from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment
from vllm.model_executor import set_random_seed
from vllm.tasks import SupportedTask
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.worker.worker_base import WorkerBase

from .runner import OptimumNeuronModelRunner


logger = logging.getLogger("Neuron")


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

    # WorkerBase methods that are expected to be implemented
    # Note that some of the methods related to features we explicitly don't support
    # (e.g., prefix caching, lora) are omitted here because they won't be called

    def init_device(self) -> None:
        # Initialize distributed environment.
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

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        # The optimum-neuron vLLM plugin only supports text generation.
        return ("generate",)

    @torch.inference_mode()
    def execute_model(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput | None:
        # Main execution method called repeatedly by the vLLM scheduler.
        return self.model_runner.execute_model(scheduler_output)
