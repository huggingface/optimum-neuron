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
import logging

from vllm.platforms.interface import Platform, PlatformEnum


logger = logging.getLogger("Neuron")


class OptimumNeuronPlatform(Platform):
    _enum = PlatformEnum.NEURON
    device_name: str = "neuron"
    device_type: str = "neuron"
    ray_device_key: str = "neuron_cores"
    device_control_env_var: str = "NEURON_RT_VISIBLE_CORES"

    @classmethod
    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "neuron"

    @classmethod
    def is_async_output_supported(cls, enforce_eager: bool | None) -> bool:
        return False

    @classmethod
    def check_and_update_config(cls, vllm_config) -> None:
        """Check and update the vLLM configuration for the Optimum Neuron platform.

        Unsupported configuration parameters are rejected, and the configuration is modified
        to set the worker class to `OptimumNeuronWorker`, which will laod and run the target
        model using `optimum-neuron`.
        """
        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            # Update the config to set the worker class to OptimumNeuronWorker
            parallel_config.worker_cls = "optimum.neuron.vllm.worker.OptimumNeuronWorker"

        if parallel_config.world_size > 1:
            parallel_config.distributed_executor_backend = "uni"

        if vllm_config.cache_config and vllm_config.model_config:
            # optimum-neuron only supports blocks equal to the maximum sequence length
            vllm_config.cache_config.block_size = vllm_config.model_config.max_model_len

        if vllm_config.model_config and vllm_config.model_config.use_mla:
            raise ValueError(
                "MLA (Multi-Layer Attention) is not supported on Optimum Neuron platform. "
                "Please set `use_mla` to False in the model configuration."
            )

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        return False
