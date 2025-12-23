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

from vllm.platforms.interface import UnspecifiedPlatform
from vllm.utils import FlexibleArgumentParser


logger = logging.getLogger("Neuron")


class OptimumNeuronPlatform(UnspecifiedPlatform):
    device_name: str = "neuron"
    # Device type is set to "cpu" to prevent vLLM from preemptively moving tensors
    # to the XLA device and trigger spurious neuron runtime intializations.
    # The CPU tensors will be moved when required to the XLA device by the neuron SDK.
    device_type: str = "cpu"
    ray_device_key: str = "neuron_cores"
    device_control_env_var: str = "NEURON_RT_VISIBLE_CORES"

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "neuron"

    @classmethod
    def pre_register_and_update(cls, parser: FlexibleArgumentParser | None = None) -> None:
        from vllm.config import model

        # Patch ModelConfig to avoid hard-coded check in vLLM
        def verify_with_parallel_config(self, parallel_config) -> None:
            # The original method checks that the tensor_parallel_size divides
            # the number of attention heads, which is not necessarily true for
            # Neuron models (e.g., Llama 4 Scout 17B with TP=32).
            # We override the method to skip this check.
            logger.info("Disabling ModelConfig verification with parallel config for Optimum Neuron platform (class).")
            pass

        model.ModelConfig.verify_with_parallel_config = verify_with_parallel_config

    @classmethod
    def check_and_update_config(cls, vllm_config) -> None:
        """Check and update the vLLM configuration for the Optimum Neuron platform.

        Unsupported configuration parameters are rejected, and the configuration is modified
        to set the worker class to `OptimumNeuronWorker`, which will load and run the target
        model using `optimum-neuron`.
        """
        parallel_config = vllm_config.parallel_config
        if parallel_config.worker_cls == "auto":
            # Update the config to set the worker class to OptimumNeuronWorker
            parallel_config.worker_cls = "optimum.neuron.vllm.worker.OptimumNeuronWorker"

        if parallel_config.world_size > 1:
            parallel_config.distributed_executor_backend = "uni"

        if vllm_config.cache_config:
            # Disable prefix-caching as it's not supported on optimum-neuron
            vllm_config.cache_config.enable_prefix_caching = False
            if vllm_config.model_config:
                # optimum-neuron only supports blocks equal to the maximum sequence length
                vllm_config.cache_config.block_size = vllm_config.model_config.max_model_len

        if vllm_config.model_config:
            if vllm_config.model_config.use_mla:
                raise ValueError(
                    "MLA (Multi-Layer Attention) is not supported on Optimum Neuron platform. "
                    "Please set `use_mla` to False in the model configuration."
                )

            # Patch ModelConfig to avoid hard-coded check in vLLM
            def verify_with_parallel_config(parallel_config) -> None:
                # The original method checks that the tensor_parallel_size divides
                # the number of attention heads, which is not necessarily required for
                # Neuron models, since we use padding (e.g., Llama 4 Scout 17B with TP=32).
                # We override the method to skip this check.
                logger.info(
                    "Disabling ModelConfig verification with parallel config for Optimum Neuron platform (instance)."
                )
                pass

            vllm_config.model_config.verify_with_parallel_config = verify_with_parallel_config

    @classmethod
    def is_pin_memory_available(cls) -> bool:
        return False

    @classmethod
    def use_all_gather(cls) -> bool:
        return True
