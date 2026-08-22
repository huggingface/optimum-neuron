# coding=utf-8
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
import os


logger = logging.getLogger("Neuron")


def register():
    """
    Register the Optimum Neuron platform plugin for vLLM.
    This function is called to ensure that the plugin is registered when the package is imported.
    """
    # vLLM's V1 engine forks an EngineCore process by default
    # (VLLM_WORKER_MULTIPROC_METHOD=fork). Forking after the Neuron runtime and
    # torch have initialized their native thread pools leaves the child with a
    # dead neuron::ThreadPool, so weight loading deadlocks in
    # neuron::parallel_load (or aborts with "Invalid thread pool!"). Force spawn
    # so the EngineCore starts from a clean interpreter. setdefault respects an
    # explicit user override. This mirrors what `optimum-cli neuron serve` does.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    logger.info("Optimum Neuron platform plugin registered for vLLM.")
    return "optimum.neuron.vllm.platform.OptimumNeuronPlatform"
