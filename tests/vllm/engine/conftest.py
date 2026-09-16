# coding=utf-8
# Copyright 2026 The HuggingFace Team. All rights reserved.
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
import pytest


@pytest.fixture
def vllm_llm():
    """Return a factory creating vLLM engines that are shut down after the test.

    vLLM runs the engine core in a spawned process that is only stopped when the
    ``LLM`` object is finalized. Leaving that to the garbage collector is not
    deterministic: when the object survives the test, the engine core is still
    running at interpreter exit, where multiprocessing joins it and pytest hangs
    forever. Shutting the engine down here keeps the teardown independent of the
    garbage collector, including when the test fails.
    """
    from vllm import LLM

    engines = []

    def create(**kwargs) -> LLM:
        llm = LLM(**kwargs)
        engines.append(llm)
        return llm

    yield create

    for llm in engines:
        llm.llm_engine.engine_core.shutdown()
