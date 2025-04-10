# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
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
from pathlib import Path

from .hub_cache import ModelCacheEntry, create_hub_compile_cache_proxy, hub_neuronx_cache


logger = logging.getLogger(__name__)


def cache_traced_neuron_artifacts(neuron_dir: Path, cache_entry: ModelCacheEntry):
    # Use the context manager just for creating registry, AOT compilation won't leverage `create_compile_cache`
    # in `libneuronxla`, so we will need to cache compiled artifacts to local manually.
    with hub_neuronx_cache("inference", entry=cache_entry):
        compile_cache = create_hub_compile_cache_proxy()
        model_cache_dir = compile_cache.default_cache.get_cache_dir_with_cache_key(f"MODULE_{cache_entry.hash}")
        compile_cache.upload_folder(cache_dir=model_cache_dir, src_dir=neuron_dir)

        logger.info(f"Model cached in: {model_cache_dir}.")
