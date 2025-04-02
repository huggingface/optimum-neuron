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
from typing import Dict

from ..utils.import_utils import is_neuronx_available
from .hub_cache import ModelCacheEntry, create_hub_compile_cache_proxy, hub_neuronx_cache


if is_neuronx_available():
    from libneuronxla.neuron_cc_cache import (
        CacheUrl,
        CompileCache,
        CompileCacheFs,
        CompileCacheS3,
        create_compile_cache,
    )
else:

    class CacheUrl:
        pass

    class CompileCache:
        pass

    class CompileCacheFs:
        pass

    class CompileCacheS3:
        pass

    def create_compile_cache():
        pass


logger = logging.getLogger(__name__)

CACHE_WHITE_LIST = [
    "_name_or_path",
    "transformers_version",
    "_diffusers_version",
    "eos_token_id",
    "bos_token_id",
    "pad_token_id",
    "torchscript",
    "torch_dtype",
    "_commit_hash",
    "sample_size",
    "projection_dim",
    "_use_default_values",
    "_attn_implementation_autoset",
]
NEURON_CONFIG_WHITE_LIST = ["input_names", "output_names", "model_type"]

DEFAULT_PATH_FOR_NEURON_CC_WRAPPER = Path(__file__).parent.as_posix()


def _prepare_config_for_matching(entry_config: Dict, target_entry: ModelCacheEntry, model_type: str):
    if model_type == "stable-diffusion":
        # Remove neuron config for comparison as the target does not have it
        neuron_config = entry_config["unet"].pop("neuron")
        non_checked_components = [
            "vae",
            "vae_encoder",
            "vae_decoder",
        ]  # Exclude vae configs from the check for now since it's complex and not mandatory
        for param in non_checked_components:
            entry_config.pop(param, None)
            target_entry.config.pop(param, None)
        target_entry_config = target_entry.config
    else:
        # Remove neuron config for comparison as the target does not have it
        neuron_config = entry_config.pop("neuron")
        entry_config = {"model": entry_config}
        target_entry_config = {"model": target_entry.config}

    return entry_config, target_entry_config, neuron_config


def lookup_matched_entries(entry_config, target_entry, white_list, model_entries, model_type: str):
    is_matched = True
    entry_config, target_entry_config, neuron_config = _prepare_config_for_matching(
        entry_config, target_entry, model_type
    )
    for name, value in entry_config.items():
        if isinstance(value, dict):
            for param in white_list:
                value.pop(param, None)
                target_entry_config[name].pop(param, None)
            for term in set(entry_config[name]).intersection(set(target_entry_config[name])):
                if entry_config[name][term] != target_entry_config[name][term]:
                    is_matched = False
                    break
        else:
            if value != target_entry_config[name]:
                is_matched = False
                break
    if is_matched:
        neuron_config.pop("model_type", None)
        model_entries.append(neuron_config)

    return model_entries


def cache_traced_neuron_artifacts(neuron_dir: Path, cache_entry: ModelCacheEntry):
    # Use the context manager just for creating registry, AOT compilation won't leverage `create_compile_cache`
    # in `libneuronxla`, so we will need to cache compiled artifacts to local manually.
    with hub_neuronx_cache("inference", entry=cache_entry):
        compile_cache = create_hub_compile_cache_proxy()
        model_cache_dir = compile_cache.default_cache.get_cache_dir_with_cache_key(f"MODULE_{cache_entry.hash}")
        compile_cache.upload_folder(cache_dir=model_cache_dir, src_dir=neuron_dir)

        logger.info(f"Model cached in: {model_cache_dir}.")
