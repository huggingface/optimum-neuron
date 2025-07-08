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

import copy
import json
from tempfile import TemporaryDirectory
from typing import Any, Dict, Union

from huggingface_hub import HfApi
from transformers import PretrainedConfig

from .cache_entry import CACHE_WHITE_LIST, ModelCacheEntry


NEURON_CONFIG_WHITE_LIST = ["input_names", "output_names", "model_type"]


def _exclude_white_list_from_config(
    config: Dict, white_list: List | None = None, neuron_white_list: List | None = None
):
    if white_list is None:
        white_list = CACHE_WHITE_LIST

    if neuron_white_list is None:
        neuron_white_list = NEURON_CONFIG_WHITE_LIST

    for param in white_list:
        config.pop(param, None)

    if "neuron" in config:
        for param in neuron_white_list:
            config["neuron"].pop(param, None)

    return config


def _clean_configs(
    configs: dict[str, PretrainedConfig | dict[str, Any]],
    white_list: List | None = None,
    neuron_white_list: List | None = None,
):
    """Only applied on traced TorchScript models."""
    clean_configs = {}
    no_check_components = [
        "vae",
        "vae_encoder",
        "vae_decoder",
    ]  # Exclude vae configs from stable diffusion pipeline since it's complex and not mandatory
    for name, config in configs.items():
        if name in no_check_components:
            continue
        config = copy.deepcopy(config).to_diff_dict() if isinstance(config, PretrainedConfig) else config
        config = _exclude_white_list_from_config(config, white_list, neuron_white_list)
        clean_configs[name] = config
    return clean_configs


def _prepare_configs_for_matching(configs: Dict, model_type: str):
    if model_type == "stable-diffusion":
        non_checked_components = [
            "vae",
            "vae_encoder",
            "vae_decoder",
        ]  # Exclude vae configs from the check for now since it's complex and not mandatory
    else:
        # FIXME: this should be done also for diffusion transformer
        raise NotImplementedError
    new_configs = {}
    for name in configs:
        if name in non_checked_components:
            continue
        new_configs[name] = copy.deepcopy(configs[name])
        # Remove neuron config for comparison
        if "neuron" in new_configs[name]:
            new_configs[name].pop("neuron")

    return new_configs


class MultiModelCacheEntry(ModelCacheEntry):
    """A class describing a model cache entry

    Args:
        model_id (`str`):
            The model id, used as a key for the cache entry.
        model_type (`str`):
            The model type, also used as a key for the cache entry.
        configs (`dict[str, dict[str, Any]]`):
            The configurations for the multi models pipeline.

    """

    def __init__(self, model_id: str, configs: dict[str, PretrainedConfig | dict[str, Any]]):
        self._configs = _clean_configs(configs)
        if "unet" in self._configs:
            model_type = "stable-diffusion"
        elif "transformer" in self._configs:
            model_type = "diffusion-transformer"
        elif "encoder" in self._configs:
            model_type = self._configs["encoder"]["model_type"]
        else:
            raise NotImplementedError
        # Task is None for multi model cache entries since we cache the whole pipeline
        # and not a single combination of sub-models corresponding to one of the tasks
        super().__init__(model_id, model_type, task=None)

    # ModelCacheEntry API implementation

    def to_dict(self) -> dict[str, Any]:
        return self._configs

    @classmethod
    def from_dict(cls, model_id: str, configs: dict[str, Any]) -> "MultiModelCacheEntry":
        return cls(model_id=model_id, configs=configs)

    @property
    def neuron_config(self) -> dict[str, Any]:
        # FIXME: Return neuron config of one of the models
        if self.model_type == "stable-diffusion":
            config = self._configs["unet"]
        else:
            raise NotImplementedError
        return config.get("neuron", None)

    def has_same_arch(self, other: "MultiModelCacheEntry"):
        if not isinstance(other, MultiModelCacheEntry):
            return False
        if self.model_type != other.model_type:
            return False
        # When comparing configs we remove the neuron configs
        configs = _prepare_configs_for_matching(self._configs, self.model_type)
        other_configs = _prepare_configs_for_matching(other._configs, other.model_type)
        if configs.keys() != other_configs.keys():
            return False
        for name, config in configs.items():
            other_config = other_configs[name]

            # We only verify that one of the configs contains the other
            # This is because the configs are stripped down when serialized
            def contains(container: dict[str, Any], containee: dict[str, Any]):
                for name, value in containee.items():
                    if name not in container:
                        return False
                    if value != container[name]:
                        return False
                return True

            if not contains(config, other_config) and not contains(other_config, config):
                return False
        return True

    @classmethod
    def from_hub(cls, model_id: str):
        api = HfApi()
        repo_files = api.list_repo_files(model_id)
        config_pattern = "/config.json"
        config_files = [path for path in repo_files if config_pattern in path]
        lookup_configs = {}
        with TemporaryDirectory() as tmpdir:
            for model_path in config_files:
                local_path = api.hf_hub_download(model_id, model_path, local_dir=tmpdir)
                with open(local_path) as f:
                    entry_config = json.load(f)
                    white_list = CACHE_WHITE_LIST
                    for param in white_list:
                        entry_config.pop(param, None)
                    lookup_configs[model_path.split("/")[-2]] = entry_config

        return cls(model_id, lookup_configs)
