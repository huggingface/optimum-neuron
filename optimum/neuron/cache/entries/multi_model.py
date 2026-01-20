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
import glob
import hashlib
import json
import os
from tempfile import TemporaryDirectory
from typing import Any

from huggingface_hub import HfApi
from transformers import PretrainedConfig

from .cache_entry import CACHE_WHITE_LIST, ModelCacheEntry


NEURON_CONFIG_WHITE_LIST = ["input_names", "output_names", "model_type"]


def _exclude_white_list_from_config(
    config: dict, white_list: list | None = None, neuron_white_list: list | None = None
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
    white_list: list | None = None,
    neuron_white_list: list | None = None,
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


def _prepare_configs_for_matching(configs: dict, model_type: str):
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


def _merge_configs(local_path: str) -> dict[str, Any]:
    """Get all config.json files in the local path recursively and merge them into a single config."""
    config_files = glob.glob(os.path.join(local_path, "**", "config.json"), recursive=True)
    lookup_configs = {}
    for config_file in config_files:
        with open(config_file) as f:
            entry_config = json.load(f)
            for param in CACHE_WHITE_LIST:
                entry_config.pop(param, None)
            lookup_configs[config_file.split("/")[-2]] = entry_config
    return lookup_configs


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

    def arch_digest(self) -> str:
        arch_dict = {
            "model_type": self.model_type,
            "configs": _prepare_configs_for_matching(self._configs, self.model_type),
        }
        arch_json = json.dumps(arch_dict, sort_keys=True).encode("utf-8")
        return hashlib.sha256(arch_json).hexdigest()

    @classmethod
    def from_hub(cls, model_id: str):
        api = HfApi()
        repo_files = api.list_repo_files(model_id)
        config_pattern = "/config.json"
        config_files = [path for path in repo_files if config_pattern in path]
        with TemporaryDirectory() as tmpdir:
            for model_path in config_files:
                api.hf_hub_download(model_id, model_path, local_dir=tmpdir)
            lookup_configs = _merge_configs(tmpdir)
        return cls(model_id, lookup_configs)

    @classmethod
    def from_local_path(cls, model_id: str):
        return cls(model_id, _merge_configs(model_id))
