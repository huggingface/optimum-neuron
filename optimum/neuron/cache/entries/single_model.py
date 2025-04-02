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
from typing import Any, Dict, Union

from transformers import AutoConfig, PretrainedConfig

from .cache_entry import CACHE_WHITE_LIST, ModelCacheEntry


class SingleModelCacheEntry(ModelCacheEntry):
    """A class describing a model cache entry

    Args:
        model_id (`str`):
            The model id, used as a key for the cache entry.
        config (`transformers.PretrainedConfig`):
            The configuration of the model.

    """

    def __init__(self, model_id: str, task: str, config: Union[PretrainedConfig, Dict[str, Any]]):
        config = copy.deepcopy(config)
        # Remove keys set to default values
        self._config = config.to_diff_dict() if isinstance(config, PretrainedConfig) else config
        # Store neuron config separately (if any) as eventually it will be passed separately
        self._neuron_config = self._config.pop("neuron", None)
        # Also remove keys in white-list
        for key in CACHE_WHITE_LIST:
            self._config.pop(key, None)
        super().__init__(model_id, self._config["model_type"], task)

    # ModelCacheEntry API implementation

    def to_dict(self) -> Dict[str, Any]:
        # Add neuron config when serializing
        config = copy.deepcopy(self._config)
        config["neuron"] = self._neuron_config
        return config

    @classmethod
    def from_dict(cls, model_id: str, task: str, config: Dict[str, Any]) -> "SingleModelCacheEntry":
        return cls(model_id=model_id, task=task, config=config)

    @property
    def neuron_config(self) -> Dict[str, Any]:
        return self._neuron_config

    def has_same_arch(self, other: "SingleModelCacheEntry"):
        if not isinstance(other, SingleModelCacheEntry):
            return False
        return self.model_type == other.model_type and self.task == other.task and self._config == other._config

    @classmethod
    def from_hub(cls, model_id: str, task: str):
        config = AutoConfig.from_pretrained(model_id)
        return cls(model_id, task, config)
