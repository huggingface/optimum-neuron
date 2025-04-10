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
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict

from huggingface_hub import HfApi


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


@dataclass
class ModelCacheEntry:
    """A class describing a model cache entry

    Args:
        model_id (`str`):
            The model id, used as a key for the cache entry.
        model_type (`str`):
            The model type, also used as a key for the cache entry.
        task (`str`):
            The task name.
    """

    model_id: str
    model_type: str
    task: str

    # ModelCacheEntry API to be implemented by subclasses

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    def neuron_config(self) -> Dict[str, Any]:
        raise NotImplementedError

    def has_same_arch(self, other: "ModelCacheEntry"):
        raise NotImplementedError

    # Generic methods relying on the API above

    @property
    def hash(self):
        hash_gen = hashlib.sha512()
        hash_gen.update(self.serialize().encode("utf-8"))
        return str(hash_gen.hexdigest())[:20]

    @staticmethod
    def create(model_id: str, task: str):
        from .multi_model import MultiModelCacheEntry
        from .single_model import SingleModelCacheEntry

        api = HfApi()
        if api.file_exists(model_id, "config.json"):
            return SingleModelCacheEntry.from_hub(model_id, task)
        elif api.file_exists(model_id, "model_index.json"):
            return MultiModelCacheEntry.from_hub(model_id)
        raise ValueError(f"Unable to evaluate model type for {model_id}: is it a diffusers or transformers model ?")

    def serialize(self) -> str:
        cache_dict = self.to_dict()
        cache_dict["_entry_class"] = self.__class__.__name__
        cache_dict["_model_id"] = self.model_id
        cache_dict["_task"] = self.task
        return json.dumps(cache_dict, indent=2, sort_keys=True)

    @staticmethod
    def deserialize(data: str) -> "ModelCacheEntry":
        cache_dict = json.loads(data)
        entry_class = cache_dict.pop("_entry_class")
        model_id = cache_dict.pop("_model_id")
        task = cache_dict.pop("_task")
        if entry_class == "SingleModelCacheEntry":
            from .single_model import SingleModelCacheEntry

            return SingleModelCacheEntry.from_dict(model_id, task, cache_dict)
        elif entry_class == "MultiModelCacheEntry":
            from .multi_model import MultiModelCacheEntry

            return MultiModelCacheEntry.from_dict(model_id, cache_dict)
        raise ValueError(f"Invalid cache entry of type {entry_class}")
