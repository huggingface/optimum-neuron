import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

from huggingface_hub import HfApi
from transformers import PretrainedConfig


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
    """

    model_id: str
    model_type: str

    def to_dict(self) -> Dict[str, Any]:
        raise NotImplementedError

    @property
    def hash(self):
        hash_gen = hashlib.sha512()
        hash_gen.update(self.serialize().encode("utf-8"))
        return str(hash_gen.hexdigest())[:20]

    def neuron_config(self) -> Dict[str, Any]:
        raise NotImplementedError

    def has_same_arch(self, other: "ModelCacheEntry"):
        raise NotImplementedError

    @classmethod
    def from_hub(cls, model_id: str):
        raise NotImplementedError

    @staticmethod
    def create(model_id: str, config: Optional[Union[PretrainedConfig, Dict[str, Any]]] = None):
        from .multi_model import MultiModelCacheEntry
        from .single_model import SingleModelCacheEntry

        if config is not None:
            if isinstance(config, PretrainedConfig):
                return SingleModelCacheEntry(model_id, config)
            return MultiModelCacheEntry(model_id, config)
        # No config was provided: fetch it from the hub
        api = HfApi()
        if api.file_exists(model_id, "config.json"):
            return SingleModelCacheEntry.from_hub(model_id)
        elif api.file_exist(model_id, "model_index.json"):
            return MultiModelCacheEntry.from_hub(model_id)
        raise ValueError(f"Unable to evaluate model type for {model_id}: is it a diffusers or transformers model ?")

    def serialize(self) -> str:
        cache_dict = self.to_dict()
        cache_dict["_entry_class"] = self.__class__.__name__
        cache_dict["_model_id"] = self.model_id
        return json.dumps(cache_dict, sort_keys=True)

    @staticmethod
    def deserialize(data: str) -> "ModelCacheEntry":
        cache_dict = json.loads(data)
        entry_class = cache_dict.pop("_entry_class")
        model_id = cache_dict.pop("_model_id")
        if entry_class == "SingleModelCacheEntry":
            from .single_model import SingleModelCacheEntry

            return SingleModelCacheEntry.from_dict(model_id, cache_dict)
        elif entry_class == "MultiModelCacheEntry":
            from .multi_model import MultiModelCacheEntry

            return MultiModelCacheEntry.from_dict(model_id, cache_dict)
        raise ValueError(f"Invalid cache entry of type {entry_class}")
