import copy
import json
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

    def __init__(self, model_id: str, config: Union[PretrainedConfig, Dict[str, Any]]):
        # Remove keys set to default values
        self.config = config.to_diff_dict() if isinstance(config, PretrainedConfig) else config
        # Store neuron config separately (if any) as eventually it will be passed separately
        self.neuron_config = self.config.pop("neuron", None)
        # Also remove keys in white-list
        for key in CACHE_WHITE_LIST:
            self.config.pop(key, None)
        super().__init__(model_id, self.config["model_type"])

    def to_dict(self) -> Dict[str, Any]:
        # Add neuron config when serializing
        config = copy.deepcopy(self.config)
        config["neuron"] = self.neuron_config
        return config

    @classmethod
    def from_dict(cls, model_id: str, config: Dict[str, Any]) -> "SingleModelCacheEntry":
        return cls(model_id, config)

    def has_same_arch(self, other: "SingleModelCacheEntry"):
        if not isinstance(other, SingleModelCacheEntry):
            return False
        return self.config == other.config

    @classmethod
    def from_hub(cls, model_id: str):
        config = AutoConfig.from_pretrained(model_id)
        return cls(model_id, config)
