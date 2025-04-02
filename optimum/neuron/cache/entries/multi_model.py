import copy
import json
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Union

from huggingface_hub import HfApi
from transformers import PretrainedConfig

from .cache_entry import CACHE_WHITE_LIST, ModelCacheEntry


NEURON_CONFIG_WHITE_LIST = ["input_names", "output_names", "model_type"]


def _exclude_white_list_from_config(
    config: Dict, white_list: Optional[List] = None, neuron_white_list: Optional[List] = None
):
    if white_list is None:
        white_list = CACHE_WHITE_LIST

    if neuron_white_list is None:
        neuron_white_list = NEURON_CONFIG_WHITE_LIST

    for param in white_list:
        config.pop(param, None)

    for param in neuron_white_list:
        config["neuron"].pop(param, None)

    return config


def _build_cache_config(
    configs: Dict[str, Union[PretrainedConfig, Dict[str, Any]]],
    white_list: Optional[List] = None,
    neuron_white_list: Optional[List] = None,
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

    if len(clean_configs) > 1:
        if "unet" in configs:
            # stable diffusion
            clean_configs["model_type"] = "stable-diffusion"
        elif "transformer" in configs:
            # diffusion transformer
            clean_configs["model_type"] = "diffusion-transformer"
        else:
            # seq-to-seq
            clean_configs["model_type"] = next(iter(clean_configs.values()))["model_type"]

        return clean_configs
    else:
        return next(iter(clean_configs.values()))


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


class MultiModelCacheEntry(ModelCacheEntry):
    """A class describing a model cache entry

    Args:
        model_id (`str`):
            The model id, used as a key for the cache entry.
        model_type (`str`):
            The model type, also used as a key for the cache entry.
        configs (`Dict[str, Dict[str, Any]]`):
            The configurations for the multi models pipeline.

    """

    def __init__(self, model_id: str, model_type: str, configs: Dict[str, Union[PretrainedConfig, Dict[str, Any]]]):
        super().__init__(model_id, model_type)
        self.config = _build_cache_config(configs)

    def to_json(self) -> str:
        return json.dumps(self.config, sort_keys=True)

    @classmethod
    def from_json(cls, model_id: str, config: Dict[str, Any]) -> "MultiModelCacheEntry":
        return cls(model_id, config)

    @classmethod
    def from_hub(cls, model_id):
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

        if "unet" in lookup_configs:
            model_type = "stable-diffusion"
        if "transformer" in lookup_configs:
            model_type = "diffusion-transformer"
        return cls(model_id, model_type, lookup_configs)
