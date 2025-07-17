# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Utilities of various sorts."""

import copy
import inspect
import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import torch
from huggingface_hub import model_info, snapshot_download
from requests import HTTPError
from transformers import PretrainedConfig

from ...utils import is_diffusers_available, logging
from .import_utils import is_torch_neuronx_available


if is_torch_neuronx_available():
    from torch_neuronx import DataParallel

if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel

    if is_diffusers_available():
        from diffusers import ModelMixin


logger = logging.get_logger()


def is_precompilation() -> bool:
    return os.environ.get("NEURON_EXTRACT_GRAPHS_ONLY") == "1"


def is_main_worker(global_main: bool = True) -> bool:
    if torch.distributed.is_initialized():
        import torch_xla.core.xla_model as xm
        import torch_xla.runtime as xr

        return xr.global_ordinal() == 0 if global_main else xm.get_local_ordinal() == 0
    return True


# From https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
def string_to_bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise TypeError(
            f"Truthy value expected: got {v} but expected one of yes/no, true/false, t/f, y/n, 1/0 (case insensitive)."
        )


def args_and_kwargs_to_kwargs_only(
    f: Callable,
    args: tuple[Any, ...] | None = None,
    kwargs: dict[str, Any] | None = None,
    include_default_values: bool = False,
) -> dict[str, Any]:
    """
    Takes a function `f`, the `args` and `kwargs` provided to the function call, and returns the save arguments in the
    keyword arguments format.

    Args:
        f (`Callable`):
            The function that is being called.
        args (`tuple[Any, ...] | None`, defaults to `None`):
            The args given to `f`.
        kwargs (`dict[str, Any] | None`, defaults to `None`):
            The kwargs given to `f`.
        include_default_values (`bool`, defaults to `False`):
            Whether or not the return keyword arguments should contain parameters that were not in `args` and `kwargs`
            which have defaults values.

    Returns:
        `dict[str, Any]`: The same arguments all formated as keyword arguments.
    """
    if args is None:
        args = ()
    if kwargs is None:
        kwargs = {}
    sig = inspect.signature(f)
    param_names = list(sig.parameters)
    result = dict(zip(param_names, args))
    result.update(kwargs)
    if include_default_values:
        for param in sig.parameters.values():
            if param.name in result:
                continue
            if param.default != inspect.Parameter.empty:
                result[param.name] = param.default
    return result


def replace_weights(
    model: "torch.jit._script.RecursiveScriptModule | DataParallel",
    weights: dict[str, torch.Tensor] | torch.nn.Module,
    prefix: str = "model",
):
    """
    Replaces the weights in a Neuron Model with weights from another model, the original neuron model should have separated weights(by setting `inline_weights_to_neff=False` during the tracing).
    """

    if isinstance(weights, torch.nn.Module):
        weights = weights.state_dict()

    # extract module paths from the weights c module
    if is_torch_neuronx_available() and isinstance(model, DataParallel):
        model_weights = model.module.weights
    else:
        model_weights = model.weights
    code = model_weights._c.code
    start_str = "__parameters__ = ["
    end_str = "]\n"
    module_paths = code.split(start_str)[1].split(end_str)[0].strip()[:-1:].replace('"', "").split(", ")
    module_paths = [module_path for module_path in module_paths if module_path != ""]

    for module_path in module_paths:
        if len(re.findall("\w\d+", module_path)) > 0:
            continue
        else:
            model_weights._c.setattr(
                module_path, weights[module_path.replace(prefix + "->", "", 1).replace("->", ".")]
            )


def check_if_weights_replacable(
    config: "PretrainedConfig | dict[str, PretrainedConfig]",
    weights: dict[str, torch.Tensor] | torch.nn.Module | None,
):
    def _is_weights_neff_separated(config):
        return not config.neuron.get("inline_weights_to_neff", True) if hasattr(config, "neuron") else False

    if isinstance(config, PretrainedConfig):
        is_weights_neff_separated = _is_weights_neff_separated(config)
    elif isinstance(config, dict):
        is_weights_neff_separated = []
        for _, config_value in config.items():
            is_weights_neff_separated.append(_is_weights_neff_separated(config_value))
        is_weights_neff_separated = all(is_weights_neff_separated)

    if weights is not None and not is_weights_neff_separated:
        raise RuntimeError(
            "Unable to replace weights of the neuron model since its weights and neff are not separated, please set `inline_weights_to_neff=False` when converting the model to Neuron format."
        )


class DiffusersPretrainedConfig(PretrainedConfig):
    """override to update `model_type`."""

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary.

        Returns:
            :obj:`dict[str, any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        return output


def get_stable_diffusion_configs(
    models_for_export: dict[str, "PreTrainedModel | ModelMixin"],
):
    subfolders = ["text_encoder", "text_encoder_2", "unet", "vae"]
    configs = {}
    for name in subfolders:
        if name in models_for_export:
            configs[name] = models_for_export[name].config

    return configs


def map_torch_dtype(dtype: str | torch.dtype):
    dtype_mapping = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "int32": torch.int32,
        "int64": torch.int64,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }

    if isinstance(dtype, str) and dtype in dtype_mapping:
        dtype = dtype_mapping.get(dtype)

    return dtype


# Adapted from https://github.com/huggingface/diffusers/blob/d4dc4d7654d4855e516be1989f54bfeea736090e/src/diffusers/utils/hub_utils.py#L341
def get_checkpoint_shard_files(
    pretrained_model_name_or_path,
    index_filename,
    subfolder="",
    cache_dir=None,
    proxies=None,
    local_files_only=False,
    token=None,
    user_agent=None,
    revision=None,
):
    """
    For a given model:

    - download and cache all the shards of a sharded checkpoint if `pretrained_model_name_or_path` is a model ID on the
      Hub
    - returns the list of paths to all the shards, as well as some metadata.
    """
    if not os.path.isfile(index_filename):
        raise ValueError(f"Can't find a checkpoint index ({index_filename}) in {pretrained_model_name_or_path}.")

    with open(index_filename, "r") as f:
        index = json.loads(f.read())

    original_shard_filenames = sorted(set(index["weight_map"].values()))
    sharded_metadata = index["metadata"]
    sharded_metadata["all_checkpoint_keys"] = list(index["weight_map"].keys())
    sharded_metadata["weight_map"] = index["weight_map"].copy()
    shards_path = os.path.join(pretrained_model_name_or_path, subfolder)

    if not os.path.isdir(pretrained_model_name_or_path):
        # If pretrained_model_name_or_path is a model identifier on the Hub.
        allow_patterns = original_shard_filenames
        if subfolder is not None:
            allow_patterns = [os.path.join(subfolder, p) for p in allow_patterns]
        ignore_patterns = ["*.json", "*.md"]
        # `model_info` call must guarded with the above condition.
        model_files_info = model_info(pretrained_model_name_or_path, revision=revision, token=token)
        for shard_file in original_shard_filenames:
            shard_file_present = any(shard_file in k.rfilename for k in model_files_info.siblings)
            if not shard_file_present:
                raise EnvironmentError(
                    f"{shards_path} does not appear to have a file named {shard_file} which is "
                    "required according to the checkpoint index."
                )
        try:
            # Load from URL
            cached_folder = snapshot_download(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                allow_patterns=allow_patterns,
                ignore_patterns=ignore_patterns,
                user_agent=user_agent,
            )
            if subfolder is not None:
                cached_folder = os.path.join(cached_folder, subfolder)
            shards_path = cached_folder

        # We have already dealt with RepositoryNotFoundError and RevisionNotFoundError when getting the index, so
        # we don't have to catch them here. We have also dealt with EntryNotFoundError.
        except HTTPError as e:
            hf_co_resolve_endpoint = os.environ.get("HF_ENDPOINT", "https://huggingface.co")
            raise EnvironmentError(
                f"We couldn't connect to '{hf_co_resolve_endpoint}' to load {pretrained_model_name_or_path}. You should try"
                " again after checking your internet connection."
            ) from e

    # List of paths for sharded ckpts.
    shards_file_paths = []
    for shard_filename in original_shard_filenames:
        shards_file_paths.append(Path(shards_path, shard_filename).as_posix())

    return shards_file_paths, sharded_metadata
