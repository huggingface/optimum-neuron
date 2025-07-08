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
import functools
import inspect
import json
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Tuple, Union

import torch
from huggingface_hub import model_info, snapshot_download
from requests import HTTPError
from transformers import PretrainedConfig
from transformers.modeling_utils import _add_variant
from transformers.utils import (
    FLAX_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    cached_file,
    download_url,
    has_file,
    is_remote_url,
)

from ...utils import is_diffusers_available, logging
from .import_utils import is_torch_neuronx_available
from .require_utils import requires_safetensors


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
def string_to_bool(v: Union[str, bool]) -> bool:
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
    args: Optional[Tuple[Any, ...]] = None,
    kwargs: Optional[Dict[str, Any]] = None,
    include_default_values: bool = False,
) -> Dict[str, Any]:
    """
    Takes a function `f`, the `args` and `kwargs` provided to the function call, and returns the save arguments in the
    keyword arguments format.

    Args:
        f (`Callable`):
            The function that is being called.
        args (`Optional[Tuple[Any, ...]]`, defaults to `None`):
            The args given to `f`.
        kwargs (`Optional[Dict[str, Any]]`, defaults to `None`):
            The kwargs given to `f`.
        include_default_values (`bool`, defaults to `False`):
            Whether or not the return keyword arguments should contain parameters that were not in `args` and `kwargs`
            which have defaults values.

    Returns:
        `Dict[str, Any]`: The same arguments all formated as keyword arguments.
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


def _original_filename_to_safetensors_filename(filename: str) -> str:
    """Transforms the filename for any kind of checkpoint to a safetensors equivalent."""
    _, extension = filename.rsplit(".", maxsplit=1)
    pattern = rf"\w+(-[0-9]*-of-[0-9]*)?\.{extension}"
    match_ = re.match(pattern, filename)
    if not match_:
        raise ValueError(f"Could not convert {filename} to a safetensor filename.")
    group_1 = match_.group(1)
    index_out_of_total_str = group_1 if group_1 is not None else ""
    safetensor_filename, safetensor_extension = SAFE_WEIGHTS_NAME.rsplit(".", maxsplit=1)
    return f"{safetensor_filename}{index_out_of_total_str}.{safetensor_extension}"


@requires_safetensors
def convert_checkpoint_to_safetensors(
    weight_file: Union[str, Path],
    output_dir: Optional[Union[str, Path]] = None,
    safetensors_weight_filename_prefix: Optional[str] = None,
    log: bool = False,
) -> Path:
    """
    Converts a PyTorch model checkpoint to a `safetensors` model checkpoint.

    Args:
        weight_file (`Union[str, Path]`):
            The path to the PyTorch model checkpoint.
        output_dir (`Optional[Union[str, Path]]`, defaults to `None`):
            The output directory where the `safetensors` checkpoint will be saved.
            If left unspecified, the parent directory of the PyTorch checkpoint will be used.
        safetensors_weight_filename_prefix (`Optional[str]`, defaults to `None`):
            If specified, the name of the converted file will be prefixed by "safetensors_weight_filename_prefix-".
        log (`bool`, defaults to `False`):
            Whether or not the function should log which file it is converting.


    Returns:
        `Path`: The path to the `safetensors` checkpoint.
    """
    from safetensors.torch import save_file

    if not isinstance(weight_file, Path):
        weight_file = Path(weight_file)

    if output_dir is None:
        output_dir = weight_file.parent

    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    if weight_file.suffix != ".bin":
        raise ValueError("Can only convert PyTorch checkpoints to safetensors.")

    safetensors_filename = _original_filename_to_safetensors_filename(weight_file.name)
    if safetensors_weight_filename_prefix is not None:
        safetensors_filename = f"{safetensors_weight_filename_prefix}-{safetensors_filename}"
    safetensors_path = output_dir / safetensors_filename

    already_exists = safetensors_path.is_file()
    is_distributed = torch.distributed.is_initialized()
    is_main_process = is_distributed and torch.distributed.get_rank() == 0

    # Only one worker will load the checkpoint (potentially huge) and perform the conversion.
    if not already_exists and (not is_distributed or is_main_process):
        if log:
            logger.info(f"Converting {weight_file} to safetensors")
        checkpoint = torch.load(weight_file, map_location=torch.device("cpu"))
        data_pointers = set()
        for k, v in checkpoint.items():
            if v.data_ptr() in data_pointers:
                v = v.detach().clone()
            v = v.contiguous()
            checkpoint[k] = v
            data_pointers.add(v.data_ptr())
        save_file(checkpoint, safetensors_path)
        del checkpoint

    return safetensors_path


@functools.wraps(cached_file)
def distributed_friendly_cached_file(*args, **kwargs):
    import torch_xla.core.xla_model as xm

    if is_main_worker():
        output = cached_file(*args, **kwargs)
    xm.rendezvous("Cached file done")
    if not is_main_worker():
        output = cached_file(*args, **kwargs)
    return output


def download_checkpoints_in_cache(
    pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    local_files_only: bool = False,
    token: Optional[Union[str, bool]] = None,
    revision: str = "main",
    use_safetensors: Optional[bool] = None,
    use_safetensors_in_priority: Optional[bool] = None,
    convert_to_safetensors: bool = False,
    **kwargs,
):
    """
    Downloads checkpoint to the cache or returns the path to the already downloaded files.

    Note: This is a transformed version of `transformers.PreTrainedModel.from_pretrained` where only the part about
    downloading checkpoints has been kept. At the end of the function a custom part has been added handling the
    conversion to safetensors if needed.
    """
    kwargs.pop("state_dict", None)
    from_tf = kwargs.pop("from_tf", False)
    from_flax = kwargs.pop("from_flax", False)
    resume_download = kwargs.pop("resume_download", None)
    proxies = kwargs.pop("proxies", None)
    kwargs.pop("output_loading_info", False)
    kwargs.pop("use_auth_token", None)
    kwargs.pop("trust_remote_code", None)
    _ = kwargs.pop("mirror", None)
    from_pipeline = kwargs.pop("_from_pipeline", None)
    from_auto_class = kwargs.pop("_from_auto", False)
    kwargs.pop("torch_dtype", None)
    kwargs.pop("low_cpu_mem_usage", None)
    kwargs.pop("device_map", None)
    kwargs.pop("max_memory", None)
    kwargs.pop("offload_folder", None)
    kwargs.pop("offload_state_dict", False)
    kwargs.pop("load_in_8bit", False)
    kwargs.pop("load_in_4bit", False)
    kwargs.pop("quantization_config", None)
    subfolder = kwargs.pop("subfolder", "")
    commit_hash = kwargs.pop("_commit_hash", None)
    variant = kwargs.pop("variant", None)

    # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
    # index of the files.
    is_sharded = False
    sharded_metadata = None

    # Load model
    user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
    if from_pipeline is not None:
        user_agent["using_pipeline"] = from_pipeline

    if pretrained_model_name_or_path is not None:
        pretrained_model_name_or_path = str(pretrained_model_name_or_path)
        is_local = os.path.isdir(pretrained_model_name_or_path)
        if is_local:
            if from_tf and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + ".index")
            ):
                # Load from a TF 1.0 checkpoint in priority if from_tf
                archive_file = os.path.join(pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + ".index")
            elif from_tf and os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME)):
                # Load from a TF 2.0 checkpoint in priority if from_tf
                archive_file = os.path.join(pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME)
            elif from_flax and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)
            ):
                # Load from a Flax checkpoint in priority if from_flax
                archive_file = os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)
            elif use_safetensors is not False and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant))
            ):
                # Load from a safetensors checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant)
                )
            elif use_safetensors is not False and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant))
            ):
                # Load from a sharded safetensors checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)
                )
                is_sharded = True
            elif use_safetensors_in_priority is not False and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant))
            ):
                # Load from a safetensors checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_NAME, variant)
                )
            elif use_safetensors_in_priority is not False and os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant))
            ):
                # Load from a sharded safetensors checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)
                )
                is_sharded = True
            elif os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_NAME, variant))
            ):
                # Load from a PyTorch checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_NAME, variant)
                )
            elif os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant))
            ):
                # Load from a sharded PyTorch checkpoint
                archive_file = os.path.join(
                    pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant)
                )
                is_sharded = True
            # At this stage we don't have a weight file so we will raise an error.
            elif os.path.isfile(
                os.path.join(pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + ".index")
            ) or os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME)):
                raise EnvironmentError(
                    f"Error no file named {_add_variant(WEIGHTS_NAME, variant)} found in directory"
                    f" {pretrained_model_name_or_path} but there is a file for TensorFlow weights. Use"
                    " `from_tf=True` to load this model from those weights."
                )
            elif os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)):
                raise EnvironmentError(
                    f"Error no file named {_add_variant(WEIGHTS_NAME, variant)} found in directory"
                    f" {pretrained_model_name_or_path} but there is a file for Flax weights. Use `from_flax=True`"
                    " to load this model from those weights."
                )
            elif use_safetensors:
                raise EnvironmentError(
                    f"Error no file named {_add_variant(SAFE_WEIGHTS_NAME, variant)} found in directory"
                    f" {pretrained_model_name_or_path}."
                )
            else:
                raise EnvironmentError(
                    f"Error no file named {_add_variant(WEIGHTS_NAME, variant)}, {TF2_WEIGHTS_NAME},"
                    f" {TF_WEIGHTS_NAME + '.index'} or {FLAX_WEIGHTS_NAME} found in directory"
                    f" {pretrained_model_name_or_path}."
                )
        elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path)):
            archive_file = pretrained_model_name_or_path
            is_local = True
        elif os.path.isfile(os.path.join(subfolder, pretrained_model_name_or_path + ".index")):
            if not from_tf:
                raise ValueError(
                    f"We found a TensorFlow checkpoint at {pretrained_model_name_or_path + '.index'}, please set "
                    "from_tf to True to load from this checkpoint."
                )
            archive_file = os.path.join(subfolder, pretrained_model_name_or_path + ".index")
            is_local = True
        elif is_remote_url(pretrained_model_name_or_path):
            filename = pretrained_model_name_or_path
            resolved_archive_file = download_url(pretrained_model_name_or_path)
        else:
            # set correct filename
            if from_tf:
                filename = TF2_WEIGHTS_NAME
            elif from_flax:
                filename = FLAX_WEIGHTS_NAME
            elif use_safetensors is not False:
                filename = _add_variant(SAFE_WEIGHTS_NAME, variant)
            elif use_safetensors_in_priority is not False:
                filename = _add_variant(SAFE_WEIGHTS_NAME, variant)
            else:
                filename = _add_variant(WEIGHTS_NAME, variant)

            try:
                # Load from URL or cache if already cached
                cached_file_kwargs = {
                    "cache_dir": cache_dir,
                    "force_download": force_download,
                    "proxies": proxies,
                    "resume_download": resume_download,
                    "local_files_only": local_files_only,
                    "use_auth_token": token,
                    "user_agent": user_agent,
                    "revision": revision,
                    "subfolder": subfolder,
                    "_raise_exceptions_for_missing_entries": False,
                    "_commit_hash": commit_hash,
                }

                resolved_archive_file = distributed_friendly_cached_file(
                    pretrained_model_name_or_path, filename, **cached_file_kwargs
                )

                # Since we set _raise_exceptions_for_missing_entries=False, we don't get an exception but a None
                # result when internet is up, the repo and revision exist, but the file does not.
                if resolved_archive_file is None and filename == _add_variant(SAFE_WEIGHTS_NAME, variant):
                    # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                    resolved_archive_file = distributed_friendly_cached_file(
                        pretrained_model_name_or_path,
                        _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant),
                        **cached_file_kwargs,
                    )
                    if resolved_archive_file is not None:
                        is_sharded = True
                    elif use_safetensors:
                        raise EnvironmentError(
                            f" {_add_variant(SAFE_WEIGHTS_NAME, variant)} or "
                            f"{_add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)} and thus cannot be loaded with "
                            "`safetensors`. Please make sure that the model has been saved with "
                            "`safe_serialization=True` or do not set `use_safetensors=True`."
                        )
                    else:
                        # This repo has no safetensors file of any kind, we switch to PyTorch.
                        filename = _add_variant(WEIGHTS_NAME, variant)
                        resolved_archive_file = distributed_friendly_cached_file(
                            pretrained_model_name_or_path, filename, **cached_file_kwargs
                        )
                if resolved_archive_file is None and filename == _add_variant(WEIGHTS_NAME, variant):
                    # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                    resolved_archive_file = distributed_friendly_cached_file(
                        pretrained_model_name_or_path,
                        _add_variant(WEIGHTS_INDEX_NAME, variant),
                        **cached_file_kwargs,
                    )
                    if resolved_archive_file is not None:
                        is_sharded = True
                if resolved_archive_file is None:
                    # Otherwise, maybe there is a TF or Flax model file.  We try those to give a helpful error
                    # message.
                    has_file_kwargs = {
                        "revision": revision,
                        "proxies": proxies,
                        "use_auth_token": token,
                    }
                    if has_file(pretrained_model_name_or_path, TF2_WEIGHTS_NAME, **has_file_kwargs):
                        raise EnvironmentError(
                            f"{pretrained_model_name_or_path} does not appear to have a file named"
                            f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file for TensorFlow weights."
                            " Use `from_tf=True` to load this model from those weights."
                        )
                    elif has_file(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME, **has_file_kwargs):
                        raise EnvironmentError(
                            f"{pretrained_model_name_or_path} does not appear to have a file named"
                            f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file for Flax weights. Use"
                            " `from_flax=True` to load this model from those weights."
                        )
                    elif variant is not None and has_file(
                        pretrained_model_name_or_path, WEIGHTS_NAME, **has_file_kwargs
                    ):
                        raise EnvironmentError(
                            f"{pretrained_model_name_or_path} does not appear to have a file named"
                            f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file without the variant"
                            f" {variant}. Use `variant=None` to load this model from those weights."
                        )
                    else:
                        raise EnvironmentError(
                            f"{pretrained_model_name_or_path} does not appear to have a file named"
                            f" {_add_variant(WEIGHTS_NAME, variant)}, {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or"
                            f" {FLAX_WEIGHTS_NAME}."
                        )
            except EnvironmentError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                # to the original exception.
                raise
            except Exception:
                # For any other exception, we throw a generic error.
                raise EnvironmentError(
                    f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it"
                    " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                    f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                    f" directory containing a file named {_add_variant(WEIGHTS_NAME, variant)},"
                    f" {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or {FLAX_WEIGHTS_NAME}."
                )

        if is_local:
            resolved_archive_file = archive_file
    else:
        resolved_archive_file = None

    # We'll need to download and cache each checkpoint shard if the checkpoint is sharded.
    if is_sharded:
        # resolved_archive_file becomes a list of files that point to the different checkpoint shards in this case.
        resolved_archive_file, sharded_metadata = get_checkpoint_shard_files(
            pretrained_model_name_or_path,
            resolved_archive_file,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            resume_download=resume_download,
            local_files_only=local_files_only,
            use_auth_token=token,
            user_agent=user_agent,
            revision=revision,
            subfolder=subfolder,
            _commit_hash=commit_hash,
        )

    # TODO: this whole bulk is not very optimized, improve it once the tests are written.
    if convert_to_safetensors:
        maybe_to_convert = resolved_archive_file
        if not isinstance(maybe_to_convert, list):
            maybe_to_convert = [maybe_to_convert]

        filenames_to_safetensors_filenames = {}
        for filename in maybe_to_convert:
            filename = Path(filename)
            if filename.suffix == ".safetensors":
                filenames_to_safetensors_filenames[filename.name] = filename
            elif filename.suffix == ".bin":
                output_path = convert_checkpoint_to_safetensors(
                    filename, safetensors_weight_filename_prefix="converted", log=True
                )
                filenames_to_safetensors_filenames[filename.name] = output_path
            else:
                raise ValueError("Only PyTorch and safetensors files are supported.")

        if sharded_metadata is not None:
            weight_map = sharded_metadata["weight_map"]
            for weight_name, torch_filename in weight_map.items():
                weight_map[weight_name] = filenames_to_safetensors_filenames[torch_filename]

        if isinstance(resolved_archive_file, list):
            resolved_archive_file = [
                filenames_to_safetensors_filenames[Path(filename).name] for filename in resolved_archive_file
            ]
        else:
            resolved_archive_file = filenames_to_safetensors_filenames[Path(resolved_archive_file).name]

    return resolved_archive_file, sharded_metadata


def replace_weights(
    model: Union[torch.jit._script.RecursiveScriptModule, "DataParallel"],
    weights: Union[Dict[str, torch.Tensor], torch.nn.Module],
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
    config: Union["PretrainedConfig", Dict[str, "PretrainedConfig"]],
    weights: Optional[Union[Dict[str, torch.Tensor], torch.nn.Module]],
):
    def _is_weights_neff_separated(config):
        return not config.neuron.get("inline_weights_to_neff", True) if hasattr(config, "neuron") else False

    if isinstance(config, PretrainedConfig):
        is_weights_neff_separated = _is_weights_neff_separated(config)
    elif isinstance(config, Dict):
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
            :obj:`Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self.__dict__)
        return output


def get_stable_diffusion_configs(
    models_for_export: Dict[str, Union["PreTrainedModel", "ModelMixin"]],
):
    subfolders = ["text_encoder", "text_encoder_2", "unet", "vae"]
    configs = {}
    for name in subfolders:
        if name in models_for_export:
            configs[name] = models_for_export[name].config

    return configs


def map_torch_dtype(dtype: Union[str, torch.dtype]):
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
