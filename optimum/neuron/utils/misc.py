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

import inspect
import os
import re
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
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
from transformers.utils.hub import get_checkpoint_shard_files

from ...utils import logging
from .require_utils import requires_safetensors


logger = logging.get_logger()


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


def download_checkpoints_in_cache(
    pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    force_download: bool = False,
    local_files_only: bool = False,
    token: Optional[Union[str, bool]] = None,
    revision: str = "main",
    use_safetensors: Optional[bool] = None,
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
    resume_download = kwargs.pop("resume_download", False)
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
                resolved_archive_file = cached_file(pretrained_model_name_or_path, filename, **cached_file_kwargs)

                # Since we set _raise_exceptions_for_missing_entries=False, we don't get an exception but a None
                # result when internet is up, the repo and revision exist, but the file does not.
                if resolved_archive_file is None and filename == _add_variant(SAFE_WEIGHTS_NAME, variant):
                    # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                    resolved_archive_file = cached_file(
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
                        resolved_archive_file = cached_file(
                            pretrained_model_name_or_path, filename, **cached_file_kwargs
                        )
                if resolved_archive_file is None and filename == _add_variant(WEIGHTS_NAME, variant):
                    # Maybe the checkpoint is sharded, we try to grab the index name in this case.
                    resolved_archive_file = cached_file(
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
