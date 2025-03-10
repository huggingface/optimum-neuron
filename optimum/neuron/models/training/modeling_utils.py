# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import os
import json
import warnings
import copy
import functools
from threading import Thread
from typing import Union, Optional, Type

import torch

from safetensors import safe_open
from safetensors.torch import load_file as safe_load_file
from safetensors.torch import save_file as safe_save_file

from transformers import PreTrainedModel, PretrainedConfig
from transformers.modeling_utils import SpecificPreTrainedModelType, no_init_weights, load_state_dict, get_state_dict_dtype, _add_variant
from transformers.utils.hub import get_checkpoint_shard_files
from transformers.safetensors_conversion import auto_conversion
from transformers.utils import (
    ContextManagers,
    find_adapter_config_file,
    logging,
    ACCELERATE_MIN_VERSION,
    ADAPTER_SAFE_WEIGHTS_NAME,
    ADAPTER_WEIGHTS_NAME,
    CONFIG_NAME,
    DUMMY_INPUTS,
    FLAX_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    ContextManagers,
    ModelOutput,
    PushToHubMixin,
    cached_file,
    copy_func,
    download_url,
    extract_commit_hash,
    has_file,
    is_accelerate_available,
    is_bitsandbytes_available,
    is_flash_attn_2_available,
    is_offline_mode,
    is_optimum_available,
    is_peft_available,
    is_remote_url,
    is_safetensors_available,
    is_torch_flex_attn_available,
    is_torch_greater_or_equal,
    is_torch_sdpa_available,
    is_torch_xla_available,
    logging,
    replace_return_docstrings,
    strtobool,
)

from ...accelerate import ModelParallelismConfig
from ...utils.patching import Patcher




logger = logging.get_logger(__name__)

def parallel_load_state_dict(
    mp_config: ModelParallelismConfig,
    checkpoint_file: Union[str, os.PathLike],
    is_quantized: bool = False,
    map_location: Optional[Union[str, torch.device]] = None,
    weights_only: bool = True,
):
    # Steps:
    # 1. If the checkpoint_file is a torch checkpoint, convert it to safetensors
    # 2. Loop over the tp ranks in groups of mp.config.num_local_ranks_per_steps
    # 3. For each step, loop over the parameter names to load
    # 4. Shard them according to the tp_rank
    pass


class NeuronModelMixin:

    @classmethod
    def from_pretrained(
        cls: Type[SpecificPreTrainedModelType],
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
        cache_dir: Optional[Union[str, os.PathLike]] = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: Optional[Union[str, bool]] = None,
        revision: str = "main",
        use_safetensors: Optional[bool] = None,
        weights_only: bool = True,
        **kwargs,
    ) -> SpecificPreTrainedModelType:
        state_dict = kwargs.pop("state_dict", None)
        from_tf = kwargs.pop("from_tf", False)
        from_flax = kwargs.pop("from_flax", False)
        resume_download = kwargs.pop("resume_download", None)
        proxies = kwargs.pop("proxies", None)
        output_loading_info = kwargs.pop("output_loading_info", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        _ = kwargs.pop("mirror", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        _fast_init = kwargs.pop("_fast_init", True)
        torch_dtype = kwargs.pop("torch_dtype", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", None)
        device_map = kwargs.pop("device_map", None)
        max_memory = kwargs.pop("max_memory", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_state_dict = kwargs.pop("offload_state_dict", False)
        offload_buffers = kwargs.pop("offload_buffers", False)
        load_in_8bit = kwargs.pop("load_in_8bit", False)
        load_in_4bit = kwargs.pop("load_in_4bit", False)
        quantization_config = kwargs.pop("quantization_config", None)
        subfolder = kwargs.pop("subfolder", "")
        commit_hash = kwargs.pop("_commit_hash", None)
        variant = kwargs.pop("variant", None)
        adapter_kwargs = kwargs.pop("adapter_kwargs", {})
        adapter_name = kwargs.pop("adapter_name", "default")
        use_flash_attention_2 = kwargs.pop("use_flash_attention_2", False)
        generation_config = kwargs.pop("generation_config", None)

        gguf_file = kwargs.pop("gguf_file", None)
        # Cache path to the GGUF file
        gguf_path = None

        if use_auth_token is not None:
            warnings.warn(
                "The `use_auth_token` argument is deprecated and will be removed in v5 of Transformers. Please use `token` instead.",
                FutureWarning,
            )
            if token is not None:
                raise ValueError(
                    "`token` and `use_auth_token` are both specified. Please set only the argument `token`."
                )
            token = use_auth_token

        if token is not None and adapter_kwargs is not None and "token" not in adapter_kwargs:
            adapter_kwargs["token"] = token

        if use_safetensors is None and not is_safetensors_available():
            use_safetensors = False
        if trust_remote_code is True:
            logger.warning(
                "The argument `trust_remote_code` is to be used with Auto classes. It has no effect here and is"
                " ignored."
            )

        if gguf_file is not None and not is_accelerate_available():
            raise ValueError("accelerate is required when loading a GGUF file `pip install accelerate`.")

        if commit_hash is None:
            if not isinstance(config, PretrainedConfig):
                # We make a call to the config file first (which may be absent) to get the commit hash as soon as possible
                resolved_config_file = cached_file(
                    pretrained_model_name_or_path,
                    CONFIG_NAME,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=subfolder,
                    _raise_exceptions_for_gated_repo=False,
                    _raise_exceptions_for_missing_entries=False,
                    _raise_exceptions_for_connection_errors=False,
                )
                commit_hash = extract_commit_hash(resolved_config_file, commit_hash)
            else:
                commit_hash = getattr(config, "_commit_hash", None)

        if is_peft_available():
            _adapter_model_path = adapter_kwargs.pop("_adapter_model_path", None)

            if _adapter_model_path is None:
                _adapter_model_path = find_adapter_config_file(
                    pretrained_model_name_or_path,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    resume_download=resume_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    _commit_hash=commit_hash,
                    **adapter_kwargs,
                )
            if _adapter_model_path is not None and os.path.isfile(_adapter_model_path):
                with open(_adapter_model_path, "r", encoding="utf-8") as f:
                    _adapter_model_path = pretrained_model_name_or_path
                    pretrained_model_name_or_path = json.load(f)["base_model_name_or_path"]
        else:
            _adapter_model_path = None

        if device_map is not None:
            raise RuntimeError("Device map is not supported yet.")

        if low_cpu_mem_usage is not None:
            raise RuntimeError("Low cpu memory usage is not supported for optimum-neuron.")

        if load_in_4bit or load_in_8bit:
            raise RuntimeError("Quantization is not supported yet.")

        from_pt = not (from_tf | from_flax)

        user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config_class.from_pretrained(
                config_path,
                cache_dir=cache_dir,
                return_unused_kwargs=True,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                subfolder=subfolder,
                _from_auto=from_auto_class,
                _from_pipeline=from_pipeline,
                **kwargs,
            )
        else:
            # In case one passes a config to `from_pretrained` + "attn_implementation"
            # override the `_attn_implementation` attribute to `attn_implementation` of the kwargs
            # Please see: https://github.com/huggingface/transformers/issues/28038

            # Overwrite `config._attn_implementation` by the one from the kwargs --> in auto-factory
            # we pop attn_implementation from the kwargs but this handles the case where users
            # passes manually the config to `from_pretrained`.
            config = copy.deepcopy(config)

            kwarg_attn_imp = kwargs.pop("attn_implementation", None)
            if kwarg_attn_imp is not None:
                config._attn_implementation = kwarg_attn_imp

            model_kwargs = kwargs

        # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
        # index of the files.
        is_sharded = False
        sharded_metadata = None
        # Load model
        loading_info = None

        # Keep in fp32 modules
        keep_in_fp32_modules = None
        use_keep_in_fp32_modules = False

        if gguf_file is not None and hf_quantizer is not None:
            raise ValueError(
                "You cannot combine Quantization and loading a model from a GGUF file, try again by making sure you did not passed a `quantization_config` or that you did not load a quantized model from the Hub."
            )

        if pretrained_model_name_or_path is not None and gguf_file is None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            is_local = os.path.isdir(pretrained_model_name_or_path)
            if is_local:
                if from_tf and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + ".index")
                ):
                    # Load from a TF 1.0 checkpoint in priority if from_tf
                    archive_file = os.path.join(pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + ".index")
                elif from_tf and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME)
                ):
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
                    os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)
                    )
                ):
                    # Load from a sharded safetensors checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)
                    )
                    is_sharded = True
                elif not use_safetensors and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_NAME, variant))
                ):
                    # Load from a PyTorch checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_NAME, variant)
                    )
                elif not use_safetensors and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant))
                ):
                    # Load from a sharded PyTorch checkpoint
                    archive_file = os.path.join(
                        pretrained_model_name_or_path, subfolder, _add_variant(WEIGHTS_INDEX_NAME, variant)
                    )
                    is_sharded = True
                # At this stage we don't have a weight file so we will raise an error.
                elif not use_safetensors and (
                    os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, TF_WEIGHTS_NAME + ".index"))
                    or os.path.isfile(os.path.join(pretrained_model_name_or_path, subfolder, TF2_WEIGHTS_NAME))
                ):
                    raise EnvironmentError(
                        f"Error no file named {_add_variant(WEIGHTS_NAME, variant)} found in directory"
                        f" {pretrained_model_name_or_path} but there is a file for TensorFlow weights. Use"
                        " `from_tf=True` to load this model from those weights."
                    )
                elif not use_safetensors and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)
                ):
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
                        f"Error no file named {_add_variant(WEIGHTS_NAME, variant)}, {_add_variant(SAFE_WEIGHTS_NAME, variant)},"
                        f" {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME + '.index'} or {FLAX_WEIGHTS_NAME} found in directory"
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
                        "token": token,
                        "user_agent": user_agent,
                        "revision": revision,
                        "subfolder": subfolder,
                        "_raise_exceptions_for_gated_repo": False,
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
                            if revision == "main":
                                resolved_archive_file, revision, is_sharded = auto_conversion(
                                    pretrained_model_name_or_path, **cached_file_kwargs
                                )
                            cached_file_kwargs["revision"] = revision
                            if resolved_archive_file is None:
                                raise EnvironmentError(
                                    f"{pretrained_model_name_or_path} does not appear to have a file named"
                                    f" {_add_variant(SAFE_WEIGHTS_NAME, variant)} or {_add_variant(SAFE_WEIGHTS_INDEX_NAME, variant)} "
                                    "and thus cannot be loaded with `safetensors`. Please make sure that the model has "
                                    "been saved with `safe_serialization=True` or do not set `use_safetensors=True`."
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
                    if not local_files_only and not is_offline_mode():
                        if resolved_archive_file is not None:
                            if filename in [WEIGHTS_NAME, WEIGHTS_INDEX_NAME]:
                                # If the PyTorch file was found, check if there is a safetensors file on the repository
                                # If there is no safetensors file on the repositories, start an auto conversion
                                safe_weights_name = SAFE_WEIGHTS_INDEX_NAME if is_sharded else SAFE_WEIGHTS_NAME
                                has_file_kwargs = {
                                    "revision": revision,
                                    "proxies": proxies,
                                    "token": token,
                                    "cache_dir": cache_dir,
                                    "local_files_only": local_files_only,
                                }
                                cached_file_kwargs = {
                                    "cache_dir": cache_dir,
                                    "force_download": force_download,
                                    "resume_download": resume_download,
                                    "local_files_only": local_files_only,
                                    "user_agent": user_agent,
                                    "subfolder": subfolder,
                                    "_raise_exceptions_for_gated_repo": False,
                                    "_raise_exceptions_for_missing_entries": False,
                                    "_commit_hash": commit_hash,
                                    **has_file_kwargs,
                                }
                                if not has_file(pretrained_model_name_or_path, safe_weights_name, **has_file_kwargs):
                                    Thread(
                                        target=auto_conversion,
                                        args=(pretrained_model_name_or_path,),
                                        kwargs={"ignore_errors_during_conversion": True, **cached_file_kwargs},
                                        name="Thread-auto_conversion",
                                    ).start()
                        else:
                            # Otherwise, no PyTorch file was found, maybe there is a TF or Flax model file.
                            # We try those to give a helpful error message.
                            has_file_kwargs = {
                                "revision": revision,
                                "proxies": proxies,
                                "token": token,
                                "cache_dir": cache_dir,
                                "local_files_only": local_files_only,
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
                                    f" {_add_variant(WEIGHTS_NAME, variant)}, {_add_variant(SAFE_WEIGHTS_NAME, variant)},"
                                    f" {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or {FLAX_WEIGHTS_NAME}."
                                )

                except EnvironmentError:
                    # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted
                    # to the original exception.
                    raise
                except Exception as e:
                    # For any other exception, we throw a generic error.
                    raise EnvironmentError(
                        f"Can't load the model for '{pretrained_model_name_or_path}'. If you were trying to load it"
                        " from 'https://huggingface.co/models', make sure you don't have a local directory with the"
                        f" same name. Otherwise, make sure '{pretrained_model_name_or_path}' is the correct path to a"
                        f" directory containing a file named {_add_variant(WEIGHTS_NAME, variant)},"
                        f" {TF2_WEIGHTS_NAME}, {TF_WEIGHTS_NAME} or {FLAX_WEIGHTS_NAME}."
                    ) from e

            if is_local:
                logger.info(f"loading weights file {archive_file}")
                resolved_archive_file = archive_file
            else:
                logger.info(f"loading weights file {filename} from cache at {resolved_archive_file}")
        elif gguf_file:
            from .modeling_gguf_pytorch_utils import load_gguf_checkpoint

            # Case 1: the GGUF file is present locally
            if os.path.isfile(gguf_file):
                gguf_path = gguf_file
            # Case 2: The GGUF path is a location on the Hub
            # Load from URL or cache if already cached
            else:
                cached_file_kwargs = {
                    "cache_dir": cache_dir,
                    "force_download": force_download,
                    "proxies": proxies,
                    "resume_download": resume_download,
                    "local_files_only": local_files_only,
                    "token": token,
                    "user_agent": user_agent,
                    "revision": revision,
                    "subfolder": subfolder,
                    "_raise_exceptions_for_gated_repo": False,
                    "_raise_exceptions_for_missing_entries": False,
                    "_commit_hash": commit_hash,
                }

                gguf_path = cached_file(pretrained_model_name_or_path, gguf_file, **cached_file_kwargs)

            # we need a dummy model to help rename state_dict
            with torch.device("meta"):
                dummy_model = cls(config)
            state_dict = load_gguf_checkpoint(gguf_path, return_tensors=True, model_to_load=dummy_model)["tensors"]

            resolved_archive_file = None
            is_sharded = False
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
                token=token,
                user_agent=user_agent,
                revision=revision,
                subfolder=subfolder,
                _commit_hash=commit_hash,
            )

        if (
            is_safetensors_available()
            and isinstance(resolved_archive_file, str)
            and resolved_archive_file.endswith(".safetensors")
        ):
            with safe_open(resolved_archive_file, framework="pt") as f:
                metadata = f.metadata()

            if metadata is None:
                # Assume it's a pytorch checkpoint (introduced for timm checkpoints)
                pass
            elif metadata.get("format") == "pt":
                pass
            elif metadata.get("format") == "tf":
                from_tf = True
                logger.info("A TensorFlow safetensors file is being loaded in a PyTorch model.")
            elif metadata.get("format") == "flax":
                from_flax = True
                logger.info("A Flax safetensors file is being loaded in a PyTorch model.")
            elif metadata.get("format") == "mlx":
                # This is a mlx file, we assume weights are compatible with pt
                pass
            else:
                raise ValueError(
                    f"Incompatible safetensors file. File metadata is not ['pt', 'tf', 'flax', 'mlx'] but {metadata.get('format')}"
                )

        from_pt = not (from_tf | from_flax)

        # load pt weights early so that we know which dtype to init the model under
        if from_pt:
            if not is_sharded and state_dict is None:
                # Time to load the checkpoint
                state_dict = load_state_dict(resolved_archive_file, weights_only=weights_only)

            # set dtype to instantiate the model under:
            # 1. If torch_dtype is not None, we use that dtype
            # 2. If torch_dtype is "auto", we auto-detect dtype from the loaded state_dict, by checking its first
            #    weights entry that is of a floating type - we assume all floating dtype weights are of the same dtype
            # we also may have config.torch_dtype available, but we won't rely on it till v5
            dtype_orig = None

            if torch_dtype is not None:
                if isinstance(torch_dtype, str):
                    if torch_dtype == "auto":
                        if hasattr(config, "torch_dtype") and config.torch_dtype is not None:
                            torch_dtype = config.torch_dtype
                            logger.info(f"Will use torch_dtype={torch_dtype} as defined in model's config object")
                        else:
                            if is_sharded and "dtype" in sharded_metadata:
                                torch_dtype = sharded_metadata["dtype"]
                            elif not is_sharded:
                                torch_dtype = get_state_dict_dtype(state_dict)
                            else:
                                one_state_dict = load_state_dict(resolved_archive_file[0], weights_only=weights_only)
                                torch_dtype = get_state_dict_dtype(one_state_dict)
                                del one_state_dict  # free CPU memory
                            logger.info(
                                "Since the `torch_dtype` attribute can't be found in model's config object, "
                                "will use torch_dtype={torch_dtype} as derived from model's weights"
                            )
                    elif hasattr(torch, torch_dtype):
                        torch_dtype = getattr(torch, torch_dtype)
                    else:
                        raise ValueError(
                            f'`torch_dtype` can be one of: `torch.dtype`, `"auto"` or a string of a valid `torch.dtype`, but received {torch_dtype}'
                        )
                dtype_orig = cls._set_default_torch_dtype(torch_dtype)

            # Check if `_keep_in_fp32_modules` is not None
            use_keep_in_fp32_modules = (cls._keep_in_fp32_modules is not None) and (
                (torch_dtype == torch.float16) or hasattr(hf_quantizer, "use_keep_in_fp32_modules")
            )

            if is_sharded:
                loaded_state_dict_keys = sharded_metadata["all_checkpoint_keys"]
            else:
                loaded_state_dict_keys = list(state_dict.keys())
            if (
                gguf_path is None
                and (low_cpu_mem_usage or (use_keep_in_fp32_modules and is_accelerate_available()))
                and pretrained_model_name_or_path is not None
            ):
                # In case some weights need to be kept in float32 and accelerate is not installed,
                # we later on want to take the path where state_dict is not None, that is the one
                # that do not require accelerate.
                state_dict = None

        config.name_or_path = pretrained_model_name_or_path

        # Instantiate model.
        init_contexts = [no_init_weights(_enable=_fast_init)]
        tp_device = None

        config = copy.deepcopy(config)  # We do not want to modify the config inplace in from_pretrained.
        if not getattr(config, "_attn_implementation_autoset", False):
            config = cls._autoset_attn_implementation(
                config, use_flash_attention_2=use_flash_attention_2, torch_dtype=torch_dtype, device_map=device_map
            )

        with ContextManagers(init_contexts):
            # Let's make sure we don't run the init function of buffer modules
            model = cls(config, *model_args, **model_kwargs)

        # make sure we use the model's config since the __init__ call might have copied it
        config = model.config

        print("Loaded", model)

        return model


    # @classmethod
    # def from_pretrained(cls, *args, **kwargs):
    #     mp_config = args[1][1]
    #     partial_parallel_load_state_dict = functools.partial(parallel_load_state_dict, mp_config=mp_config)
    #     patcher = Patcher([("transformers.modeling_utils.load_state_dict", partial_parallel_load_state_dict)])
    #     with patcher:
    #         return super().from_pretrained(*args, **kwargs)

