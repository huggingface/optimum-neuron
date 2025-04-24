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

import copy
import gc
import json
import math
import os
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, Literal, Optional, Type, Union

import torch
from safetensors import safe_open
from transformers import PretrainedConfig
from transformers.modeling_utils import (
    SpecificPreTrainedModelType,
    get_parameter_dtype,
    get_state_dict_dtype,
    load_state_dict,
    no_init_weights,
    set_initialized_submodules,
)
from transformers.utils import (
    CONFIG_NAME,
    ContextManagers,
    cached_file,
    extract_commit_hash,
    find_adapter_config_file,
    is_offline_mode,
    is_peft_available,
    is_safetensors_available,
    logging,
)

from ...utils.import_utils import is_neuronx_distributed_available, is_torch_xla_available
from ...utils.misc import download_checkpoints_in_cache, is_main_worker, is_precompilation
from .transformations_utils import (
    adapt_state_dict,
    set_module_names_in_transformation_specs,
)


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    from torch_xla.utils.checkpoint import checkpoint

if is_neuronx_distributed_available():
    import neuronx_distributed
    from neuronx_distributed.kernels.flash_attn import nki_flash_attn_func
    from neuronx_distributed.parallel_layers.layers import BaseParallelLinear
    from neuronx_distributed.parallel_layers.utils import get_local_world_size, move_model_to_device
else:
    # This is a placeholder for the nki_flash_attn_func function for doc building.
    def nki_flash_attn_func(*args, **kwargs):
        pass


logger = logging.get_logger(__name__)

MODEL_PARALLEL_SHARDS_DIR_NAME = "shards"

ALL_ATTENTION_FUNCTIONS: Dict[str, Dict[str, Callable]] = {
    "flash_attention_2": nki_flash_attn_func,
}


class NeuronModelMixin:
    @classmethod
    def _check_and_enable_flash_attn_2(
        cls,
        config,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        check_device_map: bool = True,
        hard_check_only: bool = False,
    ) -> PretrainedConfig:
        """
        Checks the availability of Flash Attention 2 and compatibility with the current model.

        If all checks pass and `hard_check_only` is False, the method will set the config attribute `attn_implementation` to "flash_attention_2" so that the model can initialize the correct attention module.
        """
        if not cls._supports_flash_attn_2:
            raise ValueError(
                f"{cls.__name__} does not support Flash Attention 2.0 yet. Please request to add support where"
                f" the model is hosted, on its model hub page: https://huggingface.co/{config._name_or_path}/discussions/new"
                " or in the Transformers GitHub repo: https://github.com/huggingface/transformers/issues/new"
            )

        if torch_dtype is None:
            logger.warning_once(
                "You are attempting to use Flash Attention 2.0 without specifying a torch dtype. This might lead to unexpected behaviour"
            )
        elif torch_dtype is not None and torch_dtype not in [torch.float16, torch.bfloat16]:
            logger.warning_once(
                "Flash Attention 2.0 only supports torch.float16 and torch.bfloat16 dtypes, but"
                f" the current dype in {cls.__name__} is {torch_dtype}. You should run training or inference using Automatic Mixed-Precision via the `with torch.autocast(device_type='torch_device'):` decorator,"
                ' or load the model with the `torch_dtype` argument. Example: `model = AutoModel.from_pretrained("openai/whisper-tiny", attn_implementation="flash_attention_2", torch_dtype=torch.float16)`'
            )

        # The check `torch.empty(0).device.type != "xla"` is needed as the model may be initialized after `torch.set_default_device` has been called,
        # or the model may be initialized under the context manager `with torch.device("cuda"):`.
        if check_device_map and device_map is None and torch.empty(0).device.type != "xla":
            logger.warning_once(
                "You are attempting to use Flash Attention 2.0 with a model not initialized on XLA. Make sure to move the model to XLA"
                " after initializing it on CPU with `model.to('xla')`."
            )
        elif (
            check_device_map
            and device_map is not None
            and isinstance(device_map, dict)
            and ("cpu" in device_map.values() or "disk" in device_map.values())
        ):
            raise ValueError(
                "You are attempting to use Flash Attention 2.0 with a model dispatched on CPU or disk. This is not supported. Please make sure to "
                "initialise the model on XLA by passing a device_map that contains only GPU devices as keys."
            )
        if not hard_check_only:
            config._attn_implementation = "flash_attention_2"
        return config

    @classmethod
    def _autoset_attn_implementation(
        cls,
        config,
        use_flash_attention_2: bool = False,
        torch_dtype: Optional[torch.dtype] = None,
        device_map: Optional[Union[str, Dict[str, int]]] = None,
        check_device_map: bool = True,
    ):
        """
        Automatically checks and dispatches to a default attention implementation. In order of priority:
            1. An implementation specified in `config._attn_implementation` (due for example to the argument attn_implementation="sdpa" in from_pretrained).
            2. DEPRECATED: if use_flash_attention_2 is set to `True` and `flash_attn` is available, flash attention. (`LlamaFlashAttention` for example)
            3. SDPA implementation, if available and supported by the model type. (`LlamaSdpaAttention` for example)
            4. The default model's implementation otherwise (`LlamaAttention` for example) .
        """
        # Here we use config._attn_implementation_internal to check whether the attention implementation was explicitely set by the user.
        # The property `PretrainedConfig._attn_implementation` is never `None`, for backward compatibility (always fall back on "eager").
        # The `hasattr` here is used as some Transformers tests for some reason do not call PretrainedConfig __init__ (e.g. test_no_super_init_config_and_model)
        requested_attn_implementation = None
        if hasattr(config, "_attn_implementation_internal") and config._attn_implementation_internal is not None:
            if config._attn_implementation != "flash_attention_2" and use_flash_attention_2:
                raise ValueError(
                    f'Both attn_implementation="{config._attn_implementation}" and `use_flash_attention_2=True` were used when loading the model, which are not compatible.'
                    ' We recommend to just use `attn_implementation="flash_attention_2"` when loading the model.'
                )

            if not isinstance(config._attn_implementation, dict) and config._attn_implementation not in [
                "eager"
            ] + list(ALL_ATTENTION_FUNCTIONS.keys()):
                message = f'Specified `attn_implementation="{config._attn_implementation}"` is not supported. The only possible arguments are `attn_implementation="eager"` (manual attention implementation)'
                if cls._supports_flash_attn_2:
                    message += (
                        ', `"attn_implementation=flash_attention_2"` (implementation using nki flash attention 2)'
                    )
                # Keeping this if supported one day.
                # if cls._supports_flex_attn:
                #     message += (
                #         ', `"attn_implementation=flex_attention"` (implementation using torch\'s flex_attention)'
                #     )
                raise ValueError(message + ".")

            # If a config is passed with a preset attn_implementation, we skip the automatic dispatch and use the user-provided config, with hard checks that the requested attention implementation is available.
            requested_attn_implementation = config._attn_implementation_internal

        # Composite models consisting of several PretrainedModels have to specify attention impl as a dict
        # where keys are sub-config names. But most people will specify one `str` which means that should dispatch it
        # for all sub-models.
        # Below we check if a config is composite and manually prepare a dict of attn impl if not already passed as a dict.
        # Later each sub-module will dispatch with its own attn impl, by calling `XXXModel._from_config(config.text_config)`
        # If any of sub-modules doesn't support requested attn, an error will be raised. See https://github.com/huggingface/transformers/pull/32238
        for key in config.sub_configs.keys():
            sub_config = getattr(config, key)
            curr_attn_implementation = (
                requested_attn_implementation
                if not isinstance(requested_attn_implementation, dict)
                else requested_attn_implementation.get(key, None)
            )
            sub_config._attn_implementation_internal = curr_attn_implementation

        if use_flash_attention_2:
            logger.warning_once(
                'The model was loaded with use_flash_attention_2=True, which is deprecated and may be removed in a future release. Please use `attn_implementation="flash_attention_2"` instead.'
            )
            config._attn_implementation = "flash_attention_2"

        if config._attn_implementation == "flash_attention_2":
            cls._check_and_enable_flash_attn_2(
                config,
                torch_dtype=torch_dtype,
                device_map=device_map,
                hard_check_only=False,
                check_device_map=check_device_map,
            )
        # Keeping it if supported one day.
        # elif requested_attn_implementation == "flex_attention":
        #     config = cls._check_and_enable_flex_attn(config, hard_check_only=True)
        elif requested_attn_implementation in list(ALL_ATTENTION_FUNCTIONS.keys()):
            config._attn_implementation = requested_attn_implementation
        elif isinstance(requested_attn_implementation, dict):
            config._attn_implementation = None
        else:
            config._attn_implementation = "eager"

        config._attn_implementation_autoset = True
        return config

    # This method uses `torch.xla.utils.checkpoint.checkpoint` instead of the torch one.
    def _set_gradient_checkpointing(self, enable: bool = True, gradient_checkpointing_func: Callable = checkpoint):
        return super()._set_gradient_checkpointing(
            enable=enable, gradient_checkpointing_func=gradient_checkpointing_func
        )

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
        kwargs.pop("from_tf", False)
        kwargs.pop("from_flax", False)
        resume_download = kwargs.pop("resume_download", None)
        proxies = kwargs.pop("proxies", None)
        kwargs.pop("output_loading_info", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        _ = kwargs.pop("mirror", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        _fast_init = kwargs.pop("_fast_init", True)
        torch_dtype = kwargs.pop("torch_dtype", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", None)
        device_map = kwargs.pop("device_map", None)
        kwargs.pop("max_memory", None)
        kwargs.pop("offload_folder", None)
        kwargs.pop("offload_state_dict", False)
        kwargs.pop("offload_buffers", False)
        load_in_8bit = kwargs.pop("load_in_8bit", False)
        load_in_4bit = kwargs.pop("load_in_4bit", False)
        kwargs.pop("quantization_config", None)
        subfolder = kwargs.pop("subfolder", "")
        commit_hash = kwargs.pop("_commit_hash", None)
        kwargs.pop("variant", None)
        adapter_kwargs = kwargs.pop("adapter_kwargs", {})
        kwargs.pop("adapter_name", "default")
        use_flash_attention_2 = kwargs.pop("use_flash_attention_2", False)
        kwargs.pop("generation_config", None)

        kwargs.pop("gguf_file", None)
        # Cache path to the GGUF file

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

        if device_map not in [None, "xla", "cpu"]:
            raise RuntimeError('The only device map values supported are: `None`, "cpu" or "xla".')

        if low_cpu_mem_usage is not None:
            raise RuntimeError("Low cpu memory usage is not supported for optimum-neuron.")

        if load_in_4bit or load_in_8bit:
            raise RuntimeError("Quantization is not supported yet.")

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

        filenames, sharded_metadata = download_checkpoints_in_cache(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            force_download=force_download,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            use_safetensors=use_safetensors,
            use_safetensors_in_priority=True,
            convert_to_safetensors=True,
            **kwargs,
        )

        is_sharded = sharded_metadata is not None

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
                            one_state_dict = load_state_dict(filenames[0], weights_only=weights_only)
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
            config.torch_dtype = torch_dtype

        config.name_or_path = pretrained_model_name_or_path

        # Instantiate model.
        init_contexts = [no_init_weights(_enable=_fast_init)]

        config = copy.deepcopy(config)  # We do not want to modify the config inplace in from_pretrained.
        # This is due to a Neuron compiler bug, and it should be removed when the bug is fixed.
        should_fake_tie = config.tie_word_embeddings
        config.tie_word_embeddings = False
        if not getattr(config, "_attn_implementation_autoset", False):
            # We do not check for the device_map because we are going to move the model to XLA anyway on our own.
            config = cls._autoset_attn_implementation(
                config,
                use_flash_attention_2=use_flash_attention_2,
                torch_dtype=torch_dtype,
                device_map=device_map,
                check_device_map=False,
            )

        with ContextManagers(init_contexts):
            # Let's make sure we don't run the init function of buffer modules
            model = cls(config, *model_args, **model_kwargs)

        # make sure we use the model's config since the __init__ call might have copied it
        config = model.config

        state_dict = {}

        mp_config = model.mp_config
        num_local_ranks_per_step = mp_config.num_local_ranks_per_step
        local_world_size = get_local_world_size()
        local_rank = xm.get_local_ordinal()
        if num_local_ranks_per_step <= 0:
            num_local_ranks_per_step = local_world_size

        for worker in range(math.ceil(local_world_size / num_local_ranks_per_step)):
            model_to_load = model
            if local_rank // num_local_ranks_per_step == worker:
                if sharded_metadata:
                    weight_map = sharded_metadata["weight_map"]
                else:
                    filename = Path(filenames)
                    # TODO: manage the safetensor check dependency.
                    with safe_open(filename, framework="pt", device="cpu") as fp:
                        weight_map = dict.fromkeys(fp.keys(), filename)

                for weight_name, filename in weight_map.items():
                    with safe_open(filename, framework="pt", device="cpu") as fp:
                        state_dict[weight_name] = fp.get_tensor(weight_name)

                prefix = model.base_model_prefix
                loaded_keys = state_dict.keys()
                expected_keys = model.state_dict().keys()
                if len(prefix) > 0:
                    has_prefix_module = any(s.startswith(prefix) for s in loaded_keys)
                    expects_prefix_module = any(s.startswith(prefix) for s in expected_keys)
                else:
                    has_prefix_module = False
                    expects_prefix_module = False

                # key re-naming operations are never done on the keys
                # that are loaded, but always on the keys of the newly initialized model
                remove_prefix_from_model = not has_prefix_module and expects_prefix_module
                add_prefix_to_model = has_prefix_module and not expects_prefix_module

                if remove_prefix_from_model:
                    model_to_load = getattr(model, prefix)
                elif add_prefix_to_model:
                    state_dict = {".".join([prefix, key]): value for key, value in state_dict.items()}

                # This is required to have the specs properly defined.
                set_module_names_in_transformation_specs(model_to_load)

                # Adapts the state dict to the custom model.
                state_dict = adapt_state_dict(model_to_load, state_dict, inplace=False)
                model_to_load.load_state_dict(state_dict, strict=False)

                if torch_dtype is not None:
                    model = model.to(torch_dtype)
                if device_map == "xla":
                    move_model_to_device(model, xm.xla_device())

                gc.collect()
                model.tie_weights()

                # Now we set the modules names using the full model regardless of prefixes.
                # This is this name that will be saved and used when re-loading the model.
                set_module_names_in_transformation_specs(model)

                # It is important to initialize modules that are not in the state dict.
                if _fast_init:
                    # We call "set_initialized_submodules" twice:
                    # One with `model_to_load` to handle the base submodules from the state dict
                    # And one with `model`, which should contain anything that is not in the base model.
                    not_initialized_submodules = set_initialized_submodules(model_to_load, state_dict.keys())
                    if model is not model_to_load:
                        not_initialized_submodules.update(set_initialized_submodules(model, state_dict.keys()))
                    for name, mod in not_initialized_submodules.items():
                        if getattr(mod, "_is_hf_initialized", False):
                            # It means that it was set as initialized by the first `set_initialized_submodules`, we can
                            # skip.
                            continue
                        else:
                            logger.debug(f"Initializing {name} with default weights")
                            if isinstance(mod, BaseParallelLinear):
                                mod.initialize_weight_and_bias()
                            else:
                                model._initialize_weights(mod)

            xm.rendezvous(f"load_state_dict_{worker}")

        # Currently tie_word_embeddings leads to a compiler bug.
        # If weights are initially tied, we still copy the value but we do not tie them.
        if should_fake_tie:
            with torch.no_grad():
                model.get_output_embeddings().weight.data.copy_(model.get_input_embeddings().weight)

        xm.mark_step()

        return model

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        is_main_process: Union[bool, Literal["auto"]] = "auto",
        state_dict: Optional[dict] = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: Union[int, str] = "5GB",
        safe_serialization: bool = False,
        variant: Optional[str] = None,
        token: Optional[Union[str, bool]] = None,
        save_peft_format: bool = True,
        optimizer: Optional[torch.optim.Optimizer] = None,
        **kwargs,
    ):
        if is_precompilation():
            return
        if is_main_process == "auto":
            is_main_process = is_main_worker()

        use_auth_token = kwargs.pop("use_auth_token", None)
        kwargs.pop("ignore_metadata_errors", False)

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

        if token is not None:
            kwargs["token"] = token

        _hf_peft_config_loaded = getattr(self, "_hf_peft_config_loaded", False)

        if "save_config" in kwargs:
            warnings.warn(
                "`save_config` is deprecated and will be removed in v5 of Transformers. Use `is_main_process` instead."
            )
            is_main_process = kwargs.pop("save_config")
        if safe_serialization:
            raise logger.error(
                "`safe_serialization` is not supported when saving the sharded checkpoints. It is possible to consolidate the model weights into `safetensors` format."
            )

        if os.path.isfile(save_directory):
            logger.error(f"Provided path ({save_directory}) should be a directory, not a file")
            return

        save_directory = Path(save_directory)

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            raise RuntimeError(
                "`push_to_hub` is not supported because checkpoints are sharded. Consolidate them then push to hub."
            )

        model_to_save = self

        # save the string version of dtype to the config, e.g. convert torch.float32 => "float32"
        # we currently don't use this setting automatically, but may start to use with v5
        dtype = get_parameter_dtype(model_to_save)
        model_to_save.config.torch_dtype = str(dtype).split(".")[1]

        # Attach architecture to the config
        model_to_save.config.architectures = [model_to_save.__class__.__name__]

        # Unset attn implementation so it can be set to another one when loading back
        model_to_save.config._attn_implementation_autoset = False

        # Save the config
        if is_main_process:
            if not _hf_peft_config_loaded:
                # If the model config has set attributes that should be in the generation config, move them there.
                misplaced_generation_parameters = model_to_save.config._get_non_default_generation_parameters()
                if self.can_generate() and len(misplaced_generation_parameters) > 0:
                    warnings.warn(
                        "Moving the following attributes in the config to the generation config: "
                        f"{misplaced_generation_parameters}. You are seeing this warning because you've set "
                        "generation parameters in the model config, as opposed to in the generation config.",
                        UserWarning,
                    )
                    for param_name, param_value in misplaced_generation_parameters.items():
                        setattr(model_to_save.generation_config, param_name, param_value)
                        setattr(model_to_save.config, param_name, None)

                model_to_save.config.save_pretrained(save_directory)
            if self.can_generate():
                model_to_save.generation_config.save_pretrained(save_directory)

            if _hf_peft_config_loaded:
                logger.info(
                    "Detected adapters on the model, saving the model in the PEFT format, only adapter weights will be saved."
                )
                state_dict = model_to_save.get_adapter_state_dict()

                if save_peft_format:
                    logger.info(
                        "To match the expected format of the PEFT library, all keys of the state dict of adapters will be pre-pended with `base_model.model`."
                    )
                    peft_state_dict = {}
                    for key, value in state_dict.items():
                        peft_state_dict[f"base_model.model.{key}"] = value
                    state_dict = peft_state_dict

                active_adapter = self.active_adapters()

                if len(active_adapter) > 1:
                    raise ValueError(
                        "Multiple active adapters detected, saving multiple active adapters is not supported yet. You can save adapters separately one by one "
                        "by iteratively calling `model.set_adapter(adapter_name)` then `model.save_pretrained(...)`"
                    )
                active_adapter = active_adapter[0]

                current_peft_config = self.peft_config[active_adapter]
                current_peft_config.save_pretrained(save_directory)

            with open(save_directory / "mp_config.json", "w") as f:
                mp_config_data = asdict(self.mp_config)
                if isinstance(mp_config_data["checkpoint_dir"], Path):
                    mp_config_data["checkpoint_dir"] = mp_config_data["checkpoint_dir"].as_posix()
                f.write(json.dumps(mp_config_data, indent=4))

        # Saving the metadata required to consolidate the checkpoints properly.
        # if get_data_parallel_rank() == 0 and get_tensor_model_parallel_rank() == 0:
        #     metadata = create_parameter_metadata(model_to_save)
        #     pp_rank = get_pipeline_model_parallel_rank()
        #     metadata_path = save_directory / MODEL_PARALLEL_SHARDS_DIR_NAME / f"mp_metadata_pp_rank_{pp_rank}.json"
        #     metadata_path.parent.mkdir(parents=True, exist_ok=True)
        #     with open(metadata_path, "w") as f:
        #         f.write(json.dumps(metadata, indent=4))

        neuronx_distributed.trainer.save_checkpoint(
            save_directory.as_posix(),
            tag=MODEL_PARALLEL_SHARDS_DIR_NAME,
            model=self,
            optimizer=optimizer,
            use_xser=self.mp_config.use_xser,
            async_save=self.mp_config.async_save,
            num_workers=self.mp_config.num_local_ranks_per_step,
        )
