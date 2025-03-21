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
from pathlib import Path
from typing import Callable, Dict, Optional, Type, Union

import torch
from safetensors import safe_open
from transformers import PretrainedConfig
from transformers.modeling_utils import (
    SpecificPreTrainedModelType,
    no_init_weights,
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

from ...distributed.utils import (
    create_kv_proj_local_weight_from_regular_weight,
    create_query_or_output_projection_local_weight_from_regular_weight,
)
from ...utils.import_utils import is_neuronx_distributed_available, is_torch_xla_available
from ...utils.misc import download_checkpoints_in_cache


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

if is_neuronx_distributed_available():
    from neuronx_distributed.kernels.flash_attn import nki_flash_attn_func
    from neuronx_distributed.parallel_layers.layers import create_local_weight
    from neuronx_distributed.parallel_layers.parallel_state import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_size,
    )
    from neuronx_distributed.parallel_layers.utils import get_local_world_size, move_model_to_device


logger = logging.get_logger(__name__)

ALL_ATTENTION_FUNCTIONS: Dict[str, Dict[str, Callable]] = {
    "flash_attention_2": nki_flash_attn_func,
}

def create_local_fused_weight(tp_rank, tp_size, individual_weights, partition_dim, out_weight=None):
    weight_lists = []
    for weight in individual_weights:
        weight_list = torch.split(weight, weight.size(partition_dim) //  tp_size, dim=partition_dim)[tp_rank::tp_size]
        weight_lists.append(weight_list)

    with torch.no_grad():
        return torch.cat(
            [torch.cat(weight_list, dim=partition_dim) for weight_list in weight_lists],
            dim=partition_dim,
            out=out_weight,
        )



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
                    message += ', `"attn_implementation=flash_attention_2"` (implementation using nki flash attention 2)'
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
        kwargs.pop("output_loading_info", False)
        use_auth_token = kwargs.pop("use_auth_token", None)
        trust_remote_code = kwargs.pop("trust_remote_code", None)
        _ = kwargs.pop("mirror", None)
        from_pipeline = kwargs.pop("_from_pipeline", None)
        from_auto_class = kwargs.pop("_from_auto", False)
        _fast_init = kwargs.pop("_fast_init", True)
        torch_dtype = kwargs.pop("torch_dtype", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", None)
        # The default is device_map = None in transformers.
        # Here we want to move the model to the device by default to free up the RAM.
        # device_map = kwargs.pop("device_map", "xla")
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
            # dtype_orig = cls._set_default_torch_dtype(torch_dtype)
            config.torch_dtype = torch_dtype

        config.name_or_path = pretrained_model_name_or_path

        # Instantiate model.
        init_contexts = [no_init_weights(_enable=_fast_init)]

        config = copy.deepcopy(config)  # We do not want to modify the config inplace in from_pretrained.
        if not getattr(config, "_attn_implementation_autoset", False):
            # We do not check for the device_map because we are going to move the model to XLA anyway on our own.
            config = cls._autoset_attn_implementation(
                config, use_flash_attention_2=use_flash_attention_2, torch_dtype=torch_dtype, device_map=device_map, check_device_map=False,
            )

        with ContextManagers(init_contexts):
            # Let's make sure we don't run the init function of buffer modules
            model = cls(config, *model_args, **model_kwargs)

        # make sure we use the model's config since the __init__ call might have copied it
        config = model.config

        state_dict  = {}

        mp_config = model.mp_config
        num_local_ranks_per_step = mp_config.num_local_ranks_per_step
        local_world_size = get_local_world_size()
        local_rank = xm.get_local_ordinal()
        tp_size = get_tensor_model_parallel_size()
        tp_rank = get_tensor_model_parallel_rank()
        if num_local_ranks_per_step <= 0:
            num_local_ranks_per_step = local_world_size

        for worker in range(math.ceil(local_world_size / num_local_ranks_per_step)):
            if local_rank // num_local_ranks_per_step == worker:
                if sharded_metadata:
                    weight_map = sharded_metadata["weight_map"]
                else:
                    filename = Path(filenames)
                    # TODO: manage the safetensor check dependency.
                    with safe_open(filename, framework="pt", device="cpu") as fp:
                        weight_map = {weight_name: filename for weight_name in fp.keys()}

                state_dict = {}
                for weight_name, filename in weight_map.items():
                    with safe_open(filename, framework="pt", device="cpu") as fp:
                        state_dict[weight_name] = fp.get_tensor(weight_name)


                model2state_dict = {}
                for name, module in model.named_modules():
                    if hasattr(module, "fused_linears"):
                        for fused_linear_specs, linears_specs in module.fused_linears.items():
                            if isinstance(fused_linear_specs, tuple):
                                fused_linear_name = fused_linear_specs[0]
                                param_names = [fused_linear_specs[1]]
                            else:
                                fused_linear_name = fused_linear_specs
                                fused_linear = module.get_submodule(fused_linear_name)
                                param_names = ["weight", "bias"] if fused_linear.bias is not None else ["weight"]

                            for param_name in param_names:
                                linear_param_names = []
                                linear_names = []
                                for linear_specs in linears_specs:
                                    if isinstance(linear_specs, tuple):
                                        linear_names.append(linear_specs[0])
                                        linear_param_names.append(linear_specs[1])
                                    else:
                                        linear_names.append(linear_specs)
                                        linear_param_names.append(param_name)
                                fused_linear_full_name = f"{name}.{fused_linear_name}.{param_name}"
                                linear_full_names = [f"{name}.{linear_name}.{linear_param_name}" for linear_name, linear_param_name in zip(linear_names, linear_param_names)]
                                model2state_dict[fused_linear_full_name] = ("fused_linears", linear_full_names)
                    if hasattr(module, "gqa_qkv_specs"):
                        specs = module.gqa_qkv_specs
                        fuse_qkv = specs["fuse_qkv"]
                        bias = specs["bias"]

                        if fuse_qkv:
                            param_names = ["weight", "bias"] if bias else ["weight"]
                            for param_name in param_names:
                                fused_linear_full_name = f"{name}.{specs['gqa_qkv_name_projection']}.{param_name}_qkv"
                                linear_full_names = [
                                    f"{name}.{specs['query_projection']}.{param_name}",
                                    f"{name}.{specs['key_projection']}.{param_name}",
                                    f"{name}.{specs['value_projection']}.{param_name}",
                                ]
                                model2state_dict[fused_linear_full_name] = ("gqa_qkv", linear_full_names, specs)
                        else:
                            gqa_qkv_projection_name = f"{name}.{specs['gqa_qkv_projection']}"
                            model2state_dict[f"{gqa_qkv_projection_name}.weight_q"] = ("gqa_qkv", f"{name}.{specs['query_projection']}.weight", specs)
                            model2state_dict[f"{gqa_qkv_projection_name}.weight_k"] = ("gqa_qkv", f"{name}.{specs['key_projection']}.weight", specs)
                            model2state_dict[f"{gqa_qkv_projection_name}.weight_v"] = ("gqa_qkv", f"{name}.{specs['value_projection']}.weight", specs)

                        # Handling output projection.
                        output_projection_name = f"{name}.{specs['output_projection']}"
                        model2state_dict[f"{output_projection_name}.weight"] = ("gqa_qkv_output_projection", f"{output_projection_name}.weight", specs)
                        if module.get_submodule(specs["output_projection"]).bias is not None:
                            model2state_dict[f"{output_projection_name}.bias"] = ("gqa_qkv_output_projection", f"{output_projection_name}.bias", specs)

                for name, param in model.named_parameters():
                    if hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel:
                        if param.partition_dim not in [0, 1]:
                            raise Exception(f"Partiton value of 0,1 are supported, found {param.partition_dim}.")
                        case, *information = model2state_dict.get(name, ("local", name))
                        if case == "local":
                            full_weight = state_dict[name]
                            per_partition_size = full_weight.shape[param.partition_dim] // tp_size
                            print(name, param.partition_dim, per_partition_size, param.partition_stride)
                            state_dict[name] = create_local_weight(
                                full_weight, param.partition_dim, per_partition_size, param.partition_stride
                            )
                        elif case == "fused_linears":
                            state_dict_names = information[0]
                            full_weights = [state_dict.pop(key) for key in state_dict_names]
                            per_partition_size = full_weights[0].shape[param.partition_dim] // tp_size
                            state_dict[name] = create_local_fused_weight(tp_rank, tp_size, full_weights, param.partition_dim)
                        elif case == "gqa_qkv":
                            specs = information[1]
                            if specs["fuse_qkv"]:
                                linear_full_names = information[0]
                                q_name, k_name, v_name = linear_full_names
                                full_weights = [
                                    create_query_or_output_projection_local_weight_from_regular_weight(
                                        state_dict.pop(q_name),
                                        specs["num_attention_heads"],
                                        specs["num_key_value_heads"],
                                        specs["kv_size_multiplier"],
                                        "query",
                                    ),
                                    create_kv_proj_local_weight_from_regular_weight(
                                        state_dict.pop(k_name),
                                        specs["kv_size_multiplier"],
                                        specs["kv_output_size_per_partition"]
                                    ),
                                    create_kv_proj_local_weight_from_regular_weight(
                                        state_dict.pop(v_name),
                                        specs["kv_size_multiplier"],
                                        specs["kv_output_size_per_partition"]
                                    ),
                                ]
                                state_dict[name] = torch.cat(full_weights, dim=0)
                            else:
                                if "weight_q" in name:
                                    state_dict[name] = create_query_or_output_projection_local_weight_from_regular_weight(
                                        state_dict.pop(information[0]),
                                        specs["num_attention_heads"],
                                        specs["num_key_value_heads"],
                                        specs["kv_size_multiplier"],
                                        "query",
                                    )
                                else:
                                    state_dict[name] = create_kv_proj_local_weight_from_regular_weight(
                                        state_dict.pop(information[0]),
                                        specs["kv_size_multiplier"],
                                        specs["kv_output_size_per_partition"]
                                    )
                        elif case == "gqa_qkv_output_projection":
                            specs = information[1]
                            state_dict[name] = create_query_or_output_projection_local_weight_from_regular_weight(
                                state_dict.pop(information[0]),
                                specs["num_attention_heads"],
                                specs["num_key_value_heads"],
                                specs["kv_size_multiplier"],
                                "output",
                            )

                model.load_state_dict(state_dict, strict=True)
                if torch_dtype is not None:
                    model = model.to(torch_dtype)
                if device_map == "xla":
                    move_model_to_device(model, xm.xla_device())
                gc.collect()
                model.tie_weights()

            xm.rendezvous(f"load_state_dict_{worker}")

        return model
