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
import json
import math
import os
import warnings
from pathlib import Path
from typing import Optional, Type, Union

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
    is_torch_xla_available,
    logging,
)

from ...accelerate import ModelParallelismConfig
from ...utils.import_utils import is_neuronx_distributed_available, is_torch_xla_available
from ...utils.misc import download_checkpoints_in_cache


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

if is_neuronx_distributed_available():
    from neuronx_distributed.parallel_layers.layers import create_local_weight
    from neuronx_distributed.parallel_layers.parallel_state import (
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_size,
    )
    from neuronx_distributed.parallel_layers.utils import get_local_world_size




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

        config.name_or_path = pretrained_model_name_or_path

        # Instantiate model.
        init_contexts = [no_init_weights(_enable=_fast_init)]

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
                print(f"Loading for worker {worker}")
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
                        for fused_linear_name, linear_names in module.fused_linears.items():
                            fused_linear = module.get_submodule(fused_linear_name)
                            param_names = ["weight", "bias"] if fused_linear.bias is not None else ["weight"]
                            for param_name in param_names:
                                fused_linear_full_name = f"{name}.{fused_linear_name}.{param_name}"
                                linear_full_names = [f"{name}.{linear_name}.{param_name}" for linear_name in linear_names]
                                model2state_dict[fused_linear_full_name]  = linear_full_names

                for name, param in model.state_dict(keep_vars=True).items():
                    if hasattr(param, "tensor_model_parallel") and param.tensor_model_parallel:
                        if param.partition_dim not in [0, 1]:
                            raise Exception(f"Partiton value of 0,1 are supported, found {param.partition_dim}.")
                        state_dict_names = model2state_dict.get(name, [name])
                        full_weights = [state_dict[key] for key in state_dict_names]
                        per_partition_size = full_weights[0].shape[param.partition_dim] // tp_size
                        if len(full_weights) == 1:
                            state_dict[name] = create_local_weight(
                                full_weights[0], param.partition_dim, per_partition_size, param.partition_stride
                            )
                        else:
                            state_dict[name] = create_local_fused_weight(tp_rank, tp_size, full_weights, param.partition_dim)

                model.load_state_dict(state_dict, strict=False)
            xm.rendezvous(f"load_state_dict_{worker}")

        return model

