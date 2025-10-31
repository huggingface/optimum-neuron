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

import collections
import copy
import gc
import json
import math
import os
import re
import warnings
from dataclasses import asdict
from pathlib import Path
from threading import Thread
from typing import Callable, Literal, Type

import neuronx_distributed
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
import transformers
from accelerate.utils import find_tied_parameters
from neuronx_distributed.kernels.flash_attn import nki_flash_attn_func
from neuronx_distributed.modules.qkv_linear import GQAQKVColumnParallelLinear
from neuronx_distributed.parallel_layers.layers import (
    BaseParallelLinear,
    ColumnParallelLinear,
    ParallelEmbedding,
)
from neuronx_distributed.parallel_layers.parallel_state import (
    get_data_parallel_rank,
    get_pipeline_model_parallel_rank,
    get_pipeline_model_parallel_size,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_size,
)
from neuronx_distributed.parallel_layers.utils import (
    get_local_world_size,
    move_model_to_device,
)
from safetensors import safe_open
from torch import nn
from torch_xla.utils.checkpoint import checkpoint
from transformers import PretrainedConfig
from transformers.modeling_utils import (
    SpecificPreTrainedModelType,
    _add_variant,
    get_parameter_dtype,
    get_state_dict_dtype,
    load_state_dict,
    no_init_weights,
)
from transformers.pytorch_utils import id_tensor_storage
from transformers.quantizers import AutoHfQuantizer
from transformers.safetensors_conversion import auto_conversion
from transformers.utils import (
    CONFIG_NAME,
    FLAX_WEIGHTS_NAME,
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    TF2_WEIGHTS_NAME,
    TF_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
    ContextManagers,
    cached_file,
    download_url,
    extract_commit_hash,
    find_adapter_config_file,
    has_file,
    is_offline_mode,
    is_peft_available,
    is_remote_url,
    is_safetensors_available,
    logging,
)
from transformers.utils.hub import get_checkpoint_shard_files

from ...utils.misc import is_main_worker, is_precompilation
from .config import TrainingNeuronConfig
from .pipeline_utils import (
    MetaParametersOnly,
    get_pipeline_parameters_for_current_stage,
    move_params_to_cpu,
)
from .transformations_utils import (
    adapt_state_dict,
    create_parameter_metadata,
    get_tensor_model_parallel_attributes,
    specialize_transformation_specs_for_model,
)


logger = logging.get_logger(__name__)

MODEL_PARALLEL_SHARDS_DIR_NAME = "shards"

ALL_ATTENTION_FUNCTIONS: dict[str, dict[str, Callable]] = {
    "flash_attention_2": nki_flash_attn_func,
}


def _wait_previous_async_save(checkpoint_dir: Path):
    # Async save: wait for the previous async save to finish before starting a new one.
    # There is this mechanism in neuronx_distributed already, but it breaks when we change the name of the
    # directory between saves, which we done (checkpoint-10, checkpoint-20, etc).
    # Reference: https://github.com/aws-neuron/neuronx-distributed/blob/main/src/neuronx_distributed/trainer/checkpoint.py#L136-L139
    # So we add this extra wait here that uses the same mechanism but prior to calling the save function.
    g_iostate = neuronx_distributed.trainer.checkpoint.g_iostate
    if g_iostate is not None:
        g_iostate.wait_save(async_remove=True)
        if xr.global_ordinal() == 0:
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            (checkpoint_dir / "done").touch(exist_ok=True)


def _load_state_dict_into_model(model_to_load, state_dict, start_prefix):
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, "_metadata", None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    error_msgs = []

    # PyTorch's `_load_from_state_dict` does not copy parameters in a module's descendants
    # so we need to apply the function recursively.
    def load(module: nn.Module, state_dict, prefix=""):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})

        args = (state_dict, prefix, local_metadata, True, [], [], error_msgs)
        # Parameters of module and children will start with prefix. We can exit early if there are none in this
        # state_dict
        if len([key for key in state_dict if key.startswith(prefix)]) > 0:
            # ** Difference from original _load_state_dict_into_model **
            # We do not add the code related to `deepspeed` here, since we do not support it.

            # ** Difference from original _load_state_dict_into_model **
            # module._load_from_state_dict can mutate the parameters in the module, we must cache the tensor parallel
            # metadata.
            tensor_model_parallel_attributes = {
                k: get_tensor_model_parallel_attributes(v) for k, v in module._parameters.items()
            }

            module._load_from_state_dict(*args)

            # Restoring the tensor model parallel attributes.
            for name, param in module._parameters.items():
                attributes = tensor_model_parallel_attributes[name]
                for attr_name, attr in attributes.items():
                    setattr(param, attr_name, attr)

        for name, child in module._modules.items():
            if child is not None:
                load(child, state_dict, prefix + name + ".")

    load(model_to_load, state_dict, prefix=start_prefix)
    # Delete `state_dict` so it could be collected by GC earlier. Note that `state_dict` is a copy of the argument, so
    # it's safe to delete it.
    del state_dict

    return error_msgs


class NeuronModelMixin:
    SUPPORTS_PIPELINE_PARALLELISM: bool = False
    PIPELINE_TRANSFORMER_LAYER_CLS: Type | None = None
    PIPELINE_INPUT_NAMES: list[str] | None = None
    PIPELINE_LEAF_MODULE_CLASSE_NAMES: list[str] | None = None

    @classmethod
    def supports_pipeline_parallelism(cls) -> bool:
        """
        Returns whether the model supports pipeline parallelism.
        """
        if cls.SUPPORTS_PIPELINE_PARALLELISM:
            if cls.PIPELINE_TRANSFORMER_LAYER_CLS is None or cls.PIPELINE_INPUT_NAMES is None:
                raise ValueError(
                    f"{cls.__name__} supports pipeline parallelism but does not have the required attributes "
                    "`PIPELINE_TRANSFORMER_LAYER_CLS` and `PIPELINE_INPUT_NAMES` set."
                )
            return True
        return False

    @property
    def parameters_for_current_stage(self) -> set[str]:
        """
        Returns the names of the parameters that are in the current pipeline stage.
        If pipeline parallelism is not used, this returns the names of all the parameters of the model.
        """
        if getattr(self, "_parameter_names_for_current_pp_rank", None) is None:
            self._parameter_names_for_current_pp_rank = get_pipeline_parameters_for_current_stage(self)
        return self._parameter_names_for_current_pp_rank

    def _check_and_adjust_attn_implementation(
        self, attn_implementation: str | None, is_init_check: bool = False
    ) -> str:
        attn_implementation = attn_implementation or "eager"
        attn_implementation: str = self.get_correct_attn_implementation(attn_implementation, is_init_check)
        return attn_implementation

    def _flash_attn_2_can_dispatch(self, is_init_check: bool = False) -> bool:
        dtype = self.config.dtype

        if not self._supports_flash_attn:
            raise ValueError(
                f"{self.__class__.__name__} does not support Flash Attention 2.0 yet. Please request to add support here: "
                "https://github.com/huggingface/optimum-neuron/issues"
            )

        if dtype is None:
            logger.warning_once(
                "You are attempting to use Flash Attention 2 without specifying a dtype. This might lead to unexpected behaviour"
            )

        # If no error raise by this point, we can return `True`
        return True

    def _flash_attn_3_can_dispatch(self, is_init_check: bool = False) -> bool:
        raise ValueError("Flash Attention 3.0 is not supported in optimum-neuron.")

    def _sdpa_can_dispatch(self, is_init_check: bool = False) -> bool:
        raise ValueError("SDPA is not supported in optimum-neuron.")

    def _flex_attn_can_dispatch(self, is_init_check: bool = False) -> bool:
        raise ValueError("Flex Attention is not supported in optimum-neuron.")

    # This method uses `torch.xla.utils.checkpoint.checkpoint` instead of the torch one.
    def _set_gradient_checkpointing(self, enable: bool = True, gradient_checkpointing_func: Callable = checkpoint):
        return super()._set_gradient_checkpointing(
            enable=enable, gradient_checkpointing_func=gradient_checkpointing_func
        )

    @staticmethod
    def _fix_state_dict_key_on_load(key) -> tuple[str, bool]:
        # Rename LayerNorm beta & gamma params for some early models ported from Tensorflow (e.g. Bert)
        # This rename is logged.
        if key.endswith("LayerNorm.beta"):
            return key.replace("LayerNorm.beta", "LayerNorm.bias"), True
        if key.endswith("LayerNorm.gamma"):
            return key.replace("LayerNorm.gamma", "LayerNorm.weight"), True

        # Rename weight norm parametrizations to match changes across torch versions.
        # Impacts a number of speech/wav2vec models. e.g. Hubert, Wav2Vec2, and others.
        # This rename is not logged.
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            if key.endswith("weight_g"):
                return key.replace("weight_g", "parametrizations.weight.original0"), True
            # ** Difference from original _fix_state_dict_key_on_load **
            # If we do not check that `"qkv_proj"` is not in the key this method changes the name of the value weights
            # when using GQAQKVColumnParallelLinear.
            if key.endswith("weight_v") and "qkv_proj" not in key:
                return key.replace("weight_v", "parametrizations.weight.original1"), True
        else:
            if key.endswith("parametrizations.weight.original0"):
                return key.replace("parametrizations.weight.original0", "weight_g"), True
            if key.endswith("parametrizations.weight.original1"):
                return key.replace("parametrizations.weight.original1", "weight_v"), True

        return key, False

    @classmethod
    def _fix_state_dict_keys_on_load(cls, state_dict):
        """Fixes state dict keys by replacing legacy parameter names with their modern equivalents.
        Logs if any parameters have been renamed.

        NOTE: this function comes from tranformers 4.49.0, and it has been removed afterwards. We keep it here
        to prevent having to modify all the from_pretrained code here.
        """

        renamed_keys = {}
        state_dict_keys = list(state_dict.keys())
        for key in state_dict_keys:
            new_key, has_changed = cls._fix_state_dict_key_on_load(key)
            if has_changed:
                state_dict[new_key] = state_dict.pop(key)

                # track gamma/beta rename for logging
                if key.endswith("LayerNorm.gamma"):
                    renamed_keys["LayerNorm.gamma"] = (key, new_key)
                elif key.endswith("LayerNorm.beta"):
                    renamed_keys["LayerNorm.beta"] = (key, new_key)

        if renamed_keys:
            warning_msg = f"A pretrained model of type `{cls.__name__}` "
            warning_msg += "contains parameters that have been renamed internally (a few are listed below but more are present in the model):\n"
            for old_key, new_key in renamed_keys.values():
                warning_msg += f"* `{old_key}` -> `{new_key}`\n"
            warning_msg += "If you are using a model from the Hub, consider submitting a PR to adjust these weights and help future users."
            logger.info_once(warning_msg)

        return state_dict

    @classmethod
    def _load_pretrained_model(
        cls,
        model,
        resolved_archive_file,
        pretrained_model_name_or_path,
        ignore_mismatched_sizes=False,
        sharded_metadata=None,
        _fast_init=True,
        device_map=None,
        dtype=None,
        weights_only=True,
    ):
        # This should always be a list but, just to be sure.
        if not isinstance(resolved_archive_file, list):
            resolved_archive_file = [resolved_archive_file]

        is_sharded = sharded_metadata is not None

        # ** Difference from original _load_pretrained_model **
        # We infer the loaded_keys as follows.
        state_dict = None
        if is_sharded:
            loaded_keys = sharded_metadata["all_checkpoint_keys"]
        else:
            state_dict = load_state_dict(
                resolved_archive_file[0], is_quantized=False, map_location=None, weights_only=weights_only
            )
            loaded_keys = list(state_dict.keys())

        # ** Difference from original _load_pretrained_model **
        # We do not support device_map="disk" for some of the weights so we do not add the code associated to it.

        # tie the model weights before retrieving the state_dict
        model.tie_weights()

        # Retrieve missing & unexpected_keys
        # ** Difference from original _load_pretrained_model **
        # We load dynamically a fake model on the meta device with the original implementation to get the keys.
        orig_transformers_cls = getattr(transformers, cls.__name__, None)
        with torch.device("meta"):
            # We must set the attention implementation to eager to avoid transformers checks which will fail on Neuron cores.
            clone_config = copy.deepcopy(model.config)
            clone_config._attn_implementation = "eager"
            meta_orig_model = orig_transformers_cls(clone_config)
        model_state_dict = meta_orig_model.state_dict()
        expected_keys = list(model_state_dict.keys())
        prefix = model.base_model_prefix

        # ** Difference from original _load_pretrained_model **
        # We do not support quantization so we do not add the code associated to it here.

        original_loaded_keys = loaded_keys
        loaded_keys = [cls._fix_state_dict_key_on_load(key)[0] for key in loaded_keys]

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
            _prefix = f"{prefix}."
            expected_keys_not_prefixed = [s for s in expected_keys if not s.startswith(_prefix)]
            expected_keys = [s[len(_prefix) :] if s.startswith(_prefix) else s for s in expected_keys]
        elif add_prefix_to_model:
            expected_keys = [".".join([prefix, s]) for s in expected_keys]

        missing_keys = sorted(set(expected_keys) - set(loaded_keys))
        unexpected_keys = set(loaded_keys) - set(expected_keys)

        # Remove nonpersistent buffers from unexpected keys: they are not in the state dict but will be in the model
        # buffers
        model_buffers = {n for n, _ in model.named_buffers()}
        if remove_prefix_from_model:
            model_buffers = {key[len(_prefix) :] if key.startswith(_prefix) else key for key in model_buffers}
        elif add_prefix_to_model:
            model_buffers = {".".join([prefix, key]) for key in model_buffers}
        unexpected_keys = sorted(unexpected_keys - model_buffers)

        # Clean up buffer for `inv-freq` because RoPE embedding moved under base model (https://github.com/huggingface/transformers/pull/34858)
        has_inv_freq_buffers = any(buffer.endswith("rotary_emb.inv_freq") for buffer in model_buffers)
        if has_inv_freq_buffers:
            unexpected_keys = {k for k in unexpected_keys if "rotary_emb.inv_freq" not in k}

        model.tie_weights()
        if device_map is None:
            ptrs = collections.defaultdict(list)
            for name, tensor in model.state_dict().items():
                id_tensor = id_tensor_storage(tensor)
                ptrs[id_tensor].append(name)

            # These are all the pointers of shared tensors.
            tied_params = [names for _, names in ptrs.items() if len(names) > 1]
        else:
            # id function doesn't work for meta tensor so we need this function
            tied_params = find_tied_parameters(model)

        new_tied_params = []
        for group in tied_params:
            if remove_prefix_from_model:
                group = [key[len(_prefix) :] if key.startswith(_prefix) else key for key in group]
            elif add_prefix_to_model:
                group = [".".join([prefix, key]) for key in group]
            new_tied_params.append(group)
            missing_in_group = [k for k in missing_keys if k in group]
            if len(missing_in_group) > 0 and len(missing_in_group) < len(group):
                missing_keys = [k for k in missing_keys if k not in missing_in_group]

        # If the model has tied parameters, we make sure they have the same names as in the state dict.
        tied_params = new_tied_params

        # Some models may have keys that are not in the state by design, removing them before needlessly warning
        # the user.
        if cls._keys_to_ignore_on_load_missing is not None:
            for pat in cls._keys_to_ignore_on_load_missing:
                missing_keys = [k for k in missing_keys if re.search(pat, k) is None]
        # If the key is not in model.parameters_for_current_stage, it is not missing, we just do not need it here.
        missing_keys = [k for k in missing_keys if k in model.parameters_for_current_stage]

        if cls._keys_to_ignore_on_load_unexpected is not None:
            for pat in cls._keys_to_ignore_on_load_unexpected:
                unexpected_keys = [k for k in unexpected_keys if re.search(pat, k) is None]

        # ** Difference from original _load_pretrained_model **
        # We do not support quantization so we do not add the code associated to it here.
        # We also skip the code related to `low_cpu_mem_usage`.

        # retrieve uninitialized modules and initialize before maybe overriding that with the pretrained weights.
        if _fast_init:
            if not ignore_mismatched_sizes:
                if remove_prefix_from_model:
                    _loaded_keys = [f"{prefix}.{k}" for k in loaded_keys]
                elif add_prefix_to_model:
                    _loaded_keys = [k[len(prefix) + 1 :] for k in loaded_keys]
                else:
                    _loaded_keys = loaded_keys

                # Mark loaded parameters/buffers as initialized (transformers 4.56.0+ approach)
                for key in model.state_dict():
                    if key in _loaded_keys:
                        param_or_buffer = model.get_parameter_or_buffer(key)
                        if param_or_buffer is not None:
                            param_or_buffer._is_hf_initialized = True

                # If we're about to tie the output embeds to the input embeds we don't need to init them
                if (
                    hasattr(model.config.get_text_config(decoder=True), "tie_word_embeddings")
                    and model.config.get_text_config(decoder=True).tie_word_embeddings
                ):
                    output_embeddings = model.get_output_embeddings()
                    if output_embeddings is not None:
                        # Still need to initialize if there is a bias term since biases are not tied.
                        if not hasattr(output_embeddings, "bias") or output_embeddings.bias is None:
                            output_embeddings._is_hf_initialized = True

                # Set the flag on modules recursively
                def set_is_initialized_for_modules(module):
                    if (
                        all(getattr(child, "_is_hf_initialized", False) for child in module.children())
                        and all(
                            getattr(param, "_is_hf_initialized", False) for param in module.parameters(recurse=False)
                        )
                        and all(
                            getattr(buffer, "_is_hf_initialized", False)
                            for buffer in module.buffers(recurse=False)
                            if buffer not in module._non_persistent_buffers_set
                        )
                    ):
                        module._is_hf_initialized = True

                model.apply(set_is_initialized_for_modules)
                not_initialized_submodules = {
                    name: mod for name, mod in model.named_modules() if not getattr(mod, "_is_hf_initialized", False)
                }
            else:
                not_initialized_submodules = dict(model.named_modules())

            # ** Difference from original _load_pretrained_model **
            # We initialize the parallel modules.
            for name, mod in not_initialized_submodules.items():
                if isinstance(mod, GQAQKVColumnParallelLinear):
                    # There is a bug in initialization for this module.
                    # In any case, we will always have weights for this in the case of `from_pretrained`.
                    continue
                elif isinstance(mod, BaseParallelLinear):
                    mod.initialize_weight_and_bias()

            # ** Difference from original _load_pretrained_model **
            # We do not add deepspeed related code.

            model.apply(model._initialize_weights)

        # ** Difference from original _load_pretrained_model **
        # We do not add keep_in_fp32_modules related code since it is not supported.

        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ""
        model_to_load = model
        if len(cls.base_model_prefix) > 0 and not hasattr(model, cls.base_model_prefix) and has_prefix_module:
            start_prefix = cls.base_model_prefix + "."
        if len(cls.base_model_prefix) > 0 and hasattr(model, cls.base_model_prefix) and not has_prefix_module:
            model_to_load = getattr(model, cls.base_model_prefix)
            base_model_expected_keys = list(model_to_load.state_dict().keys())
            if any(key in expected_keys_not_prefixed and key not in base_model_expected_keys for key in loaded_keys):
                raise ValueError(
                    "The state dictionary of the model you are trying to load is corrupted. Are you sure it was "
                    "properly saved?"
                )
            if device_map is not None:
                device_map = {k.replace(f"{cls.base_model_prefix}.", ""): v for k, v in device_map.items()}

        def _find_mismatched_keys(
            state_dict,
            model_state_dict,
            loaded_keys,
            original_loaded_keys,
            add_prefix_to_model,
            remove_prefix_from_model,
            ignore_mismatched_sizes,
        ):
            mismatched_keys = []
            if ignore_mismatched_sizes:
                for checkpoint_key, model_key in zip(original_loaded_keys, loaded_keys):
                    # If the checkpoint is sharded, we may not have the key here.
                    if checkpoint_key not in state_dict:
                        continue
                    if remove_prefix_from_model:
                        # The model key starts with `prefix` but `checkpoint_key` doesn't so we add it.
                        model_key = f"{prefix}.{model_key}"
                    elif add_prefix_to_model:
                        # The model key doesn't start with `prefix` but `checkpoint_key` does so we remove it.
                        model_key = ".".join(model_key.split(".")[1:])

                    if (
                        model_key in model_state_dict
                        and state_dict[checkpoint_key].shape != model_state_dict[model_key].shape
                    ):
                        if (
                            state_dict[checkpoint_key].shape[-1] == 1
                            and state_dict[checkpoint_key].numel() * 2 == model_state_dict[model_key].numel()
                        ):
                            # This skips size mismatches for 4-bit weights. Two 4-bit values share an 8-bit container, causing size differences.
                            # Without matching with module type or parameter type it seems like a practical way to detect valid 4bit weights.
                            pass
                        else:
                            mismatched_keys.append(
                                (checkpoint_key, state_dict[checkpoint_key].shape, model_state_dict[model_key].shape)
                            )
                            del state_dict[checkpoint_key]
            return mismatched_keys

        # ** Difference from original _load_pretrained_model **
        # We do not handle the `device_map` here, since our cases are much simpler.

        # ** Difference from original _load_pretrained_model **
        # We specialize the transformation specs for the model, this is required to have the specs properly defined.
        specialize_transformation_specs_for_model(model_to_load)

        # ** Difference from original _load_pretrained_model **
        # We do not add GGUF or low_cpu_mem_usage related code here.

        error_msgs = []
        mismatched_keys = []

        # ** Difference from original _load_pretrained_model **
        # We do not add the offload_index code here, since we do not support it.

        if len(resolved_archive_file) > 1:
            resolved_archive_file = logging.tqdm(resolved_archive_file, desc="Loading checkpoint shards")

        # In case some parameters weights are sharded across multiple files, we keep track of them to be able to adapt
        # them in successive calls to `adapt_state_dict`.
        upstanding_sharded_params = {}
        for shard_file in resolved_archive_file:
            # ** Difference from original _load_pretrained_model **
            # We do not use map_location here so we do not add the code associated to it.

            # We only need to load the state dict if it is a sharded checkpoint because if it is not sharded, the only
            # state dict that needs to be loaded was already loaded to get `loaded_keys`.
            if is_sharded:
                state_dict = load_state_dict(
                    shard_file, is_quantized=False, map_location=None, weights_only=weights_only
                )

            # Remove "duplicated" parameters that are tied together in the state dict.
            for group in tied_params:
                for param_name in group[1:]:
                    state_dict.pop(param_name, None)

            # ** Difference from original _load_state_dict_into_model **
            # We adapt the state dict to the custom model.
            state_dict = adapt_state_dict(
                model_to_load,
                state_dict,
                upstanding_sharded_params=upstanding_sharded_params,
                inplace=True,
            )

            # We need to remove the keys only after adapting the state dict otherwise the parameter names might not
            # match between the custom model and the checkpoint.
            for key in list(state_dict.keys()):
                # If the key is a parameter that is not needed for the current pipeline stage, we remove it.
                if key not in model.parameters_for_current_stage:
                    del state_dict[key]
            gc.collect()

            if get_pipeline_model_parallel_size() > 1:
                move_params_to_cpu(model_to_load, model.parameters_for_current_stage)

            # Mistmatched keys contains tuples key/shape1/shape2 of weights in the checkpoint that have a shape not
            # matching the weights in the model.
            mismatched_keys += _find_mismatched_keys(
                state_dict,
                model_state_dict,
                loaded_keys,
                original_loaded_keys,
                add_prefix_to_model,
                remove_prefix_from_model,
                ignore_mismatched_sizes,
            )

            # ** Difference from original _load_pretrained_model **
            # We do not add the code related to `low_cpu_mem_usage` here.

            # Sharded checkpoint or whole
            fixed_state_dict = cls._fix_state_dict_keys_on_load(state_dict)
            error_msgs += _load_state_dict_into_model(model_to_load, fixed_state_dict, start_prefix)

            # force memory release
            del state_dict
            gc.collect()

            # ** Difference from original _load_pretrained_model **
            # We do not add the offload_index code here, since we do not support it.

        # ** Difference from original _load_pretrained_model **
        # We specialize the specs on the full model regardless of prefixes.
        # This is this name that will be saved and used when re-loading the model.
        specialize_transformation_specs_for_model(model)

        if len(error_msgs) > 0:
            error_msg = "\n\t".join(error_msgs)
            if "size mismatch" in error_msg:
                error_msg += (
                    "\n\tYou may consider adding `ignore_mismatched_sizes=True` in the model `from_pretrained` method."
                )
            raise RuntimeError(f"Error(s) in loading state_dict for {model.__class__.__name__}:\n\t{error_msg}")

        if len(unexpected_keys) > 0:
            archs = [] if model.config.architectures is None else model.config.architectures
            warner = logger.warning if model.__class__.__name__ in archs else logger.info
            warner(
                f"Some weights of the model checkpoint at {pretrained_model_name_or_path} were not used when"
                f" initializing {model.__class__.__name__}: {unexpected_keys}\n- This IS expected if you are"
                f" initializing {model.__class__.__name__} from the checkpoint of a model trained on another task or"
                " with another architecture (e.g. initializing a BertForSequenceClassification model from a"
                " BertForPreTraining model).\n- This IS NOT expected if you are initializing"
                f" {model.__class__.__name__} from the checkpoint of a model that you expect to be exactly identical"
                " (initializing a BertForSequenceClassification model from a BertForSequenceClassification model)."
            )
        else:
            logger.info(f"All model checkpoint weights were used when initializing {model.__class__.__name__}.\n")
        if len(missing_keys) > 0:
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized: {missing_keys}\nYou should probably"
                " TRAIN this model on a down-stream task to be able to use it for predictions and inference."
            )
        elif len(mismatched_keys) == 0:
            logger.info(
                f"All the weights of {model.__class__.__name__} were initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path}.\nIf your task is similar to the task the model of the checkpoint"
                f" was trained on, you can already use {model.__class__.__name__} for predictions without further"
                " training."
            )
        if len(mismatched_keys) > 0:
            mismatched_warning = "\n".join(
                [
                    f"- {key}: found shape {shape1} in the checkpoint and {shape2} in the model instantiated"
                    for key, shape1, shape2 in mismatched_keys
                ]
            )
            logger.warning(
                f"Some weights of {model.__class__.__name__} were not initialized from the model checkpoint at"
                f" {pretrained_model_name_or_path} and are newly initialized because the shapes did not"
                f" match:\n{mismatched_warning}\nYou should probably TRAIN this model on a down-stream task to be able"
                " to use it for predictions and inference."
            )

        return model, missing_keys, unexpected_keys, mismatched_keys, error_msgs

    @classmethod
    def from_pretrained(
        cls: Type[SpecificPreTrainedModelType],
        pretrained_model_name_or_path: str | os.PathLike | None,
        trn_config: TrainingNeuronConfig,
        *model_args,
        config: PretrainedConfig | str | os.PathLike | None = None,
        cache_dir: str | os.PathLike | None = None,
        ignore_mismatched_sizes: bool = False,
        force_download: bool = False,
        local_files_only: bool = False,
        token: str | bool | None = None,
        revision: str = "main",
        use_safetensors: bool | None = None,
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
        dtype = kwargs.pop("dtype", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        # For BC on torch_dtype argument (deprecated in favor of dtype)
        if torch_dtype is not None:
            dtype = dtype if dtype is not None else torch_dtype
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", None)
        device_map = kwargs.pop("device_map", None)
        kwargs.pop("max_memory", None)
        offload_folder = kwargs.pop("offload_folder", None)
        offload_state_dict = kwargs.pop("offload_state_dict", False)
        load_in_8bit = kwargs.pop("load_in_8bit", False)
        load_in_4bit = kwargs.pop("load_in_4bit", False)
        quantization_config = kwargs.pop("quantization_config", None)
        subfolder = kwargs.pop("subfolder", "")
        commit_hash = kwargs.pop("_commit_hash", None)
        variant = kwargs.pop("variant", None)
        adapter_kwargs = kwargs.pop("adapter_kwargs", None)
        adapter_name = kwargs.pop("adapter_name", None)
        use_flash_attention_2 = kwargs.pop("use_flash_attention_2", False)
        kwargs.pop("generation_config", None)
        gguf_file = kwargs.pop("gguf_file", None)

        if state_dict is not None:
            raise NotImplementedError(
                "Providing a `state_dict` to `from_pretrained` is not supported in optimum-neuron."
            )

        if from_tf or from_flax:
            raise NotImplementedError(
                "Loading from TensorFlow or Flax is not supported in optimum-neuron. Please use PyTorch weights."
            )

        if low_cpu_mem_usage is not None:
            raise NotImplementedError("`low_cpu_mem_usage` is not supported in optimum-neuron.")

        if use_flash_attention_2:
            raise ValueError(
                '`use_flash_attention_2` is deprecated, use `attn_implementation="flash_attention_2". instead'
            )

        # We support less features from the device_map since moving to device is handled by the compiler.
        # Only `None`, "xla" and "cpu" as device_map values are supported.
        if device_map not in [None, "xla", "cpu"]:
            raise RuntimeError('The only device map values supported are: `None`, "cpu" or "xla".')

        if offload_folder is not None or offload_state_dict:
            raise NotImplementedError("`offload_folder` and `offload_state_dict` are not supported in optimum-neuron.")

        if load_in_8bit or load_in_4bit or quantization_config is not None:
            raise NotImplementedError("Quantization is not supported yet.")

        if gguf_file is not None:
            raise NotImplementedError("GGUF files are not supported in optimum-neuron.")

        if adapter_name is not None or adapter_kwargs is not None:
            raise NotImplementedError(
                "Loading adapters directly from {cls.__name__}.from_pretrained is not supported. "
                "Please use the NeuronPeftModelForXXX classes to load adapters."
            )

        # ** Difference from original from_pretrained **
        # Here we ignore the `tp_plan` argument and handle tensor parallelism ourselves.
        tp_plan = kwargs.pop("tp_plan", None)
        if tp_plan is not None:
            logger.info("optimum-neuron handles tensor parallelism on its own. `tp_plan` is ignored.")
            tp_plan = None

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
            # We do not support loading adapters directly from the `from_pretrained` method.
            # We check if the provided model name or path is an adapter, and fail if needed.
            _adapter_model_path = find_adapter_config_file(
                pretrained_model_name_or_path,
                cache_dir=cache_dir,
                force_download=force_download,
                resume_download=resume_download,
                proxies=proxies,
                local_files_only=local_files_only,
                _commit_hash=commit_hash,
            )
            if _adapter_model_path is not None and os.path.isfile(_adapter_model_path):
                with open(_adapter_model_path, "r", encoding="utf-8") as f:
                    _adapter_model_path = pretrained_model_name_or_path
                    pretrained_model_name_or_path = json.load(f)["base_model_name_or_path"]
        else:
            _adapter_model_path = None

        if _adapter_model_path is not None:
            raise NotImplementedError(
                f"Loading adapters directly from {cls.__name__}.from_pretrained is not supported. "
                "Please use the NeuronPeftModelForXXX classes to load adapters."
            )

        user_agent = {"file_type": "model", "framework": "pytorch", "from_auto_class": from_auto_class}
        if from_pipeline is not None:
            user_agent["using_pipeline"] = from_pipeline

        if is_offline_mode() and not local_files_only:
            logger.info("Offline mode: forcing local_files_only=True")
            local_files_only = True

        # Load config if we don't provide a configuration
        if not isinstance(config, PretrainedConfig):
            config_path = config if config is not None else pretrained_model_name_or_path
            config, model_kwargs = cls.config.from_pretrained(
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
            config = copy.deepcopy(config)
            model_kwargs = kwargs

        if "attn_implementation" in kwargs:
            config._attn_implementation = kwargs.pop("attn_implementation")

        # ** Difference from original from_pretrained **
        # In the original from_pretrained, here there was some initialization to support quantization.
        # We just do not add it here because it is not supported.
        pre_quantized = hasattr(config, "quantization_config")
        if pre_quantized and not AutoHfQuantizer.supports_quant_method(config.quantization_config):
            pre_quantized = False

        if pre_quantized or quantization_config is not None:
            raise NotImplementedError("Quantization is not supported in optimum-neuron.")

        # This variable will flag if we're loading a sharded checkpoint. In this case the archive file is just the
        # index of the files.
        is_sharded = False
        sharded_metadata = None
        # Load model
        loading_info = None

        # ** Difference from original from_pretrained **
        # We set a few variables that will be needed later in the code.
        if trn_config is None:
            raise ValueError("`trn_config` is required to load a model in optimum-neuron.")
        num_local_ranks_per_step = trn_config.num_local_ranks_per_step
        local_world_size = get_local_world_size()
        local_rank = xm.get_local_ordinal()
        if num_local_ranks_per_step <= 0:
            num_local_ranks_per_step = local_world_size

        if pretrained_model_name_or_path is not None:
            pretrained_model_name_or_path = str(pretrained_model_name_or_path)
            is_local = os.path.isdir(pretrained_model_name_or_path)
            if is_local:
                if use_safetensors is not False and os.path.isfile(
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
                        f" {pretrained_model_name_or_path} but there is a file for TensorFlow weights. These weights "
                        "cannot be loaded in optimum-neuron."
                    )
                elif not use_safetensors and os.path.isfile(
                    os.path.join(pretrained_model_name_or_path, subfolder, FLAX_WEIGHTS_NAME)
                ):
                    raise EnvironmentError(
                        f"Error no file named {_add_variant(WEIGHTS_NAME, variant)} found in directory"
                        f" {pretrained_model_name_or_path} but there is a file for Flax weights. These weights cannot"
                        " be loaded in optimum-neuron."
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
                raise ValueError(
                    f"We found a TensorFlow checkpoint at {pretrained_model_name_or_path + '.index'}, this checkpoint "
                    "cannot be loaded in optimum-neuron."
                )
            elif is_remote_url(pretrained_model_name_or_path):
                filename = pretrained_model_name_or_path
                resolved_archive_file = download_url(pretrained_model_name_or_path)
            else:
                # set correct filename
                if use_safetensors is not False:
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
                                    " These weights cannot be loaded in optimum-neuron."
                                )
                            elif has_file(pretrained_model_name_or_path, FLAX_WEIGHTS_NAME, **has_file_kwargs):
                                raise EnvironmentError(
                                    f"{pretrained_model_name_or_path} does not appear to have a file named"
                                    f" {_add_variant(WEIGHTS_NAME, variant)} but there is a file for Flax weights. "
                                    " These weights cannot be loaded in optimum-neuron."
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
            elif metadata.get("format") == "mlx":
                # This is a mlx file, we assume weights are compatible with pt
                pass
            else:
                raise ValueError(
                    f"Incompatible safetensors file. File metadata is not ['pt', 'tf', 'flax', 'mlx'] but {metadata.get('format')}"
                )

        # ** Difference from original from_pretrained **
        # We do not load the state_dict (when not sharded) here as it is done in the original implementation.
        # We do it only in `cls._load_state_dict`.

        # Set dtype to instantiate the model under:
        # 1. If torch_dtype is not None, we use that dtype
        # 2. If torch_dtype is "auto", we auto-detect dtype from the loaded state_dict, by checking its first
        #    weights entry that is of a floating type - we assume all floating dtype weights are of the same dtype
        # We also may have config.dtype available, but we won't rely on it till v5
        dtype_orig = None

        if dtype is not None:
            if isinstance(dtype, str):
                if dtype == "auto":
                    if hasattr(config, "dtype") and config.dtype is not None:
                        dtype = config.dtype
                        logger.info(f"Will use dtype={dtype} as defined in model's config object")
                    else:
                        if is_sharded and "dtype" in sharded_metadata:
                            dtype = sharded_metadata["dtype"]
                        elif not is_sharded:
                            # ** Difference from original from_pretrained **
                            # Here we load the state dict only if we end up in this case, otherwise we defer the
                            # loading for later.
                            for worker in range(math.ceil(local_world_size / num_local_ranks_per_step)):
                                if local_rank // num_local_ranks_per_step == worker:
                                    one_time_state_dict = load_state_dict(
                                        resolved_archive_file, weights_only=weights_only
                                    )
                                    dtype = get_state_dict_dtype(one_time_state_dict)
                                    del one_time_state_dict
                                xm.rendezvous(f"auto dtype_{worker}")
                        else:
                            one_state_dict = load_state_dict(resolved_archive_file[0], weights_only=weights_only)
                            dtype = get_state_dict_dtype(one_state_dict)
                            del one_state_dict  # free CPU memory
                        logger.info(
                            "Since the `dtype` attribute can't be found in model's config object, "
                            "will use dtype={dtype} as derived from model's weights"
                        )
                elif hasattr(torch, dtype):
                    dtype = getattr(torch, dtype)
                    for sub_config_key in config.sub_configs.keys():
                        sub_config = getattr(config, sub_config_key)
                        sub_config.dtype = dtype
            elif isinstance(dtype, torch.dtype):
                for sub_config_key in config.sub_configs.keys():
                    sub_config = getattr(config, sub_config_key)
                    sub_config.dtype = dtype
            elif isinstance(dtype, dict):
                for key, curr_dtype in dtype.items():
                    if hasattr(config, key):
                        value = getattr(config, key)
                        value.dtype = curr_dtype
                # main torch dtype for modules that aren't part of any sub-config
                dtype = dtype.get("")
                config.dtype = dtype
                if isinstance(dtype, str) and hasattr(torch, dtype):
                    dtype = getattr(torch, dtype)
                elif dtype is None:
                    dtype = torch.float32
            else:
                raise ValueError(
                    f"`dtype` can be one of: `torch.dtype`, `'auto'`, a string of a valid `torch.dtype` or a `dict` with valid `dtype` "
                    f"for each sub-config in composite configs, but received {dtype}"
                )

            dtype_orig = cls._set_default_dtype(dtype)
        else:
            # set fp32 as the default dtype for BC
            default_dtype = str(torch.get_default_dtype()).split(".")[-1]
            config.dtype = default_dtype
            for key in config.sub_configs.keys():
                value = getattr(config, key)
                value.dtype = default_dtype

        # ** Difference from original from_pretrained **
        # We do not handle `use_keep_in_fp32_modules` here since it is not relevant for us.

        # ** Difference from original from_pretrained **
        # We do not create the `loaded_state_dict_keys` variable here as it is done in the original implementation,
        # instead we compute these keys in `cls._load_pretrained_model`.

        config.name_or_path = pretrained_model_name_or_path

        # Instantiate model.
        init_contexts = [no_init_weights()]

        # If we are using pipeline parallelism, we need to use the meta device for parameters only while keeping buffers on CPU.
        if trn_config.pipeline_parallel_size > 1:
            init_contexts.append(MetaParametersOnly())

        # ** Difference from original from_pretrained **
        # In the original from_pretrained implementation there is deepspeed and low_cpu_mem_usage code for
        # `init_contexts`.
        # We do not put it here since we do not support it.

        config = copy.deepcopy(config)  # We do not want to modify the config inplace in from_pretrained.

        # ** Difference from original from_pretrained **
        # We make sure that config.dtype is of type torch.dtype.
        # We do not change the config inplace since we are working from a deepcopy.
        config.dtype = dtype

        # ** Difference from original from_pretrained **
        # We do not support the `tie_word_embeddings` feature in pipeline parallelism.
        # Instead when `config.tie_word_embeddings` is set to True, we set it to False and simply clone the data between
        # the tied weights.

        with ContextManagers(init_contexts):
            # Let's make sure we don't run the init function of buffer modules
            model = cls(config, trn_config, *model_args, **model_kwargs)

        if trn_config.pipeline_parallel_size > 1:
            move_params_to_cpu(model, model.parameters_for_current_stage)

        # make sure we use the model's config since the __init__ call might have copied it
        config = model.config

        # ** Difference from original from_pretrained **
        # Here there is some code related to quantization, we skip it.
        # We also skip the code related to `device_map` since we do not support the cases it handles.

        # ** Difference from original from_pretrained **
        # We do not add cases for `from_tf` and `from_flax` since we do not support them.

        # restore default dtype
        if dtype_orig is not None:
            torch.set_default_dtype(dtype_orig)

        # ** Difference from original from_pretrained **
        # Here we load the pretrained model by group of ranks.
        # For pipeline parallelism, we only load the parameters needed for the current stage.
        # The `cls._load_pretrained_model` method takes a subset of the original parameters because we support a subset
        # of the original features.

        for worker in range(math.ceil(local_world_size / num_local_ranks_per_step)):
            if local_rank // num_local_ranks_per_step == worker:
                (
                    model,
                    missing_keys,
                    unexpected_keys,
                    mismatched_keys,
                    error_msgs,
                ) = cls._load_pretrained_model(
                    model,
                    resolved_archive_file,
                    pretrained_model_name_or_path,
                    ignore_mismatched_sizes=ignore_mismatched_sizes,
                    sharded_metadata=sharded_metadata,
                    _fast_init=_fast_init,
                    device_map=device_map,
                    dtype=dtype,
                    weights_only=weights_only,
                )

            xm.rendezvous(f"load_state_dict_{worker}")

        # make sure token embedding weights are still tied if needed
        model.tie_weights()

        # Set model in evaluation mode to deactivate DropOut modules by default
        model.eval()

        # ** Difference from original from_pretrained **
        # We skip the code about generation since we do not support this.

        # ** Difference from original from_pretrained **
        # We skip the code about the hf_quantizer, not supported.

        if device_map == "xla":
            move_model_to_device(model, xm.xla_device())

        if output_loading_info:
            if loading_info is None:
                loading_info = {
                    "missing_keys": missing_keys,
                    "unexpected_keys": unexpected_keys,
                    "mismatched_keys": mismatched_keys,
                    "error_msgs": error_msgs,
                }
            return model, loading_info

        return model

    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        is_main_process: bool | Literal["auto"] = "auto",
        state_dict: dict | None = None,
        save_function: Callable = torch.save,
        push_to_hub: bool = False,
        max_shard_size: int | str = "5GB",
        safe_serialization: bool = False,
        variant: str | None = None,
        token: str | bool | None = None,
        save_peft_format: bool = True,
        optimizer: torch.optim.Optimizer | None = None,
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

        if "save_config" in kwargs:
            warnings.warn(
                "`save_config` is deprecated and will be removed in v5 of Transformers. Use `is_main_process` instead."
            )
            is_main_process = kwargs.pop("save_config")
        if safe_serialization:
            raise NotImplementedError(
                "`safe_serialization` is not supported when saving the sharded checkpoints. It is possible to "
                "consolidate the model weights into `safetensors` format."
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

        # We need to specialize the transformation specs for the model before saving.
        specialize_transformation_specs_for_model(model_to_save)

        # save the string version of dtype to the config, e.g. convert torch.float32 => "float32"
        # we currently don't use this setting automatically, but may start to use with v5
        dtype = get_parameter_dtype(model_to_save)
        model_to_save.config.dtype = str(dtype).split(".")[1]

        # Attach architecture to the config
        model_to_save.config.architectures = [model_to_save.__class__.__name__]

        # Unset attn implementation so it can be set to another one when loading back
        model_to_save.config._attn_implementation_autoset = False

        # Save the model
        if self.trn_config.async_save:
            _wait_previous_async_save(save_directory / MODEL_PARALLEL_SHARDS_DIR_NAME)

        neuronx_distributed.trainer.save_checkpoint(
            save_directory.as_posix(),
            tag=MODEL_PARALLEL_SHARDS_DIR_NAME,
            model=model_to_save,
            optimizer=optimizer,
            use_xser=self.trn_config.use_xser,
            async_save=self.trn_config.async_save,
            num_workers=self.trn_config.num_local_ranks_per_step,
        )

        # Save the metadata required to consolidate the checkpoints properly.
        if get_data_parallel_rank() == 0 and get_tensor_model_parallel_rank() == 0:
            metadata = create_parameter_metadata(model_to_save)
            pp_rank = get_pipeline_model_parallel_rank()
            metadata_path = save_directory / MODEL_PARALLEL_SHARDS_DIR_NAME / f"mp_metadata_pp_rank_{pp_rank}.json"
            metadata_path.parent.mkdir(parents=True, exist_ok=True)
            with open(metadata_path, "w") as f:
                f.write(json.dumps(metadata, indent=4))

        # Save the config
        if is_main_process:
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

            with open(save_directory / "trn_config.json", "w") as f:
                trn_config_data = asdict(self.trn_config)
                if isinstance(trn_config_data["checkpoint_dir"], Path):
                    trn_config_data["checkpoint_dir"] = trn_config_data["checkpoint_dir"].as_posix()
                f.write(json.dumps(trn_config_data, indent=4))

    def resize_token_embeddings(
        self,
        new_num_tokens: int | None = None,
        pad_to_multiple_of: int | None = None,
        mean_resizing: bool = False,
    ) -> "nn.Embedding | ParallelEmbedding":
        embeddings = super().resize_token_embeddings(new_num_tokens, pad_to_multiple_of, mean_resizing)
        # The way the vocab size by the main method is wrong when using ParallelEmbedding.
        # So we reset it here.
        self.config.get_text_config().vocab_size = embeddings.num_embeddings
        self.vocab_size = embeddings.num_embeddings
        return embeddings

    def _get_resized_embeddings(
        self,
        old_embeddings: nn.Embedding | ParallelEmbedding,
        new_num_tokens: int | None = None,
        pad_to_multiple_of: int | None = None,
        mean_resizing: bool = False,
    ):
        """
        Override of transformers method to handle ParallelEmbedding layers in tensor parallel scenarios.
        Falls back to the base implementation for regular nn.Embedding layers.
        """
        if isinstance(old_embeddings, ParallelEmbedding):
            return self._get_resized_parallel_embeddings(
                old_embeddings, new_num_tokens, pad_to_multiple_of, mean_resizing
            )
        else:
            # Fall back to standard transformers method for regular embeddings
            return super()._get_resized_embeddings(old_embeddings, new_num_tokens, pad_to_multiple_of, mean_resizing)

    def _get_resized_lm_head(
        self,
        old_lm_head: nn.Linear | ColumnParallelLinear,
        new_num_tokens: int | None = None,
        transposed: bool = False,
        mean_resizing: bool = False,
    ):
        """
        Override of transformers method to handle ColumnParallelLinear layers in tensor parallel scenarios.
        Falls back to the base implementation for regular nn.Linear layers.
        """
        if isinstance(old_lm_head, ColumnParallelLinear):
            return self._get_resized_parallel_lm_head(old_lm_head, new_num_tokens, transposed, mean_resizing)
        else:
            # Fall back to standard transformers method for regular linear layers
            return super()._get_resized_lm_head(old_lm_head, new_num_tokens, transposed, mean_resizing)

    def _get_resized_parallel_embeddings(
        self,
        old_embeddings: ParallelEmbedding,
        new_num_tokens: int | None = None,
        pad_to_multiple_of: int | None = None,
        mean_resizing: bool = False,
    ) -> ParallelEmbedding:
        """
        Builds a resized ParallelEmbedding from a provided ParallelEmbedding.
        """
        if mean_resizing:
            raise NotImplementedError(
                "Mean resizing is not supported for ParallelEmbedding layers. "
                "Please use standard initialization for resizing."
            )

        # Handle padding to multiple
        if pad_to_multiple_of is not None:
            if not isinstance(pad_to_multiple_of, int):
                raise ValueError(
                    f"Asking to pad the embedding matrix to a multiple of `{pad_to_multiple_of}`, which is not an "
                    "integer. Please make sure to pass an integer"
                )
            if new_num_tokens is None:
                new_num_tokens = old_embeddings.num_embeddings
            new_num_tokens = ((new_num_tokens + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of

        if new_num_tokens is None:
            return old_embeddings

        # Get TP configuration
        tp_size = get_tensor_model_parallel_size()

        # Ensure vocab size is divisible by TP size
        if new_num_tokens % tp_size != 0:
            raise ValueError(
                f"New vocabulary size ({new_num_tokens}) must be divisible by tensor parallel size ({tp_size})"
            )

        old_num_tokens_global = old_embeddings.num_embeddings
        old_num_tokens_local = old_embeddings.num_embeddings_per_partition
        old_embedding_dim = old_embeddings.embedding_dim

        new_num_tokens_local = new_num_tokens // tp_size

        if old_num_tokens_global == new_num_tokens:
            return old_embeddings

        # Create new ParallelEmbedding with the same configuration
        new_embeddings = ParallelEmbedding(
            new_num_tokens,
            old_embedding_dim,
            padding_idx=old_embeddings.padding_idx,
            max_norm=old_embeddings.max_norm,
            norm_type=old_embeddings.norm_type,
            scale_grad_by_freq=old_embeddings.scale_grad_by_freq,
            sparse=old_embeddings.sparse,
            device=old_embeddings.weight.device,
            dtype=old_embeddings.weight.dtype,
            shard_across_embedding=old_embeddings.shard_across_embedding,
            pad=old_embeddings.pad,
            tensor_model_parallel_group=old_embeddings.tensor_model_parallel_group,
            sequence_parallel_enabled=old_embeddings.sequence_parallel_enabled,
            sequence_dimension=old_embeddings.sequence_dim,
        )

        # Copy existing weights
        n_local = min(old_num_tokens_local, new_num_tokens_local)
        with torch.no_grad():
            new_embeddings.weight.data[:n_local, :] = old_embeddings.weight.data[:n_local, :]

        # Initialize new tokens if expanding
        if new_num_tokens_local > old_num_tokens_local:
            added_tokens_local = new_num_tokens_local - old_num_tokens_local
            # Use standard initialization
            with torch.no_grad():
                # Initialize the new token embeddings with the model's standard initialization
                if hasattr(self, "_init_weights"):
                    # Create a temporary embedding to get proper initialization
                    temp_embedding = torch.nn.Embedding(added_tokens_local, old_embedding_dim)
                    self._init_weights(temp_embedding)
                    new_embeddings.weight.data[old_num_tokens_local:, :] = temp_embedding.weight.data
                else:
                    # Fallback to normal initialization
                    std = getattr(self.config, "initializer_range", 0.02)
                    new_embeddings.weight.data[old_num_tokens_local:, :].normal_(mean=0.0, std=std)

        return new_embeddings

    def _get_resized_parallel_lm_head(
        self,
        old_lm_head: ColumnParallelLinear,
        new_num_tokens: int | None = None,
        transposed: bool | None = False,
        mean_resizing: bool = False,
    ) -> ColumnParallelLinear:
        """
        Builds a resized ColumnParallelLinear from a provided ColumnParallelLinear.
        """
        if mean_resizing:
            raise NotImplementedError(
                "Mean resizing is not supported for ColumnParallelLinear layers. "
                "Please use standard initialization for resizing."
            )

        from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_size

        if new_num_tokens is None:
            return old_lm_head

        # Get TP configuration
        tp_size = get_tensor_model_parallel_size()

        # Ensure vocab size is divisible by TP size
        if new_num_tokens % tp_size != 0:
            raise ValueError(
                f"New vocabulary size ({new_num_tokens}) must be divisible by tensor parallel size ({tp_size})"
            )

        old_num_tokens_global = old_lm_head.output_size
        old_num_tokens_local = old_lm_head.weight.shape[0]  # First dimension for ColumnParallelLinear

        new_num_tokens_local = new_num_tokens // tp_size

        if old_num_tokens_global == new_num_tokens:
            return old_lm_head

        # Create new ColumnParallelLinear with the same configuration
        new_lm_head = ColumnParallelLinear(
            old_lm_head.input_size,
            new_num_tokens,
            bias=old_lm_head.bias is not None,
            gather_output=old_lm_head.gather_output,
            dtype=old_lm_head.dtype,
            device=old_lm_head.weight.device,
            stride=old_lm_head.stride,
            init_method=old_lm_head.arg_init_method,
            sequence_parallel_enabled=old_lm_head.sequence_parallel_enabled,
            sequence_dimension=old_lm_head.sequence_dimension,
            keep_master_weight=old_lm_head.keep_master_weight,
            skip_bias_add=old_lm_head.skip_bias_add,
            pad=old_lm_head.pad,
            tensor_model_parallel_group=old_lm_head.tensor_parallel_group,
            reduce_dtype=old_lm_head.reduce_dtype,
        )

        # Copy existing weights
        n_local = min(old_num_tokens_local, new_num_tokens_local)
        with torch.no_grad():
            if transposed:
                # Weight is [input_size, output_size_local]
                new_lm_head.weight.data[:, :n_local] = old_lm_head.weight.data[:, :n_local]
            else:
                # Weight is [output_size_local, input_size] (standard ColumnParallelLinear)
                new_lm_head.weight.data[:n_local, :] = old_lm_head.weight.data[:n_local, :]

            # Copy bias if present
            if old_lm_head.bias is not None and new_lm_head.bias is not None:
                new_lm_head.bias.data[:n_local] = old_lm_head.bias.data[:n_local]

        # Initialize new tokens if expanding
        if new_num_tokens_local > old_num_tokens_local:
            added_tokens_local = new_num_tokens_local - old_num_tokens_local
            # Use standard initialization
            with torch.no_grad():
                # Initialize the new token weights with the model's standard initialization
                if hasattr(self, "_init_weights"):
                    # Create a temporary linear layer to get proper initialization
                    if transposed:
                        temp_linear = torch.nn.Linear(
                            added_tokens_local, old_lm_head.input_size, bias=old_lm_head.bias is not None
                        )
                        self._init_weights(temp_linear)
                        new_lm_head.weight.data[:, old_num_tokens_local:] = temp_linear.weight.data.T
                        if temp_linear.bias is not None and new_lm_head.bias is not None:
                            new_lm_head.bias.data[old_num_tokens_local:] = temp_linear.bias.data
                    else:
                        temp_linear = torch.nn.Linear(
                            old_lm_head.input_size, added_tokens_local, bias=old_lm_head.bias is not None
                        )
                        self._init_weights(temp_linear)
                        new_lm_head.weight.data[old_num_tokens_local:, :] = temp_linear.weight.data
                        if temp_linear.bias is not None and new_lm_head.bias is not None:
                            new_lm_head.bias.data[old_num_tokens_local:] = temp_linear.bias.data
                else:
                    # Fallback to normal initialization
                    std = getattr(self.config, "initializer_range", 0.02)
                    if transposed:
                        new_lm_head.weight.data[:, old_num_tokens_local:].normal_(mean=0.0, std=std)
                    else:
                        new_lm_head.weight.data[old_num_tokens_local:, :].normal_(mean=0.0, std=std)
                    if new_lm_head.bias is not None:
                        new_lm_head.bias.data[old_num_tokens_local:].zero_()

        return new_lm_head
