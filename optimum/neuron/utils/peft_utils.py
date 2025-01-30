# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Utilities related to the PEFT library and support."""

import collections
import functools
import os
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import torch

from .import_utils import is_peft_available
from .patching import Patcher, replace_class_in_inheritance_hierarchy
from .require_utils import requires_neuronx_distributed, requires_safetensors
from .training_utils import _get_model_param_count


if is_peft_available():
    from peft import PeftModel
    from peft import get_peft_model as orig_get_peft_model
    from peft.utils import (
        SAFETENSORS_WEIGHTS_NAME,
        WEIGHTS_NAME,
        id_tensor_storage,
        set_peft_model_state_dict,
    )
    from peft.utils import (
        get_peft_model_state_dict as orig_get_peft_model_state_dict,
    )

else:
    SAFETENSORS_WEIGHTS_NAME = WEIGHTS_NAME = ""

    class PeftModel:
        pass

    def orig_get_peft_model(*args, **kwargs):
        pass

    def orig_get_peft_model_state_dict(*args, **kwargs):
        pass

    def set_peft_model_state_dict(*args, **kwargs):
        pass

    def id_tensor_storage(*args, **kwargs):
        pass


ADAPTER_MODEL_PARALLEL_SHARDS_DIR_NAME = "adapter_shards"


@requires_neuronx_distributed
def has_valid_embedding_base_layer(layer):
    """Check if the layer has an embedding base layer"""
    from neuronx_distributed.parallel_layers.layers import BaseParallelLinear, ParallelEmbedding

    return hasattr(layer, "base_layer") and isinstance(
        layer.base_layer, (torch.nn.Linear, torch.nn.Embedding, ParallelEmbedding, BaseParallelLinear)
    )


@functools.wraps(orig_get_peft_model_state_dict)
def get_peft_model_state_dict(*args, **kwargs):
    with Patcher([("peft.utils.save_and_load.has_valid_embedding_base_layer", has_valid_embedding_base_layer)]):
        return orig_get_peft_model_state_dict(*args, **kwargs)


class NeuronPeftModel(PeftModel):
    @requires_neuronx_distributed
    @requires_safetensors
    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = True,
        selected_adapters: Optional[List[str]] = None,
        save_embedding_layers: Union[str, bool] = "auto",
        is_main_process: bool = True,
        convert_pissa_to_lora: Optional[str] = None,
        async_save: bool = False,
        **kwargs: Any,
    ):
        import neuronx_distributed
        import torch_xla.core.xla_model as xm
        from neuronx_distributed.parallel_layers.parallel_state import (
            get_data_parallel_rank,
            get_pipeline_model_parallel_rank,
            get_pipeline_model_parallel_size,
            get_tensor_model_parallel_rank,
            get_tensor_model_parallel_size,
            model_parallel_is_initialized,
        )
        from neuronx_distributed.parallel_layers.utils import move_all_tensor_to_cpu
        from safetensors.torch import save_file as safe_save_file

        if model_parallel_is_initialized():
            should_write_data = get_data_parallel_rank() == 0
            is_model_paralllel = get_tensor_model_parallel_size() > 1 or get_pipeline_model_parallel_size() > 1
        else:
            should_write_data = xm.is_master_ordinal(local=True)
            is_model_paralllel = False

        if os.path.isfile(save_directory):
            raise ValueError(f"Provided path ({save_directory}) should be a directory, not a file")

        if selected_adapters is None:
            selected_adapters = list(self.peft_config.keys())
        else:
            if any(
                selected_adapter_name not in list(self.peft_config.keys())
                for selected_adapter_name in selected_adapters
            ):
                raise ValueError(
                    f"You passed an invalid `selected_adapters` arguments, current supported adapter names are"
                    f" {list(self.peft_config.keys())} - got {selected_adapters}."
                )

        def save_pissa_as_lora(peft_config, convert_pissa_to_lora, output_state_dict, kwargs):
            if not str(peft_config.init_lora_weights).startswith("pissa"):
                warnings.warn("`convert_pissa_to_lora` only works for converting a PiSSA adapter to a LoRA adapter")
            initial_adapter = os.path.basename(convert_pissa_to_lora)
            self.load_adapter(
                os.path.dirname(convert_pissa_to_lora), subfolder=initial_adapter, adapter_name=initial_adapter
            )
            if str(self.peft_config[initial_adapter].init_lora_weights).startswith("pissa"):
                raise ValueError(
                    "The `init_lora_weights` parameter of the initial PiSSA adapter should be set to `True`. "
                    "Otherwise, `self.load_adapter` will subtract the principal singular value and vector again based on the residual model."
                )
            output_state_dict = self.base_model.subtract_pissa_init(output_state_dict, initial_adapter, kwargs)
            self.delete_adapter(adapter_name)
            return output_state_dict

        if is_main_process:
            os.makedirs(save_directory, exist_ok=True)
            self.create_or_update_model_card(save_directory)

        for adapter_name in selected_adapters:
            peft_config = self.peft_config[adapter_name]
            # save only the trainable weights
            output_state_dict = get_peft_model_state_dict(
                self,
                state_dict=kwargs.get("state_dict", None),
                adapter_name=adapter_name,
                save_embedding_layers=save_embedding_layers,
            )
            output_dir = os.path.join(save_directory, adapter_name) if adapter_name != "default" else save_directory
            os.makedirs(output_dir, exist_ok=True)

            if is_model_paralllel:
                if convert_pissa_to_lora is not None:
                    output_state_dict = save_pissa_as_lora(
                        peft_config, convert_pissa_to_lora, output_state_dict, kwargs
                    )

                # Because `neuronx_distributed.trainer.save_checkpoint` only accepts `torch.nn.Module` we create a fake
                # module containing the state dict.
                class DummyModule(torch.nn.Module):
                    def state_dict(self):
                        return output_state_dict

                adapter_shards_dir_model = os.path.join(output_dir, "adapter_shards", "model")
                if not os.path.isdir(adapter_shards_dir_model):
                    os.makedirs(adapter_shards_dir_model, exist_ok=True)

                dummy_mod = DummyModule()
                neuronx_distributed.trainer.save_checkpoint(
                    output_dir,
                    tag="adapter_shards",
                    model=dummy_mod,
                    async_save=async_save,
                )

                # Importing here to avoid circular imports.
                from ..distributed.utils import get_parameters_tp_metadata

                metadata = {}
                named_parameters_without_adapter_name = {
                    n.replace(f".{adapter_name}", ""): p for n, p in self.named_parameters()
                }
                metadata["sharded_metadata"] = {
                    k: asdict(v) for k, v in get_parameters_tp_metadata(named_parameters_without_adapter_name).items()
                }
                metadata["gqa_qkv_metadata"] = self._gqa_qkv_metadata

                if get_data_parallel_rank() == 0 and get_tensor_model_parallel_rank() == 0:
                    pp_rank = get_pipeline_model_parallel_rank()
                    metadata_path = (
                        Path(output_dir) / ADAPTER_MODEL_PARALLEL_SHARDS_DIR_NAME / f"mp_metadata_pp_rank_{pp_rank}.pt"
                    )
                    # Checking that the parent directory exists, it should exist, but let's make sure since g_iostate.end() is
                    # called at the end of `neuronx_distributed.trainer.save_checkpoint` and it can remove checkpoint
                    # directories if the max limit has been reached.
                    if metadata_path.parent.is_dir():
                        torch.save(metadata, metadata_path)

            elif is_main_process and safe_serialization:
                output_state_dict = move_all_tensor_to_cpu(output_state_dict, convert=should_write_data)
                # Section copied from: https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_utils.py#L2111-L2134
                # Safetensors does not allow tensor aliasing.
                # We're going to remove aliases before saving
                ptrs = collections.defaultdict(list)
                for name, tensor in output_state_dict.items():
                    # Sometimes in the state_dict we have non-tensor objects.
                    # e.g. in bitsandbytes we have some `str` objects in the state_dict
                    if isinstance(tensor, torch.Tensor):
                        ptrs[id_tensor_storage(tensor)].append(name)
                    else:
                        # In the non-tensor case, fall back to the pointer of the object itself
                        ptrs[id(tensor)].append(name)

                # These are all the pointers of shared tensors.
                shared_ptrs = {ptr: names for ptr, names in ptrs.items() if len(names) > 1}

                for _, names in shared_ptrs.items():
                    # Here we just clone the shared tensors to avoid tensor aliasing which is
                    # not supported in safetensors.
                    for shared_tensor_name in names[1:]:
                        output_state_dict[shared_tensor_name] = output_state_dict[shared_tensor_name].clone()
                if convert_pissa_to_lora is not None:
                    output_state_dict = save_pissa_as_lora(
                        peft_config, convert_pissa_to_lora, output_state_dict, kwargs
                    )
                safe_save_file(
                    output_state_dict,
                    os.path.join(output_dir, SAFETENSORS_WEIGHTS_NAME),
                    metadata={"format": "pt"},
                )
            elif is_main_process:
                output_state_dict = move_all_tensor_to_cpu(output_state_dict, convert=should_write_data)
                if convert_pissa_to_lora is not None:
                    output_state_dict = save_pissa_as_lora(
                        peft_config, convert_pissa_to_lora, output_state_dict, kwargs
                    )
                torch.save(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

            # save the config and change the inference mode to `True`
            if peft_config.base_model_name_or_path is None:
                peft_config.base_model_name_or_path = (
                    self.base_model.__dict__.get("name_or_path", None)
                    if peft_config.is_prompt_learning
                    else self.base_model.model.__dict__.get("name_or_path", None)
                )
            inference_mode = peft_config.inference_mode
            peft_config.inference_mode = True

            if peft_config.task_type is None:
                # deal with auto mapping
                base_model_class = self._get_base_model_class(
                    is_prompt_tuning=peft_config.is_prompt_learning,
                )
                parent_library = base_model_class.__module__

                auto_mapping_dict = {
                    "base_model_class": base_model_class.__name__,
                    "parent_library": parent_library,
                }
            else:
                auto_mapping_dict = None

            if is_main_process:
                if convert_pissa_to_lora is not None:
                    peft_config.init_lora_weights = True
                    peft_config.r *= 2
                    peft_config.lora_alpha *= 2
                peft_config.save_pretrained(output_dir, auto_mapping_dict=auto_mapping_dict)
            peft_config.inference_mode = inference_mode

    def get_nb_trainable_parameters(self) -> Tuple[int, int]:
        return _get_model_param_count(self)


@functools.wraps(orig_get_peft_model)
def get_peft_model(*args, **kwargs):
    peft_model = orig_get_peft_model(*args, **kwargs)
    replace_class_in_inheritance_hierarchy(peft_model, PeftModel, NeuronPeftModel)
    return peft_model
