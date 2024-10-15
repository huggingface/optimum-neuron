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
import math
import os
import warnings
from dataclasses import asdict
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.utils import is_peft_available

from .patching import Patcher, replace_class_in_inheritance_hierarchy
from .require_utils import requires_neuronx_distributed, requires_safetensors
from .training_utils import _get_model_param_count


if is_peft_available():
    from peft import PeftModel
    from peft import get_peft_model as orig_get_peft_model
    from peft.tuners.lora import LoraLayer
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

    class LoraLayer:
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
                    os.makedirs(adapter_shards_dir_model)

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


class LoraGQAQKVParallelLinear(torch.nn.Module, LoraLayer):
    r"""
    When the target layer parallel_linear is GQAQKVColumnParallelLinear, in order to keep the input and output shapes
    consistent, we perform column segmentation on lora_B, while lora_A is still a complete linear layer.
    """

    def __init__(
        self,
        base_layer: torch.nn.Module,
        adapter_name: str,
        r: int = 0,
        lora_alpha: int = 1,
        lora_dropout: float = 0.0,
        fan_in_fan_out: bool = False,  # Set this to True if the layer to replace stores weight like (fan_in, fan_out)
        is_target_conv_1d_layer: bool = False,
        init_lora_weights: Union[bool, str] = True,
        use_rslora: bool = False,
        use_dora: bool = False,
        **kwargs,
    ):
        super().__init__()
        LoraLayer.__init__(self, base_layer, **kwargs)
        self.fan_in_fan_out = fan_in_fan_out

        self._active_adapter = adapter_name
        self.update_layer(
            adapter_name,
            r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            init_lora_weights=init_lora_weights,
            use_rslora=use_rslora,
            use_dora=use_dora,
        )

    @requires_neuronx_distributed
    def update_layer(self, adapter_name, r, lora_alpha, lora_dropout, init_lora_weights, use_rslora, use_dora):

        base_layer = self.get_base_layer()

        self.r[adapter_name] = r
        self.lora_alpha[adapter_name] = lora_alpha
        if lora_dropout > 0.0:
            lora_dropout_layer = nn.Dropout(p=lora_dropout)
        else:
            lora_dropout_layer = nn.Identity()

        self.lora_dropout[adapter_name] = lora_dropout_layer

        # GQAQKVColumnParallelLinear specific
        self.num_attention_heads = base_layer.num_attention_heads
        self.num_key_value_heads = base_layer.num_key_value_heads
        self.sequence_parallel_enabled = base_layer.sequence_parallel_enabled
        self.kv_size_multiplier = base_layer.kv_size_multiplier
        self.gather_output = base_layer.gather_output

        self.lora_A[adapter_name] = nn.Linear(
            in_features=self.in_features, out_features=r, bias=False, dtype=torch.float32
        )
        from ..distributed.utils import OptimumGQAQKVColumnParallelLinear

        self.lora_B[adapter_name] = OptimumGQAQKVColumnParallelLinear(
            base_layer.query_proj_name,
            base_layer.key_proj_name,
            base_layer.value_proj_name,
            base_layer.output_proj_name,
            base_layer.num_attention_heads,
            base_layer.num_key_value_heads,
            input_size=r,
            output_sizes=self.out_features,
            bias=False,
            gather_output=self.gather_output,
            dtype=torch.float32,
            # TODO: add init method
            # init_method=init_method,
            kv_size_multiplier=self.kv_size_multiplier,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
        )

        if use_rslora:
            self.scaling[adapter_name] = lora_alpha / math.sqrt(r)
        else:
            self.scaling[adapter_name] = lora_alpha / r

        # TODO: add support for more initialization method just as in here:
        # https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py#L128-L139
        if init_lora_weights:
            init_lora_weights = "default"
        self.init_lora_parameters(adapter_name, init_lora_weights)

        if use_dora:
            self.dora_init(adapter_name)
            self.use_dora[adapter_name] = True
        else:
            self.use_dora[adapter_name] = False

        self.set_adapter(self.active_adapters)

    def init_lora_parameters(self, adapter_name, init_lora_weights):
        init_lora_weights = init_lora_weights.lower()
        assert init_lora_weights in ["default", "gaussian"]

        if init_lora_weights == "default":
            # initialize A the same way as the default for nn.Linear and B to zero
            # https://github.com/microsoft/LoRA/blob/a0a92e0f26c067cf94747bdbf1ce73793fa44d19/loralib/layers.py#L124
            nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
        elif init_lora_weights == "gaussian":
            nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.lora_rank)
        else:
            raise ValueError(f"Unknown LoRA parameters initialization with {init_lora_weights}")

        q, k, v = self.get_qkv(self.lora_B[adapter_name])
        nn.init.zeros_(q.data)
        nn.init.zeros_(k.data)
        nn.init.zeros_(v.data)

    def merge(self) -> None:
        """
        Merge the adapter weights into the base weights
        """
        weight_q, weight_k, weight_v = self.get_qkv(self.base_layer)
        delta_weight_q, delta_weight_k, delta_weight_v = self.get_delta_weight()

        weight_q.data += delta_weight_q
        weight_k.data += delta_weight_k
        weight_v.data += delta_weight_v
        self.merged = True

    def unmerge(self) -> None:
        """
        This method unmerges merged adapter layers from the base weights.
        """
        if not self.merged:
            warnings.warn("Already unmerged. Nothing to do.")
            return

        q, k, v = self.get_qkv(self.base_layer)
        delta_weight_q, delta_weight_k, delta_weight_v = self.get_delta_weight()

        q.data -= delta_weight_q
        k.data -= delta_weight_k
        v.data -= delta_weight_v
        self.merged = False

    def get_qkv(self, layer):
        return layer.weight_q, layer.weight_k, layer.weight_v

    def get_delta_weight(self) -> torch.Tensor:
        weight_A = self.lora_A.weight
        q_lora_B, k_lora_B, v_lora_B = self.get_qkv(self.lora_B)

        output_q = (q_lora_B @ weight_A) * self.scaling
        output_k = (k_lora_B @ weight_A) * self.scaling
        output_v = (v_lora_B @ weight_A) * self.scaling

        return output_q, output_k, output_v

    def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        self._check_forward_args(x, *args, **kwargs)
        adapter_names = kwargs.pop("adapter_names", None)

        previous_dtype = x.dtype
        if self.disable_adapters:
            if self.merged:
                self.unmerge()
            output_q, output_k, output_v = self.base_layer(x, *args, **kwargs)
        elif adapter_names is not None:
            raise RuntimeError("Providing specific adapter names is not supported yet.")
        elif self.merged:
            output_q, output_k, output_v = self.base_layer(x, *args, **kwargs)
        else:
            output_q, output_k, output_v = self.base_layer(x, *args, **kwargs)
            for active_adapter in self.active_adapters:
                if active_adapter not in self.lora_A.keys():
                    continue
                lora_A = self.lora_A[active_adapter]
                dropout = self.lora_dropout[active_adapter]
                scaling = self.scaling[active_adapter]
                x = x.to(lora_A.weight.dtype)

                q_lora_B, k_lora_B, v_lora_B = self.get_qkv(self.lora_B[active_adapter])
                dropout_input = lora_A(dropout(x))
                lora_q_output, lora_k_output, lora_v_output = self._lora_forward(
                    dropout_input, q_lora_B, k_lora_B, v_lora_B
                )

                output_q += lora_q_output * scaling
                output_k += lora_k_output * scaling
                output_v += lora_v_output * scaling

        return output_q.to(previous_dtype), output_k.to(previous_dtype), output_v.to(previous_dtype)

    @requires_neuronx_distributed
    def _lora_forward(self, input, weight_q, weight_k, weight_v):
        # Matrix multiply.
        from neuronx_distributed.modules.qkv_linear import gqa_qkv_linear_with_async_allreduce

        output_parallel_q, output_parallel_k, output_parallel_v = gqa_qkv_linear_with_async_allreduce(
            input=input,
            weight_q=weight_q,
            weight_k=weight_k,
            weight_v=weight_v,
            bias_q=None,
            bias_k=None,
            bias_v=None,
            async_grad_allreduce=not self.sequence_parallel_enabled,
            sequence_parallel_enabled=self.sequence_parallel_enabled,
            kv_size_multiplier=self.kv_size_multiplier,
        )
        if self.gather_output:
            # All-gather across the partitions.
            assert not self.sequence_parallel_enabled
            output_q = gather_from_tensor_model_parallel_region(output_parallel_q)
            output_k = gather_from_tensor_model_parallel_region(output_parallel_k)
            output_v = gather_from_tensor_model_parallel_region(output_parallel_v)
        else:
            output_q, output_k, output_v = output_parallel_q, output_parallel_k, output_parallel_v
        return output_q, output_k, output_v
