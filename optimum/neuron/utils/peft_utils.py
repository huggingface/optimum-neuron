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
import functools
import gc
from typing import Any, List, Optional, Union

from transformers.utils import is_peft_available

from .patching import replace_class_in_inheritance_hierarchy
from .require_utils import requires_neuronx_distributed


if is_peft_available():
    from peft import PeftModel
    from peft import get_peft_model as orig_get_peft_model
    from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict

else:

    class PeftModel:
        pass

    def orig_get_peft_model(*args, **kwargs):
        pass

    def get_peft_model_state_dict(*args, **kwargs):
        pass

    def set_peft_model_state_dict(*args, **kwargs):
        pass


class NeuronPeftModel(PeftModel):
    @requires_neuronx_distributed
    def save_pretrained(
        self,
        save_directory: str,
        safe_serialization: bool = True,
        selected_adapters: Optional[List[str]] = None,
        save_embedding_layers: Union[str, bool] = "auto",
        is_main_process: bool = True,
        convert_pissa_to_lora: Optional[str] = None,
        **kwargs: Any,
    ):
        import torch_xla.core.xla_model as xm
        from neuronx_distributed.parallel_layers.parallel_state import (
            get_data_parallel_rank,
            model_parallel_is_initialized,
        )
        from neuronx_distributed.parallel_layers.utils import move_all_tensor_to_cpu

        if model_parallel_is_initialized():
            should_write_data = get_data_parallel_rank() == 0
        else:
            should_write_data = xm.is_master_ordinal(local=True)

        if selected_adapters is None:
            selected_adapters = list(self.peft_config.keys())

        orig_state_dicts = {}
        cpu_state_dicts = {}
        for adapter_name in selected_adapters:
            state_dict = get_peft_model_state_dict(
                self,
                state_dict=kwargs.get("state_dict", None),
                adapter_name=adapter_name,
                save_embedding_layers=save_embedding_layers,
            )
            cpu_state_dict = move_all_tensor_to_cpu(state_dict, convert=should_write_data)
            orig_state_dicts[adapter_name] = state_dict
            cpu_state_dicts[adapter_name] = cpu_state_dict

        for adapter_name, state_dict in cpu_state_dicts.items():
            set_peft_model_state_dict(self, state_dict, adapter_name=adapter_name)

        output = None
        if should_write_data:
            output = super().save_pretrained(
                save_directory,
                safe_serialization=safe_serialization,
                selected_adapters=selected_adapters,
                save_embedding_layers=save_embedding_layers,
                is_main_process=is_main_process,
                convert_pissa_to_lora=convert_pissa_to_lora,
            )

        for adapter_name, state_dict in orig_state_dicts.items():
            set_peft_model_state_dict(self, state_dict, adapter_name=adapter_name)

        xm.mark_step()
        del cpu_state_dicts
        gc.collect()
        return output


@functools.wraps(orig_get_peft_model)
def get_peft_model(*args, **kwargs):
    peft_model = orig_get_peft_model(*args, **kwargs)
    replace_class_in_inheritance_hierarchy(peft_model, PeftModel, NeuronPeftModel)
    return peft_model
