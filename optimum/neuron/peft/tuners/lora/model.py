# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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


from ....utils.import_utils import is_peft_available
from .layer import NEURON_LORA_MODULES


if is_peft_available():
    from peft.tuners.lora import LoraModel
else:

    class LoraModel:
        pass


class NeuronLoraModel(LoraModel):
    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False):
        # We experiment with the custom modules feature for LoRA instead of overriding the methods.
        adapter_config = config[adapter_name]
        adapter_config._register_custom_module(NEURON_LORA_MODULES)
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)
