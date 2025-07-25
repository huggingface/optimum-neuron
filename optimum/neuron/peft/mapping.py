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


from ..utils.import_utils import is_peft_available
from .peft_model import NeuronPeftModel, NeuronPeftModelForCausalLM
from .tuners import NeuronLoraModel


if is_peft_available():
    from peft.tuners.tuners_utils import BaseTuner
else:

    class BaseTuner:
        pass


PEFT_TYPE_TO_TUNER_MAPPING: dict[str, type[BaseTuner]] = {
    "LORA": NeuronLoraModel,
}
MODEL_TYPE_TO_PEFT_MODEL_MAPPING: dict[str, type[NeuronPeftModel]] = {
    "CAUSAL_LM": NeuronPeftModelForCausalLM,
}
