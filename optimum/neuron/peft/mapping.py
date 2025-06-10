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

from typing import Optional

from transformers import PreTrainedModel

from ..utils.import_utils import is_peft_available
from ..utils.patching import Patcher
from .peft_model import NeuronPeftModel, NeuronPeftModelForCausalLM
from .tuners import NeuronLoraModel


if is_peft_available():
    from peft.config import PeftConfig
    from peft.mapping import MODEL_TYPE_TO_PEFT_MODEL_MAPPING, PEFT_TYPE_TO_CONFIG_MAPPING
    from peft.mapping import PEFT_TYPE_TO_TUNER_MAPPING as ORIG_PEFT_TYPE_TO_TUNER_MAPPING
    from peft.mapping import get_peft_model as orig_get_peft_model
    from peft.mixed_model import PeftMixedModel
    from peft.peft_model import PeftModel
    from peft.tuners.tuners_utils import BaseTuner
    from peft.utils import PeftType
else:

    class BaseTuner:
        pass

    class PeftConfig:
        pass

    class PeftMixedModel:
        pass

    class PeftModel:
        pass

    PEFT_TYPE_TO_CONFIG_MAPPING = {}
    MODEL_TYPE_TO_PEFT_MODEL_MAPPING = {}
    ORIG_PEFT_TYPE_TO_TUNER_MAPPING = {}

    def orig_get_peft_model(*args, **kwargs):
        pass

    class PeftType:
        pass


# These mappings should represent the same thing as far as I understand.
# We define both to match what is being done in peft.
PEFT_TYPE_TO_TUNER_MAPPING: dict[str, type[BaseTuner]] = {
    "LORA": NeuronLoraModel,
}
PEFT_TYPE_TO_MODEL_MAPPING: dict[str, type[BaseTuner]] = {
    PeftType.LORA: NeuronLoraModel,
}
MODEL_TYPE_TO_PEFT_MODEL_MAPPING: dict[str, type[NeuronPeftModel]] = {
    "CAUSAL_LM": NeuronPeftModelForCausalLM,
}


def get_peft_model(
    model: PreTrainedModel,
    peft_config: PeftConfig,
    adapter_name: str = "default",
    mixed: bool = False,
    autocast_adapter_dtype: bool = True,
    revision: Optional[str] = None,
    low_cpu_mem_usage: bool = False,
) -> PeftModel | PeftMixedModel:
    if peft_config.peft_type not in PEFT_TYPE_TO_TUNER_MAPPING:
        raise ValueError(
            "PEFT type {peft_config.peft_type} not supported in Optimum Neuron. Supported types are: "
            f"{list(PEFT_TYPE_TO_TUNER_MAPPING.keys())}"
        )

    # The PEFT_TYPE_TO_MODEL_MAPPING is used to map the PEFT type to the model type.
    # Instead of rewriting the whole function, we just patch this dictionary to use our own peft tuners.
    patcher = Patcher(
        [
            ("peft.peft_model.PEFT_TYPE_TO_MODEL_MAPPING", PEFT_TYPE_TO_MODEL_MAPPING),
            ("peft.mapping.MODEL_TYPE_TO_PEFT_MODEL_MAPPING", MODEL_TYPE_TO_PEFT_MODEL_MAPPING),
        ]
    )
    with patcher:
        peft_model = orig_get_peft_model(
            model,
            peft_config,
            adapter_name=adapter_name,
            mixed=mixed,
            autocast_adapter_dtype=autocast_adapter_dtype,
            revision=revision,
            low_cpu_mem_usage=low_cpu_mem_usage,
        )
    return peft_model
