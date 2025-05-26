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

from .granite.modeling_granite import GraniteForCausalLM
from .llama.modeling_llama import LlamaForCausalLM
from .modeling_utils import ALL_ATTENTION_FUNCTIONS, NeuronModelMixin, NotSupportedError
from .transformations_utils import (
    CustomModule,
    FusedLinearsSpec,
    GQAQKVColumnParallelLinearSpec,
    ModelWeightTransformationSpec,
    ModelWeightTransformationSpecs,
    adapt_state_dict,
    adapt_peft_config_for_model,
    create_parameter_metadata,
    to_original_weights,
)
