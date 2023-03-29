# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

from .argument_utils import convert_neuronx_compiler_args_to_neuron
from .import_utils import is_neuron_available, is_neuronx_available
from .training_utils import (
    FirstAndLastDataset,
    Patcher,
    is_model_officially_supported,
    is_precompilation,
    patch_forward,
    patch_model,
    patch_transformers_for_neuron_sdk,
    patched_finfo,
    prepare_environment_for_neuron,
)
