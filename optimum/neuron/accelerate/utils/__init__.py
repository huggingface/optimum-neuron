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

from .dataclasses import MixedPrecisionConfig, MixedPrecisionMode
from .misc import patch_accelerate_is_torch_xla_available
from .operations import (
    broadcast_object,
    broadcast_object_to_data_parallel_group,
    broadcast_object_to_pipeline_model_parallel_group,
    broadcast_object_to_tensor_model_parallel_group,
    gather_object,
    gather_object_from_data_parallel_group,
    gather_object_from_pipeline_model_parallel_group,
    gather_object_from_tensor_model_parallel_group,
)
