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

import os

from torch_neuronx.utils import get_platform_target


SUPPORTED_INSTANCE_TYPES = ["trn1", "inf2", "trn1n", "trn2"]


def get_neuron_instance_type(instance_type: str | None) -> str:
    # Autodetect the platform
    if instance_type is None:
        instance_type = get_platform_target()

    if instance_type not in SUPPORTED_INSTANCE_TYPES:
        raise ValueError(
            f"{instance_type} is not a valid instance type, supported instance types are: {SUPPORTED_INSTANCE_TYPES}."
        )
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = instance_type

    return instance_type


def is_cpu_only_instance():
    instance_type = get_platform_target()
    cpu_only = instance_type not in SUPPORTED_INSTANCE_TYPES
    return cpu_only
