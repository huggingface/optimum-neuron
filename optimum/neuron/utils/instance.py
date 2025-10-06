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
import functools
import os

from .system import get_available_cores


SUPPORTED_INSTANCE_TYPES = ["trn1", "inf2", "trn1n", "trn2"]
INSTANCE_VALUE_MAP = {
    "trn1": "trn1",
    "trn1n": "trn1",
    "inf2": "trn1",
    "trn2": "trn2",
}


@functools.cache
def current_instance_type() -> str:
    if get_available_cores() == 0:
        raise RuntimeError("No Neuron device detected.")
    fpath = "/sys/devices/virtual/dmi/id/product_name"
    try:
        with open(fpath, "r") as f:
            fc = f.readline()
    except IOError:
        raise RuntimeError("Unable to read Neuron platform instance type.")
    instance_type = fc.split(".")[0]
    return normalize_instance_type(instance_type)


def normalize_instance_type(instance_type: str) -> str:
    if instance_type not in SUPPORTED_INSTANCE_TYPES:
        raise ValueError(
            f"{instance_type} is not a valid instance type, supported instance types are: {SUPPORTED_INSTANCE_TYPES}."
        )

    # Normalize instance type
    return INSTANCE_VALUE_MAP[instance_type]


def define_target_instance_type(instance_type: str | None = None) -> str:
    if instance_type is not None:
        return normalize_instance_type(instance_type)
    elif os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE") is not None:
        return normalize_instance_type(os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE"))
    else:
        return current_instance_type()
