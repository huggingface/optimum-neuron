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


SUPPORTED_INSTANCE_TYPES = ["trn1", "inf2", "trn1n", "trn2"]
INSTANCE_VALUE_MAP = {
    "trn1": "trn1",
    "trn1n": "trn1",
    "inf2": "trn1",
    "trn2": "trn2",
}


def auto_detect_platform() -> str:
    fpath = "/sys/devices/virtual/dmi/id/product_name"
    try:
        with open(fpath, "r") as f:
            fc = f.readline()
    except IOError:
        raise RuntimeError(
            "Unable to read platform target. If running on CPU, please supply \
            target instance type, with one of options trn1, inf2, trn1n, or trn2."
        )
    instance_type = fc.split(".")[0]
    return instance_type


def get_neuron_instance_type(instance_type: str | None) -> str:
    # Autodetect the platform
    if instance_type is None:
        instance_type = auto_detect_platform()

    if instance_type not in SUPPORTED_INSTANCE_TYPES:
        raise ValueError(
            f"{instance_type} is not a valid instance type, supported instance types are: {SUPPORTED_INSTANCE_TYPES}."
        )

    # Normalize instance type
    instance_type = INSTANCE_VALUE_MAP[instance_type]
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = instance_type

    return instance_type


def is_cpu_only_instance():
    instance_type = auto_detect_platform()
    cpu_only = instance_type not in SUPPORTED_INSTANCE_TYPES
    return cpu_only
