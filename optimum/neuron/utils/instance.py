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
import logging
import os

from .system import get_available_cores


logger = logging.getLogger(__name__)

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


def align_compilation_target(target: str, override: bool):
    """A helper to align the NEURON_PLATFORM_TARGET_OVERRIDE environment variable with the target instance type.

    Args:
        target (`str`):
            The target instance type. Must be one of `SUPPORTED_INSTANCE_TYPES`.
        override (`bool`):
            If a different compilation target is set in the environment, it can be overridden.
    """
    target = normalize_instance_type(target)
    env_target = os.environ.get("NEURON_PLATFORM_TARGET_OVERRIDE", None)
    if env_target == target:
        # The compilation target is already correctly set
        return target
    if env_target is not None and not override:
        # Another compilation target is already set and we don't want to override it
        return env_target
    # The compilation target is not set
    if get_available_cores() > 0:
        current_target = current_instance_type()
        if target == current_target:
            # No need to override the compilation target as it matches the current instance type
            return current_target
        elif not override:
            raise ValueError(
                f"The current platform is {current_target} but we are compiling for {target}."
                f" Please set the NEURON_PLATFORM_TARGET_OVERRIDE to {target} or use the optimum-cli"
            )
        logger.info(f"The current instance type is {current_target}, but we are compiling for {target}.")
    else:
        logger.info(f"Setting the compilation target to {target}.")
    os.environ["NEURON_PLATFORM_TARGET_OVERRIDE"] = target
    return target


def define_instance_type_with_default_value(instance_type: str | None = None):
    if instance_type is None:
        if get_available_cores() == 0:
            instance_type = "trn1"
            logger.info(f"No Neuron device detected, we are compiling for {instance_type}.")
        else:
            instance_type = current_instance_type()
    else:
        instance_type = normalize_instance_type(instance_type)

    return instance_type
