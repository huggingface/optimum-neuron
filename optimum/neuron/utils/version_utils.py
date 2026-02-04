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
"""Version utilities."""

import functools
import re
from importlib import metadata

from packaging import version

from .import_utils import is_neuronx_available


def get_pinned_version(package_name: str) -> str:
    """
    Get the pinned version of a package from the `optimum-neuron` package metadata.

    Args:
        package_name (`str`): The name of the package to get the pinned version for.
    Returns:
        `str`: The pinned version of the package.
    Raises:
        `SystemError`: If there is an error parsing the package metadata.
        `ValueError`: If no pinned version is found for the package.
    """
    requires = metadata.requires("optimum-neuron")
    if requires is None:
        raise SystemError("An error occured while parsing package metadata")
    candidates = [r for r in requires if r.startswith(package_name)]
    if len(candidates) == 1 and f"{package_name}==" in candidates[0]:
        match = re.search(f"{package_name}==([0-9\.]+)", candidates[0])
        if match is not None:
            return match.group(1)
    raise ValueError(f"No pinned version found for package {package_name}")


@functools.cache
def get_neuronxcc_version() -> str:
    try:
        import neuronxcc
    except ImportError:
        raise ModuleNotFoundError("NeuronX Compiler python package is not installed.")
    return neuronxcc.__version__


@functools.cache
def get_torch_xla_version() -> str:
    try:
        import torch_xla
    except ImportError:
        raise ModuleNotFoundError("`torch_xla` python package is not installed.")
    return torch_xla.__version__


@functools.cache
def get_neuronx_distributed_version() -> str:
    try:
        import neuronx_distributed  # noqa: F401
    except ImportError:
        raise ModuleNotFoundError("`neuronx_distributed` python package is not installed.")
    return metadata.version("neuronx_distributed")


@functools.cache
def get_torch_version() -> str:
    try:
        import torch
    except ImportError:
        raise ModuleNotFoundError("`torch` python package is not installed.")
    return torch.__version__


def check_compiler_compatibility(compiler_type: str, compiler_version: str):
    if compiler_type == "neuronx-cc":
        compiler_available_fn = is_neuronx_available
        installed_compiler_version_fn = get_neuronxcc_version
    else:
        raise RuntimeError(f"Pretrained model compiler type {compiler_type} not recognized.")

    if not compiler_available_fn():
        raise RuntimeError(f"Pretrained model was compiled for {compiler_type}, but {compiler_type} is not installed.")

    if version.parse(compiler_version) > version.parse(installed_compiler_version_fn()):
        raise RuntimeError(
            f"Pretrained model is compiled with {compiler_type}({compiler_version}) newer than current compiler ({installed_compiler_version_fn()}),"
            " which may cause runtime incompatibilities."
        )


def check_compiler_compatibility_for_stable_diffusion():
    if not is_neuronx_available():
        raise RuntimeError(
            "Stable diffusion models are supported only on neuronx devices (inf2 / trn1), but neuronx-cc is not installed."
        )
    installed_compiler_version = get_neuronxcc_version()
    if version.parse(installed_compiler_version) < version.parse("2.6"):
        raise RuntimeError(
            f"Stable diffusion models are supported from neuronx-cc 2.6, but you have {installed_compiler_version}, please upgrade it."
        )
