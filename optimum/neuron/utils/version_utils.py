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

from packaging import version

from .import_utils import is_neuronx_available


_neuronxcc_version: str | None = None
_torch_xla_version: str | None = None
_neuronx_distributed_version: str | None = None
_torch_version: str | None = None


def get_neuronxcc_version() -> str:
    global _neuronxcc_version
    if _neuronxcc_version is not None:
        return _neuronxcc_version
    try:
        import neuronxcc
    except ImportError:
        raise ModuleNotFoundError("NeuronX Compiler python package is not installed.")
    _neuronxcc_version = neuronxcc.__version__
    return _neuronxcc_version


def get_torch_xla_version() -> str:
    global _torch_xla_version
    if _torch_xla_version is not None:
        return _torch_xla_version
    try:
        import torch_xla
    except ImportError:
        raise ModuleNotFoundError("`torch_xla` python package is not installed.")
    _torch_xla_version = torch_xla.__version__
    return _torch_xla_version


def get_neuronx_distributed_version() -> str:
    global _neuronx_distributed_version
    if _neuronx_distributed_version is not None:
        return _neuronx_distributed_version
    try:
        import neuronx_distributed
    except ImportError:
        raise ModuleNotFoundError("`neuronx_distributed` python package is not installed.")
    _neuronx_distributed_version = neuronx_distributed.__version__
    return _neuronx_distributed_version


def get_torch_version() -> str:
    global _torch_version
    if _torch_version is not None:
        return _torch_version
    try:
        import torch
    except ImportError:
        raise ModuleNotFoundError("`torch` python package is not installed.")
    _torch_version = torch.__version__
    return _torch_version


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
