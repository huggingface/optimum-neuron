# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Utilities related to initialization of `torch_xla` and `torch_neuronx`"""

import os
import re

import torch
import torch_xla.distributed.xla_backend as xbn

from optimum.utils import logging

from ..cache.training import patch_neuron_cc_wrapper


logger = logging.get_logger()


def init_process_group():
    if os.environ.get("TORCHELASTIC_RUN_ID"):
        if not isinstance(torch.distributed.group.WORLD, xbn.ProcessGroupXla):
            torch.distributed.init_process_group(backend="xla")
            if not isinstance(torch.distributed.group.WORLD, xbn.ProcessGroupXla):
                raise AssertionError("Failed to initialize torch.distributed process group using XLA backend.")


def set_neuron_cc_flag(flag_name: str, flag_value: str):
    """
    Sets a specific Neuron compiler flag in the `NEURON_CC_FLAGS` environment variable.
    If the flag is already present, its value is updated.

    Args:
        flag_name (`str`):
            The name of the flag to set (e.g., `--auto-cast`).
        flag_value (`str`):
            The value to set for the flag (e.g., `none`, `bf16`, etc.).
    """
    neuron_cc_flags = os.environ.get("NEURON_CC_FLAGS", "")
    match_ = re.search(rf"{re.escape(flag_name)}\s?\=?\s?\w+", neuron_cc_flags)
    if match_ is not None:
        neuron_cc_flags = (
            neuron_cc_flags[: match_.start(0)] + f"{flag_name}={flag_value}" + neuron_cc_flags[match_.end(0) :]
        )
    else:
        neuron_cc_flags += f" {flag_name}={flag_value}"
    os.environ["NEURON_CC_FLAGS"] = neuron_cc_flags


def set_common_flags():
    """
    Sets environment variables for transformer-based models training with AWS Neuron.
    """
    model_type = os.environ.get("OPTIMUM_NEURON_COMMON_FLAGS_MODEL_TYPE", "")
    if model_type != "":
        os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + f" --model-type={model_type}"
    # Setting MALLOC_ARENA_MAX is needed because of a memory issue in XLA/glic, otherwise OOM can happen during
    # checkpointing. More information here:
    # https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/torch/torch-neuronx/index.html#memory-leaking-in-glibc
    os.environ["MALLOC_ARENA_MAX"] = "64"
    # Setting the path to use our patched version of the `neuron_cc_wrapper`.
    patch_neuron_cc_wrapper(restore_path=False).__enter__()
