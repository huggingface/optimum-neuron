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
from typing import TYPE_CHECKING, Dict

import torch

from .require_utils import requires_torch_xla


if TYPE_CHECKING:
    from transformers import PreTrainedModel


_MODEL_TYPE_TO_OPTLEVEL: Dict[str, str] = {
    "default": "-O2",
    "llama": "-O1",
}


@requires_torch_xla
def init_process_group():
    if os.environ.get("TORCHELASTIC_RUN_ID"):
        import torch_xla.distributed.xla_backend as xbn

        if not isinstance(torch.distributed.group.WORLD, xbn.ProcessGroupXla):
            torch.distributed.init_process_group(backend="xla")
            if not isinstance(torch.distributed.group.WORLD, xbn.ProcessGroupXla):
                raise AssertionError("Failed to initialize torch.distributed process group using XLA backend.")


def set_common_neuron_cc_flags():
    """
    Prepares the system environment for Transformers models training on AWS Neuron.
    """
    # Set compiler flag to compile for transformer model type
    os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + " --model-type=transformer"
    # Setting MALLOC_ARENA_MAX is needed because of a memory issue in XLA/glic, otherwise OOM can happen during
    # checkpointing. More information here:
    # https://awsdocs-neuron.readthedocs-hosted.com/en/latest/release-notes/torch/torch-neuronx/index.html#memory-leaking-in-glibc
    os.environ["MALLOC_ARENA_MAX"] = "64"


def set_neuron_cc_flags_for_torch_amp():
    torch.cuda.is_bf16_supported = lambda: True
    neuron_cc_flags = os.environ.get("NEURON_CC_FLAGS", "")
    match_ = re.search(r"--auto-cast\s?\=?\s?\w+", neuron_cc_flags)
    if match_ is not None:
        neuron_cc_flags = neuron_cc_flags[: match_.start(0)] + neuron_cc_flags[match_.end(0) :]
    os.environ["NEURON_CC_FLAGS"] = f"{neuron_cc_flags} --auto-cast=none"


def set_neuron_cc_optlevel_for_model(model: "PreTrainedModel", optlevel: str = "auto"):
    """
    Sets the Neuron compiler optimization level considering both `model` and `optlevel`.
    If `optlevel` is different than `"auto"`, it will be set to that value, otherwise the default value for a given
    model is used.
    """
    if optlevel == "auto":
        optlevel = _MODEL_TYPE_TO_OPTLEVEL.get(model.config.model_type, _MODEL_TYPE_TO_OPTLEVEL["default"])
    neuron_cc_flags = os.environ.get("NEURON_CC_FLAGS", "")
    match_ = re.search(r"-O[123]", neuron_cc_flags)
    if match_:
        neuron_cc_flags = neuron_cc_flags[: match_.start(0)] + f"{optlevel}" + neuron_cc_flags[match_.end(0) + 1 :]
    else:
        neuron_cc_flags += f" {optlevel} "
    os.environ["NEURON_CC_FLAGS"] = neuron_cc_flags


def set_neuron_cc_flags_for_model(model: "PreTrainedModel"):
    """
    Sets flags for the Neuron compiler depending on the model.
    """
    neuron_cc_flags = os.environ.get("NEURON_CC_FLAGS", "")
    if "ForCausalLM" or "ForConditionalGeneration" in model.__class__.__name__:
        distribution_strategy = "--distribution-strategy=llm-training"
        if distribution_strategy not in neuron_cc_flags:
            os.environ["NEURON_CC_FLAGS"] = neuron_cc_flags + f" {distribution_strategy}"
