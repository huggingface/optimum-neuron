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

import torch

from ...utils.import_utils import is_peft_available
from ...utils.patching import Patcher


if is_peft_available():
    from peft.utils.save_and_load import get_peft_model_state_dict as orig_get_peft_model_state_dict
else:

    def orig_get_peft_model_state_dict(*args, **kwargs):
        pass


def has_valid_embedding_base_layer(layer):
    """Check if the layer has an embedding base layer"""
    from neuronx_distributed.parallel_layers.layers import BaseParallelLinear, ParallelEmbedding

    return hasattr(layer, "base_layer") and isinstance(
        layer.base_layer, (torch.nn.Linear, torch.nn.Embedding, ParallelEmbedding, BaseParallelLinear)
    )


@functools.wraps(orig_get_peft_model_state_dict)
def get_peft_model_state_dict(*args, **kwargs):
    """Get the state dict of the PEFT model"""
    with Patcher([("peft.utils.save_and_load.has_valid_embedding_base_layer", has_valid_embedding_base_layer)]):
        return orig_get_peft_model_state_dict(*args, **kwargs)
