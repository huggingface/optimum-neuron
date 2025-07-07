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
# Adapted from https://github.com/aws-neuron/neuronx-distributed-inference/blob/9993358ce052fd7a1bb4a7497a6318aac36ed95c/src/neuronx_distributed_inference/modules/kvcache/utils.py

import torch
from torch_neuronx.xla_impl.ops import xla_hlo_call


@xla_hlo_call
def fill_prefix(tensor, update):
    scribe = tensor.scribe
    dtype = tensor.dtype
    shape = tensor.sizes
    start_indices = [scribe.u32.Constant(constant_value=0)] * len(shape)
    return dtype[shape].DynamicUpdateSlice(tensor, update, *start_indices)


def dynamic_update_slice(tensor: torch.Tensor, update: torch.Tensor, start_indices: list[torch.Tensor]):
    """
    Directly invoke DynamicUpdateSlice XLA op
    https://openxla.org/xla/operation_semantics#dynamicupdateslice
    """

    @xla_hlo_call
    def xla_dynamic_update_slice(tensor, update, *start_indices):
        dtype = tensor.dtype
        shape = tensor.sizes
        return dtype[shape].DynamicUpdateSlice(tensor, update, *start_indices)

    assert len(start_indices) == tensor.dim(), "not enough indices to index into tensor"
    return xla_dynamic_update_slice(tensor, update, *start_indices)
