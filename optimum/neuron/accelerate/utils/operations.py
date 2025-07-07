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
"""Custom operations related to accelerate for Neuron."""

import torch
from accelerate.utils.operations import recursively_apply

from ...utils.require_utils import requires_torch_xla


@requires_torch_xla
def _xla_gather(tensor, out_of_graph: bool = False):
    import torch_xla.core.xla_model as xm

    from neuronx_distributed.parallel_layers.parallel_state import (
        get_data_parallel_group,
        model_parallel_is_initialized,
    )

    groups = None
    if model_parallel_is_initialized():
        groups = get_data_parallel_group(as_list=True)

    def _xla_gather_one(tensor):
        if tensor.ndim == 0:
            tensor = tensor.clone()[None]
        # Can only gather contiguous tensors
        if not tensor.is_contiguous():
            tensor = tensor.contiguous()

        if out_of_graph:
            gathered_tensors = xm.mesh_reduce("nested_xla_gather", tensor, lambda x: x)
            if groups is not None:
                new_gathered_tensors = []
                # Since groups is containing list of group of replicas, we consider that visiting the first group of
                # replicas is enough since the value should be the same across other axes.
                replicas_to_consider = set(groups[0])
                for idx, tensor in enumerate(gathered_tensors):
                    if idx not in replicas_to_consider:
                        continue
                    new_gathered_tensors.append(tensor)
                gathered_tensors = new_gathered_tensors
            gathered = torch.cat(gathered_tensors)
        else:
            gathered = xm.all_gather(tensor, groups=groups, pin_layout=False)
        return gathered

    res = recursively_apply(_xla_gather_one, tensor, error_on_other_type=True)
    xm.mark_step()
    return res
