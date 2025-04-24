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

from typing import Optional

import torch.nn as nn

from ...distributed.utils import parallel_cross_entropy
from ...utils import is_neuronx_distributed_available


if is_neuronx_distributed_available():
    from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_size


_PARALLEL_CROSS_ENTROPY_SHOULD_PRESERVE_INPUT: bool = False


def fixed_cross_entropy(source, target, num_items_in_batch: Optional[int] = None, ignore_index: int = -100, **kwargs):
    reduction = "sum" if num_items_in_batch is not None else "mean"
    tp_size = get_tensor_model_parallel_size()
    if tp_size > 1:
        if _PARALLEL_CROSS_ENTROPY_SHOULD_PRESERVE_INPUT:
            source = source.clone()
        loss_function = parallel_cross_entropy
    else:
        loss_function = nn.functional.cross_entropy
    loss = loss_function(source, target, ignore_index=ignore_index, reduction=reduction)
    if reduction == "sum":
        loss = loss / num_items_in_batch
    return loss


def ForCausalLMLoss(
    logits, labels, vocab_size: int, num_items_in_batch: Optional[int] = None, ignore_index: int = -100, **kwargs
):
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()
    labels = labels.to(logits.device)
    # Shift so that tokens < n predict n
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    shift_logits = shift_logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(shift_logits.device)
    loss = fixed_cross_entropy(shift_logits, shift_labels, num_items_in_batch, ignore_index, **kwargs)
    return loss
