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

from typing import Callable

import torch
from transformers import PretrainedConfig
from transformers.cache_utils import Cache

from ....utils import logging
from ...utils.training_utils import is_main_worker_for_metrics
from .config import TrainingNeuronConfig
from .pipeline_utils import dynamic_torch_fx_wrap


_LOGGED_WARNING_FLASH_ATTENTION_2 = False

logger = logging.get_logger(__name__)


def create_causal_mask(
    config: PretrainedConfig,
    trn_config: TrainingNeuronConfig,
    input_embeds: torch.Tensor,
    attention_mask: torch.Tensor | None,
    cache_position: torch.Tensor,
    past_key_values: Cache | None = None,
    position_ids: torch.Tensor = None,
    or_mask_function: Callable | None = None,
    and_mask_function: Callable | None = None,
):
    global _LOGGED_WARNING_FLASH_ATTENTION_2
    if past_key_values is not None:
        raise RuntimeError("`past_key_values` is not supported for training in optimum-neuron.")

    if or_mask_function is not None or and_mask_function is not None:
        raise RuntimeError(
            "`or_mask_function` and `and_mask_function` are not supported for training in optimum-neuron."
        )

    if config._attn_implementation == "flash_attention_2":
        if not _LOGGED_WARNING_FLASH_ATTENTION_2 and is_main_worker_for_metrics():
            _LOGGED_WARNING_FLASH_ATTENTION_2 = True
            logger.warning(
                "You are using `flash_attention_2` as attention implementation. "
                "In this case, only a causal mask is supported and it is computed inside the attention operator, the "
                "attention mask passed to the model will be ignored."
            )
        return None

    dtype, device = input_embeds.dtype, input_embeds.device
    if trn_config.sequence_parallel_enabled:
        batch_size = input_embeds.shape[1]
        sequence_length = input_embeds.shape[0] * trn_config.tensor_parallel_size
    else:
        batch_size = input_embeds.shape[0]
        sequence_length = input_embeds.shape[1]

    target_length = attention_mask.shape[-1] if attention_mask is not None else sequence_length + 1

    causal_mask = _compute_causal_mask(
        attention_mask, cache_position, batch_size, sequence_length, target_length, dtype, device
    )
    return causal_mask


@dynamic_torch_fx_wrap
def _compute_causal_mask(attention_mask, cache_position, batch_size, sequence_length, target_length, dtype, device):
    # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        min_dtype = torch.finfo(dtype).min
        causal_mask = torch.full((sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.contiguous()
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(causal_mask.device)
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                padding_mask, min_dtype
            )
    return causal_mask
