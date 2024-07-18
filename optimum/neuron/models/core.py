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
"""Core functions and classes to adapt models for Neuron."""

import functools
import gc
import math
import os
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Callable, Dict, Optional, Union

import torch
import torch.nn as nn
from transformers.modeling_utils import get_parameter_dtype

from ..utils.patching import Patcher
from ..utils.require_utils import requires_neuronx_distributed, requires_safetensors


if TYPE_CHECKING:
    from transformers import PreTrainedModel


_ORIG_TORCH_FINFO = torch.finfo


def create_patched_finfo(xla_downcast_bf16: bool = False, use_amp: bool = False, xla_use_bf16: bool = False):
    def patched_finfo(dtype):
        if xla_downcast_bf16 or use_amp or xla_use_bf16:
            return _ORIG_TORCH_FINFO(torch.bfloat16)
        return _ORIG_TORCH_FINFO(dtype)

    return patched_finfo


def create_patched_get_parameter_dtype(
    xla_downcast_bf16: bool = False, use_amp: bool = False, xla_use_bf16: bool = False
):
    def patched_get_parameter_dtype(module):
        dtype = get_parameter_dtype(module)
        if xla_downcast_bf16 or use_amp or xla_use_bf16:
            return torch.bfloat16
        return dtype

    return patched_get_parameter_dtype


@requires_neuronx_distributed
@requires_safetensors
def torch_xla_safe_save_file(
    tensors: Dict[str, torch.Tensor],
    filename: Union[str, "os.PathLike"],
    metadata: Optional[Dict[str, str]] = None,
    master_only: bool = True,
    global_master: bool = False,
):
    """
    Torch XLA compatible implementation of `safetensors.torch.save_file`.
    """
    from neuronx_distributed.parallel_layers.utils import move_all_tensor_to_cpu
    from safetensors.torch import save_file
    from torch_xla.core.xla_model import is_master_ordinal

    should_write_data = not master_only or is_master_ordinal(local=not global_master)
    cpu_data = move_all_tensor_to_cpu(tensors, convert=should_write_data)
    if should_write_data:
        save_file(cpu_data, filename, metadata=metadata)


@requires_neuronx_distributed
def create_patched_save_pretrained(orig_save_pretrained_function: Callable[["PreTrainedModel"], None]):
    """
    Creates a wrapper around the `transformers.modeling_utils.PreTrainedModel.save_pretrained` method.
    This methods calls `tensor.data_ptr()` on the model parameters, which causes segmentation fault when the tensors
    are on the XLA device. To prevent that, this wrapper calls `save_pretrained` with the model on the CPU device.
    """
    import torch_xla.core.xla_model as xm
    from neuronx_distributed.parallel_layers.parallel_state import (
        get_data_parallel_rank,
        model_parallel_is_initialized,
    )
    from neuronx_distributed.parallel_layers.utils import move_all_tensor_to_cpu

    orig_self = orig_save_pretrained_function.__self__
    orig_func = orig_save_pretrained_function.__func__

    patcher = Patcher([("transformers.modeling_utils.safe_save_file", torch_xla_safe_save_file)])

    @functools.wraps(orig_func)
    def wrapper(*args, **kwargs):
        self = args[0]
        if model_parallel_is_initialized():
            should_write_data = get_data_parallel_rank() == 0
        else:
            should_write_data = xm.is_master_ordinal(local=True)
        orig_state_dict = self.state_dict()
        cpu_state_dict = move_all_tensor_to_cpu(self.state_dict(), convert=should_write_data)
        self.load_state_dict(cpu_state_dict, assign=True)
        output = None
        if should_write_data:
            with patcher:
                output = orig_func(*args, **kwargs)
        self.load_state_dict(orig_state_dict, assign=True)
        xm.mark_step()
        del cpu_state_dict
        gc.collect()
        return output

    return wrapper.__get__(orig_self)


class PatchedModule(ABC):
    """
    Abstract class that represents a module that is being "patched" or adapted for Neuron.
    """

    @abstractmethod
    def from_original(cls, orig_module: torch.nn.Module, **options) -> "PatchedModule":
        """
        Constructs a `PatchedModule` instance from the original module it is adapting.
        """
        pass


class NeuronAttention(PatchedModule):
    """
    Abstract class that represents a Neuron-adapted version of an attention mechanism.
    It provides getters and setters that are useful to enable / disable Neuron features in the attention computation.
    """

    @property
    def sequence_parallel_enabled(self) -> bool:
        return getattr(self, "_sequence_parallel_enabled", False)

    @sequence_parallel_enabled.setter
    def sequence_parallel_enabled(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("sequence_parallel_enabled must be a boolean value.")
        self._sequence_parallel_enabled = value

    @property
    def flash_attention_enabled(self) -> bool:
        return getattr(self, "_flash_attention_enabled", False)

    @flash_attention_enabled.setter
    def flash_attention_enabled(self, value: bool):
        if not isinstance(value, bool):
            raise ValueError("flash_attention_enabled must be a boolean value.")
        self._flash_attention_enabled = value


class CoreAttention(nn.Module):
    """
    Implements the classical attention computation.
    """

    def forward(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        attention_dropout: float = 0.0,
        attention_mask: Optional[torch.tensor] = None,
    ) -> torch.Tensor:
        bsz, num_heads, q_len, head_dim = query_states.shape
        kv_seq_len = key_states.shape[-2]
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

        if attn_weights.size() != (bsz, num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:  # no matter the length, we just slice it
            # This is the Transformers way of applying the mask.
            causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
            attn_weights = attn_weights + causal_mask
        else:
            # This is the recommended way for Neuron. This way the attention is not passed as an argument
            # avoiding communication that is not needed.
            causal_mask = torch.triu(torch.ones((1, 1, q_len, kv_seq_len), device="xla"), diagonal=1).bool()
            mask_value = torch.finfo(attn_weights.dtype).min
            attn_weights = attn_weights.masked_fill_(causal_mask, mask_value)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.double).to(query_states.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=attention_dropout, training=self.training)

        attn_output = torch.matmul(attn_weights, value_states)
        return attn_output
