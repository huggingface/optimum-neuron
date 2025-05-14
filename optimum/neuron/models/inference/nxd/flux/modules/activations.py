# coding=utf-8
# Copyright 2025 HuggingFace Inc.
#
# This implementation is derived from the Diffusers library.
# The original codebase has been optimized and modified to achieve optimal performance
# characteristics when executed on Amazon Neuron devices.
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

"""
Adapted from `neuronx_distributed_inference/models/diffusers/activations.py`.
"""

import torch
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear
from torch import nn as nn
from torch.nn import functional as F


class NeuronGELU(nn.Module):
    r"""
    GELU activation function with tanh approximation support with `approximate="tanh"`.

    Parameters:
        dim_in (`int`): The number of channels in the input.
        dim_out (`int`): The number of channels in the output.
        approximate (`str`, *optional*, defaults to `"none"`): If `"tanh"`, use tanh approximation.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
        self,
        dim_in: int,
        dim_out: int,
        approximate: str = "none",
        bias: bool = True,
        gather_output: bool = True,
        reduce_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.proj = ColumnParallelLinear(
            dim_in, dim_out, bias=bias, gather_output=gather_output, reduce_dtype=reduce_dtype
        )
        self.approximate = approximate

    def gelu(self, gate: torch.Tensor) -> torch.Tensor:
        return F.gelu(gate, approximate=self.approximate)

    def forward(self, hidden_states):
        hidden_states = self.proj(hidden_states)
        hidden_states = self.gelu(hidden_states)
        return hidden_states


class FP32SiLU(nn.Module):
    r"""
    SiLU activation function with input upcasted to torch.float32.
    """

    def __init__(self):
        super().__init__()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return F.silu(inputs.float(), inplace=False).to(inputs.dtype)


ACTIVATION_FUNCTIONS = {
    "swish": nn.SiLU(),
    "silu": nn.SiLU(),
    "mish": nn.Mish(),
    "gelu": nn.GELU(),
    "relu": nn.ReLU(),
}


def get_activation(act_fn: str) -> nn.Module:
    """Helper function to get activation function from string.

    Args:
        act_fn (str): Name of activation function.

    Returns:
        nn.Module: Activation function.
    """

    act_fn = act_fn.lower()
    if act_fn in ACTIVATION_FUNCTIONS:
        return ACTIVATION_FUNCTIONS[act_fn]
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")
