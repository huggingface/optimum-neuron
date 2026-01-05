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
Adapted from `neuronx_distributed_inference/models/diffusers/normalization.py`.
"""

from typing import Optional, Tuple

import torch
from neuronx_distributed.parallel_layers.layer_norm import LayerNorm
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear  # noqa: E402
from torch import nn as nn

from ....backend.utils.layer_boundary_marker import ModuleMarkerEndWrapper, ModuleMarkerStartWrapper
from .embeddings import (
    NeuronCombinedTimestepLabelEmbeddings,
)


class NeuronAdaLayerNormZeroSingle(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(
        self,
        embedding_dim: int,
        norm_type="layer_norm",
        bias=True,
        use_parallel_layer=True,
        reduce_dtype=torch.bfloat16,
    ):
        super().__init__()

        self.silu = nn.SiLU()
        self.linear = (
            ColumnParallelLinear(
                embedding_dim,
                3 * embedding_dim,
                bias=bias,
                gather_output=True,
                reduce_dtype=reduce_dtype,
            )
            if use_parallel_layer
            else nn.Linear(embedding_dim, 3 * embedding_dim, bias=bias)
        )
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        else:
            raise ValueError(f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm'.")

    def forward(
        self,
        x: torch.Tensor,
        emb: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        emb = self.linear(self.silu(emb))
        shift_msa, scale_msa, gate_msa = emb.chunk(3, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa


class NeuronAdaLayerNormZero(nn.Module):
    r"""
    Norm layer adaptive layer norm zero (adaLN-Zero).

    Parameters:
        embedding_dim (`int`): The size of each embedding vector.
        num_embeddings (`int`): The size of the embeddings dictionary.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_embeddings: Optional[int] = None,
        norm_type="layer_norm",
        bias=True,
        use_parallel_layer=True,
        reduce_dtype=torch.bfloat16,
    ):
        super().__init__()
        if num_embeddings is not None:
            self.emb = NeuronCombinedTimestepLabelEmbeddings(num_embeddings, embedding_dim)
        else:
            self.emb = None

        self.silu = nn.SiLU()
        self.linear = (
            ColumnParallelLinear(
                embedding_dim,
                6 * embedding_dim,
                bias=bias,
                gather_output=True,
                reduce_dtype=reduce_dtype,
            )
            if use_parallel_layer
            else nn.Linear(embedding_dim, 6 * embedding_dim, bias=bias)
        )
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, elementwise_affine=False, eps=1e-6)
        else:
            raise ValueError(f"Unsupported `norm_type` ({norm_type}) provided. Supported ones are: 'layer_norm'.")

    def forward(
        self,
        x: torch.Tensor,
        timestep: Optional[torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        hidden_dtype: Optional[torch.dtype] = None,
        emb: Optional[torch.Tensor] = None,
        hlomarker: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if self.emb is not None:
            emb = self.emb(timestep, class_labels, hidden_dtype=hidden_dtype)
        emb = self.linear(self.silu(emb))
        if hlomarker:
            emb = ModuleMarkerEndWrapper()(emb)
            x, emb = ModuleMarkerStartWrapper()(x, emb)
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = emb.chunk(6, dim=1)
        x = self.norm(x) * (1 + scale_msa[:, None]) + shift_msa[:, None]
        return x, gate_msa, shift_mlp, scale_mlp, gate_mlp


class NeuronAdaLayerNormContinuous(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        conditioning_embedding_dim: int,
        # NOTE: It is a bit weird that the norm layer can be configured to have scale and shift parameters
        # because the output is immediately scaled and shifted by the projected conditioning embeddings.
        # Note that AdaLayerNorm does not let the norm layer have scale and shift parameters.
        # However, this is how it was implemented in the original code, and it's rather likely you should
        # set `elementwise_affine` to False.
        elementwise_affine=True,
        eps=1e-5,
        bias=True,
        norm_type="layer_norm",
        use_parallel_layer=True,
        reduce_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.silu = nn.SiLU()
        self.linear = (
            ColumnParallelLinear(
                conditioning_embedding_dim,
                embedding_dim * 2,
                bias=bias,
                gather_output=True,
                reduce_dtype=reduce_dtype,
            )
            if use_parallel_layer
            else nn.Linear(conditioning_embedding_dim, embedding_dim * 2, bias=bias)
        )
        if norm_type == "layer_norm":
            self.norm = LayerNorm(embedding_dim, eps, elementwise_affine, bias)
        else:
            raise ValueError(f"unknown norm_type {norm_type}")

    def forward(self, x: torch.Tensor, conditioning_embedding: torch.Tensor) -> torch.Tensor:
        # convert back to the original dtype in case `conditioning_embedding`` is upcasted to float32 (needed for hunyuanDiT)
        emb = self.linear(self.silu(conditioning_embedding).to(x.dtype))
        scale, shift = torch.chunk(emb, 2, dim=1)
        x = self.norm(x) * (1 + scale)[:, None, :] + shift[:, None, :]
        return x
