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

"""
Adapted from `neuronx_distributed_inference/models/diffusers/flux/modeling_flux.py`.
"""

import logging
import math
import os
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from neuronx_distributed.parallel_layers.layer_norm import LayerNorm
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.mappings import reduce_from_tensor_model_parallel_region
from diffusers.models.transformers.transformer_qwenimage import QwenEmbedRope

from ..backend.modules.diffusion.embeddings import (
    FluxPosEmbed,
    NeuronCombinedTimestepGuidanceTextProjEmbeddings,
    NeuronCombinedTimestepTextProjEmbeddings,
    apply_rotary_emb as apply_rotary_emb_qwen,
    NeuronQwenTimestepProjEmbeddings,
)
from ..backend.modules.diffusion.normalization import (
    NeuronAdaLayerNormContinuous,
    NeuronAdaLayerNormZero,
    NeuronAdaLayerNormZeroSingle,
)
from ..backend.modules.diffusion.attention import NeuronAttention, NeuronFeedForward


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class NeuronQwenEmbedRope(nn.Module):
    pass


class NeuronQwenDoubleStreamAttnProcessor2_0:
    pass


class NeuronQwenImageTransformer2DModel(torch.nn.Module):
    pass


class NeuronQwenImageTransformerBlock(nn.Module):
    pass
