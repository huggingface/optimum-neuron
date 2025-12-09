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
# Adapted from https://github.com/aws-neuron/neuronx-distributed-inference/blob/9993358ce052fd7a1bb4a7497a6318aac36ed95c/src/neuronx_distributed_inference/models/config.py

import torch

from ....configuration_utils import NeuronConfig, register_neuron_config
from ....utils import DTYPE_MAPPER


NEURON_CONFIG_FILE = "neuron_config.json"


def to_dict(obj):
    if type(obj) is dict:
        return {k: to_dict(v) for k, v in obj.items()}
    elif type(obj) is list:
        return [to_dict(v) for v in obj]
    elif hasattr(obj, "__dict__"):
        return {k: to_dict(v) for k, v in obj.__dict__.items()}
    elif type(obj) is torch.dtype:
        return str(obj).split(".")[1]
    else:
        return obj


class IncompatibleConfigError(ValueError):
    pass


@register_neuron_config
class NxDNeuronConfig(NeuronConfig):
    """
    Base config class for inference in NxD.

    This class contains attributes that are needed for various inference
    optimization/features in NxD.

    These attributes are a subset of the attributes in the original NxDI config class.
    """

    def __init__(
        self,
        checkpoint_id: str = None,
        checkpoint_revision: str = None,
        batch_size: int | None = 1,
        max_batch_size: int | None = None,
        continuous_batching: bool | None = False,
        speculation_length: int | None = 0,
        sequence_length: int | None = 128,
        tp_degree: int | None = 1,
        ep_degree: int | None = 1,
        pp_degree: int | None = 1,
        sequence_parallel_enabled: bool | None = False,
        torch_dtype: str | torch.dtype | None = torch.bfloat16,
        n_active_tokens: int | None = None,
        max_context_length: int | None = None,
        output_logits: bool | None = False,
        fused_qkv: bool | None = False,
        target: str | None = None,  # set to "trn2" for trn2
        on_device_sampling: bool | None = False,
        max_topk: int | None = 256,
        start_rank_id: int | None = 0,
        local_ranks_size: int | None = None,
        capacity_factor: float = None,
        glu_mlp: bool = True,
    ) -> None:
        # Required to retrieve a checkpoint from the hub
        self.checkpoint_id = checkpoint_id
        self.checkpoint_revision = checkpoint_revision
        # Basic config for inference in NxD
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.tp_degree = tp_degree
        self.torch_dtype = torch_dtype
        if isinstance(self.torch_dtype, str):
            self.torch_dtype = DTYPE_MAPPER.pt(self.torch_dtype)
        self.n_active_tokens = self.sequence_length if n_active_tokens is None else n_active_tokens
        self.output_logits = output_logits

        # fallback to sequence_length is for compatibility with vllm
        self.max_context_length = max_context_length
        if self.max_context_length is None:
            self.max_context_length = sequence_length

        # Graph transforms
        self.fused_qkv = fused_qkv

        # Continuous batching
        self.continuous_batching = continuous_batching
        self.max_batch_size = batch_size if max_batch_size is None else max_batch_size

        # On-device sampling
        self.on_device_sampling = on_device_sampling
        self.max_topk = max_topk

        # Speculative decoding
        self.speculation_length = speculation_length
        if self.speculation_length > 0:
            if self.on_device_sampling:
                raise IncompatibleConfigError("Speculative decoding is incompatible with on-device sampling")

        # Distributed config
        self.pp_degree = pp_degree
        self.ep_degree = ep_degree
        self.sequence_parallel_enabled = sequence_parallel_enabled

        # Multi-node
        # TODO: Check if start_rank_id can be modified dynamically at runtime
        # Otherwise, we need multiple exports for different start_rank_id
        self.start_rank_id = start_rank_id
        self.local_ranks_size = local_ranks_size
        if self.local_ranks_size is None:
            self.local_ranks_size = self.world_size

        self.target = target

        # MoE specific
        self.capacity_factor = float(capacity_factor) if capacity_factor is not None else None
        self.glu_mlp = glu_mlp

    @property
    def ctx_batch_size(self) -> int:
        return 1 if self.continuous_batching else self.batch_size

    @property
    def tkg_batch_size(self) -> int:
        return self.batch_size

    @property
    def world_size(self) -> int:
        """
        The total number of ranks in the distributed setup.
        """
        return self.tp_degree * self.pp_degree * self.ep_degree

    @property
    def weights_to_skip_layout_optimization(self) -> list[str]:
        """
        list of weights to skip layout optimization.

        Can be overridden by subclasses to specify weights that should not be optimized.
        """
        return []

    @property
    def logical_nc_config(self) -> int:
        return 2 if self.target == "trn2" else 1
