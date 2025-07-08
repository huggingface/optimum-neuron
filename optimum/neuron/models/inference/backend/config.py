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
from ....utils import map_torch_dtype


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
        torch_dtype: str | torch.dtype | None = torch.bfloat16,
        rpl_reduce_dtype: str | torch.dtype | None = None,
        n_active_tokens: int | None = None,
        max_context_length: int | None = None,
        output_logits: bool | None = False,
        padding_side: str | None = "right",
        fused_qkv: bool | None = False,
        vocab_parallel: bool | None = False,
        sequence_parallel_enabled: bool | None = False,
        is_chunked_prefill: bool | None = False,
        flash_decoding_enabled: bool | None = False,
        async_mode: bool | None = False,
        qk_layernorm: bool | None = False,
        attn_kernel_enabled: bool | None = False,
        qkv_kernel_enabled: bool | None = False,
        mlp_kernel_enabled: bool | None = False,
        mlp_kernel_fuse_residual_add: bool | None = False,
        enable_bucketing: bool | None = False,
        target: str | None = None,  # set to "trn2" for trn2
        logical_nc_config: int | None = 1,
        cc_pipeline_tiling_factor: int | None = 2,
        num_cores_per_group: int | None = 1,
        on_device_sampling: bool | None = False,
        max_topk: int | None = 256,
        start_rank_id: int | None = 0,
        local_ranks_size: int | None = None,
        capacity_factor: float = None,
        glu_mlp: bool = True,
    ) -> None:
        # TODO: these flags are suposed to work in NxDI. Either make them work or remove them
        if is_chunked_prefill:
            raise ValueError("`is_chunked_prefill` is not supported in optimum-neuron.")
        if flash_decoding_enabled:
            raise ValueError("`flash_decoding_enabled` is not supported in optimum-neuron.")
        if async_mode:
            raise ValueError("`async_mode` is not supported in optimum-neuron.")
        if qkv_kernel_enabled or mlp_kernel_enabled:
            raise ValueError("`qkv_kernel_enabled` and `mlp_kernel_enabled` are not supported for trn1 chips.")
        if vocab_parallel:
            raise ValueError("`vocab_parallel` is not supported in optimum-neuron.")
        if qk_layernorm:
            raise ValueError(
                "`qk_layernorm` is not supported in optimum-neuron. It is actually a modeling flag that affects the attention layer."
            )
        # Required to retrieve a checkpoint from the hub
        self.checkpoint_id = checkpoint_id
        self.checkpoint_revision = checkpoint_revision
        # Basic config for inference in NxD
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.tp_degree = tp_degree
        self.torch_dtype = torch_dtype
        if isinstance(self.torch_dtype, str):
            self.torch_dtype = map_torch_dtype(self.torch_dtype)
        self.n_active_tokens = self.sequence_length if n_active_tokens is None else n_active_tokens
        self.output_logits = output_logits

        self.padding_side = padding_side

        self.rpl_reduce_dtype = torch_dtype if rpl_reduce_dtype is None else rpl_reduce_dtype
        if isinstance(self.rpl_reduce_dtype, str):
            self.rpl_reduce_dtype = map_torch_dtype(self.rpl_reduce_dtype)

        # fallback to sequence_length is for compatibility with vllm
        self.max_context_length = max_context_length
        if self.max_context_length is None:
            self.max_context_length = sequence_length

        # Graph transforms
        self.fused_qkv = fused_qkv

        # Functional parallelism
        self.vocab_parallel = vocab_parallel
        self.sequence_parallel_enabled = sequence_parallel_enabled
        self.is_chunked_prefill = is_chunked_prefill

        # Continuous batching
        # TODO: Check if we really need different batch size for CTE and TKG, given
        # that we anyway provide two different config instance for them.
        self.continuous_batching = continuous_batching
        self.max_batch_size = batch_size if max_batch_size is None else max_batch_size

        # On-device sampling
        self.on_device_sampling = on_device_sampling
        self.max_topk = max_topk

        # async
        self.async_mode = async_mode

        # Bucketing
        self.enable_bucketing = enable_bucketing

        # Speculative decoding
        self.speculation_length = speculation_length
        if self.speculation_length > 0:
            if self.async_mode:
                raise IncompatibleConfigError("Speculative Decoding is not yet supported with async.")
            if self.on_device_sampling:
                raise IncompatibleConfigError("Speculative decoding is incompatible with on-device sampling")

        # Distributed config
        self.pp_degree = pp_degree
        self.ep_degree = ep_degree

        # QK layer normalization
        self.qk_layernorm = qk_layernorm

        # Multi-node
        # TODO: Check if start_rank_id can be modified dynamically at runtime
        # Otherwise, we need multiple exports for different start_rank_id
        self.start_rank_id = start_rank_id
        self.local_ranks_size = local_ranks_size
        if self.local_ranks_size is None:
            self.local_ranks_size = self.world_size

        # Flash decoding
        self.flash_decoding_enabled = flash_decoding_enabled
        self.num_cores_per_group = num_cores_per_group

        # Kernels
        self.attn_kernel_enabled = attn_kernel_enabled
        self.qkv_kernel_enabled = qkv_kernel_enabled
        self.mlp_kernel_enabled = mlp_kernel_enabled
        self.mlp_kernel_fuse_residual_add = mlp_kernel_fuse_residual_add

        # compiler flags
        self.logical_nc_config = logical_nc_config
        self.cc_pipeline_tiling_factor = cc_pipeline_tiling_factor
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
