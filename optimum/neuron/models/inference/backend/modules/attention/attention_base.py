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
# Adapted from https://github.com/aws-neuron/neuronx-distributed-inference/blob/9993358ce052fd7a1bb4a7497a6318aac36ed95c/src/neuronx_distributed_inference/modules/attention/attention_base.py
import logging
import math
import warnings
from enum import Enum

import torch
from torch import Tensor, nn
from torch.distributed import ProcessGroup
from transformers import PretrainedConfig

from .utils import (
    apply_rotary_pos_emb,
    manual_softmax,
    move_heads_front,
    repeat_kv,
)


# Try except for the compatibility with older compiler version
try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel  # noqa: E402
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel  # noqa: E402

import neuronx_distributed as nxd
from neuronx_distributed.parallel_layers import parallel_state, utils  # noqa: E402
from neuronx_distributed.parallel_layers.layers import SPMDRank
from neuronxcc.nki.language import nc
from torch_neuronx.xla_impl.ops import nki_jit  # noqa: E402

from ...config import NxDNeuronConfig
from .gqa import GroupQueryAttention_O, GroupQueryAttention_QKV  # noqa: E402


logger = logging.getLogger("Neuron")

_flash_fwd_call = nki_jit()(attention_isa_kernel)


class FlashAttentionStrategy(Enum):
    NONE = 0
    UNSHARDED_KERNEL = 1
    SHARDED_KERNEL = 2


class NeuronAttentionBase(nn.Module):
    """
    This base attention class implements the core Neuron related adaptation including
    1. replaces the q_proj, k_proj, v_proj with column parallel layer
    2. replaces the o_proj with row parallel layer
    3. update self.num_head to be self.num_head / tp_degree
    4. update self.num_key_value_heads to be self.num_key_value_heads / tp_degree
    5. update forward() method to adjust to changes from self.num_head
    """

    def __init__(
        self,
        config: PretrainedConfig,
        neuron_config: NxDNeuronConfig,
        tensor_model_parallel_group: ProcessGroup | None = None,
        qkv_proj_bias: bool = False,
        o_proj_bias: bool = False,
        qk_scale: float | None = None,
    ):
        if not parallel_state.model_parallel_is_initialized():
            raise ValueError(
                "Neuron Attention has to be initialized in a distributed env. Please use neuronx_distributed"
                " module to initialize a distributed env."
            )
        super().__init__()

        if tensor_model_parallel_group is not None:
            self.tensor_model_parallel_group = tensor_model_parallel_group
            self.rank_util = SPMDRank(world_size=self.tensor_model_parallel_group.size())
        else:
            self.tensor_model_parallel_group = nxd.parallel_layers.parallel_state.get_tensor_model_parallel_group()
            self.rank_util = SPMDRank(world_size=self.tensor_model_parallel_group.size())

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None)
        # Head dim could be present but set to None in some models
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.torch_dtype = neuron_config.torch_dtype
        self.rms_norm_eps = config.rms_norm_eps
        self._qk_scale = qk_scale

        self.o_proj_layer_name = "o_proj"

        self.qkv_proj = GroupQueryAttention_QKV(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            tp_degree=neuron_config.tp_degree,
            dtype=self.torch_dtype,
            bias=qkv_proj_bias,
            gather_output=False,
            fused_qkv=neuron_config.fused_qkv,
            tensor_model_parallel_group=self.tensor_model_parallel_group,
            rms_norm_eps=self.rms_norm_eps,
            logical_nc_config=neuron_config.logical_nc_config,
        )
        self.o_proj = GroupQueryAttention_O(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            tp_degree=neuron_config.tp_degree,
            dtype=self.torch_dtype,
            bias=o_proj_bias,
            input_is_parallel=True,
            layer_name=self.o_proj_layer_name,
            tensor_model_parallel_group=self.tensor_model_parallel_group,
            rpl_reduce_dtype=neuron_config.torch_dtype,
        )
        self.num_heads = utils.divide(self.qkv_proj.get_num_attention_heads(), neuron_config.tp_degree)
        self.num_key_value_heads = utils.divide(self.qkv_proj.get_num_key_value_heads(), neuron_config.tp_degree)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        # By default we do not use layernorm in q and k projection
        # This can be changed in the subclass if needed: maybe make it a parameter?
        self.q_layernorm = None
        self.k_layernorm = None
        self.logical_nc_config = neuron_config.logical_nc_config

    @property
    def qk_scale(self):
        return self._qk_scale or (1.0 / math.sqrt(self.head_dim))

    def scaled_qk(self, Q, K, attention_mask):
        qk_scale = self.qk_scale
        QK = torch.matmul(Q, K.transpose(2, 3)) * qk_scale
        QK = torch.where(attention_mask, QK, torch.finfo(QK.dtype).min)
        return QK

    def prep_qkv_tensors(
        self,
        position_ids,
        hidden_states,
        past_key_value,
        cos_cache=None,
        sin_cache=None,
        rmsnorm=None,
    ):
        """take care of the shape, layout, group query, custom position encoding, etc."""
        Q, K, V = self.qkv_proj(hidden_states=hidden_states, rmsnorm=rmsnorm)

        # Divide hidden_dim across heads for MHA
        # Change layout: BSHD -> BHSD
        bsz, q_len, _ = hidden_states.size()

        Q = move_heads_front(Q, bsz, q_len, self.num_heads, self.head_dim, layernorm=self.q_layernorm)
        K = move_heads_front(K, bsz, q_len, self.num_key_value_heads, self.head_dim, layernorm=self.k_layernorm)
        V = move_heads_front(V, bsz, q_len, self.num_key_value_heads, self.head_dim, layernorm=None)

        # Rotate Q and K
        if self.rotary_emb is not None:
            if cos_cache is None or sin_cache is None:
                cos_cache, sin_cache = self.rotary_emb(V, position_ids)

            Q, K = apply_rotary_pos_emb(Q, K, cos_cache, sin_cache)

        return Q, K, V, cos_cache, sin_cache

    def perform_prefill(self, Q, K, V, q_len, bsz, attention_mask) -> Tensor:
        """attention computation at prefilling (context encoding) phase"""
        K_active = repeat_kv(K, self.num_key_value_groups)
        V_active = repeat_kv(V, self.num_key_value_groups)

        flash_attn_strategy = self.get_flash_attention_strategy(q_len)
        logger.debug(f"Flash attention strategy: {flash_attn_strategy}")

        if flash_attn_strategy != FlashAttentionStrategy.NONE:
            logger.debug(f"ATTN kernel: logical_nc_config={self.logical_nc_config}")

            # original shape of q, k, v is BHSD, and expected output is also BHSD.
            logger.debug(f"Using flash_fwd for Q.shape={Q.shape}")
            # make sure to cast inputs to torch_dtype (this is needed because the downcast to bf16
            # might happen after the kernel hlo creation step). Also convert shapes as expected by the kernel.

            # original Q shape: batch, num_heads, seqlen, d_head
            Q = (
                Q.permute(0, 1, 3, 2)  # after permute: batch, num_heads, d_head, seqlen
                .reshape((bsz * self.num_heads, self.head_dim, q_len))
                .to(self.torch_dtype)
            )
            Q = Q * self.qk_scale
            K_active = (
                K_active.permute(0, 1, 3, 2).reshape((bsz * self.num_heads, self.head_dim, q_len)).to(self.torch_dtype)
            )
            V_active = V_active.reshape((bsz * self.num_heads, q_len, self.head_dim)).to(self.torch_dtype)
            # shape: (B*H)DS
            attn_output = torch.zeros(bsz * self.num_heads, self.head_dim, q_len, dtype=Q.dtype, device=Q.device)

            logger.debug("Input parameter shapes")
            logger.debug(f"Q input shape {Q.shape}")
            logger.debug(f"K input shape {K_active.shape}")
            logger.debug(f"V input shape {V_active.shape}")
            logger.debug(f"Attn output shape {attn_output.shape}")

            if flash_attn_strategy == FlashAttentionStrategy.SHARDED_KERNEL:
                grid = (nc(self.logical_nc_config),)

                _flash_fwd_call[grid](
                    Q,
                    K_active,
                    V_active,
                    1.0,
                    attn_output,
                    kernel_name="CausalAttentionMMSoftmaxMMWithoutSwap",
                )
            elif flash_attn_strategy == FlashAttentionStrategy.UNSHARDED_KERNEL:
                _flash_fwd_call(
                    Q,
                    K_active,
                    V_active,
                    1.0,
                    attn_output,
                    kernel_name="CausalAttentionMMSoftmaxMMWithoutSwap",
                )
            else:
                raise ValueError(f"Invalid flash attention strategy: {flash_attn_strategy}")

            # shape: BHDS
            attn_output = attn_output.reshape((bsz, self.num_heads, self.head_dim, q_len))
            logger.debug(f"Attn output after reshape {attn_output.shape}")
        else:
            logger.debug("ATTN: native compiler")
            logger.debug(f"Not using flash_fwd for Q.shape={Q.shape}")
            active_scores = self.scaled_qk(Q, K_active, attention_mask)
            active_scores = nn.functional.softmax(active_scores, dim=-1, dtype=torch.float32).to(Q.dtype)
            attn_output = torch.matmul(active_scores, V_active)
        return attn_output, flash_attn_strategy

    def get_flash_attention_strategy(self, q_len) -> FlashAttentionStrategy:
        """
        Gets the flash attention strategy.

        For LNC1, use the unsharded kernel if sequence length is at least 4096 to get the best performance.
        The unsharded kernel requires a sequence length of at least 512.

        For LNC2, use the sharded kernel if sequence length is divisible by 1024. Otherwise, use no
        kernel, because the unsharded kernel has worse performance than no kernel.
        The sharded kernel requires a sequence length of at least 1024.

        These constraints may change later.

        TODO: Throw an exception instead of disabling flash attention if explicitly enabled but not eligible.
              This must consider bucketing to avoid throwing an exception for smaller buckets.
        """
        if self._qk_scale is not None:
            # If a custom qk_scale is provided, flash attention is not supported.
            return FlashAttentionStrategy.NONE
        if int(self.logical_nc_config) > 1:
            if q_len < 1024:
                return FlashAttentionStrategy.NONE

            if q_len % 1024 == 0:
                return FlashAttentionStrategy.SHARDED_KERNEL
            else:
                warnings.warn("Flash attention disabled. LNC2 requires seq_len % 1024 for flash attn to be performant")
                return FlashAttentionStrategy.NONE

        # If seq_len is at least 4096, enable flash attn automatically to improve performance.
        if q_len >= 4096:
            return FlashAttentionStrategy.UNSHARDED_KERNEL

        return FlashAttentionStrategy.NONE

    def compute_for_token_gen(self, Q, K, V, position_ids, past_key_value, attention_mask, active_mask) -> Tensor:
        """attention computation at token generation phase"""
        is_speculation = position_ids.shape[-1] > 1

        # Attention computation: softmax((Q.K/√dkv) + mask).V
        # i. prior (cached) KV
        K_prior = past_key_value[0]
        V_prior = past_key_value[1]
        K_prior = repeat_kv(K_prior, self.num_key_value_groups)
        V_prior = repeat_kv(V_prior, self.num_key_value_groups)
        prior_scores = torch.matmul(Q, K_prior.transpose(2, 3)) * self.qk_scale
        prior_scores = torch.where(attention_mask, prior_scores, torch.finfo(prior_scores.dtype).min)
        prior_scores = prior_scores.to(torch.float32)

        # ii. active (current/new) KV
        K_active = repeat_kv(K, self.num_key_value_groups)
        V_active = repeat_kv(V, self.num_key_value_groups)
        active_scores = torch.matmul(Q, K_active.transpose(2, 3)) * self.qk_scale
        if is_speculation:
            active_scores = torch.where(active_mask, active_scores, torch.finfo(active_scores.dtype).min)
        active_scores = active_scores.to(torch.float32)

        # iii. attention scores
        softmax_prior, softmax_active = manual_softmax(prior_scores, active_scores)
        softmax_prior, softmax_active = softmax_prior.to(Q.dtype), softmax_active.to(Q.dtype)
        attn_prior = torch.matmul(softmax_prior, V_prior)
        attn_active = torch.matmul(softmax_active, V_active)
        attn_output = attn_prior + attn_active

        return attn_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        active_mask: torch.LongTensor | None = None,
        cos_cache: torch.Tensor | None = None,
        sin_cache: torch.Tensor | None = None,
        rmsnorm=None,
    ) -> tuple[Tensor, tuple[Tensor, Tensor] | None]:
        """Implements each layer's forward pass for the attention block."""
        bsz, q_len, _ = hidden_states.size()

        Q, K, V, cos_cache, sin_cache = self.prep_qkv_tensors(
            position_ids,
            hidden_states,
            past_key_value,
            cos_cache=cos_cache,
            sin_cache=sin_cache,
            rmsnorm=rmsnorm,
        )

        flash_attn_strategy = FlashAttentionStrategy.NONE
        if past_key_value is None:
            attn_output, flash_attn_strategy = self.perform_prefill(Q, K, V, q_len, bsz, attention_mask)
        else:
            attn_output = self.compute_for_token_gen(
                Q, K, V, position_ids, past_key_value, attention_mask, active_mask
            )

        if flash_attn_strategy != FlashAttentionStrategy.NONE:
            # transpose BHDS -> BSHD
            # this layout avoids additional transposes between attention kernel and output projection
            attn_output = attn_output.permute(0, 3, 1, 2)
        else:
            # transpose BHSD -> BSHD
            attn_output = attn_output.transpose(1, 2).contiguous()

        # merge multi head hidden
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        # Z = Z.Wo
        attn_output = self.o_proj(attn_output)

        past_key_value: tuple[Tensor, Tensor] = (K, V)

        return attn_output, past_key_value, cos_cache, sin_cache
