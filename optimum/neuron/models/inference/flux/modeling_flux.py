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
import torch.nn.functional as F
from neuronx_distributed.parallel_layers.layer_norm import LayerNorm
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.mappings import reduce_from_tensor_model_parallel_region

from ..backend.modules.diffusion.attention import NeuronFeedForward, NeuronAttention
from ..backend.modules.diffusion.embeddings import (
    FluxPosEmbed,
    NeuronCombinedTimestepGuidanceTextProjEmbeddings,
    NeuronCombinedTimestepTextProjEmbeddings,
)
from ..backend.modules.diffusion.normalization import (
    NeuronAdaLayerNormContinuous,
    NeuronAdaLayerNormZero,
    NeuronAdaLayerNormZeroSingle,
)


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class NeuronFluxTransformer2DModel(torch.nn.Module):
    """
    The Transformer model introduced in Flux.

    Reference: https://blackforestlabs.ai/announcing-black-forest-labs/

    Parameters:
        patch_size (`int`): Patch size to turn the input data into small patches.
        in_channels (`int`, *optional*, defaults to 16): The number of channels in the input.
        num_layers (`int`, *optional*, defaults to 18): The number of layers of MMDiT blocks to use.
        num_single_layers (`int`, *optional*, defaults to 18): The number of layers of single DiT blocks to use.
        attention_head_dim (`int`, *optional*, defaults to 64): The number of channels in each head.
        num_attention_heads (`int`, *optional*, defaults to 18): The number of heads to use for multi-head attention.
        joint_attention_dim (`int`, *optional*): The number of `encoder_hidden_states` dimensions to use.
        pooled_projection_dim (`int`): Number of dimensions to use when projecting the `pooled_projections`.
        guidance_embeds (`bool`, defaults to False): Whether to use guidance embeddings.
    """

    def __init__(
        self,
        patch_size: int = 1,
        in_channels: int = 64,
        out_channels: int | None = None,
        num_layers: int = 19,
        num_single_layers: int = 38,
        attention_head_dim: int = 128,
        num_attention_heads: int = 24,
        joint_attention_dim: int = 4096,
        pooled_projection_dim: int = 768,
        guidance_embeds: bool = False,
        axes_dims_rope: list[int] = [16, 56, 56],
        reduce_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        self.config = SimpleNamespace(
            patch_size=patch_size,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            num_single_layers=num_single_layers,
            attention_head_dim=attention_head_dim,
            num_attention_heads=num_attention_heads,
            joint_attention_dim=joint_attention_dim,
            pooled_projection_dim=pooled_projection_dim,
            guidance_embeds=guidance_embeds,
        )

        self.out_channels = out_channels or in_channels
        self.inner_dim = self.config.num_attention_heads * self.config.attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=axes_dims_rope)

        text_time_guidance_cls = (
            NeuronCombinedTimestepGuidanceTextProjEmbeddings
            if guidance_embeds
            else NeuronCombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=self.config.pooled_projection_dim,
            reduce_dtype=reduce_dtype,
        )

        # We can't use gather_output=False, there is a LayerNorm at the beginning of the next FluxTransformerBlock
        self.context_embedder = ColumnParallelLinear(
            self.config.joint_attention_dim,
            self.inner_dim,
            gather_output=True,
            reduce_dtype=reduce_dtype,
        )
        self.x_embedder = ColumnParallelLinear(
            self.config.in_channels, self.inner_dim, gather_output=True, reduce_dtype=reduce_dtype
        )

        self.transformer_blocks = nn.ModuleList(
            [
                NeuronFluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    reduce_dtype=reduce_dtype,
                )
                for i in range(self.config.num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                NeuronFluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config.num_attention_heads,
                    attention_head_dim=self.config.attention_head_dim,
                    reduce_dtype=reduce_dtype,
                )
                for i in range(self.config.num_single_layers)
            ]
        )

        # this might be different to DiT, this is not AdaLayerNormZero.
        # The key difference is in the initialization strategy.
        # AdaLayerNormZero starts with an identity mapping (no transformation) and gradually learns during training
        self.norm_out = NeuronAdaLayerNormContinuous(
            self.inner_dim,
            self.inner_dim,
            elementwise_affine=False,
            eps=1e-6,
            use_parallel_layer=True,
        )
        self.proj_out = ColumnParallelLinear(
            self.inner_dim,
            patch_size * patch_size * self.out_channels,
            bias=True,
            gather_output=True,
            reduce_dtype=reduce_dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        image_rotary_emb: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: dict[str, Any] | None = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        controlnet_blocks_repeat: bool = False,
    ):
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).

        Returns:
            the sample tensor.
        """
        if joint_attention_kwargs is not None:
            joint_attention_kwargs = joint_attention_kwargs.copy()

        if joint_attention_kwargs is not None and joint_attention_kwargs.get("scale", None) is not None:
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
            )

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.norm_out(hidden_states, temb)
        output = self.proj_out(hidden_states)
        return output


class NeuronFluxSingleTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        reduce_dtype=torch.bfloat16,
        mlp_ratio=4.0,
    ):
        super().__init__()
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = NeuronAdaLayerNormZeroSingle(dim, use_parallel_layer=True)
        self.proj_mlp = ColumnParallelLinear(dim, self.mlp_hidden_dim, gather_output=False, reduce_dtype=reduce_dtype)
        self.act_mlp = nn.GELU(approximate="tanh")
        # To avoid all_gathers after Q K V projections in the Attention block, we use two separated proj_outs,
        # one for the MLP output and one for the Attention output at the end we simply add them together, it's same as
        # the original implementation that concatenates the outputs along the hidden dimension, and then split them
        # again with the RowParallelLinear and gather the results with All-Reduce
        #
        # Here we also disabled the reduce_output of RowParallelLinear to merge two All-Reduces into one
        # bias add must be done after All-Reduce, so we skip it here and add it later
        self.proj_out_attn = RowParallelLinear(
            dim,
            dim,
            input_is_parallel=True,
            reduce_dtype=reduce_dtype,
            reduce_output=False,
            skip_bias_add=True,
        )
        # We don't want to add the bias twice, so disable one of them
        self.proj_out_mlp = RowParallelLinear(
            self.mlp_hidden_dim,
            dim,
            input_is_parallel=True,
            bias=False,
            reduce_dtype=reduce_dtype,
            reduce_output=False,
        )

        self.attn = NeuronAttention(
            query_dim=dim,
            cross_attention_dim=None,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            bias=True,
            qk_norm="rms_norm",
            eps=1e-6,
            pre_only=True,
            reduce_dtype=reduce_dtype,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
    ):
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        gate = gate.unsqueeze(1)
        out_attn, bias = self.proj_out_attn(attn_output)
        out_mlp = self.proj_out_mlp(mlp_hidden_states)
        proj_out = reduce_from_tensor_model_parallel_region(
            out_attn + out_mlp, process_group=self.proj_out_attn.tensor_parallel_group
        )
        hidden_states = gate * (proj_out + bias)
        hidden_states = residual + hidden_states
        if hidden_states.dtype == torch.float16:
            hidden_states = hidden_states.clip(-65504, 65504)

        return hidden_states


class NeuronFluxTransformerBlock(nn.Module):
    r"""
    A Transformer block following the MMDiT architecture, introduced in Stable Diffusion 3.

    Reference: https://arxiv.org/abs/2403.03206

    Parameters:
        dim (`int`): The number of channels in the input and output.
        num_attention_heads (`int`): The number of heads to use for multi-head attention.
        attention_head_dim (`int`): The number of channels in each head.
        context_pre_only (`bool`): Boolean to determine if we should add some blocks associated with the
            processing of `context` conditions.
    """

    def __init__(
        self,
        dim,
        num_attention_heads,
        attention_head_dim,
        reduce_dtype=torch.bfloat16,
        qk_norm="rms_norm",
        eps=1e-6,
    ):
        super().__init__()

        self.norm1 = NeuronAdaLayerNormZero(dim)

        self.norm1_context = NeuronAdaLayerNormZero(dim)

        self.attn = NeuronAttention(
            query_dim=dim,
            cross_attention_dim=None,
            added_kv_proj_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            out_dim=dim,
            context_pre_only=False,
            bias=True,
            qk_norm=qk_norm,
            eps=eps,
            reduce_dtype=reduce_dtype,
        )

        self.norm2 = LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = NeuronFeedForward(dim=dim, dim_out=dim, activation_fn="gelu-approximate", reduce_dtype=reduce_dtype)

        self.norm2_context = LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff_context = NeuronFeedForward(
            dim=dim, dim_out=dim, activation_fn="gelu-approximate", reduce_dtype=reduce_dtype
        )

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(hidden_states, emb=temb)

        (
            norm_encoder_hidden_states,
            c_gate_msa,
            c_shift_mlp,
            c_scale_mlp,
            c_gate_mlp,
        ) = self.norm1_context(encoder_hidden_states, emb=temb)
        # Attention.
        attn_output, context_attn_output = self.attn(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
        )

        # Process attention outputs for the `hidden_states`.
        attn_output = gate_msa.unsqueeze(1) * attn_output
        hidden_states = hidden_states + attn_output

        norm_hidden_states = self.norm2(hidden_states)
        norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

        ff_output = self.ff(norm_hidden_states)
        ff_output = gate_mlp.unsqueeze(1) * ff_output

        hidden_states = hidden_states + ff_output

        # Process attention outputs for the `encoder_hidden_states`.

        context_attn_output = c_gate_msa.unsqueeze(1) * context_attn_output
        encoder_hidden_states = encoder_hidden_states + context_attn_output

        norm_encoder_hidden_states = self.norm2_context(encoder_hidden_states)
        norm_encoder_hidden_states = norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states
