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
from typing import Any, TYPE_CHECKING
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neuronx_distributed.utils.utils import hardware
from neuronx_distributed.parallel_layers.layer_norm import LayerNorm
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    SPMDRank,
)
from neuronx_distributed.parallel_layers.mappings import (
    gather_from_tensor_model_parallel_region_with_dim,
    reduce_from_tensor_model_parallel_region,
)
from neuronx_distributed.parallel_layers.parallel_state import (
    get_data_parallel_group, 
    get_tensor_model_parallel_size,
    get_world_group,
)
from ...backend.modules.rms_norm import NeuronRMSNorm
from ...backend.modules.attention.utils import transpose_parallel_linear_layer
from ...backend.utils.distributed import get_dp_rank_spmd, split_along_dim
from ...backend.utils.layer_boundary_marker import ModuleMarkerStartWrapper, ModuleMarkerEndWrapper
from .modules.kernels import matmul_o_proj_kernel
from .modules.activations import NeuronGELU
from .modules.embeddings import (
    FluxPosEmbed,
    NeuronCombinedTimestepGuidanceTextProjEmbeddings,
    NeuronCombinedTimestepTextProjEmbeddings,
    apply_rotary_emb,
)
from .modules.normalization import (
    NeuronAdaLayerNormContinuous,
    NeuronAdaLayerNormZero,
    NeuronAdaLayerNormZeroSingle,
)


try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel  # noqa: E402
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel  # noqa: E402

from neuronxcc.nki.language import nc
from torch_neuronx.utils import get_platform_target
from torch_neuronx.xla_impl.ops import nki_jit  # noqa: E402

if TYPE_CHECKING:
    from .......exporters.neuron.base import NeuronDefaultConfig

_HARDWARE = hardware(get_platform_target())

_flash_fwd_call = nki_jit()(attention_isa_kernel)

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def attention_wrapper_sharded_without_swap(query, key, value):
    bs, n_head, q_len, d_head = query.shape  # my change
    q = query.clone().permute(0, 1, 3, 2).reshape((bs * n_head, d_head, q_len))
    k = key.clone().permute(0, 1, 3, 2).reshape((bs * n_head, d_head, q_len))
    v = value.clone().reshape((bs * n_head, q_len, d_head))

    attn_output = torch.zeros((bs * n_head, q_len, d_head), dtype=torch.bfloat16, device=q.device)

    vc_size = int(os.getenv("NEURON_RT_VIRTUAL_CORE_SIZE", "1"))
    use_sharded_attention_kernel = vc_size == 2
    scale = 1 / math.sqrt(d_head)

    if use_sharded_attention_kernel:
        grid = (nc(2),)
        _flash_fwd_call[grid](q, k, v, scale, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap")
    else:
        _flash_fwd_call(q, k, v, scale, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap")

    attn_output = attn_output.reshape((bs, n_head, q_len, d_head))

    return attn_output


class FluxNxDConfig: 
    def __init__(
        self, 
        neuron_config,
        patch_size, 
        in_channels, 
        out_channels, 
        num_layers, 
        num_single_layers, 
        attention_head_dim, 
        num_attention_heads, 
        joint_attention_dim, 
        pooled_projection_dim, 
        guidance_embeds, 
    ): 
        self.neuron_config = neuron_config
        self.patch_size = patch_size 
        self.in_channels = in_channels 
        self.out_channels = out_channels 
        self.num_layers = num_layers 
        self.num_single_layers = num_single_layers 
        self.attention_head_dim = attention_head_dim 
        self.num_attention_heads = num_attention_heads 
        self.joint_attention_dim = joint_attention_dim 
        self.pooled_projection_dim = pooled_projection_dim 
        self.guidance_embeds = guidance_embeds


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
        config: "NeuronDefaultConfig",
    ):
        super().__init__()
        self.config = config

        self.data_parallel_group = get_data_parallel_group()
        self.global_rank = SPMDRank(world_size=get_world_group().size())
        self.context_parallel_enabled = (self.config.world_size != self.config.tensor_parallel_size)
        self.enable_out_proj_kernel = (
            _HARDWARE == hardware.TRN2 and self.config.world_size != self.config.tensor_parallel_size
        )  # only supports 1024x1024 inputs for now

        self.out_channels = self.config._config.in_channels
        self.inner_dim = self.config._config.num_attention_heads * self.config._config.attention_head_dim

        self.pos_embed = FluxPosEmbed(theta=10000, axes_dim=(16, 56, 56))

        text_time_guidance_cls = (
            NeuronCombinedTimestepGuidanceTextProjEmbeddings
            if self.config._config.guidance_embeds
            else NeuronCombinedTimestepTextProjEmbeddings
        )
        self.time_text_embed = text_time_guidance_cls(
            embedding_dim=self.inner_dim,
            pooled_projection_dim=self.config._config.pooled_projection_dim,
            reduce_dtype=self.config.float_dtype,
        )

        # We can't use gather_output=False, there is a LayerNorm at the beginning of the next FluxTransformerBlock
        self.context_embedder = ColumnParallelLinear(
            self.config._config.joint_attention_dim,
            self.inner_dim,
            gather_output=True,
            reduce_dtype=self.config.float_dtype,
        )
        self.x_embedder = ColumnParallelLinear(
            self.config._config.in_channels, self.inner_dim, gather_output=True, reduce_dtype=self.config.float_dtype,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                NeuronFluxTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config._config.num_attention_heads,
                    attention_head_dim=self.config._config.attention_head_dim,
                    reduce_dtype=self.config.float_dtype,
                    context_parallel_enabled=self.context_parallel_enabled,
                    enable_out_proj_kernel=self.enable_out_proj_kernel,
                )
                for i in range(self.config._config.num_layers)
            ]
        )

        self.single_transformer_blocks = nn.ModuleList(
            [
                NeuronFluxSingleTransformerBlock(
                    dim=self.inner_dim,
                    num_attention_heads=self.config._config.num_attention_heads,
                    attention_head_dim=self.config._config.attention_head_dim,
                    reduce_dtype=self.config.float_dtype,
                    context_parallel_enabled=self.context_parallel_enabled,
                )
                for i in range(self.config._config.num_single_layers)
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
            self.config.patch_size * self.config.patch_size * self.out_channels,
            bias=True,
            gather_output=True,
            reduce_dtype=self.config.float_dtype,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        guidance: torch.Tensor = None,
        image_rotary_emb: torch.Tensor = None,
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

        if (
            joint_attention_kwargs is not None
            and joint_attention_kwargs.get("scale", None) is not None
        ):
            logger.warning(
                "Passing `scale` via `joint_attention_kwargs` when not using the PEFT backend is ineffective."
            )

        hidden_states = self.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000

        temb = (
            self.time_text_embed(timestep, pooled_projections)
            if not self.config._config.guidance_embeds
            else self.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.context_embedder(encoder_hidden_states)

        dp_rank = get_dp_rank_spmd(
            global_rank=self.global_rank.get_rank(),
            tp_degree=get_tensor_model_parallel_size(),
        )

        rotary_emb_text = None
        rotary_emb_image = None

        # scatter inputs to DP group
        if self.context_parallel_enabled:
            # TODO: see if rotary split can only be done in one denoising step and reuse
            rotary_emb_text = image_rotary_emb[: encoder_hidden_states.shape[1]]
            rotary_emb_image = image_rotary_emb[encoder_hidden_states.shape[1] :]
            rotary_emb_text = split_along_dim(
                rotary_emb_text, dim=0, rank=dp_rank, data_parallel_group=self.data_parallel_group
            )
            rotary_emb_image = split_along_dim(
                rotary_emb_image, dim=0, rank=dp_rank, data_parallel_group=self.data_parallel_group
            )
            hidden_states = split_along_dim(
                hidden_states, dim=1, rank=dp_rank, data_parallel_group=self.data_parallel_group
            )
            encoder_hidden_states = split_along_dim(
                encoder_hidden_states,
                dim=1,
                rank=dp_rank,
                data_parallel_group=self.data_parallel_group,
            )

        hidden_states, encoder_hidden_states = ModuleMarkerStartWrapper()(
            hidden_states, encoder_hidden_states
        )
        for index_block, block in enumerate(self.transformer_blocks):
            encoder_hidden_states, hidden_states = block(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
                rotary_emb_text=rotary_emb_text,
                rotary_emb_image=rotary_emb_image,
            )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states
                        + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block // interval_control]
                    )
        hidden_states, encoder_hidden_states = ModuleMarkerEndWrapper()(
            hidden_states, encoder_hidden_states
        )
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.single_transformer_blocks):
            if index_block % 2 == 0:
                hidden_states = ModuleMarkerStartWrapper()(hidden_states)
            hidden_states = block(
                hidden_states=hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                joint_attention_kwargs=joint_attention_kwargs,
                rotary_emb_text=rotary_emb_text,
                rotary_emb_image=rotary_emb_image,
            )

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.single_transformer_blocks) / len(
                    controlnet_single_block_samples
                )
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )
            if index_block % 2 == 1:
                hidden_states = ModuleMarkerEndWrapper()(hidden_states)

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.norm_out(hidden_states, temb)

        output = self.proj_out(hidden_states)

        if self.context_parallel_enabled:
            # gather output
            output = gather_from_tensor_model_parallel_region_with_dim(
                output, gather_dim=1, process_group=self.data_parallel_group
            )

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
        context_parallel_enabled=False,
    ):
        super().__init__()

        self.context_parallel_enabled = context_parallel_enabled
        self.mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm = NeuronAdaLayerNormZeroSingle(dim, use_parallel_layer=True)
        self.proj_mlp = ColumnParallelLinear(
            dim, self.mlp_hidden_dim, gather_output=False, reduce_dtype=reduce_dtype
        )
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

        self.attn = NeuronFluxAttention(
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
            context_parallel_enabled=self.context_parallel_enabled,
        )

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        temb: torch.FloatTensor,
        image_rotary_emb=None,
        joint_attention_kwargs=None,
        rotary_emb_text=None,
        rotary_emb_image=None,
    ):
        residual = hidden_states
        norm_hidden_states, gate = self.norm(hidden_states, emb=temb)
        mlp_hidden_states = self.act_mlp(self.proj_mlp(norm_hidden_states))
        attn_output = self.attn(
            hidden_states=norm_hidden_states,
            image_rotary_emb=image_rotary_emb,
            rotary_emb_text=rotary_emb_text,
            rotary_emb_image=rotary_emb_image,
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
        context_parallel_enabled=False,
        enable_out_proj_kernel=False,
    ):
        super().__init__()

        self.context_parallel_enabled = context_parallel_enabled
        self.enable_out_proj_kernel = enable_out_proj_kernel
        self.norm1 = NeuronAdaLayerNormZero(dim)

        self.norm1_context = NeuronAdaLayerNormZero(dim)

        self.attn = NeuronFluxAttention(
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
            context_parallel_enabled=self.context_parallel_enabled,
            enable_out_proj_kernel=self.enable_out_proj_kernel,
        )

        self.norm2 = LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.ff = NeuronFeedForward(
            dim=dim, dim_out=dim, activation_fn="gelu-approximate", reduce_dtype=reduce_dtype
        )

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
        rotary_emb_text=None,
        rotary_emb_image=None,
        joint_attention_kwargs=None,
    ):
        norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
            hidden_states,
            emb=temb,
            hlomarker=True,
        )

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
            rotary_emb_text=rotary_emb_text,
            rotary_emb_image=rotary_emb_image,
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
        norm_encoder_hidden_states = (
            norm_encoder_hidden_states * (1 + c_scale_mlp[:, None]) + c_shift_mlp[:, None]
        )

        context_ff_output = self.ff_context(norm_encoder_hidden_states)
        encoder_hidden_states = encoder_hidden_states + c_gate_mlp.unsqueeze(1) * context_ff_output
        if encoder_hidden_states.dtype == torch.float16:
            encoder_hidden_states = encoder_hidden_states.clip(-65504, 65504)

        return encoder_hidden_states, hidden_states


class NeuronFeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        dropout (`float`, *optional*, defaults to 0.0): The dropout probability to use.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        final_dropout (`bool` *optional*, defaults to False): Apply a final dropout.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
        self,
        dim: int,
        dim_out: int | None = None,
        mult: int = 4,
        dropout: float = 0.0,
        activation_fn: str = "geglu",
        final_dropout: bool = False,
        inner_dim=None,
        bias: bool = True,
        reduce_dtype: torch.dtype = torch.bfloat16,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu-approximate":
            act_fn = NeuronGELU(
                dim,
                inner_dim,
                approximate="tanh",
                bias=bias,
                gather_output=False,
                reduce_dtype=reduce_dtype,
            )

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # project dropout
        self.net.append(nn.Dropout(dropout))
        # project out
        self.net.append(
            RowParallelLinear(
                inner_dim, dim_out, bias=bias, input_is_parallel=True, reduce_dtype=reduce_dtype
            )
        )
        # FF as used in Vision Transformer, MLP-Mixer, etc. have a final dropout
        if final_dropout:
            self.net.append(nn.Dropout(dropout))

    def forward(self, hidden_states: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            logger.warning(
                "The `scale` argument is deprecated and will be ignored. Please remove it, "
                "as passing it will raise an error in the future. `scale` should directly be "
                "passed while calling the underlying pipeline component i.e., "
                "via `cross_attention_kwargs`."
            )
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states


class NeuronFluxAttention(nn.Module):
    r"""
    A cross attention layer.

    Parameters:
        query_dim (`int`):
            The number of channels in the query.
        cross_attention_dim (`int`, *optional*):
            The number of channels in the encoder_hidden_states. If not given, defaults to `query_dim`.
        heads (`int`,  *optional*, defaults to 8):
            The number of heads to use for multi-head attention.
        kv_heads (`int`,  *optional*, defaults to `None`):
            The number of key and value heads to use for multi-head attention. Defaults to `heads`. If
            `kv_heads=heads`, the model will use Multi Head Attention (MHA), if `kv_heads=1` the model will use Multi
            Query Attention (MQA) otherwise GQA is used.
        dim_head (`int`,  *optional*, defaults to 64):
            The number of channels in each head.
        dropout (`float`, *optional*, defaults to 0.0):
            The dropout probability to use.
        bias (`bool`, *optional*, defaults to False):
            Set to `True` for the query, key, and value linear layers to contain a bias parameter.
        upcast_attention (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the attention computation to `float32`.
        upcast_softmax (`bool`, *optional*, defaults to False):
            Set to `True` to upcast the softmax computation to `float32`.
        cross_attention_norm (`str`, *optional*, defaults to `None`):
            The type of normalization to use for the cross attention. Can be `None`, `layer_norm`, or `group_norm`.
        cross_attention_norm_num_groups (`int`, *optional*, defaults to 32):
            The number of groups to use for the group norm in the cross attention.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
        norm_num_groups (`int`, *optional*, defaults to `None`):
            The number of groups to use for the group norm in the attention.
        spatial_norm_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the spatial normalization.
        out_bias (`bool`, *optional*, defaults to `True`):
            Set to `True` to use a bias in the output linear layer.
        scale_qk (`bool`, *optional*, defaults to `True`):
            Set to `True` to scale the query and key by `1 / sqrt(dim_head)`.
        eps (`float`, *optional*, defaults to 1e-5):
            An additional value added to the denominator in group normalization that is used for numerical stability.
        rescale_output_factor (`float`, *optional*, defaults to 1.0):
            A factor to rescale the output by dividing it with this value.
        residual_connection (`bool`, *optional*, defaults to `False`):
            Set to `True` to add the residual connection to the output.
        _from_deprecated_attn_block (`bool`, *optional*, defaults to `False`):
            Set to `True` if the attention block is loaded from a deprecated state dict.
    """

    def __init__(
        self,
        query_dim: int,
        cross_attention_dim: int | None = None,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
        bias: bool = False,
        upcast_attention: bool = False,
        upcast_softmax: bool = False,
        cross_attention_norm: str | None = None,
        qk_norm: str | None = None,
        added_kv_proj_dim: int | None = None,
        added_proj_bias: bool | None = True,
        out_bias: bool = True,
        scale_qk: bool = True,
        eps: float = 1e-5,
        rescale_output_factor: float = 1.0,
        residual_connection: bool = False,
        _from_deprecated_attn_block: bool = False,
        out_dim: int = None,
        out_context_dim: int = None,
        context_pre_only=None,
        pre_only=False,
        is_causal: bool = False,
        pad_heads: bool = True,
        reduce_dtype: torch.dtype = torch.bfloat16,
        context_parallel_enabled=False,
        enable_out_proj_kernel=False,
    ):
        super().__init__()

        self.data_parallel_group = get_data_parallel_group()
        self.context_parallel_enabled = context_parallel_enabled
        self.enable_out_proj_kernel = enable_out_proj_kernel
        self.query_dim = query_dim
        self.use_bias = bias
        self.is_cross_attention = cross_attention_dim is not None
        self.cross_attention_dim = (
            cross_attention_dim if cross_attention_dim is not None else query_dim
        )
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.rescale_output_factor = rescale_output_factor
        self.residual_connection = residual_connection
        self.dropout = dropout
        self.fused_projections = False
        self.out_dim = out_dim if out_dim is not None else query_dim
        self.out_context_dim = out_context_dim if out_context_dim is not None else query_dim
        self.context_pre_only = context_pre_only
        self.pre_only = pre_only
        self.is_causal = is_causal

        # we make use of this private variable to know whether this class is loaded
        # with an deprecated state dict so that we can convert it on the fly
        self._from_deprecated_attn_block = _from_deprecated_attn_block

        self.scale_qk = scale_qk
        self.scale = dim_head**-0.5 if self.scale_qk else 1.0

        self.heads = out_dim // dim_head if out_dim is not None else heads

        tp_degree = get_tensor_model_parallel_size()
        if pad_heads:
            self.heads = math.ceil(heads / tp_degree) * tp_degree
        self.padded_inner_dim = dim_head * self.heads
        # Only shard the heads, dim_head is unchanged.
        # So that the original RMSNorm and apply_rotary_emb implementations still work
        self.heads = self.heads // tp_degree

        self.added_kv_proj_dim = added_kv_proj_dim

        self.group_norm = None

        self.spatial_norm = None

        if qk_norm is None:
            self.norm_q = None
            self.norm_k = None
        elif qk_norm == "rms_norm":
            self.norm_q = NeuronRMSNorm(dim_head, eps=eps)
            self.norm_k = NeuronRMSNorm(dim_head, eps=eps)
        else:
            raise ValueError(f"unknown qk_norm: {qk_norm}. Should be None,'rms_norm'")

        if cross_attention_norm is None:
            self.norm_cross = None
        else:
            raise ValueError(
                f"unknown cross_attention_norm: {cross_attention_norm}. Should be None"
            )

        self.to_q = ColumnParallelLinear(
            query_dim,
            self.padded_inner_dim,
            bias=bias,
            gather_output=False,
            reduce_dtype=reduce_dtype,
        )
        self.to_k = ColumnParallelLinear(
            self.cross_attention_dim,
            self.padded_inner_dim,
            bias=bias,
            gather_output=False,
            reduce_dtype=reduce_dtype,
        )
        self.to_v = ColumnParallelLinear(
            self.cross_attention_dim,
            self.padded_inner_dim,
            bias=bias,
            gather_output=False,
            reduce_dtype=reduce_dtype,
        )

        self.added_proj_bias = added_proj_bias
        if self.added_kv_proj_dim is not None:
            self.add_k_proj = ColumnParallelLinear(
                added_kv_proj_dim,
                self.padded_inner_dim,
                bias=added_proj_bias,
                gather_output=False,
                reduce_dtype=reduce_dtype,
            )
            self.add_v_proj = ColumnParallelLinear(
                added_kv_proj_dim,
                self.padded_inner_dim,
                bias=added_proj_bias,
                gather_output=False,
                reduce_dtype=reduce_dtype,
            )
            if self.context_pre_only is not None:
                self.add_q_proj = ColumnParallelLinear(
                    added_kv_proj_dim,
                    self.padded_inner_dim,
                    bias=added_proj_bias,
                    gather_output=False,
                    reduce_dtype=reduce_dtype,
                )
        else:
            self.add_q_proj = None
            self.add_k_proj = None
            self.add_v_proj = None

        if not self.pre_only:
            self.to_out = nn.ModuleList([])
            self.to_out.append(
                RowParallelLinear(
                    self.padded_inner_dim,
                    self.out_dim,
                    bias=out_bias,
                    input_is_parallel=True,
                    reduce_dtype=reduce_dtype,
                )
            )
            self.to_out.append(nn.Dropout(dropout))
            if self.enable_out_proj_kernel:
                self.to_out[0].weight = transpose_parallel_linear_layer(self.to_out[0].weight)
        else:
            self.to_out = None

        if self.context_pre_only is not None and not self.context_pre_only:
            self.to_add_out = RowParallelLinear(
                self.padded_inner_dim,
                self.out_context_dim,
                bias=out_bias,
                input_is_parallel=True,
                reduce_dtype=reduce_dtype,
            )
        else:
            self.to_add_out = None

        if qk_norm is not None and added_kv_proj_dim is not None:
            if qk_norm == "rms_norm":
                self.norm_added_q = NeuronRMSNorm(dim_head, eps=eps)
                self.norm_added_k = NeuronRMSNorm(dim_head, eps=eps)
            else:
                raise ValueError(
                    f"unknown qk_norm: {qk_norm}. Should be one of `None,'layer_norm','fp32_layer_norm','rms_norm'`"
                )
        else:
            self.norm_added_q = None
            self.norm_added_k = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        image_rotary_emb: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        encoder_hidden_states: torch.Tensor | None = None,
        rotary_emb_text: torch.Tensor = None,
        rotary_emb_image: torch.Tensor = None,
    ) -> torch.Tensor:
        r"""
        The forward method of the `Attention` class.

        Args:
            hidden_states (`torch.Tensor`):
                The hidden states of the query.
            encoder_hidden_states (`torch.Tensor`, *optional*):
                The hidden states of the encoder.
            attention_mask (`torch.Tensor`, *optional*):
                The attention mask to use. If `None`, no mask is applied.
            **cross_attention_kwargs:
                Additional keyword arguments to pass along to the cross attention.

        Returns:
            `torch.Tensor`: The output of the attention layer.
        """
        batch_size, _, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        # `sample` projections.
        query = self.to_q(hidden_states)
        key = self.to_k(hidden_states)
        value = self.to_v(hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // self.heads

        query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
        # We skip V transpose for non_masked, context parallel single transformer.
        # We keep V seqlen to dim=1 for the context_parallel attention wrapper.
        if (
            self.context_parallel_enabled
            and encoder_hidden_states is None
            and attention_mask is None
        ):
            value = value.view(batch_size, -1, self.heads, head_dim)
        else:
            value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)

        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        if self.context_parallel_enabled:

            # need this for single transformer attention where inputs for text and image are merged
            if encoder_hidden_states is None:
                rotary_emb_image = torch.cat([rotary_emb_text, rotary_emb_image], dim=0)

            # apply rotary before gather and cat with text tokens as emb is split
            # along seq len for image and text
            if rotary_emb_image is not None:
                query = apply_rotary_emb(query, rotary_emb_image)
                key = apply_rotary_emb(key, rotary_emb_image)

            if encoder_hidden_states is not None:
                query, key, value = ModuleMarkerEndWrapper()(query, key, value)
                encoder_hidden_states, query, key, value = ModuleMarkerStartWrapper()(
                    encoder_hidden_states, query, key, value
                )

        # the attention in FluxSingleTransformerBlock does not use `encoder_hidden_states`
        if encoder_hidden_states is not None:
            # `context` projections.
            encoder_hidden_states_query_proj = self.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, self.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, self.heads, head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, self.heads, head_dim
            ).transpose(1, 2)

            if self.norm_added_q is not None:
                encoder_hidden_states_query_proj = self.norm_added_q(
                    encoder_hidden_states_query_proj
                )
            if self.norm_added_k is not None:
                encoder_hidden_states_key_proj = self.norm_added_k(encoder_hidden_states_key_proj)

            if self.context_parallel_enabled:

                # apply rotary before gather and cat with image tokens as emb is split
                # along seq len for image and text
                if rotary_emb_text is not None:
                    encoder_hidden_states_query_proj = apply_rotary_emb(
                        encoder_hidden_states_query_proj, rotary_emb_text
                    )
                    encoder_hidden_states_key_proj = apply_rotary_emb(
                        encoder_hidden_states_key_proj, rotary_emb_text
                    )

                # gather k and v from dp group - [B, H, S, D]
                stacked_kv = torch.stack([key, value], dim=0)
                # after gather => [2, B, H, S, D]
                stacked_kv = gather_from_tensor_model_parallel_region_with_dim(
                    stacked_kv,
                    gather_dim=3,
                    process_group=self.data_parallel_group,
                )
                key, value = torch.unbind(stacked_kv, dim=0)
                # gather k and v - [B, H, S, D]
                stacked_kv_enc = torch.stack(
                    [encoder_hidden_states_key_proj, encoder_hidden_states_value_proj], dim=0
                )
                # after gather => [2, B, H, S, D]
                stacked_kv_enc = gather_from_tensor_model_parallel_region_with_dim(
                    stacked_kv_enc,
                    gather_dim=3,
                    process_group=self.data_parallel_group,
                )
                encoder_hidden_states_key_proj, encoder_hidden_states_value_proj = torch.unbind(
                    stacked_kv_enc, dim=0
                )

            # attention
            # the concatenation is happening along the sequence dimension after the transpose operation above.
            query = torch.cat([encoder_hidden_states_query_proj, query], dim=2)
            key = torch.cat([encoder_hidden_states_key_proj, key], dim=2)
            value = torch.cat([encoder_hidden_states_value_proj, value], dim=2)
        elif self.context_parallel_enabled:
            # Single Transformer Case
            # In the non_masked case K and V have the same shape.
            if attention_mask is not None:
                # gather k and v from dp group - [B, H, S, D]
                stacked_kv = torch.stack([key, value], dim=0)
                # after gather => [2, B, H, S, D]
                stacked_kv = gather_from_tensor_model_parallel_region_with_dim(
                    stacked_kv,
                    gather_dim=3,
                    process_group=self.data_parallel_group,
                )
                key, value = torch.unbind(stacked_kv, dim=0)
            else:
                # Removed all_gather from here and moved it inside the context parallel attention kernel wrapper.
                hidden_states = attention_wrapper_context_parallel_single_transformer(
                    query, key, value, self.data_parallel_group
                )

        # apply rotary for non CP case
        if not self.context_parallel_enabled:
            if image_rotary_emb is not None:
                query = apply_rotary_emb(query, image_rotary_emb)
                key = apply_rotary_emb(key, image_rotary_emb)

        if attention_mask is not None:
            hidden_states = F.scaled_dot_product_attention(
                query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
            )
        else:
            # Already ran SDPA in the single_trasformer_block, context parallel attention case above.
            # Use the original SDPA wrapper for the remaining cases.
            if not self.context_parallel_enabled or encoder_hidden_states is not None:
                hidden_states = attention_wrapper_sharded_without_swap(query, key, value)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, self.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        if encoder_hidden_states is not None:
            # splitting along the sequence dimension
            encoder_hidden_states, hidden_states = (
                hidden_states[:, : encoder_hidden_states.shape[1]],
                hidden_states[:, encoder_hidden_states.shape[1] :],
            )

            if self.enable_out_proj_kernel:     # Executing out projection kernel.
                grid = (nc(2),)
                hidden_states = matmul_o_proj_kernel[grid](
                    hidden_states.transpose(1, 2), self.to_out[0].weight
                )
            else:
                hidden_states = self.to_out[0](hidden_states)
                hidden_states = self.to_out[1](hidden_states)

            encoder_hidden_states = self.to_add_out(encoder_hidden_states)

            return hidden_states, encoder_hidden_states
        else:
            if self.padded_inner_dim != self.out_dim:
                # For the single transformer block, we don't have an output projection to remove the padded hidden dimension
                # So we just cut them here base on the expected out_dim (they are all zeros)
                return hidden_states[..., : self.out_dim]
            else:
                return hidden_states


# Context Parallel Attention Helper Function, CC-Ops + Attention Kernel
# Inputs:
# * Query: assumes [bs, heads, seqlen, head_dim] shape.
# * Key  : assumes [bs, heads, seqlen, head_dim] shape.
# * Value: assumes [bs, seqlen, heads, head_dim] shape.
#
# Note: Used in the case of unmasked, context_parallel attention, single_transformer blocks.
# In this case, we omit V transpose on top of attention forward function (unlike for Q an K).
# This is why V shape is different than Q and K shape.
# apply_rotary_emb(...) and norm(...) is not applied to V,
# and we need to transpose seqlen to dim=0 anyways,
# for purpose of applying all_gather (all_gather requirement).
def attention_wrapper_context_parallel_single_transformer(query, key, value, process_group):
    bs, n_head, q_len, d_head = query.shape
    # K is assumed to be transposed in CP attention forward function.
    k_len = key.shape[2]
    # V is not transposed in CP attention forward function.
    v_len = value.shape[1]

    query = query.reshape((bs * n_head, q_len, d_head))
    key = key.reshape((bs * n_head, k_len, d_head))
    value = value.reshape((v_len, bs * n_head, d_head))

    # We are gathering on seqlen, move seqlen to dim=0.
    # K shape: [seqlen, bs * heads, d_head]
    key = key.transpose(0, 1)

    # Seqlen is already dim=0 in the case of V.
    value = gather_from_tensor_model_parallel_region_with_dim(
        value,
        gather_dim=0,
        process_group=process_group,
    )
    # Change V shape back to [bs * heads, seqlen, d_head] for SDPA kernel below.
    value = value.permute(1, 0, 2)

    key = gather_from_tensor_model_parallel_region_with_dim(
        key,
        gather_dim=0,
        process_group=process_group,
    )
    # Change K shape to [bs * heads, d_heads, seqlen] for SDPA kernel below.
    # Same shape as in the original "attention_wrapper_sharded_without_swap" SDPA wrapper.
    key = key.permute(1, 2, 0)

    scale = 1 / math.sqrt(d_head)
    attn_output = torch.zeros(
        (bs * n_head, q_len, d_head), dtype=torch.bfloat16, device=query.device
    )

    vc_size = int(os.getenv("NEURON_RT_VIRTUAL_CORE_SIZE", "1"))
    use_sharded_attention_kernel = vc_size == 2

    if use_sharded_attention_kernel:
        grid = (nc(2),)
        _flash_fwd_call[grid](
            query, key, value, scale, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap"
        )
    else:
        _flash_fwd_call(
            query, key, value, scale, attn_output, kernel_name="AttentionMMSoftmaxMMWithoutSwap"
        )

    attn_output = attn_output.reshape((bs, n_head, q_len, d_head))
    return attn_output
    