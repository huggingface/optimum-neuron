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
# Adapted from:
# - https://github.com/aws-neuron/neuronx-distributed-inference/tree/main/src/neuronx_distributed_inference/models/gpt_oss
# - https://github.com/huggingface/transformers/tree/main/src/transformers/models/gpt_oss
"""PyTorch GPT-OSS model for NxD inference."""

import logging
import math
import warnings

import torch
from neuronx_distributed.modules.moe.expert_mlps_v2 import ExpertMLPsV2
from neuronx_distributed.modules.moe.model import MoE
from neuronx_distributed.modules.moe.moe_configs import RoutedExpertsMLPOpsConfig
from neuronx_distributed.modules.moe.routing import RouterTopK
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
)
from torch import nn
from transformers import PretrainedConfig
from transformers.integrations.mxfp4 import convert_moe_packed_tensors
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig

from ..backend.config import NxDNeuronConfig
from ..backend.modules.attention.attention_base import NeuronAttentionBase
from ..backend.modules.decoder import NxDDecoderModelForCausalLM, NxDModelForCausalLM
from ..backend.modules.rms_norm import NeuronRMSNorm


logger = logging.getLogger("Neuron")


def convert_gate_up_proj_hf_transposed(gate_up_proj: torch.Tensor, is_bias: bool = False) -> torch.Tensor:
    """Convert gate_up_proj from HF-transposed interleaved format to NxD chunked format.

    HF's convert_moe_packed_tensors returns weights with transpose(1,2) applied:
    - Weight: [E, H, 2*I] with gate and up INTERLEAVED along last dim: [g0, u0, g1, u1, ...]
    - Bias: [E, 2*I] with gate and up INTERLEAVED: [g0, u0, g1, u1, ...]

    NxD expects chunked format (gate then up):
    - Weight: [E, H, 2*I] with [gate..., up...] along last dim
    - Bias: [E, 2*I] with [gate..., up...] along last dim

    Args:
        gate_up_proj: Tensor of shape [E, H, 2*I] (weight) or [E, 2*I] (bias), interleaved
        is_bias: Whether this is a bias tensor

    Returns:
        Tensor with same shape but de-interleaved (chunked format)
    """
    # De-interleave along the last dimension
    # Interleaved: [g0, u0, g1, u1, g2, u2, ...] → Chunked: [g0, g1, g2, ..., u0, u1, u2, ...]
    gate = gate_up_proj[..., ::2]  # Even indices: gate elements
    up = gate_up_proj[..., 1::2]  # Odd indices: up elements
    # Concatenate to get chunked format: [gate..., up...]
    return torch.cat((gate, up), dim=-1)


def convert_gpt_oss_to_neuron_state_dict(
    state_dict: dict, config: GptOssConfig, neuron_config: NxDNeuronConfig
) -> dict:
    """Convert GPT-OSS HF state dict to Neuron format.

    This handles:
    1. Dequantization of MXFP4 weights to bfloat16
    2. Conversion of gate_up_proj from interleaved [E, H, 2*I] to chunked [E, H, 2*I]
    3. down_proj: HF already provides [E, H, I] which is correct for NxD (no conversion needed)
    4. Router weight renaming: mlp.router.weight -> mlp.router.linear_router.weight

    Args:
        state_dict: HF checkpoint state dict
        config: GPT-OSS model configuration
        neuron_config: Neuron configuration

    Returns:
        Converted state dict for Neuron
    """
    new_state_dict = {}
    num_layers = config.num_hidden_layers
    dtype = neuron_config.torch_dtype

    # Detect key prefix (HF state dicts may have "model." prefix)
    sample_key = next(iter(state_dict.keys()))
    key_prefix = "model." if sample_key.startswith("model.") else ""

    for layer_idx in range(num_layers):
        layer_prefix = f"{key_prefix}layers.{layer_idx}"
        output_prefix = f"layers.{layer_idx}"

        # Check if we have MXFP4 quantized weights (blocks/scales format)
        gate_up_blocks_key = f"{layer_prefix}.mlp.experts.gate_up_proj_blocks"
        gate_up_scales_key = f"{layer_prefix}.mlp.experts.gate_up_proj_scales"
        down_blocks_key = f"{layer_prefix}.mlp.experts.down_proj_blocks"
        down_scales_key = f"{layer_prefix}.mlp.experts.down_proj_scales"

        if gate_up_blocks_key in state_dict:
            # MXFP4 quantized weights - dequantize using HF function
            logger.debug(f"Dequantizing MXFP4 weights for layer {layer_idx}")

            # Dequantize gate_up_proj: convert_moe_packed_tensors returns [E, H, 2*I] interleaved
            gate_up_blocks = state_dict[gate_up_blocks_key]
            gate_up_scales = state_dict[gate_up_scales_key]
            gate_up_proj = convert_moe_packed_tensors(gate_up_blocks, gate_up_scales, dtype=dtype)
            # De-interleave from [E, H, 2*I] interleaved to [E, H, 2*I] chunked
            gate_up_proj = convert_gate_up_proj_hf_transposed(gate_up_proj, is_bias=False)

            # Dequantize down_proj: convert_moe_packed_tensors returns [E, H, I]
            # HF's transpose at end of dequant gives us exactly what NxD expects
            down_blocks = state_dict[down_blocks_key]
            down_scales = state_dict[down_scales_key]
            down_proj = convert_moe_packed_tensors(down_blocks, down_scales, dtype=dtype)

            new_state_dict[f"{output_prefix}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj.to(dtype)
            new_state_dict[f"{output_prefix}.mlp.expert_mlps.mlp_op.down_proj.weight"] = down_proj.to(dtype)

        else:
            # Non-quantized weights (already dequantized via HF Mxfp4Config(dequantize=True))
            # HF convert_moe_packed_tensors returns weights with transpose(1,2) applied:
            # - gate_up_proj: [E, H, 2*I] INTERLEAVED along last dim - need to de-interleave
            # - down_proj: [E, H, I] which is exactly what NxD expects
            gate_up_key = f"{layer_prefix}.mlp.experts.gate_up_proj"
            down_key = f"{layer_prefix}.mlp.experts.down_proj"

            if gate_up_key in state_dict:
                gate_up_proj = state_dict[gate_up_key]
                # HF dequant returns [E, H, 2*I] interleaved, NxD expects [E, H, 2*I] chunked
                gate_up_proj = convert_gate_up_proj_hf_transposed(gate_up_proj, is_bias=False)
                new_state_dict[f"{output_prefix}.mlp.expert_mlps.mlp_op.gate_up_proj.weight"] = gate_up_proj.to(dtype)

            if down_key in state_dict:
                down_proj = state_dict[down_key]
                # HF dequant returns [E, H, I] which is exactly what NxD expects (no transpose needed)
                new_state_dict[f"{output_prefix}.mlp.expert_mlps.mlp_op.down_proj.weight"] = down_proj.to(dtype)

        # Router weights and bias
        router_weight_key = f"{layer_prefix}.mlp.router.weight"
        if router_weight_key in state_dict:
            new_state_dict[f"{output_prefix}.mlp.router.linear_router.weight"] = state_dict[router_weight_key].to(
                dtype
            )

        router_bias_key = f"{layer_prefix}.mlp.router.bias"
        if router_bias_key in state_dict:
            new_state_dict[f"{output_prefix}.mlp.router.linear_router.bias"] = state_dict[router_bias_key].to(dtype)

        # Expert bias (if present)
        gate_up_bias_key = f"{layer_prefix}.mlp.experts.gate_up_proj_bias"
        down_bias_key = f"{layer_prefix}.mlp.experts.down_proj_bias"

        if gate_up_bias_key in state_dict:
            # Bias is also interleaved, de-interleave it
            gate_up_bias = convert_gate_up_proj_hf_transposed(state_dict[gate_up_bias_key], is_bias=True)
            new_state_dict[f"{output_prefix}.mlp.expert_mlps.mlp_op.gate_up_proj.bias"] = gate_up_bias.to(dtype)
        if down_bias_key in state_dict:
            new_state_dict[f"{output_prefix}.mlp.expert_mlps.mlp_op.down_proj.bias"] = state_dict[down_bias_key].to(
                dtype
            )

        # Sinks: map self_attn.sinks -> self_attn.learned_sinks.sink
        sinks_key = f"{layer_prefix}.self_attn.sinks"
        if sinks_key in state_dict:
            new_state_dict[f"{output_prefix}.self_attn.learned_sinks.sink"] = state_dict[sinks_key].to(dtype)

        # Rank utility tensor for attention (required for tensor parallel operations)
        new_state_dict[f"{output_prefix}.self_attn.rank_util.rank"] = torch.arange(
            0, neuron_config.tp_degree, dtype=torch.int32
        )

    # Global rank utility tensor for base model
    new_state_dict["rank_util.rank"] = torch.arange(0, neuron_config.tp_degree, dtype=torch.int32)

    # Copy all non-MoE weights directly
    for key, value in state_dict.items():
        # Skip keys we've already processed
        if any(
            pattern in key
            for pattern in [
                "mlp.experts.gate_up_proj",
                "mlp.experts.down_proj",
                "mlp.router.weight",
                "mlp.router.bias",
                "self_attn.sinks",  # Already mapped to learned_sinks.sink
            ]
        ):
            continue

        # Strip "model." prefix if present for output key
        output_key = key[len(key_prefix) :] if key.startswith(key_prefix) else key

        # Direct copy for other weights (if not already in new_state_dict)
        if output_key not in new_state_dict:
            new_state_dict[output_key] = value.to(dtype) if torch.is_floating_point(value) else value

    return new_state_dict


class GptOssRotaryEmbedding(nn.Module):
    """GPT-OSS Rotary Embedding with NTK-by-parts and YaRN concentration scaling.

    This matches the NxDI implementation exactly.
    See YaRN paper: https://arxiv.org/abs/2309.00071

    Args:
        dim: Head dimension for rotary embedding
        base: Base for frequency computation (rope_theta)
        initial_context_length: Original context length for scaling computation
        scaling_factor: Factor for extending context (rope_scaling_factor)
        ntk_alpha: NTK alpha parameter (corresponds to beta_slow in HF config)
        ntk_beta: NTK beta parameter (corresponds to beta_fast in HF config)
    """

    def __init__(
        self,
        dim: int,
        base: float = 10000.0,
        initial_context_length: int = 4096,
        scaling_factor: float = 1.0,
        ntk_alpha: float = 1.0,
        ntk_beta: float = 32.0,
    ):
        super().__init__()
        self.dim = dim
        self.base = base
        self.initial_context_length = initial_context_length
        self.scaling_factor = scaling_factor
        self.ntk_alpha = ntk_alpha
        self.ntk_beta = ntk_beta
        self.register_buffer("inv_freq", None, persistent=False)
        self.concentration = None

    def get_inv_freqs_and_concentration(self, device):
        """Compute inverse frequencies and concentration factor.

        Uses NTK-by-parts interpolation when scaling_factor > 1.0.
        See YaRN paper: https://arxiv.org/abs/2309.00071
        """
        freq = self.base ** (torch.arange(0, self.dim, 2, dtype=torch.float, device=device) / self.dim)

        if self.scaling_factor > 1.0:
            # YaRN concentration factor
            concentration = 0.1 * math.log(self.scaling_factor) + 1.0

            d_half = self.dim / 2

            # NTK by parts: compute low and high frequency boundaries
            low = d_half * math.log(self.initial_context_length / (self.ntk_beta * 2 * math.pi)) / math.log(self.base)
            high = (
                d_half * math.log(self.initial_context_length / (self.ntk_alpha * 2 * math.pi)) / math.log(self.base)
            )
            assert 0 < low < high < d_half - 1, f"Invalid NTK bounds: 0 < {low} < {high} < {d_half - 1}"

            # Interpolation (scaled) and extrapolation (unscaled) inverse frequencies
            interpolation = 1.0 / (self.scaling_factor * freq)
            extrapolation = 1.0 / freq

            # Linear ramp between low and high
            ramp = (torch.arange(d_half, dtype=torch.float32, device=freq.device) - low) / (high - low)
            mask = 1 - ramp.clamp(0, 1)

            # Blend interpolation and extrapolation based on mask
            inv_freq = interpolation * (1 - mask) + extrapolation * mask
        else:
            concentration = 1.0
            inv_freq = 1.0 / freq

        return inv_freq, concentration

    @torch.no_grad()
    def forward(self, x, position_ids):
        """Compute rotary embeddings.

        Args:
            x: Input tensor of shape [batch_size, num_heads, seq_len, head_dim]
            position_ids: Position indices of shape [batch_size, seq_len]

        Returns:
            Tuple of (cos, sin) tensors for rotary embedding, scaled by concentration
        """
        if self.inv_freq is None or self.concentration is None:
            self.inv_freq, self.concentration = self.get_inv_freqs_and_concentration(x.device)

        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)

        # Apply concentration scaling (YaRN attention scaling)
        cos = emb.cos() * self.concentration
        sin = emb.sin() * self.concentration

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


class NeuronGptOssAttention(NeuronAttentionBase):
    """GPT-OSS attention layer for Neuron.

    Key differences from base:
    - Uses YaRN rotary embedding
    - Has explicit head_dim (not derived from hidden_size / num_heads)
    - Has attention bias
    """

    def __init__(
        self,
        config: GptOssConfig,
        neuron_config: NxDNeuronConfig,
    ):
        # GPT-OSS has explicit head_dim in config
        head_dim = getattr(config, "head_dim", None)
        if head_dim is None:
            head_dim = config.hidden_size // config.num_attention_heads
            logger.warning(
                f"head_dim not found in config, using computed value: {head_dim}. "
                "This may be incorrect for GPT-OSS models."
            )

        # GPT-OSS uses attention bias
        qkv_proj_bias = getattr(config, "attention_bias", True)
        o_proj_bias = getattr(config, "attention_bias", True)

        super().__init__(
            config,
            neuron_config,
            qkv_proj_bias=qkv_proj_bias,
            o_proj_bias=o_proj_bias,
            learned_sinks_size=1,  # GPT-OSS uses learned sinks
        )

        # Override head_dim after parent init
        self.head_dim = head_dim

        # Set up GPT-OSS rotary embedding with NTK-by-parts scaling
        # Extract scaling parameters from rope_scaling dict
        rope_scaling = getattr(config, "rope_scaling", {}) or {}
        scaling_factor = rope_scaling.get("factor", 32.0)
        # Note: HF config uses beta_slow/beta_fast, NxDI uses ntk_alpha/ntk_beta
        # beta_slow corresponds to ntk_alpha, beta_fast corresponds to ntk_beta
        ntk_alpha = rope_scaling.get("beta_slow", 1.0)
        ntk_beta = rope_scaling.get("beta_fast", 32.0)
        initial_context_length = rope_scaling.get("original_max_position_embeddings", 4096)

        self.rotary_emb = GptOssRotaryEmbedding(
            dim=head_dim,
            base=config.rope_theta,
            initial_context_length=initial_context_length,
            scaling_factor=scaling_factor,
            ntk_alpha=ntk_alpha,
            ntk_beta=ntk_beta,
        )


def initialize_gpt_oss_moe(
    config: GptOssConfig,
    neuron_config: NxDNeuronConfig,
) -> MoE:
    """Initialize GPT-OSS MoE module with all required settings.

    This matches NxDI's NeuronGptOssMoE initialization exactly.
    GPT-OSS uses:
    - Router with softmax activation and FP32 dtype
    - ExpertMLPsV2 with bias enabled
    - SwiGLU activation with scaling factor 1.702 and bias 1

    Args:
        config: GPT-OSS model configuration
        neuron_config: Neuron configuration

    Returns:
        Initialized MoE module
    """
    # Import GLUType for proper enum value
    from neuronx_distributed.modules.moe.moe_configs import GLUType
    from neuronx_distributed.modules.moe.moe_process_group import init_tensor_expert_parallel_moe_process_groups

    # Initialize MoE parallel state (required for ExpertMLPsV2)
    # This sets up _MOE_CTE_TENSOR_MODEL_PARALLEL_GROUP and other MoE-specific groups
    tp_degree = neuron_config.tp_degree
    init_tensor_expert_parallel_moe_process_groups(
        tkg_tp_degree=tp_degree,
        tkg_ep_degree=1,  # No expert parallelism
        cte_tp_degree=tp_degree,
        cte_ep_degree=1,  # No expert parallelism
    )

    # Router with softmax activation and bias
    router = RouterTopK(
        num_experts=config.num_local_experts,
        top_k=config.num_experts_per_tok,
        hidden_size=config.hidden_size,
        dtype=torch.float32,  # Router uses FP32
        act_fn="softmax",
        sequence_parallel_enabled=getattr(neuron_config, "sequence_parallel_enabled", False),
        sequence_dimension=1,
        bias=True,  # GPT-OSS uses router bias
    )

    # Expert MLPs with bias and GPT-OSS specific settings
    expert_mlps = ExpertMLPsV2(
        routed_experts_mlp_config=RoutedExpertsMLPOpsConfig(
            num_experts=config.num_local_experts,
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            top_k=config.num_experts_per_tok,
            hidden_act=config.hidden_act,  # "sigmoid" for GPT-OSS
            glu_mlp=True,  # GPT-OSS uses SwiGLU
            bias=True,  # GPT-OSS uses expert bias
            glu_type=GLUType.SWIGLU,
            hidden_act_scaling_factor=1.702,
            hidden_act_bias=1,
            early_expert_affinity_modulation=False,
            normalize_top_k_affinities=False,
        ),
        sequence_parallel_enabled=getattr(neuron_config, "sequence_parallel_enabled", False),
        dtype=neuron_config.torch_dtype,
        tensor_model_parallel_group=parallel_state.get_tensor_model_parallel_group(),
        expert_model_parallel_group=parallel_state.get_expert_model_parallel_group(),
    )

    moe = MoE(
        router=router,
        expert_mlps=expert_mlps,
        sequence_parallel_enabled=getattr(neuron_config, "sequence_parallel_enabled", False),
        sequence_dimension=1,
    )

    # Set MoE module in eval mode
    moe.eval()
    return moe


class NeuronGptOssDecoderLayer(nn.Module):
    """GPT-OSS decoder layer for Neuron inference.

    Implements pre-norm transformer layer with:
    - Self-attention with YaRN RoPE
    - MoE FFN with top-k routing
    """

    def __init__(self, config: GptOssConfig, neuron_config: NxDNeuronConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.layer_idx = layer_idx

        # Attention
        self.self_attn = NeuronGptOssAttention(config, neuron_config)

        # MoE module - use GPT-OSS specific initialization
        self.mlp = initialize_gpt_oss_moe(config, neuron_config)

        # Layer norms
        self.input_layernorm = NeuronRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = NeuronRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        **kwargs,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        """Forward pass for decoder layer.

        Args:
            hidden_states: Input tensor of shape [batch, seq_len, hidden_size]
            attention_mask: Attention mask
            position_ids: Position indices
            past_key_value: Cached key/value states

        Returns:
            Tuple of (hidden_states, present_key_value, cos_cache, sin_cache)
        """
        if "padding_mask" in kwargs:
            warnings.warn("Passing `padding_mask` is deprecated. Please use `attention_mask` instead.")

        residual = hidden_states

        # Pre-norm and self-attention
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, present_key_value, cos_cache, sin_cache = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # MoE FFN
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)[0]  # MoE returns (output, router_logits)
        hidden_states = residual + hidden_states

        return (hidden_states, present_key_value, cos_cache, sin_cache)


class NxDGptOssModel(NxDDecoderModelForCausalLM):
    """GPT-OSS model for NxD inference.

    This model is traced for compilation.
    """

    def __init__(self, config: GptOssConfig, neuron_config: NxDNeuronConfig):
        super().__init__(config, neuron_config)

        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            dtype=neuron_config.torch_dtype,
            shard_across_embedding=True,
        )
        self.layers = nn.ModuleList(
            [
                NeuronGptOssDecoderLayer(config, neuron_config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm = NeuronRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            gather_output=not neuron_config.on_device_sampling,
            bias=False,
        )


class GptOssNxDModelForCausalLM(NxDModelForCausalLM):
    """GPT-OSS causal language model for NxD inference.

    This is the main entry point for GPT-OSS inference on Neuron.
    """

    _model_cls = NxDGptOssModel

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: PretrainedConfig, neuron_config: NxDNeuronConfig
    ) -> dict:
        return convert_gpt_oss_to_neuron_state_dict(state_dict, config, neuron_config)

    @classmethod
    def get_compiler_args(cls, neuron_config: NxDNeuronConfig) -> str:
        compiler_args = "--enable-saturate-infinity --enable-mixed-precision-accumulation --model-type transformer -O1"
        # Add flags for cc-overlap
        compiler_args += " --tensorizer-options='--enable-ccop-compute-overlap --cc-pipeline-tiling-factor=2'"
        compiler_args += " --auto-cast=none"
        # Enable vector-offset DGE
        compiler_args += " --internal-enable-dge-levels vector_dynamic_offsets"
        return compiler_args

    @classmethod
    def _get_neuron_config(
        cls,
        checkpoint_id: str,
        checkpoint_revision: str,
        instance_type: str,
        batch_size: int,
        sequence_length: int,
        tensor_parallel_size: int,
        dtype: torch.dtype,
    ):
        return NxDNeuronConfig(
            checkpoint_id=checkpoint_id,
            checkpoint_revision=checkpoint_revision,
            batch_size=batch_size,
            sequence_length=sequence_length,
            tp_degree=tensor_parallel_size,
            torch_dtype=dtype,
            target=instance_type,
            # MoE-specific settings
            glu_mlp=True,  # GPT-OSS uses SwiGLU (gate * up)
        )
