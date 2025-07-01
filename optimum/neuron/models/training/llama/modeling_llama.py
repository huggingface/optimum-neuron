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
"""LlaMa model implementation for Neuron."""

from functools import partial
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.processing_utils import Unpack
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import LossKwargs, can_return_tuple, logging

from ....utils import is_neuronx_distributed_available, is_torch_xla_available
from ..config import TrainingNeuronConfig
from ..loss_utils import ForCausalLMLoss
from ..modeling_utils import NeuronModelMixin
from ..pipeline_utils import dynamic_torch_fx_wrap
from ..transformations_utils import (
    CustomModule,
    FusedLinearsSpec,
    GQAQKVColumnParallelLinearSpec,
    ModelWeightTransformationSpecs,
)


if is_torch_xla_available():
    from torch_xla.utils.checkpoint import checkpoint

if is_neuronx_distributed_available():
    import neuronx_distributed.parallel_layers.utils as neuronx_dist_utils
    from neuronx_distributed.kernels.flash_attn import nki_flash_attn_func
    from neuronx_distributed.modules.qkv_linear import GQAQKVColumnParallelLinear
    from neuronx_distributed.parallel_layers.layers import (
        ColumnParallelLinear,
        ParallelEmbedding,
        RowParallelLinear,
    )
    from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_size


logger = logging.get_logger(__name__)


def _init_normal(std, w):
    return nn.init.normal_(w, mean=0.0, std=std)


class LlamaRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6, sequence_parallel_enabled=False):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        setattr(self.weight, "sequence_parallel_enabled", sequence_parallel_enabled)
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(LlamaRMSNorm)


class LlamaRotaryEmbedding(nn.Module):
    def __init__(self, config: LlamaConfig, device=None):
        super().__init__()
        # BC: "rope_type" was originally "type"
        if hasattr(config, "rope_scaling") and config.rope_scaling is not None:
            self.rope_type = config.rope_scaling.get("rope_type", config.rope_scaling.get("type"))
        else:
            self.rope_type = "default"
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config
        self.rope_init_fn = ROPE_INIT_FUNCTIONS[self.rope_type]

        inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.original_inv_freq = self.inv_freq

    def _dynamic_frequency_update(self, position_ids, device):
        """
        dynamic RoPE layers should recompute `inv_freq` in the following situations:
        1 - growing beyond the cached sequence length (allow scaling)
        2 - the current sequence length is in the original scale (avoid losing precision with small sequences)
        """
        seq_len = torch.max(position_ids) + 1
        if seq_len > self.max_seq_len_cached:  # growth
            inv_freq, self.attention_scaling = self.rope_init_fn(self.config, device, seq_len=seq_len)
            self.register_buffer("inv_freq", inv_freq, persistent=False)  # TODO joao: may break with compilation
            self.max_seq_len_cached = seq_len

        if seq_len < self.original_max_seq_len and self.max_seq_len_cached > self.original_max_seq_len:  # reset
            # This .to() is needed if the model has been moved to a device after being initialized (because
            # the buffer is automatically moved, but not the original copy)
            self.original_inv_freq = self.original_inv_freq.to(device)
            self.register_buffer("inv_freq", self.original_inv_freq, persistent=False)
            self.max_seq_len_cached = self.original_max_seq_len

    @torch.no_grad()
    def forward(self, x, position_ids):
        if "dynamic" in self.rope_type:
            self._dynamic_frequency_update(position_ids, device=x.device)

        # Core RoPE block
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()
        # Force float32 (see https://github.com/huggingface/transformers/pull/29285)
        device_type = x.device.type
        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()

        # Advanced RoPE types (e.g. yarn) apply a post-processing scaling factor, equivalent to scaling attention
        cos = cos * self.attention_scaling
        sin = sin * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x, flash_attn, transpose):
    if flash_attn and transpose:
        x1 = x[:, :, : x.shape[-2] // 2, :]
        x2 = x[:, :, x.shape[-2] // 2 :, :]
        return torch.cat((-x2, x1), dim=-2)
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q, k, cos, sin, position_ids=None, unsqueeze_dim=1, flash_attn=False, transpose_nki_inputs=False
):
    if unsqueeze_dim != 1:
        raise NotImplementedError(
            f"Unsqueeze dimension {unsqueeze_dim} is not supported. Only unsqueeze_dim=1 is currently implemented."
        )
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    if flash_attn and transpose_nki_inputs:
        cos = cos.permute(0, 1, 3, 2)  # [bs, 1, dim, seq_len]
        sin = sin.permute(0, 1, 3, 2)  # [bs, 1, dim, seq_len]
    q_embed = (q * cos) + (rotate_half(q, flash_attn, transpose_nki_inputs) * sin)
    k_embed = (k * cos) + (rotate_half(k, flash_attn, transpose_nki_inputs) * sin)
    return q_embed, k_embed


class LlamaMLP(nn.Module, CustomModule):
    def __init__(self, config, trn_config: TrainingNeuronConfig):
        nn.Module.__init__(self)
        self.config = config
        self.trn_config = trn_config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]

        tp_size = get_tensor_model_parallel_size()
        if self.intermediate_size % tp_size != 0:
            raise RuntimeError(
                f"Intermediate size {self.intermediate_size} must be divisible by the tensor model parallel size "
                f"{tp_size}."
            )
        self.split_size = self.intermediate_size // tp_size

        init_method = partial(_init_normal, config.initializer_range)

        # Defines the MLP weight transformation specs
        self.specs = ModelWeightTransformationSpecs(
            specs=FusedLinearsSpec(
                fused_linear_name="gate_up_proj",
                linear_names=["gate_proj", "up_proj"],
                bias=False,
                fuse_axis="column",
                original_dims=[self.intermediate_size] * 2,
            )
        )
        self.gate_up_proj = ColumnParallelLinear(
            self.hidden_size,
            2 * self.intermediate_size,
            stride=2,
            bias=False,
            gather_output=False,
            init_method=init_method,
            sequence_parallel_enabled=self.trn_config.sequence_parallel_enabled,
            sequence_dimension=0,
            dtype=self.config.torch_dtype,
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=init_method,
            sequence_parallel_enabled=self.trn_config.sequence_parallel_enabled,
            sequence_dimension=0,
            dtype=self.config.torch_dtype,
        )

    def forward(self, x):
        gate_proj, up_proj = self.gate_up_proj(x).split(self.split_size, dim=2)

        def activation_mlp(gate_proj, up_proj):
            activation_output = self.act_fn(gate_proj)
            return activation_output * up_proj

        # We checkpoint the MLP compute too, since we see extra data movement which is more
        # expensive than the recompute in this case.
        if self.trn_config.gradient_checkpointing:
            intermediate_states = checkpoint(activation_mlp, gate_proj, up_proj)
        else:
            intermediate_states = self.act_fn(gate_proj) * up_proj
        down_proj = self.down_proj(intermediate_states)

        return down_proj


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    causal: bool = False,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        attn_weights = attn_weights + causal_mask
    elif causal:
        # ** Difference from the original eager_attention_forward implementation **
        # Instead of using the attention mask, we re-compute a causal mask.
        # It is more efficient, the only issue is that we do not support custom attention masks.
        causal_mask = torch.triu(torch.ones((1, 1, query.size(2), key.size(2)), device="xla"), diagonal=1).bool()
        min_value = torch.finfo(attn_weights.dtype).min
        attn_weights = attn_weights.masked_fill_(causal_mask, min_value)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)

    return attn_output, attn_weights


class LlamaAttention(nn.Module, CustomModule):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, trn_config: TrainingNeuronConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        if (self.hidden_size % self.num_heads) != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.trn_config = trn_config

        init_method = partial(_init_normal, config.initializer_range)

        tp_size = get_tensor_model_parallel_size()
        self.qkv_linear = (self.num_key_value_heads < tp_size) or (self.num_key_value_heads % tp_size != 0)
        if self.qkv_linear:
            if trn_config.kv_size_multiplier is None:
                self.kv_size_multiplier = trn_config.auto_kv_size_multiplier(self.num_key_value_heads)
            else:
                self.kv_size_multiplier = trn_config.kv_size_multiplier
        else:
            self.kv_size_multiplier = 1

        self.specs = ModelWeightTransformationSpecs()

        if self.qkv_linear:
            self.qkv_proj = GQAQKVColumnParallelLinear(
                self.hidden_size,
                [self.num_heads * self.head_dim, self.num_key_value_heads * self.head_dim],
                bias=False,
                gather_output=False,
                init_method=init_method,
                sequence_parallel_enabled=trn_config.sequence_parallel_enabled,
                kv_size_multiplier=self.kv_size_multiplier,
                fuse_qkv=trn_config.fuse_qkv,
                dtype=self.config.torch_dtype,
            )

            gqa_qkv_specs = GQAQKVColumnParallelLinearSpec(
                gqa_qkv_projection_name="qkv_proj",
                query_projection_name="q_proj",
                key_projection_name="k_proj",
                value_projection_name="v_proj",
                output_projection_name="o_proj",
                num_attention_heads=self.num_heads,
                num_key_value_heads=self.num_key_value_heads,
                kv_size_multiplier=self.kv_size_multiplier,
                q_output_size_per_partition=self.qkv_proj.q_output_size_per_partition,
                kv_output_size_per_partition=self.qkv_proj.kv_output_size_per_partition,
                fuse_qkv=trn_config.fuse_qkv,
                bias=False,
            )
            self.specs.add_spec(gqa_qkv_specs)
        elif trn_config.fuse_qkv and self.num_heads == self.num_key_value_heads:
            self.qkv_proj = ColumnParallelLinear(
                self.hidden_size,
                3 * self.num_heads * self.head_dim,
                stride=3,
                bias=False,
                gather_output=False,
                init_method=init_method,
                sequence_parallel_enabled=trn_config.sequence_parallel_enabled,
                sequence_dimension=0,
                dtype=self.config.torch_dtype,
            )
            self.specs.add_spec(
                FusedLinearsSpec(
                    fused_linear_name="qkv_proj",
                    linear_names=["q_proj", "k_proj", "v_proj"],
                    bias=False,
                    fuse_axis="column",
                    original_dims=[self.num_heads * self.head_dim] * 3,
                )
            )
            self.split_size = self.num_heads * self.head_dim // tp_size
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=init_method,
                sequence_parallel_enabled=trn_config.sequence_parallel_enabled,
                sequence_dimension=0,
                dtype=self.config.torch_dtype,
            )
            self.k_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=init_method,
                sequence_parallel_enabled=trn_config.sequence_parallel_enabled,
                sequence_dimension=0,
                dtype=self.config.torch_dtype,
            )
            self.v_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=init_method,
                sequence_parallel_enabled=trn_config.sequence_parallel_enabled,
                sequence_dimension=0,
                dtype=self.config.torch_dtype,
            )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=init_method,
            sequence_parallel_enabled=trn_config.sequence_parallel_enabled,
            sequence_dimension=0,
            dtype=self.config.torch_dtype,
        )
        self.num_heads = neuronx_dist_utils.divide(config.num_attention_heads, tp_size)
        self.num_key_value_heads = neuronx_dist_utils.divide(
            config.num_key_value_heads * self.kv_size_multiplier, tp_size
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

    @property
    def use_flash_attention_v2(self):
        return self.config._attn_implementation == "flash_attention_2"

    @property
    def use_ring_attention(self):
        # We do not support ring attention for now, it should be added.
        return False

    def permute_qkv_for_attn(
        self, query_states, key_states, value_states, bsz, q_len, num_heads, num_key_value_heads, head_dim
    ):
        if self.trn_config.transpose_nki_inputs and self.use_flash_attention_v2 and not self.use_ring_attention:
            # Expected output shape is (batch size, num heads, head dim, sequence length)
            if self.trn_config.sequence_parallel_enabled:
                query_states = query_states.view(q_len, bsz, num_heads, head_dim).permute(1, 2, 3, 0)
                key_states = key_states.view(q_len, bsz, num_key_value_heads, head_dim).permute(1, 2, 3, 0)
                value_states = value_states.view(q_len, bsz, num_key_value_heads, head_dim).permute(1, 2, 3, 0)
            else:
                query_states = query_states.view(bsz, q_len, num_heads, head_dim).permute(0, 2, 3, 1)
                key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).permute(0, 2, 3, 1)
                value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).permute(0, 2, 3, 1)
        elif self.trn_config.sequence_parallel_enabled:
            # Expected output shape is (batch size, num heads, sequence length, head dim)
            query_states = query_states.view(q_len, bsz, num_heads, head_dim).permute(1, 2, 0, 3)
            key_states = key_states.view(q_len, bsz, num_key_value_heads, head_dim).permute(1, 2, 0, 3)
            value_states = value_states.view(q_len, bsz, num_key_value_heads, head_dim).permute(1, 2, 0, 3)
        else:
            # Expected output shape is (batch size, num heads, sequence length, head dim)
            query_states = query_states.view(bsz, q_len, num_heads, head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, num_key_value_heads, head_dim).transpose(1, 2)

        return query_states, key_states, value_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if self.trn_config.sequence_parallel_enabled:
            q_len, bsz, _ = hidden_states.size()
            q_len = q_len * get_tensor_model_parallel_size()
        else:
            bsz, q_len, _ = hidden_states.size()

        if self.trn_config.fuse_qkv and self.num_heads == self.num_key_value_heads and self.kv_size_multiplier == 1:
            qkv_states = self.qkv_proj(hidden_states)
            query_states, key_states, value_states = qkv_states.split(self.split_size, dim=2)
        elif self.qkv_linear:
            query_states, key_states, value_states = self.qkv_proj(hidden_states)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        query_states, key_states, value_states = self.permute_qkv_for_attn(
            query_states, key_states, value_states, bsz, q_len, self.num_heads, self.num_key_value_heads, self.head_dim
        )

        cos, sin = position_embeddings

        query_states, key_states = apply_rotary_pos_emb(
            query_states,
            key_states,
            cos,
            sin,
            flash_attn=self.use_flash_attention_v2,
            transpose_nki_inputs=self.trn_config.transpose_nki_inputs,
        )
        if self.use_flash_attention_v2:
            attn_output = nki_flash_attn_func(
                query_states,
                repeat_kv(key_states, self.num_key_value_groups),
                repeat_kv(value_states, self.num_key_value_groups),
                dropout_p=0.0,  # We never apply dropout in the flash attention path because it produces NaNs.
                softmax_scale=self.scaling,
                causal=True,
                mixed_precision=True,
                transpose_nki_inputs=self.trn_config.transpose_nki_inputs,
            )
            attn_output = nn.functional.dropout(attn_output, p=0.0 if not self.training else self.attention_dropout)
            attn_weights = None
        else:
            attn_output, attn_weights = eager_attention_forward(
                self,
                query_states,
                key_states,
                value_states,
                attention_mask,
                self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                causal=attention_mask is None,
                **kwargs,
            )

        if self.trn_config.sequence_parallel_enabled:
            attn_output = attn_output.permute(2, 0, 1, 3)
            attn_output = attn_output.reshape(q_len, bsz, self.num_heads * self.head_dim)
        else:
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaConfig, trn_config: TrainingNeuronConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, trn_config=trn_config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config, trn_config)
        self.input_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, sequence_parallel_enabled=trn_config.sequence_parallel_enabled
        )
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, sequence_parallel_enabled=trn_config.sequence_parallel_enabled
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,  # necessary, but kept here for BC
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]]:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)

        return outputs


class LlamaPreTrainedModel(PreTrainedModel):
    config_class = LlamaConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["LlamaDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn_2 = True
    _supports_sdpa = False
    _supports_flex_attn = False
    _supports_cache_class = False
    _supports_quantized_cache = False
    _supports_static_cache = False
    _supports_attention_backend = False

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class LlamaModel(NeuronModelMixin, LlamaPreTrainedModel):
    def __init__(self, config: LlamaConfig, trn_config: TrainingNeuronConfig):
        LlamaPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.trn_config = trn_config

        init_method = partial(_init_normal, config.initializer_range)
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            self.padding_idx,
            init_method=init_method,
            sequence_parallel_enabled=trn_config.sequence_parallel_enabled,
            dtype=config.torch_dtype,
        )
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config, trn_config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, sequence_parallel_enabled=trn_config.sequence_parallel_enabled
        )
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        self.gradient_checkpointing = self.trn_config.gradient_checkpointing

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> BaseModelOutputWithPast:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        current_length = (
            inputs_embeds.size(0) * self.trn_config.tensor_parallel_size
            if self.trn_config.sequence_parallel_enabled
            else inputs_embeds.size(1)
        )
        cache_position = torch.arange(0, current_length, device=inputs_embeds.device)

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        if self.trn_config.recompute_causal_mask:
            causal_mask = None
        else:
            causal_mask = self._update_causal_mask(attention_mask, inputs_embeds, cache_position)

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = checkpoint(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    output_attentions,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    output_attentions=output_attentions,
                    position_embeddings=position_embeddings,
                    **flash_attn_kwargs,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output

    def _update_causal_mask(
        self,
        attention_mask: torch.Tensor,
        input_tensor: torch.Tensor,
        cache_position: torch.Tensor,
    ):
        if self.config._attn_implementation == "flash_attention_2":
            if attention_mask is not None and (attention_mask == 0.0).any():
                raise RuntimeError(f"Only a causal mask is supported with {self.config._attn_implementation}.")
            return None

        dtype, device = input_tensor.dtype, input_tensor.device
        if self.trn_config.sequence_parallel_enabled:
            sequence_length = input_tensor.shape[0] * self.trn_config.tensor_parallel_size
        else:
            sequence_length = input_tensor.shape[1]

        target_length = attention_mask.shape[-1] if isinstance(attention_mask, torch.Tensor) else sequence_length + 1

        # In case the provided `attention` mask is 2D, we generate a causal mask here (4D).
        batch_size = input_tensor.shape[1] if self.trn_config.sequence_parallel_enabled else input_tensor.shape[0]
        causal_mask = self._prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=batch_size,
        )

        return causal_mask

    @staticmethod
    @dynamic_torch_fx_wrap
    def _prepare_4d_causal_attention_mask_with_cache_position(
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        **kwargs,
    ):
        if attention_mask is not None and attention_mask.dim() == 4:
            # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
            causal_mask = attention_mask
        else:
            min_dtype = torch.finfo(dtype).min
            causal_mask = torch.full(
                (sequence_length, target_length), fill_value=min_dtype, dtype=dtype, device=device
            )
            if sequence_length != 1:
                causal_mask = torch.triu(causal_mask, diagonal=1)
            causal_mask *= torch.arange(target_length, device=device) > cache_position.reshape(-1, 1)
            causal_mask = causal_mask[None, None, :, :].expand(batch_size, 1, -1, -1)
            if attention_mask is not None:
                causal_mask = causal_mask.clone()  # copy to contiguous memory for in-place edit
                mask_length = attention_mask.shape[-1]
                padding_mask = causal_mask[:, :, :, :mask_length] + attention_mask[:, None, None, :].to(
                    causal_mask.device
                )
                padding_mask = padding_mask == 0
                causal_mask[:, :, :, :mask_length] = causal_mask[:, :, :, :mask_length].masked_fill(
                    padding_mask, min_dtype
                )

        return causal_mask


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


class LlamaForCausalLM(NeuronModelMixin, LlamaPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    SUPPORTS_PIPELINE_PARALLELISM = True
    PIPELINE_TRANSFORMER_LAYER_CLS = LlamaDecoderLayer
    PIPELINE_INPUT_NAMES = ["input_ids", "attention_mask", "labels"]
    PIPELINE_LEAF_MODULE_CLASSE_NAMES = ["LlamaRMSNorm", "LlamaRotaryEmbedding"]

    def __init__(self, config, trn_config: TrainingNeuronConfig):
        LlamaPreTrainedModel.__init__(self, config)
        self.model = LlamaModel(config, trn_config)
        self.trn_config = trn_config

        init_method = partial(_init_normal, config.initializer_range)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            gather_output=False,
            init_method=init_method,
            sequence_parallel_enabled=trn_config.sequence_parallel_enabled,
            sequence_dimension=0,
            dtype=self.config.torch_dtype,
        )

        self.vocab_size = config.vocab_size // get_tensor_model_parallel_size()

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @can_return_tuple
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            **kwargs,
        )

        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states)

        if self.trn_config.sequence_parallel_enabled:
            logits = logits.transpose(0, 1).contiguous()

        loss = None
        if labels is not None:
            loss = ForCausalLMLoss(logits=logits, labels=labels, vocab_size=self.vocab_size, **kwargs)

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
