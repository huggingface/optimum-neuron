# coding=utf-8
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""PyTorch LLaMA model for NXD inference."""
import warnings

import torch
from neuronx_distributed.parallel_layers import parallel_state  # noqa: E402
from neuronx_distributed.parallel_layers.layers import (  # noqa: E402
    ColumnParallelLinear,  # noqa: E402
    ParallelEmbedding,  # noqa: E402
    RowParallelLinear,  # noqa: E402
)  # noqa: E402
from torch import nn
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaMLP,
    LlamaModel,
)


from models.attention.attention_base import NeuronAttentionBase  # noqa: E402
from models.attention.utils import RotaryEmbedding  # noqa: E402
from models.custom_calls import CustomRMSNorm  # noqa: E402


class NeuronLlamaMLP(LlamaMLP):
    """
    This class just replace the linear layers (gate_proj, up_proj and down_proj) with column and row parallel layers
    """

    def __init__(self, config: LlamaConfig):
        assert parallel_state.model_parallel_is_initialized()
        super().__init__(config)

        self.gate_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=config.torch_dtype,
            pad=True,
        )
        self.up_proj = ColumnParallelLinear(
            self.hidden_size,
            self.intermediate_size,
            bias=False,
            gather_output=False,
            dtype=config.torch_dtype,
            pad=True,
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            dtype=config.torch_dtype,
            pad=True,
        )


class NeuronLlamaAttention(NeuronAttentionBase):
    """
    Compared with LlamaAttention, this class just
    1. replaces the q_proj, k_proj, v_proj with column parallel layer
    2. replaces the o_proj with row parallel layer
    3. update self.num_head to be self.num_head / tp_degree
    4. update self.num_key_value_heads to be self.num_key_value_heads / tp_degree
    5. update forward() method to adjust to changes from self.num_head
    """

    def __init__(self, config: LlamaConfig, layer_idx: int):
        super().__init__()

        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.padding_side = config.padding_side
        self.torch_dtype = config.torch_dtype

        if parallel_state.model_parallel_is_initialized():
            self.tp_degree = parallel_state.get_tensor_model_parallel_size()
        else:
            self.tp_degree = 1
        self.fused_qkv = False
        self.clip_qkv = None

        self.init_gqa_properties()

        self.init_rope()

    def init_rope(self):
        if not hasattr(self.config, "rope_scaling") or self.config.rope_scaling is None:
            self.rotary_emb = RotaryEmbedding(
                self.head_dim,
                max_position_embeddings=self.max_position_embeddings,
                base=self.rope_theta,
            )
        else:
            scaling_type = self.config.rope_scaling["type"]
            scaling_factor = self.config.rope_scaling["factor"]
            if scaling_type == "linear":
                self.rotary_emb = LlamaLinearScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            elif scaling_type == "dynamic":
                self.rotary_emb = LlamaDynamicNTKScalingRotaryEmbedding(
                    self.head_dim,
                    max_position_embeddings=self.max_position_embeddings,
                    scaling_factor=scaling_factor,
                    base=self.rope_theta,
                )
            else:
                raise ValueError(f"Unknown RoPE scaling type {scaling_type}")


class NeuronLlamaDecoderLayer(LlamaDecoderLayer):
    """
    Just replace the attention with the NXD version, and MLP with the NXD version
    """

    def __init__(self, config: LlamaConfig, layer_idx: int):
        assert parallel_state.model_parallel_is_initialized()
        super().__init__(config, layer_idx)
        if config._attn_implementation != "eager":
            warnings.warn(
                "Ignoring _attn_implementation = {config._attn_implementation} parameter: only default attention is supported for Neuron"
            )
        # Replace standard layers by parallel layers
        self.self_attn = NeuronLlamaAttention(config=config, layer_idx=layer_idx)
        self.mlp = NeuronLlamaMLP(config)
        self.input_layernorm = CustomRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = CustomRMSNorm(config.hidden_size, eps=config.rms_norm_eps)


class NeuronLlamaModel(LlamaModel):
    """
    The neuron version of the LlamaModel
    """

    def __init__(self, config: LlamaConfig):
        assert parallel_state.model_parallel_is_initialized()
        super().__init__(config)
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            config.pad_token_id,
            dtype=config.torch_dtype,
            shard_across_embedding=True,
            # We choose to shard across embedding dimension because this stops XLA from introducing
            # rank specific constant parameters into the HLO. We could shard across vocab, but that
            # would require us to use non SPMD parallel_model_trace.
            pad=True,
        )
        self.lm_head = ColumnParallelLinear(config.hidden_size, config.vocab_size, bias=False, pad=True)

        self.layers = nn.ModuleList(
            [NeuronLlamaDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = CustomRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids,
        attention_mask,
        position_ids,
        past_key_values,
    ):
        # We need to override forward because the standard forward makes too much assumptions
        # about the cache implementation

        batch_size, seq_length = input_ids.shape[:2]

        inputs_embeds = self.embed_tokens(input_ids)
        position_ids = position_ids.view(-1, seq_length).long()

        # embed positions
        hidden_states = inputs_embeds

        # decoder layers
        next_decoder_cache = ()

        for decoder_layer in self.layers:

            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=False,
                use_cache=True,
            )

            hidden_states = layer_outputs[0]

            next_decoder_cache += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        hidden_size = hidden_states.shape[-1]
        index = torch.max(position_ids, dim=1, keepdim=True).indices
        index = index.unsqueeze(1).expand(batch_size, 1, hidden_size)
        hidden_states = torch.gather(hidden_states, dim=1, index=index)

        logits = self.lm_head(hidden_states)
        logits = logits.float()

        return logits, next_decoder_cache
