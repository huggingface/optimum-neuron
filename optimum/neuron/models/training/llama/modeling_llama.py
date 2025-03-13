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
"""LlaMa model implementation for Neuron"""

import math
import os
from functools import partial
from typing import List, Optional, Tuple, Union, Callable

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
import torch_xla.core.xla_model as xm
from torch_xla.utils.checkpoint import checkpoint
from torch import nn

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutputWithPast,
    CausalLMOutputWithPast,
)
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LLAMA_INPUTS_DOCSTRING,
    LLAMA_START_DOCSTRING,
)
from transformers.models.llama.modeling_llama import LlamaAttention as LlamaAttentionHF
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer as LlamaDecoderLayerHF,
)
from transformers.models.llama.modeling_llama import (
    LlamaForCausalLM as LlamaForCausalLMHF,
)
from transformers.models.llama.modeling_llama import (
    LlamaForSequenceClassification,
)
from transformers.models.llama.modeling_llama import LlamaMLP as LlamaMLPHF
from transformers.models.llama.modeling_llama import LlamaModel as LlamaModelHF
from transformers.models.llama.modeling_llama import LlamaPreTrainedModel
from transformers.models.llama.modeling_llama import LlamaRMSNorm as LlamaRMSNormHF
from transformers.models.llama.modeling_llama import (
    KwargsForCausalLM,
    LlamaRotaryEmbedding,
    apply_rotary_pos_emb,
    repeat_kv,
    rotate_half,
)
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.processing_utils import Unpack
from transformers.cache_utils import Cache, DynamicCache, StaticCache
from transformers.utils import (
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    logging,
    replace_return_docstrings,
)

import neuronx_distributed.parallel_layers.utils as neuronx_dist_utils
from neuronx_distributed.kernels.flash_attn import nki_flash_attn_func
from neuronx_distributed.modules.qkv_linear import GQAQKVColumnParallelLinear
from neuronx_distributed.parallel_layers import mappings
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)
from neuronx_distributed.parallel_layers.loss_functions import parallel_cross_entropy
from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_size

from ....accelerate import ModelParallelismConfig
from ...loss_utils import ForCausalLMLoss
from .. import NeuronModelMixin


logger = logging.get_logger(__name__)

def _init_normal(std, w):
    return nn.init.normal_(w, mean=0.0, std=std)


class LlamaRMSNorm(LlamaRMSNormHF):
    def __init__(self, hidden_size, eps=1e-6, sequence_parallel_enabled=False):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__(hidden_size, eps=eps)
        setattr(self.weight, "sequence_parallel_enabled", sequence_parallel_enabled)


class LlamaMLP(LlamaMLPHF):
    def __init__(self, config, mp_config: ModelParallelismConfig):
        nn.Module.__init__(self)
        self.config = config
        self.mp_config=mp_config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.act_fn = ACT2FN[config.hidden_act]

        init_method = partial(_init_normal, config.initializer_range)
        self.fused_linears = {
            "gate_up_proj": ["gate_proj", "up_proj"],
        }
        self.gate_up_proj = ColumnParallelLinear(
            self.hidden_size,
            2 * self.intermediate_size,
            stride=2,
            bias=False,
            gather_output=False,
            init_method=init_method,
            sequence_parallel_enabled=self.mp_config.sequence_parallel_enabled,
            dtype=self.config.torch_dtype,
        )
        self.down_proj = RowParallelLinear(
            self.intermediate_size,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=init_method,
            sequence_parallel_enabled=self.mp_config.sequence_parallel_enabled,
            dtype=self.config.torch_dtype,
        )
        self.split_size = self.intermediate_size // get_tensor_model_parallel_size()

    def forward(self, x):
        gate_proj, up_proj = self.gate_up_proj(x).split(self.split_size, dim=2)

        def activation_mlp(gate_proj, up_proj):
            activation_output = self.act_fn(gate_proj)
            return activation_output * up_proj

        # We checkpoint the MLP compute too, since we see extra data movement which is more
        # expensive than the recompute in this case.
        if self.mp_config.gradient_checkpointing:
            intermediate_states = checkpoint(activation_mlp, gate_proj, up_proj)
        else:
            intermediate_states = self.act_fn(gate_proj) * up_proj
        down_proj = self.down_proj(intermediate_states)

        return down_proj

def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        # Instead of using the attention mask, we re-compute a causal mask.
        # It is more efficient, the only issue is that we do not support custom attention masks.
        # Change this if a customer requests for it.
        causal_mask = torch.triu(torch.ones((1, 1, query.size(2), key.size(2)), device="xla"), diagonal=1).bool()
        min_value = torch.finfo(attn_weights.dtype).min
        attn_weights = attn_weights.masked_fill_(causal_mask, min_value)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value_states)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights

class LlamaAttention(LlamaAttentionHF):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config: LlamaConfig, mp_config: ModelParallelismConfig, layer_idx: int):
        nn.Module.__init__(self)
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", self.hidden_size // self.num_heads)
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.mp_config = mp_config

        init_method = partial(_init_normal, config.initializer_range)
    
        self.qkv_linear = self.num_key_value_heads < get_tensor_model_parallel_size()
        if self.qkv_linear:
            if mp_config.kv_size_multiplier is None:
                self.kv_size_multiplier = mp_config.auto_kv_size_multiplier(self.num_key_value_heads)
            else:
                self.kv_size_multiplier = mp_config.kv_size_multiplier
        else:
            self.kv_size_multiplier = 1

        if self.qkv_linear:
            self.qkv_proj = GQAQKVColumnParallelLinear(
                self.hidden_size,
                [self.num_heads * self.head_dim, self.num_key_value_heads * self.head_dim],
                bias=False,
                gather_output=False,
                init_method=init_method,
                sequence_parallel_enabled=mp_config.sequence_parallel_enabled,
                kv_size_multiplier=self.kv_size_multiplier,
                fuse_qkv=mp_config.fuse_qkv,
                dtype=self.config.torch_dtype,
            )
        elif mp_config.fuse_qkv and self.num_heads == self.num_key_value_heads:
            self.fused_linears = {
                "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            }
            self.qkv_proj = ColumnParallelLinear(
                self.hidden_size,
                3 * self.num_heads * self.head_dim,
                stride=3,
                bias=False,
                gather_output=False,
                init_method=init_method,
                sequence_parallel_enabled=mp_config.sequence_parallel_enabled,
                dtype=self.config.torch_dtype,
            )
            self.split_size = self.num_heads * self.head_dim // get_tensor_model_parallel_size()
        else:
            self.q_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=init_method,
                sequence_parallel_enabled=mp_config.sequence_parallel_enabled,
                dtype=self.config.torch_dtype,
            )
            self.k_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=init_method,
                sequence_parallel_enabled=mp_config.sequence_parallel_enabled,
                dtype=self.config.torch_dtype,
            )
            self.v_proj = ColumnParallelLinear(
                self.hidden_size,
                self.num_key_value_heads * self.head_dim,
                bias=False,
                gather_output=False,
                init_method=init_method,
                sequence_parallel_enabled=mp_config.sequence_parallel_enabled,
                dtype=self.config.torch_dtype,
            )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            init_method=init_method,
            sequence_parallel_enabled=mp_config.sequence_parallel_enabled,
            dtype=self.config.torch_dtype,
        )
        tp_size = get_tensor_model_parallel_size()
        self.num_heads = neuronx_dist_utils.divide(config.num_attention_heads, tp_size)
        self.num_key_value_heads = neuronx_dist_utils.divide(
            config.num_key_value_heads * self.kv_size_multiplier, tp_size
        )
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_value: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        bsz, q_len, _ = hidden_states.size()
        if self.mp_config.sequence_parallel_enabled:
            q_len, bsz, _ = hidden_states.size()
            q_len = q_len * get_tensor_model_parallel_size()

        if (
            self.mp_config.fuse_qkv
            and self.num_heads == self.num_key_value_heads
            and self.kv_size_multiplier == 1
        ):
            qkv_states = self.qkv_proj(hidden_states)
            query_states, key_states, value_states = qkv_states.split(self.split_size, dim=2)
        elif self.qkv_linear:
            query_states, key_states, value_states = self.qkv_proj(hidden_states)
        else:
            query_states = self.q_proj(hidden_states)
            key_states = self.k_proj(hidden_states)
            value_states = self.v_proj(hidden_states)

        if self.mp_config.sequence_parallel_enabled:
            query_states = query_states.view(q_len, bsz, self.num_heads, self.head_dim).permute(1, 2, 0, 3)
            key_states = key_states.view(q_len, bsz, self.num_key_value_heads, self.head_dim).permute(1, 2, 0, 3)
            value_states = value_states.view(q_len, bsz, self.num_key_value_heads, self.head_dim).permute(1, 2, 0, 3)
        else:
            query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
            key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
            value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:
            # sin and cos are specific to RoPE models; cache_position needed for the static cache
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)

        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            if self.config._attn_implementation != "flash_attention_2":
                raise RuntimeError(
                    f"Unsupported attention implementation for Neuron: {self.config._attn_implementation}"
                )
            else:
                # It could be this one day if we support multiple attention implementations, but for now it's just 
                # eager or flash attention.
                # attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
                attention_interface = nki_flash_attn_func

        output = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        if self.config._attn_implementation != "eager":
            attn_output = output[0]
            attn_weights = None
        else:
            attn_output, attn_weights = output

        if self.mp_config.sequence_parallel_enabled:
            attn_output = attn_output.permute(2, 0, 1, 3)
            attn_output = attn_output.reshape(q_len, bsz, self.hidden_size // get_tensor_model_parallel_size())
        else:
            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.reshape(bsz, q_len, self.hidden_size // get_tensor_model_parallel_size())

        attn_output = self.o_proj(attn_output)

        return attn_output, attn_weights

class LlamaDecoderLayer(LlamaDecoderLayerHF):
    def __init__(self, config: LlamaConfig, mp_config: ModelParallelismConfig, layer_idx: int):
        nn.Module.__init__(self)

        self.hidden_size = config.hidden_size

        self.self_attn = LlamaAttention(config=config, mp_config=mp_config, layer_idx=layer_idx)

        self.mlp = LlamaMLP(config, mp_config)
        self.input_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, sequence_parallel_enabled=mp_config.sequence_parallel_enabled
        )
        self.post_attention_layernorm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, sequence_parallel_enabled=mp_config.sequence_parallel_enabled
        )


class LlamaModel(NeuronModelMixin, LlamaModelHF):
    """
    Transformer decoder consisting of *config.num_hidden_layers* layers. Each layer is a [`LlamaDecoderLayer`]

    Args:
        config: LlamaConfig
    """

    def __init__(self, config: LlamaConfig, mp_config: ModelParallelismConfig):
        LlamaPreTrainedModel.__init__(self, config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        init_method = partial(_init_normal, config.initializer_range)
        self.embed_tokens = ParallelEmbedding(
            config.vocab_size, config.hidden_size, self.padding_idx, init_method=init_method,
            sequence_parallel_enabled=mp_config.sequence_parallel_enabled, dtype=config.torch_dtype,
        )
        self.layers = nn.ModuleList([LlamaDecoderLayer(config, mp_config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, sequence_parallel_enabled=mp_config.sequence_parallel_enabled
        )
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        self.gradient_checkpointing = False

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **flash_attn_kwargs: Unpack[FlashAttentionKwargs],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if self.gradient_checkpointing and self.training and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
            )
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # This is not needed since we recompute a causal mask at each attention layer.
        # We keep this in case we want to support custom attention masks in the future.
        # causal_mask = self._update_causal_mask(
        #     attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
        # )
        causal_mask = None

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
                    past_key_values,
                    output_attentions,
                    use_cache,
                    cache_position,
                    position_embeddings,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    past_key_value=past_key_values,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                    cache_position=cache_position,
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
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )
        return output if return_dict else output.to_tuple()

class LlamaForCausalLM(NeuronModelMixin, LlamaForCausalLMHF):
    _tied_weights_keys = ["lm_head.weight"]
    _tp_plan = {"lm_head": "colwise_rep"}

    def __init__(self, config, mp_config: ModelParallelismConfig):
        LlamaForCausalLMHF.__init__(self,config)
        self.model = LlamaModel(config, mp_config)
        self.mp_config = mp_config

        init_method = partial(_init_normal, config.initializer_range)
        self.lm_head = ColumnParallelLinear(
            config.hidden_size,
            config.vocab_size,
            bias=False,
            gather_output=False,
            init_method=init_method,
            sequence_parallel_enabled=mp_config.sequence_parallel_enabled,
            dtype=self.config.torch_dtype,
        )

        self.vocab_size = config.vocab_size // get_tensor_model_parallel_size()

        # Initialize weights and apply final processing
        self.post_init()

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        hidden_states = outputs[0]
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

        if self.mp_config.sequence_parallel_enabled:
            logits = logits.transpose(0, 1).contiguous()

        loss = None
        if labels is not None:
            loss = ForCausalLMLoss(logits=logits, labels=labels, vocab_size=self.vocab_size, **kwargs)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
