# coding=utf-8
# Copyright 2025 IBM and the HuggingFace Inc. team. All rights reserved.
#
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


import torch
from neuronx_distributed.parallel_layers.mappings import (
    gather_from_sequence_parallel_region,
    scatter_to_sequence_parallel_region,
)
from torch import nn
from torch_xla.utils.checkpoint import checkpoint
from transformers.loss.loss_utils import ForCausalLMLoss
from transformers.modeling_flash_attention_utils import FlashAttentionKwargs
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.models.granite.configuration_granite import GraniteConfig
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs, can_return_tuple, logging

from ..config import TrainingNeuronConfig
from ..llama.modeling_llama import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaForCausalLM,
    LlamaModel,
    LlamaPreTrainedModel,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)
from ..masking_utils import create_causal_mask


# Wrap the gather and scatter functions to ensure they are properly traced by `torch.fx`.
gather_from_sequence_parallel_region = torch.fx.wrap(gather_from_sequence_parallel_region)
scatter_to_sequence_parallel_region = torch.fx.wrap(scatter_to_sequence_parallel_region)

logger = logging.get_logger(__name__)


class GraniteAttention(LlamaAttention):
    def __init__(self, config: GraniteConfig, trn_config: TrainingNeuronConfig, layer_idx: int):
        super().__init__(config, trn_config, layer_idx)
        self.scaling = config.attention_multiplier


class GraniteDecoderLayer(LlamaDecoderLayer):
    def __init__(self, config: GraniteConfig, trn_config: TrainingNeuronConfig, layer_idx: int):
        super().__init__(config, trn_config, layer_idx)
        self.residual_multiplier = config.residual_multiplier
        self.self_attn = GraniteAttention(config=config, trn_config=trn_config, layer_idx=layer_idx)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,  # necessary, but kept here for BC
        **kwargs: Unpack[TransformersKwargs],
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        # Self Attention
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states * self.residual_multiplier

        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states * self.residual_multiplier  # main diff with Llama

        return hidden_states


class GraniteModel(LlamaModel):
    config = GraniteConfig
    _no_split_modules = ["GraniteDecoderLayer"]

    def __init__(self, config: GraniteConfig, trn_config: TrainingNeuronConfig):
        LlamaPreTrainedModel.__init__(self, config)
        self.embedding_multiplier = config.embedding_multiplier
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.trn_config = trn_config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)

        self.layers = nn.ModuleList(
            [GraniteDecoderLayer(config, trn_config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps, sequence_parallel_enabled=trn_config.sequence_parallel_enabled
        )
        self.rotary_emb = LlamaRotaryEmbedding(config=config)

        self.gradient_checkpointing = self.trn_config.gradient_checkpointing

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if self.trn_config.sequence_parallel_enabled:
            inputs_embeds = inputs_embeds.transpose(0, 1).contiguous()
            inputs_embeds = scatter_to_sequence_parallel_region(inputs_embeds)

        inputs_embeds = inputs_embeds * self.embedding_multiplier  # main diff with Llama

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
            causal_mask = create_causal_mask(
                config=self.config,
                trn_config=self.trn_config,
                input_embeds=inputs_embeds,
                attention_mask=attention_mask,
                cache_position=cache_position,
                past_key_values=None,  # Not used in training
                position_ids=position_ids,
            )

        hidden_states = inputs_embeds

        # create position embeddings to be shared across the decoder layers
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Decoder layers
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            if self.gradient_checkpointing and self.training:
                hidden_states = checkpoint(
                    decoder_layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_ids,
                    position_embeddings,
                    **kwargs,
                )
            else:
                hidden_states = decoder_layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_ids=position_ids,
                    position_embeddings=position_embeddings,
                )

        hidden_states = self.norm(hidden_states)

        output = BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
        )
        return output


class KwargsForCausalLM(FlashAttentionKwargs, TransformersKwargs): ...


class GraniteForCausalLM(LlamaForCausalLM):
    config_class = GraniteConfig

    SUPPORTS_PIPELINE_PARALLELISM = False
    PIPELINE_TRANSFORMER_LAYER_CLS = GraniteDecoderLayer
    PIPELINE_INPUT_NAMES = ["input_ids", "attention_mask", "labels"]
    PIPELINE_LEAF_MODULE_CLASSE_NAMES = ["LlamaRMSNorm", "LlamaRotaryEmbedding"]

    def __init__(self, config, trn_config: TrainingNeuronConfig):
        LlamaPreTrainedModel.__init__(self, config)
        self.trn_config = trn_config
        self.model = GraniteModel(config, trn_config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        **kwargs: Unpack[KwargsForCausalLM],
    ) -> CausalLMOutputWithPast:
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
        logits = logits / self.config.logits_scaling  # main diff with Llama

        if self.trn_config.sequence_parallel_enabled:
            logits = gather_from_sequence_parallel_region(logits)
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
