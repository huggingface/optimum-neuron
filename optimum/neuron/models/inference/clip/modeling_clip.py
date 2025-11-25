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
"""CLIP model on Neuron devices."""

import logging

import torch
import torch.nn as nn
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear
from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_size
from transformers import (
    AutoModel,
    AutoModelForImageClassification,
)
from transformers.activations import ACT2FN
from transformers.modeling_attn_mask_utils import (
    _create_4d_causal_attention_mask,
    _prepare_4d_attention_mask,
)
from transformers.modeling_outputs import BaseModelOutput, BaseModelOutputWithPooling, ImageClassifierOutput
from transformers.models.clip.modeling_clip import CLIPOutput

from optimum.neuron.modeling_traced import NeuronTracedModel
from optimum.neuron.utils.doc import (
    _GENERIC_PROCESSOR,
    _PROCESSOR_FOR_IMAGE,
    NEURON_IMAGE_CLASSIFICATION_EXAMPLE,
    NEURON_IMAGE_INPUTS_DOCSTRING,
    NEURON_MODEL_START_DOCSTRING,
    NEURON_MULTIMODAL_FEATURE_EXTRACTION_EXAMPLE,
    NEURON_TEXT_IMAGE_INPUTS_DOCSTRING,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)


logger = logging.getLogger(__name__)


@add_start_docstrings(
    """
    Bare CLIP Model without any specific head on top, used for the task "feature-extraction".
    """,
    NEURON_MODEL_START_DOCSTRING,
)
class NeuronCLIPModel(NeuronTracedModel):
    auto_model_class = AutoModel

    @add_start_docstrings_to_model_forward(
        NEURON_TEXT_IMAGE_INPUTS_DOCSTRING
        + NEURON_MULTIMODAL_FEATURE_EXTRACTION_EXAMPLE.format(
            processor_class=_GENERIC_PROCESSOR,
            model_class="NeuronCLIPModel",
            checkpoint="optimum/clip-vit-base-patch32-neuronx",
        )
    )
    def forward(
        self,
        input_ids: torch.Tensor,
        pixel_values: torch.Tensor,
        attention_mask: torch.Tensor,
        **kwargs,
    ):
        neuron_inputs = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
            "attention_mask": attention_mask,
        }

        with self.neuron_padding_manager(neuron_inputs) as inputs:
            outputs = self.model(*inputs)
            text_last_hidden_state = self.remove_padding([outputs[4][0]], dims=[1], indices=[input_ids.shape[1]])[
                0
            ]  # Remove padding on batch_size(0)

            text_outputs = BaseModelOutputWithPooling(
                last_hidden_state=text_last_hidden_state,
                pooler_output=outputs[4][1],
            )
            vision_outputs = BaseModelOutputWithPooling(last_hidden_state=outputs[5][0], pooler_output=outputs[5][1])

        return CLIPOutput(
            logits_per_image=outputs[0],
            logits_per_text=outputs[1],
            text_embeds=outputs[2],
            image_embeds=outputs[3],
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )


@add_start_docstrings(
    """
    CLIP vision encoder with an image classification head on top (a linear layer on top of the pooled final hidden states of the patch tokens) e.g. for ImageNet.
    """,
    NEURON_MODEL_START_DOCSTRING,
)
class NeuronCLIPForImageClassification(NeuronTracedModel):
    auto_model_class = AutoModelForImageClassification

    @property
    def dtype(self) -> "torch.dtype | None":
        """
        Torch dtype of the inputs to avoid error in transformers on casting a BatchFeature to type None.
        """
        return getattr(self.config.neuron, "input_dtype", torch.float32)

    @add_start_docstrings_to_model_forward(
        NEURON_IMAGE_INPUTS_DOCSTRING.format("batch_size, num_channels, height, width")
        + NEURON_IMAGE_CLASSIFICATION_EXAMPLE.format(
            processor_class=_PROCESSOR_FOR_IMAGE,
            model_class="NeuronCLIPForImageClassification",
            checkpoint="optimum/clip-vit-base-patch32-image-classification-neuronx",
        )
    )
    def forward(
        self,
        pixel_values: torch.Tensor,
        **kwargs,
    ):
        neuron_inputs = {"pixel_values": pixel_values}

        # run inference
        with self.neuron_padding_manager(neuron_inputs) as inputs:
            outputs = self.model(*inputs)  # shape: [batch_size, num_channels, image_size, image_size]
            outputs = self.remove_padding(
                outputs, dims=[0], indices=[pixel_values.shape[0]]
            )  # Remove padding on batch_size(0)

        logits = outputs[0]

        return ImageClassifierOutput(logits=logits)


#######################################################
##############            NxD            ##############
#######################################################

class NeuronCLIPAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config._config.hidden_size
        self.num_heads = config._config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        if self.head_dim * self.num_heads != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim} and `num_heads`:"
                f" {self.num_heads})."
            )
        self.scale = self.head_dim**-0.5
        self.dropout = config._config.attention_dropout

        self.q_proj = ColumnParallelLinear(config._config.hidden_size, config._config.hidden_size, gather_output=False)
        self.k_proj = ColumnParallelLinear(config._config.hidden_size, config._config.hidden_size, gather_output=False)
        self.v_proj = ColumnParallelLinear(config._config.hidden_size, config._config.hidden_size, gather_output=False)
        self.out_proj = RowParallelLinear(config._config.hidden_size, config._config.hidden_size, input_is_parallel=True)

        tp_size = get_tensor_model_parallel_size()
        self.num_heads = self.num_heads // tp_size

        assert self.num_heads * self.head_dim == (config._config.hidden_size // tp_size)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        causal_attention_mask: torch.Tensor | None = None,
        output_attentions: bool | None = True,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Input shape: Batch x Time x Channel"""

        bsz, tgt_len, embed_dim = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scale
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.view(*proj_shape)
        value_states = value_states.view(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        # apply the causal_attention_mask first
        if causal_attention_mask is not None:
            if causal_attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is"
                    f" {causal_attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + causal_attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)

        if output_attentions:
            # this operation is a bit akward, but it's required to
            # make sure that attn_weights keeps its gradient.
            # In order to do so, attn_weights have to reshaped
            # twice and have to be reused in the following
            attn_weights_reshaped = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights_reshaped.view(bsz * self.num_heads, tgt_len, src_len)
        else:
            attn_weights_reshaped = None

        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)
        attn_output = attn_output.reshape(bsz, tgt_len, -1)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights_reshaped


class NeuronCLIPMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config._config.hidden_act]
        self.fc1 = ColumnParallelLinear(config._config.hidden_size, config._config.intermediate_size, gather_output=False)
        self.fc2 = RowParallelLinear(config._config.intermediate_size, config._config.hidden_size, input_is_parallel=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class NeuronCLIPEncoderLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embed_dim = config._config.hidden_size
        self.self_attn = NeuronCLIPAttention(config)

        self.layer_norm1 = nn.LayerNorm(self.embed_dim, eps=config._config.layer_norm_eps)
        self.mlp = NeuronCLIPMLP(config)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim, eps=config._config.layer_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor,
        output_attentions: bool | None = True,
    ) -> tuple[torch.FloatTensor]:
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`): attention mask of size
                `(batch, 1, tgt_len, src_len)` where padding elements are indicated by very large negative values.
                `(config.encoder_attention_heads,)`.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        hidden_states = self.layer_norm1(hidden_states)
        hidden_states, attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs


class NeuronCLIPEncoder(nn.Module):
    """
    Parallel Transformer encoder consisting of `config.num_hidden_layers` self attention layers.
    Each layer is a [`NeuronCLIPEncoderLayer`].

    Args:
        config: CLIPConfig
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([NeuronCLIPEncoderLayer(config) for _ in range(config._config.num_hidden_layers)])
        self.gradient_checkpointing = False

    def forward(
        self,
        inputs_embeds,
        attention_mask: torch.Tensor | None = None,
        causal_attention_mask: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> tuple | BaseModelOutput:
        encoder_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        hidden_states = inputs_embeds
        for idx, encoder_layer in enumerate(self.layers):
            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)
            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(encoder_layer),
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                )
            else:
                layer_outputs = encoder_layer(
                    hidden_states,
                    attention_mask,
                    causal_attention_mask,
                    output_attentions=output_attentions,
                )

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_attentions = all_attentions + (layer_outputs[1],)

        if output_hidden_states:
            encoder_states = encoder_states + (hidden_states,)

        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
        )


class NeuronCLIPTextEmbeddings(nn.Module):
    def __init__(self, config):
        super().__init__()
        embed_dim = config._config.hidden_size

        self.token_embedding = nn.Embedding(config._config.vocab_size, embed_dim)
        self.position_embedding = nn.Embedding(config._config.max_position_embeddings, embed_dim)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer(
            "position_ids",
            torch.arange(config._config.max_position_embeddings).expand((1, -1)),
            persistent=False,
        )

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
    ) -> torch.Tensor:
        seq_length = input_ids.shape[-1] if input_ids is not None else inputs_embeds.shape[-2]

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)

        position_embeddings = self.position_embedding(position_ids)
        embeddings = inputs_embeds + position_embeddings

        return embeddings


class NeuronCLIPTextTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        embed_dim = config._config.hidden_size
        self.embeddings = NeuronCLIPTextEmbeddings(config)
        self.encoder = NeuronCLIPEncoder(config)
        self.final_layer_norm = nn.LayerNorm(embed_dim, eps=config._config.layer_norm_eps)

        # For `pooled_output` computation
        self.eos_token_id = config._config.eos_token_id

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> tuple | BaseModelOutputWithPooling:
        if input_ids is None:
            raise ValueError("You have to specify input_ids")

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_shape[-1])

        hidden_states = self.embeddings(input_ids=input_ids, position_ids=position_ids)

        # CLIP's text model uses causal mask, prepare it here.
        # https://github.com/openai/CLIP/blob/cfcffb90e69f37bf2ff1e988237a0fbe41f33c04/clip/model.py#L324
        causal_attention_mask = _create_4d_causal_attention_mask(
            input_shape, hidden_states.dtype, device=hidden_states.device
        )

        # expand attention_mask
        # always requires in eager attention computation
        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            attention_mask = _prepare_4d_attention_mask(attention_mask, hidden_states.dtype)

        encoder_outputs = self.encoder(
            inputs_embeds=hidden_states,
            attention_mask=attention_mask,
            causal_attention_mask=causal_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        last_hidden_state = encoder_outputs[0]
        last_hidden_state = self.final_layer_norm(last_hidden_state)

        if self.eos_token_id == 2:
            # The `eos_token_id` was incorrect before PR #24773: Let's keep what have been done here.
            # A CLIP model with such `eos_token_id` in the config can't work correctly with extra new tokens added
            # ------------------------------------------------------------
            # text_embeds.shape = [batch_size, sequence_length, transformer.width]
            # take features from the eot embedding (eot_token is the highest number in each sequence)
            # casting to torch.int for onnx compatibility: argmax doesn't support int64 inputs with opset 14
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                input_ids.to(dtype=torch.int, device=last_hidden_state.device).argmax(dim=-1),
            ]
        else:
            # The config gets updated `eos_token_id` from PR #24773 (so the use of exta new tokens is possible)
            pooled_output = last_hidden_state[
                torch.arange(last_hidden_state.shape[0], device=last_hidden_state.device),
                # We need to get the first position of `eos_token_id` value (`pad_token_ids` might equal to `eos_token_id`)
                # Note: we assume each sequence (along batch dim.) contains an  `eos_token_id` (e.g. prepared by the tokenizer)
                (input_ids.to(dtype=torch.int, device=last_hidden_state.device) == self.eos_token_id)
                .int()
                .argmax(dim=-1),
            ]

        if not return_dict:
            return (last_hidden_state, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=last_hidden_state,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class NeuronCLIPTextModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.neuron_text_encoder = NeuronCLIPTextTransformer(config)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict:bool | None = None,
    ) -> tuple | BaseModelOutputWithPooling:
        return self.neuron_text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
