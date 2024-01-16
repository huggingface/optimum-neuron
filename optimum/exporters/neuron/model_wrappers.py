# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
"""Model wrappers for Neuron export."""

from typing import TYPE_CHECKING, List, Optional

import torch
from transformers.models.t5.modeling_t5 import T5LayerCrossAttention


if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel


class UnetNeuronWrapper(torch.nn.Module):
    def __init__(self, model, input_names: List[str]):
        super().__init__()
        self.model = model
        self.input_names = input_names

    def forward(self, *inputs):
        if len(inputs) != len(self.input_names):
            raise ValueError(
                f"The model needs {len(self.input_names)} inputs: {self.input_names}."
                f" But only {len(input)} inputs are passed."
            )

        ordered_inputs = dict(zip(self.input_names, inputs))

        added_cond_kwargs = {
            "text_embeds": ordered_inputs.pop("text_embeds", None),
            "time_ids": ordered_inputs.pop("time_ids", None),
        }
        sample = ordered_inputs.pop("sample", None)
        timestep = ordered_inputs.pop("timestep").float().expand((sample.shape[0],))

        out_tuple = self.model(
            sample=sample,
            timestep=timestep,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
            **ordered_inputs,
        )

        return out_tuple


# Adapted from https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/torch-neuronx/t5-inference-tutorial.html
class T5EncoderWrapper(torch.nn.Module):
    """Wrapper to trace the encoder and the kv cache initialization in the decoder."""

    def __init__(
        self,
        model: "PreTrainedModel",
        num_beams: int = 1,
        device: str = "xla",
        tp_degree: Optional[int] = None,
    ):
        super().__init__()
        self.model = model
        self.config = model.config
        self.num_beams = num_beams
        self.device = device
        self.tp_degree = tp_degree

    def forward(self, input_ids, attention_mask):
        # Infer shapes
        batch_size = input_ids.shape[0]
        sequence_length = input_ids.shape[1]

        encoder_output = self.model.encoder(
            input_ids=input_ids, attention_mask=attention_mask, output_attentions=False, output_hidden_states=False
        )

        last_hidden_state = encoder_output["last_hidden_state"]
        encoder_hidden_states = torch.concat(
            [tensor.unsqueeze(0).repeat(self.num_beams, 1, 1) for tensor in last_hidden_state]
        )

        decoder_blocks = self.model.decoder.block
        present_key_value_states_sa = []
        present_key_value_states_ca = []

        for block in decoder_blocks:
            # Cross attention has to be initialized with the encoder hidden state
            cross_attention: T5LayerCrossAttention = block.layer[1]
            attention = cross_attention.EncDecAttention

            def shape(states):
                """projection"""
                return states.view(
                    self.num_beams * batch_size, -1, self.config.num_heads, attention.key_value_proj_dim
                ).transpose(1, 2)

            key_states = shape(attention.k(encoder_hidden_states))
            value_states = shape(attention.v(encoder_hidden_states))

            # cross_attn_kv_state
            present_key_value_states_ca.append(key_states)
            present_key_value_states_ca.append(value_states)

            # Self attention kv states are initialized to zeros. This is done to keep the size of the kv cache tensor constant.
            # The kv cache is padded here to keep a fixed shape.
            # [key states]
            present_key_value_states_sa.append(
                torch.zeros(
                    (self.num_beams * batch_size, self.config.num_heads, sequence_length - 1, self.config.d_kv),
                    dtype=torch.float32,
                    device=self.device,
                )
            )
            # [value states]
            present_key_value_states_sa.append(
                torch.zeros(
                    (self.num_beams * batch_size, self.config.num_heads, sequence_length - 1, self.config.d_kv),
                    dtype=torch.float32,
                    device=self.device,
                )
            )

        return present_key_value_states_sa + present_key_value_states_ca


# Adapted from https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/torch-neuronx/t5-inference-tutorial.html
class T5DecoderWrapper(torch.nn.Module):
    """Wrapper to trace the decoder with past keys values with a language head."""

    def __init__(
        self,
        model: "PreTrainedModel",
        batch_size: int,
        sequence_length: int,
        num_beams: int = 1,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        device: str = "xla",
        tp_degree: Optional[int] = None,
    ):
        super().__init__()
        self.model = model
        self.config = model.config
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_beams = num_beams
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions
        self.device = device
        self.tp_degree = tp_degree

        # Initialize KV cache (num_beams, n_heads, seq_length, dim_per_head)
        if device == "cpu":
            self.past_key_values_sa = [
                torch.ones(
                    (num_beams, self.config.num_heads, self.sequence_length - 1, self.config.d_kv), dtype=torch.float32
                )
                for _ in range(self.config.num_decoder_layers * 2)
            ]
            self.past_key_values_ca = [
                torch.ones(
                    (num_beams, self.config.num_heads, self.sequence_length, self.config.d_kv), dtype=torch.float32
                )
                for _ in range(self.config.num_decoder_layers * 2)
            ]
        elif device == "xla":
            self.past_key_values_sa = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(
                        torch.ones(
                            (
                                self.batch_size * self.num_beams,
                                self.config.num_heads,
                                sequence_length - 1,
                                self.config.d_kv,
                            ),
                            dtype=torch.float32,
                        ),
                        requires_grad=False,
                    )
                    for _ in range(self.config.num_decoder_layers * 2)
                ]
            )
            self.past_key_values_ca = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(
                        torch.ones(
                            (
                                self.batch_size * self.num_beams,
                                self.config.num_heads,
                                sequence_length,
                                self.config.d_kv,
                            ),
                            dtype=torch.float32,
                        ),
                        requires_grad=False,
                    )
                    for _ in range(self.config.num_decoder_layers * 2)
                ]
            )

    def update_past(self, past_key_values):
        new_past_sa = []
        new_past_ca = []
        for past_layer in past_key_values:
            new_past_layer = list(past_layer)
            for i in range(len(new_past_layer[:2])):
                new_past_layer[i] = past_layer[i][:, :, 1:]
            new_past_sa += [
                new_past_layer[:2],
            ]
            new_past_ca += [
                new_past_layer[2:],
            ]
        return new_past_sa, new_past_ca

    def reorder_cache(self, past_key_values, beam_idx):
        for i in range(len(past_key_values)):
            gather_index = beam_idx.view([beam_idx.shape[0], 1, 1, 1]).expand_as(past_key_values[i])
            past_key_values[i] = torch.gather(past_key_values[i], dim=0, index=gather_index)
        return past_key_values

    def forward(
        self,
        input_ids,
        decoder_attention_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        beam_idx,
        beam_scores,
        **kwargs,
    ):
        if self.num_beams > 1:
            # We reorder the cache based on the beams selected in each iteration. Required step for beam search.
            past_key_values_sa = self.reorder_cache(self.past_key_values_sa, beam_idx)
            past_key_values_ca = self.reorder_cache(self.past_key_values_ca, beam_idx)
        else:
            # We do not need to reorder for greedy sampling
            past_key_values_sa = self.past_key_values_sa
            past_key_values_ca = self.past_key_values_ca

        # The cache is stored in a flatten form. We order the cache per layer before passing it to the decoder.
        # Each layer has 4 tensors, so we group by 4.
        past_key_values = [
            [*past_key_values_sa[i * 2 : i * 2 + 2], *past_key_values_ca[i * 2 : i * 2 + 2]]
            for i in range(0, int(len(past_key_values_ca) / 2))
        ]

        decoder_output = self.model.decoder(
            input_ids=input_ids,
            attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=True,
            output_attentions=self.output_attentions,
            output_hidden_states=self.output_hidden_states,
        )

        last_hidden_state = decoder_output["last_hidden_state"]
        past_key_values = decoder_output["past_key_values"]
        if self.output_hidden_states:
            decoder_hidden_states = list(
                decoder_output["hidden_states"]
            )  # flatten `hidden_states` which is a tuple of tensors

        if self.output_attentions:
            decoder_attentions = list(
                decoder_output["attentions"]
            )  # flatten `decoder_attentions` which is a tuple of tensors
            cross_attentions = list(
                decoder_output["cross_attentions"]
            )  # flatten `cross_attentions` which is a tuple of tensors

        if self.config.tie_word_embeddings:
            # Rescale output before projecting on vocab
            # See https://github.com/tensorflow/mesh/blob/fa19d69eafc9a482aff0b59ddd96b025c0cb207d/mesh_tensorflow/transformer/transformer.py#L586
            last_hidden_state = last_hidden_state * (self.model.config.d_model**-0.5)

        lm_logits = self.model.lm_head(last_hidden_state)

        past_key_values_sa, past_key_values_ca = self.update_past(past_key_values)

        # We flatten the cache to a single array. This is required for the input output aliasing to work
        past_key_values_sa = [vec for kv_per_layer in past_key_values_sa for vec in kv_per_layer]
        past_key_values_ca = [vec for kv_per_layer in past_key_values_ca for vec in kv_per_layer]

        if self.device == "cpu":
            self.past_key_values_sa = past_key_values_sa
            self.past_key_values_ca = past_key_values_ca

        # We calculate topk inside the wrapper
        next_token_logits = lm_logits[:, -1, :]

        if self.num_beams > 1:
            # This section of beam search is run outside the decoder in the huggingface t5 implementation.
            # To maximize the computation within the neuron device, we move this within the wrapper
            logit_max, _ = torch.max(next_token_logits, dim=-1, keepdim=True)
            logsumexp = torch.log(torch.exp(next_token_logits - logit_max).sum(dim=-1, keepdim=True))
            next_token_scores = next_token_logits - logit_max - logsumexp
            next_token_scores = next_token_scores + beam_scores[:, None].expand_as(next_token_scores)

            # reshape for beam search
            vocab_size = next_token_scores.shape[-1]
            next_token_scores = next_token_scores.view(self.batch_size, self.num_beams * vocab_size)
            next_token_scores = next_token_scores * 1

            # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
            next_token_scores, next_tokens = torch.topk(
                next_token_scores, 2 * self.num_beams, dim=1, largest=True, sorted=True
            )

            next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
            next_tokens = next_tokens % vocab_size

            neuron_outputs = [next_token_scores, next_tokens, next_indices] + past_key_values_sa + past_key_values_ca

        else:
            # Greedy
            next_tokens = torch.argmax(next_token_logits, dim=-1)

            neuron_outputs = [next_tokens] + past_key_values_sa + past_key_values_ca

        if self.output_hidden_states:
            neuron_outputs += decoder_hidden_states

        if self.output_attentions:
            neuron_outputs += decoder_attentions
            neuron_outputs += cross_attentions

        return neuron_outputs


class SentenceTransformersTransformerNeuronWrapper(torch.nn.Module):
    def __init__(self, model, input_names: List[str]):
        super().__init__()
        self.model = model
        self.input_names = input_names

    def forward(self, input_ids, attention_mask):
        out_tuple = self.model({"input_ids": input_ids, "attention_mask": attention_mask})

        return out_tuple["token_embeddings"], out_tuple["sentence_embedding"]


class SentenceTransformersCLIPNeuronWrapper(torch.nn.Module):
    def __init__(self, model, input_names: List[str]):
        super().__init__()
        self.model = model
        self.input_names = input_names

    def forward(self, input_ids, pixel_values, attention_mask):
        vision_outputs = self.model[0].model.vision_model(pixel_values=pixel_values)
        image_embeds = self.model[0].model.visual_projection(vision_outputs[1])

        text_outputs = self.model[0].model.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_embeds = self.model[0].model.text_projection(text_outputs[1])

        if len(self.model) > 1:
            image_embeds = self.model[1:](image_embeds)
            text_embeds = self.model[1:](text_embeds)

        return (text_embeds, image_embeds)
