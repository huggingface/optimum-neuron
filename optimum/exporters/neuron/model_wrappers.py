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

from typing import TYPE_CHECKING

import torch
from transformers.cache_utils import EncoderDecoderCache
from transformers.models.t5.modeling_t5 import T5LayerCrossAttention

from ...neuron.utils import is_neuronx_available
from .utils import f32Wrapper


if is_neuronx_available():
    import torch_xla.core.xla_model as xm

import neuronx_distributed
from neuronx_distributed.parallel_layers.layers import ColumnParallelLinear, RowParallelLinear


if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel
    from transformers.models.t5.modeling_t5 import T5Attention, T5LayerFF


class UnetNeuronWrapper(torch.nn.Module):
    def __init__(self, model, input_names: list[str], device: str | None = None):
        super().__init__()
        self.model = model
        self.input_names = input_names
        self.device = device

    def forward(self, *inputs):
        if len(inputs) != len(self.input_names):
            raise ValueError(
                f"The model needs {len(self.input_names)} inputs: {self.input_names}."
                f" But only {len(inputs)} inputs are passed."
            )

        ordered_inputs = dict(zip(self.input_names, inputs))

        added_cond_kwargs = {
            "text_embeds": ordered_inputs.pop("text_embeds", None),
            "time_ids": ordered_inputs.pop("time_ids", None),
            "image_embeds": ordered_inputs.pop("image_embeds", None)
            or ordered_inputs.pop("image_enc_hidden_states", None),
        }
        sample = ordered_inputs.pop("sample", None)
        timestep = ordered_inputs.pop("timestep").float().expand((sample.shape[0],))
        encoder_hidden_states = ordered_inputs.pop("encoder_hidden_states", None)

        # Re-build down_block_additional_residual
        down_block_additional_residuals = ()
        down_block_additional_residuals_names = [
            name for name in ordered_inputs.keys() if "down_block_additional_residuals" in name
        ]
        for name in down_block_additional_residuals_names:
            value = ordered_inputs.pop(name)
            down_block_additional_residuals += (value,)

        mid_block_additional_residual = ordered_inputs.pop("mid_block_additional_residual", None)

        out_tuple = self.model(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            down_block_additional_residuals=(
                down_block_additional_residuals if down_block_additional_residuals else None
            ),
            mid_block_additional_residual=mid_block_additional_residual,
            added_cond_kwargs=added_cond_kwargs,
            return_dict=False,
        )

        return out_tuple


class PixartTransformerNeuronWrapper(torch.nn.Module):
    def __init__(self, model, input_names: list[str], device: str = None):
        super().__init__()
        self.model = model
        self.dtype = model.dtype
        self.input_names = input_names
        self.device = device

    def forward(self, *inputs):
        if len(inputs) != len(self.input_names):
            raise ValueError(
                f"The model needs {len(self.input_names)} inputs: {self.input_names}."
                f" But only {len(input)} inputs are passed."
            )

        ordered_inputs = dict(zip(self.input_names, inputs))

        sample = ordered_inputs.pop("sample", None)
        encoder_hidden_states = ordered_inputs.pop("encoder_hidden_states", None)
        timestep = ordered_inputs.pop("timestep", None)
        encoder_attention_mask = ordered_inputs.pop("encoder_attention_mask", None)

        # Additional conditions
        out_tuple = self.model(
            hidden_states=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            added_cond_kwargs={"resolution": None, "aspect_ratio": None},
            return_dict=False,
        )

        return out_tuple


class FluxTransformerNeuronWrapper(torch.nn.Module):
    def __init__(self, model, input_names: list[str], device: str = None):
        super().__init__()
        self.model = model
        self.dtype = model.dtype
        self.input_names = input_names
        self.device = device

    def forward(self, *inputs):
        if len(inputs) != len(self.input_names):
            raise ValueError(
                f"The model needs {len(self.input_names)} inputs: {self.input_names}."
                f" But only {len(input)} inputs are passed."
            )

        ordered_inputs = dict(zip(self.input_names, inputs))

        hidden_states = ordered_inputs.pop("hidden_states", None)
        encoder_hidden_states = ordered_inputs.pop("encoder_hidden_states", None)
        pooled_projections = ordered_inputs.pop("pooled_projections", None)
        timestep = ordered_inputs.pop("timestep", None)
        guidance = ordered_inputs.pop("guidance", None)
        image_rotary_emb = ordered_inputs.pop("image_rotary_emb", None)

        out_tuple = self.model(
            hidden_states=hidden_states,
            timestep=timestep,
            guidance=guidance,
            pooled_projections=pooled_projections,
            encoder_hidden_states=encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            joint_attention_kwargs=None,
            return_dict=False,
        )

        return out_tuple


class ControlNetNeuronWrapper(torch.nn.Module):
    def __init__(self, model, input_names: list[str], device: str = None):
        super().__init__()
        self.model = model
        self.input_names = input_names
        self.device = device

    def forward(self, *inputs):
        if len(inputs) != len(self.input_names):
            raise ValueError(
                f"The model needs {len(self.input_names)} inputs: {self.input_names}."
                f" But only {len(input)} inputs are passed."
            )

        ordered_inputs = dict(zip(self.input_names, inputs))

        sample = ordered_inputs.pop("sample", None)
        timestep = ordered_inputs.pop("timestep", None)
        encoder_hidden_states = ordered_inputs.pop("encoder_hidden_states", None)
        controlnet_cond = ordered_inputs.pop("controlnet_cond", None)
        conditioning_scale = ordered_inputs.pop("conditioning_scale", None)

        # Additional conditions for the Stable Diffusion XL UNet.
        added_cond_kwargs = {
            "text_embeds": ordered_inputs.pop("text_embeds", None),
            "time_ids": ordered_inputs.pop("time_ids", None),
        }

        out_tuple = self.model(
            sample=sample,
            timestep=timestep,
            encoder_hidden_states=encoder_hidden_states,
            controlnet_cond=controlnet_cond,
            conditioning_scale=conditioning_scale,
            added_cond_kwargs=added_cond_kwargs,
            guess_mode=False,  # TODO: support guess mode of ControlNet
            return_dict=False,
            **ordered_inputs,
        )

        return out_tuple


# Adapted from https://github.com/aws-neuron/aws-neuron-samples/blob/master/torch-neuronx/inference/hf_pretrained_pixart_alpha_inference_on_inf2.ipynb
# For text encoding
class T5EncoderWrapper(torch.nn.Module):
    def __init__(
        self,
        model: "PreTrainedModel",
        sequence_length: int,
        batch_size: int | None = None,
        device: str = "xla",
        tensor_parallel_size: int = 1,
    ):
        super().__init__()
        self.model = model
        self.config = model.config
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.device = device
        self.tensor_parallel_size = tensor_parallel_size

        for block in self.model.encoder.block:
            block.layer[1].DenseReluDense.act = torch.nn.GELU(approximate="tanh")
        precomputed_bias = (
            self.model.encoder.block[0].layer[0].SelfAttention.compute_bias(self.sequence_length, self.sequence_length)
        )
        if self.tensor_parallel_size > 1:
            self.model = self.parallelize(self.model)
            precomputed_bias_tp = self.shard_weights(precomputed_bias, 1)
            self.model.encoder.block[0].layer[0].SelfAttention.compute_bias = (
                lambda *args, **kwargs: precomputed_bias_tp
            )
        else:
            self.model.encoder.block[0].layer[0].SelfAttention.compute_bias = lambda *args, **kwargs: precomputed_bias

    def parallelize(self, model):
        for _, block in enumerate(model.encoder.block):
            selfAttention = block.layer[0].SelfAttention
            ff = block.layer[1]
            layer_norm_0 = block.layer[0].layer_norm.to(torch.float32)
            layer_norm_1 = block.layer[1].layer_norm.to(torch.float32)
            block.layer[1] = self.shard_ff(ff)
            block.layer[0].SelfAttention = self.shard_self_attention(selfAttention, self.tensor_parallel_size)
            block.layer[0].layer_norm = f32Wrapper(layer_norm_0)
            block.layer[1].layer_norm = f32Wrapper(layer_norm_1)
        final_layer_norm = model.encoder.final_layer_norm.to(torch.float32)
        model.encoder.final_layer_norm = f32Wrapper(final_layer_norm)

        return model

    @staticmethod
    def shard_weights(data, dim):
        tp_rank = neuronx_distributed.parallel_layers.parallel_state.get_tensor_model_parallel_rank()
        s = data.shape[dim] // neuronx_distributed.parallel_layers.parallel_state.get_tensor_model_parallel_size()
        if dim == 0:
            return data[s * tp_rank : s * (tp_rank + 1)].clone()
        elif dim == 1:
            return data[:, s * tp_rank : s * (tp_rank + 1)].clone()

    @staticmethod
    def shard_ff(ff: "T5LayerFF"):
        orig_wi_0 = ff.DenseReluDense.wi_0  # only applicable for T5 with gated silu`
        ff.DenseReluDense.wi_0 = ColumnParallelLinear(
            orig_wi_0.in_features, orig_wi_0.out_features, bias=False, gather_output=False
        )
        ff.DenseReluDense.wi_0.weight.data = T5EncoderWrapper.shard_weights(orig_wi_0.weight.data, 0)
        orig_wi_1 = ff.DenseReluDense.wi_1
        ff.DenseReluDense.wi_1 = ColumnParallelLinear(
            orig_wi_1.in_features, orig_wi_1.out_features, bias=False, gather_output=False
        )
        ff.DenseReluDense.wi_1.weight.data = T5EncoderWrapper.shard_weights(orig_wi_1.weight.data, 0)
        orig_wo = ff.DenseReluDense.wo
        ff.DenseReluDense.wo = RowParallelLinear(
            orig_wo.in_features, orig_wo.out_features, bias=False, input_is_parallel=True
        )
        ff.DenseReluDense.wo.weight.data = T5EncoderWrapper.shard_weights(orig_wo.weight.data, 1)
        ff.DenseReluDense.act = torch.nn.GELU(approximate="tanh")
        return ff

    @staticmethod
    def shard_self_attention(selfAttention: "T5Attention", tensor_parallel_size: int):
        orig_inner_dim = selfAttention.q.out_features
        dim_head = orig_inner_dim // selfAttention.n_heads
        selfAttention.n_heads = selfAttention.n_heads // tensor_parallel_size
        selfAttention.inner_dim = dim_head * selfAttention.n_heads
        orig_q = selfAttention.q
        selfAttention.q = ColumnParallelLinear(
            selfAttention.q.in_features, selfAttention.q.out_features, bias=False, gather_output=False
        )
        selfAttention.q.weight.data = T5EncoderWrapper.shard_weights(orig_q.weight.data, 0)
        del orig_q
        orig_k = selfAttention.k
        selfAttention.k = ColumnParallelLinear(
            selfAttention.k.in_features,
            selfAttention.k.out_features,
            bias=(selfAttention.k.bias is not None),
            gather_output=False,
        )
        selfAttention.k.weight.data = T5EncoderWrapper.shard_weights(orig_k.weight.data, 0)
        del orig_k
        orig_v = selfAttention.v
        selfAttention.v = ColumnParallelLinear(
            selfAttention.v.in_features,
            selfAttention.v.out_features,
            bias=(selfAttention.v.bias is not None),
            gather_output=False,
        )
        selfAttention.v.weight.data = T5EncoderWrapper.shard_weights(orig_v.weight.data, 0)
        del orig_v
        orig_out = selfAttention.o
        selfAttention.o = RowParallelLinear(
            selfAttention.o.in_features,
            selfAttention.o.out_features,
            bias=(selfAttention.o.bias is not None),
            input_is_parallel=True,
        )
        selfAttention.o.weight.data = T5EncoderWrapper.shard_weights(orig_out.weight.data, 1)
        del orig_out
        return selfAttention

    def forward(self, input_ids):
        return self.model(input_ids, output_hidden_states=False)


# Adapted from https://awsdocs-neuron.readthedocs-hosted.com/en/latest/src/examples/pytorch/torch-neuronx/t5-inference-tutorial.html
# For text encoding + KV cache initialization
class T5EncoderForSeq2SeqLMWrapper(torch.nn.Module):
    """Wrapper to trace the encoder and the kv cache initialization in the decoder."""

    def __init__(
        self,
        model: "PreTrainedModel",
        sequence_length: int | None = None,
        batch_size: int | None = None,
        num_beams: int = 1,
        device: str = "xla",
        tensor_parallel_size: int = 1,
    ):
        super().__init__()
        self.model = model
        self.config = model.config
        self.num_beams = num_beams
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.device = device
        self.tensor_parallel_size = tensor_parallel_size
        self.num_attention_heads_per_partition = self.config.num_heads  # when tensor_parallel_size=1

        if self.tensor_parallel_size > 1:
            self.num_attention_heads_per_partition = (
                self.num_attention_heads_per_partition
                // neuronx_distributed.parallel_layers.parallel_state.get_tensor_model_parallel_size()
            )
            self.past_key_values_sa = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(
                        torch.ones(
                            (
                                self.num_beams * batch_size,
                                self.num_attention_heads_per_partition,
                                self.sequence_length - 1,
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
                                self.num_beams * batch_size,
                                self.num_attention_heads_per_partition,
                                self.sequence_length,
                                self.config.d_kv,
                            ),
                            dtype=torch.float32,
                        ),
                        requires_grad=False,
                    )
                    for _ in range(self.config.num_decoder_layers * 2)
                ]
            )

    def forward(self, input_ids, attention_mask):
        # Infer shapes of dummy inputs used for tracing
        batch_size = input_ids.shape[0]
        sequence_length = input_ids.shape[1]
        if self.sequence_length is not None:
            assert self.sequence_length, (
                f"Different sequence length for the parallel partition({self.sequence_length}) and for dummy inputs({sequence_length}). Make sure that they have the same value."
            )
        if self.batch_size is not None:
            assert self.batch_size, (
                f"Different batch size for the parallel partition({self.batch_size}) and for dummy inputs({batch_size}). Make sure that they have the same value."
            )

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

        for i, block in enumerate(decoder_blocks):
            # Cross attention has to be initialized with the encoder hidden state
            cross_attention: T5LayerCrossAttention = block.layer[1]
            attention = cross_attention.EncDecAttention

            def shape(states):
                """projection"""
                return states.view(
                    self.num_beams * batch_size,
                    -1,
                    self.num_attention_heads_per_partition,
                    attention.key_value_proj_dim,
                ).transpose(1, 2)

            key_states = shape(attention.k(encoder_hidden_states))
            value_states = shape(attention.v(encoder_hidden_states))

            if not self.tensor_parallel_size > 1:
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
            else:
                present_key_value_states_ca.append((self.past_key_values_ca[i * 2] * 0) + key_states)
                present_key_value_states_ca.append((self.past_key_values_ca[i * 2 + 1] * 0) + value_states)
                present_key_value_states_sa.append(
                    self.past_key_values_sa[i * 2]
                    * torch.zeros(
                        (
                            self.num_beams * self.batch_size,
                            self.num_attention_heads_per_partition,
                            self.sequence_length - 1,
                            self.config.d_kv,
                        ),
                        dtype=torch.float32,
                        device=self.device,
                    )
                )
                present_key_value_states_sa.append(
                    self.past_key_values_sa[i * 2 + 1]
                    * torch.zeros(
                        (
                            self.num_beams * self.batch_size,
                            self.num_attention_heads_per_partition,
                            self.sequence_length - 1,
                            self.config.d_kv,
                        ),
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
        tensor_parallel_size: int = 1,
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
        self.tensor_parallel_size = tensor_parallel_size

        self.num_attention_heads_per_partition = self.config.num_heads
        if tensor_parallel_size > 1:
            self.num_attention_heads_per_partition = (
                self.num_attention_heads_per_partition
                // neuronx_distributed.parallel_layers.parallel_state.get_tensor_model_parallel_size()
            )

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
                                self.num_attention_heads_per_partition,
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
                                self.num_attention_heads_per_partition,
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
            past_key_values=EncoderDecoderCache.from_legacy_cache(past_key_values),
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
    def __init__(self, model, input_names: list[str], device: str = None):
        super().__init__()
        self.model = model
        self.input_names = input_names
        self.device = device

    def forward(self, input_ids, attention_mask):
        out_tuple = self.model({"input_ids": input_ids, "attention_mask": attention_mask})

        return out_tuple["token_embeddings"], out_tuple["sentence_embedding"]


class CLIPVisionWithProjectionNeuronWrapper(torch.nn.Module):
    def __init__(
        self,
        model,
        input_names: list[str],
        output_hidden_states: bool = True,
        device: str = None,
    ):
        super().__init__()
        self.model = model
        self.input_names = input_names
        self.output_hidden_states = output_hidden_states
        self.device = device

    def forward(self, pixel_values):
        vision_outputs = self.model.vision_model(
            pixel_values=pixel_values, output_hidden_states=self.output_hidden_states
        )
        pooled_output = vision_outputs[1]
        image_embeds = self.model.visual_projection(pooled_output)

        outputs = (image_embeds, vision_outputs.last_hidden_state)

        if self.output_hidden_states:
            outputs += (vision_outputs.hidden_states,)
        return outputs


class SentenceTransformersCLIPNeuronWrapper(torch.nn.Module):
    def __init__(self, model, input_names: list[str], device: str = None):
        super().__init__()
        self.model = model
        self.input_names = input_names
        self.device = device

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


class WhisperEncoderWrapper(torch.nn.Module):
    """Wrapper to trace the forward of Whisper encoder."""

    def __init__(
        self,
        model: "PreTrainedModel",
        batch_size: int,
        device: str = None,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.config = model.config
        self.batch_size = batch_size
        self.device = device

    def forward(
        self,
        input_features,
        decoder_input_ids,
        **kwargs,
    ):
        # encoder
        encoder_outputs = self.model.model.encoder(
            input_features=input_features,
            return_dict=True,
        )
        # 1st decoder + proj_out
        decoder_outputs = self.model.model.decoder(
            input_ids=decoder_input_ids,
            encoder_hidden_states=encoder_outputs[0],
            use_cache=False,
            return_dict=True,
        )
        lm_logits = self.model.proj_out(decoder_outputs[0])

        return (lm_logits, encoder_outputs.last_hidden_state)


class WhisperDecoderWrapper(torch.nn.Module):
    """Wrapper to trace the forward of Whisper decoder."""

    def __init__(
        self,
        model: "PreTrainedModel",
        batch_size: int,
        sequence_length: int,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        device: str = None,
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.config = model.config
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.output_hidden_states = output_hidden_states
        self.output_attentions = output_attentions
        self.device = device if device else xm.xla_device()

    def forward(
        self,
        input_ids,
        encoder_hidden_states,
        **kwargs,
    ):
        cache_position = torch.arange(input_ids.shape[1]).to(self.device)
        outputs = self.model.model.decoder(
            input_ids=input_ids,
            encoder_hidden_states=encoder_hidden_states,
            use_cache=False,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=True,
            cache_position=cache_position,
        )
        lm_logits = self.model.proj_out(outputs[0])
        return lm_logits


class NoCacheModelWrapper(torch.nn.Module):
    def __init__(self, model: "PreTrainedModel", input_names: list[str]):
        super().__init__()
        self.model = model
        self.input_names = input_names

    def forward(self, *input):
        ordered_inputs = dict(zip(self.input_names, input))
        outputs = self.model(use_cache=False, **ordered_inputs)

        return outputs
