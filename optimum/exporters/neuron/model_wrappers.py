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


if TYPE_CHECKING:
    from transformers.modeling_utils import PreTrainedModel


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


class NoCacheModelWrapper(torch.nn.Module):
    def __init__(self, model: "PreTrainedModel", input_names: list[str]):
        super().__init__()
        self.model = model
        self.input_names = input_names

    def forward(self, *input):
        ordered_inputs = dict(zip(self.input_names, input))
        outputs = self.model(use_cache=False, **ordered_inputs)

        return outputs
