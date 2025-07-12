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
from transformers import (
    AutoModel,
    AutoModelForImageClassification,
)
from transformers.modeling_outputs import BaseModelOutputWithPooling, ImageClassifierOutput
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
