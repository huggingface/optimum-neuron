# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""YOLOS model on Neuron devices."""

import logging

import torch
from transformers import AutoModelForObjectDetection
from transformers.modeling_outputs import ModelOutput

from ...modeling_traced import NeuronTracedModel
from ...utils.doc import (
    _PROCESSOR_FOR_IMAGE,
    NEURON_IMAGE_INPUTS_DOCSTRING,
    NEURON_MODEL_START_DOCSTRING,
    NEURON_OBJECT_DETECTION_EXAMPLE,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)


logger = logging.getLogger(__name__)


@add_start_docstrings(
    """
    Neuron Model with object detection heads on top, for tasks such as COCO detection.
    """,
    NEURON_MODEL_START_DOCSTRING,
)
class NeuronYolosForObjectDetection(NeuronTracedModel):
    auto_model_class = AutoModelForObjectDetection

    @property
    def dtype(self) -> "torch.dtype | None":
        """
        Torch dtype of the inputs to avoid error in transformers on casting a BatchFeature to type None.
        """
        return getattr(self.config.neuron, "input_dtype", torch.float32)

    @add_start_docstrings_to_model_forward(
        NEURON_IMAGE_INPUTS_DOCSTRING.format("batch_size, num_channels, height, width")
        + NEURON_OBJECT_DETECTION_EXAMPLE.format(
            processor_class=_PROCESSOR_FOR_IMAGE,
            model_class="NeuronYolosForObjectDetection",
            checkpoint="optimum/yolos-tiny-neuronx-bs1",
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
        pred_boxes = outputs[1]
        last_hidden_state = outputs[2]

        return ModelOutput(logits=logits, pred_boxes=pred_boxes, last_hidden_state=last_hidden_state)
