# coding=utf-8
# Copyright 2026 The HuggingFace Team. All rights reserved.
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

import os
from typing import Any

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

from optimum.neuron import NeuronModelForImageTextToText
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx


@is_inferentia_test
@requires_neuronx
def test_vlm_generation_base(any_vlm_generate_model: dict[str, Any]):
    """Test that a VLM can generate tokens with and without pixel_values."""
    neuron_model = NeuronModelForImageTextToText.from_pretrained(any_vlm_generate_model["neuron_model_path"])
    batch_size = neuron_model.neuron_config.batch_size
    input_length = 10

    # Text-only generation (no images)
    input_ids = torch.ones((batch_size, input_length), dtype=torch.int64)
    outputs = neuron_model.generate(input_ids, max_new_tokens=5)
    assert outputs.shape[0] == batch_size, "Output batch size should match input batch size"
    assert outputs.shape[1] > input_length, "Output should be longer than input"

    # Generation with dummy pixel_values
    image_size = neuron_model.neuron_config.image_size
    max_num_images = neuron_model.neuron_config.max_num_images
    pixel_values = torch.zeros((batch_size, max_num_images, 3, image_size, image_size))
    outputs_with_images = neuron_model.generate(input_ids, pixel_values=pixel_values, max_new_tokens=5)
    assert outputs_with_images.shape[0] == batch_size


@is_inferentia_test
@requires_neuronx
def test_vlm_generation_greedy_expectations(any_vlm_generate_model: dict[str, Any]):
    """Test that VLM greedy generation matches the HF reference model (text-only path)."""
    model_id = any_vlm_generate_model["model_id"]
    neuron_model_path = any_vlm_generate_model["neuron_model_path"]

    neuron_model = NeuronModelForImageTextToText.from_pretrained(neuron_model_path)
    processor = AutoProcessor.from_pretrained(neuron_model_path)
    prompt = "What is Deep Learning?"
    inputs = processor(text=prompt, return_tensors="pt")
    max_new_tokens = 10

    # CPU reference
    cpu_model = AutoModelForImageTextToText.from_pretrained(model_id)
    cpu_outputs = cpu_model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)

    neuron_outputs = neuron_model.generate(inputs["input_ids"], max_new_tokens=max_new_tokens, do_sample=False)
    assert torch.equal(neuron_outputs, cpu_outputs), "Neuron and CPU outputs differ at the token level"


@is_inferentia_test
@requires_neuronx
def test_vlm_generation_with_image(any_vlm_generate_model: dict[str, Any]):
    """Test VLM greedy generation with a real image matches the HF CPU reference."""
    image_path = os.path.join(os.path.dirname(__file__), "venus_botticelli.png")
    image = Image.open(image_path).convert("RGB")

    model_id = any_vlm_generate_model["model_id"]
    neuron_model_path = any_vlm_generate_model["neuron_model_path"]

    processor = AutoProcessor.from_pretrained(neuron_model_path)

    # SmolVLM / Idefics3 requires the chat template to emit image tokens correctly.
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "Can you describe this image?"},
            ],
        }
    ]
    text = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=text, images=[image], return_tensors="pt")
    max_new_tokens = 20

    # CPU reference
    cpu_model = AutoModelForImageTextToText.from_pretrained(model_id)
    cpu_outputs = cpu_model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
    cpu_text = processor.decode(cpu_outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    # Neuron model
    neuron_model = NeuronModelForImageTextToText.from_pretrained(neuron_model_path)
    neuron_outputs = neuron_model.generate(
        inputs["input_ids"],
        pixel_values=inputs["pixel_values"],
        do_sample=False,
        max_new_tokens=max_new_tokens,
    )
    neuron_text = processor.decode(neuron_outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)

    # Check output is non-empty and matches the CPU reference.
    assert len(neuron_text.strip()) > 0, "Neuron model produced empty output"
    assert cpu_text == neuron_text, f"Neuron and CPU outputs differ.\nNeuron: {neuron_text!r}\nCPU:    {cpu_text!r}"
    assert torch.equal(neuron_outputs, cpu_outputs), "Neuron and CPU outputs differ at the token level"
