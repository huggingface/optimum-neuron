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
from tempfile import TemporaryDirectory

import pytest
import torch

from optimum.neuron import NeuronFluxKontextPipeline, NeuronFluxPipeline
from optimum.neuron.utils.testing_utils import requires_neuronx


@pytest.fixture(scope="module")
@requires_neuronx
def neuron_flux_tp2_path():
    compiler_args = {"auto_cast": "none"}
    input_shapes = {
        "batch_size": 1,
        "height": 8,
        "width": 8,
        "num_images_per_prompt": 1,
        "sequence_length": 256,
    }

    neuron_pipeline = NeuronFluxPipeline.from_pretrained(
        "hf-internal-testing/tiny-flux-pipe-gated-silu",
        export=True,
        torch_dtype=torch.bfloat16,
        tensor_parallel_size=2,
        dynamic_batch_size=False,
        disable_neuron_cache=True,
        **input_shapes,
        **compiler_args,
    )
    model_dir = TemporaryDirectory()
    model_path = model_dir.name
    neuron_pipeline.save_pretrained(model_path)
    del neuron_pipeline
    yield model_path


@pytest.fixture(scope="module")
@requires_neuronx
def neuron_flux_kontext_tp2_path():
    compiler_args = {"auto_cast": "none"}
    input_shapes = {
        "batch_size": 1,
        "height": 8,
        "width": 8,
        "num_images_per_prompt": 1,
        "sequence_length": 256,
    }

    neuron_pipeline = NeuronFluxKontextPipeline.from_pretrained(
        "hf-internal-testing/tiny-flux-kontext-pipe-gated-silu",
        export=True,
        torch_dtype=torch.bfloat16,
        tensor_parallel_size=2,
        dynamic_batch_size=False,
        disable_neuron_cache=True,
        **input_shapes,
        **compiler_args,
    )
    model_dir = TemporaryDirectory()
    model_path = model_dir.name
    neuron_pipeline.save_pretrained(model_path)
    del neuron_pipeline
    yield model_path
