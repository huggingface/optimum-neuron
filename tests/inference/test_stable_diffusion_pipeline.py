# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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

from parameterized import parameterized

from optimum.neuron import NeuronStableDiffusionPipeline
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx
from optimum.utils import logging
from optimum.utils.testing_utils import require_diffusers

from .inference_utils import NeuronModelTestMixin


logger = logging.get_logger()


@is_inferentia_test
@requires_neuronx
@require_diffusers
class NeuronModelForMultipleChoiceIntegrationTest(NeuronModelTestMixin):
    NEURON_MODEL_CLASS = NeuronStableDiffusionPipeline
    STATIC_INPUTS_SHAPES = {"batch_size": 1, "height": 64, "width": 64}
    SUPPORTED_ARCHITECTURES = [
        "stable-diffusion",
    ]

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_compare_to_diffusers_dyn_bs(self, model_arch):
        pass

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_compare_to_diffusers_non_dyn_bas(self, model_arch):
        pass
