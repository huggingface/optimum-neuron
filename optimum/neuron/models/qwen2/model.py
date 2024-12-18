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
# ==============================================================================

from transformers import PretrainedConfig

from ...backends.hlo.config import NeuronConfig
from ...backends.hlo.dtypes import to_torch_dtype
from ..llama.model import LlamaHloModel
from .modules import Qwen2ForCausalLM


class Qwen2ForSampling(LlamaHloModel):
    """The Qwen2 model is essentially a LLama model with bias in attention projections.

    The implementation in this class is very similar to the one used for Llama in Tnx.
    The only difference is the class used to instantiate the CPU model, that adds
    biases for attention projections.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        neuron_config: NeuronConfig,
    ):
        dtype = to_torch_dtype(neuron_config.amp)
        super().__init__(config, neuron_config, cpu_model=Qwen2ForCausalLM(config, dtype))
