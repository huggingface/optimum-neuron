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

from ..backend.config import HloNeuronConfig
from ..backend.dtypes import to_torch_dtype
from ..backend.modeling_decoder import HloModelForCausalLM
from ..llama.model import LlamaHloModel
from .modules import Qwen2ForCausalLM


class Qwen2HloModel(LlamaHloModel):
    """The Qwen2 model is essentially a LLama model with bias in attention projections.

    The implementation in this class is very similar to the one used for Llama in Tnx.
    The only difference is the class used to instantiate the CPU model, that adds
    biases for attention projections.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        neuron_config: HloNeuronConfig,
    ):
        dtype = to_torch_dtype(neuron_config.auto_cast_type)
        super().__init__(config, neuron_config, cpu_model=Qwen2ForCausalLM(config, dtype))


class Qwen2HloModelForCausalLM(HloModelForCausalLM):
    """Qwen2HloModelForCausalLM is a wrapper around Qwen2HloModel for causal language modeling tasks.

    It inherits from HloModelForCausalLM and provides the specific export configuration for Qwen2 models.
    """

    neuron_model_class = Qwen2HloModel

    @classmethod
    def _get_neuron_config(
        cls,
        checkpoint_id: str,
        checkpoint_revision: str,
        batch_size: int,
        sequence_length: int,
        auto_cast_type: str,
        tensor_parallel_size: int,
    ):
        return HloModelForCausalLM._get_neuron_config(
            checkpoint_id=checkpoint_id,
            checkpoint_revision=checkpoint_revision,
            batch_size=batch_size,
            sequence_length=sequence_length,
            tensor_parallel_size=tensor_parallel_size,
            auto_cast_type=auto_cast_type,
            fuse_qkv=False,  # Don't fuse QKV because of the bias in Qwen2
        )
