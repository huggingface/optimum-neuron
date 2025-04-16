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
from ..backend.modeling_decoder import HloModelForCausalLM
from ..llama.model import LlamaHloModel
from .hlo import GraniteGraphBuilder


class GraniteHloModel(LlamaHloModel):
    """The Granite model is a LLama model with 4 scalar multpliers that are applied to:
    - the embeddings,
    - the QK product in the attention (instead of the static 1/sqrt(num_heads))
    - the MLP outputs
    - the lm_head logits
    The implementation in this class is very similar to the one used for Llama.
    The only differences are:
    - the specific graph builder used to insert the scaling operations,
    - the overloaded forward to add multiplication of the logits by the logits multiplier.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        neuron_config: HloNeuronConfig,
    ):
        super().__init__(config, neuron_config, hlo_builder=GraniteGraphBuilder(config, neuron_config))

    def forward(self, input_ids, cache_ids, start_ids):
        logits = super().forward(input_ids, cache_ids, start_ids)
        # Granite specific: divide logits by scaling factor
        return logits / self.config.logits_scaling


class GraniteHloModelForCausalLM(HloModelForCausalLM):
    """GraniteHloModelForCausalLM is a wrapper around GraniteHloModel for causal language modeling tasks.

    It inherits from HloModelForCausalLM and provides the specific export configuration for granite models.
    """

    neuron_model_class = GraniteHloModel

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
        )
