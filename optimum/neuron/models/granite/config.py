# coding=utf-8
# Copyright 2024 The HuggingFace Team. All rights reserved.
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

from transformers import PretrainedConfig
from transformers_neuronx.llama.config import LlamaConfig


class GraniteConfig(LlamaConfig):
    """The Granite model uses the same configuration as the TnX LLama model"""

    def __init__(
        self, config: PretrainedConfig, n_positions: int, batch_size: int, amp: str, tp_degree: int, **kwargs
    ):
        super().__init__(config, n_positions, batch_size, amp, tp_degree, **kwargs)
        self.model_type = "granite"
        # These are parameters specific to the granite modeling
        self.attention_multiplier = config.attention_multiplier
        self.embedding_multiplier = config.embedding_multiplier
        self.logits_scaling = config.logits_scaling
        self.residual_multiplier = config.residual_multiplier
