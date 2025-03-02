# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
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
from transformers.models.phi import PhiConfig

from ...backends.hlo import module


class Phi4ForCausalLM(module.PretrainedModel):
    def __init__(self, config: PhiConfig, dtype):
        super().__init__()
        self.model = Phi4Model(config, dtype)
        self.lm_head = module.LowMemoryLazyLinear(config.vocab_size, dtype=dtype, bias=False)

    def get_tied_parameters(self):
        return [(self.model.embed_tokens.weight, self.lm_head.weight)]

    def get_base_model(self):
        return self.model


class Phi4Model(module.LowMemoryModule):
    def __init__(self, config: PhiConfig, dtype):
        super().__init__()
        self.embed_tokens = module.LowMemoryEmbedding(config.vocab_size, config.hidden_size)
        self.layers = module.LowMemoryModuleList(
            [Phi4DecoderLayer(config, dtype) for _ in range(config.num_hidden_layers)]
        )
        self.norm = Phi4RMSNorm()


class Phi4RMSNorm(module.LowMemoryModule):
    def __init__(self) -> None:
        super().__init__()
        self.weight = module.UninitializedParameter()


class Phi4DecoderLayer(module.LowMemoryModule):
    def __init__(self, config: PhiConfig, dtype):
        super().__init__()
        self.self_attn = Phi4Attention(config, dtype)
        self.mlp = Phi4MLP(config, dtype)
        self.input_layernorm = Phi4RMSNorm()
        self.post_attention_layernorm = Phi4RMSNorm()


class Phi4Attention(module.LowMemoryModule):
    def __init__(self, config: PhiConfig, dtype):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        op_size = config.num_attention_heads * self.head_dim + 2 * (config.num_key_value_heads * self.head_dim)
        self.qkv_proj = module.LowMemoryLazyLinear(op_size, bias=False, dtype=dtype)
        self.o_proj = module.LowMemoryLazyLinear(self.hidden_size, bias=False, dtype=dtype)


class Phi4MLP(module.LowMemoryModule):
    def __init__(self, config, dtype):
        super().__init__()
        self.gate_up_proj = module.LowMemoryLazyLinear(2 * config.intermediate_size, bias=False, dtype=dtype)
        self.down_proj = module.LowMemoryLazyLinear(config.hidden_size, bias=False, dtype=dtype)
