# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
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
from ...backends.hlo import module


class LlamaForCausalLM(module.PretrainedModel):
    def __init__(self, config, dtype):
        super().__init__()
        self.model = LlamaModel(config, dtype)
        self.lm_head = module.LowMemoryLazyLinear(config.vocab_size, dtype=dtype, bias=False)

    def get_tied_parameters(self):
        return [(self.model.embed_tokens.weight, self.lm_head.weight)]

    def get_base_model(self):
        return self.model


class LlamaModel(module.LowMemoryModule):
    def __init__(self, config, dtype):
        super().__init__()
        self.embed_tokens = module.LowMemoryEmbedding(config.vocab_size, config.hidden_size)
        self.layers = module.LowMemoryModuleList(
            [LlamaDecoderLayer(config, dtype) for _ in range(config.num_hidden_layers)]
        )
        self.norm = LlamaRMSNorm()


class LlamaRMSNorm(module.LowMemoryModule):
    def __init__(self) -> None:
        super().__init__()
        self.weight = module.UninitializedParameter()


class LlamaDecoderLayer(module.LowMemoryModule):
    def __init__(self, config, dtype):
        super().__init__()
        self.self_attn = LlamaAttention(config, dtype)
        self.mlp = LlamaMLP(config, dtype)
        self.input_layernorm = LlamaRMSNorm()
        self.post_attention_layernorm = LlamaRMSNorm()


class LlamaAttention(module.LowMemoryModule):
    def __init__(self, config, dtype):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.q_proj = module.LowMemoryLazyLinear(self.num_heads * self.head_dim, bias=False, dtype=dtype)
        self.k_proj = module.LowMemoryLazyLinear(self.num_heads * self.head_dim, bias=False, dtype=dtype)
        self.v_proj = module.LowMemoryLazyLinear(self.num_heads * self.head_dim, bias=False, dtype=dtype)
        self.o_proj = module.LowMemoryLazyLinear(self.hidden_size, bias=False, dtype=dtype)


class LlamaMLP(module.LowMemoryModule):
    def __init__(self, config, dtype):
        super().__init__()
        self.gate_proj = module.LowMemoryLazyLinear(config.intermediate_size, bias=False, dtype=dtype)
        self.up_proj = module.LowMemoryLazyLinear(config.intermediate_size, bias=False, dtype=dtype)
        self.down_proj = module.LowMemoryLazyLinear(config.hidden_size, bias=False, dtype=dtype)
