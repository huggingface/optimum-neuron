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

import torch
from transformers import PretrainedConfig

from ...backends.hlo.config import NeuronConfig
from ...backends.hlo.dtypes import to_torch_dtype
from ..llama.model import LlamaHloModel
from .modules import Phi3ForCausalLM


class Phi3HloModel(LlamaHloModel):
    """The Phi3 model is essentially a LLama model with fused qkv and gate_up projections.

    The implementation in this class is very similar to the one used for Llama in Tnx.
    The only difference is that the fused qkv and gate/up linear projection are split when
    loading weights (note that they might be fused again when transferring the weights to the
    neuron device if the NeuronConfig specifies it).
    """

    def __init__(
        self,
        config: PretrainedConfig,
        neuron_config: NeuronConfig,
    ):
        dtype = to_torch_dtype(neuron_config.amp)
        super().__init__(config, neuron_config, cpu_model=Phi3ForCausalLM(config, dtype))

    def load_weights(self):
        # Materialize the embedding to CPU
        self.cpu_model.model.embed_tokens.materialize()

        for layer in self.cpu_model.model.layers:
            layer.materialize()
            attn = layer.self_attn
            mlp = layer.mlp
            new_layer = self.decoder_lm_head.new_layer()
            new_layer.add_pre_attention_layer_norm(layer.input_layernorm.weight.detach(), None)
            # Transpose and split fused qkv_proj into separate weights
            fused_attn = attn.qkv_proj.weight.clone().detach().T
            # Extract the larger query weights first
            q_features = attn.num_heads * attn.head_dim
            q_weight = fused_attn[:, :q_features]
            # Then split the remaining into key and value weights
            k_weight, v_weight = torch.chunk(fused_attn[:, q_features:], 2, dim=1)
            new_layer.add_attention_query(q_weight, None)
            new_layer.add_attention_key(k_weight, None)
            new_layer.add_attention_value(v_weight, None)
            if self.neuron_config and self.neuron_config.attn_output_transposed:
                new_layer.add_attention_output(attn.o_proj.weight.T.detach(), None, sharding=0, transposed=True)
            else:
                new_layer.add_attention_output(attn.o_proj.weight.detach(), None, sharding=1, transposed=False)

            new_layer.add_pre_mlp_layer_norm(layer.post_attention_layernorm.weight.detach(), None)
            # Tanspose and split fused mlp into separate weights
            fused_gate_up = mlp.gate_up_proj.weight.clone().detach().T
            gate, up = torch.chunk(fused_gate_up, 2, dim=1)
            new_layer.add_parameter(gate, sharding=1, allow_transform=True)
            new_layer.add_parameter(up, sharding=1, allow_transform=True)
            new_layer.add_parameter(mlp.down_proj.weight, sharding=1)
            new_layer.to_neuron()
            layer.nullify()

        ln_f = self.cpu_model.model.norm
        ln_f.materialize()
        self.decoder_lm_head.add_final_layer_norm(ln_f.weight.detach(), None)
        ln_f.nullify()

        lm_head = self.cpu_model.lm_head
        lm_head.materialize()
        self.decoder_lm_head.add_lm_head(lm_head.weight.detach().T)
        lm_head.nullify()

        self.decoder_lm_head.to_neuron()
        self.decoder_lm_head.use_executor = True

        self.decoder_lm_head_for_context.load_shared_weights(self.decoder_lm_head)
        self.decoder_lm_head_for_context.use_executor = True
