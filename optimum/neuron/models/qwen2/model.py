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

from transformers import PretrainedConfig
from transformers_neuronx.base import NeuronHloDecoderModel
from transformers_neuronx.config import NeuronConfig
from transformers_neuronx.decoder import DecoderGraph
from transformers_neuronx.dtypes import to_torch_dtype
from transformers_neuronx.llama.hlo import LlamaGraphBuilder
from transformers_neuronx.ops import init_neuron

from .modules import Qwen2ForCausalLM


class Qwen2ForSampling(NeuronHloDecoderModel):
    """The Qwen2 model is essentially a LLama model with bias in linear projections.

    The implementation in this class is very similar to the one used for Llama in Tnx.
    The only differences are:
    - the config (Qwen2Config) and base model (Qwen2ForCausalLM) used in __init__,
    - the addition of biases parameters when loading weights from the checkpoint model.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        neuron_config: NeuronConfig,
    ):
        dtype = to_torch_dtype(neuron_config.amp)
        super().__init__(Qwen2ForCausalLM, config, dtype)
        self.config = config
        self.neuron_config = neuron_config if neuron_config else NeuronConfig()

        hlo_builder = LlamaGraphBuilder(config, neuron_config=self.neuron_config)
        self.decoder_param_set = DecoderGraph(
            config=config,
            neuron_config=neuron_config,
            n_active_tokens=1,
            builder=hlo_builder,
        )
        self.decoder_lm_head = self.decoder_param_set.init_token_decoder(model_obj=self)
        self.decoder_lm_head_for_context = self.decoder_param_set.init_context_decoder(model_obj=self)

    def load_weights(self):
        # Materialize the embedding to CPU
        self.chkpt_model.model.embed_tokens.materialize()

        for layer in self.chkpt_model.model.layers:
            layer.materialize()
            attn = layer.self_attn
            mlp = layer.mlp
            new_layer = self.decoder_lm_head.new_layer()
            new_layer.add_pre_attention_layer_norm(layer.input_layernorm.weight.detach(), None)
            new_layer.add_attention_query(attn.q_proj.weight.detach().T, attn.q_proj.bias.detach())
            new_layer.add_attention_key(attn.k_proj.weight.detach().T, attn.k_proj.bias.detach())
            new_layer.add_attention_value(attn.v_proj.weight.detach().T, attn.v_proj.bias.detach())
            if self.neuron_config and self.neuron_config.attn_output_transposed:
                new_layer.add_attention_output(attn.o_proj.weight.T.detach(), None, sharding=0, transposed=True)
            else:
                new_layer.add_attention_output(attn.o_proj.weight.detach(), None, sharding=1, transposed=False)
            new_layer.add_pre_mlp_layer_norm(layer.post_attention_layernorm.weight.detach(), None)

            new_layer.add_parameter(mlp.gate_proj.weight.T, sharding=1, allow_transform=True)
            new_layer.add_parameter(mlp.up_proj.weight.T, sharding=1, allow_transform=True)
            new_layer.add_parameter(mlp.down_proj.weight, sharding=1)
            new_layer.to_neuron()
            layer.nullify()
        ln_f = self.chkpt_model.model.norm
        ln_f.materialize()
        self.decoder_lm_head.add_final_layer_norm(ln_f.weight.detach(), None)

        lm_head = self.chkpt_model.lm_head
        lm_head.materialize()
        self.decoder_lm_head.add_lm_head(lm_head.weight.detach().T)
        lm_head.nullify()

        self.decoder_lm_head.to_neuron()
        self.decoder_lm_head.use_executor = True

        self.decoder_lm_head_for_context.load_shared_weights(self.decoder_lm_head)
        self.decoder_lm_head_for_context.use_executor = True
