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
from transformers_neuronx import ops

from ...backends.hlo.config import NeuronConfig
from ...backends.hlo.dtypes import to_torch_dtype
from ..llama.model import LlamaHloModel
from .modules import Phi4ForCausalLM


class Phi4ForSampling(LlamaHloModel):
    """The Phi4 model is essentially a LLama model with fused qkv and gate_up projections.

    The implementation in this class is very similar to the one used for Llama in Tnx.
    The only differences are:
    - the config (Phi4Config) and base model (Phi4ForCausalLM) used in __init__,
    - the addition of biases parameters when loading weights from the checkpoint model.
    """

    def __init__(
        self,
        config: PretrainedConfig,
        neuron_config: NeuronConfig,
    ):
        dtype = to_torch_dtype(neuron_config.amp)
        super().__init__(config, neuron_config, cpu_model=Phi4ForCausalLM(config, dtype))

    def load_weights(self):
        self.materialize_embeddings()
        ops.init()

        for layer_id, layer in enumerate(self.chkpt_model.model.layers):
            if layer_id not in self.layers_after_partition:
                continue
            layer.materialize()
            attn = layer.self_attn
            mlp = layer.mlp
            if self.neuron_config and self.neuron_config.quant:
                is_unit_scale = self.neuron_config.quant.is_unit_scale(layer_id)
            else:
                is_unit_scale = False

            # Split fused qkv_proj and mlp into separate weights
            fused_attn = attn.qkv_proj.weight.clone().detach()
            fused_gate_up = mlp.gate_up_proj.weight.clone().detach()
            q_weight, k_weight, v_weight = torch.chunk(fused_attn, 3, dim=0)
            gate, up = torch.chunk(fused_gate_up, 2, dim=0)

            new_layer = self.decoder_lm_head.new_layer(is_unit_scale=is_unit_scale)
            new_layer.add_pre_attention_layer_norm(layer.input_layernorm.weight.detach(), None)
            new_layer.add_attention_query(q_weight)
            new_layer.add_attention_key(k_weight)
            new_layer.add_attention_value(v_weight)
            if self.neuron_config and self.neuron_config.attn_output_transposed:
                new_layer.add_attention_output(attn.o_proj.weight.T.detach(), None, sharding=0, transposed=True)
            else:
                new_layer.add_attention_output(attn.o_proj.weight.detach(), None, sharding=1, transposed=False)
            new_layer.add_pre_mlp_layer_norm(layer.post_attention_layernorm.weight.detach(), None)

            # Note: Automatic MLP padding is safe since zeros are *only* introduced to intermediary state
            if self.neuron_config.fuse_mlp:
                assert fused_gate_up.shape[0] % self.config.tp_degree == 0, (
                    f"mlp weights are not divisible by tp_degree {self.config.tp_degree}"
                )
                new_layer.add_mlp_input(fused_gate_up)
                if self.neuron_config.mlp_out_weight_transpose:
                    new_layer.add_mlp_output(
                        mlp.down_proj.weight.T.detach(),
                        None,
                        sharding=0,
                        transposed=True,
                    )
                else:
                    new_layer.add_mlp_output(
                        mlp.down_proj.weight.detach(),
                        None,
                        sharding=1,
                        transposed=False,
                    )
            else:
                new_layer.add_parameter(gate, sharding=1, allow_pad=True, allow_quantize=True, allow_transform=True)
                new_layer.add_parameter(up, sharding=1, allow_pad=True, allow_quantize=True, allow_transform=True)
                if self.neuron_config.weight_tiling:
                    new_layer.add_parameter(
                        mlp.down_proj.weight.T, sharding=0, allow_pad=True, allow_quantize=True, allow_transform=True
                    )
                else:
                    if self.neuron_config.mlp_out_weight_transpose:
                        new_layer.add_parameter(
                            mlp.down_proj.weight.T, sharding=0, allow_pad=True, allow_quantize=True
                        )
                    else:
                        new_layer.add_parameter(
                            mlp.down_proj.weight, sharding=1, allow_pad=True, allow_quantize=True, out_feature_dim=0
                        )
            new_layer.to_neuron()
            layer.nullify()
        if self.neuron_config.shard_over_sequence:
            self.decoder_lm_head.add_pre_layer_parameter(torch.arange(self.config.tp_degree), sharding=0)
        # For pipeline parallel, we need to load ln and lm_head for now even if the pipeline stage doesn't compute the, because
        # 1) we need the ln_lm_head hlo for pp0 to get the logits shape and dtype
        # 2) we don't needs these for intermediate pp stages, but to keep things simple, just include ln_lm_head for all pp stages for now
        # 3) to get ln_lm_head hlo, we need to do weight loading and sharding
        # 4) this will introduce extra memory allocation, but ln_lm_head i/o tensor is much smaller and we can get rid of it when we can construct hlo in init
        ln_f = self.chkpt_model.model.norm
        ln_f.materialize()
        self.decoder_lm_head.add_final_layer_norm(ln_f.weight.detach(), None)

        lm_head = self.chkpt_model.lm_head
        lm_head.materialize()
        self.decoder_lm_head.add_lm_head(lm_head.weight.detach().T)
        if self.neuron_config.on_device_embedding:
            if self.neuron_config.sequence_parallel_norm:
                self.decoder_lm_head.add_pre_layer_parameter(
                    self.chkpt_model.model.embed_tokens.weight, sharding=None, allow_pad=True
                )
            else:
                self.decoder_lm_head.add_pre_layer_parameter(
                    self.chkpt_model.model.embed_tokens.weight, sharding=1, allow_pad=True
                )
        lm_head.nullify()

        self.decoder_lm_head.to_neuron()
        self.init_rest_of_model()
