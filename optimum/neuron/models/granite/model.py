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
from transformers_neuronx import base, bucket, decoder, utils
from transformers_neuronx.config import NeuronConfig
from transformers_neuronx.constants import LAYOUT_HSB

from .config import GraniteConfig
from .hlo import GraniteForSamplingNoEmbeddingHlo
from .modules import GraniteForCausalLM


class GraniteForSampling(base.NeuronModelBase):
    """The Granite model is a LLama model with 4 scalar multpliers that are applied to:
    - the embeddings,
    - the QK product in the attention (instead of the static 1/sqrt(num_heads))
    - the MLP outputs
    - the lm_head logits
    The implementation in this class is very similar to the one used for Llama in Tnx.
    The only differences are:
    - the config (GraniteConfig) and base model (GraniteForCausalLM) used in __init__,
    - the multiplication of the logits by the logits multiplier
    """

    def __init__(
        self,
        config: PretrainedConfig,
        *,
        n_positions: int = 2048,
        batch_size: int = 1,
        amp: str = "f32",
        tp_degree: int = 2,
        context_length_estimate: int = None,
        neuron_config: NeuronConfig = None,
        prefixed_length: int = 0,
        **kwargs,
    ):
        config = GraniteConfig(config, n_positions, batch_size, amp, tp_degree)
        super().__init__(GraniteForCausalLM, config)
        self.context_pre_hook = None
        self.context_hook = None
        self.config = config
        self.neuron_config = neuron_config if neuron_config else NeuronConfig()
        self.prefixed_length = prefixed_length
        self.batch_sizes = bucket.batch_sizes(batch_size)
        self.context_batch_sizes = (
            [1] if self.neuron_config and self.neuron_config.continuous_batching else self.batch_sizes
        )
        hlo_builder = GraniteForSamplingNoEmbeddingHlo(config, neuron_config=self.neuron_config)

        self.decoder_param_set = decoder.DecoderLmHeadForSamplingNoEmbedding(
            tp_degree=tp_degree,
            n_positions_list=self.token_buckets,
            n_active_tokens=1,
            batch_size=self.batch_sizes,
            attention_head_size=config.attention_head_size,
            amp=amp,
            num_layers=self.config.num_hidden_layers,
            n_head=config.num_attention_heads,
            n_kv_head=config.num_key_value_heads,
            neuron_config=self.neuron_config,
            allow_pad=True,
            builder=hlo_builder,
        )
        self.decoder_lm_head = self.decoder_param_set.init_token_decoder(model_obj=self)
        self.decoder_lm_head_for_context = self.decoder_param_set.init_context_decoder(model_obj=self)

    def load_weights(self):
        self.materialize_embeddings()

        for layer in self.chkpt_model.model.layers:
            layer.materialize()
            attn = layer.self_attn
            mlp = layer.mlp
            is_unit_scale = False
            new_layer = self.decoder_lm_head.new_layer(is_unit_scale=is_unit_scale)
            if self.neuron_config.has_pre_attention_norm:
                new_layer.add_pre_attention_layer_norm(layer.input_layernorm.weight.detach(), None)
            new_layer.add_attention_query(attn.q_proj.weight.detach().T, None)
            new_layer.add_attention_key(attn.k_proj.weight.detach().T, None)
            new_layer.add_attention_value(attn.v_proj.weight.detach().T, None)
            if self.neuron_config and self.neuron_config.attn_output_transposed:
                new_layer.add_attention_output(attn.o_proj.weight.T.detach(), None, sharding=0, transposed=True)
            else:
                new_layer.add_attention_output(attn.o_proj.weight.detach(), None, sharding=1, transposed=False)

            if self.neuron_config.fused_rmsnorm_mlp:
                dummy_post_attention_ln_weight = torch.ones_like(layer.post_attention_layernorm.weight.detach())
                new_layer.add_pre_mlp_layer_norm(dummy_post_attention_ln_weight, None)
            else:
                new_layer.add_pre_mlp_layer_norm(layer.post_attention_layernorm.weight.detach(), None)

            # Note: Automatic MLP padding is safe since zeros are *only* introduced to intermediary state
            if self.neuron_config.fused_rmsnorm_mlp:
                fused_pre_mlp_ln_gate_weight = (
                    mlp.gate_proj.weight
                    * layer.post_attention_layernorm.weight.detach().to(dtype=mlp.gate_proj.weight.dtype)
                )
                new_layer.add_parameter(
                    fused_pre_mlp_ln_gate_weight.T, sharding=1, allow_pad=True, allow_quantize=True
                )
                fused_pre_mlp_ln_up_weight = mlp.up_proj.weight * layer.post_attention_layernorm.weight.detach().to(
                    dtype=mlp.up_proj.weight.dtype
                )
                new_layer.add_parameter(fused_pre_mlp_ln_up_weight.T, sharding=1, allow_pad=True, allow_quantize=True)
                new_layer.add_parameter(
                    mlp.down_proj.weight.T, sharding=0, allow_pad=True, allow_quantize=True, out_feature_dim=0
                )
            elif self.neuron_config.fuse_mlp:
                assert all(
                    getattr(mlp, attr, None) for attr in ["gate_proj", "up_proj"]
                ), "fuse_mlp need to have gate and up proj weights"
                assert all(
                    getattr(mlp, attr, None).weight.shape[0] % self.config.tp_degree == 0
                    for attr in ["gate_proj", "up_proj"]
                ), f" mlp weights are not  divisible tp_degree {self.config.tp_degree}"
                mlp_in_weight = utils.interleave_mlp(
                    mlp.gate_proj.weight, mlp.up_proj.weight, tp_degree=self.config.tp_degree, dim=0
                )
                new_layer.add_mlp_input(mlp_in_weight.T.detach(), None)
                new_layer.add_mlp_output(
                    mlp.down_proj.weight.detach(),
                    None,
                    sharding=1,
                    transposed=False,
                )
            else:
                new_layer.add_parameter(mlp.gate_proj.weight.T, sharding=1, allow_pad=True, allow_transform=True)
                new_layer.add_parameter(mlp.up_proj.weight.T, sharding=1, allow_pad=True, allow_transform=True)
                new_layer.add_parameter(
                    mlp.down_proj.weight,
                    sharding=1,
                    allow_pad=True,
                )
            new_layer.to_neuron()
            layer.nullify()

        ln_f = self.chkpt_model.model.norm
        ln_f.materialize()
        self.decoder_lm_head.add_final_layer_norm(ln_f.weight.detach(), None)
        ln_f.nullify()

        lm_head = self.chkpt_model.lm_head
        lm_head.materialize()
        self.decoder_lm_head.add_lm_head(lm_head.weight.detach().T)
        lm_head.nullify()

        self.decoder_lm_head.to_neuron()
        self.init_rest_of_model()
        self.maybe_nullify_embeddings()

    def materialize_embeddings(self):
        # Materialize the embedding to CPU
        self.chkpt_model.model.embed_tokens.materialize()

    def maybe_nullify_embeddings(self):
        if self.neuron_config.on_device_embedding:
            self.chkpt_model.model.embed_tokens.nullify()

    def init_rest_of_model(self):
        # Pipeline sparallel deosn't support executor right now
        self.decoder_lm_head.use_executor = True

    def set_prefixed(self, input_ids):
        self.prefixed_input_ids = input_ids[:, : self.prefixed_length]
        prefixed_length = self.prefixed_length
        self.prefixed_length = 0
        self.forward(self.prefixed_input_ids)
        self.prefixed_length = prefixed_length

    def preprocess_and_embed(self, input_ids, cache_ids=None, start_ids=None, **kwargs):
        padded_inputs, *rst = self._preprocess(input_ids, start_ids=start_ids, cache_ids=cache_ids, **kwargs)
        if not self.neuron_config.on_device_embedding:
            input_embeddings = self.chkpt_model.model.embed_tokens(padded_inputs)
            if self.neuron_config.attention_layout == LAYOUT_HSB:
                input_embeddings = input_embeddings.transpose(0, -1).contiguous()
        else:
            # embedding layer is on device and will be computed as part of self._forward(), so don't compute here
            input_embeddings = None
        return input_embeddings, *rst

    def forward(self, input_ids, cache_ids, start_ids):
        original_input_ids = input_ids
        input_embeddings, *rst = self.preprocess_and_embed(input_ids, cache_ids, start_ids)
        logits = self._forward(input_embeddings, *rst)
        # Granite specific: divide logits by scaling factor
        logits = logits / self.config.logits_scaling
        return self._postprocess(original_input_ids, logits, start_ids=start_ids)
