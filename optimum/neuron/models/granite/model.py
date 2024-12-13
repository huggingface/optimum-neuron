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
import warnings

import torch
from transformers import PretrainedConfig
from transformers_neuronx import base, bucket, decoder, ops, utils
from transformers_neuronx.config import NeuronConfig
from transformers_neuronx.constants import KV_SHARD_PAD, LAYOUT_HSB

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
        context_unroll: int = None,
        unroll: int = None,
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
        if self.neuron_config.shard_over_sequence:
            n_kv_head = self.config.num_key_value_heads
            kv_shard_degree = self.config.tp_degree // n_kv_head
            assert kv_shard_degree <= KV_SHARD_PAD, "increase kv_shard degree is higher than default 128"
            warnings.warn(f"shard over sequence enabled, increasing n_positions {n_positions} by 128")
            if isinstance(n_positions, list):
                npos = sorted(n_positions)
                npos[-1] += KV_SHARD_PAD
            else:
                npos = n_positions + KV_SHARD_PAD
            self.config.n_positions = npos
            config.n_positions = npos
            n_positions = npos
        if self.neuron_config.on_device_generation:
            self.neuron_config.on_device_generation.vocab_size = self.config.vocab_size

        self.layers_after_partition = self.neuron_config.auto_layer_partition(config.num_hidden_layers)
        self.prefixed_length = prefixed_length

        if context_unroll is None:
            context_unroll = len(self.layers_after_partition)
        self.context_unroll = context_unroll

        if unroll is None:
            unroll = len(self.layers_after_partition)
        self.unroll = unroll

        self.token_buckets = bucket.token_sizes(n_positions)
        self.context_buckets = bucket.context_sizes(context_length_estimate, self.token_buckets)
        # input length should be  divisable by tp_degree to activate seq paralle
        if neuron_config and neuron_config.sequence_parallel_norm:
            for bucket_size in self.context_buckets:
                if (
                    bucket_size > neuron_config.sequence_parallel_norm_threshold
                    and bucket_size % self.config.tp_degree != 0
                ):
                    raise ValueError(
                        f"Sequence parallel normalization requires the bucket size ({bucket_size}) to be divisible by the tensor parallel degree ({self.config.tp_degree})"
                    )
        self.window_context_buckets = []
        if prefixed_length:
            if prefixed_length not in self.context_buckets:
                self.context_buckets.append(prefixed_length)
                self.context_buckets = sorted(self.context_buckets)

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
            num_layers=len(self.layers_after_partition),
            n_head=config.num_attention_heads,
            n_kv_head=config.num_key_value_heads,
            unroll=unroll,
            neuron_config=self.neuron_config,
            allow_pad=True,
            builder=hlo_builder,
        )
        self.decoder_lm_head = self.decoder_param_set.init_token_decoder(
            unroll=self.unroll, buckets=self.token_buckets, model_obj=self
        )
        self.decoder_lm_head_for_context = self.decoder_param_set.init_context_decoder(
            unroll=self.context_unroll, buckets=self.context_buckets, model_obj=self
        )
        self.decoder_lm_head_for_speculation = {}
        self.decoder_lm_head_for_window_context = {}

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
            new_layer = self.decoder_lm_head.new_layer(is_unit_scale=is_unit_scale)
            new_layer.add_pre_attention_layer_norm(layer.input_layernorm.weight.detach(), None)
            new_layer.add_attention_query(attn.q_proj.weight.detach().T, None)
            new_layer.add_attention_key(attn.k_proj.weight.detach().T, None)
            new_layer.add_attention_value(attn.v_proj.weight.detach().T, None)
            if self.neuron_config and self.neuron_config.attn_output_transposed:
                new_layer.add_attention_output(attn.o_proj.weight.T.detach(), None, sharding=0, transposed=True)
            else:
                new_layer.add_attention_output(attn.o_proj.weight.detach(), None, sharding=1, transposed=False)
            new_layer.add_pre_mlp_layer_norm(layer.post_attention_layernorm.weight.detach(), None)

            # Note: Automatic MLP padding is safe since zeros are *only* introduced to intermediary state
            if self.neuron_config.fuse_mlp:
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
                new_layer.add_parameter(
                    mlp.gate_proj.weight.T, sharding=1, allow_pad=True, allow_quantize=True, allow_transform=True
                )
                new_layer.add_parameter(
                    mlp.up_proj.weight.T, sharding=1, allow_pad=True, allow_quantize=True, allow_transform=True
                )
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

    def materialize_embeddings(self):
        # Materialize the embedding to CPU
        self.chkpt_model.model.embed_tokens.materialize()

    def init_rest_of_model(self):
        # Pipeline sparallel deosn't support executor right now
        if not self.neuron_config.is_pp():
            self.decoder_lm_head.use_executor = True

        if self.context_buckets:
            for context_length_estimate in self.context_buckets:
                for batch_size in self.context_batch_sizes:
                    model = self.decoder_lm_head.build_weight_shared(
                        share_caches=True, new=self.decoder_lm_head_for_context[context_length_estimate, batch_size]
                    )
                    # PERF: No latency improvement seen in multi-layer models from executor
                    # Pipeline parallel deosn't support executor right now
                    if self.context_unroll == self.config.num_hidden_layers and not self.neuron_config.is_pp():
                        model.use_executor = True
                    self.decoder_lm_head_for_context[context_length_estimate, batch_size] = model

        if self.decoder_lm_head_for_speculation:
            for i, k in enumerate(self.decoder_lm_head_for_speculation):
                model = self.decoder_lm_head.build_weight_shared(
                    share_caches=True,
                    new=self.decoder_lm_head_for_speculation[k],
                    embed_weight=self.chkpt_model.model.embed_tokens.weight,
                )
                self.decoder_lm_head_for_speculation[k] = model

        if self.decoder_lm_head_for_window_context:
            for i, k in enumerate(self.decoder_lm_head_for_window_context):
                model = self.decoder_lm_head.build_weight_shared(
                    share_caches=True, new=self.decoder_lm_head_for_window_context[k]
                )
                self.decoder_lm_head_for_window_context[k] = model

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
        return padded_inputs, input_embeddings, *rst

    def forward(self, input_ids, cache_ids=None, start_ids=None, last_token_id=None, input_embeddings=None, **kwargs):
        if last_token_id is not None:  # preprocess_and_embed() has already been invoked
            rst = cache_ids, start_ids, last_token_id
        else:  # invoke preprocess_and_embed()
            input_ids, input_embeddings, *rst = self.preprocess_and_embed(input_ids, cache_ids, start_ids, **kwargs)
        # either input_embeddings are generated (off device embedding), or input_ids will be padded from preprocess_and_embed (on device embedding)
        inputs = input_embeddings if input_embeddings is not None else input_ids
        logits = self._forward(inputs, *rst)
        logits = self._postprocess(logits, start_ids=start_ids, **kwargs)
        return logits
