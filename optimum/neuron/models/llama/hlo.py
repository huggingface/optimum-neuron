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
from typing import Optional

from transformers.models.llama import LlamaConfig

from ...backends.hlo import functional
from ...backends.hlo.config import Layout, NeuronConfig
from ...backends.hlo.decoder import DecoderGraphBuilder
from ...backends.hlo.layers import attention, rotary, transformer
from ...backends.hlo.utils import get_qkv_padding


class LlamaGraphBuilder(DecoderGraphBuilder):
    def __init__(self, config: LlamaConfig, neuron_config: Optional[NeuronConfig] = None):
        super().__init__(config, neuron_config)

    def pre_layer(
        self,
        hidden,
        cache_ids,
        start_ids,
    ):
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        pos_embed = rotary.hlo_rotary_embedding(
            hidden.dtype,
            head_dim,
            cache_ids,
            base=self.config.rope_theta,
            rope_scaling=self.config.rope_scaling,
        )

        mask, active_mask = functional.attention_mask(
            cache_ids,
            start_ids,
            self.neuron_config.n_positions,
        )

        return hidden, cache_ids, start_ids, pos_embed, mask, active_mask

    def layer(
        self,
        hidden,
        cache_ids,
        start_ids,
        pos_embed,
        mask,
        active_mask,
        attn_k_cache,
        attn_v_cache,
        pre_attn_ln_weight,
        pre_attn_ln_bias,
        attn_q_weight,
        attn_q_bias,
        attn_k_weight,
        attn_k_bias,
        attn_v_weight,
        attn_v_bias,
        attn_out_weight,
        attn_out_bias,
        post_attn_ln_weight,
        post_attn_ln_bias,
        pre_mlp_ln_weight,
        pre_mlp_ln_bias,
        mlp_in_weight,
        mlp_in_bias,
        mlp_out_weight,
        mlp_out_bias,
        post_mlp_ln_weight,
        post_mlp_ln_bias,
        in0_weight=None,
        in1_weight=None,
        out_weight=None,
    ):
        eps = self.config.rms_norm_eps
        is_bsh = self.neuron_config and self.neuron_config.attention_layout == Layout.BSH
        ln_hidden = functional.rms_norm(
            hidden,
            pre_attn_ln_weight,
            eps,
            dim=2 if is_bsh else 0,
        )
        attn_output, out_attn_k_cache, out_attn_v_cache = self.attention(
            ln_hidden,
            cache_ids,
            start_ids,
            pos_embed,
            mask,
            active_mask,
            attn_k_cache,
            attn_v_cache,
            attn_q_weight,
            attn_q_bias,
            attn_k_weight,
            attn_k_bias,
            attn_v_weight,
            attn_v_bias,
            attn_out_weight,
            attn_out_bias,
        )
        hidden = functional.add(attn_output, hidden)
        gated_mlp = functional.gated_mlp_bsh if is_bsh else functional.gated_mlp
        rms_norm_dim = 2 if is_bsh else 0
        norm_hidden = functional.rms_norm(
            hidden,
            pre_mlp_ln_weight,
            eps,
            dim=rms_norm_dim,
        )

        mlp_hidden = gated_mlp(
            norm_hidden,
            in0_weight,
            in1_weight,
            out_weight,
            activation_function="silu",
            neuron_config=self.neuron_config,
        )
        res_hidden = functional.add(mlp_hidden, hidden)
        return res_hidden, out_attn_k_cache, out_attn_v_cache

    def ln_lm_head(
        self,
        hidden,
        last_token_id,
        is_prefill,
        rms_weight,
        unused_bias,
        lm_head_weight,
        lm_head_bias,
    ):
        logits = transformer.rms_lm_head(
            self.neuron_config.tp_degree,
            hidden,
            last_token_id,
            rms_weight,
            lm_head_weight,
            lm_head_bias,
            is_prefill=is_prefill,
            eps=self.config.rms_norm_eps,
            neuron_config=self.neuron_config,
        )
        return logits

    def attention(
        self,
        hidden,
        cache_ids,
        start_ids,
        pos_embed,
        mask,
        active_mask,
        cached_keys,
        cached_values,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        v_weight,
        v_bias,
        out_weight,
        out_bias,
        qkv_tuple: tuple = None,
    ):
        d_head = self.config.hidden_size // self.config.num_attention_heads
        tp_degree = self.neuron_config.tp_degree

        # Compute the expected number of KV heads (Used in case fused QKV is used)
        n_kv_heads_tp = None
        if self.config.num_key_value_heads is not None:
            n_head = self.config.num_attention_heads
            n_kv_head = self.config.num_key_value_heads
            n_head_padded, n_kv_head_padded = get_qkv_padding(n_head, n_kv_head, self.neuron_config)
            n_kv_heads_tp = n_kv_head_padded // tp_degree

        # Q = (hidden @ wQ) + bQ
        # K = (hidden @ wK) + bK
        # V = (hidden @ wV) + bV
        if qkv_tuple:
            # If computed already, skip computation here
            assert active_mask is None
            query, key, value = qkv_tuple
        else:
            query, key, value = attention.query_key_value(
                hidden,
                q_weight,
                q_bias,
                k_weight,
                k_bias,
                v_weight,
                v_bias,
                d_head,
                neuron_config=self.neuron_config,
                n_kv_heads_tp=n_kv_heads_tp,
            )

        # Q = Rotate(Q)
        # K = Rotate(K)
        query, key = rotary.rotate_half(
            query,
            key,
            pos_embed,
            tp_degree=tp_degree,
        )

        # Q = Q / sqrt(d_head)
        query = attention.scale(query, d_head)

        # Single Token Generation (Decode)
        if active_mask is not None:
            cached_keys_s = cached_keys
            cached_values_s = cached_values

            # Sp = Q @ Kp
            prior_scores = attention.score(
                query,
                cached_keys_s,
                n_kv_heads=self.config.num_key_value_heads,
            )
            prior_scores = attention.mask(
                prior_scores,
                mask,
                tp_degree=tp_degree,
            )

            # Sa = Q @ Ka
            active_score = attention.score(
                query,
                key,
                n_kv_heads=self.config.num_key_value_heads,
            )
            active_score = attention.mask(
                active_score,
                active_mask,
                tp_degree=tp_degree,
            )

            # C = softmax(Sa, Sp) @ (Va, Vp)
            context = attention.context(
                prior_scores,
                active_score,
                cached_values_s,
                value,
                n_kv_heads=self.config.num_key_value_heads,
                tp_degree=self.neuron_config.tp_degree,
            )

            # KCache[I], VCache[I] = K, V
            updated_keys, updated_values = attention.fused_kv_update_cache(
                cached_keys,
                cached_values,
                cache_ids,
                key,
                value,
                start_ids,
                neuron_config=self.neuron_config,
            )

        # Multi-Token Context Encoding
        else:
            context = None
            if self.neuron_config.allow_flash_attention:
                context = attention.flash_attention(query, key, value)
            if context is None:
                # S = Q @ K

                score = attention.score(
                    query,
                    key,
                    n_kv_heads=self.config.num_key_value_heads,
                )
                score = attention.mask(
                    score,
                    mask,
                    tp_degree=tp_degree,
                )
                context = attention.context_combined(
                    score,
                    value,
                    n_kv_heads=self.config.num_key_value_heads,
                )

            # KCache, VCache = K, V
            if cached_keys.sizes == key.sizes:
                updated_keys, updated_values = key, value
            else:
                updated_keys, updated_values = attention.fused_kv_update_cache(
                    cached_keys,
                    cached_values,
                    cache_ids,
                    key,
                    value,
                    start_ids,
                    neuron_config=self.neuron_config,
                )

        # O = (C @ wO) + bO
        output = attention.output(context, out_weight, out_bias, tp_degree, self.neuron_config)
        return output, updated_keys, updated_values
