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
from typing import Optional

from transformers_neuronx import constants, hlo, utils
from transformers_neuronx.config import NeuronConfig
from transformers_neuronx.constants import LAYOUT_BSH, LAYOUT_HSB
from transformers_neuronx.hlo import dequantize_kv_cache_direct_cast, quantize_kv_cache_direct_cast
from transformers_neuronx.layers import attention, attention_utils, flash_decoding, rotary, transformer
from transformers_neuronx.nki.compile import nki_call

from .config import GraniteConfig


def scale_mul(t, scale):
    """Multiply a tensor by a float scale"""
    dtype = t.dtype
    # Convert float to a constant scalar tensor of the target dtype
    scale_t = dtype.Constant(constant_value=scale)
    # Expand the scalar tensor to the target shape
    scale_br_t = dtype[t.sizes].Broadcast(scale_t, dimensions=[])
    return dtype[t.sizes].Multiply(t, scale_br_t)


class GraniteForSamplingNoEmbeddingHlo:

    def __init__(self, config: GraniteConfig, neuron_config: Optional[NeuronConfig] = None):
        self.config = config
        self.neuron_config = neuron_config
        self.n_positions = None
        self.num_active_blocks = None

    @property
    def shard_over_batch(self):
        # Property access allows fallback configuration to be enabled after construction
        return (
            self.neuron_config is not None
            and self.neuron_config.group_query_attention == constants.GQA.SHARD_OVER_BATCH
        )

    def inputs(self, scribe, dtype, n_active_tokens, batch_size):
        tensors, dims = transformer.inputs(
            scribe,
            dtype,
            batch_size,
            n_active_tokens,
            self.config.hidden_size,
            self.neuron_config,
            self.config.tp_degree,
        )

        return tensors, dims

    def token_tree_inputs(self, scribe, dtype, n_active_tokens, batch_size):
        tensors, dims = self.inputs(scribe, dtype, n_active_tokens, batch_size)
        s32 = scribe.s32
        cache_2d = self.neuron_config and self.neuron_config.use_2d_cache_ids
        # Allow tree based speculation inputs
        if cache_2d:
            position_sizes = batch_size, n_active_tokens
            previous_cache_ids = s32[position_sizes].Parameter(parameter_number=4)
            reorder_mapping = s32[position_sizes].Parameter(parameter_number=5)
        else:
            previous_cache_ids = s32[n_active_tokens].Parameter(parameter_number=4)
            reorder_mapping = s32[n_active_tokens].Parameter(parameter_number=5)
        seq_slice_dim = 1 if cache_2d else 0

        return (*tensors, previous_cache_ids, reorder_mapping), (*dims, seq_slice_dim, seq_slice_dim)

    def embedding(self, input_ids, cache_ids, start_ids, last_token_id, *weights):
        if self.neuron_config.shard_over_sequence and self.neuron_config.on_device_embedding:
            *rst, embed_weight = weights
        else:
            embed_weight, *rst = weights
        dtype = getattr(input_ids.scribe, self.config.amp)
        if self.neuron_config.on_device_embedding and self.neuron_config.sequence_parallel_norm:
            hidden = hlo.embedding(embed_weight, input_ids, tp_degree=1, dtype=dtype)
        else:
            hidden = hlo.embedding(embed_weight, input_ids, tp_degree=self.config.tp_degree, dtype=dtype)
        if self.config.hidden_size % self.config.tp_degree != 0:
            hidden = hlo.slice_along(hidden, dim=-1, limit=self.config.hidden_size, start=0)
        if self.neuron_config.attention_layout == LAYOUT_HSB:
            hidden = hlo.transpose210(hidden)
        return hidden

    def token_tree_embedding(
        self, input_ids, cache_ids, start_ids, last_token_id, previous_cache_ids, reorder_mapping, *weights
    ):
        return self.embedding(input_ids, cache_ids, start_ids, last_token_id, *weights)

    def pre_layer(self, hidden, cache_ids, start_ids, last_token_id, *weights):
        # TODO: move this fallback calculation to decoder.py
        if self.num_active_blocks is None and self.neuron_config.optimized_paged_attention:
            max_model_len = self.neuron_config.continuous_batching.max_model_len
            max_num_seqs = self.neuron_config.continuous_batching.max_num_seqs
            block_size = self.neuron_config.continuous_batching.block_size
            self.num_active_blocks = (max_model_len * max_num_seqs // block_size) - 2

        if self.neuron_config.optimized_paged_attention and len(last_token_id.sizes) == 2:
            # For decoding with multiple KV cache blocks:
            # - cache_ids are used as context_lens
            # - start_ids are used as slot_mapping
            # - last_token_id is used as block_tables
            # The function below transforms 2D block_tables into 1D active block table
            last_token_id = attention_utils.active_block_tables(
                block_tables=last_token_id,
                context_lens=cache_ids,
                num_active_blocks=self.num_active_blocks,
                neuron_config=self.neuron_config,
            )
            max_num_seqs = self.neuron_config.continuous_batching.max_num_seqs
            block_size = self.neuron_config.continuous_batching.block_size
            block_to_seq = attention_utils.block_to_seq_indexing(
                context_lens=cache_ids, num_seqs=max_num_seqs, num_blocks=self.num_active_blocks, block_size=block_size
            )
        else:
            block_to_seq = None

        # Granite specific: embeddings are multiplied by embedding_multiplier
        hidden = scale_mul(hidden, self.config.embedding_multiplier)

        head_dim = self.config.attention_head_size
        pos_embed = rotary.hlo_rotary_embedding(
            hidden.dtype,
            int(head_dim * self.config.rotary_percentage),
            cache_ids,
            base=self.config.rope_theta,
            interpolation_factor=self.config.position_interpolation_factor,
            rope_scaling=self.config.rope_scaling,
        )
        core_id = None

        # flash decoding
        if self.neuron_config.shard_over_sequence:
            core_id, *rst = weights
            n_kv_heads = (
                self.config.num_key_value_heads
                if hasattr(self.config, "num_key_value_heads")
                else self.config.num_attention_heads
            )
            cores_per_kv_head = self.config.tp_degree // n_kv_heads
            self.cores_per_kv_head = cores_per_kv_head if cores_per_kv_head > 1 else self.config.tp_degree
            cache_ids, mask, active_mask = flash_decoding.convert_attn_mask_and_cache_id(
                cache_ids, start_ids, core_id, self.n_positions, cores_per_kv_head=self.cores_per_kv_head
            )
        else:
            mask, active_mask = hlo.attention_mask(
                cache_ids,
                start_ids,
                self.n_positions,
                last_token_id=last_token_id,
                num_active_blocks=self.num_active_blocks,
                neuron_config=self.neuron_config,
            )

        return hidden, last_token_id, pos_embed, cache_ids, start_ids, block_to_seq, mask, active_mask, core_id

    def token_tree_pre_layer(
        self, hidden, cache_ids, start_ids, last_token_id, previous_cache_ids, reorder_mapping, *weights
    ):
        hidden, last_token_id, pos_embed, cache_ids, start_ids, block_to_seq, mask, active_mask, core_id = (
            self.pre_layer(hidden, cache_ids, start_ids, last_token_id, *weights)
        )
        if self.neuron_config.on_device_embedding:
            embed_weight, token_tree_mask = weights
        else:
            token_tree_mask, *rst = weights
        active_mask = hlo.token_tree_attention_mask(token_tree_mask, active_mask)
        return (
            hidden,
            last_token_id,
            pos_embed,
            cache_ids,
            start_ids,
            block_to_seq,
            previous_cache_ids,
            reorder_mapping,
            mask,
            active_mask,
            core_id,
        )

    def layer(
        self,
        hidden,
        last_token_id,
        pos_embed,
        cache_ids,
        start_ids,
        block_to_seq,
        mask,
        active_mask,
        core_id,
        attn_k_cache,
        attn_v_cache,
        pre_attn_ln_weight,
        pre_attn_ln_bias,
        fused_pre_attn_ln_qkv_weight,
        attn_q_weight,
        attn_q_scales,
        attn_q_bias,
        attn_k_weight,
        attn_k_scales,
        attn_k_bias,
        attn_v_weight,
        attn_v_scales,
        attn_v_bias,
        attn_out_weight,
        attn_out_scales,
        attn_out_bias,
        post_attn_ln_weight,
        post_attn_ln_bias,
        pre_mlp_ln_weight,
        pre_mlp_ln_bias,
        mlp_in_weight,
        mlp_in_scales,
        mlp_in_bias,
        mlp_out_weight,
        mlp_out_scales,
        mlp_out_bias,
        post_mlp_ln_weight,
        post_mlp_ln_bias,
        in0_weight=None,
        in0_scales=None,
        in1_weight=None,
        in1_scales=None,
        out_weight=None,
        out_scales=None,
    ):
        eps = self.config.rms_norm_eps
        is_bsh = self.neuron_config and self.neuron_config.attention_layout == LAYOUT_BSH
        if self.neuron_config and self.neuron_config.fused_rmsnorm_qkv and active_mask is None:
            assert fused_pre_attn_ln_qkv_weight is not None
            attn_output, out_attn_k_cache, out_attn_v_cache = self.fused_rmsnorm_qkv(
                hidden,
                None,
                eps,
                cache_ids,
                start_ids,
                last_token_id,
                block_to_seq,
                pos_embed,
                mask,
                active_mask,
                core_id,
                attn_k_cache,
                attn_v_cache,
                fused_pre_attn_ln_qkv_weight,
                attn_q_scales,
                attn_q_bias,
                attn_k_weight,
                attn_k_scales,
                attn_k_bias,  # should be none
                attn_v_weight,
                attn_v_scales,
                attn_v_bias,  # should be none
                attn_out_weight,
                attn_out_scales,
                attn_out_bias,
            )
        else:
            ln_hidden = (
                hlo.rms_norm(
                    hidden, pre_attn_ln_weight, eps, neuron_config=self.neuron_config, tp_degree=self.config.tp_degree
                )
                if is_bsh
                else hlo.rms_norm(
                    hidden,
                    pre_attn_ln_weight,
                    eps,
                    dim=0,
                    neuron_config=self.neuron_config,
                    tp_degree=self.config.tp_degree,
                )
            )
            attn_output, out_attn_k_cache, out_attn_v_cache = self.attention(
                ln_hidden,
                cache_ids,
                start_ids,
                last_token_id,
                block_to_seq,
                pos_embed,
                mask,
                active_mask,
                core_id,
                attn_k_cache,
                attn_v_cache,
                attn_q_weight,
                attn_q_scales,
                attn_q_bias,
                attn_k_weight,
                attn_k_scales,
                attn_k_bias,
                attn_v_weight,
                attn_v_scales,
                attn_v_bias,
                attn_out_weight,
                attn_out_scales,
                attn_out_bias,
            )
        # Granite specific: attention output is multiplied by residual multiplier
        attn_output = scale_mul(attn_output, self.config.residual_multiplier)
        hidden = hlo.add(attn_output, hidden)
        gated_mlp = hlo.gated_mlp_bsh if is_bsh else hlo.gated_mlp
        rms_norm_dim = 2 if is_bsh else 0
        norm_hidden = hlo.rms_norm(
            hidden,
            pre_mlp_ln_weight,
            eps,
            dim=rms_norm_dim,
            neuron_config=self.neuron_config,
            tp_degree=self.config.tp_degree,
        )
        if self.neuron_config.fuse_mlp:
            assert all(
                map(lambda x: not (x), [in0_weight, in1_weight, out_weight, in0_scales, in1_scales, out_scales])
            ), "in0, in1 and out weights have to be None"
            in0_weight, in0_scales = mlp_in_weight, mlp_in_scales
            out_weight, out_scales = mlp_out_weight, mlp_out_scales

        mlp_hidden = gated_mlp(
            norm_hidden,
            in0_weight,
            in1_weight,
            out_weight,
            in0_scales=in0_scales,
            in1_scales=in1_scales,
            out_scales=out_scales,
            activation_function="silu",
            tp_degree=self.config.tp_degree,
            neuron_config=self.neuron_config,
        )
        # Granite specific: MLP output is multiplied by residual_multiplier
        mlp_hidden = scale_mul(mlp_hidden, self.config.residual_multiplier)
        res_hidden = hlo.add(mlp_hidden, hidden)
        return res_hidden, out_attn_k_cache, out_attn_v_cache

    def token_tree_layer(
        self,
        hidden,
        last_token_id,
        pos_embed,
        cache_ids,
        start_ids,
        block_to_seq,
        previous_cache_ids,
        reorder_mapping,
        mask,
        active_mask,
        core_id,
        attn_k_cache,
        attn_v_cache,
        pre_attn_ln_weight,
        pre_attn_ln_bias,
        fused_pre_attn_ln_qkv_weight,
        attn_q_weight,
        attn_q_scales,
        attn_q_bias,
        attn_k_weight,
        attn_k_scales,
        attn_k_bias,
        attn_v_weight,
        attn_v_scales,
        attn_v_bias,
        attn_out_weight,
        attn_out_scales,
        attn_out_bias,
        post_attn_ln_weight,
        post_attn_ln_bias,
        pre_mlp_ln_weight,
        pre_mlp_ln_bias,
        mlp_in_weight,
        mlp_in_scales,
        mlp_in_bias,
        mlp_out_weight,
        mlp_out_scales,
        mlp_out_bias,
        post_mlp_ln_weight,
        post_mlp_ln_bias,
        in0_weight,
        in0_scales,
        in1_weight,
        in1_scales,
        out_weight,
        out_scales,
    ):
        eps = self.config.rms_norm_eps
        is_bsh = self.neuron_config and self.neuron_config.attention_layout == LAYOUT_BSH
        ln_hidden = (
            hlo.rms_norm(
                hidden, pre_attn_ln_weight, eps, neuron_config=self.neuron_config, tp_degree=self.config.tp_degree
            )
            if is_bsh
            else hlo.rms_norm(
                hidden,
                pre_attn_ln_weight,
                eps,
                dim=0,
                neuron_config=self.neuron_config,
                tp_degree=self.config.tp_degree,
            )
        )
        reordered_attn_k_cache, reordered_attn_v_cache = attention.reorder_kv_cache(
            attn_k_cache, attn_v_cache, previous_cache_ids, reorder_mapping, neuron_config=self.neuron_config
        )
        attn_output, out_attn_k_cache, out_attn_v_cache = self.attention(
            ln_hidden,
            cache_ids,
            start_ids,
            last_token_id,
            block_to_seq,
            pos_embed,
            mask,
            active_mask,
            core_id,
            reordered_attn_k_cache,
            reordered_attn_v_cache,
            attn_q_weight,
            attn_q_scales,
            attn_q_bias,
            attn_k_weight,
            attn_k_scales,
            attn_k_bias,
            attn_v_weight,
            attn_v_scales,
            attn_v_bias,
            attn_out_weight,
            attn_out_scales,
            attn_out_bias,
        )
        hidden = hlo.add(attn_output, hidden)
        gated_mlp = hlo.gated_mlp_bsh if is_bsh else hlo.gated_mlp
        rms_norm_dim = 2 if is_bsh else 0
        norm_hidden = hlo.rms_norm(
            hidden,
            pre_mlp_ln_weight,
            eps,
            dim=rms_norm_dim,
            neuron_config=self.neuron_config,
            tp_degree=self.config.tp_degree,
        )
        mlp_hidden = gated_mlp(
            norm_hidden,
            in0_weight,
            in1_weight,
            out_weight,
            in0_scales=in0_scales,
            in1_scales=in1_scales,
            out_scales=out_scales,
            activation_function="silu",
            tp_degree=self.config.tp_degree,
            neuron_config=self.neuron_config,
        )
        res_hidden = hlo.add(mlp_hidden, hidden)
        return res_hidden, out_attn_k_cache, out_attn_v_cache

    def ln_lm_head(
        self, hidden, last_token_id, rms_weight, unused_bias, lm_head_weight, lm_head_bias, return_all_outputs=True
    ):
        logits = transformer.rms_lm_head(
            self.config.tp_degree,
            hidden,
            last_token_id,
            rms_weight,
            lm_head_weight,
            lm_head_bias,
            return_all_outputs,
            eps=self.config.rms_norm_eps,
            neuron_config=self.neuron_config,
        )
        return logits

    def fused_rmsnorm_qkv(
        self,
        hidden,
        pre_attn_ln_weight,
        eps,
        cache_ids,
        start_ids,
        last_token_id,
        block_to_seq,
        pos_embed,
        mask,
        active_mask,
        core_id,
        attn_k_cache,
        attn_v_cache,
        attn_q_weight,
        attn_q_scales,
        attn_q_bias,
        attn_k_weight,
        attn_k_scales,
        attn_k_bias,  # should be none
        attn_v_weight,
        attn_v_scales,
        attn_v_bias,  # should be none
        attn_out_weight,
        attn_out_scales,
        attn_out_bias,
    ):
        # TODO: refactor below
        from neuronxcc.nki._private_kernels.fused_linear import fused_rms_norm_qkv

        def _kernel(h, w, output):
            return fused_rms_norm_qkv(h, w, output, eps=eps)

        n_seqs, n_active_tokens, _ = hidden.sizes
        d_head = self.config.attention_head_size
        tp_degree = self.config.tp_degree

        # Compute the expected number of KV heads (Used in case fused QKV is used)
        n_kv_heads_tp = None
        if self.config.num_key_value_heads is not None:
            n_head = self.config.num_attention_heads
            n_kv_head = self.config.num_key_value_heads
            n_head, n_kv_head_padded = utils.get_qkv_padding(n_head, n_kv_head, tp_degree, self.neuron_config)
            n_kv_heads_tp = n_kv_head_padded // tp_degree

        _, hidden_size_tp = attn_q_weight.sizes

        n_total_heads_tp = hidden_size_tp // d_head
        n_heads_tp = n_total_heads_tp - 2 * n_kv_heads_tp
        # Q hidden size
        hidden_size_tp = d_head * n_heads_tp

        nki_output = nki_call(
            _kernel,
            hidden,
            attn_q_weight,
            output_HloShapes=[hidden.dtype[hidden.sizes[0], hidden.sizes[1], attn_q_weight.sizes[-1]]],
        )
        slice_lim = nki_output.sizes[-1] // (n_heads_tp + 2 * n_kv_heads_tp)
        query = hlo.slice_along(nki_output, -1, n_heads_tp * slice_lim, start=0)
        key = hlo.slice_along(nki_output, -1, (n_heads_tp + n_kv_heads_tp) * slice_lim, start=n_heads_tp * slice_lim)
        value = hlo.slice_along(
            nki_output,
            -1,
            (n_heads_tp + 2 * n_kv_heads_tp) * slice_lim,
            start=(n_heads_tp + n_kv_heads_tp) * slice_lim,
        )

        # shard over head (llama/hlo.py)
        active_q_sizes = n_active_tokens, n_seqs, n_heads_tp, d_head
        active_kv_sizes = n_active_tokens, n_seqs, n_kv_heads_tp, d_head
        query = hlo.reshape(query, active_q_sizes)
        key = hlo.reshape(key, active_kv_sizes)
        value = hlo.reshape(value, active_kv_sizes)
        assert all(
            [
                attn_q_scales is None,
                attn_q_bias is None,
                attn_k_weight is None,
                attn_k_scales is None,
                attn_k_bias is None,
                attn_v_weight is None,
                attn_v_scales is None,
                attn_v_bias is None,
            ]
        )

        # Pass QKV tuple since it will not be computed in the attention block
        attn_output, out_attn_k_cache, out_attn_v_cache = self.attention(
            nki_output,
            cache_ids,
            start_ids,
            last_token_id,
            block_to_seq,
            pos_embed,
            mask,
            active_mask,
            core_id,
            attn_k_cache,
            attn_v_cache,
            attn_q_weight,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            attn_out_weight,
            attn_out_scales,
            attn_out_bias,
            qkv_tuple=(query, key, value),
        )
        return attn_output, out_attn_k_cache, out_attn_v_cache

    def attention(
        self,
        hidden,
        cache_ids,
        start_ids,
        last_token_id,
        block_to_seq,
        pos_embed,
        mask,
        active_mask,
        core_id,
        cached_keys,
        cached_values,
        q_weight,
        q_scales,
        q_bias,
        k_weight,
        k_scales,
        k_bias,
        v_weight,
        v_scales,
        v_bias,
        out_weight,
        out_scales,
        out_bias,
        qkv_tuple: tuple = None,
    ):
        d_head = self.config.attention_head_size
        tp_degree = self.config.tp_degree

        # Compute the expected number of KV heads (Used in case fused QKV is used)
        n_kv_heads_tp = None
        if self.config.num_key_value_heads is not None:
            n_head = self.config.num_attention_heads
            n_kv_head = self.config.num_key_value_heads
            n_head, n_kv_head_padded = utils.get_qkv_padding(n_head, n_kv_head, tp_degree, self.neuron_config)
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
                q_scales,
                q_bias,
                k_weight,
                k_scales,
                k_bias,
                v_weight,
                v_scales,
                v_bias,
                d_head,
                neuron_config=self.neuron_config,
                tp_degree=tp_degree,  # TODO: include tp_degree into neuron_config
                shard_over_batch=self.shard_over_batch,
                n_kv_heads_tp=n_kv_heads_tp,
            )

        # Q = Rotate(Q)
        # K = Rotate(K)
        query, key = rotary.rotate_half(
            query,
            key,
            pos_embed,
            self.config.rotary_percentage,
            tp_degree=tp_degree,
            shard_over_batch=self.shard_over_batch,
        )

        # Granite specific: instead of dividing the QK product, multiply it by the attention_multiplier
        query = scale_mul(query, self.config.attention_multiplier)

        # In BSH cache layout, the output of QKV linear projection is still kept as SBH for all QKV.
        bsh_cache_layout = False
        batch_dim = 1
        if self.neuron_config is not None:
            bsh_cache_layout = self.neuron_config.cache_layout == constants.LAYOUT_BSH
        if bsh_cache_layout:
            query, key, value = attention_utils.transpose_qkv(query, key, value)
            batch_dim = 0

        # Single Token Generation ("Prefetch"-style) ans speculative forward
        if active_mask is not None:

            n_active_tokens = key.sizes[1] if bsh_cache_layout else key.sizes[0]
            if n_active_tokens > 1 and self.neuron_config and self.neuron_config.continuous_batching:
                # For speculative forward + continuous batching, slice out samples in the batch size
                # corresponding to the batch size of the speculative head
                slice_sizes = [1] * len(cached_keys.sizes)
                if cached_keys.sizes[batch_dim] == 1:
                    # Use hlo.select for batch size 1 as index select is prohibitively slow
                    # TODO: revert to hlo.index_select once its faster P126527643
                    cached_keys_s = hlo.select(
                        cached_keys, batch_dim, hlo.reshape(start_ids, slice_sizes), keepdim=True
                    )
                    cached_values_s = hlo.select(
                        cached_values, batch_dim, hlo.reshape(start_ids, slice_sizes), keepdim=True
                    )
                else:
                    cached_keys_s = hlo.index_select(cached_keys, batch_dim, start_ids)
                    cached_values_s = hlo.index_select(cached_values, batch_dim, start_ids)
                if self.neuron_config and self.neuron_config.kv_cache_quant:
                    cached_keys_s = dequantize_kv_cache_direct_cast(cached_keys_s, self.neuron_config)
                    cached_values_s = dequantize_kv_cache_direct_cast(cached_values_s, self.neuron_config)
            elif self.neuron_config and self.neuron_config.paged_attention:
                # For decoding with multiple KV cache blocks, start_ids are used as block_tables
                cached_keys_s = attention_utils.gather_blocks(
                    cached_keys, block_tables=last_token_id, neuron_config=self.neuron_config
                )
                cached_values_s = attention_utils.gather_blocks(
                    cached_values, block_tables=last_token_id, neuron_config=self.neuron_config
                )
                if self.neuron_config and self.neuron_config.kv_cache_quant:
                    cached_keys_s = dequantize_kv_cache_direct_cast(cached_keys_s, self.neuron_config)
                    cached_values_s = dequantize_kv_cache_direct_cast(cached_values_s, self.neuron_config)
            elif self.neuron_config and self.neuron_config.kv_cache_quant:
                cached_keys_s = dequantize_kv_cache_direct_cast(cached_keys, self.neuron_config)
                cached_values_s = dequantize_kv_cache_direct_cast(cached_values, self.neuron_config)
            else:
                cached_keys_s = cached_keys
                cached_values_s = cached_values
            # Communication 1: all-gather query from cores
            if (n_active_tokens != self.n_positions) and self.neuron_config.shard_over_sequence:
                query = flash_decoding.gather_query_group(query, self.cores_per_kv_head, n_head, tp_degree)

            # Sp = Q @ Kp
            prior_scores = attention.score(
                query,
                cached_keys_s,
                n_kv_heads=self.config.num_key_value_heads,
                tp_degree=tp_degree,
                block_to_seq=block_to_seq,
                neuron_config=self.neuron_config,
            )
            prior_scores = attention.mask(
                prior_scores, mask, tp_degree=tp_degree, shard_over_batch=self.shard_over_batch
            )

            # Sa = Q @ Ka
            active_score = attention.score(
                query,
                key,
                n_kv_heads=self.config.num_key_value_heads,
                tp_degree=tp_degree,
                neuron_config=self.neuron_config,
            )
            active_score = attention.mask(
                active_score, active_mask, tp_degree=tp_degree, shard_over_batch=self.shard_over_batch
            )

            # C = softmax(Sa, Sp) @ (Va, Vp)
            if self.neuron_config.shard_over_sequence:
                dtype = query.dtype
                context = flash_decoding.context(
                    prior_scores,
                    active_score,
                    cached_values_s,
                    value,
                    core_id,
                    mask,
                    active_mask,
                    n_kv_heads=self.config.num_key_value_heads,
                    n_heads=n_head,
                    dtype=dtype,
                    tp_degree=tp_degree,
                    neuron_config=self.neuron_config,
                    shard_over_batch=self.shard_over_batch,
                )
                cache_ids, value, key = flash_decoding.select_values_within_bound(
                    cache_ids, value, key, self.cores_per_kv_head, core_id, dim=0
                )

            else:
                context = attention.context(
                    prior_scores,
                    active_score,
                    cached_values_s,
                    value,
                    n_kv_heads=self.config.num_key_value_heads,
                    tp_degree=tp_degree,
                    context_lens=cache_ids,
                    num_active_blocks=self.num_active_blocks,
                    block_to_seq=block_to_seq,
                    neuron_config=self.neuron_config,
                )

            # KCache[I], VCache[I] = K, V
            updated_keys, updated_values = attention.fused_kv_update_cache(
                cached_keys, cached_values, cache_ids, key, value, start_ids, neuron_config=self.neuron_config
            )

        # Multi-Token Context Encoding
        else:
            _, batch_size, _, _ = query.sizes
            if self.neuron_config.lhs_aligned or batch_size == 1:
                context = attention.flash_attention(query, key, value)
            else:
                # do not use flash attention for lhs padded (right aligned) batch > 1 case
                # because it does not correctly take mask into account
                context = None

            if context is None:
                # S = Q @ K

                score = attention.score(
                    query,
                    key,
                    n_kv_heads=self.config.num_key_value_heads,
                    tp_degree=tp_degree,
                    neuron_config=self.neuron_config,
                )
                score = attention.mask(score, mask, tp_degree=tp_degree, shard_over_batch=self.shard_over_batch)
                context = attention.context_combined(
                    score,
                    value,
                    n_kv_heads=self.config.num_key_value_heads,
                    tp_degree=tp_degree,
                    neuron_config=self.neuron_config,
                )

            if self.neuron_config.shard_over_sequence:
                cache_ids, value, key = flash_decoding.select_values_within_bound(
                    cache_ids, value, key, self.cores_per_kv_head, core_id, dim=0
                )
            # KCache, VCache = K, V
            if cached_keys.sizes == key.sizes:
                if self.neuron_config and self.neuron_config.kv_cache_quant:
                    updated_keys = quantize_kv_cache_direct_cast(key, self.neuron_config)
                    updated_values = quantize_kv_cache_direct_cast(value, self.neuron_config)
                else:
                    updated_keys, updated_values = key, value
            else:
                updated_keys, updated_values = attention.fused_kv_update_cache(
                    cached_keys, cached_values, cache_ids, key, value, start_ids, neuron_config=self.neuron_config
                )

        # O = (C @ wO) + bO
        output = attention.output(context, out_weight, out_scales, out_bias, tp_degree, self.neuron_config)
        return output, updated_keys, updated_values
