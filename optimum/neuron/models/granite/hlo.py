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

from transformers.models.granite import GraniteConfig
from transformers_neuronx import constants, hlo, utils
from transformers_neuronx.config import NeuronConfig
from transformers_neuronx.constants import LAYOUT_BSH, LAYOUT_HSB
from transformers_neuronx.layers import attention, rotary, transformer
from transformers_neuronx.nki.compile import nki_call

from optimum.utils import logging


logger = logging.get_logger()


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
            self.neuron_config.tp_degree,
        )

        return tensors, dims

    def embedding(self, input_ids, cache_ids, start_ids, last_token_id, block_tables, context_lens, *weights):
        embed_weight, *rst = weights
        dtype = getattr(input_ids.scribe, self.neuron_config.amp)
        hidden = hlo.embedding(embed_weight, input_ids, tp_degree=self.neuron_config.tp_degree, dtype=dtype)
        if self.config.hidden_size % self.neuron_config.tp_degree != 0:
            hidden = hlo.slice_along(hidden, dim=-1, limit=self.config.hidden_size, start=0)
        if self.neuron_config.attention_layout == LAYOUT_HSB:
            hidden = hlo.transpose210(hidden)
        return hidden

    def pre_layer(
        self, hidden, cache_ids, start_ids, last_token_id, block_tables, context_lens, *weights, position_ids=None
    ):
        block_to_seq = None
        cached_mask = None
        cached_to_contexted = None
        active_to_contexted = None
        core_id = None

        # Granite specific: embeddings are multiplied by embedding_multiplier
        hidden = scale_mul(hidden, self.config.embedding_multiplier)

        head_dim = self.config.hidden_size // self.config.num_attention_heads
        position_ids = cache_ids if position_ids is None else position_ids
        pos_embed = rotary.hlo_rotary_embedding(
            hidden.dtype,
            head_dim,
            position_ids,
            base=self.config.rope_theta,
            rope_scaling=self.config.rope_scaling,
        )

        # flash decoding
        mask, active_mask = hlo.attention_mask(
            cache_ids,
            start_ids,
            self.n_positions,
            last_token_id=last_token_id,
            neuron_config=self.neuron_config,
            context_lens=context_lens,
        )

        return (
            hidden,
            last_token_id,
            pos_embed,
            cache_ids,
            start_ids,
            block_to_seq,
            mask,
            active_mask,
            core_id,
            block_tables,
            cached_mask,
            cached_to_contexted,
            active_to_contexted,
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
        block_tables,
        cached_mask,
        cached_to_contexted,
        active_to_contexted,
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
        is_first_last_layer=False,
    ):
        local_args = {**locals()}
        local_args.pop("self")

        # Initialize with kernels
        enable_qkv_kernel, enable_mlp_kernel = False, False
        if self.neuron_config and self.neuron_config.fused_rmsnorm_qkv:
            try:
                from neuronxcc.nki._private_kernels.qkv import rmsnorm_qkv_isa_fused_add_kernel  # noqa: F401

                enable_qkv_kernel = True
            except Exception:
                logger.warning("No QKV kernel found")
        if self.neuron_config and self.neuron_config.fused_rmsnorm_mlp:
            try:
                from neuronxcc.nki._private_kernels.mlp import mlp_isa_kernel  # noqa: F401

                enable_mlp_kernel = True
            except Exception:
                logger.warning("No MLP kernel found")
            enable_mlp_kernel = True

        if (not enable_qkv_kernel and not enable_mlp_kernel) or active_mask is not None:
            return self.flat_compiler_layer(**local_args)

        local_args["enable_qkv_kernel"] = enable_qkv_kernel
        local_args["enable_mlp_kernel"] = enable_mlp_kernel
        return self.native_kernel_layer(**local_args)

    def flat_compiler_layer(
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
        block_tables,
        cached_mask,
        cached_to_contexted,
        active_to_contexted,
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
        is_first_last_layer=False,
    ):
        eps = self.config.rms_norm_eps
        is_bsh = self.neuron_config and self.neuron_config.attention_layout == LAYOUT_BSH
        if self.neuron_config.has_pre_attention_norm:
            ln_hidden = (
                hlo.rms_norm(
                    hidden,
                    pre_attn_ln_weight,
                    eps,
                    neuron_config=self.neuron_config,
                    tp_degree=self.neuron_config.tp_degree,
                )
                if is_bsh
                else hlo.rms_norm(
                    hidden,
                    pre_attn_ln_weight,
                    eps,
                    dim=0,
                    neuron_config=self.neuron_config,
                    tp_degree=self.neuron_config.tp_degree,
                )
            )
        else:
            ln_hidden = hidden
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
            block_tables,
            cached_mask,
            cached_to_contexted,
            active_to_contexted,
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
            tp_degree=self.neuron_config.tp_degree,
        )
        if self.neuron_config.fuse_mlp:
            assert all(
                (not (x) for x in [in0_weight, in1_weight, out_weight])
            ), "in0, in1 and out weights have to be None"
            in0_weight = mlp_in_weight
            out_weight = mlp_out_weight

        mlp_hidden = gated_mlp(
            norm_hidden,
            in0_weight,
            in1_weight,
            out_weight,
            activation_function="silu",
            tp_degree=self.neuron_config.tp_degree,
            neuron_config=self.neuron_config,
        )
        # Granite specific: MLP output is multiplied by residual_multiplier
        mlp_hidden = scale_mul(mlp_hidden, self.config.residual_multiplier)
        res_hidden = hlo.add(mlp_hidden, hidden)
        return res_hidden, out_attn_k_cache, out_attn_v_cache

    def native_kernel_layer(
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
        block_tables,
        cached_mask,
        cached_to_contexted,
        active_to_contexted,
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
        is_first_last_layer=False,
        enable_qkv_kernel=False,
        enable_mlp_kernel=False,
    ):
        eps = self.config.rms_norm_eps
        is_bsh = self.neuron_config and self.neuron_config.attention_layout == LAYOUT_BSH
        assert is_bsh
        rms_norm_dim = 2 if is_bsh else 0

        from neuronxcc.nki._private_kernels.mlp import mlp_fused_add_isa_kernel, mlp_isa_kernel

        # lambda functions for calling kernels
        def _mlp_fused_add_kernel(attn_output, hidden, ln_w, gate_w, up_w, down_w, out, fused_rmsnorm=True):
            mlp_fused_add_isa_kernel(
                attn_output, hidden, ln_w, gate_w, up_w, down_w, out, "MLP", fused_rmsnorm=fused_rmsnorm
            )

        def _mlp_kernel(hidden, ln_w, gate_w, up_w, down_w, out, fused_rmsnorm=False):
            mlp_isa_kernel(hidden, ln_w, gate_w, up_w, down_w, out, "MLP", fused_rmsnorm=fused_rmsnorm)

        if enable_qkv_kernel:
            fused_out = self.fused_rmsnorm_qkv(
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
                block_tables,
                cached_mask,
                cached_to_contexted,
                active_to_contexted,
                attn_k_cache,
                attn_v_cache,
                attn_q_weight,
                attn_q_bias,
                attn_k_weight,
                attn_k_bias,  # should be none
                attn_v_weight,
                attn_v_bias,  # should be none
                attn_out_weight,
                attn_out_bias,
            )
            if len(fused_out) == 3:
                attn_output, out_attn_k_cache, out_attn_v_cache = fused_out
            else:
                attn_output, out_attn_k_cache, out_attn_v_cache, fused_added_hidden = fused_out
        else:
            ln_hidden = (
                hlo.rms_norm(
                    hidden,
                    pre_attn_ln_weight,
                    eps,
                    neuron_config=self.neuron_config,
                    tp_degree=self.neuron_config.tp_degree,
                )
                if is_bsh
                else hlo.rms_norm(
                    hidden,
                    pre_attn_ln_weight,
                    eps,
                    dim=0,
                    neuron_config=self.neuron_config,
                    tp_degree=self.neuron_config.tp_degree,
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
                block_tables,
                cached_mask,
                cached_to_contexted,
                active_to_contexted,
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

        if isinstance(hidden, tuple):
            hidden = hidden[0]

        if enable_mlp_kernel:
            # In TP, we can fuse residual add and rms norm into the kernel
            if is_first_last_layer or not enable_qkv_kernel:
                hidden_add = hlo.add(attn_output, hidden)
            mlp_result = nki_call(
                _mlp_fused_add_kernel,
                attn_output,
                hidden,
                pre_mlp_ln_weight,
                in0_weight,
                in1_weight,
                out_weight,
                output_HloShapes=[hidden.dtype[hidden.sizes[0], hidden.sizes[1], hidden.sizes[2]]],
            )
            dtype, replica_groups = utils.parse_dtype_replica_groups(self.neuron_config, self.neuron_config.tp_degree)
            mlp_hidden = hlo.all_reduce_sum(
                mlp_result, self.neuron_config.tp_degree, dtype=dtype, replica_groups=replica_groups
            )
            if is_first_last_layer or not enable_qkv_kernel:
                return hlo.add(mlp_hidden, hidden_add), out_attn_k_cache, out_attn_v_cache

            return (hidden, mlp_hidden, attn_output), out_attn_k_cache, out_attn_v_cache
        else:
            hidden = hlo.add(attn_output, hidden)
            gated_mlp = hlo.gated_mlp_bsh if is_bsh else hlo.gated_mlp
            norm_hidden = hlo.rms_norm(
                hidden,
                pre_mlp_ln_weight,
                eps,
                dim=rms_norm_dim,
                neuron_config=self.neuron_config,
                tp_degree=self.neuron_config.tp_degree,
            )
            mlp_hidden = gated_mlp(
                norm_hidden,
                in0_weight,
                in1_weight,
                out_weight,
                activation_function="silu",
                tp_degree=self.neuron_config.tp_degree,
                neuron_config=self.neuron_config,
            )
            if is_first_last_layer or not enable_qkv_kernel:
                return hlo.add(mlp_hidden, hidden), out_attn_k_cache, out_attn_v_cache
            return (hidden, mlp_hidden, attn_output), out_attn_k_cache, out_attn_v_cache

    def ln_lm_head(
        self, hidden, last_token_id, rms_weight, unused_bias, lm_head_weight, lm_head_bias, is_prefill=True
    ):
        logits = transformer.rms_lm_head(
            self.neuron_config.tp_degree,
            hidden,
            last_token_id,
            rms_weight,
            lm_head_weight,
            lm_head_bias,
            is_prefill,
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
        block_tables,
        cached_mask,
        cached_to_contexted,
        active_to_contexted,
        attn_k_cache,
        attn_v_cache,
        attn_q_weight,
        attn_q_bias,
        attn_k_weight,
        attn_k_bias,  # should be none
        attn_v_weight,
        attn_v_bias,  # should be none
        attn_out_weight,
        attn_out_bias,
    ):
        from neuronxcc.nki._private_kernels.qkv import rmsnorm_qkv_isa_fused_add_kernel, rmsnorm_qkv_isa_kernel

        def _kernel(h, w, ln_w, output):
            return rmsnorm_qkv_isa_kernel(h, w, ln_w, output, "QKV")

        def _fused_out_kernel(h0, h1, h2, w, ln_w, output):
            # This kernel will perform h0 = h0 + h1 + h2 (writing results in-place to an input buffer
            # FIXME: allow for multiple outputs
            return rmsnorm_qkv_isa_fused_add_kernel(h0, h1, h2, w, ln_w, output, "QKV")

        fused_add = False
        if isinstance(hidden, tuple):
            fused_add = True
            hidden, mlp_out, attn_out = hidden

        n_seqs, n_active_tokens, _ = hidden.sizes
        d_head = self.config.hidden_size // self.config.num_attention_heads
        tp_degree = self.neuron_config.tp_degree

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

        if fused_add:
            nki_output = nki_call(
                _fused_out_kernel,
                hidden,
                mlp_out,
                attn_out,
                attn_q_weight,
                pre_attn_ln_weight,
                output_HloShapes=[hidden.dtype[n_seqs, n_active_tokens, hidden_size_tp]],
            )
        else:
            nki_output = nki_call(
                _kernel,
                hidden,
                attn_q_weight,
                pre_attn_ln_weight,
                output_HloShapes=[hidden.dtype[n_seqs, n_active_tokens, hidden_size_tp]],
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
                attn_q_bias is None,
                attn_k_weight is None,
                attn_k_bias is None,
                attn_v_weight is None,
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
            block_tables,
            cached_mask,
            cached_to_contexted,
            active_to_contexted,
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
            attn_out_bias,
            qkv_tuple=(query, key, value),
        )
        if fused_add:
            return attn_output, out_attn_k_cache, out_attn_v_cache, hidden
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
        block_tables,
        cached_mask,
        cached_to_contexted,
        active_to_contexted,
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
            n_head_padded, n_kv_head_padded = utils.get_qkv_padding(n_head, n_kv_head, tp_degree, self.neuron_config)
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
            tp_degree=tp_degree,
            shard_over_batch=self.shard_over_batch,
        )

        # Granite specific: instead of dividing the QK product, multiply it by the attention_multiplier
        query = scale_mul(query, self.config.attention_multiplier)

        batch_dim = 1
        # Single Token Generation ("Prefetch"-style) ans speculative forward
        if active_mask is not None:

            n_active_tokens = key.sizes[0]
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
                elif cached_keys.sizes[batch_dim] == start_ids.sizes[0]:
                    # For batched speculative decoding, we will select kv caches for all sequences. No need to do
                    # index select, which is slow
                    cached_keys_s = cached_keys
                    cached_values_s = cached_values
                else:
                    # for multi prompt use case, cached_keys.sizes[batch_dim] can still be larger than 1, so we
                    # need to use start_ids size to determine if we want to select kv cache.
                    cached_keys_s = hlo.index_select(cached_keys, batch_dim, start_ids)
                    cached_values_s = hlo.index_select(cached_values, batch_dim, start_ids)
            else:
                cached_keys_s = cached_keys
                cached_values_s = cached_values

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
            context = attention.context(
                prior_scores,
                active_score,
                cached_values_s,
                value,
                n_kv_heads=self.config.num_key_value_heads,
                tp_degree=tp_degree,
                neuron_config=self.neuron_config,
            )

            # KCache[I], VCache[I] = K, V
            updated_keys, updated_values = attention.fused_kv_update_cache(
                cached_keys, cached_values, cache_ids, key, value, start_ids, neuron_config=self.neuron_config
            )

        # Multi-Token Context Encoding
        else:
            batch_size = query.sizes[batch_dim]
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

            # KCache, VCache = K, V
            if cached_keys.sizes == key.sizes:
                updated_keys, updated_values = key, value
            else:
                updated_keys, updated_values = attention.fused_kv_update_cache(
                    cached_keys, cached_values, cache_ids, key, value, start_ids, neuron_config=self.neuron_config
                )

        # O = (C @ wO) + bO
        output = attention.output(context, out_weight, out_bias, tp_degree, self.neuron_config)
        return output, updated_keys, updated_values
