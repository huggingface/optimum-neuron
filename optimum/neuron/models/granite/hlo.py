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
from transformers_neuronx.layers import attention, attention_utils, rotary, transformer
from transformers_neuronx.nki.compile import nki_call

from optimum.utils import logging

from .config import GraniteConfig


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

    def eagle_draft_inputs(
        self,
        scribe,
        dtype,
        n_active_tokens,
        batch_size,
        token_tree=False,
        k=0,
        n_leaves=0,
        depth=0,
        n_entrees=0,
        width=0,
    ):
        tensors, dims = self.inputs(scribe, dtype, n_active_tokens, batch_size)
        hidden_sizes = batch_size, n_active_tokens, self.config.hidden_size
        prev_hidden = dtype[hidden_sizes].Parameter(parameter_number=6)
        if not token_tree:
            return (*tensors, prev_hidden), (*dims, 1)
        s32 = scribe.s32
        tree_mask_sizes = k, k
        tree_mask = s32[tree_mask_sizes].Parameter(parameter_number=7)
        indices_sizes = batch_size, k - 1
        update_indices = s32[indices_sizes].Parameter(parameter_number=8)
        hidden_update_sizes = batch_size, k - 1
        hidden_update_indices = s32[hidden_update_sizes].Parameter(parameter_number=9)
        cache_update_sizes = batch_size, depth
        cache_gather_indices = s32[cache_update_sizes].Parameter(parameter_number=10)
        cache_scatter_indices = s32[cache_update_sizes].Parameter(parameter_number=11)
        pos_sizes = batch_size, k
        position_ids = s32[pos_sizes].Parameter(parameter_number=12)
        path_sizes = n_leaves, depth
        all_paths = s32[path_sizes].Parameter(parameter_number=13)
        return (
            *tensors,
            prev_hidden,
            tree_mask,
            update_indices,
            hidden_update_indices,
            cache_gather_indices,
            cache_scatter_indices,
            position_ids,
            all_paths,
        ), (*dims, 1, 1, 1, 1, 1, 1, 1, 1)

    def embedding(self, input_ids, cache_ids, start_ids, last_token_id, block_tables, context_lens, *weights):
        core_id = None
        embed_weight, *rst = weights
        dtype = getattr(input_ids.scribe, self.config.amp)
        if self.neuron_config.on_device_embedding and self.neuron_config.sequence_parallel_norm:
            hidden = hlo.embedding(
                embed_weight,
                input_ids,
                tp_degree=self.config.tp_degree,
                dim=0,
                dtype=dtype,
                core_id=core_id,
                sequence_parallel=self.neuron_config.is_sequence_parallel,
            )
        else:
            hidden = hlo.embedding(embed_weight, input_ids, tp_degree=self.config.tp_degree, dtype=dtype)
        if self.config.hidden_size % self.config.tp_degree != 0:
            hidden = hlo.slice_along(hidden, dim=-1, limit=self.config.hidden_size, start=0)
        if self.neuron_config.attention_layout == LAYOUT_HSB:
            hidden = hlo.transpose210(hidden)
        return hidden

    def pre_layer(
        self, hidden, cache_ids, start_ids, last_token_id, block_tables, context_lens, *weights, position_ids=None
    ):
        # TODO: move this fallback calculation to decoder.py
        if self.num_active_blocks is None and self.neuron_config.optimized_paged_attention:
            max_model_len = self.neuron_config.continuous_batching.max_model_len
            max_num_seqs = self.neuron_config.continuous_batching.max_num_seqs
            block_size = self.neuron_config.continuous_batching.block_size
            self.num_active_blocks = (max_model_len * max_num_seqs // block_size) - 2

        block_to_seq = None
        cached_mask = None
        cached_to_contexted = None
        active_to_contexted = None
        core_id = None
        if self.neuron_config.shard_over_sequence or (
            self.neuron_config.sequence_parallel_norm and self.neuron_config.on_device_embedding
        ):
            core_id, *rst = weights
        if self.neuron_config.shard_over_sequence:
            n_kv_heads = (
                self.config.num_key_value_heads
                if hasattr(self.config, "num_key_value_heads")
                else self.config.num_attention_heads
            )
            cores_per_kv_head = self.config.tp_degree // n_kv_heads
            self.cores_per_kv_head = cores_per_kv_head if cores_per_kv_head > 1 else self.config.tp_degree
            cores_per_q_head = self.config.tp_degree // self.config.num_attention_heads
            self.cores_per_kv_head = (
                self.cores_per_kv_head // cores_per_q_head if cores_per_q_head else self.cores_per_kv_head
            )
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
        elif self.neuron_config.enable_chunked_prefill:
            # - cache_ids are used as position_ids of each token
            # - start_ids are used as slot_mapping
            # - last_token_id is used as new token length for each sequence
            context_lens_2d = hlo.unsqueeze(context_lens, 1)
            seq_lens = hlo.add(context_lens, last_token_id)
            block_size = self.neuron_config.continuous_batching.block_size
            if self.neuron_config.shard_over_sequence:
                core_sos_rank = hlo.remainder(core_id, cores_per_kv_head)
                core_sos_rank = hlo.cast(core_sos_rank, seq_lens.scribe.s32)
                sharded_block_size = block_size // cores_per_kv_head
                block_tables = attention_utils.active_block_tables(
                    block_tables=block_tables,
                    context_lens=hlo.unsqueeze(seq_lens, 1),
                    num_active_blocks=self.num_active_blocks,
                    neuron_config=self.neuron_config,
                )
                start_ids, active_token_mask = attention_utils.sharded_slot_mapping(
                    start_ids, cache_ids, block_size, core_sos_rank, sos_degree=cores_per_kv_head
                )
                max_num_keys = (self.num_active_blocks + 1) * sharded_block_size
                _, n_active_tokens = cache_ids.sizes
                cached_to_contexted, cached_to_contexted_idx, active_to_contexted, sharded_seq_lens = (
                    attention_utils.sharded_kv_indexing(
                        seq_lens,
                        last_token_id,
                        cache_ids,
                        max_num_keys,
                        n_active_tokens,
                        block_size,
                        block_tables,
                        core_sos_rank,
                        active_token_mask,
                        sos_degree=cores_per_kv_head,
                    )
                )
            else:
                block_tables = attention_utils.active_block_tables(
                    block_tables=block_tables,
                    context_lens=context_lens_2d,
                    num_active_blocks=self.num_active_blocks,
                    neuron_config=self.neuron_config,
                )
                max_num_keys = self.num_active_blocks * block_size + self.n_positions
                cached_mask, cached_to_contexted, active_to_contexted = attention_utils.contexted_kv_indexing(
                    query_lens=last_token_id, key_lens=seq_lens, max_num_keys=max_num_keys, block_size=block_size
                )

        # Granite specific: embeddings are multiplied by embedding_multiplier
        hidden = scale_mul(hidden, self.config.embedding_multiplier)

        head_dim = self.config.attention_head_size
        position_ids = cache_ids if position_ids is None else position_ids
        pos_embed = rotary.hlo_rotary_embedding(
            hidden.dtype,
            int(head_dim * self.config.rotary_percentage),
            position_ids,
            base=self.config.rope_theta,
            interpolation_factor=self.config.position_interpolation_factor,
            rope_scaling=self.config.rope_scaling,
        )

        # flash decoding
        mask, active_mask = hlo.attention_mask(
            cache_ids,
            start_ids,
            self.n_positions,
            last_token_id=last_token_id,
            num_active_blocks=self.num_active_blocks,
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

    def eagle_draft_pre_layer(
        self, hidden, cache_ids, start_ids, last_token_id, block_tables, context_lens, *weights, position_ids=None
    ):

        if (
            self.neuron_config.shard_over_sequence or self.neuron_config.sequence_parallel_norm
        ) and self.neuron_config.on_device_embedding:
            core_id, embed_weight, *rst = weights
        else:
            embed_weight, *rst = weights

        if self.config.bias:
            fc_weight, fc_bias, *rst = rst
        else:
            fc_weight, *rst = rst
            fc_bias = None
        hidden = hlo.dot_add(fc_weight, hidden, fc_bias, 0, 2, 0)
        hidden = hlo.permute(hidden, [1, 2, 0])
        hidden = hlo.all_gather(hidden, 2, self.config.tp_degree)
        # hidden = hlo.dot_add(hidden, fc_weight, fc_bias, 2, 0, 2)
        return self.pre_layer(
            hidden,
            cache_ids,
            start_ids,
            last_token_id,
            block_tables,
            context_lens,
            *weights,
            position_ids=position_ids,
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
            tp_degree=self.config.tp_degree,
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
            tp_degree=self.config.tp_degree,
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
            if self.neuron_config.is_sequence_parallel:
                # In sequence parallel, we cannot fuse residual add and rms norm into the kernel
                hidden = hlo.add(attn_output, hidden)
                norm_hidden = hlo.rms_norm(
                    hidden,
                    pre_mlp_ln_weight,
                    eps,
                    dim=rms_norm_dim,
                    neuron_config=self.neuron_config,
                    tp_degree=self.config.tp_degree,
                )
                mlp_result = nki_call(
                    _mlp_kernel,
                    norm_hidden,
                    pre_mlp_ln_weight,
                    in0_weight,
                    in1_weight,
                    out_weight,
                    output_HloShapes=[
                        norm_hidden.dtype[norm_hidden.sizes[0], norm_hidden.sizes[1], norm_hidden.sizes[2]]
                    ],
                )
                dtype, replica_groups = utils.parse_dtype_replica_groups(self.neuron_config, self.config.tp_degree)
                mlp_hidden = hlo.reduce_scatter_sum(
                    mlp_result, tp_degree=self.config.tp_degree, dim=1, replica_groups=replica_groups, dtype=dtype
                )
                return hlo.add(mlp_hidden, hidden), out_attn_k_cache, out_attn_v_cache

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
            dtype, replica_groups = utils.parse_dtype_replica_groups(self.neuron_config, self.config.tp_degree)
            mlp_hidden = hlo.all_reduce_sum(
                mlp_result, self.config.tp_degree, dtype=dtype, replica_groups=replica_groups
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
                tp_degree=self.config.tp_degree,
            )
            mlp_hidden = gated_mlp(
                norm_hidden,
                in0_weight,
                in1_weight,
                out_weight,
                activation_function="silu",
                tp_degree=self.config.tp_degree,
                neuron_config=self.neuron_config,
            )
            if is_first_last_layer or not enable_qkv_kernel:
                return hlo.add(mlp_hidden, hidden), out_attn_k_cache, out_attn_v_cache
            return (hidden, mlp_hidden, attn_output), out_attn_k_cache, out_attn_v_cache

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
        d_head = self.config.attention_head_size
        tp_degree = self.config.tp_degree

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

        if (
            (active_mask is None and not self.neuron_config.enable_chunked_prefill)
            and self.neuron_config.shard_over_sequence
            and self.neuron_config.duplicate_q_weight_sos
        ):
            # slice on computed qeury when sos and duplicate Q weights is on

            # q / kv -> number of q per core after replication
            # core_id % tp/kv -> kv replication degree on cores
            # q / tp -> actual q per core before replication
            slice_start = hlo.remainder(
                hlo.reshape(core_id, []), core_id.dtype.Constant(constant_value=self.neuron_config.kv_replication)
            )
            slice_size = self.neuron_config.n_head_padded // tp_degree

            slice_start = hlo.multiply(slice_start, slice_start.dtype.Constant(constant_value=slice_size))

            query = hlo.dynamic_slice_along(query, 2, start=slice_start, size=slice_size)

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
            elif self.neuron_config and self.neuron_config.paged_attention:
                # For decoding with multiple KV cache blocks, start_ids are used as block_tables
                cached_keys_s = attention_utils.gather_blocks(
                    cached_keys, block_tables=last_token_id, neuron_config=self.neuron_config
                )
                cached_values_s = attention_utils.gather_blocks(
                    cached_values, block_tables=last_token_id, neuron_config=self.neuron_config
                )
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
            batch_size = query.sizes[batch_dim]
            if (
                (self.neuron_config.lhs_aligned or batch_size == 1)
                and not self.neuron_config.enable_chunked_prefill
                and not self.neuron_config.bsh_cache_layout
            ):
                context = attention.flash_attention(query, key, value)
            else:
                # do not use flash attention for lhs padded (right aligned) batch > 1 case
                # because it does not correctly take mask into account
                context = None

            if context is None:
                if self.neuron_config.enable_chunked_prefill:
                    if self.neuron_config.shard_over_sequence:
                        # Communication 1: all-gather query from cores
                        if not self.neuron_config.duplicate_q_weight_sos:
                            query = flash_decoding.gather_query_group(query, self.cores_per_kv_head, n_head, tp_degree)
                        # S = Q @ K (This matmul wastes some computation)
                        contexted_keys = attention_utils.gather_sharded_kv(
                            cached_keys,
                            active_idx=cached_to_contexted,
                            active_tokens=key,
                            active_token_idx=active_to_contexted,
                        )
                        score = attention.score(
                            query,
                            contexted_keys,
                            n_kv_heads=self.config.num_key_value_heads,
                            tp_degree=tp_degree,
                            neuron_config=self.neuron_config,
                        )
                        score = attention.mask(score, mask, tp_degree=tp_degree)
                        # FlashAttention-Style Communication
                        f32 = score.scribe.f32
                        score = hlo.cast(score, f32)
                        max_score_local = hlo.reduce_max(score, dim=3)
                        max_score_local_br = hlo.broadcast(max_score_local, score.sizes, [0, 1, 2])
                        score = hlo.exp(hlo.subtract(score, max_score_local_br))
                        l_sum_score_local = hlo.reduce_sum(score, dim=3)

                        # Value Combination
                        score = hlo.cast(score, cached_values.dtype)
                        contexted_values = attention_utils.gather_sharded_kv(
                            cached_values,
                            active_idx=cached_to_contexted,
                            active_tokens=value,
                            active_token_idx=active_to_contexted,
                        )
                        context = attention.context_combined(
                            score,
                            contexted_values,
                            n_kv_heads=self.config.num_key_value_heads,
                            dtype=score.scribe.f32,
                            tp_degree=tp_degree,
                            neuron_config=self.neuron_config,
                            skip_softmax=True,
                        )
                        # Communication 2: softmax correction
                        context = attention_utils.sharded_softmax_correction(
                            context,
                            max_score_local,
                            l_sum_score_local,
                            core_id,
                            tp_degree=tp_degree,
                            sos_degree=self.cores_per_kv_head,
                        )
                        # Communication 3: reduce-scatter partial context
                        num_groups = tp_degree // self.cores_per_kv_head
                        replica_groups = utils.build_replica_groups(
                            num_groups=num_groups, group_size=self.cores_per_kv_head, interleave=False
                        )
                        context = hlo.reduce_scatter_sum(
                            context, tp_degree=self.cores_per_kv_head, dim=2, replica_groups=replica_groups
                        )
                        context = hlo.cast(context, hidden.dtype)
                    else:
                        # S = Q @ K
                        cached_keys_gathered = attention_utils.gather_blocks(
                            cached_keys, block_tables=block_tables, neuron_config=self.neuron_config
                        )
                        contexted_keys = attention_utils.contexted_kv(
                            cached_keys_gathered, key, cached_mask, cached_to_contexted, active_to_contexted
                        )
                        score = attention.score(
                            query,
                            contexted_keys,
                            n_kv_heads=self.config.num_key_value_heads,
                            tp_degree=tp_degree,
                            neuron_config=self.neuron_config,
                        )

                        score = attention.mask(score, mask, tp_degree=tp_degree)

                        # C = softmax(Sa, Sp) @ (Va, Vp)
                        cached_values_gathered = attention_utils.gather_blocks(
                            cached_values, block_tables=block_tables, neuron_config=self.neuron_config
                        )
                        contexted_values = attention_utils.contexted_kv(
                            cached_values_gathered, value, cached_mask, cached_to_contexted, active_to_contexted
                        )
                        context = attention.context_combined(
                            score,
                            contexted_values,
                            n_kv_heads=self.config.num_key_value_heads,
                            tp_degree=tp_degree,
                            neuron_config=self.neuron_config,
                        )
                else:
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
