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
import itertools
import os
import warnings
from abc import ABC, abstractmethod

import torch
from transformers import PretrainedConfig

from . import functional, ops
from .base import NeuronBaseSerializer, NeuronModelBase
from .compiler import (
    DataTypeConverter,
    ParallelKernel,
    compile_py_func,
    gen_zero_input,
    gen_zero_output,
)
from .config import GQA, Layout, NeuronConfig
from .dtypes import to_torch_dtype
from .parallel import ParallelTensorManipulator
from .utils import (
    get_pad_size,
    get_qkv_padding,
    interleave_qkv,
    maybe_pad_tensor,
    pad_interleaved,
    round_up_to_divisor,
)


class GraphBuilder(ABC):
    def __init__(self, config: PretrainedConfig, neuron_config: NeuronConfig):
        self.config = config
        self.neuron_config = neuron_config


class DecoderGraphBuilder(GraphBuilder):
    NUM_INPUTS = 4

    def inputs(
        self,
        scribe,
        dtype,
        batch_size,
        n_active_tokens,
    ):
        """
        Defines the set of required inputs for all decoder models.

        Args:
            scribe: The PyHLO scribe object to write operations with.
            dtype: The data type of the hidden state.
            batch_size: The active batch size (may differ from cache batch size)
            n_active_tokens: The number of active tokens to process. During context
                prefill, this will be larger than 1. During autogregressive
                token generation this will be exactly equal to 1.

        Returns:
            hidden: The hidden state (Assumed to be embedded on CPU)
            cache_ids: The positions to update in the KV cache. This is 1d when
                using RHS-alignment since all batch lines update the same
                places in the KV cache. This is 2d when LHS-alignment since
                each batch line can update a different offset in the KV cache.
            start_ids: The offset into each batch line. When using
                LHS-alignment, this indicates the start offset. When using
                RHS-alignment, this indicates the batch line to update.
            last_token_id: An integer index (along the sequence dimenson) which
                indicates which is the last token. This is used in the language
                model head to slice the hidden state.
            sequence_slice_dimensions: The dimension of each input tensor which can
                be sliced during token generation.
        """
        s32 = scribe.s32

        if self.neuron_config.attention_layout == Layout.BSH:
            hidden_sizes = batch_size, n_active_tokens, self.config.hidden_size
        else:  # HASB LAyout
            hidden_sizes = self.config.hidden_size, n_active_tokens, batch_size

        hidden = dtype[hidden_sizes].Parameter(parameter_number=0)
        if self.neuron_config.continuous_batching:
            position_sizes = batch_size, n_active_tokens
            cache_ids = s32[position_sizes].Parameter(parameter_number=1)  # 2d cache_ids
        else:
            cache_ids = s32[n_active_tokens].Parameter(parameter_number=1)  # 1d cache_ids

        start_ids = s32[batch_size].Parameter(parameter_number=2)

        # Build parameters for last_token_id and others
        if self.neuron_config.continuous_batching:
            # regular token gen
            last_token_id = s32[batch_size].Parameter(parameter_number=3)
        else:
            last_token_id = s32[1].Parameter(parameter_number=3)

        return hidden, cache_ids, start_ids, last_token_id

    @abstractmethod
    def pre_layer(self, hidden, cache_ids, start_ids):
        """
        Provides the pre-layer graph.

        Provides the graph for the initializations to be performed after the creation of the embeddings and before
        going through the Decoder layers.

        It includes in particular the creation of:
        - the position embeddings,
        - the masks (see details in parameters).

        Args:
            hidden: The hidden state (Assumed to be embedded on CPU)
            cache_ids: The positions to update in the KV cache. This is 1d when
                using RHS-alignment since all batch lines update the same
                places in the KV cache. This is 2d when LHS-alignment since
                each batch line can update a different offset in the KV cache.
            start_ids: The offset into each batch line. When using
                LHS-alignment, this indicates the start offset. When using
                RHS-alignment, this indicates the batch line to update.
        Returns:
            hidden: The updated hidden state
            cache_ids: The updated cached_ids
            start_ids: The updated start_ids
            pos_embed:  a tuple containing the position embeddings
            mask: The mask used for the score calculations based on KV cached values
            active_mask: The mask used for the score calculations based on the active token only.
        """
        raise NotImplementedError

    @abstractmethod
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
        *weights,
    ):
        """
        Provides the graph for each Decoder layer forward.

        Args:
            hidden: The hidden state before that layer
            cache_ids: The positions to update in the KV cache. This is 1d when
                using RHS-alignment since all batch lines update the same
                places in the KV cache. This is 2d when LHS-alignment since
                each batch line can update a different offset in the KV cache.
            start_ids: The offset into each batch line. When using
                LHS-alignment, this indicates the start offset. When using
                RHS-alignment, this indicates the batch line to update.
            cache_ids: The updated cached_ids
            start_ids: The updated start_ids
            pos_embed:  a tuple containing the position embeddings
            mask: The mask used for the score calculations based on KV cached values
            active_mask: The mask used for the score calculations based on the active token only.
            attn_k_cache: the attention key cache for that layer.
            attn_v_cache: the attention value cache for that layer.
            weights: the layer weights (as a list of model specific args)
        Returns:
            hidden: The updated hidden state
            attn_k_cache: the updated attention key cache.
            attn_v_cache: the updated attention value cache.
        """
        raise NotImplementedError

    def ln_lm_head(self, hidden, last_token_id, is_prefill, *weights):
        """
        Provides the graph for the Decoder normalization + generation head.

        Args:
            hidden: The hidden state as evaluated by all layers.
            last_token_id: the id of the last token.
            is_prefill: a boolean indicating if the current graph is doing prefill (context encoding) or decode (token generation)
            weights: the combined normalization + head weights (as a list of model specific args)
        Returns:
            logits: the scores for each candidate token in the batch.
        """
        raise NotImplementedError


class DecoderGraph(NeuronBaseSerializer):
    def __init__(
        self,
        n_active_tokens,
        config: PretrainedConfig,
        neuron_config: NeuronConfig,
        is_prefill=True,
        builder=None,
        tag=None,
    ):
        super().__init__()
        self.n_active_tokens = n_active_tokens
        self.config = config
        self.neuron_config = neuron_config
        self.batch_size = self.neuron_config.batch_size
        if self.neuron_config.continuous_batching and is_prefill:
            # Always use a batch_size of 1 when using continuous batching
            self.batch_size = 1
        self.is_prefill = is_prefill
        self.layers = []
        self.ln_f_weight = None
        self.ln_f_bias = None
        self.lm_head_weight = None
        self.lm_head_bias = None
        self.inputs_sdim = None
        self.ln_lm_head_params = []
        self.program = None
        self.use_executor = False
        self.return_ranks = -1
        self.builder = builder
        self.check_gqa_fallback()
        self.tag = tag

    def check_gqa_fallback(self):
        """
        Check if a fallback mechanism is needed for a user-provided GQA config.

        The initial fallback mechanism will be that no special GQA configuration
        is used. This will attempt to evenly distribute Q and KV heads to all
        NeuronCores.

        The second (safest) fallback mechanism will replicate the KV heads to be
        equal to the number of Q heads. This makes the GQA model look identical
        to an MHA model.
        """
        gqa = self.neuron_config.group_query_attention

        if gqa is None:
            # MHA Early exit - This avoids emitting irrelevant GQA warnings
            if self.config.num_attention_heads == self.config.num_key_value_heads:
                return
            self.neuron_config.group_query_attention = GQA.SHARD_OVER_HEADS

        if gqa == GQA.REPLICATED_HEADS:
            return

        if self.config.num_key_value_heads % self.neuron_config.tp_degree != 0:
            warnings.warn(
                f"KV head replication will be enabled since the number of KV "
                f"heads ({self.config.num_key_value_heads}) is not evenly divisible by the "
                f"tensor parallel degree ({self.neuron_config.tp_degree})"
            )
            self.neuron_config.group_query_attention = GQA.REPLICATED_HEADS

    @classmethod
    def init_context_decoder(cls, config, neuron_config, builder):
        return cls(
            config=config,
            neuron_config=neuron_config,
            n_active_tokens=neuron_config.n_positions,
            is_prefill=True,
            builder=builder,
            tag="context",
        )

    @classmethod
    def init_token_decoder(cls, config, neuron_config, builder):
        return cls(
            n_active_tokens=1,
            config=config,
            neuron_config=neuron_config,
            is_prefill=False,
            builder=builder,
            tag="token",
        )

    def enable_executor(self, return_ranks=-1):
        self.return_ranks = return_ranks
        self.program.enable_executor()

    def new_layer(self):
        layer = DecoderLayer(
            self.config,
            self.neuron_config,
            self.batch_size,
            n_active_tokens=self.n_active_tokens,
            layer_num=len(self.layers),
        )
        self.layers.append(layer)
        return layer

    def add_final_layer_norm(self, weight, bias):
        self.ln_f_weight = weight
        self.ln_f_bias = bias

    def add_lm_head(self, weight, bias=None):
        self.lm_head_weight = weight
        self.lm_head_bias = bias

    def to_neuron(self):
        manipulator = MaybeParallelTensorManipulator(self.neuron_config.tp_degree)

        self.ln_f_weight = manipulator.duplicate(self.ln_f_weight)
        self.ln_f_bias = manipulator.duplicate(self.ln_f_bias)
        _, vocab_size = self.lm_head_weight.shape
        # Pad vocab size such that it can be divided by the following factor
        divisor = int(os.environ.get("NEURON_VOCAB_PAD_DIVISOR", str(self.neuron_config.tp_degree)))
        vocab_pad = get_pad_size(vocab_size, divisor)
        lm_head_weight = torch.nn.functional.pad(self.lm_head_weight, (0, vocab_pad, 0, 0))
        self.lm_head_weight = manipulator.shard_along(lm_head_weight, dim=1)
        ln_lm_head_params = [self.ln_f_weight, self.ln_f_bias, self.lm_head_weight]
        ln_lm_head_params = [param for param in ln_lm_head_params if param is not None]
        if self.lm_head_bias is not None:
            self.lm_head_bias = manipulator.shard_along(self.lm_head_bias, dim=0)
            ln_lm_head_params.append(self.lm_head_bias)
        self.ln_lm_head_params = ln_lm_head_params
        self.program = self._build_program()

    def load_shared_weights(self, src_graph):
        for layer in src_graph.layers:
            new_layer = self.new_layer()
            new_layer.assign_parameters(layer)
            new_layer.assign_caches(layer)
            new_layer.extra_parameters = layer.extra_parameters
        self.add_final_layer_norm(src_graph.ln_f_weight, src_graph.ln_f_bias)
        self.add_lm_head(src_graph.lm_head_weight, src_graph.lm_head_bias)
        ln_lm_head_params = [self.ln_f_weight, self.ln_f_bias, self.lm_head_weight]
        ln_lm_head_params = [param for param in ln_lm_head_params if param is not None]
        if self.lm_head_bias is not None:
            ln_lm_head_params.append(self.lm_head_bias)
        self.ln_lm_head_params = ln_lm_head_params
        self.program = self._build_program()

    def setup(self):
        self.program.setup(self.layers, self.ln_lm_head_params)
        if self.use_executor:
            self.enable_executor()

    def reset(self):
        for layer in self.layers:
            layer.reset()

    def forward(self, *inputs):
        """
        This path makes the assumption that inputs are correctly sized for a
        sequence length of 1. This allows us to avoid checking buckets, slicing,
        etc.
        """
        if self.use_executor:
            outputs = self.program.execute(
                *inputs,
                return_ranks=self.return_ranks,
            )
        else:
            self.program.inputs_host_to_device(inputs)
            self.program.run()
            outputs = self.program.maybe_logits_device_to_host(return_ranks=self.return_ranks)

        return outputs

    def _build_program(self):
        def hlo_forward_wrapper(scribe):
            dtype = getattr(scribe, self.neuron_config.amp)

            # Create user parameters
            hidden, cache_ids, start_ids, last_token_id = self.builder.inputs(
                scribe, dtype, self.batch_size, self.n_active_tokens
            )
            param_builder = DecoderParameterBuilder(scribe, self.builder.NUM_INPUTS)

            # Create inputs for all weights & caches
            in_caches, layers_weights, lm_head_params = self._hlo_parameters(
                self.neuron_config.n_positions, self.batch_size, param_builder
            )

            # Unroll the graph
            logits, out_caches = self.hlo_forward(
                hidden,
                cache_ids,
                start_ids,
                last_token_id,
                in_caches,
                layers_weights,
                lm_head_params,
            )
            self._hlo_cache_aliases(in_caches, out_caches)

            # Set the output
            out_caches = itertools.chain(*out_caches)
            if self.neuron_config.log_softmax_scores:
                logits, scores = self._hlo_post_layer(logits)
                outputs = [logits, scores, *out_caches]
            else:
                outputs = [logits, *out_caches]

            # Filter out the None's in outputs
            outputs = [o for o in outputs if o is not None]
            root_shapes = [shape.dtype[shape.sizes] for shape in outputs]
            return scribe.tuple(*root_shapes).Tuple(*outputs)

        hlo_module = compile_py_func(hlo_forward_wrapper)
        return DecoderProgramFullyUnrolled(
            self.neuron_config,
            self.layers,
            hlo_module,
            self.builder.NUM_INPUTS,
            self.batch_size,
            tag=self.tag,
        )

    def hlo_forward(
        self,
        hidden,
        cache_ids,
        start_ids,
        last_token_id,
        layers_caches,
        layers_weights,
        lm_head_params,
    ):
        """Only used at compilation time to create the HLO graph"""
        hidden, cache_ids, start_ids, pos_embed, mask, active_mask = self.builder.pre_layer(
            hidden, cache_ids, start_ids
        )
        output_caches = []
        for caches, weights in zip(layers_caches, layers_weights):
            attn_k_cache, attn_v_cache = [maybe_transfer_with_static_ring(cache) for cache in caches]
            weights = [maybe_transfer_with_static_ring(weight) for weight in weights]
            hidden, attn_k_cache, attn_v_cache = self.builder.layer(
                hidden,
                cache_ids,
                start_ids,
                pos_embed,
                mask,
                active_mask,
                attn_k_cache,
                attn_v_cache,
                *weights,
            )
            output_caches.append([attn_k_cache, attn_v_cache])
        logits = self.builder.ln_lm_head(
            hidden,
            last_token_id,
            self.is_prefill,
            *lm_head_params,
        )
        return logits, output_caches

    def _hlo_parameters(self, n_positions, batch_size, param_builder):
        layers_caches, layers_weights = self._hlo_layers_params(param_builder, self.layers, n_positions)
        lm_head_params = self._hlo_lm_head_params(param_builder)
        return layers_caches, layers_weights, lm_head_params

    def all_parameters(self):
        """
        Get all the parameters for the current model.

        NOTE: It is extremely important that these tensors are returned in the
              same order as the parameters returned in the _hlo_parameters
              function. If this is not done correctly, the HLO parameter and
              the corresponding weight tensor cannot be assocated.
        """
        parameters = []

        # Layer caches
        for layer in self.layers:
            for cache in layer.attn_k_cache, layer.attn_v_cache:
                parameters.append(cache)

        # Layer weights
        for layer in self.layers:
            parameters.extend(layer.all_parameters())

        # LM head parameters
        parameters.append(self.ln_f_weight)
        parameters.append(self.ln_f_bias)
        parameters.append(self.lm_head_weight)
        parameters.append(self.lm_head_bias)

        return parameters

    def valid_parameters(self):
        parameters = self.all_parameters()
        return [par for par in parameters if par is not None]

    def _hlo_layers_params(self, param_builder, layers, n_positions):
        layers_caches = []
        dim_size = {0: n_positions}
        for layer in layers:
            layer_caches = []
            for cache in layer.attn_k_cache, layer.attn_v_cache:
                par = param_builder.from_tensor(cache, dim_size=dim_size)
                layer_caches.append(par)
            layers_caches.append(layer_caches)
        layers_weights = []
        for layer in layers:
            layer_weights = [param_builder.from_tensor(weight) for weight in layer.all_parameters()]
            layers_weights.append(layer_weights)
        return layers_caches, layers_weights

    def _hlo_cache_aliases(self, in_caches, out_caches):
        assert len(in_caches) == len(out_caches)
        for src, dst in zip(itertools.chain(*in_caches), itertools.chain(*out_caches)):
            if dst is not None:
                assert src is not None, "out_cache must alias with a valid cache!"
                dst.set_alias_to(src, must=True)

    def _hlo_lm_head_params(self, param_builder):
        ln_f_weight = param_builder.from_tensor(self.ln_f_weight)
        ln_f_bias = param_builder.from_tensor(self.ln_f_bias)
        head_weight = param_builder.from_tensor(self.lm_head_weight)
        head_bias = param_builder.from_tensor(self.lm_head_bias)
        ln_f_weight = maybe_transfer_with_static_ring(ln_f_weight)
        ln_f_bias = maybe_transfer_with_static_ring(ln_f_bias)
        head_weight = maybe_transfer_with_static_ring(head_weight)
        head_bias = maybe_transfer_with_static_ring(head_bias)
        return ln_f_weight, ln_f_bias, head_weight, head_bias

    def _hlo_post_layer(self, logits):
        return self.post_layer_builder(logits)

    # Mainly used for serialization purposes.
    # Defines how to access all the kernels.
    def get_all_kernels(self):
        return [self.program.kernel]


def read_n_active_tokens(hlo_module):
    return hlo_module.host_program_shape.parameters[0].dimensions[1]


def maybe_transfer_with_static_ring(shape):
    if shape is None:
        return None
    return functional.transfer_with_static_ring(shape)


class MaybePadder:
    def __init__(self, size, padding="end", split_size=None, interleaved_factor=None) -> None:
        self.split_size = split_size
        self.size = size
        self.padding = padding
        self.interleaved_factor = interleaved_factor

    def __call__(self, weight, dim):
        if self.padding == "end":
            return maybe_pad_tensor(weight, dim, self.size, left=False)
        else:
            if weight is None:
                return weight
            assert self.padding == "interleaved", f"Invalid padding mode {self.padding}"
            assert self.interleaved_factor, "interleaved_factor is not provided"
            # when split_size is set, we first split the target weight at dim
            # into (split_size x ?), for example, to do interleaved padding on of KV weight
            # we first need to reshape it into (hidden, num_kv_head, d_head)
            # and then apply interleaved padding on num_kv_head
            weight_shapes = list(weight.shape)

            padded_shape = weight_shapes.copy()
            padded_shape[dim] = self.size

            new_size = self.size
            if self.split_size:
                assert weight_shapes[dim] % self.split_size == 0, (
                    f"shape on dim_{dim} {weight_shapes[dim]} cannot be evenly divisible by provided split_size {self.split_size}"
                )
                new_shape = (
                    weight_shapes[:dim]
                    + [self.split_size]
                    + [weight_shapes[dim] // self.split_size]
                    + weight_shapes[dim + 1 :]
                )
                weight = weight.view(new_shape)
                new_size = self.size // (weight_shapes[dim] // self.split_size)
            res = pad_interleaved(
                weight,
                dim,
                new_size,
                weight.shape[dim] // self.interleaved_factor,
                (new_size - weight.shape[dim]) // self.interleaved_factor,
            )
            return res.view(padded_shape)


class DecoderLayer:
    def __init__(
        self,
        config,
        neuron_config,
        batch_size,
        n_active_tokens=None,
        layer_num=None,
    ):
        super().__init__()
        self.config = config
        self.neuron_config = neuron_config
        self.batch_size = batch_size
        self.pre_attn_ln_weight = None
        self.pre_attn_ln_bias = None
        self.attn_q_weight = None
        self.attn_q_bias = None
        self.attn_k_weight = None
        self.attn_k_bias = None
        self.attn_v_weight = None
        self.attn_v_bias = None
        self.attn_out_weight = None
        self.attn_out_bias = None
        self.post_attn_ln_weight = None
        self.post_attn_ln_bias = None
        self.pre_mlp_ln_weight = None
        self.pre_mlp_ln_bias = None
        self.mlp_in_weight = None
        self.mlp_in_bias = None
        self.mlp_out_weight = None
        self.mlp_out_bias = None
        self.post_mlp_ln_weight = None
        self.post_mlp_ln_bias = None
        # Create KV caches for each batch_size
        self.attn_k_cache = None
        self.attn_v_cache = None
        self.cache_shape = None
        self.n_head = config.num_attention_heads
        self.n_kv_head = config.num_key_value_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.extra_parameters = []
        self.attn_out_sharding = 0
        self.attn_out_transposed = True
        self.mlp_out_sharding = 0
        self.mlp_out_transposed = True
        self.kv_replication = 1  # default value to denote weight replication factor
        self.layer_num = layer_num

    def add_parameter(self, param, sharding=None, allow_transform=False):
        self.extra_parameters.append((param, sharding, allow_transform))

    def add_pre_attention_layer_norm(self, weight, bias):
        self.pre_attn_ln_weight = weight
        self.pre_attn_ln_bias = bias

    def add_attention_query(self, weight, bias):
        self.attn_q_weight = weight
        self.attn_q_bias = bias

    def add_attention_key(self, weight, bias):
        self.attn_k_weight = weight
        self.attn_k_bias = bias

    def add_attention_value(self, weight, bias):
        self.attn_v_weight = weight
        self.attn_v_bias = bias

    def add_attention_output(
        self,
        weight,
        bias,
        sharding=0,
        transposed=True,
        out_feature_dim=None,
        contract_dims=None,
        pad=True,
    ):
        self.attn_out_weight = weight
        self.attn_out_bias = bias
        self.attn_out_sharding = sharding
        self.attn_out_transposed = transposed
        self.attn_out_feature_dim = out_feature_dim
        self.attn_out_contract_dims = contract_dims
        self.attn_out_pad = pad

    def add_pre_mlp_layer_norm(self, weight, bias):
        self.pre_mlp_ln_weight = weight
        self.pre_mlp_ln_bias = bias

    def add_mlp_input(self, weight, bias):
        self.mlp_in_weight = weight
        self.mlp_in_bias = bias

    def add_mlp_output(self, weight, bias, sharding=0, transposed=True):
        self.mlp_out_weight = weight
        self.mlp_out_bias = bias
        self.mlp_out_sharding = sharding
        self.mlp_out_transposed = transposed

    def to_neuron(self):
        # Apply QKV weight paddings + head replication to evenly split weights
        # for the specified tensor parallelism
        # Hidden size padding
        _, hidden_size = self.attn_q_weight.shape
        n_heads = hidden_size // self.attention_head_size

        n_head_padded, n_kv_heads_padded = get_qkv_padding(n_heads, self.n_kv_head, self.neuron_config)

        hidden_size_padded = hidden_size_padded_qkv = n_head_padded * self.attention_head_size
        qkv_maybe_pad = MaybePadder(hidden_size_padded_qkv)
        attn_out_maybe_pad = MaybePadder(hidden_size_padded)

        # Adjust padding strategy if we can use less K/V replication
        # with interleaved padding.
        extra_heads = n_head_padded - n_heads
        if (
            self.n_head != self.n_kv_head
            and self.neuron_config.group_query_attention == GQA.REPLICATED_HEADS
            and self.neuron_config.tp_degree % self.n_kv_head == 0
            and extra_heads % self.n_kv_head == 0
            and extra_heads > 0
        ):
            qkv_maybe_pad = MaybePadder(
                hidden_size_padded_qkv,
                padding="interleaved",
                split_size=n_heads,
                interleaved_factor=self.n_kv_head,
            )
            attn_out_maybe_pad = MaybePadder(
                hidden_size_padded,
                padding="interleaved",
                split_size=n_heads,
                interleaved_factor=self.n_kv_head,
            )

        self.attn_q_weight = qkv_maybe_pad(self.attn_q_weight, dim=1)
        self.attn_q_bias = qkv_maybe_pad(self.attn_q_bias, dim=0)

        if n_kv_heads_padded != self.n_kv_head:
            if n_kv_heads_padded % self.n_kv_head == 0:
                ratio = int(n_kv_heads_padded / self.n_kv_head)
            else:
                ratio = int((n_kv_heads_padded - extra_heads) / self.n_kv_head)

            # Full replication: replicate KV heads to original Q heads and then do padding
            if n_head_padded == n_kv_heads_padded and extra_heads > 0:
                ratio = int((n_kv_heads_padded - extra_heads) / self.n_kv_head)

            def repeat(weight):
                if weight is None:
                    return weight
                shape = weight.shape[:-1] + (
                    self.n_kv_head,
                    weight.shape[-1] // self.n_kv_head,
                )
                weight = weight.view(shape)
                weight = torch.repeat_interleave(weight, repeats=ratio, dim=-2)
                shape = weight.shape[:-2] + (weight.shape[-1] * weight.shape[-2],)
                return weight.view(shape)

            def pad_kv_no_repeat(weight, pad_size):
                if weight is None:
                    return weight
                shape = weight.shape[:-1] + (
                    self.n_kv_head,
                    weight.shape[-1] // self.n_kv_head,
                )
                weight = weight.view(shape)
                weight = torch.nn.functional.pad(weight, (0, 0, 0, pad_size))
                shape = weight.shape[:-2] + (weight.shape[-1] * weight.shape[-2],)
                return weight.view(shape)

            if ratio == 0:
                # in case no replication is needed, pad kv based on n_kv_heads_padded calculated above
                self.attn_k_weight = pad_kv_no_repeat(self.attn_k_weight, n_kv_heads_padded - self.n_kv_head)
                self.attn_v_weight = pad_kv_no_repeat(self.attn_v_weight, n_kv_heads_padded - self.n_kv_head)
                self.attn_k_bias = pad_kv_no_repeat(self.attn_k_bias, n_kv_heads_padded - self.n_kv_head)
                self.attn_v_bias = pad_kv_no_repeat(self.attn_v_bias, n_kv_heads_padded - self.n_kv_head)
                self.n_kv_head = n_kv_heads_padded
            else:
                self.attn_k_weight = repeat(self.attn_k_weight)
                self.attn_v_weight = repeat(self.attn_v_weight)
                self.attn_k_bias = repeat(self.attn_k_bias)
                self.attn_v_bias = repeat(self.attn_v_bias)
                self.n_kv_head *= ratio
            self.kv_replication = ratio
            # FIXME: As a workaround to get kv_replication info (after padding) in HLO construction
            self.neuron_config.kv_replication = self.kv_replication

        if self.n_head == self.n_kv_head:
            self.attn_k_weight = qkv_maybe_pad(self.attn_k_weight, dim=1)
            self.attn_k_bias = qkv_maybe_pad(self.attn_k_bias, dim=0)

            self.attn_v_weight = qkv_maybe_pad(self.attn_v_weight, dim=1)
            self.attn_v_bias = qkv_maybe_pad(self.attn_v_bias, dim=0)

        if self.neuron_config.fuse_qkv:
            fused_qkv_weight = interleave_qkv(
                self.attn_q_weight,
                self.attn_k_weight,
                self.attn_v_weight,
                self.neuron_config.tp_degree,
                dim=1,
            )
            if self.attn_q_bias is not None:
                fused_qkv_bias = interleave_qkv(
                    self.attn_q_bias,
                    self.attn_k_bias,
                    self.attn_v_bias,
                    self.neuron_config.tp_degree,
                    dim=0,
                )
            else:
                fused_qkv_bias = None
            self.attn_k_weight = None
            self.attn_k_bias = None
            self.attn_v_weight = None
            self.attn_v_bias = None
        if self.attn_out_pad:
            self.attn_out_weight = attn_out_maybe_pad(self.attn_out_weight, dim=self.attn_out_sharding)
        # Intermediate MLP layer padding
        if self.mlp_in_weight is not None:
            _, intermediate_size = self.mlp_in_weight.shape
            intermediate_size_padded = round_up_to_divisor(intermediate_size, self.neuron_config.tp_degree)
            maybe_pad = MaybePadder(intermediate_size_padded)

            self.mlp_in_weight = maybe_pad(self.mlp_in_weight, dim=1)
            self.mlp_in_bias = maybe_pad(self.mlp_in_bias, dim=0)
            self.mlp_out_weight = maybe_pad(self.mlp_out_weight, dim=self.mlp_out_sharding)
        # End of replication + padding code

        maybe_manipulator = MaybeParallelTensorManipulator(self.neuron_config.tp_degree)
        maybe_duplicate = maybe_manipulator.duplicate
        maybe_shard_along = maybe_manipulator.shard_along
        maybe_primary_only = maybe_manipulator.primary_only
        maybe_shard_along_and_transform = maybe_manipulator.shard_along_and_transform
        self.pre_attn_ln_weight = maybe_duplicate(self.pre_attn_ln_weight)
        self.pre_attn_ln_bias = maybe_duplicate(self.pre_attn_ln_bias)
        qkv_weight_sharder = maybe_shard_along
        if self.neuron_config and self.neuron_config.fuse_qkv:
            self.attn_q_weight = qkv_weight_sharder(fused_qkv_weight, dim=1)
            self.attn_q_bias = maybe_shard_along(fused_qkv_bias, dim=0)
        else:
            self.attn_q_weight = qkv_weight_sharder(self.attn_q_weight, dim=1)
            self.attn_q_bias = maybe_shard_along(self.attn_q_bias, dim=0)
        self.attn_k_weight = qkv_weight_sharder(self.attn_k_weight, dim=1)
        self.attn_k_bias = maybe_shard_along(self.attn_k_bias, dim=0)
        self.attn_v_weight = qkv_weight_sharder(self.attn_v_weight, dim=1)
        self.attn_v_bias = maybe_shard_along(self.attn_v_bias, dim=0)
        self.attn_out_weight = maybe_shard_along(self.attn_out_weight, dim=self.attn_out_sharding)
        self.attn_out_bias = maybe_primary_only(self.attn_out_bias)
        self.post_attn_ln_weight = maybe_duplicate(self.post_attn_ln_weight)
        self.post_attn_ln_bias = maybe_duplicate(self.post_attn_ln_bias)
        self.pre_mlp_ln_weight = maybe_duplicate(self.pre_mlp_ln_weight)
        self.pre_mlp_ln_bias = maybe_duplicate(self.pre_mlp_ln_bias)
        if self.mlp_in_weight is not None:
            self.mlp_in_weight = maybe_shard_along_and_transform(self.mlp_in_weight, 1)
            self.mlp_in_bias = maybe_shard_along(self.mlp_in_bias, dim=0)
            self.mlp_out_weight = maybe_shard_along_and_transform(self.mlp_out_weight, dim=self.mlp_out_sharding)
            self.mlp_out_bias = maybe_primary_only(self.mlp_out_bias)
        self.post_mlp_ln_weight = maybe_duplicate(self.post_mlp_ln_weight)
        self.post_mlp_ln_bias = maybe_duplicate(self.post_mlp_ln_bias)

        extras = []
        for param, dim, allow_transform in self.extra_parameters:
            size = round_up_to_divisor(param.shape[dim], self.neuron_config.tp_degree)
            param = maybe_pad_tensor(param, dim, size)

            if allow_transform:
                param = maybe_shard_along_and_transform(param, dim)
            else:
                param = maybe_manipulator.duplicate_or_shard_along(param, dim)

            extras.append(param)

        self.extra_parameters = extras
        self.init_caches()

    def init_caches(self):
        n_heads_kv_cache = self.n_kv_head

        # Compute the hidden size taking a potential padding into account. We must
        # allow the KV cache to be padded so it can be evenly divisible across
        # NeuronCores.
        n_heads_kv_cache = round_up_to_divisor(self.n_kv_head, self.neuron_config.tp_degree)
        # Select manipulator based on device
        manipulator = ParallelTensorManipulator(self.neuron_config.tp_degree)
        cpu_cache_shape = [
            self.neuron_config.n_positions,
            self.batch_size,
            n_heads_kv_cache,
            self.attention_head_size,
        ]
        self.cache_shape = [
            self.neuron_config.n_positions,
            self.batch_size,
            n_heads_kv_cache // self.neuron_config.tp_degree,
            self.attention_head_size,
        ]
        cache_dtype = to_torch_dtype(self.neuron_config.amp)
        cpu_cache = torch.zeros(cpu_cache_shape, dtype=cache_dtype)
        assert (n_heads_kv_cache >= self.neuron_config.tp_degree) and (
            n_heads_kv_cache % self.neuron_config.tp_degree == 0
        ), (
            f"cannot shard along kv_heads dimension: n_kv_head={n_heads_kv_cache}, tp_degree={self.neuron_config.tp_degree}"
        )
        self.attn_k_cache = manipulator.shard_along(cpu_cache, dim=2)
        self.attn_v_cache = manipulator.shard_along(cpu_cache, dim=2)

    def assign_caches(self, layer):
        self.attn_k_cache = layer.attn_k_cache
        self.attn_v_cache = layer.attn_v_cache
        self.cache_shape = layer.cache_shape

    def all_parameters(self):
        return [
            self.pre_attn_ln_weight,
            self.pre_attn_ln_bias,
            self.attn_q_weight,
            self.attn_q_bias,
            self.attn_k_weight,
            self.attn_k_bias,
            self.attn_v_weight,
            self.attn_v_bias,
            self.attn_out_weight,
            self.attn_out_bias,
            self.post_attn_ln_weight,
            self.post_attn_ln_bias,
            self.pre_mlp_ln_weight,
            self.pre_mlp_ln_bias,
            self.mlp_in_weight,
            self.mlp_in_bias,
            self.mlp_out_weight,
            self.mlp_out_bias,
            self.post_mlp_ln_weight,
            self.post_mlp_ln_bias,
            *self.extra_parameters,
        ]

    def valid_parameters(self):
        return [par for par in self.all_parameters() if par is not None]

    def reset(self):
        # CPU compilation sometimes returns tensors in a list, eg. [tensor(...), tensor(...)]
        if isinstance(self.attn_k_cache, list):
            self.attn_k_cache = torch.cat(self.attn_k_cache)
        zero_cache = torch.zeros(
            self.attn_k_cache.shape,
            dtype=self.attn_k_cache.dtype,
        )
        zero_cache = [zero_cache for _ in range(self.neuron_config.tp_degree)]
        ops.parallel_write(self.attn_k_cache, zero_cache)
        ops.parallel_write(self.attn_v_cache, zero_cache)

    def assign_parameters(self, layer):
        self.pre_attn_ln_weight = layer.pre_attn_ln_weight
        self.pre_attn_ln_bias = layer.pre_attn_ln_bias
        self.attn_q_weight = layer.attn_q_weight
        self.attn_q_bias = layer.attn_q_bias
        self.attn_k_weight = layer.attn_k_weight
        self.attn_k_bias = layer.attn_k_bias
        self.attn_v_weight = layer.attn_v_weight
        self.attn_v_bias = layer.attn_v_bias
        self.attn_out_weight = layer.attn_out_weight
        self.attn_out_bias = layer.attn_out_bias
        self.post_attn_ln_weight = layer.post_attn_ln_weight
        self.post_attn_ln_bias = layer.post_attn_ln_bias
        self.pre_mlp_ln_weight = layer.pre_mlp_ln_weight
        self.pre_mlp_ln_bias = layer.pre_mlp_ln_bias
        self.mlp_in_weight = layer.mlp_in_weight
        self.mlp_in_bias = layer.mlp_in_bias
        self.mlp_out_weight = layer.mlp_out_weight
        self.mlp_out_bias = layer.mlp_out_bias
        self.post_mlp_ln_weight = layer.post_mlp_ln_weight
        self.post_mlp_ln_bias = layer.post_mlp_ln_bias
        self.extra_parameters = layer.extra_parameters


class MaybeParallelTensorManipulator:
    def __init__(self, tp_degree):
        self.manipulator = ParallelTensorManipulator(tp_degree)

    def duplicate(self, tensor):
        if tensor is None:
            return None
        return self.manipulator.duplicate(tensor)

    def shard_along(self, tensor, dim):
        if tensor is None:
            return None
        return self.manipulator.shard_along(tensor, dim)

    def primary_only(self, tensor):
        if tensor is None:
            return None
        return self.manipulator.primary_only(tensor)

    def duplicate_or_shard_along(self, tensor, dim):
        if dim is None:
            return self.duplicate(tensor)
        return self.shard_along(tensor, dim)

    def shard_along_and_transform(self, tensor, dim):
        if tensor is None:
            return None
        tensors = self.manipulator.shard_along_on_cpu(tensor, dim)
        return ops.parallel_to_nc(tensors)


class DecoderParameterBuilder:
    def __init__(self, scribe, parameter_number):
        self.scribe = scribe
        self.parameter_number = parameter_number
        self.dtype_converter = DataTypeConverter()

    def from_tensor(self, tensor, dim_size=None):
        if tensor is None:
            return None
        # Tensor may be a list of tensors (e.g. [tensor(...)]) during CPU compilation flow
        if isinstance(tensor, list):
            tensor = tensor[0]
        name = self.dtype_converter.torch2name(tensor.dtype)
        dtype = getattr(self.scribe, name)
        sizes = list(tensor.shape)
        if dim_size is not None:
            for dim, size in dim_size.items():
                sizes[dim] = size
        param = dtype[sizes].Parameter(parameter_number=self.parameter_number)
        self.parameter_number += 1
        return param


class DecoderProgram:
    def __init__(
        self,
        neuron_config,
        layers,
        hlo_module,
        num_inputs,
        batch_size,
        tag=None,
        num_exec_repetition=1,
    ):
        self.neuron_config = neuron_config
        self.layers = layers
        self.batch_size = batch_size
        self.input_buffers = [gen_zero_input(hlo_module, idx) for idx in range(num_inputs)]
        kernel_tag = f"seqlen{neuron_config.n_positions}-batch{batch_size}"
        if tag is not None:
            kernel_tag = f"{tag}-seqlen{neuron_config.n_positions}-batch{batch_size}"
        self.kernel = ParallelKernel(
            hlo_module,
            neuron_config.tp_degree,
            g_start_device_id=0,
            g_device_count=neuron_config.tp_degree,
            tag=kernel_tag,
            num_exec_repetition=num_exec_repetition,
        )
        self.n_active_tokens = read_n_active_tokens(hlo_module)
        self.tag = tag
        self.manipulator = ParallelTensorManipulator(neuron_config.tp_degree)

    def setup(self, io_ring_cache_size):
        self.input_buffers = [self.manipulator.duplicate(buf) for buf in self.input_buffers]
        self.logits_buffer = self.manipulator.duplicate(self.logits_buffer)
        self.kernel.load(io_ring_cache_size)

    def inputs_host_to_device(self, input_tensors):
        assert not (len(input_tensors) == 5 and len(self.input_buffers) == 6)
        for idx, (buf, tensor) in enumerate(zip(self.input_buffers, input_tensors)):
            tensor = tensor.to(buf.dtype)
            tensor = self.manipulator.duplicate_on_cpu(tensor)
            assert buf.shape == tensor[0].shape, (
                f"Copying tensor from host to device: buffer ({buf.shape}) and tensor ({tensor[0].shape}) have different shapes!"
            )
            ops.parallel_write(buf, tensor)

    def run(self):
        raise NotImplementedError(DecoderProgram)

    def maybe_logits_device_to_host(self, return_ranks):
        if self.logits_buffer is not None:
            logits = self.manipulator.unshard_along(self.logits_buffer, dim=0)
            if return_ranks > 0:
                rank_size = logits.shape[0] // self.neuron_config.tp_degree
                logits = logits[: rank_size * return_ranks]
            return logits
        else:
            return None

    def _fill_io_tensors(self, input_tensors, output_tensors, layers):
        end = self.neuron_config.n_positions
        for layer in layers:
            for cache in layer.attn_k_cache, layer.attn_v_cache:
                cache_slice = self.manipulator.slice_on_nc(cache, 0, start=0, end=end, step=1)
                input_tensors.append(cache_slice)
                output_tensors.append(cache_slice)
        for layer in layers:
            input_tensors.extend(layer.valid_parameters())


class DecoderProgramFullyUnrolled(DecoderProgram):
    def __init__(
        self,
        neuron_config,
        layers,
        hlo_module,
        num_inputs,
        batch_size,
        tag=None,
    ):
        super().__init__(
            neuron_config,
            layers,
            hlo_module,
            num_inputs,
            batch_size,
            tag=tag,
        )
        self.logits_buffer = gen_zero_output(hlo_module, 0)
        self.memory = None
        self.executor = None

    def setup(self, layers, ln_lm_head_params):
        super().setup(io_ring_cache_size=1)

        self.memory = self.kernel.build_memory()

        # Setup the memory with input and output buffers
        input_tensors = self.input_buffers
        output_tensors = [self.logits_buffer]
        self._fill_io_tensors(input_tensors, output_tensors, layers)
        input_tensors.extend(ln_lm_head_params)
        self.memory.setup(input_tensors, output_tensors)

        # Warmup kernel to avoid unexpected initialization at runtime
        self.kernel.warmup()

    def run(self):
        self.kernel(self.memory)

    def enable_executor(self):
        input_tensors = [*self.input_buffers]
        output_tensors = [self.logits_buffer]
        self.executor = self.kernel.build_executor(self.memory, input_tensors, output_tensors)

    def execute(self, *inputs, return_ranks=-1):
        """
        Execute a kernel with using the optimized ParallelExecutor.

        Arguments:
            inputs: The set of CPU tensors to copy to each model
            return_ranks: The number of ranks to copy back to CPU
        """
        return self.executor(inputs, return_ranks)


class NeuronHloDecoderModel(NeuronModelBase):
    def __init__(self, config, neuron_config, cpu_model, hlo_builder):
        super().__init__(cpu_model)
        self.config = config
        self.neuron_config = neuron_config
        self.decoder_lm_head = DecoderGraph.init_token_decoder(config, neuron_config, hlo_builder)
        self.register_for_serialization(self.decoder_lm_head)
        self.decoder_lm_head_for_context = DecoderGraph.init_context_decoder(config, neuron_config, hlo_builder)
        self.register_for_serialization(self.decoder_lm_head_for_context)

    def reset(self):
        self.decoder_lm_head.reset()

    def decode(self, hidden, *args):
        return self.decoder_lm_head.forward(hidden, *args)

    def context(self, hidden, cache_ids, start_ids, last_token_id, *rest):
        return self.decoder_lm_head_for_context.forward(hidden, cache_ids, start_ids, last_token_id, *rest)

    def _prepare_for_par_ctx_rhs_padding(self, input_ids, cache_ids, start_ids=None, **kwargs):
        """A helper to do rhs padding on prompt for parallel context encoding model
        i.e.
            input_ids = [[111, 222, 333]]
            context_length = 3

            if context bucket size is 4
            we will pad input_ids to [[111, 222, 333, 0]]

            last_token_id = 2 (used for generation to mark the last token is at index 2 instead 3)

        Note:
            - there is no change on start_ids with right padding.
            - cache_ids will be set to [0, 1, 2, 3] in self.forward()
        """
        batch_size, context_length = input_ids.shape

        # if last_token_id not used, simply set to 0
        if self.neuron_config.continuous_batching:
            last_token_id = torch.zeros(batch_size, dtype=torch.int32)
        else:
            last_token_id = torch.as_tensor([0], dtype=torch.int32)
        if context_length == 1:
            # token generation
            return input_ids, cache_ids, last_token_id

        estimate = self.neuron_config.n_positions

        if estimate:
            # when context length is larger than estimate, last_token_id=estimate-1
            if self.neuron_config.continuous_batching:
                last_token_id = cache_ids.max(dim=1).values
            else:
                last_token_id = torch.as_tensor([min(context_length - 1, estimate - 1)], dtype=torch.int32)
            if context_length < estimate:
                input_ids = maybe_pad_tensor(input_ids, 1, estimate, left=False)
                cache_ids = self._pad_cache_ids(cache_ids, batch_size, context_length, estimate)

        return input_ids, cache_ids, last_token_id

    def _pad_cache_ids(self, cache_ids, batch_size, context_length, estimate):
        if self.neuron_config.continuous_batching:
            cache_ids = torch.arange(estimate, dtype=torch.int32)
            cache_ids = cache_ids.unsqueeze(0).expand(batch_size, estimate)
        else:
            if cache_ids is None:
                cache_ids = torch.arange(estimate, dtype=torch.int32)
            else:
                # Inputs: cache_ids = [16, 17], estimate = 512
                #
                # Process:
                # start_idx = 18, end_idx = 528 (= 512+16)
                # padded_elements =       [18, 19, ..., 511, 512, 513, ..., 525, 526, 527]
                # cache_ids_pad = [16, 17, 18, 19, ..., 511, 512, 513, ..., 525, 526, 527]
                # cache_ids =     [16, 17, 18, 19, ..., 511, 511, 511, ..., 511, 511, 511]
                start_idx = cache_ids[-1].item() + 1
                end_idx = estimate + start_idx - context_length
                pad_elements = torch.arange(start_idx, end_idx, dtype=torch.int32)
                cache_ids_pad = torch.concat([cache_ids, pad_elements], dim=0)
                cache_ids = torch.minimum(cache_ids_pad, torch.tensor(estimate - 1, dtype=torch.int32))
        return cache_ids

    def _prepare_for_continuous_batching(self, input_ids, cache_ids=None, seq_ids=None):
        n_seqs, n_active_tokens = input_ids.shape

        if seq_ids is None or not self.neuron_config.continuous_batching:
            # static batching
            return input_ids, cache_ids, seq_ids

        batch_size = self.neuron_config.batch_size

        if (n_active_tokens > 1) and cache_ids.flatten()[0].item() == 0:
            # context encoding
            n_active_seqs, n_active_tokens = input_ids.shape
            n_positions = self.neuron_config.n_positions
            assert n_active_seqs == cache_ids.shape[0], (
                f"invalid n_active_seqs ({n_active_seqs} vs {cache_ids.shape[0]})"
            )
            assert n_active_tokens <= n_positions, f"invalid input prompt length ({n_active_tokens} <= {n_positions})"
            cache_ids_pad = torch.zeros(
                n_active_seqs,
                n_positions,
                dtype=cache_ids.dtype,
                device="cpu",
            )
            for seq_id in range(n_active_seqs):
                cache_ids_pad[seq_id, :n_active_tokens] = cache_ids[seq_id, :n_active_tokens]
            return input_ids, cache_ids_pad, seq_ids

        # token generation - padding for naive continuous batching
        full_input_ids = torch.zeros(batch_size, 1, dtype=input_ids.dtype)
        full_cache_ids = torch.zeros(batch_size, 1, dtype=input_ids.dtype)
        full_seq_ids = torch.arange(batch_size, dtype=torch.int32)

        # vLLM v0.3.3 used to pass 1d seq_ids but starting with
        # v0.4.0 that is no longer the case. To ensure consistent behaviour
        # across versions, we flatten them before unsqueezing them.
        seq_ids_int64 = seq_ids.flatten().unsqueeze(-1).to(torch.int64)
        full_input_ids.scatter_(dim=0, index=seq_ids_int64, src=input_ids)
        full_cache_ids.scatter_(dim=0, index=seq_ids_int64, src=cache_ids)

        return full_input_ids, full_cache_ids, full_seq_ids

    def _preprocess(self, input_ids, start_ids=None, cache_ids=None, **kwargs):
        # enable dynamic batch size feature for continuous batching
        input_ids, cache_ids, new_start_ids = self._prepare_for_continuous_batching(input_ids, cache_ids, start_ids)

        # right pad the input_ids if neccessary
        input_ids, cache_ids, last_token_id = self._prepare_for_par_ctx_rhs_padding(
            input_ids, cache_ids, start_ids, **kwargs
        )
        start_ids = new_start_ids

        # note: this context_length is after right padded
        batch_size, context_length = input_ids.shape

        if start_ids is None:
            start_ids = torch.zeros(batch_size, dtype=torch.int32)

        if cache_ids is None:
            cache_ids = torch.arange(context_length, dtype=torch.int32)
            if self.neuron_config.continuous_batching:
                cache_ids = cache_ids.unsqueeze(0).expand(batch_size, context_length)

        return input_ids, cache_ids, start_ids, last_token_id

    def _postprocess(self, input_ids, logits, start_ids):
        if start_ids is None or (self.neuron_config.output_all_logits and logits.shape[1] > 1):
            return logits

        if not self.neuron_config.continuous_batching or input_ids.shape[-1] > 1:
            return logits

        input_batch_size = start_ids.shape[0]
        seq_ids = start_ids.flatten()
        if torch.equal(seq_ids, torch.arange(input_batch_size)):
            logits = logits[:input_batch_size]
        else:
            logits = logits[seq_ids.to(torch.long)]

        return logits

    def _context_dynamic_batching(self, hidden, *args):
        is_bsh = self.neuron_config and self.neuron_config.attention_layout == Layout.BSH
        input_batch_size = hidden.shape[0] if is_bsh else hidden.shape[2]

        running_batch_size = 1
        if input_batch_size > running_batch_size:
            assert input_batch_size % running_batch_size == 0, (
                "input batch size ({input_batch_size}) not divisible by running batch size ({running_batch_size})"
            )
            n_iters = input_batch_size // running_batch_size
            all_logits = []
            cache_ids, start_ids, last_token_id = args[0], args[1], args[2]
            for iter_id in range(n_iters):
                start_idx = iter_id * running_batch_size
                end_idx = (iter_id + 1) * running_batch_size
                if is_bsh:
                    hidden_per_batch = hidden[start_idx:end_idx, ...]
                else:
                    hidden_per_batch = hidden[..., start_idx:end_idx]
                cache_ids_per_batch = cache_ids[start_idx:end_idx, :]
                start_ids_per_batch = start_ids[start_idx:end_idx]
                last_token_id_per_batch = last_token_id[start_idx:end_idx]
                logits_per_batch = self.context(
                    hidden_per_batch,
                    cache_ids_per_batch,
                    start_ids_per_batch,
                    last_token_id_per_batch,
                )
                all_logits.append(logits_per_batch)
            logits = torch.cat(all_logits, dim=-1)
        else:
            assert input_batch_size == running_batch_size, (
                "input batch size ({input_batch_size}) not equal to running batch size ({running_batch_size})"
            )
            logits = self.context(hidden, *args)
        return logits

    def _forward(self, hidden, *args):
        _, context_length, *_ = hidden.shape

        if context_length > 1:
            continuous_batching = self.neuron_config and self.neuron_config.continuous_batching
            if continuous_batching:
                logits = self._context_dynamic_batching(hidden, *args)
            else:
                logits = self.context(hidden, *args)
        else:
            logits = self.decode(hidden, *args)

        logits = logits.to(torch.float32)
        if self.neuron_config.output_all_logits and context_length > 1:
            logits = logits.permute(2, 1, 0)
        else:
            logits = logits[: self.config.vocab_size, -1, :]
            logits = logits.transpose(0, 1)
        return logits

    def forward(
        self,
        input_ids,
        cache_ids,
        start_ids,
    ):
        original_input_ids = input_ids
        padded_inputs, *rst = self._preprocess(input_ids, start_ids=start_ids, cache_ids=cache_ids)
        # Embeddings are always evaluated on CPU
        input_embeddings = self.cpu_model.model.embed_tokens(padded_inputs)
        if self.neuron_config.attention_layout == Layout.HSB:
            input_embeddings = input_embeddings.transpose(0, -1).contiguous()
        logits = self._forward(input_embeddings, *rst)
        return self._postprocess(original_input_ids, logits, start_ids=start_ids)
