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


try:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel
except ImportError:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel


from .. import functional
from ..config import Layout, NeuronConfig
from ..nki import nki_call
from ..utils import parse_dtype_replica_groups


def update_indices_decode(cached_keys, cache_ids, neuron_config=None):
    n_positions, n_seqs, n_kv_heads, d_head = cached_keys.sizes
    cache_ids_dtype = cache_ids.dtype
    # [6,3,9] -> [(0,6),(1,3),(2,9)] -> [6*3,3*3+1,9*3+2] -> [18,10,29]
    # cache_ids * n_seqs + iota
    batch_size_br = functional.full(n_seqs, cache_ids_dtype, cache_ids.sizes)
    indices = cache_ids_dtype[cache_ids.sizes].Multiply(cache_ids, batch_size_br)
    offset = cache_ids_dtype[cache_ids.sizes].Iota(dimensions=[1])
    indices = cache_ids_dtype[cache_ids.sizes].Add(indices, offset)
    return indices


def update_indices_context(cached_keys, cache_ids, start_ids, neuron_config=None):
    # Check K/V cache layout
    n_positions, n_seqs, n_kv_heads, d_head = cached_keys.sizes
    cache_ids_dtype = cache_ids.dtype
    # [0,1,2,3] -> [(1,0),(1,1),(1,2),(1,3)] -> [1+0*3,1+1*3,1+2*3,1+3*3] -> [1,4,7,10]
    # start_ids + iota * n_seqs
    batch_size_br = functional.full(n_seqs, cache_ids_dtype, cache_ids.sizes)
    start_ids_br = functional.broadcast(start_ids, cache_ids.sizes, [1])
    indices = cache_ids_dtype[cache_ids.sizes].Iota(dimensions=[0])
    indices = cache_ids_dtype[cache_ids.sizes].Multiply(indices, batch_size_br)
    indices = cache_ids_dtype[cache_ids.sizes].Add(indices, start_ids_br)
    return indices


def wrapper_flash_attention_bir(q, k, v, out, scale=1.0, kernel_name="CausalAttentionMMSoftmaxMMWithoutSwap"):
    attention_isa_kernel(q, k, v, scale, out, kernel_name)


def query_key_value(
    hidden,
    q_weight,
    q_bias,
    k_weight,
    k_bias,
    v_weight,
    v_bias,
    d_head,
    neuron_config=None,
    n_kv_heads_tp=None,
):
    """
    Self-attention input projections.

    Q = (hidden @ wQ) + bQ
    K = (hidden @ wK) + bK
    V = (hidden @ wV) + bV

    If n_kv_heads != 0, uses multi-query, multi-group attention.
    n_kv_heads == 0 -> outputs shapes [n_active_tokens, n_seqs, n_heads_tp, d_head]
    n_kv_heads != 0 -> outputs shapes [n_active_tokens, n_seqs, n_kv_heads, n_repeats, d_head] (query)
    and [n_active_tokens, n_seqs, n_kv_heads, d_head] (key/value)
    """
    if neuron_config and neuron_config.attention_layout == Layout.BSH:
        hidden = functional.transpose210(hidden)

    hidden_size, n_active_tokens, n_seqs = hidden.sizes
    _, hidden_size_tp = q_weight.sizes
    fuse_qkv = neuron_config and neuron_config.fuse_qkv
    if fuse_qkv:
        # If KV head count explicit, find Q head count
        if n_kv_heads_tp is not None:
            n_total_heads_tp = hidden_size_tp // d_head
            n_heads_tp = n_total_heads_tp - 2 * n_kv_heads_tp
            # Q hidden size
            hidden_size_tp = d_head * n_heads_tp
            # KV hidden size
            kv_hidden_size_tp = d_head * n_kv_heads_tp
        # KV head count not specified, assume same as Q
        else:
            fused_qkv_ratio = 3  # Q + K + V
            hidden_size_tp //= fused_qkv_ratio
            kv_hidden_size_tp = hidden_size_tp
            n_heads_tp = hidden_size_tp // d_head
            n_kv_heads_tp = kv_hidden_size_tp // d_head
    else:
        _, kv_hidden_size_tp = k_weight.sizes
        n_heads_tp = hidden_size_tp // d_head
        n_kv_heads_tp = kv_hidden_size_tp // d_head

    # (h, s, b) => (h, s * b)
    hidden_r = functional.reshape(hidden, (hidden_size, n_active_tokens * n_seqs))

    # Fused MHA
    if fuse_qkv:
        # QKV = (hidden @ wQKV) + bQKV
        active_qkv = functional.dot00_add1(hidden_r, q_weight, q_bias)

        # Split
        slice_lim = active_qkv.sizes[-1] // (n_heads_tp + 2 * n_kv_heads_tp)
        active_q = functional.slice_along(active_qkv, -1, n_heads_tp * slice_lim, start=0)
        active_k = functional.slice_along(
            active_qkv,
            -1,
            (n_heads_tp + n_kv_heads_tp) * slice_lim,
            start=n_heads_tp * slice_lim,
        )
        active_v = functional.slice_along(
            active_qkv,
            -1,
            (n_heads_tp + 2 * n_kv_heads_tp) * slice_lim,
            start=(n_heads_tp + n_kv_heads_tp) * slice_lim,
        )

    # MHA & Non-sharded KV GQA
    else:
        # Q = (hidden @ wQ) + bQ
        active_q = functional.dot00_add1(hidden_r, q_weight, q_bias)

        # K = (hidden @ wK) + bK
        active_k = functional.dot00_add1(hidden_r, k_weight, k_bias)

        # V = (hidden @ wV) + bV
        active_v = functional.dot00_add1(hidden_r, v_weight, v_bias)

    # shard over head
    active_q_sizes = n_active_tokens, n_seqs, n_heads_tp, d_head
    active_kv_sizes = n_active_tokens, n_seqs, n_kv_heads_tp, d_head
    active_q = functional.reshape(active_q, active_q_sizes)
    active_k = functional.reshape(active_k, active_kv_sizes)
    active_v = functional.reshape(active_v, active_kv_sizes)

    return active_q, active_k, active_v


def fused_kv_update_cache(cached_keys, cached_vals, cache_ids, keys, vals, start_ids=None, neuron_config=None):
    """
    The fused K/V cache update is intended for reducing replicated index value calculation for both keys and values,
    since we are updating K/V with the same index offset.

    KeyCache[I], ValueCache[I] = Keys, Values
    """

    dtype = cached_keys.dtype
    use_2d_cache_ids = len(cache_ids.sizes) > 1
    if not use_2d_cache_ids:
        updated_keys = update_cache(cached_keys, cache_ids, keys)
        updated_vals = update_cache(cached_vals, cache_ids, vals)
        return updated_keys, updated_vals

    # 2D cache_ids
    cache_ids = functional.transpose(cache_ids, 0, 1)
    assign_func = functional.gen_assign_func(dtype)
    # K/V cache layout is always SBH
    n_positions, n_seqs, n_kv_heads, d_head = cached_keys.sizes
    n_active_tokens, n_active_seqs, _, _ = keys.sizes
    assert cache_ids.sizes[0] == n_active_tokens, (
        f"inconsistent sizes between cache_ids ({cache_ids.sizes}) and values ({keys.sizes})"
    )

    # reshape cache, and scatter values in a for loop.
    #
    # NOTE: Due to limitation in functional.scatter, we make cache flatten: (p0 as positions, s0 as sequences)
    #       (p0, s0), (p0, s1), (p0, s2), (p1, s0), (p1, s1), (p1, s2)
    #       This means we cannot update the sequence in the cache with one scatter op, without reordering the cache.
    kv_hidden_size = n_kv_heads * d_head
    cached_keys_r = functional.reshape(cached_keys, [n_positions * n_seqs, kv_hidden_size])
    cached_vals_r = functional.reshape(cached_vals, [n_positions * n_seqs, kv_hidden_size])

    if n_active_tokens == 1 and (n_seqs == n_active_seqs):
        # cache (2D): [n_positions * n_seqs, n_kv_heads * d_head]
        #        +---------3-4-----6-7------9-10-----------------
        # seq 0  |                [A,B]
        # seq 1  |        [C,D]
        # seq 2  |                         [E,F]
        #        +-----------------------------------------------
        # seq_ids:      cache_ids: (n_active_tokens, n_seqs)     values: (n_active_tokens, n_seqs, n_heads, d_head)
        # seq 0         [[6,7],                                  [[A,B],
        # seq 1          [3,4],                                   [C,D],
        # seq 2          [9,10]]                                  [E,F]]
        #
        keys_r = functional.reshape(keys, [n_active_seqs, kv_hidden_size])
        vals_r = functional.reshape(vals, [n_active_seqs, kv_hidden_size])

        indices = update_indices_decode(cached_keys, cache_ids, neuron_config)
        indices = functional.transpose(indices, 0, 1)

        scatter_dims = {
            "update_window_dims": [1],
            "inserted_window_dims": [0],
            "scatter_dims_to_operand_dims": [0],
            "index_vector_dim": 1,
        }
        updated_keys = functional.scatter(
            cached_keys_r,
            indices,
            keys_r,
            scatter_dims=scatter_dims,
            to_apply=assign_func,
        )
        updated_vals = functional.scatter(
            cached_vals_r,
            indices,
            vals_r,
            scatter_dims=scatter_dims,
            to_apply=assign_func,
        )

        updated_keys = functional.reshape(updated_keys, [n_positions, n_seqs, n_kv_heads, d_head])
        updated_vals = functional.reshape(updated_vals, [n_positions, n_seqs, n_kv_heads, d_head])

    elif (n_active_tokens == n_positions) and (n_seqs > n_active_seqs):
        # cache (2D): [n_positions * n_seqs, n_kv_heads * d_head]
        #        +-0-1-2-3-4-5-----------------------------------
        # seq 0  |[x,x,x,x,x,x,x,x,x,x,x,x,x,x,x,x]
        # seq 1  |[A,B,C,D,E,F] <- insert new sequence here
        # seq 2  |[y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y,y]
        #        +-----------------------------------------------
        # seq_ids:      cache_ids: (n_active_tokens, n_seqs)     values: (n_active_tokens, n_seqs, n_heads, d_head)
        # seq 1         [[0,1,2,3,4,5]]                          [[A,B,C,D,E,F]]
        keys_r = functional.reshape(keys, [n_active_tokens, kv_hidden_size])
        vals_r = functional.reshape(vals, [n_active_tokens, kv_hidden_size])

        indices = update_indices_context(cached_keys, cache_ids, start_ids, neuron_config)

        # For prefill, assuming n_active_seqs == 1, due to KV cache layout issue.
        assert n_active_seqs == 1, "n_active_seqs is expected to be 1 for 2D cache_ids"

        scatter_dims = {
            "update_window_dims": [1],
            "inserted_window_dims": [0],
            "scatter_dims_to_operand_dims": [0],
            "index_vector_dim": 1,
        }
        updated_keys = functional.scatter(
            cached_keys_r,
            indices,
            keys_r,
            scatter_dims=scatter_dims,
            to_apply=assign_func,
        )
        updated_vals = functional.scatter(
            cached_vals_r,
            indices,
            vals_r,
            scatter_dims=scatter_dims,
            to_apply=assign_func,
        )

        updated_keys = functional.reshape(updated_keys, [n_positions, n_seqs, n_kv_heads, d_head])
        updated_vals = functional.reshape(updated_vals, [n_positions, n_seqs, n_kv_heads, d_head])

    else:
        raise NotImplementedError(
            f"Updating 2D cache_ids is not implemented for "
            f"n_active_tokens={n_active_tokens}, n_positions={n_positions}, "
            f"n_seqs={n_seqs}, n_active_seqs={n_active_seqs}."
        )

    return updated_keys, updated_vals


def update_cache(cache, cache_ids, values):
    """
    Cache[I] = X
    """
    dtype = cache.dtype
    # 1D cache_ids
    scatter_dims = {
        "update_window_dims": [1, 2, 3],
        "inserted_window_dims": [0],
        "scatter_dims_to_operand_dims": [0],
        "index_vector_dim": 1,
    }
    assign_func = functional.gen_assign_func(dtype)
    updated = functional.scatter(cache, cache_ids, values, scatter_dims=scatter_dims, to_apply=assign_func)
    return updated


def scale(query, d_head):
    """
    Scales the query by the number of attention heads

    Q = Q / sqrt(d_head)
    """
    dtype = query.dtype
    scale = dtype.Constant(constant_value=d_head**0.5)
    scale_br = dtype[query.sizes].Broadcast(scale, dimensions=[])
    return dtype[query.sizes].Divide(query, scale_br)


def score(query, keys, n_kv_heads=0):
    """
    Compute the attention score by combining scaled-query & keys.

    S = Q @ K

    If n_kv_heads != 0, uses multi-query, multi-group attention.
    NOTE: Since we may pad along head dimension,
          tp_degree argument is required to be an integer for grouped-query attention models.
    """
    # Check for MQA/GQA attention
    if n_kv_heads != 0:
        _, _, n_kv_heads_tp, _ = keys.sizes
        _, _, n_heads_tp, _ = query.sizes
        n_repeats = n_heads_tp // n_kv_heads_tp
        keys = functional.repeat_kv(keys, n_repeats=n_repeats, repeat_dim=2)

    # Q @ K
    batch_dimensions = [1, 2]
    dot_dims = {
        "lhs_contracting_dimensions": [3],
        "lhs_batch_dimensions": batch_dimensions,
        "rhs_contracting_dimensions": [3],
        "rhs_batch_dimensions": batch_dimensions,
    }

    result_dot = functional.dot_general(query, keys, dimension_numbers=dot_dims)

    return result_dot


def mask(score, mask, tp_degree=None, constant_value=-30000):
    """
    Masks the computed attention scores with the attention mask.

    score = masked_fill(score, mask, -65535)
    """
    dtype = score.dtype
    score_sizes = score.sizes

    # Note: This value can cause NaN issues if it is too large
    large_neg = dtype.Constant(constant_value=constant_value)  # Valid for fp32/fp16/bf16
    large_neg_br = dtype[score_sizes].Broadcast(large_neg, dimensions=[])
    if len(mask.sizes) == 2:
        # broadcast from [n_seqs, n_active_tokens] to [n_seqs, n_heads, n_active_tokens, n_positions]
        mask_br = functional.broadcast(mask, score_sizes, [0, 2])
    else:
        # broadcast from [n_seqs, n_active_tokens, n_positions] to [n_seqs, n_heads, n_active_tokens, n_positions]
        mask_br = functional.broadcast(mask, score_sizes, [0, 2, 3])
    score = functional.masked_select(mask_br, score, large_neg_br)
    return score


def context(
    past_scores,
    active_score,
    past_values,
    active_values,
    past_mask=None,
    active_mask=None,
    n_kv_heads=0,
    dtype=None,
    tp_degree=None,
):
    """
    Compute "context" output from the QK score and value projection.

    This computes the output using split past and current values. This can be
    efficient when computing a *single* next token score since it removes the
    data dependency on an updated KV cache.

    C = softmax(S) @ V

    Implementation details:
        - If n_kv_heads != 0, uses multi-query, multi-group attention.
        - If dtype is None, uses values datatype.
        - If past_mask or active_mask is provided, apply the mask to the result
            of the softmax exp as an optimization to help with compiler
            constant propagation.
    """

    if dtype is None:
        dtype = active_score.dtype
    scribe = active_score.scribe
    f32 = scribe.f32

    n_seqs, n_heads, n_active_tokens, n_active_tokens = active_score.sizes
    _, _, _, n_positions = past_scores.sizes
    n_positions, _, n_kv_heads_tp, d_head = past_values.sizes
    n_seqs = active_values.sizes[1]
    _, n_heads_tp, _, _ = active_score.sizes

    # Upcast to f32 before computation
    past_scores = functional.cast(past_scores, f32)
    active_score = functional.cast(active_score, f32)

    # Compute maximum of both past_scores and active_scores
    reduce_max = functional.reduce_max(past_scores, dim=3)
    active_reduce_max = functional.reduce_max(active_score, dim=3)

    reduce_max = functional.maximum(reduce_max, active_reduce_max)
    reduce_max_br = functional.broadcast(reduce_max, past_scores.sizes, broadcast_dimensions=[0, 1, 2])

    # Pa = softmax(Sa)
    # Pp = softmax(Sp)
    score_shifted = functional.subtract(past_scores, reduce_max_br)
    exp = functional.exp(score_shifted)
    if past_mask is not None:
        exp = mask(
            exp,
            past_mask,
            tp_degree=tp_degree,
            constant_value=0,
        )
    denom = functional.reduce_sum(exp, dim=3)
    past_prob = functional.cast(exp, dtype)
    reduce_max_bra = functional.broadcast(
        reduce_max,
        list(reduce_max.sizes) + [n_active_tokens],
        broadcast_dimensions=[0, 1, 2],
    )
    active_score_shifted = functional.subtract(active_score, reduce_max_bra)
    active_prob = functional.exp(active_score_shifted)
    if active_mask is not None:
        active_prob = mask(
            active_prob,
            active_mask,
            tp_degree=tp_degree,
            constant_value=0,
        )
    active_denom = functional.reduce_sum(active_prob, dim=3)

    denom = functional.add(denom, active_denom)
    active_prob = functional.cast(active_prob, dtype)

    # Ca = Pa @ Va
    # Cp = Pp @ Vp
    # C = Ca + Cp
    dot_dims = {
        "lhs_contracting_dimensions": [3],
        "lhs_batch_dimensions": [0, 1],
        "rhs_contracting_dimensions": [0],
        "rhs_batch_dimensions": [1, 2],
    }
    denom = dtype[denom.sizes].Convert(denom)

    if n_kv_heads != 0:
        _, n_heads_tp, *_ = past_prob.sizes
        _, _, n_kv_heads_tp, *_ = past_values.sizes
        n_repeats = n_heads_tp // n_kv_heads_tp

        # values layout: (n_positions, n_seqs_per_nc, n_kv_heads, d_head) -> repeat_dim=2
        past_values = functional.repeat_kv(past_values, n_repeats=n_repeats, repeat_dim=2)
        active_values = functional.repeat_kv(active_values, n_repeats=n_repeats, repeat_dim=2)

    # lhs (past_prob): (n_seqs, n_heads, n_active_tokens, n_positions)
    # rhs (value):
    # - SBH cache layout: (n_positions, n_seqs, n_heads, d_head)
    rhs_contracting_dimensions = [0]
    rhs_batch_dimensions = [1, 2]
    dot_dims = {
        "lhs_contracting_dimensions": [3],
        "lhs_batch_dimensions": [0, 1],
        "rhs_contracting_dimensions": rhs_contracting_dimensions,
        "rhs_batch_dimensions": rhs_batch_dimensions,
    }

    output_dot = functional.dot_general(past_prob, past_values, dimension_numbers=dot_dims)
    active_output_dot = functional.dot_general(active_prob, active_values, dimension_numbers=dot_dims)
    output = functional.add(output_dot, active_output_dot)

    sizes = n_seqs, n_heads_tp, n_active_tokens, d_head
    denom_br = functional.broadcast(denom, sizes, broadcast_dimensions=[0, 1, 2])
    output = functional.divide(output, denom_br)
    # (n_seqs, n_heads_tp, n_active_tokens, d_head) -> (n_active_tokens, n_seqs, n_heads_tp, d_head)
    output = functional.permute(output, dimensions=[2, 0, 1, 3])
    return output


def context_combined(
    score,
    values,
    n_kv_heads=0,
    dtype=None,
    skip_softmax=False,
):
    """
    Compute "context" output from the QK score and value projection.

    This function assumes that scores and values contains the both current *and*
    past values. This is unlike the split `context` layer which assumes that
    the key/value tensors are split between current/past values. The combined
    context may be useful during input token KV cache population while the
    split context function will provide better performance during single token
    generation.

    C = softmax(S) @ V

    If n_kv_heads != 0, uses multi-query, multi-group attention.
    If dtype is None, uses values datatype.
    """
    if skip_softmax:
        probs = score
    else:
        probs = functional.softmax(score)

    n_seqs, n_heads_tp, n_active_tokens, n_positions = probs.sizes
    _, _, n_kv_heads_tp, d_head = values.sizes

    if dtype is None:
        dtype = values.dtype
    probs = functional.cast(probs, dtype)

    if n_kv_heads != 0:
        _, n_seqs, n_kv_heads_tp, d_head = values.sizes
        n_repeats = n_heads_tp // n_kv_heads_tp
        values = functional.repeat_kv(values, n_repeats=n_repeats, repeat_dim=2)

    rhs_contracting_dimensions = [0]
    rhs_batch_dimensions = [1, 2]
    dot_dims = {
        "lhs_contracting_dimensions": [3],
        "lhs_batch_dimensions": [0, 1],
        "rhs_contracting_dimensions": rhs_contracting_dimensions,
        "rhs_batch_dimensions": rhs_batch_dimensions,
    }
    result = functional.dot_general(probs, values, dimension_numbers=dot_dims, dtype=dtype)

    if n_kv_heads != 0:
        result_sizes = n_seqs, n_heads_tp, n_active_tokens, d_head
        result = functional.reshape(result, result_sizes)

    sizes = n_active_tokens, n_seqs, n_heads_tp, d_head
    result = dtype[sizes].Transpose(result, dimensions=[2, 0, 1, 3])
    return result


def output(
    context: "HloShape",  # noqa F821
    out_weight: "HloShape",  # noqa F821
    out_bias: "HloShape",  # noqa F821
    tp_degree: int,
    neuron_config: Optional[NeuronConfig] = None,
    transposed: Optional[bool] = False,
):
    """
    The output projection of a transformer applied to the attention context.

    O = (C @ wO) + bO

    Arguments:
        context: Attention context.
        out_weight: Model attention outout projection weight.
        out_bias: Model attention outout projection bias.
        tp_degree: Tensor parallelism degree.
        neuron_config: NeuronConfig object that specifies the quantization and
            collectives configurations.
        transposed: Whether the weight is transposed.

    Implementation details:

        2D out_weight case:
            Weight shape if transposed:
                [d_head * n_heads_tp, hidden]
            else:
                [hidden, d_head * n_heads_tp]

            Dot if transposed:
                (s * b, padded_h) @ (padded_h, h) contract=(1, 0) => (s * b, h)
            else:
                (s * b, padded_h) @ (h, padded_h) contract=(1, 1) => (s * b, h)

        3D out_weight case
            Weight shape if transposed:
                [d_head, n_heads_tp, hidden]
            else:
                [hidden, d_head, n_heads_tp]

            Dot if transposed:
                (s * b, d_head, n_heads_tp) @ (d_head, n_heads_tp, h) contract=((1, 2), (0, 1)) => (s * b, h)
            else:
                (s * b, d_head, n_heads_tp) @ (h, d_head, n_heads_tp) contract=((1, 2), (1, 2)) => (s * b, h)
    """
    dtype = context.dtype
    n_active_tokens, n_seqs, n_heads_tp, d_head = context.sizes
    transposed = neuron_config and neuron_config.attn_output_transposed

    if transposed:
        *_, hidden_size = out_weight.sizes
    else:
        hidden_size, *_ = out_weight.sizes
    hidden_sizes = hidden_size, n_active_tokens, n_seqs

    three_dims = len(out_weight.sizes) == 3

    if three_dims:
        result_sizes = n_active_tokens * n_seqs, n_heads_tp, d_head
    else:
        result_sizes = n_active_tokens * n_seqs, n_heads_tp * d_head

    result = functional.reshape(context, result_sizes)

    if three_dims:
        # (s * b, n_heads_tp, d_head) -> (s * b, d_head, n_heads_tp)
        result = functional.permute(result, (0, 2, 1))

    if three_dims:
        if transposed:
            lhs_contract_dims = [1, 2]
            rhs_contract_dims = [0, 1]
        else:
            lhs_contract_dims = [1, 2]
            rhs_contract_dims = [1, 2]
    else:
        if transposed:
            lhs_contract_dims = [1]
            rhs_contract_dims = [0]
        else:
            lhs_contract_dims = [1]
            rhs_contract_dims = [1]

    result = functional.dot_add(
        lhs=result,
        rhs=out_weight,
        bias=out_bias,
        lhs_contracting_dimension=lhs_contract_dims,
        rhs_contracting_dimension=rhs_contract_dims,
        bias_dimension=1,
    )

    bsh_collective = neuron_config and neuron_config.collectives_layout == Layout.BSH
    bsh_output = neuron_config and neuron_config.attention_layout == Layout.BSH

    if bsh_output or bsh_collective:
        # (s * b, h) => (b, s, h)
        result = functional.reshape(result, (n_active_tokens, n_seqs, hidden_size))
        result = functional.transpose(result, 0, 1)
    else:
        # (s * b, h) => (h, s, b)
        result = functional.transpose(result, 0, 1)
        result = functional.reshape(result, hidden_sizes)

    dtype, replica_groups = parse_dtype_replica_groups(neuron_config)
    result = functional.all_reduce_sum(result, neuron_config.tp_degree, dtype=dtype, replica_groups=replica_groups)

    # Transpose back to HSB if applicable
    if bsh_collective and not bsh_output:
        return functional.permute(result, (2, 1, 0))
    return result


def flash_attention(query, key, value):
    n_active_tokens, batch_size, n_q_heads_tp, d_head = query.sizes

    if n_active_tokens < 4096:
        # kernel gives minimal benefit for smaller sequence lengths
        return None

    if n_active_tokens <= d_head:
        # kernel assumes n_active_tokens > d_head and uses this fact to determine i/o layout
        return None

    # handle GQA by broadcasting kv
    if query.sizes[2] != key.sizes[2]:
        n_repeats = query.sizes[2] // key.sizes[2]
        key = functional.repeat_kv(key, n_repeats=n_repeats, repeat_dim=2)
        value = functional.repeat_kv(value, n_repeats=n_repeats, repeat_dim=2)

    if query.sizes[2] != key.sizes[2]:  # condition required by kernel
        return None

    # incoming qkv has shape: (n_active_tokens, batch_size, n_q_heads_tp, d_head)
    # we transpose to match expected shape by kernel
    # we also need a reshape since kernel combines batch and n heads into single dim
    query_nki = functional.reshape(
        functional.permute(query, [1, 2, 3, 0]),
        (batch_size * n_q_heads_tp, d_head, n_active_tokens),
    )
    key_nki = functional.reshape(
        functional.permute(key, [1, 2, 3, 0]),
        (batch_size * n_q_heads_tp, d_head, n_active_tokens),
    )
    value_nki = functional.reshape(
        functional.permute(value, [1, 2, 0, 3]),
        (batch_size * n_q_heads_tp, n_active_tokens, d_head),
    )
    nki_output = nki_call(
        wrapper_flash_attention_bir,
        query_nki,
        key_nki,
        value_nki,
        output_HloShapes=[query.dtype[batch_size * n_q_heads_tp, n_active_tokens, d_head]],
    )
    # kernel output (after separating batch and n heads dims) has shape:
    # (batch_size, n_q_heads_tp, n_active_tokens, d_head)
    # we permute it to (n_active_tokens, batch_size, n_q_heads_tp, d_head)
    context = functional.permute(
        functional.reshape(nki_output, (batch_size, n_q_heads_tp, n_active_tokens, d_head)),
        [2, 0, 1, 3],
    )

    return context
