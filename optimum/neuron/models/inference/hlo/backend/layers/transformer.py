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
from .. import functional
from ..config import Layout


def ln_lm_head(
    tp_degree,
    hidden,
    last_token_id,
    ln_f_weight,
    ln_f_bias,
    lm_head_weight,
    lm_head_bias,
    is_prefill=True,
    neuron_config=None,
):
    """
    Language model head with layer normalization.

    Context encoding network:
    n_active_tokens will be equal to context_length_estimate.
    In this case we slice the hidden input and compute the next token logits only for the last context token.

    Normal token gen network:
    n_active_tokens will be 1.
    No slicing required. Will return the next token logits for the current active token.

    Models: GPT2, OPT, GPT-J, GPTNeoX, BLOOM.

    logits = (layer_norm(H) @ W) + B
    """
    is_bsh = neuron_config and neuron_config.attention_layout == Layout.BSH
    if is_bsh:
        batch_size, n_active_tokens, hidden_size = hidden.sizes
    else:
        hidden_size, n_active_tokens, batch_size = hidden.sizes

    if is_prefill:
        hidden = _dynamic_logits_slice(hidden, last_token_id, neuron_config)
        # slice the hidden input and compute the next token logits only for the last context token.
        n_active_tokens = 1

    if is_bsh:
        ln_hidden = functional.layer_norm_bsh(hidden, ln_f_weight, ln_f_bias, neuron_config=None, tp_degree=tp_degree)
        ln_hidden = functional.transpose210(ln_hidden)
    else:
        ln_hidden = functional.layer_norm(hidden, ln_f_weight, ln_f_bias, neuron_config=None, tp_degree=tp_degree)
    ln_hidden = functional.reshape(ln_hidden, shape=(hidden_size, n_active_tokens * batch_size))

    logits = functional.dot00(lm_head_weight, ln_hidden)
    if lm_head_bias is not None:
        lm_head_bias = functional.broadcast(lm_head_bias, out_dim_size=logits.sizes, broadcast_dimensions=[0])
        logits = functional.add(logits, lm_head_bias)
    vocab_size, _ = logits.sizes
    return functional.reshape(logits, shape=(vocab_size, n_active_tokens, batch_size))


def rms_lm_head(
    tp_degree,
    hidden,
    last_token_id,
    rms_weight,
    lm_head_weight,
    lm_head_bias,
    is_prefill=True,
    eps=1e-6,
    neuron_config=None,
):
    """
    Language model head with rms normalization.

    Context encoding network:
    n_active_tokens will be equal to context_length_estimate.
    In this case we slice the hidden input and compute the next token logits only for the last context token.

    Normal token gen network:
    n_active_tokens will be 1.
    No slicing required. Will return the next token logits for the current active token.

    Models: LLaMa.

    logits = (rms_norm(H) @ W) + B
    """
    is_bsh = neuron_config and neuron_config.attention_layout == Layout.BSH
    if is_bsh:
        batch_size, n_active_tokens, hidden_size = hidden.sizes
    else:
        hidden_size, n_active_tokens, batch_size = hidden.sizes
    dtype = hidden.dtype

    if is_prefill:
        hidden = _dynamic_logits_slice(hidden, last_token_id, neuron_config)
        # slice the hidden input and compute the next token logits only for the last context token.
        n_active_tokens = 1

    rms_hidden = (
        functional.rms_norm(hidden, rms_weight, eps) if is_bsh else functional.rms_norm(hidden, rms_weight, eps, dim=0)
    )

    if is_bsh:
        rms_hidden = functional.transpose210(rms_hidden)
    rms_hidden = functional.reshape(rms_hidden, (hidden_size, n_active_tokens * batch_size))
    logits = functional.dot00(lm_head_weight, rms_hidden)
    if lm_head_bias is not None:
        lm_head_bias = dtype[logits.sizes].Broadcast(lm_head_bias, dimensions=[0])
        logits = dtype[logits.sizes].Add(logits, lm_head_bias)
    vocab_size, _ = logits.sizes
    return functional.reshape(logits, (vocab_size, n_active_tokens, batch_size))


def _dynamic_logits_slice(hidden, last_token_id, neuron_config):
    is_bsh = neuron_config.attention_layout == Layout.BSH
    if is_bsh:
        batch_size, n_active_tokens, hidden_size = hidden.sizes
    else:
        hidden_size, n_active_tokens, batch_size = hidden.sizes
    if neuron_config.continuous_batching:
        if not is_bsh:
            hidden = functional.transpose210(hidden)
        hidden = functional.reshape(hidden, (batch_size * n_active_tokens, hidden_size))

        # [6,3,9] -> [(0,6),(1,3),(2,9)] -> [6+0*128,3+1*128,9+2*128] -> [6,131,265]
        # last_token_id + iota * n_active_tokens
        assert last_token_id.sizes[0] == batch_size, (
            f"vectorized last_token_id length ({last_token_id.sizes[0]}) is expected to equal to batch size ({batch_size})"
        )
        offset = functional.iota(last_token_id.dtype, last_token_id.sizes, [0])
        offset = functional.multiply(offset, n_active_tokens)
        last_token_id = functional.add(last_token_id, offset)
        hidden = functional.index_select(hidden, dim=0, index=last_token_id)
        hidden = functional.reshape(hidden, (last_token_id.sizes[0], 1, hidden_size))
        if not is_bsh:
            hidden = functional.transpose210(hidden)
    else:
        hidden = functional.transpose102(hidden)
        hidden = functional.index_select(hidden, dim=0, index=last_token_id)
        hidden = functional.transpose102(hidden)
    return hidden
