import pytest
import torch
import torch_xla.core.xla_model as xm
from neuronx_distributed.kernels.flash_attn import nki_flash_attn_func
from torch import nn
from transformers import AutoConfig, set_seed

from optimum.neuron.utils.testing_utils import is_trainium_test

from .distributed_utils import distributed_test
from .utils import assert_close


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the function as implemented in the modeling code for Llama and other models.
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


@is_trainium_test
@pytest.mark.parametrize(
    "model_id, dtype",
    [
        pytest.param(
            "meta-llama/Llama-3.2-1B",
            torch.float32,
            marks=pytest.mark.xfail(strict=True, reason="Flash attention does not seem to work right in float32"),
        ),
        ("ibm-granite/granite-3.2-2b-instruct", torch.bfloat16),
    ],
    ids=["llama", "granite"],
)
@distributed_test()
def test_nki_flash_attention(model_id, dtype, set_cache_for_ci):
    """Test the flash attention kernel with a simple example, comparing
    the output with the one from the eager implementation.
    Configuration is taken from the model config.
    """
    set_seed(42)
    config = AutoConfig.from_pretrained(model_id)
    hidden_size = config.hidden_size
    num_attention_heads = config.num_attention_heads
    num_key_value_heads = config.num_key_value_heads

    batch_size = 1
    seq_len = 2048  # Flash attention requires this to be a multiple of 2048
    tp_size = 8  # Simulate as this tp size for this test (so dimensions are smaller)
    scaling = 0.015  # This is just a constant value

    head_dim = hidden_size // num_attention_heads
    num_heads = num_attention_heads // tp_size
    num_kv_heads = num_key_value_heads // tp_size
    num_kv_groups = num_heads // num_kv_heads
    device = "xla"

    query = torch.randn(batch_size, seq_len, num_heads, head_dim).transpose(1, 2).to(device=device, dtype=dtype)
    key = torch.randn(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2).to(device=device, dtype=dtype)
    value = torch.randn(batch_size, seq_len, num_kv_heads, head_dim).transpose(1, 2).to(device=device, dtype=dtype)

    key = repeat_kv(key, num_kv_groups)
    value = repeat_kv(value, num_kv_groups)

    # Eager attention forward
    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling

    causal_mask = torch.triu(torch.ones((1, 1, query.size(2), key.size(2)), device=device), diagonal=1).bool()
    min_value = torch.finfo(attn_weights.dtype).min
    attn_weights = attn_weights.masked_fill_(causal_mask, min_value)

    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    eager_attn_output = torch.matmul(attn_weights, value)
    xm.mark_step()

    # Flash attention forward
    flash_attention_output = nki_flash_attn_func(
        query,
        key,
        value,
        softmax_scale=scaling,
        causal=True,
        mixed_precision=True,
        transpose_nki_inputs=False,
    )
    xm.mark_step()
    assert_close(eager_attn_output, flash_attention_output)
