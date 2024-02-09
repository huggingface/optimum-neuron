import torch
from transformers.modeling_attn_mask_utils import (
    _prepare_4d_causal_attention_mask,
    _prepare_4d_causal_attention_mask_for_sdpa,
)


leaf_prepare_4d_causal_attention_mask = torch.fx._symbolic_trace._create_wrapped_func(
    _prepare_4d_causal_attention_mask
)

leaf_prepare_4d_causal_attention_mask = torch.fx._symbolic_trace._create_wrapped_func(
    _prepare_4d_causal_attention_mask
)

leaf_prepare_4d_causal_attention_maskfor_sdpa = torch.fx._symbolic_trace._create_wrapped_func(
    _prepare_4d_causal_attention_mask_for_sdpa
)
