import logging
import math
from typing import Optional, Tuple

import torch
from torch import nn, Tensor

from modules.attention.utils import apply_rotary_pos_emb, repeat_kv, manual_softmax, move_heads_front

# Try except for the compatibility with older compiler version
try:
    from neuronxcc.nki._private_kernels.attention import attention_isa_kernel  # noqa: E402
except ImportError:
    from neuronxcc.nki.kernels.attention import attention_isa_kernel  # noqa: E402
from torch_neuronx.xla_impl.ops import nki_jit  # noqa: E402

from modules.gqa import (  # noqa: E402
    GroupQueryAttention_O,  # noqa: E402
    GroupQueryAttention_QKV,  # noqa: E402
)  # noqa: E402

from neuronx_distributed.parallel_layers import utils  # noqa: E402

_flash_fwd_call = nki_jit()(attention_isa_kernel)


class NeuronAttentionBase(nn.Module):
    """
    This base attention class implements the core Neuron related adaptation including
    1. replaces the q_proj, k_proj, v_proj with column parallel layer
    2. replaces the o_proj with row parallel layer
    3. update self.num_head to be self.num_head / tp_degree
    4. update self.num_key_value_heads to be self.num_key_value_heads / tp_degree
    5. update forward() method to adjust to changes from self.num_head
    """

    def __init__(self):
        super().__init__()
        self.is_causal = True
        self.num_key_value_groups = None
        self.num_key_value_heads = None
        self.num_heads = None
        self.rotary_emb = None
        self.o_proj = None
        self.qkv_proj = None

    def init_gqa_properties(self):
        if (self.head_dim * self.num_attention_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_attention_heads})."
            )
        self.qkv_proj = GroupQueryAttention_QKV(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            tp_degree=self.tp_degree,
            dtype=self.torch_dtype,
            gather_output=False,
            fused_qkv=self.fused_qkv,
            clip_qkv=self.clip_qkv
        )
        self.o_proj = GroupQueryAttention_O(
            hidden_size=self.hidden_size,
            head_dim=self.head_dim,
            num_attention_heads=self.num_attention_heads,
            num_key_value_heads=self.num_key_value_heads,
            tp_degree=self.tp_degree,
            dtype=self.torch_dtype,
            input_is_parallel=True,
        )
        self.num_heads = utils.divide(self.qkv_proj.get_num_attention_heads(), self.tp_degree)
        self.num_key_value_heads = utils.divide(self.qkv_proj.get_num_key_value_heads(), self.tp_degree)
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads

    def scaled_qk(self, Q, K, attention_mask):
        QK = torch.matmul(Q, K.transpose(2, 3)) / math.sqrt(self.head_dim)
        QK = torch.where(attention_mask, QK, torch.finfo(QK.dtype).min)
        return QK

    def prep_qkv_tensors(self, position_ids, hidden_states, past_key_value):
        """ take care of the shape, layout, group query, custom position encoding, etc. """
        Q, K, V = self.qkv_proj(hidden_states=hidden_states)

        # Divide hidden_dim across heads for MHA
        # Change layout: BSHD -> BHSD
        bsz, q_len, _ = hidden_states.size()
        Q = move_heads_front(Q, bsz, q_len, self.num_heads, self.head_dim)
        K = move_heads_front(K, bsz, q_len, self.num_key_value_heads, self.head_dim)
        V = move_heads_front(V, bsz, q_len, self.num_key_value_heads, self.head_dim)

        # Rotate Q and K
        cos, sin = self.rotary_emb(V, position_ids)
        Q, K = apply_rotary_pos_emb(Q, K, cos, sin)
        return Q, K, V

    def perform_prefill(self, Q, K, V, q_len, bsz, attention_mask) -> Tensor:
        """  attention computation at prefilling (context encoding) phase """
        K_active = repeat_kv(K, self.num_key_value_groups)
        V_active = repeat_kv(V, self.num_key_value_groups)

        # use flash attention if (i) sequence length is large enough to get the best performance,
        # (ii) Q, K, and V have the same shape. Conditions can be changed in the future.
        flash_attention_eligible = q_len >= 4096 and Q.shape == K_active.shape == V_active.shape

        if flash_attention_eligible:
            # if we are using left padding, then the bzs needs be 1 (otherwise we get wrong result
            # because flash attention does not use attention_mask). In practice, we use right
            # padding so this is unlikely to cause issues
            assert self.padding_side == "right" or bsz == 1

            # original shape of q, k, v is BHSD, and expected output is also BHSD.
            logging.debug(f"Using flash_fwd for Q.shape={Q.shape}")
            # make sure to cast inputs to self.config.torch_dtype (this is needed because the downcast to bf16
            # might happen after the kernel hlo creation step). Also convert shapes as expected by the kernel.
            Q = (
                Q.permute(0, 1, 3, 2)
                .reshape((bsz * self.num_heads, self.head_dim, q_len))
                .to(self.config.torch_dtype)
            )
            Q = Q / math.sqrt(self.head_dim)
            K_active = (
                K_active.permute(0, 1, 3, 2)
                .reshape((bsz * self.num_heads, self.head_dim, q_len))
                .to(self.config.torch_dtype)
            )
            V_active = V_active.reshape((bsz * self.num_heads, q_len, self.head_dim)).to(self.config.torch_dtype)
            attn_output = torch.zeros(bsz * self.num_heads, q_len, self.head_dim, dtype=Q.dtype, device=Q.device)
            _flash_fwd_call(
                Q, K_active, V_active, 1.0, attn_output, kernel_name="CausalAttentionMMSoftmaxMMWithoutSwap"
            )
            attn_output = attn_output.reshape((bsz, self.num_heads, q_len, self.head_dim))
        else:
            logging.debug(f"Not using flash_fwd for Q.shape={Q.shape}")
            active_scores = self.scaled_qk(Q, K_active, attention_mask)
            active_scores = nn.functional.softmax(active_scores, dim=-1, dtype=torch.float32).to(Q.dtype)
            attn_output = torch.matmul(active_scores, V_active)
        return attn_output

    def compute_for_token_gen(self, Q, K, V, position_ids, past_key_value, attention_mask, active_mask) -> Tensor:
        """ attention computation at token generation phase """
        is_speculation = position_ids.shape[-1] > 1

        # Attention computation: softmax((Q.K/√dkv) + mask).V
        # i. prior (cached) KV
        K_prior = past_key_value[0]
        V_prior = past_key_value[1]
        K_prior = repeat_kv(K_prior, self.num_key_value_groups)
        V_prior = repeat_kv(V_prior, self.num_key_value_groups)
        prior_scores = torch.matmul(Q, K_prior.transpose(2, 3)) / math.sqrt(self.head_dim)
        prior_scores = torch.where(attention_mask, prior_scores, torch.finfo(prior_scores.dtype).min)
        prior_scores = prior_scores.to(torch.float32)

        # ii. active (current/new) KV
        K_active = repeat_kv(K, self.num_key_value_groups)
        V_active = repeat_kv(V, self.num_key_value_groups)
        active_scores = torch.matmul(Q, K_active.transpose(2, 3)) / math.sqrt(self.head_dim)
        if is_speculation:
            active_scores = torch.where(active_mask, active_scores, torch.finfo(active_scores.dtype).min)
        active_scores = active_scores.to(torch.float32)

        # iii. attention scores
        softmax_prior, softmax_active = manual_softmax(prior_scores, active_scores, is_speculation)
        softmax_prior, softmax_active = softmax_prior.to(Q.dtype), softmax_active.to(Q.dtype)
        attn_prior = torch.matmul(softmax_prior, V_prior)
        attn_active = torch.matmul(softmax_active, V_active)
        attn_output = attn_prior + attn_active

        return attn_output

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            past_key_value: Optional[Tuple[torch.Tensor]] = None,
            active_mask: Optional[torch.LongTensor] = None,
    ) -> Tuple[Tensor, Optional[Tuple[Tensor, Tensor]]]:
        """ Implements each layer's forward pass for the attention block. """
        bsz, q_len, _ = hidden_states.size()
        Q, K, V = self.prep_qkv_tensors(position_ids, hidden_states, past_key_value)

        if past_key_value is None:
            attn_output = self.perform_prefill(Q, K, V, q_len, bsz, attention_mask)
        else:
            attn_output = self.compute_for_token_gen(Q, K, V, position_ids, past_key_value, attention_mask, active_mask)

        # transpose BHSD -> BSHD
        attn_output = attn_output.transpose(1, 2).contiguous()

        # merge multi head hidden
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        # Z = Z.Wo
        attn_output = self.o_proj(attn_output)

        past_key_value: Tuple[Tensor, Tensor] = (K, V)

        return attn_output, past_key_value
