# NxD Hello World with llama-1B
# Model ref: https://github.com/meta-llama/llama3/blob/main/llama/model.py

import math
from typing import Optional, List
import torch
from torch import nn
import torch.nn.functional as F

from torch_neuronx.xla_impl.ops import RmsNorm
from neuronx_distributed.parallel_layers import parallel_state
from neuronx_distributed.parallel_layers.layers import (
    ColumnParallelLinear,
    ParallelEmbedding,
    RowParallelLinear,
)

from config import Config
    

def prefill_cache(cache: torch.tensor,                 
                  update: torch.tensor):

    bs, seq_len, n_kv_heads, head_dim = update.shape

    # CPU 
    indices = torch.arange(start=0, end=seq_len, device=cache.device)
    indices = indices.view(1, seq_len, 1, 1)
    indices = indices.expand(bs, seq_len, n_kv_heads, head_dim)

    return  torch.scatter(cache, 1, indices, update)


def precompute_rope(device,
                    theta,
                    head_dim, 
                    seq_len):
    # Refer: https://medium.com/@parulsharmmaa/understanding-rotary-positional-embedding-and-implementation-9f4ad8b03e32
    theta = 1.0 / (
            theta
            ** (torch.arange(0, head_dim, 2, device=device)[: (head_dim // 2)].float() / head_dim)
        )
    
    seq_idx = torch.arange(seq_len, dtype=torch.float32, device=device)

    # Outer product of theta and position index; output tensor has
    # a shape of [max_seq_len, dim // 2]
    idx_theta = torch.einsum("i, j -> ij", seq_idx, theta).float()

    # cache includes both the cos and sin components and so the output shape is
    # [max_seq_len, dim // 2, 2]
    return torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)


def rope(x, start_pos, cache):
    # input tensor has shape [b, s, n_h, h_d]
    bs, input_len, _, _ = x.shape
        
    # expand cache to batch size
    rope_cache = cache.unsqueeze(0).expand(bs, *cache.shape)

    if input_len == 1:
        # We are in decode mode, so we have a q & k of size 1. 
        # But the positions of these tokens are not necessarily the same, so 
        # we gather the RoPE cos and sine values by position id 
        index = start_pos.view(bs, 1, 1, 1).expand(bs, 1, cache.shape[-2], cache.shape[-1])
        rope_cache = torch.gather(rope_cache, dim=1, index=index.to(torch.int64))

    xshaped = x.float().reshape(*x.shape[:-1], -1, 2)

    # reshape the cache for broadcasting
    # tensor has shape [b, s, 1, h_d // 2, 2] if packed samples,
    # otherwise has shape [1, s, 1, h_d // 2, 2]
    rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)

    # tensor has shape [b, s, n_h, h_d // 2, 2]
    x_out = torch.stack(
        [
            xshaped[..., 0] * rope_cache[..., 0]
            - xshaped[..., 1] * rope_cache[..., 1],
            xshaped[..., 1] * rope_cache[..., 0]
            + xshaped[..., 0] * rope_cache[..., 1],
        ],
        -1,
    )

    # tensor has shape [b, s, n_h, h_d]
    x_out = x_out.flatten(3)
    return x_out.type_as(x)


class RMSNorm(nn.Module):
    """
    ref : https://github.com/meta-llama/llama3/blob/main/llama/model.py
    In a production usecase, from torch_neuronx.xla_impl.ops import RmsNorm
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.eps = cfg.rms_norm_eps
        self.weight = nn.Parameter(torch.ones(cfg.hidden_size, dtype=cfg.dtype))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        if x.is_cpu:
            output = self._norm(x.float()).type_as(x)
            return output * self.weight
        else:        
            return RmsNorm.apply(
                x, self.weight, self.eps, len(x.shape) - 1
            ).to(x.dtype)
    
class Attention(nn.Module):

    def __init__(self, cfg: Config, batch_size: int, seq_len: int):
        super().__init__()
                
        if parallel_state.model_parallel_is_initialized():
            
            tp_degree = parallel_state.get_tensor_model_parallel_group().size()
            
            if cfg.n_heads % tp_degree != 0:
                raise ValueError("n_heads not evenly divisible by tp degree")

            # we want atleast 1 kv head on a core
            self.n_heads = cfg.n_heads // tp_degree
            self.n_kv_heads = max(cfg.n_kv_heads // tp_degree, 1)             
            
            self.wq = ColumnParallelLinear(cfg.hidden_size, self.n_heads * tp_degree * cfg.head_dim, bias=False, gather_output = False, dtype=cfg.dtype)
            self.wk = ColumnParallelLinear(cfg.hidden_size, self.n_kv_heads * tp_degree * cfg.head_dim, bias=False, gather_output = False, dtype=cfg.dtype)
            self.wv = ColumnParallelLinear(cfg.hidden_size, self.n_kv_heads * tp_degree * cfg.head_dim, bias=False, gather_output = False, dtype=cfg.dtype)
            self.wo = RowParallelLinear(self.n_heads * tp_degree * cfg.head_dim, cfg.hidden_size, bias=False, input_is_parallel=True, dtype=cfg.dtype)
            
        else:
            self.n_heads = cfg.n_heads
            self.n_kv_heads = cfg.n_kv_heads

            self.wq = nn.Linear(cfg.hidden_size, self.n_heads * cfg.head_dim, bias=False, dtype=cfg.dtype)
            self.wk = nn.Linear(cfg.hidden_size, self.n_kv_heads * cfg.head_dim, bias=False, dtype=cfg.dtype)
            self.wv = nn.Linear(cfg.hidden_size, self.n_kv_heads * cfg.head_dim, bias=False, dtype=cfg.dtype)
            self.wo = nn.Linear(self.n_heads * cfg.head_dim, cfg.hidden_size, bias=False, dtype=cfg.dtype)

        # KV Caches
        # On NxD, your caches need to be registered parameters. Only parameters
        # can be aliased. Note, you cannot use `register_buffer` as well. 
        #
        # What is aliasing?
        # When an input and output buffer is aliased, the buffer is reused for
        # the output - we do not create new buffers for output. Aliased buffers
        # stay on device and are not returned to CPU like the output tensors.
        self.cache_k = nn.Parameter(torch.zeros((batch_size, seq_len, self.n_kv_heads, cfg.head_dim), dtype=cfg.dtype), requires_grad=False)
        self.cache_v = nn.Parameter(torch.zeros((batch_size, seq_len, self.n_kv_heads, cfg.head_dim), dtype=cfg.dtype), requires_grad=False)

        self.n_rep = self.n_heads // self.n_kv_heads
        self.head_dim = cfg.head_dim

    def forward(self,
                x: torch.Tensor,
                last_pos: torch.Tensor,
                mask: Optional[torch.Tensor],
                rope_cache: torch.tensor):
        # x (Batch, Sequence, Hidden)
        bsz, inp_len, hidden_dim = x.shape

        # BSH
        q, k, v = self.wq(x), self.wk(x), self.wv(x)

        # BSNH
        q = q.view(bsz, inp_len, self.n_heads, self.head_dim)
        k = k.view(bsz, inp_len, self.n_kv_heads, self.head_dim)
        v = v.view(bsz, inp_len, self.n_kv_heads, self.head_dim)

        if inp_len > 1:
            start_pos = torch.zeros_like(last_pos)
            q, k = rope(q, start_pos, rope_cache), rope(k, start_pos, rope_cache)
        else:
            start_pos = last_pos
            q, k = rope(q, start_pos, rope_cache), rope(k, start_pos, rope_cache)

        # Save KV Cache
        if inp_len > 1:
            indices = torch.arange(start=0, end=inp_len, dtype=torch.int64, device=q.device)
            indices = indices.view(1, inp_len, 1, 1)
            indices = indices.expand(bsz, inp_len, self.n_kv_heads, self.head_dim)
        else:
            indices = last_pos.view(bsz, 1, 1, 1).expand_as(k).to(torch.int64)
            
        updated_kcache = torch.scatter(self.cache_k, 1, indices, k)
        updated_vcache = torch.scatter(self.cache_v, 1, indices, v)

        if q.is_cpu:
            # CPU flow and XLA flow keeps a reference to the cache differently
            # On CPU we change the cache to point to the updated cache. 
            # On XLA we expect aliasing to do this in place on device. 
            self.cache_k.data = updated_kcache
            self.cache_v.data = updated_vcache

        # Note: We cannot just slice the cache to the current position. If we slice, we would change the  
        # compute shape for every decode run. On Neuron we compile for fixed shapes. So the alternative is to 
        # operate on a fixed sequence length per compilation. In this example we compute the attention 
        # for the full preallocated sequence length. We just 'read' the output at the right index. 
 
        # keys = self.cache_k[:bsz, : start_pos + inp_len]  X wrong
        # values = self.cache_v[:bsz, : start_pos + ]       X wrong

        # Yes fixed shape will cause us to waste compute. The way to work around that is to 'bucket' - compile for many
        # shapes. One way is to bucket along sequence length. Here we can slice the KV cache when you bucket along sequence length. 
        # This is an easy optimization you could do. Note, this example does not bucket. 
        keys = updated_kcache
        values = updated_vcache
        
        # With GQA, k/v heads are shared amond different q heads
        # repeat k/v heads to match q heads
        keys = torch.repeat_interleave(keys, dim=2, repeats=self.n_rep)
        values = torch.repeat_interleave(values, dim=2, repeats=self.n_rep)
        
        # bs, seqlen, head, head_dim -> bs, head, seqlen, head_dim
        q = q.transpose(1, 2) 
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)

        # Q.K^T/√d
        scores = torch.matmul(q, keys.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores + mask
        scores = F.softmax(scores.float(), dim=-1).type_as(q)
        output = torch.matmul(scores, values)
        output = output.transpose(1, 2).contiguous().view(bsz, inp_len, -1)

        # Note : As of 2.21.1, Aliased buffers on NxD has to be returned as an
        # output tensor. This is to comply with the XLA's aliasing, which expects
        # aliased input and output tensors to be alised.
        #
        # On NxD, we want to trace & compile the Model forward(). So all cache
        # buffers are passed all the way back for Model.forward() to return. 
        #
        # Planned Improvement: NxD is working on Auto-Aliasing which will 
        # remove the need to return aliased buffers simplifying development. 
        
        # return self.wo(output) 
        return self.wo(output), updated_kcache, updated_vcache


class MLP(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        if parallel_state.model_parallel_is_initialized():
            self.gate_proj = ColumnParallelLinear(cfg.hidden_size, cfg.intermediate_size, bias=False, gather_output=False, dtype=cfg.dtype)
            self.up_proj = ColumnParallelLinear(cfg.hidden_size, cfg.intermediate_size, bias=False, gather_output=False, dtype=cfg.dtype)
            self.down_proj = RowParallelLinear(cfg.intermediate_size, cfg.hidden_size, bias=False, input_is_parallel=True, dtype=cfg.dtype)
        else:
            self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False, dtype=cfg.dtype)
            self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False, dtype=cfg.dtype)
            self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False, dtype=cfg.dtype)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(torch.nn.Module):
    def __init__(self, cfg: Config, batch_size: int, seq_len: int):
        super().__init__()
        self.attention = Attention(cfg, batch_size, seq_len)
        self.mlp = MLP(cfg)
        self.attention_norm = RMSNorm(cfg)
        self.mlp_norm = RMSNorm(cfg)

    def forward(
        self,
        x: torch.Tensor,
        last_pos: torch.Tensor,
        mask: torch.Tensor,
        rope_cache: torch.Tensor,
    ):
      
        norm_h = self.attention_norm(x)
        attn_h, cache_k, cache_v = self.attention(norm_h, last_pos, mask, rope_cache)
        attn_h = x + attn_h
        
        norm_h = self.mlp_norm(attn_h)
        mlp_h = self.mlp(self.mlp_norm(norm_h))
        out = attn_h + mlp_h
        
        # Note: Relaying the cache buffers to Transformer.forward() as 
        # we want to return them as output tensors. 

        # return out
        return out, cache_k, cache_v


class Transformer(torch.nn.Module):

    def __init__(self, cfg : Config, batch_size: int, seq_len: int):
        super().__init__()

        if parallel_state.model_parallel_is_initialized():
            self.embedding = ParallelEmbedding(cfg.vocab_size,
                                               cfg.hidden_size, 
                                               shard_across_embedding=True, 
                                               dtype=cfg.dtype)
            self.output = ColumnParallelLinear(cfg.hidden_size,
                                               cfg.vocab_size,
                                               bias=False, 
                                               gather_output=True, 
                                               dtype=cfg.dtype)
        else:
            self.embedding = torch.nn.Embedding(cfg.vocab_size,
                                                cfg.hidden_size,
                                                cfg.pad_token, 
                                                dtype=cfg.dtype)
            self.output = nn.Linear(cfg.hidden_size,
                                    cfg.vocab_size,
                                    bias=False,
                                    dtype=cfg.dtype)

        self.layers = torch.nn.ModuleList()
        for _ in range(cfg.n_layers):
            self.layers.append(TransformerBlock(cfg, 
                                                batch_size=batch_size,
                                                seq_len=seq_len))

        self.bs = batch_size
        self.seq_len = seq_len

        self.norm = RMSNorm(cfg)
        self.rope_theta = cfg.rope_theta
        self.head_dim = cfg.head_dim
        self.hidden_size = cfg.hidden_size

    def forward(self,
                tokens: torch.Tensor,
                last_pos: torch.Tensor):

        _bsz, input_len = tokens.shape
        h = self.embedding(tokens)
        
        self.rope_cache = precompute_rope(device = h.device, 
                                          theta=self.rope_theta, 
                                          head_dim=self.head_dim, 
                                          seq_len=self.seq_len)

        mask = None
        if input_len > 1:
            # You cannot use -inf well on Neuron, you will run into 1003 errors
            mask = torch.full((input_len, input_len), torch.finfo(h.dtype).min)
            mask = torch.triu(mask, diagonal=1)

        k_caches = []
        v_caches = []

        for layer in self.layers:
            h, cache_k, cache_v= layer(h, last_pos, mask, self.rope_cache)
            k_caches.append(cache_k)
            v_caches.append(cache_v)

        h = self.norm(h)
        output = self.output(h).float()
        
        # We return the logits for the last token per batch. 
        # This is a simple optimization to stop moving sequence length long 
        # logits back from device to CPU for prefil. 
        if input_len > 1:
            last_pos = last_pos.view(self.bs, 1, 1).expand(self.bs, 1, self.hidden_size)
            output = torch.gather(output, dim=1, index=last_pos.to(torch.int64)) 
        # Note: We are returning K and V caches. The order in which the tensors
        # are returned is important as you will need to register the alias when 
        # tracing the model.                 
        
        if output.is_cpu:
            return output
        else:
            return output, *k_caches, *v_caches


def load_llama_checkpoint(cfg: Config,
                          model_path: str, 
                          tp_degree = 1):

    # Download model from : https://www.llama.com/llama-downloads/
    state_dict = torch.load(model_path,
                            map_location=torch.device('cpu'),
                            weights_only=True)

    # Why do we change the weight keys?
    # The modeling code does not exactly match meta's llama code. This just
    # corrects the state dict so it can be loaded to the modeling code above.

    def replace(state_dict, old_key, new_key):
        return {k.replace(old_key, new_key): v for k, v in state_dict.items()}

    state_dict = replace(state_dict, 'tok_embeddings', 'embedding')
    state_dict = replace(state_dict, 'feed_forward', 'mlp')
    state_dict = replace(state_dict, 'ffn_norm', 'mlp_norm')
    state_dict = replace(state_dict, 'mlp.w1', 'mlp.gate_proj')
    state_dict = replace(state_dict, 'mlp.w3', 'mlp.up_proj')
    state_dict = replace(state_dict, 'mlp.w2', 'mlp.down_proj')

    # The embedding and modeling head outputs are tied. We just close because 
    # model builder's sharding logic does not take care of tied weights yet. 
    state_dict['output.weight'] = state_dict['embedding.weight'].clone().detach()
    
    # We need to repeat KV heads to get atleast 1 KV head on one core
    if tp_degree > 1:
        n_repeat = tp_degree // cfg.n_kv_heads
        for lay in range(cfg.n_layers):
            wk = state_dict[f'layers.{lay}.attention.wk.weight']
            wk = torch.repeat_interleave(wk.view(cfg.n_kv_heads, cfg.head_dim, cfg.hidden_size), dim=0, repeats=n_repeat)
            state_dict[f'layers.{lay}.attention.wk.weight'] = wk.flatten(0,1)
            wv = state_dict[f'layers.{lay}.attention.wv.weight']
            wv = torch.repeat_interleave(wv.view(cfg.n_kv_heads, cfg.head_dim, cfg.hidden_size), dim=0, repeats=n_repeat)
            state_dict[f'layers.{lay}.attention.wv.weight'] = wv.flatten(0,1)

    return state_dict
