from dataclasses import dataclass

import torch


@dataclass
class Config():
    vocab_size: int
    hidden_size: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    intermediate_size: int
    dtype: torch.dtype = torch.float32
    rms_norm_eps: int = 1e-05
    rope_theta: int = 500000.0
    pad_token: int = 0

# LLama 1B
Llama1B = Config(vocab_size = 128256,
                hidden_size = 2048,
                n_layers = 16,
                n_heads = 32,
                n_kv_heads = 8,
                head_dim = 64,
                intermediate_size = 8192,
                dtype = torch.bfloat16,
                rms_norm_eps = 1e-05,
                rope_theta = 500000.0,
                pad_token = 128004
            )
