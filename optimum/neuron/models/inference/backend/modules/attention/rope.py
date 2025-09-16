"""Utility functions for neuronx_distributed.modules.attention

This module contains helper functions for the attention mechanisms,
particularly the RoPE (Rotary Position Embeddings) implementation
with frequency scaling techniques. These utilities enable extension
of effective context length beyond the original training window.
"""
# Adapted from https://github.com/aws-neuron/neuronx-distributed/blob/main/src/neuronx_distributed/modules/attention/utils.py

import math

import torch
from transformers import PretrainedConfig

from ...config import NxDNeuronConfig


def apply_scaling(freqs: torch.Tensor, config: PretrainedConfig):
    rope_scaling = getattr(config, "rope_scaling", None)
    assert rope_scaling is not None, "rope_scaling must be defined in the config to apply scaling"
    original_max_position_embeddings = rope_scaling.get("original_max_position_embeddings", None)
    assert original_max_position_embeddings is not None, (
        "original_max_position_embeddings must be defined in rope_scaling to apply scaling"
    )
    low_freq_factor = rope_scaling.get("low_freq_factor", None)
    assert low_freq_factor is not None, "low_freq_factor must be defined in rope_scaling to apply scaling"
    high_freq_factor = rope_scaling.get("high_freq_factor", None)
    assert high_freq_factor is not None, "high_freq_factor must be defined in rope_scaling to apply scaling"
    factor = rope_scaling.get("factor", None)
    assert factor is not None, "factor must be defined in rope_scaling to apply scaling"
    low_freq_wavelen = original_max_position_embeddings / low_freq_factor
    high_freq_wavelen = original_max_position_embeddings / high_freq_factor
    new_freqs = []
    for freq in freqs:
        wavelen = 2 * math.pi / freq
        if wavelen < high_freq_wavelen:
            new_freqs.append(freq)
        elif wavelen > low_freq_wavelen:
            new_freqs.append(freq / factor)
        else:
            assert low_freq_wavelen != high_freq_wavelen
            smooth = (original_max_position_embeddings / wavelen - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            new_freqs.append((1 - smooth) * freq / factor + smooth * freq)
    return torch.tensor(new_freqs, dtype=freqs.dtype, device=freqs.device)


def precompute_freqs_cis(config: PretrainedConfig, neuron_config: NxDNeuronConfig):
    head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
    end = neuron_config.max_context_length * 2
    freqs = 1.0 / (config.rope_theta ** (torch.arange(0, head_dim, 2)[: (head_dim // 2)].float() / head_dim))
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    if getattr(config, "rope_scaling", None) is not None:
        freqs = apply_scaling(freqs, config)
    freqs = torch.outer(t, freqs)
    return freqs


def apply_rotary_polar_compatible(query, key, freqs_cis):
    # Ensure freqs_cis is in FP32 for accuracy
    if freqs_cis.dtype != torch.float32:
        raise ValueError(f"Expect freqs_cis.dtype == torch.float32 to ensure accuracy, got {freqs_cis.dtype}")

    freqs_cis_real = freqs_cis.cos().unsqueeze(2)
    freqs_cis_imag = freqs_cis.sin().unsqueeze(2)

    def rotate(input):
        real = input[..., ::2]
        imag = input[..., 1::2]

        # For complex multiplication
        # (a + ib) * (c + id) = (ac - bd) + i(ad + bc)

        # ac - bd
        rot_real = (real * freqs_cis_real) - (imag * freqs_cis_imag)

        # ad + bc
        rot_imag = (real * freqs_cis_imag) + (freqs_cis_real * imag)

        return torch.cat([rot_real.unsqueeze(-1), rot_imag.unsqueeze(-1)], dim=-1).reshape(input.shape)

    query_rot = rotate(query)
    key_rot = rotate(key)

    return query_rot.type_as(query), key_rot.type_as(key)
