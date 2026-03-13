import torch

from optimum.neuron.models.inference.backend.modules.attention.padding import get_aligned_head_dim, maybe_pad_head_dim


def test_maybe_pad_head_dim_no_regression_when_aligned():
    tensor = torch.arange(2 * 128 * 3, dtype=torch.float32).reshape(256, 3)
    padded = maybe_pad_head_dim(
        tensor,
        pad_dim=0,
        source_heads=2,
        source_head_dim=128,
        target_head_dim=128,
    )
    assert padded.shape == tensor.shape
    assert torch.equal(padded, tensor)
    assert get_aligned_head_dim(128) == 128


def test_maybe_pad_head_dim_pads_each_head_tail_with_zeros():
    tensor = torch.arange(2 * 96, dtype=torch.float32)
    padded = maybe_pad_head_dim(
        tensor,
        pad_dim=0,
        source_heads=2,
        source_head_dim=96,
        target_head_dim=128,
    ).view(2, 128)
    assert torch.equal(padded[:, :96], tensor.view(2, 96))
    assert torch.count_nonzero(padded[:, 96:]) == 0
