import pytest
import torch

from optimum.neuron.generation.logits_process import FusedLogitsWarper


def test_temperature():
    logits = torch.rand([10, 10, 10])
    temperature = 0.9
    warper = FusedLogitsWarper(temperature=temperature)
    warped_logits, warped_indices = warper(logits)
    assert warped_indices is None
    assert torch.allclose(warped_logits * temperature, logits)


def shuffle_logits(logits):
    shuffled_logits = torch.empty_like(logits)
    batch_size, vocab_size = logits.shape
    for i in range(batch_size):
        shuffled_indices = torch.randperm(vocab_size)
        shuffled_logits[i] = logits[i, shuffled_indices]
    return shuffled_logits


@pytest.mark.parametrize("batch_size", [1, 2, 10])
@pytest.mark.parametrize("vocab_size", [10, 1000, 50000])
def test_top_k(batch_size, vocab_size):
    # Create sorted logits by descending order for easier comparison, then shuffle them
    sorted_logits = (
        torch.arange(start=vocab_size, end=0, step=-1, dtype=torch.float).unsqueeze(0).repeat(batch_size, 1)
    )
    shuffled_logits = shuffle_logits(sorted_logits)

    top_k = vocab_size // 2

    warper = FusedLogitsWarper(top_k=top_k)

    filtered_logits, indices = warper(shuffled_logits)

    assert filtered_logits.shape[-1] == top_k
    assert indices.shape[-1] == top_k

    for i in range(batch_size):
        # Verify indices are correct
        assert torch.equal(shuffled_logits[i, indices[i]], filtered_logits[i])
        # Since the original logits were sorted, filtered logits should match the start of the sequence
        assert torch.equal(filtered_logits[i], sorted_logits[i, :top_k])


@pytest.mark.parametrize("batch_size", [1, 2, 10])
@pytest.mark.parametrize("vocab_size", [30, 1000, 50000])
def test_top_p(batch_size, vocab_size):
    # Create normalized logits
    norm_logits = torch.zeros(batch_size, vocab_size, dtype=torch.float)
    # We have 4 buckets, each corresponding to 0.25 of the total weights
    # With populations corresponding to 0.4, 0.3, 0.2 and 0.1 percent of the vocab_size
    buckets = [0.5, 0.2, 0.2, 0.1]
    bucket_weight = 1.0 / len(buckets)
    index = 0
    for bucket in buckets:
        bucket_size = int(bucket * vocab_size)
        norm_logits[:, index : index + bucket_size] = bucket_weight / bucket_size
        index += bucket_size
    # Sanity check: the sum of the normalized logits should be one
    assert torch.allclose(torch.sum(norm_logits, axis=-1), torch.ones(batch_size))

    # The first bucket cumulated sum is 0.25: set top_p to 75 % to exclude it
    warper = FusedLogitsWarper(top_p=0.75)

    # top_p will apply a softmax, so we need to take the log of our normalized logits
    sorted_logits = torch.log(norm_logits)
    shuffled_logits = shuffle_logits(sorted_logits)

    filtered_logits, indices = warper(shuffled_logits)

    # We expect all logits but the first bucket
    expected_n_logits = int((1.0 - buckets[0]) * vocab_size)
    assert filtered_logits.shape[-1] == expected_n_logits
    assert indices.shape[-1] == expected_n_logits

    for i in range(batch_size):
        # Verify indices are correct
        assert torch.equal(shuffled_logits[i, indices[i]], filtered_logits[i])
        # Since the original logits were sorted, filtered logits should match the end of the sequence
        assert torch.equal(filtered_logits[i], sorted_logits[i, -expected_n_logits:])


def test_top_k_top_p():
    warper = FusedLogitsWarper(top_k=3, top_p=0.8)

    # Prepare logits with normalized top-3, with distributions
    # so that cumulative prob > top_p requires 3, 2, and 1 logits resp.
    norm_top3_logits = torch.tensor(
        [[0.01, 0.01, 0.25, 0.25, 0.5], [0.01, 0.01, 0.2, 0.2, 0.6], [0.01, 0.01, 0.1, 0.1, 0.8]]
    )

    # Top_p will apply a softmax, so take the log
    sorted_logits = torch.log(norm_top3_logits)
    shuffled_logits = shuffle_logits(sorted_logits)

    filtered_logits, indices = warper(shuffled_logits)

    assert filtered_logits.shape[-1] == 3
    assert torch.all(filtered_logits[0, :] == sorted_logits[0, -3:])
    assert filtered_logits[1, 0] == float("-Inf")
    assert torch.all(filtered_logits[1, 1:] == sorted_logits[1, -2:])
    assert torch.all(filtered_logits[2, :2] == float("-Inf"))
    assert filtered_logits[2, -1] == sorted_logits[2, -1]
