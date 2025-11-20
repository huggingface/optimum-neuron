# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

import math

import numpy as np
import torch
import torch.distributed as dist
from optimum.utils import logging
from torch.utils.data import Dataset
from torch.utils.data.distributed import DistributedSampler

from ..utils import is_precompilation


logger = logging.get_logger()

TRL_VERSION = "0.24.0"

def pad(
    tensors: list[torch.Tensor],
    padding_value: int = 0,
    padding_side: str = "right",
    max_length: int | None = None,
  ) -> torch.Tensor:
    """
    Pads a list of tensors to the same shape along the first dimension.
    It differs from `trl` by enfoncing the same sequence length for all tensors, which is required to avoid
    recompilation.
    """
    batch_size = len(tensors)
    if max_length is None:
        max_length = np.max([t.shape[0] for t in tensors]).tolist()

    output_shape = (max_length,) + tensors[0].shape[1:]

    # Create an output tensor filled with the padding value
    output = torch.full((batch_size, *output_shape), padding_value, dtype=tensors[0].dtype, device=tensors[0].device)

    for i, t in enumerate(tensors):
        if padding_side == "left":
            seq_start = output_shape[0] - t.shape[0]
        elif padding_side == "right":
            seq_start = 0
        else:
            raise ValueError("padding_side must be 'left' or 'right'")

        # Define the slices
        seq_slice = slice(seq_start, seq_start + t.shape[0])
        slices = (seq_slice,) + tuple(slice(0, s) for s in t.shape[1:])
        output[i][slices] = t

    return output


def neuron_parallel_compile_tokenizer_decoder_method(
    self,
    token_ids: int | list[int],
    skip_special_tokens: bool = False,
    clean_up_tokenization_spaces: bool | None = None,
    **kwargs,
) -> str:
    """
    Patched `tokenizer._decode` method for `neuron_parallel_compile`.
    This is needed because any tensor operation on the XLA device during `neuron_parallel_compile` produces rubbish
    results, which is not an issue in general, but causes failure when the token IDS end up being out of range for the
    tokenizer vocabulary.
    """
    if not is_precompilation():
        raise RuntimeError("This patch method should only be used with `neuron_parallel_compile`.")

    # We log the token IDs to force the data mouvement to CPU, which would happen during actual decoding.
    logger.debug("Using patched tokenizer.decode method for Neuron parallel compilation, token_ids = ", token_ids)

    # Returns a dummy string, we do not care about the value in this context.
    return "dummy"


def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    XLA-compatible version of nanmin that doesn't use dynamic indexing.
    Compute the minimum value of a tensor, ignoring NaNs.
    """
    mask = torch.isnan(tensor)
    if mask.all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    filled = torch.where(mask, torch.tensor(float("inf"), dtype=tensor.dtype, device=tensor.device), tensor)
    return torch.min(filled)


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    XLA-compatible version of nanmax that doesn't use dynamic indexing.
    Compute the maximum value of a tensor, ignoring NaNs.
    """
    mask = torch.isnan(tensor)
    if mask.all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    filled = torch.where(mask, torch.tensor(float("-inf"), dtype=tensor.dtype, device=tensor.device), tensor)
    return torch.max(filled)


def nanstd(tensor: torch.Tensor) -> torch.Tensor:
    """
    XLA-compatible version of nanstd.
    Compute the standard deviation of a tensor, ignoring NaNs.
    """
    # Use torch's built-in nanmean and compute variance with Bessel's correction
    variance = torch.nanmean((tensor - torch.nanmean(tensor, keepdim=True)) ** 2)
    count = torch.sum(~torch.isnan(tensor))
    variance *= count / (count - 1).clamp(min=1.0)  # Bessel's correction, avoid division by zero
    return torch.sqrt(variance)


class DistributedRepeatSampler(DistributedSampler):
    """
    Sampler that repeats the indices of a dataset in a structured manner.
    Same as `trl.trainer.utils.RepeatSampler` but adapted to work with distributed training.

    To implement it, we simply combine the logic from https://github.com/pytorch/pytorch/blob/main/torch/utils/data/distributed.py
    with the logic from https://github.com/huggingface/trl/blob/main/trl/trainer/utils.py#L1692.

    First, we distribute the dataset indices across the different ranks, then we repeat the indices on each rank.

    We inherit from `torch.utils.data.DistributedSampler` even though we override all of its methods to pass the checks
    "isinstance(sampler, DistributedSampler)" done in `torch.utils.data.DataLoader` when using distributed training.
    """

    def __init__(
        self,
        dataset: Dataset,
        mini_repeat_count: int,
        batch_size: int = 1,
        repeat_count: int = 1,
        num_replicas: int | None = None,
        rank: int | None = None,
        shuffle: bool = True,
        seed: int = 0,
        drop_last: bool = False,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore[arg-type]
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore[arg-type]
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore[arg-type]
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count
        self.shuffle = shuffle

        if shuffle:
            self.generator = torch.Generator()  # Create a local random generator
            self.generator.manual_seed(seed)

    def __iter__(self):
        # First, we produce indices for each rank.
        # That is the distributed part of the sampler.
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()  # type: ignore[arg-type]
        else:
            indices = list(range(len(self.dataset)))  # type: ignore[arg-type]

        if not self.drop_last:
            # add extra samples to make it evenly divisible
            padding_size = self.total_size - len(indices)
            if padding_size <= len(indices):
                indices += indices[:padding_size]
            else:
                indices += (indices * math.ceil(padding_size / len(indices)))[
                    :padding_size
                ]
        else:
            # remove tail of data to make it evenly divisible.
            indices = indices[: self.total_size]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank : self.total_size : self.num_replicas]
        assert len(indices) == self.num_samples

        # Second, we repeat the indices on each rank.
        # This is the non-distributed part of the sampler.
        #    [2, 4, 3, 1, 0, 6, 5]
        # -> [[2, 4, 3], [1, 0, 6], [5]]  (batch_size = 3)
        indices = [indices[i : i + self.batch_size] for i in range(0, len(indices), self.batch_size)]

        #    [[2, 4, 3], [1, 0, 6], [5]]
        # -> [[2, 4, 3], [1, 0, 6]]
        indices = [chunk for chunk in indices if len(chunk) == self.batch_size]

        for chunk in indices:
            for _ in range(self.repeat_count):
                for index in chunk:
                    for _ in range(self.mini_repeat_count):
                        yield index

    def __len__(self) -> int:
        return (self.num_samples // self.batch_size) * self.batch_size * self.mini_repeat_count * self.repeat_count

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

