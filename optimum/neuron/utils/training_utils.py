# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Training utilities"""

import contextlib
import functools
import os

import torch
from torch.utils._pytree import tree_map
from torch.utils.data import DataLoader, Dataset, IterableDataset


class FirstAndLastDataset(Dataset):
    def __init__(self, dataloader: DataLoader, num_repeat: int = 10, gradient_accumulation_steps: int = 1):
        self.dataloader = dataloader
        self.num_repeat = num_repeat * gradient_accumulation_steps
        self.samples = self.create_samples()

    def _create_samples_for_map_style_dataset(self):
        samples = []
        num_samples = len(self.dataloader.dataset)
        batch_size = self.dataloader.batch_size
        if batch_size is None and self.dataloader.batch_sampler is not None:
            batch_size = self.dataloader.batch_sampler.batch_size

        # TODO: validate that.
        if batch_size is None:
            samples = [self.dataloader.dataset[0]] * self.num_repeat + [self.dataloader.dataset[-1]] * self.num_repeat
            return samples

        num_batches = num_samples // batch_size
        remaining = num_samples % batch_size

        iterator = iter(self.dataloader)
        first_batch = next(iterator)
        samples = [first_batch] * self.num_repeat

        if num_batches >= 1 and remaining != 0:

            def map_fn(example):
                if isinstance(example, torch.Tensor):
                    return example[:remaining]
                else:
                    return example

            last_batch = tree_map(map_fn, first_batch)
            samples += [last_batch] * self.num_repeat

        return samples

    def _create_samples_for_iterable_dataset(self):
        # Will not work if the iterable dataset yields dynamic batch sizes.
        iterator = iter(self.dataloader)
        first_batch = next(iterator)
        samples = [first_batch] * self.num_repeat
        yield first_batch
        last_batch = None
        while True:
            try:
                last_batch = next(iterator)
            except StopIteration:
                if last_batch is not None:
                    samples += [last_batch] * self.num_repeat
                break
        return samples

    def create_samples(self):
        if isinstance(self.dataloader.dataset, IterableDataset):
            return self._create_samples_for_iterable_dataset()
        else:
            return self._create_samples_for_map_style_dataset()

    def __getitem__(self, idx: int):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


orig_finfo = torch.finfo


def patched_finfo(dtype):
    if dtype is torch.float32:
        return orig_finfo(torch.bfloat16)
    return orig_finfo(dtype)


class Patcher:
    def __enter__(self):
        torch.finfo = patched_finfo

    def __exit__(self, exc_type, exc_value, traceback):
        torch.finfo = orig_finfo


def patch_forward(forward_fn):
    @functools.wraps(forward_fn)
    def wrapper(*args, **kwargs):
        with Patcher():
            return forward_fn(*args, **kwargs)

    return wrapper


def patch_model(model):
    if hasattr(model.config, "layerdrop"):
        model.config.layerdrop = 0
    model.no_sync = lambda: contextlib.nullcontext()
    return model


def prepare_environment_for_neuron():
    # Set compiler flag to compile for transformer model type
    os.environ["NEURON_CC_FLAGS"] = os.environ.get("NEURON_CC_FLAGS", "") + " --model-type=transformer"


def patch_transformers_for_neuron_sdk():
    # TODO: does nothing for now but might needed to patch functions in case of bug.
    pass
