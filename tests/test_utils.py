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
# limitations under the License.
"""Tests for utility functions and classes."""

from typing import Dict, Literal, Union
from unittest import TestCase

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import BertConfig, BertForSequenceClassification, PreTrainedModel, Wav2Vec2Config, Wav2Vec2Model

from optimum.neuron.accelerate.accelerator import MODEL_PATCHING_SPECS
from optimum.neuron.utils import ModelPatcher
from optimum.neuron.utils.testing_utils import is_trainium_test
from optimum.neuron.utils.training_utils import FirstAndLastDataset, is_model_officially_supported


@is_trainium_test
def test_is_model_officially_supported():
    class DummyModelClass(PreTrainedModel):
        pass

    unsupported_model = DummyModelClass(BertConfig())
    assert is_model_officially_supported(unsupported_model) is False

    class Child(BertForSequenceClassification):
        pass

    child_model = Child(BertConfig())
    assert is_model_officially_supported(child_model) is False

    bert_model = BertForSequenceClassification(BertConfig())
    assert is_model_officially_supported(bert_model) is True


class FirstAndLastDatasetTest(TestCase):
    def _create_dataset(self, num_samples: int, dataset_type: Union[Literal["map"], Literal["iterable"]]) -> Dataset:
        random_sample = {"my_sample": torch.rand(4, 3, 24, 24)}

        class MapStyle(Dataset):
            def __init__(self, num_samples: int):
                self.num_samples = num_samples

            def __getitem__(self, key) -> Dict[str, torch.Tensor]:
                return random_sample

            def __len__(self) -> int:
                return self.num_samples

        class IterableStyle(IterableDataset):
            def __init__(self, num_samples: int):
                self.num_samples = num_samples

            def __iter__(self):
                count = 0
                while count < self.num_samples:
                    yield random_sample
                    count += 1

        dataset_class = MapStyle if dataset_type == "map" else IterableStyle
        return dataset_class(num_samples)

    def test_map_style_dataset(self):
        batch_size = 16
        gradient_accumulation_steps = 4
        world_size = 2
        non_divisible_num_samples = batch_size * 200 + 1
        divisible_num_samples = batch_size * 200
        num_repeat = 10

        # Case 1: the batch size does not divide the number of samples.
        dataloader = DataLoader(self._create_dataset(non_divisible_num_samples, "map"), batch_size=batch_size)
        first_and_last = FirstAndLastDataset(dataloader, num_repeat=num_repeat)
        self.assertEqual(len(first_and_last), num_repeat * 2)

        # Case 2: the batch size divides the number of samples.
        dataloader = DataLoader(self._create_dataset(divisible_num_samples, "map"), batch_size=batch_size)
        first_and_last = FirstAndLastDataset(dataloader, num_repeat=num_repeat)
        self.assertEqual(len(first_and_last), num_repeat)

        # Case 3: the batch size does not divide the number of samples and we have gradient accumulation / multiple processes.
        dataloader = DataLoader(self._create_dataset(non_divisible_num_samples, "map"), batch_size=batch_size)
        first_and_last = FirstAndLastDataset(
            dataloader,
            num_repeat=num_repeat,
            gradient_accumulation_steps=gradient_accumulation_steps,
            world_size=world_size,
        )
        self.assertEqual(len(first_and_last) / (gradient_accumulation_steps * world_size), num_repeat * 2)

        # Case 4: the batch size divides the number of samples and we have gradient accumulation / multiple processes.
        dataloader = DataLoader(self._create_dataset(divisible_num_samples, "map"), batch_size=batch_size)
        first_and_last = FirstAndLastDataset(
            dataloader,
            num_repeat=num_repeat,
            gradient_accumulation_steps=gradient_accumulation_steps,
            world_size=world_size,
        )
        self.assertEqual(len(first_and_last) / (gradient_accumulation_steps * world_size), num_repeat)

    def test_iterable_style_dataset(self):
        batch_size = 16
        gradient_accumulation_steps = 4
        world_size = 2
        non_divisible_num_samples = batch_size * 200 + 1
        divisible_num_samples = batch_size * 200
        num_repeat = 10

        # Case 1: the batch size does not divide the number of samples.
        dataloader = DataLoader(self._create_dataset(non_divisible_num_samples, "iterable"), batch_size=batch_size)
        first_and_last = FirstAndLastDataset(dataloader, num_repeat=num_repeat)
        self.assertEqual(len(first_and_last), num_repeat * 2)

        # Case 2: the batch size divides the number of samples.
        dataloader = DataLoader(self._create_dataset(divisible_num_samples, "iterable"), batch_size=batch_size)
        first_and_last = FirstAndLastDataset(dataloader, num_repeat=num_repeat)
        self.assertEqual(len(first_and_last), num_repeat * 2)

        # Case 3: the batch size does not divide the number of samples and we have gradient accumulation / multiple processes.
        dataloader = DataLoader(self._create_dataset(non_divisible_num_samples, "iterable"), batch_size=batch_size)
        first_and_last = FirstAndLastDataset(
            dataloader,
            num_repeat=num_repeat,
            gradient_accumulation_steps=gradient_accumulation_steps,
            world_size=world_size,
        )
        self.assertEqual(len(first_and_last) / (gradient_accumulation_steps * world_size), num_repeat * 2)

        # Case 4: the batch size divides the number of samples and we have gradient accumulation / multiple processes.
        dataloader = DataLoader(self._create_dataset(divisible_num_samples, "iterable"), batch_size=batch_size)
        first_and_last = FirstAndLastDataset(
            dataloader,
            num_repeat=num_repeat,
            gradient_accumulation_steps=gradient_accumulation_steps,
            world_size=world_size,
        )
        self.assertEqual(len(first_and_last) / (gradient_accumulation_steps * world_size), num_repeat * 2)

        # Case 5: only one batch.
        dataloader = DataLoader(self._create_dataset(batch_size, "iterable"), batch_size=batch_size)
        first_and_last = FirstAndLastDataset(dataloader, num_repeat=num_repeat)
        self.assertEqual(len(first_and_last), num_repeat)

        # Case 6: only one batch with gradient accumulation / multiple processes.
        dataloader = DataLoader(self._create_dataset(batch_size, "iterable"), batch_size=batch_size)
        first_and_last = FirstAndLastDataset(
            dataloader,
            num_repeat=num_repeat,
            gradient_accumulation_steps=gradient_accumulation_steps,
            world_size=world_size,
        )
        self.assertEqual(len(first_and_last) / (gradient_accumulation_steps * world_size), num_repeat)


def test_patch_model():
    bert_model = BertForSequenceClassification(BertConfig())
    patching_specs = []
    for spec in MODEL_PATCHING_SPECS:
        patching_specs.append((bert_model,) + spec)

    with ModelPatcher(patching_specs, ignore_missing_attributes=True):
        assert getattr(bert_model.config, "layerdrop", None) == 0
        # Checking that the context manager exists.
        with bert_model.no_sync():
            pass

    wav2vec2_model = Wav2Vec2Model(Wav2Vec2Config())
    assert (
        wav2vec2_model.config.layerdrop > 0
    ), "Default Wav2vec2Config layerdrop value is already 0 so the test will not check anything."
    patching_specs = []
    for spec in MODEL_PATCHING_SPECS:
        patching_specs.append((wav2vec2_model,) + spec)
    with ModelPatcher(patching_specs, ignore_missing_attributes=True):
        assert wav2vec2_model.config.layerdrop == 0, "layerdrop was not patched properly."

        # Checking that the context manager exists.
        with wav2vec2_model.no_sync():
            pass
