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
"""Tests for distributed utility functions and classes."""

from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from safetensors.torch import save_file

from optimum.neuron.distributed.utils import WeightInformation, load_tensor_for_weight


def test_load_tensor_for_weight():
    with TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname)
        filename = tmpdir / "tensors.safetensors"

        t1 = torch.empty((24, 24), dtype=torch.bfloat16)
        # Creating a slice from t1, meaning that it shares the same storage as t1.
        # It is important to make sure that the resulting loaded file does not have a bigger storage than needed.
        t2 = t1[:2, :2]
        save_file({"t1": t1, "t2": t2}, filename)

        weight_info_1 = WeightInformation(filename, "t1")
        weight_info_2 = WeightInformation(filename, "t2", device=torch.device("cpu"))

        loaded_t1 = load_tensor_for_weight(weight_info_1)
        loaded_t2 = load_tensor_for_weight(weight_info_2)
        loaded_sliced_t1 = load_tensor_for_weight(weight_info_1, tensor_slices=((2,), (2,)))

        assert torch.testing.assert_close(t1, loaded_t1)
        assert torch.testing.assert_close(t2, loaded_t2)
        assert torch.testing.assert_close(t2, loaded_sliced_t1)

        assert loaded_t1.numel() == loaded_t1.storage().size()
        assert loaded_t2.numel() == loaded_t2.storage().size()
        assert loaded_sliced_t1.numel() == loaded_sliced_t1.storage().size()


def test_embedding_to_parallel_embedding():
    pass
