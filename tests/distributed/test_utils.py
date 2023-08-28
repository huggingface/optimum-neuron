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

import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Literal, Union

import torch
from safetensors.torch import save_file

from optimum.neuron.distributed.utils import (
    WeightInformation,
    embedding_to_parallel_embedding,
    linear_to_parallel_linear,
    load_tensor_for_weight,
)
from optimum.neuron.utils.patching import patch_everywhere

from ..test_utils import is_trainium_test


def test_load_tensor_for_weight():
    with TemporaryDirectory() as tmpdirname:
        tmpdir = Path(tmpdirname)
        filename = tmpdir / "tensors.safetensors"

        t1 = torch.empty((24, 24), dtype=torch.bfloat16)
        # Creating a slice from t1, meaning that it shares the same storage as t1.
        # It is important to make sure that the resulting loaded file does not have a bigger storage than needed.
        t2 = t1[:2, :2].contiguous()
        save_file({"t1": t1, "t2": t2}, filename)

        weight_info_1 = WeightInformation(filename, "t1")
        weight_info_2 = WeightInformation(filename, "t2", device=torch.device("cpu"))

        loaded_t1 = load_tensor_for_weight(weight_info_1)
        loaded_t2 = load_tensor_for_weight(weight_info_2)
        loaded_sliced_t1 = load_tensor_for_weight(weight_info_1, tensor_slices=((2,), (2,)))

        torch.testing.assert_close(t1, loaded_t1)
        torch.testing.assert_close(t2, loaded_t2)
        torch.testing.assert_close(t2, loaded_sliced_t1)

        assert loaded_t1.numel() == loaded_t1.storage().size()
        assert loaded_t2.numel() == loaded_t2.storage().size()
        assert loaded_sliced_t1.numel() == loaded_sliced_t1.storage().size()


@is_trainium_test
class ParallelUtilsTestCase(unittest.TestCase):
    TP_GROUP = 0
    TP_SIZE = 8
    TP_RANK = 0

    def setUp(self):
        self.tp_group = self.TP_GROUP
        self.tp_size = self.TP_SIZE
        self.tp_rank = self.TP_RANK
        patch_everywhere("get_tensor_model_parallel_group", self.get_tensor_model_parallel_group)
        patch_everywhere("get_tensor_model_parallel_size", self.get_tensor_model_parallel_size)
        patch_everywhere("get_tensor_model_parallel_rank", self.get_tensor_model_parallel_rank)

    def get_tensor_model_parallel_group(self, as_list: bool = False):
        if as_list:
            return list(range(self.tp_size))
        return self.tp_group

    def get_tensor_model_parallel_size(self):
        return self.tp_size

    def get_tensor_model_parallel_rank(self):
        return self.tp_rank

    def _test_embedding_to_parallel_embedding(self, with_weight_info: bool):
        shard_size = 23
        vocab_size = shard_size * self.tp_size  # We need to be a multiple of self.TP_SIZE.
        embedding = torch.nn.Embedding(vocab_size, 300)
        lm_head = torch.nn.Linear(300, vocab_size)

        with TemporaryDirectory() as tmpdirname:
            tmpdir = Path(tmpdirname)
            weight_filename = tmpdir / "weights.safetensors"

            if with_weight_info:
                embedding_weight = torch.randn(vocab_size, 300)
                lm_head_weight = torch.randn(vocab_size, 300)
                lm_head_bias_weight = torch.rand(vocab_size)
                state_dict = {
                    "embedding": embedding_weight,
                    "lm_head_weight": lm_head_weight,
                    "lm_head_bias_weight": lm_head_bias_weight,
                }
                save_file(state_dict, weight_filename)
                embedding_weight_info = WeightInformation(weight_filename, "embedding")
                lm_head_weight_info = WeightInformation(weight_filename, "lm_head_weight")
                lm_head_bias_weight_info = WeightInformation(weight_filename, "lm_head_bias_weight")
            else:
                embedding_weight = embedding.weight
                lm_head_weight = lm_head.weight
                lm_head_bias_weight = lm_head.bias
                embedding_weight_info = None
                lm_head_weight_info = None
                lm_head_bias_weight_info = None

            # Without lm_head.
            for tp_rank in range(self.TP_SIZE):
                self.tp_rank = tp_rank
                parallel_embedding = embedding_to_parallel_embedding(
                    copy.deepcopy(embedding),
                    embedding_weight_info=embedding_weight_info,
                    lm_head_weight_info=lm_head_weight_info,
                    lm_head_bias_weight_info=lm_head_bias_weight_info,
                )
                torch.testing.assert_close(
                    parallel_embedding.weight, embedding_weight[tp_rank * shard_size : (tp_rank + 1) * shard_size, :]
                )

            # With embedding and lm_head untied.
            for tp_rank in range(self.TP_SIZE):
                self.tp_rank = tp_rank
                parallel_embedding, parallel_lm_head = embedding_to_parallel_embedding(
                    copy.deepcopy(embedding),
                    lm_head_layer=copy.deepcopy(lm_head),
                    embedding_weight_info=embedding_weight_info,
                    lm_head_weight_info=lm_head_weight_info,
                    lm_head_bias_weight_info=lm_head_bias_weight_info,
                )
                torch.testing.assert_close(
                    parallel_embedding.weight, embedding_weight[tp_rank * shard_size : (tp_rank + 1) * shard_size, :]
                )
                torch.testing.assert_close(
                    parallel_lm_head.weight, lm_head_weight[tp_rank * shard_size : (tp_rank + 1) * shard_size, :]
                )
                torch.testing.assert_close(parallel_lm_head.bias, lm_head_bias_weight)
                assert id(parallel_embedding.weight) != id(parallel_lm_head.weight)
                assert id(parallel_lm_head.bias) != id(lm_head.bias)

            # With embedding and lm_head tied.
            for tp_rank in range(self.TP_SIZE):
                self.tp_rank = tp_rank
                embedding_copy = copy.deepcopy(embedding)
                tied_lm_head = copy.deepcopy(lm_head)
                tied_lm_head.weight = embedding_copy.weight
                parallel_embedding, parallel_lm_head = embedding_to_parallel_embedding(
                    embedding_copy,
                    lm_head_layer=tied_lm_head,
                    embedding_weight_info=embedding_weight_info,
                    lm_head_weight_info=lm_head_weight_info,
                    lm_head_bias_weight_info=lm_head_bias_weight_info,
                )
                torch.testing.assert_close(
                    parallel_embedding.weight, embedding_weight[tp_rank * shard_size : (tp_rank + 1) * shard_size, :]
                )
                # We check that
                # parallel_lm_head.weight == embedding.weight[tp_rank * shard_size: (tp_rank + 1) * shard_size, :] because
                # embedding_to_parallel_embedding is suppose to tie parallel_embedding and parallel_lm_head in this case.
                torch.testing.assert_close(
                    parallel_lm_head.weight, embedding_weight[tp_rank * shard_size : (tp_rank + 1) * shard_size, :]
                )
                torch.testing.assert_close(parallel_lm_head.bias, lm_head_bias_weight)
                assert id(parallel_embedding.weight) == id(parallel_lm_head.weight)
                assert id(parallel_lm_head.bias) != id(lm_head.bias)

    def test_embedding_to_parallel_embedding_without_weight_info(self):
        self._test_embedding_to_parallel_embedding(False)

    def test_embedding_to_parallel_embedding_with_weight_info(self):
        self._test_embedding_to_parallel_embedding(True)

    def _test_linear_to_parallel_linear(
        self,
        with_weight_info: bool,
        use_bias: bool,
        axis: Union[Literal["row"], Literal["column"]],
        input_is_parallel: bool,
        gather_output: bool,
    ):
        shard_size = 23
        vocab_size = shard_size * self.tp_size  # We need to be a multiple of self.TP_SIZE.
        if axis == "row":
            linear = torch.nn.Linear(vocab_size, 300, bias=use_bias)
        else:
            linear = torch.nn.Linear(300, vocab_size, bias=use_bias)

        with TemporaryDirectory() as tmpdirname:
            tmpdir = Path(tmpdirname)
            weight_filename = tmpdir / "weights.safetensors"

            if with_weight_info:
                linear_weight = torch.randn(300, vocab_size) if axis == "row" else torch.randn(vocab_size, 300)
                linear_bias_weight = torch.rand(300 if axis == "row" else vocab_size)
                state_dict = {
                    "linear_weight": linear_weight,
                    "linear_bias_weight": linear_bias_weight,
                }
                save_file(state_dict, weight_filename)
                linear_weight_info = WeightInformation(weight_filename, "linear_weight")
                linear_bias_weight_info = WeightInformation(weight_filename, "linear_bias_weight")
            else:
                linear_weight = linear.weight
                linear_bias_weight = linear.bias if use_bias else None
                linear_weight_info = None
                linear_bias_weight_info = None

            for tp_rank in range(self.TP_SIZE):
                self.tp_rank = tp_rank
                parallel_linear = linear_to_parallel_linear(
                    copy.deepcopy(linear),
                    axis=axis,
                    input_is_parallel=input_is_parallel,
                    gather_output=gather_output,
                    linear_layer_weight_info=linear_weight_info,
                    linear_layer_bias_weight_info=linear_bias_weight_info,
                )
                if axis == "row":
                    weight = linear_weight[:, tp_rank * shard_size : (tp_rank + 1) * shard_size]
                    bias = linear_bias_weight if use_bias else None
                else:
                    weight = linear_weight[tp_rank * shard_size : (tp_rank + 1) * shard_size, :]
                    if gather_output:
                        bias = linear_bias_weight
                    else:
                        bias = (
                            linear_bias_weight[tp_rank * shard_size : (tp_rank + 1) * shard_size] if use_bias else None
                        )
                torch.testing.assert_close(parallel_linear.weight, weight)
                torch.testing.assert_close(parallel_linear.bias, bias)
                assert id(parallel_linear.weight) != id(weight)
                if use_bias:
                    assert id(parallel_linear.bias) != id(bias)

    def _test_linear_to_parallel_linear_in_series(self, with_weight_info: bool):
        ### Row
        # No bias, no input parallel
        self._test_linear_to_parallel_linear(with_weight_info, False, "row", False, False)
        # bias, no input parallel
        self._test_linear_to_parallel_linear(with_weight_info, True, "row", False, False)
        # bias, input_parallel
        self._test_linear_to_parallel_linear(with_weight_info, True, "row", True, False)

        ### Column
        # No bias, no gather output
        self._test_linear_to_parallel_linear(with_weight_info, False, "column", False, False)
        # bias, no gather output
        self._test_linear_to_parallel_linear(with_weight_info, True, "column", False, False)
        # bias, gather output
        self._test_linear_to_parallel_linear(with_weight_info, True, "column", False, True)

    def test_linear_to_parallel_linear_without_weight_info(self):
        self._test_linear_to_parallel_linear_in_series(False)

    def test_linear_to_parallel_linear_with_weight_info(self):
        self._test_linear_to_parallel_linear_in_series(True)
