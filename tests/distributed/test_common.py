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
"""General tests related to distributed training."""

import contextlib
import pytest
from typing import TYPE_CHECKING, Dict
from tests.distributed.utils import create_static_seed_patcher

import torch
import torch_xla.core.xla_model as xm
from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_group
from neuronx_distributed.utils.model_utils import move_model_to_device
from neuronx_distributed.pipeline import NxDPPModel

from transformers import AutoModelForCausalLM, LlamaForCausalLM

from optimum.neuron.accelerate import NeuronAccelerator
from optimum.neuron.accelerate.utils.dataclasses import ModelParallelismPlugin, NeuronDistributedType
from optimum.neuron.distributed.utils import lazy_load_for_parallelism, make_optimizer_constructor_lazy

from .distributed import DistributedTest


if TYPE_CHECKING:
    from transformers import PreTrainedModel


def create_accelerator_for_mp(tp_size: int, pp_size: int, zero_1: bool = False) -> NeuronAccelerator:
    mp_plugin = ModelParallelismPlugin(
        tensor_parallel_size=tp_size,
        parallelize_embeddings=True,
        sequence_parallel_enabled=True,
        pipeline_parallel_size=pp_size,
    )
    return NeuronAccelerator(mp_plugin=mp_plugin, zero_1=zero_1)


def get_model(tp_size: int = 1, pp_size: int = 1, lazy_load: bool = False, use_static_seed_patcher: bool = False) -> "PreTrainedModel":
    model_name = "michaelbenayoun/llama-2-tiny-16layers-random"
    if lazy_load:
        ctx = lazy_load_for_parallelism(tensor_parallel_size=tp_size, pipeline_parallel_size=pp_size)
    else:
        ctx = contextlib.nullcontext()
    if use_static_seed_patcher:
        seed_patcher = create_static_seed_patcher(LlamaForCausalLM, 42)
    else:
        seed_patcher = contextlib.nullcontext()
    with ctx:
        with seed_patcher:
            return AutoModelForCausalLM.from_pretrained(model_name)

def get_optimizer(model: torch.nn.Module, lazy: bool, with_groups: bool) -> torch.optim.Optimizer:
    adam_cls = torch.optim.AdamW
    if lazy:
        adam_cls = make_optimizer_constructor_lazy(adam_cls)
    
    if with_groups:
        groups = [
            {"params": (p for idx, p in enumerate(model.parameters()) if idx % 2 == 0), "lr": 1e-2},
            {"params": (p for idx, p in enumerate(model.parameters()) if idx % 2 == 1), "lr": 1e-6},
        ]
    else:
        groups = model.parameters()
    
    return adam_cls(groups)


class TestCommonDistributed(DistributedTest):
    # TODO: add dp + tp + pp configuration.
    @pytest.fixture(scope="class", params=[[2, 1, 1], [2, 2, 1], [2, 1, 2]], ids=["dp=2", "tp=2", "pp=2"])
    def parallel_sizes(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[False, True], ids=["no_lazy_load", "lazy_load"])
    def lazy_load(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[False, True], ids=["no_lazy_optimizer", "lazy_optimizer"])
    def lazy_optimizer(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[False, True], ids=["without_groups", "with_groups"])
    def with_groups(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[False, True], ids=["no_zero_1", "zero_1"])
    def zero_1(self, request):
        return request.param

    def test_optimizer_parameters_match_models_parameters(self, lazy_load, lazy_optimizer, with_groups, zero_1, parallel_sizes):
        num_workers, tp_size, pp_size = parallel_sizes
        dp_size = num_workers // (tp_size * pp_size)
        if dp_size == 1 and zero_1:
            pytest.skip("zero_1 needs to be tested only for dp_size > 1")

        model = get_model(tp_size=tp_size, pp_size=pp_size, lazy_load=lazy_load)
        optimizer = get_optimizer(model, lazy_optimizer, with_groups)

        accelerator = create_accelerator_for_mp(tp_size, pp_size, zero_1=zero_1)
        assert accelerator.state.distributed_type is NeuronDistributedType.MODEL_PARALLELISM

        model, optimizer = accelerator.prepare(model, optimizer)

        if isinstance(model, NxDPPModel):
            model_parameters = set(model.local_parameters())
        else:
            model_parameters = set(model.parameters())
        optimizer_parameters = set(p for group in optimizer.param_groups for p in group["params"])

        assert model_parameters == optimizer_parameters

    def test_lazy_load(self, parallel_sizes):
        _, tp_size, pp_size = parallel_sizes

        model = get_model(tp_size=tp_size, pp_size=pp_size, lazy_load=False, use_static_seed_patcher=True)
        move_model_to_device(model, xm.xla_device())
        orig_parameters: Dict[str, torch.nn.Parameter] = dict(model.named_parameters())

        accelerator = create_accelerator_for_mp(tp_size, pp_size)
        lazy_model = get_model(tp_size=tp_size, pp_size=pp_size, lazy_load=True, use_static_seed_patcher=True)
        lazy_model = accelerator.prepare(lazy_model)

        xm.mark_step()

        if pp_size > 1:
            named_parameters = lazy_model.local_named_parameters()
        else:
            named_parameters = lazy_model.named_parameters()

        for name, param in named_parameters:
            orig = orig_parameters[name]
            if orig.shape != param.shape:
                if orig.dim() == 1:
                    gather_dim = 0
                elif orig.dim() == 2:
                    gather_dim = 1 if orig.shape[0] == param.shape[0] else 0
                else:
                    raise ValueError(f"The case where the weight as a rank of {orig.dim()} is not supported.")
                gathered = [torch.empty(param.shape) for _ in range(tp_size)]
                torch.distributed.all_gather(gathered, param, group=get_tensor_model_parallel_group())
                gathered_param = torch.cat(gathered, dim=gather_dim)
                orig = orig.to("cpu")
                xm.mark_step()
            else:
                gathered_param = param
            print(f"Comparing parameter named {name}")
            torch.testing.assert_allclose(orig, gathered_param)

