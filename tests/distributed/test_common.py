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
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional, Union

import pytest
import safetensors
import torch
import torch_xla.core.xla_model as xm
from neuronx_distributed.parallel_layers.parallel_state import (
    get_pipeline_model_parallel_rank,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
)
from neuronx_distributed.parallel_layers.utils import move_all_tensor_to_cpu
from neuronx_distributed.pipeline import NxDPPModel
from neuronx_distributed.utils.model_utils import move_model_to_device
from transformers import AutoConfig, AutoTokenizer, LlamaForCausalLM

from optimum.neuron.accelerate import NeuronAccelerator
from optimum.neuron.accelerate.optimizer import NeuronAcceleratedOptimizer
from optimum.neuron.accelerate.utils.dataclasses import ModelParallelismPlugin, NeuronDistributedType
from optimum.neuron.distributed.utils import (
    TENSOR_PARALLEL_SHARDS_DIR_NAME,
    lazy_load_for_parallelism,
    make_optimizer_constructor_lazy,
)

from .distributed import DistributedTest
from .utils import create_static_seed_patcher


if TYPE_CHECKING:
    from transformers import PreTrainedModel

MODEL_NAME = "michaelbenayoun/llama-2-tiny-16layers-random"


def create_accelerator_for_mp(
    tp_size: int,
    pp_size: int,
    zero_1: bool = False,
    gradient_accumulation_steps: int = 1,
    checkpoint_dir: Optional[Union[Path, str]] = None,
) -> NeuronAccelerator:
    mp_plugin = ModelParallelismPlugin(
        tensor_parallel_size=tp_size,
        parallelize_embeddings=True,
        sequence_parallel_enabled=True,
        pipeline_parallel_size=pp_size,
        checkpoint_dir=checkpoint_dir,
    )
    return NeuronAccelerator(
        mp_plugin=mp_plugin, zero_1=zero_1, gradient_accumulation_steps=gradient_accumulation_steps
    )


def get_model(
    tp_size: int = 1,
    pp_size: int = 1,
    lazy_load: bool = False,
    from_config: bool = False,
    use_static_seed_patcher: bool = False,
) -> "PreTrainedModel":
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
            if from_config:
                return LlamaForCausalLM.from_config(AutoConfig(MODEL_NAME))
            return LlamaForCausalLM.from_pretrained(MODEL_NAME)


def get_model_inputs(include_labels: bool = True):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    inputs = tokenizer("Hello there, I'm Michael and I live in Paris!", return_tensors="pt")
    if include_labels:
        inputs["labels"] = tokenizer("Hello there, I'm Michael and I live in Paris!", return_tensors="pt")["input_ids"]
    return inputs


def get_optimizer(model: torch.nn.Module, lazy: bool = False, with_groups: bool = True) -> torch.optim.Optimizer:
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


def move_params_to_cpu(parameters):
    parameters = list(parameters)
    xm.mark_step()
    # `move_all_tensor_to_cpu` only selects `torch.Tensor`, so we need to move the parameters' data.
    cpu_params = move_all_tensor_to_cpu([p.data for p in parameters])
    return cpu_params


class TestCommonDistributed(DistributedTest):
    # TODO: add dp + tp + pp configuration.
    @pytest.fixture(scope="class", params=[[2, 1, 1], [2, 2, 1], [2, 1, 2]], ids=["dp=2", "tp=2", "pp=2"])
    def parallel_sizes(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[False, True], ids=["no_lazy_load", "lazy_load"])
    def lazy_load(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[False, True], ids=["from_config", "from_pretrained"])
    def from_config(self, request):
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

    @pytest.fixture(scope="class", params=[1, 12], ids=["no_grad_acc", "grad_acc=12"])
    def gradient_accumulation_steps(self, request):
        return request.param

    @pytest.fixture(scope="class", params=[None, 0.25], ids=["no_clip_grad_norm", "clip_grad_norm"])
    def max_grad_norm(self, request):
        return request.param

    def test_optimizer_parameters_match_models_parameters(
        self, lazy_load, lazy_optimizer, with_groups, zero_1, parallel_sizes
    ):
        num_workers, tp_size, pp_size = parallel_sizes
        dp_size = num_workers // (tp_size * pp_size)
        if dp_size == 1 and zero_1:
            pytest.skip("zero_1 needs to be tested only for dp_size > 1")

        model = get_model(tp_size=tp_size, pp_size=pp_size, lazy_load=lazy_load)
        optimizer = get_optimizer(model, lazy_optimizer, with_groups)

        accelerator = create_accelerator_for_mp(tp_size, pp_size, zero_1=zero_1)
        assert accelerator.state.distributed_type is NeuronDistributedType.MODEL_PARALLELISM

        model, optimizer = accelerator.prepare(model, optimizer)
        assert isinstance(optimizer, NeuronAcceleratedOptimizer)

        if isinstance(model, NxDPPModel):
            model_parameters = set(model.local_parameters())
        else:
            model_parameters = set(model.parameters())
        optimizer_parameters = {p for group in optimizer.param_groups for p in group["params"]}

        assert model_parameters == optimizer_parameters

    def test_optimizer_step(self, zero_1, gradient_accumulation_steps, max_grad_norm, parallel_sizes):
        num_workers, tp_size, pp_size = parallel_sizes
        dp_size = num_workers // (tp_size * pp_size)
        if dp_size == 1 and zero_1:
            pytest.skip("zero_1 needs to be tested only for dp_size > 1")

        model = get_model(tp_size=tp_size, pp_size=pp_size)
        optimizer = get_optimizer(model)

        accelerator = create_accelerator_for_mp(
            tp_size, pp_size, zero_1=zero_1, gradient_accumulation_steps=gradient_accumulation_steps
        )

        model, optimizer = accelerator.prepare(model, optimizer)
        assert isinstance(optimizer, NeuronAcceleratedOptimizer)

        inputs = get_model_inputs()

        def move_grads_to_cpu(parameters):
            grads = [p.grad for p in parameters]
            # xm.mark_step()
            grads = move_all_tensor_to_cpu(grads)
            # grads = [grad.to("cpu") for grad in grads]
            return grads

        inputs = {k: v.to(xm.xla_device()) for k, v in inputs.items()}
        current_parameters = move_params_to_cpu(
            model.parameters() if isinstance(model, torch.nn.Module) else model.local_parameters()
        )

        for step in range(2 * gradient_accumulation_steps):
            xm.mark_step()
            with accelerator.accumulate():
                if pp_size > 1:
                    orig_parameters = current_parameters
                    loss = model.run_train(**inputs)
                    xm.mark_step()

                    if max_grad_norm is not None:
                        accelerator.clip_grad_norm_(model.local_parameters(), max_norm=max_grad_norm, norm_type=2)
                        for param in model.local_parameters():
                            assert torch.linalg.norm(param.grad, p=2) <= max_grad_norm

                    # Checking that at least some of the parameters have a gradient.
                    assert any(torch.any(param.grad != 0) for param in model.local_parameters())

                    optimizer.step()
                    model.zero_grad()

                    # At this point, no parameter should have a gradient.
                    assert all(torch.all(param.grad == 0) for param in model.local_parameters())

                    current_parameters = list(model.local_parameters())
                else:
                    orig_parameters = current_parameters
                    outputs = model(**inputs)
                    loss = outputs["loss"]
                    loss.backward()

                    if max_grad_norm is not None:
                        accelerator.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm, norm_type=2)

                    # Checking that at least some of the parameters have a gradient.
                    grads_on_cpu = move_grads_to_cpu(model.parameters())
                    # assert any(torch.any(grad != 0) for grad in grads_on_cpu)

                    optimizer.step()

                    # Checking here that the norm has been clipped because it happens during the optimizer steps in some cases.
                    if max_grad_norm is not None:
                        grads_on_cpu = move_grads_to_cpu(model.parameters())
                        assert all(torch.linalg.norm(grad, p=2) <= max_grad_norm for grad in grads_on_cpu)

                    model.zero_grad()

                    # At this point, no parameter should have a gradient.
                    grads_on_cpu = move_grads_to_cpu(model.parameters())
                    assert all(torch.all(grad == 0) for grad in grads_on_cpu)

                    current_parameters = move_params_to_cpu(model.parameters())

                with torch.no_grad():
                    if step % gradient_accumulation_steps != 0:
                        assert all(torch.all(p1 == p2) for (p1, p2) in zip(orig_parameters, current_parameters))
                    else:
                        assert all(torch.all(p1 != p2) for (p1, p2) in zip(orig_parameters, current_parameters))

    def test_lazy_load(self, from_config, parallel_sizes):
        _, tp_size, pp_size = parallel_sizes

        model = get_model(
            tp_size=tp_size, pp_size=pp_size, lazy_load=False, from_config=from_config, use_static_seed_patcher=True
        )
        move_model_to_device(model, xm.xla_device())
        orig_parameters: Dict[str, torch.nn.Parameter] = dict(model.named_parameters())

        accelerator = create_accelerator_for_mp(tp_size, pp_size)
        lazy_model = get_model(
            tp_size=tp_size, pp_size=pp_size, lazy_load=True, from_config=from_config, use_static_seed_patcher=True
        )
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

    def test_save_model_and_load_model(self, parallel_sizes, tmpdir, monkeypatch):
        tmpdir = Path(tmpdir)
        _, tp_size, pp_size = parallel_sizes
        tp_rank = get_tensor_model_parallel_rank()
        pp_rank = get_pipeline_model_parallel_rank()

        model = get_model(tp_size=tp_size, pp_size=pp_size, lazy_load=False)

        accelerator = create_accelerator_for_mp(tp_size, pp_size)
        model = accelerator.prepare(model)
        accelerator.save_state(tmpdir.as_posix())

        if pp_size > 1:
            # We need to disable `NxDPPModel._set_distributed` since it is already done during the creation of the
            # first model, otherwise creating new `NxDPPModel`s will fail.
            monkeypatch.setattr(NxDPPModel, "_set_distributed", lambda _: _)

        tmpdir_content = [path.name for path in tmpdir.glob("**/*")]
        pytorch_checkpoint_exists = "pytorch_model.bin" in tmpdir_content
        safetensors_checkpoint_exists = "model.safetensors" in tmpdir_content

        if tp_size > 1 or pp_size > 1:
            ref_data_file_name = f"tp_rank_{tp_rank:02d}_pp_rank_{pp_rank:02d}"
            tensors_directory = f"{ref_data_file_name}.tensors"
            assert not pytorch_checkpoint_exists
            assert not safetensors_checkpoint_exists
            assert TENSOR_PARALLEL_SHARDS_DIR_NAME in tmpdir_content
            assert ref_data_file_name in tmpdir_content
            assert tensors_directory in tmpdir_content
        else:
            assert pytorch_checkpoint_exists or safetensors_checkpoint_exists

        # Making sure that we end-up with a different model when starting over.
        new_model = get_model(tp_size=tp_size, pp_size=pp_size, lazy_load=False)
        new_accelerator = create_accelerator_for_mp(tp_size, pp_size)
        new_model = new_accelerator.prepare(new_model)

        if pp_size == 1:
            model_parameters = move_params_to_cpu(model.parameters())
            new_model_parameters = move_params_to_cpu(new_model.parameters())
        else:
            model_parameters = move_params_to_cpu(model.local_parameters())
            new_model_parameters = move_params_to_cpu(new_model.local_parameters())

        assert any(
            torch.all(p1 == 0.0) or torch.all(p1 == 1.0) or torch.all(p1 != p2)
            for p1, p2 in zip(model_parameters, new_model_parameters)
        )

        # Checking that when providing a checkpoint, we end-up with the same model as the original.
        new_model = get_model(tp_size=tp_size, pp_size=pp_size, lazy_load=False)
        new_accelerator = create_accelerator_for_mp(tp_size, pp_size, checkpoint_dir=tmpdir)
        new_model = new_accelerator.prepare(new_model)

        # If there is no model parallelism, the checkpoint weights will not be loaded automatically since we do not
        # call parallelize, so we do it manually.
        if tp_size == 1 and pp_size == 1:
            if pytorch_checkpoint_exists:
                filename = "pytorch_model.bin"
                checkpoint_path = tmpdir / filename
                new_model.load_state_dict(torch.load(checkpoint_path))
            else:
                filename = "model.safetensors"
                checkpoint_path = tmpdir / filename
                new_model.load_state_dict(safetensors.torch.load_file(checkpoint_path))

        if pp_size == 1:
            model_parameters = move_params_to_cpu(model.parameters())
            new_model_parameters = move_params_to_cpu(new_model.parameters())
        else:
            model_parameters = move_params_to_cpu(model.local_parameters())
            new_model_parameters = move_params_to_cpu(new_model.local_parameters())

        assert all(torch.all(p1 == p2) for p1, p2 in zip(model_parameters, new_model_parameters))
