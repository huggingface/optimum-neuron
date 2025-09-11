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


import pytest
import torch
import torch_xla.core.xla_model as xm
from neuronx_distributed.parallel_layers.parallel_state import (
    get_pipeline_model_parallel_size,
    get_tensor_model_parallel_size,
)
from neuronx_distributed.parallel_layers.utils import move_all_tensor_to_cpu
from neuronx_distributed.pipeline import NxDPPModel
from neuronx_distributed.utils.model_utils import move_model_to_device

from optimum.neuron.accelerate.optimizer import NeuronAcceleratedOptimizer
from optimum.neuron.models.training import LlamaForCausalLM as NeuronLlamaForCausalLM
from optimum.neuron.models.training.config import TrainingNeuronConfig
from optimum.neuron.utils.testing_utils import is_trainium_test

from ..distributed_utils import distributed_test
from .utils import create_accelerator, get_model_inputs


MODEL_NAME = "michaelbenayoun/llama-2-tiny-16layers-random"
MODEL_NAME_WITH_4_KV_HEADS = "michaelbenayoun/llama-2-tiny-4kv-heads-4layers-random"


def move_params_to_cpu(parameters):
    parameters = list(parameters)
    xm.mark_step()
    # `move_all_tensor_to_cpu` only selects `torch.Tensor`, so we need to move the parameters' data.
    cpu_params = move_all_tensor_to_cpu([p.data for p in parameters])
    return cpu_params


@pytest.mark.parametrize(
    "gradient_accumulation_steps,max_grad_norm",
    [
        [1, None],
        [12, 0.01],
        [1, 0.01],
        [12, None],
    ],
    ids=[
        "grad_acc_1-no_grad_norm",
        "grad_acc_12-grad_norm_0.01",
        "grad_acc_1-grad_norm_0.01",
        "grad_acc_12-no_grad_norm",
    ],
)
@distributed_test(world_size=32, tp_size=2, pp_size=4)
@is_trainium_test
def test_optimizer_step(gradient_accumulation_steps, max_grad_norm, set_cache_for_ci):
    tp_size = get_tensor_model_parallel_size()
    pp_size = get_pipeline_model_parallel_size()

    accelerator = create_accelerator(
        tp_size, pp_size, zero_1=False, gradient_accumulation_steps=gradient_accumulation_steps
    )

    trn_config = TrainingNeuronConfig(tensor_parallel_size=tp_size, pipeline_parallel_size=pp_size)
    model = NeuronLlamaForCausalLM.from_pretrained(MODEL_NAME_WITH_4_KV_HEADS, trn_config)

    model = accelerator.prepare(model)

    if tp_size == pp_size == 1:
        move_model_to_device(model, xm.xla_device())

    parameters = list(model.local_parameters()) if isinstance(model, NxDPPModel) else list(model.parameters())
    groups = [
        {"params": (p for idx, p in enumerate(parameters) if idx % 2 == 0), "lr": 1e-2},
        {"params": (p for idx, p in enumerate(parameters) if idx % 2 == 1), "lr": 1e-6},
    ]
    optimizer = torch.optim.AdamW(groups)
    optimizer = accelerator.prepare(optimizer)
    assert isinstance(optimizer, NeuronAcceleratedOptimizer), "Optimizer is not a NeuronAcceleratedOptimizer."

    inputs = get_model_inputs(model, MODEL_NAME_WITH_4_KV_HEADS)

    def move_grads_to_cpu(parameters):
        grads = [p.grad for p in parameters]
        grads = move_all_tensor_to_cpu(grads)
        return grads

    if pp_size == 1:
        inputs = {k: v.to(xm.xla_device()) for k, v in inputs.items()}

    current_parameters = move_params_to_cpu(parameters)

    for step in range(int(1.5 * gradient_accumulation_steps)):
        is_optimizer_update_step = (step + 1) % gradient_accumulation_steps == 0
        with accelerator.accumulate(model):
            if pp_size > 1:
                orig_parameters = current_parameters
                loss = model.run_train(**inputs)
                xm.mark_step()

                # Checking that at least some of the parameters have a gradient.
                grads_on_cpu = move_grads_to_cpu(model.local_parameters())
                assert any(torch.all(grad != 0) for grad in grads_on_cpu), "Expected some gradients to be non-zero."

                if is_optimizer_update_step:
                    if max_grad_norm is not None:
                        accelerator.clip_grad_norm_(
                            model.local_parameters(),
                            max_norm=max_grad_norm,
                            norm_type=2,
                            postpone_clipping_to_optimizer_step=True,
                        )
                    optimizer.step()

                    # Checking only after an actual optimizer step that the norm has been clipped because it happens
                    # during the optimizer step in some cases.
                    if max_grad_norm is not None:
                        grads_on_cpu = move_grads_to_cpu(model.local_parameters())
                        norms = [torch.linalg.vector_norm(grad, 2) for grad in grads_on_cpu]
                        total_norm = torch.linalg.vector_norm(torch.stack(norms), 2)
                        assert total_norm <= max_grad_norm, "Expected the total norm to be clipped."

                    optimizer.zero_grad()

                    grads_on_cpu = move_grads_to_cpu(model.local_parameters())
                    # At this point, no parameter should have a gradient.
                    assert all(grad is None or torch.all(grad == 0) for grad in grads_on_cpu), (
                        "Expected no gradients after zero_grad()."
                    )

                current_parameters = move_params_to_cpu(model.local_parameters())
            else:
                orig_parameters = current_parameters
                outputs = model(**inputs)
                loss = outputs["loss"]
                xm.mark_step()
                loss.backward()

                # Checking that at least some of the parameters have a gradient.
                grads_on_cpu = move_grads_to_cpu(model.parameters())
                assert any(torch.all(grad != 0) for grad in grads_on_cpu), "Expected some gradients to be non-zero."

                if is_optimizer_update_step:
                    if max_grad_norm is not None:
                        accelerator.clip_grad_norm_(
                            model.parameters(),
                            max_norm=max_grad_norm,
                            norm_type=2,
                            postpone_clipping_to_optimizer_step=True,
                        )

                    optimizer.step()

                    # Checking only after an actual optimizer step that the norm has been clipped because it happens
                    # during the optimizer step in some cases.
                    if max_grad_norm is not None:
                        grads_on_cpu = move_grads_to_cpu(model.parameters())
                        norms = [torch.linalg.vector_norm(grad, 2) for grad in grads_on_cpu]
                        total_norm = torch.linalg.vector_norm(torch.stack(norms), 2)
                        assert total_norm <= max_grad_norm, "Expected the total norm to be clipped."

                    optimizer.zero_grad()

                    # At this point, no parameter should have a gradient.
                    grads_on_cpu = move_grads_to_cpu(model.parameters())
                    assert all(grad is None or torch.all(grad == 0) for grad in grads_on_cpu), (
                        "Expected no gradients after zero_grad()."
                    )

                current_parameters = move_params_to_cpu(model.parameters())

            if is_optimizer_update_step:
                assert any(torch.any(p1 != p2) for (p1, p2) in zip(orig_parameters, current_parameters)), (
                    "Expected some parameters to have changed after an optimizer step."
                )
            else:
                assert all(torch.all(p1 == p2) for (p1, p2) in zip(orig_parameters, current_parameters)), (
                    "Expected no parameters to have changed before an optimizer step."
                )
