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

from collections import defaultdict
from pathlib import Path

import pytest
import torch
import torch_xla.core.xla_model as xm
import torch_xla.runtime as xr
from neuronx_distributed.modules.qkv_linear import GQAQKVColumnParallelLinear
from neuronx_distributed.parallel_layers.parallel_state import (
    get_pipeline_model_parallel_size,
    get_tensor_model_parallel_size,
)
from neuronx_distributed.parallel_layers.utils import move_all_tensor_to_cpu
from neuronx_distributed.pipeline import NxDPPModel
from neuronx_distributed.utils.model_utils import move_model_to_device
from peft import PeftModelForCausalLM
from transformers import LlamaForCausalLM

from optimum.neuron.accelerate.optimizer import NeuronAcceleratedOptimizer
from optimum.neuron.models.training import LlamaForCausalLM as NeuronLlamaForCausalLM
from optimum.neuron.models.training.checkpointing import consolidate_model_parallel_checkpoints_to_unified_checkpoint
from optimum.neuron.models.training.config import TrainingNeuronConfig
from optimum.neuron.models.training.pipeline_utils import create_nxdpp_model
from optimum.neuron.peft import NeuronPeftModelForCausalLM
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
    "zero_1,gradient_accumulation_steps,max_grad_norm",
    [
        [False, 1, None],
        [False, 12, 0.01],
        [True, 1, 0.01],
        [True, 12, None],
    ],
    ids=[
        "zero_1=False,gradient_accumulation_steps=1,max_grad_norm=None",
        "zero_1=False,gradient_accumulation_steps=12,max_grad_norm=0.01",
        "zero_1=True,gradient_accumulation_steps=1,max_grad_norm=0.01",
        "zero_1=True,gradient_accumulation_steps=12,max_grad_norm=None",
    ],
)
@distributed_test(world_size=32, tp_size=2, pp_size=4)
@is_trainium_test
def test_optimizer_step(zero_1, gradient_accumulation_steps, max_grad_norm, set_cache_for_ci):
    world_size = xr.world_size()
    tp_size = get_tensor_model_parallel_size()
    pp_size = get_pipeline_model_parallel_size()
    dp_size = world_size // (tp_size * pp_size)
    if dp_size == 1 and zero_1:
        pytest.skip("zero_1 needs to be tested only for dp_size > 1")

    # TODO: investigate that with the AWS team to find a solution.
    if dp_size > 1 and zero_1 and max_grad_norm is not None:
        pytest.skip("Gradient clipping seems to not work properly with ZeRO-1.")

    accelerator = create_accelerator(
        tp_size, pp_size, zero_1=zero_1, gradient_accumulation_steps=gradient_accumulation_steps
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


@pytest.mark.parametrize(
    "world_size,tp_size,pp_size,kv_size_multiplier,fuse_qkv",
    [
        [32, 2, 4, None, False],
        [8, 8, 1, None, False],
        [8, 8, 1, 4, False],
        [8, 8, 1, 4, True],
    ],
    ids=[
        "dp=4,tp=2,pp=4",
        "dp=1,tp=8,kv_size_multiplier=None,GQAQKVColumnParallelLinear",
        "dp=1,tp=8,kv_size_multiplier=4,GQAQKVColumnParallelLinear",
        "dp=1,tp=8,kv_size_multiplier=4,GQAQKVColumnParallelLinear,fuse_qkv",
    ],
)
@pytest.mark.parametrize(
    "use_xser",
    [True, False],
    ids=["use_xser=True", "use_xser=False"],
)
@distributed_test()
@is_trainium_test
def test_consolidate_custom_model_parallel_checkpoints(
    tmpdir, world_size, tp_size, pp_size, kv_size_multiplier, fuse_qkv, use_xser, set_cache_for_ci
):
    tmpdir = Path(tmpdir)
    orig_model = LlamaForCausalLM.from_pretrained(MODEL_NAME_WITH_4_KV_HEADS)

    if xr.global_ordinal() == 0:
        orig_model.save_pretrained(tmpdir / "orig_model", safe_serialization=False)

    trn_config = TrainingNeuronConfig(
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        use_xser=use_xser,
        async_save=False,
        fuse_qkv=fuse_qkv,
        kv_size_multiplier=kv_size_multiplier,
    )
    custom_model = NeuronLlamaForCausalLM.from_pretrained(MODEL_NAME_WITH_4_KV_HEADS, trn_config)

    if pp_size > 1:
        custom_model = create_nxdpp_model(custom_model)

    custom_model.save_pretrained(tmpdir / "custom_model")

    xm.rendezvous("Saving done.")

    if xr.global_ordinal() == 0:
        consolidate_model_parallel_checkpoints_to_unified_checkpoint(
            tmpdir / "custom_model",
            tmpdir / "consolidated_model",
            save_format="pytorch",
        )
        orig_state_dict = torch.load(tmpdir / "orig_model" / "pytorch_model.bin", weights_only=True)
        consolidated_state_dict = torch.load(tmpdir / "consolidated_model" / "pytorch_model.bin", weights_only=True)

        assert orig_state_dict.keys() == consolidated_state_dict.keys(), (
            "Keys of the original state dict and consolidated state dict do not match."
        )
        for key in orig_state_dict:
            orig_tensor = orig_state_dict[key]
            consolidated_tensor = consolidated_state_dict[key]
            torch.testing.assert_close(orig_tensor, consolidated_tensor)


@pytest.mark.parametrize(
    "world_size,tp_size,pp_size,kv_size_multiplier,fuse_qkv",
    [
        [8, 2, 1, None, False],
        [8, 8, 1, 4, False],
        [8, 8, 1, 4, True],
    ],
    ids=[
        "dp=4,tp=2",
        "dp=1,tp=8,kv_size_multiplier=4,GQAQKVColumnParallelLinear",
        "dp=1,tp=8,kv_size_multiplier=4,GQAQKVColumnParallelLinear,fuse_qkv",
    ],
)
@pytest.mark.parametrize(
    "use_xser",
    [True, False],
    ids=["use_xser=True", "use_xser=False"],
)
@distributed_test()
@is_trainium_test
def test_consolidate_custom_lora_model_parallel_checkpoints(
    tmpdir, world_size, tp_size, pp_size, kv_size_multiplier, fuse_qkv, use_xser, set_cache_for_ci
):
    tmpdir = Path(tmpdir)
    orig_model = LlamaForCausalLM.from_pretrained(MODEL_NAME_WITH_4_KV_HEADS)

    first_lora_adapter_model_name_or_path = "michaelbenayoun/lora-qkv-included-llama-2-tiny-4kv-heads-4layers-random"
    second_lora_adapter_model_name_or_path = (
        "michaelbenayoun/lora-2-qkv-included-llama-2-tiny-4kv-heads-4layers-random"
    )

    # Loading the LoRA adapters into the original model.
    orig_model = PeftModelForCausalLM.from_pretrained(
        orig_model,
        first_lora_adapter_model_name_or_path,
        adapter_name="default",
    )
    orig_model.load_adapter(
        second_lora_adapter_model_name_or_path,
        adapter_name="test",
    )

    if xr.global_ordinal() == 0:
        orig_model.save_pretrained(tmpdir / "orig_model", safe_serialization=False)

    trn_config = TrainingNeuronConfig(
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
        use_xser=use_xser,
        async_save=False,
        fuse_qkv=fuse_qkv,
    )
    custom_model = NeuronLlamaForCausalLM.from_pretrained(MODEL_NAME_WITH_4_KV_HEADS, trn_config)

    # Loading the LoRA adapters into the custom model.
    custom_model = NeuronPeftModelForCausalLM.from_pretrained(
        custom_model,
        first_lora_adapter_model_name_or_path,
        adapter_name="default",
    )
    custom_model.load_adapter(
        second_lora_adapter_model_name_or_path,
        adapter_name="test",
    )

    has_gqa_qkv_column_parallel_linear = any(isinstance(m, GQAQKVColumnParallelLinear) for m in custom_model.modules())

    # Some weights need to be averaged before comparing them.
    # For now it is only the case for the LoRA A weights when there is linear fusion involved.
    # We specify this in a dictionary where:
    #  - the key is the suffix of the weight that we want to average
    # - the value is a list of suffixes that we want to average with the key suffix.
    key_suffixes_of_weights_to_average = {
        "gate_proj.lora_A.weight": ["gate_proj.lora_A.weight", "up_proj.lora_A.weight"],
        "up_proj.lora_A.weight": ["gate_proj.lora_A.weight", "up_proj.lora_A.weight"],
    }
    if has_gqa_qkv_column_parallel_linear:
        key_suffixes_of_weights_to_average.update(
            {
                "q_proj.lora_A.weight": ["q_proj.lora_A.weight", "k_proj.lora_A.weight", "v_proj.lora_A.weight"],
                "k_proj.lora_A.weight": ["q_proj.lora_A.weight", "k_proj.lora_A.weight", "v_proj.lora_A.weight"],
                "v_proj.lora_A.weight": ["q_proj.lora_A.weight", "k_proj.lora_A.weight", "v_proj.lora_A.weight"],
            }
        )

    custom_model.save_pretrained(tmpdir / "custom_model")
    xm.rendezvous("Saving done.")

    if xr.global_ordinal() == 0:
        consolidate_model_parallel_checkpoints_to_unified_checkpoint(
            tmpdir / "custom_model",
            tmpdir / "consolidated_model",
            save_format="pytorch",
        )
        for adapter_name in ["default", "test"]:
            if adapter_name == "default":
                orig_state_dict = torch.load(tmpdir / "orig_model" / "adapter_model.bin", weights_only=True)
                consolidated_state_dict = torch.load(
                    tmpdir / "consolidated_model" / "adapter_model.bin", weights_only=True
                )
            else:
                orig_state_dict = torch.load(
                    tmpdir / "orig_model" / adapter_name / "adapter_model.bin", weights_only=True
                )
                consolidated_state_dict = torch.load(
                    tmpdir / "consolidated_model" / adapter_name / "adapter_model.bin", weights_only=True
                )

            assert orig_state_dict.keys() == consolidated_state_dict.keys(), (
                f"Keys of the original state dict and consolidated state dict do not match for adapter {adapter_name}."
            )
            for key in orig_state_dict:
                orig_tensor = orig_state_dict[key]
                consolidated_tensor = consolidated_state_dict[key]
                print(f"Testing that {key} match for adapter {adapter_name}")
                if any(key.endswith(suffix) for suffix in key_suffixes_of_weights_to_average):
                    continue
                else:
                    torch.testing.assert_close(orig_tensor, consolidated_tensor)

            # For the weights that need to be averaged before compared, we do it here.
            orig_tensors = defaultdict(list)
            for key_suffix, suffixes in key_suffixes_of_weights_to_average.items():
                for key in orig_state_dict.keys():
                    # If the key ends with the key_suffix, it means that the associated weight needs to be averaged
                    # with weights that end with the suffixes.
                    if key.endswith(key_suffix):
                        # key_prefix is basically the fully qualified name of the module that contains the weight.
                        key_prefix = key[: -len(key_suffix)]
                        # We collect all the tensors that need to be averaged.
                        for name, tensor in orig_state_dict.items():
                            for suffix in suffixes:
                                if name.endswith(suffix):
                                    # name_prefix is the fully qualified name of the module that contains the weight.
                                    name_prefix = name[: -len(suffix)]
                                    # We only keep the tensors that are from the same module as the key.
                                    if name_prefix == key_prefix:
                                        orig_tensors[key].append(tensor)

            for key, tensors in orig_tensors.items():
                orig_tensor = torch.mean(torch.stack(tensors, dim=0), dim=0)
                consolidated_tensor = consolidated_state_dict[key]
                print(f"Testing that {key} match for adapter {adapter_name}")
                torch.testing.assert_close(orig_tensor, consolidated_tensor)


@distributed_test(world_size=32, tp_size=2, pp_size=4)
@is_trainium_test
def test_each_pp_rank_only_loads_relevant_parameters(set_cache_for_ci):
    tp_size = get_tensor_model_parallel_size()
    pp_size = get_pipeline_model_parallel_size()
    trn_config = TrainingNeuronConfig(
        tensor_parallel_size=tp_size,
        pipeline_parallel_size=pp_size,
    )
    model = NeuronLlamaForCausalLM.from_pretrained(MODEL_NAME_WITH_4_KV_HEADS, trn_config)
    parameters_on_cpu = {n for n, p in model.named_parameters() if p.device == torch.device("cpu")}
    parameters_on_meta = {n for n, p in model.named_parameters() if p.device == torch.device("meta")}

    accelerator = create_accelerator(tp_size, pp_size)

    nxd_pp_model = accelerator.prepare(model)

    local_parameters = {n for n, _ in nxd_pp_model.local_named_parameters()}
    other_parameters = {n for n, _ in nxd_pp_model.named_parameters() if n not in local_parameters}

    diff = local_parameters ^ parameters_on_cpu
    assert diff != {}, f"Expected that only the parameters of the current PP rank are on CPU. Got {diff} instead."

    diff = other_parameters ^ parameters_on_meta
    assert diff != {}, (
        f"Expected that the parameters of the other PP ranks are on the meta device. Got {diff} instead."
    )
