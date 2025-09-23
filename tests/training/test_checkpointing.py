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
from peft import PeftModelForCausalLM
from transformers import LlamaForCausalLM

from optimum.neuron.accelerate.accelerator import NeuronAccelerator
from optimum.neuron.models.training import LlamaForCausalLM as NeuronLlamaForCausalLM
from optimum.neuron.models.training.checkpointing import consolidate_model_parallel_checkpoints_to_unified_checkpoint
from optimum.neuron.models.training.config import TrainingNeuronConfig
from optimum.neuron.models.training.pipeline_utils import create_nxdpp_model
from optimum.neuron.peft import NeuronPeftModelForCausalLM
from optimum.neuron.utils.testing_utils import is_trainium_test

from ..distributed_utils import distributed_test


MODEL_NAME = "michaelbenayoun/llama-2-tiny-16layers-random"
MODEL_NAME_WITH_4_KV_HEADS = "michaelbenayoun/llama-2-tiny-4kv-heads-4layers-random"


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
        [32, 2, 4, None, False],
        [8, 8, 1, 4, False],
        [8, 8, 1, 4, True],
    ],
    ids=[
        "8_2_1",
        "32_2_4",
        "8_8_1-kv_size_multiplier_4-GQAQKVColumnParallelLinear",
        "8_8_1,kv_size_multiplier_4-GQAQKVColumnParallelLinear-fuse_qkv",
    ],
)
@pytest.mark.parametrize(
    "use_xser",
    [True, False],
    ids=["xser", "no_xser"],
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
        kv_size_multiplier=kv_size_multiplier,
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
    accelerator = NeuronAccelerator(trn_config=trn_config)
    custom_model = accelerator.prepare_model(custom_model)

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
