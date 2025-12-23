# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

import copy

import torch
import torch_xla
import torch_xla.core.xla_model as xm
from neuronx_distributed.parallel_layers.parallel_state import (
    get_tensor_model_parallel_group,
)

from ...models.training import create_parameter_metadata
from ...models.training.transformations_utils import to_original_weights


def get_original_merged_weights_for_vllm(model) -> dict[str, torch.Tensor]:
    """
    Gets original (unsharded, untransformed) weights from a NeuronPeftModel for vLLM.

    Steps:
        1. Merge LoRA adapters in-place on each TP shard. This way we go from "base + LoRA" to "merged weights" on each
           shard.
        2. Gather the sharded state dicts from all TP ranks.
        3. Get transformation specs and metadata required to revert to original weights.
        4. Use `to_original_weights` to revert the gathered sharded weights to original weights.
        5. Unmerge LoRA adapters to restore the model state.
    """
    from ...peft import NeuronPeftModel

    if not isinstance(model, NeuronPeftModel):
        raise TypeError(f"Expected NeuronPeftModel, got {type(model).__name__}")

    tp_group = get_tensor_model_parallel_group(as_list=True)

    # Step 1: Merge LoRA adapters (modifies weights in-place on each shard)
    model.merge_adapter()
    torch_xla.sync()

    # Step 2: Gather state dict across TP ranks
    # Get local state dict (sharded weights from this rank)
    # Strip PEFT prefixes to make it look like a regular (non-PEFT) model
    local_state_dict = {}
    for name, param in model.named_parameters():
        # Skip LoRA adapter parameters (lora_A, lora_B, lora_embedding_A, lora_embedding_B, etc.)
        if "lora" in name.lower():
            continue

        # Strip PEFT prefixes: "base_model.model." and ".base_layer"
        # This makes merged PEFT weights look like regular model weights
        clean_name = name.removeprefix("base_model.model.").replace(".base_layer", "")

        # Skip modules_to_save and original_module (PEFT-specific)
        if "modules_to_save" in clean_name or "original_module" in clean_name:
            continue

        local_state_dict[clean_name] = param.data

    # Gather all TP rank state dicts into format: {name: [tensor_rank0, tensor_rank1, ...]}
    sharded_state_dicts = {}
    for name, local_tensor in local_state_dict.items():
        gathered = xm.all_gather(local_tensor, dim=0, groups=tp_group)
        gathered_tensors = list(torch.split(gathered, local_tensor.size(0), dim=0))
        sharded_state_dicts[name] = gathered_tensors
    torch_xla.sync()

    # Step 3: Get transformation specs and metadata
    # For NeuronPeftModel: model.base_model.model is the actual NeuronModelForCausalLM
    base_model = model.base_model.model

    # Get transformation specs from base model
    transformation_specs = []
    for module in base_model.modules():
        if hasattr(module, "specs"):
            transformation_specs.append(copy.deepcopy(module.specs))

    for specs in transformation_specs:
        specs.module_fully_qualified_name = specs.module_fully_qualified_name.removeprefix(
            "base_model.model."
        ).replace(".base_layer", "")
        for spec in specs:
            spec.peft_type = None

    # Create parameter metadata from peft model
    metadata = create_parameter_metadata(model)
    parameters_metadata = metadata["parameters"]

    # Clean parameter names in metadata to match the cleaned state dict
    cleaned_parameters_metadata = {}
    for name, param_metadata in parameters_metadata.items():
        # Skip LoRA parameters
        if "lora" in name.lower():
            continue

        # Strip PEFT prefixes
        clean_name = name.removeprefix("base_model.model.").replace(".base_layer", "")

        # Skip modules_to_save and original_module
        if "modules_to_save" in clean_name or "original_module" in clean_name:
            continue

        cleaned_parameters_metadata[clean_name] = param_metadata
    parameters_metadata = cleaned_parameters_metadata

    # Step 5: Transform to original weights
    original_state_dict = to_original_weights(
        transformation_specs,
        sharded_state_dicts,
        parameters_metadata,
    )

    # Step 6: Unmerge LoRA adapters to restore model state
    model.unmerge_adapter()
    torch_xla.sync()

    return original_state_dict
