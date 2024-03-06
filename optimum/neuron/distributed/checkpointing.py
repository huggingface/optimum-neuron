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
"""Functions handling checkpointing under parallel settings."""

import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Union

import torch
from transformers.modeling_utils import shard_checkpoint
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME

from ..utils.require_utils import requires_neuronx_distributed, requires_safetensors
from .utils import TENSOR_PARALLEL_SHARDS_DIR_NAME, ParameterMetadata


def consolidate_tensor_parallel_checkpoints(
    sharded_checkpoints: List[Path], load_function: Callable[[Union[str, Path]], Dict[str, Any]]
) -> Dict[str, "torch.Tensor"]:
    state_dicts = []
    sharded_checkpoints = sorted(sharded_checkpoints)
    for sharded_checkpoint in sharded_checkpoints:
        if not sharded_checkpoint.is_file():
            continue
        state_dicts.append(load_function(sharded_checkpoint.as_posix()))

    parameter_names = state_dicts[0]["model"].keys()
    sharded_metadatas = {
        name: [
            (
                ParameterMetadata(**state_dict["sharded_metadata"][name])
                if name in state_dict["sharded_metadata"]
                else ParameterMetadata("tied")
            )
            for state_dict in state_dicts
        ]
        for name in parameter_names
    }

    gqa_qkv_metadata = state_dicts[0]["gqa_qkv_metadata"]
    kv_size_multiplier = gqa_qkv_metadata["kv_size_multiplier"]
    original_parameter_names_to_gqa_qkv_names = gqa_qkv_metadata["original_names_to_gqa_qkv_names"]
    gqa_qkv_names_to_original_names = {v: k for k, v in original_parameter_names_to_gqa_qkv_names.items()}

    consolidated_state_dict = {}
    for name in parameter_names:
        is_gqa_qkv_weight = name in gqa_qkv_names_to_original_names
        if is_gqa_qkv_weight:
            original_name = gqa_qkv_names_to_original_names[name]
            weight_name = name.rsplit(".", maxsplit=1)[1]
        else:
            original_name = name
            weight_name = ""  # Not needed.
        # For now all parameter metadatas are equal so it is enough to take the first element.
        # This might not be the case anymore when `ParameterMetadata` uses slices.
        metadata = sharded_metadatas[name][0]
        if metadata.is_tied:
            consolidated_state_dict[original_name] = state_dicts[0]["model"][name]
        else:
            params = [state_dict["model"][name] for state_dict in state_dicts]

            full_param = torch.cat(
                params,
                dim=metadata.partition_dim,
            )

            if weight_name in ["weight_k", "weight_v", "bias_k", "bias_v"]:
                full_param = torch.chunk(full_param, kv_size_multiplier, dim=0)[0].clone()

            consolidated_state_dict[original_name] = full_param

    return consolidated_state_dict


@requires_neuronx_distributed
def consolidate_model_parallel_checkpoints(checkpoint_dir: Union[str, Path]) -> Dict[str, "torch.Tensor"]:
    from neuronx_distributed.parallel_layers.checkpointing import _xser_load

    if not isinstance(checkpoint_dir, Path):
        checkpoint_dir = Path(checkpoint_dir)

    if checkpoint_dir.name != TENSOR_PARALLEL_SHARDS_DIR_NAME:
        if (checkpoint_dir / TENSOR_PARALLEL_SHARDS_DIR_NAME).is_dir():
            checkpoint_dir = checkpoint_dir / TENSOR_PARALLEL_SHARDS_DIR_NAME
        else:
            raise ValueError(f"Could not find the tensor parallel shards from {checkpoint_dir}")

    # Regular case: the checkpoint was saved without xser.
    sharded_checkpoints = list(checkpoint_dir.glob("tp_rank_*/checkpoint.pt"))
    load_function = torch.load

    # If no file was found, maybe the checkpoint was saved with xser.
    if not sharded_checkpoints:
        sharded_checkpoints = checkpoint_dir.glob("tp_rank_*")
        sharded_checkpoints = [p for p in sharded_checkpoints if not p.name.endswith("tensors")]
        load_function = _xser_load

    if not sharded_checkpoints:
        raise ValueError(f"Could not find any sharded checkpoint in {checkpoint_dir.as_posix()}")

    def get_checkpoint_name(checkpoint_path: Path) -> str:
        name = checkpoint_path.name
        if name == "checkpoint.pt":
            name = checkpoint_path.parent.name
        return name

    pp_size = max((int(get_checkpoint_name(checkpoint_path)[-2:]) for checkpoint_path in sharded_checkpoints)) + 1
    checkpoints_grouped_by_pp_ranks = [[] for _ in range(pp_size)]
    for pp_rank in range(pp_size):
        for checkpoint_path in sharded_checkpoints:
            checkpoint_name = get_checkpoint_name(checkpoint_path)
            if int(checkpoint_name[-2:]) == pp_rank:
                checkpoints_grouped_by_pp_ranks[pp_rank].append(checkpoint_path)

    consolidated_state_dict = {}
    for checkpoint_group_for_pp_rank in checkpoints_grouped_by_pp_ranks:
        consolidated_for_pp_rank = consolidate_tensor_parallel_checkpoints(checkpoint_group_for_pp_rank, load_function)
        consolidated_state_dict.update(**consolidated_for_pp_rank)

    for key, tensor in consolidated_state_dict.items():
        consolidated_state_dict[key] = tensor.to("cpu")

    return consolidated_state_dict


@requires_safetensors
def consolidate_model_parallel_checkpoints_to_unified_checkpoint(
    checkpoint_dir: Union[str, Path],
    output_dir: Union[str, Path],
    save_format: Literal["pytorch", "safetensors"] = "safetensors",
):
    from safetensors.torch import save_file

    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    state_dict = consolidate_model_parallel_checkpoints(checkpoint_dir)
    shards, index = shard_checkpoint(
        state_dict, weights_name=SAFE_WEIGHTS_NAME if save_format == "safetensors" else WEIGHTS_NAME
    )
    for shard_file, shard in shards.items():
        if save_format == "safetensors":
            save_file(shard, output_dir / shard_file, metadata={"format": "pt"})
        else:
            torch.save(shard, output_dir / shard_file)
    if index is not None:
        save_index_file = SAFE_WEIGHTS_INDEX_NAME if save_format == "safetensors" else WEIGHTS_INDEX_NAME
        with open(output_dir / save_index_file, "w") as fp:
            content = json.dumps(index, indent=2, sort_keys=True) + "\n"
            fp.write(content)
