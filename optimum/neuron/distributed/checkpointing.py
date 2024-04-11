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
from .utils import MODEL_PARALLEL_SHARDS_DIR_NAME, ParameterMetadata, compute_query_indices_for_rank


def create_gqa_query_or_output_projection_weight_from_full_weight(
    full_weight: torch.Tensor,
    tp_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    kv_size_multiplier: int,
    query_or_output: Union[Literal["query"], Literal["output"]],
):
    assert query_or_output in ["query", "output"]
    assert full_weight.device == torch.device("cpu")
    if query_or_output == "query":
        hidden_size = full_weight.size(1)
    else:
        hidden_size = full_weight.size(0)
        full_weight = full_weight.transpose(0, 1)

    indices = [
        compute_query_indices_for_rank(tp_size, tp_rank, num_attention_heads, num_key_value_heads, kv_size_multiplier)
        for tp_rank in range(tp_size)
    ]
    indices = torch.cat(indices, dim=0)
    reversed_indices = torch.sort(indices, dim=0).indices

    full_weight = full_weight.reshape(num_attention_heads, -1, hidden_size)
    full_weight = full_weight[reversed_indices]
    full_weight = full_weight.reshape(-1, hidden_size)

    if query_or_output == "output":
        full_weight = full_weight.transpose(0, 1)

    return full_weight


def consolidate_tensor_parallel_checkpoints(
    sharded_checkpoints: List[Path],
    load_function: Callable[[Union[str, Path]], Dict[str, Any]],
    metadata: Dict[str, Any],
) -> Dict[str, "torch.Tensor"]:
    state_dicts = []
    sharded_checkpoints = sorted(sharded_checkpoints)
    for sharded_checkpoint in sharded_checkpoints:
        if not sharded_checkpoint.is_file():
            continue
        state_dicts.append(load_function(sharded_checkpoint.as_posix()))

    parameter_names = state_dicts[0].keys()
    sharded_metadatas = {
        name: (
            ParameterMetadata(**metadata["sharded_metadata"][name])
            if name in metadata["sharded_metadata"]
            else ParameterMetadata("tied")
        )
        for name in parameter_names
    }

    gqa_qkv_metadata = metadata["gqa_qkv_metadata"]
    original_parameter_names_to_gqa_qkv_names = gqa_qkv_metadata["original_names_to_gqa_qkv_names"]
    gqa_qkv_output_projections_names = gqa_qkv_metadata["output_projections_names"]
    gqa_qkv_names_to_original_names = {v: k for k, v in original_parameter_names_to_gqa_qkv_names.items()}

    consolidated_state_dict = {}
    for name in parameter_names:
        # We need to handle the mapping between the GQA parameter names and the original names.
        is_gqa_qkv_weight = name in gqa_qkv_names_to_original_names
        if is_gqa_qkv_weight:
            original_name = gqa_qkv_names_to_original_names[name]
            weight_name = name.rsplit(".", maxsplit=1)[1]
        else:
            original_name = name
            weight_name = ""  # Not needed.

        # For now all parameter metadatas are equal so it is enough to take the first element.
        # This might not be the case anymore when `ParameterMetadata` uses slices.
        sharded_metadata = sharded_metadatas[name]
        if sharded_metadata.is_tied:
            consolidated_state_dict[original_name] = state_dicts[0][name].to("cpu")
        else:
            weights = [state_dict[name] for state_dict in state_dicts]
            tp_size = len(weights)
            full_weight = torch.cat(
                weights,
                dim=sharded_metadata.partition_dim,
            )
            full_weight = full_weight.to("cpu")
            if weight_name in ["weight_k", "weight_v", "bias_k", "bias_v"]:
                full_weight = (
                    torch.chunk(full_weight, gqa_qkv_metadata["kv_size_multiplier"], dim=0)[0].detach().clone()
                )
            elif weight_name == "weight_q" or original_name in gqa_qkv_output_projections_names:
                full_weight = create_gqa_query_or_output_projection_weight_from_full_weight(
                    full_weight,
                    tp_size,
                    gqa_qkv_metadata["num_attention_heads"],
                    gqa_qkv_metadata["num_key_value_heads"],
                    gqa_qkv_metadata["kv_size_multiplier"],
                    "query" if weight_name == "weight_q" else "output",
                )
            consolidated_state_dict[original_name] = full_weight

    return consolidated_state_dict


@requires_neuronx_distributed
def consolidate_model_parallel_checkpoints(checkpoint_dir: Union[str, Path]) -> Dict[str, "torch.Tensor"]:
    from neuronx_distributed.parallel_layers.checkpointing import _xser_load

    if not isinstance(checkpoint_dir, Path):
        checkpoint_dir = Path(checkpoint_dir)

    if checkpoint_dir.name != MODEL_PARALLEL_SHARDS_DIR_NAME:
        if (checkpoint_dir / MODEL_PARALLEL_SHARDS_DIR_NAME).is_dir():
            checkpoint_dir = checkpoint_dir / MODEL_PARALLEL_SHARDS_DIR_NAME
        else:
            raise ValueError(f"Could not find the tensor parallel shards from {checkpoint_dir}")

    model_checkpoint_dir = checkpoint_dir / "model"

    # Case 1: the checkpoint was saved with xser.
    sharded_checkpoints = list(model_checkpoint_dir.glob("dp_rank*.tensors"))
    if sharded_checkpoints:
        sharded_checkpoints = model_checkpoint_dir.glob("dp_rank_*")
        sharded_checkpoints = [
            p for p in sharded_checkpoints if not (p.name.endswith("info.pt") or p.name.endswith("tensors"))
        ]
        load_function = _xser_load

    # Case 2: If no file was found, maybe the checkpoint was saved without xser.
    if not sharded_checkpoints:
        sharded_checkpoints = list(model_checkpoint_dir.glob("dp_rank_*.pt"))
        load_function = torch.load

    if not sharded_checkpoints:
        raise ValueError(f"Could not find any sharded checkpoint in {model_checkpoint_dir.as_posix()}")

    pp_size = max((int(checkpoint_path.stem[-2:]) for checkpoint_path in sharded_checkpoints)) + 1
    checkpoints_grouped_by_pp_ranks = [[] for _ in range(pp_size)]
    metadatas = []
    for pp_rank in range(pp_size):
        for checkpoint_path in sharded_checkpoints:
            checkpoint_name = checkpoint_path.stem
            if int(checkpoint_name[-2:]) == pp_rank:
                checkpoints_grouped_by_pp_ranks[pp_rank].append(checkpoint_path)
        metadatas.append(torch.load(checkpoint_dir / f"mp_metadata_pp_rank_{pp_rank}.pt"))

    consolidated_state_dict = {}
    for pp_rank, checkpoint_group_for_pp_rank in enumerate(checkpoints_grouped_by_pp_ranks):
        consolidated_for_pp_rank = consolidate_tensor_parallel_checkpoints(
            checkpoint_group_for_pp_rank, load_function, metadatas[pp_rank]
        )
        consolidated_state_dict.update(**consolidated_for_pp_rank)

    for key, tensor in consolidated_state_dict.items():
        consolidated_state_dict[key] = tensor

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
