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
from typing import Dict, Literal, Union

import torch
from transformers.modeling_utils import shard_checkpoint
from transformers.utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME, WEIGHTS_INDEX_NAME, WEIGHTS_NAME

from ..utils.require_utils import requires_safetensors
from .utils import TENSOR_PARALLEL_SHARDS_DIR_NAME, ParameterMetadata


def consolidate_tensor_parallel_checkpoints(checkpoint_dir: Union[str, Path]) -> Dict[str, "torch.Tensor"]:
    if not isinstance(checkpoint_dir, Path):
        checkpoint_dir = Path(checkpoint_dir)

    if checkpoint_dir.name != TENSOR_PARALLEL_SHARDS_DIR_NAME:
        if (checkpoint_dir / TENSOR_PARALLEL_SHARDS_DIR_NAME).is_dir():
            checkpoint_dir = checkpoint_dir / TENSOR_PARALLEL_SHARDS_DIR_NAME
        else:
            raise ValueError(f"Could not find the tensor parallel shards from {checkpoint_dir}")

    state_dicts = []

    for sharded_checkpoint in sorted(checkpoint_dir.glob("tp_rank_*/checkpoint.pt")):
        if not sharded_checkpoint.is_file():
            continue
        state_dicts.append(torch.load(sharded_checkpoint))

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

    consolidated_state_dict = {}
    for name in parameter_names:
        # For now all parameter metadatas are equal so it is enough to take the first element.
        # This might not be the case anymore when `ParameterMetadata` uses slices.
        metadata = sharded_metadatas[name][0]
        if metadata.is_tied:
            consolidated_state_dict[name] = state_dicts[0]["model"][name]
        else:
            params = [state_dict["model"][name] for state_dict in state_dicts]
            consolidated_state_dict[name] = torch.cat(
                params,
                dim=metadata.partition_dim,
            )

    return consolidated_state_dict


@requires_safetensors
def consolidate_tensor_parallel_checkpoints_to_unified_checkpoint(
    checkpoint_dir: Union[str, Path],
    output_dir: Union[str, Path],
    save_format: Literal["pytorch", "safetensors"] = "safetensors",
):
    from safetensors.torch import save_file

    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    state_dict = consolidate_tensor_parallel_checkpoints(checkpoint_dir)
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
