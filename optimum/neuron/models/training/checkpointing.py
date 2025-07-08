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

import json
import os
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from transformers.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    SAFE_WEIGHTS_NAME,
    WEIGHTS_INDEX_NAME,
    WEIGHTS_NAME,
)

from ...utils.import_utils import is_peft_available
from .modeling_utils import MODEL_PARALLEL_SHARDS_DIR_NAME
from .transformations_utils import ModelWeightTransformationSpecs, to_original_weights


if is_peft_available():
    from peft.utils.constants import (
        SAFETENSORS_WEIGHTS_NAME as PEFT_SAFETENSORS_WEIGHTS_NAME,
    )
    from peft.utils.constants import (
        WEIGHTS_NAME as PEFT_WEIGHTS_NAME,
    )
else:
    PEFT_SAFETENSORS_WEIGHTS_NAME = PEFT_WEIGHTS_NAME = ""


def xser_load_on_cpu(path: str):
    """
    Modified version from neuronx_distributed `_xser_load` function load located at:
    https://github.com/aws-neuron/neuronx-distributed/blob/e83494557cb4c5b7e185ccf6c9240bfed9a1993d/src/neuronx_distributed/parallel_layers/checkpointing.py#L252
    Instead of moving the loaded tensors to the XLA device, it keeps them on CPU.
    """
    import torch_xla.core.xla_model as xm
    import torch_xla.utils.serialization as xser

    ref_data = torch.load(path, weights_only=False)

    def convert_fn(tensors):
        rewritten_tensors = []
        for t in tensors:
            rewritten_tensors.append(torch.load(os.path.join(path + ".tensors", "tensor_{}.pt".format(t.tid))))
        return rewritten_tensors

    def select_fn(v):
        return type(v) is xser.TensorReference

    return xm.ToXlaTensorArena(convert_fn, select_fn).transform(ref_data)


def consolidate_tensor_parallel_checkpoints(
    sharded_checkpoints: List[Path],
    load_function: Callable[[Union[str, Path]], Dict[str, Any]],
    metadata: Dict[str, Any],
    adapter_name: Optional[str] = None,
) -> Dict[str, "torch.Tensor"]:
    state_dicts = []
    sharded_checkpoints = sorted(sharded_checkpoints)
    for sharded_checkpoint in sharded_checkpoints:
        if not sharded_checkpoint.is_file():
            continue
        state_dicts.append(load_function(sharded_checkpoint.as_posix()))

    parameters_metadata = metadata["parameters"]
    transformation_specs_metadata = metadata["model_weight_transformation_specs"]

    # We recreate the transformation specs from the metadata.
    transformations_specs = []
    for specs_metadata in transformation_specs_metadata:
        specs = ModelWeightTransformationSpecs.from_metadata(specs_metadata)
        transformations_specs.append(specs)

    # We transform the sharded state dicts as follows:
    # [state_dict_tp_rank_0, state_dict_tp_rank_1, ...]
    #   ->  {
    #           key: [state_dict_tp_rank_0[key], state_dict_tp_rank_1[key], ...],
    #           for key in state_dict_tp_rank_0.keys()
    #       }
    parameter_names = state_dicts[0].keys()
    sharded_state_dicts = {name: [state_dict[name] for state_dict in state_dicts] for name in parameter_names}

    consolidated_state_dict = to_original_weights(
        transformations_specs, sharded_state_dicts, parameters_metadata, adapter_name=adapter_name
    )

    return consolidated_state_dict


def consolidate_model_parallel_checkpoints(
    checkpoint_dir: Path, adapter_name: Optional[str] = None
) -> Dict[str, "torch.Tensor"]:
    model_checkpoint_dir = checkpoint_dir / "model"

    # Case 1: the checkpoint was saved with xser.
    sharded_checkpoints = list(model_checkpoint_dir.glob("dp_rank*.tensors"))
    if sharded_checkpoints:
        sharded_checkpoints = model_checkpoint_dir.glob("dp_rank_*")
        sharded_checkpoints = [
            p for p in sharded_checkpoints if not (p.name.endswith("info.pt") or p.name.endswith("tensors"))
        ]
        load_function = xser_load_on_cpu

    # Case 2: If no file was found, maybe the checkpoint was saved without xser.
    if not sharded_checkpoints:
        sharded_checkpoints = list(model_checkpoint_dir.glob("dp_rank_*.pt"))
        load_function = partial(torch.load, weights_only=True)

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
        if (checkpoint_dir / f"mp_metadata_pp_rank_{pp_rank}.pt").is_file():
            metadatas.append(torch.load(checkpoint_dir / f"mp_metadata_pp_rank_{pp_rank}.pt"))
        else:
            with open(checkpoint_dir / f"mp_metadata_pp_rank_{pp_rank}.json") as fp:
                metadatas.append(json.load(fp))

    consolidated_state_dict = {}
    for pp_rank, checkpoint_group_for_pp_rank in enumerate(checkpoints_grouped_by_pp_ranks):
        consolidated_for_pp_rank = consolidate_tensor_parallel_checkpoints(
            checkpoint_group_for_pp_rank,
            load_function,
            metadatas[pp_rank],
            adapter_name=adapter_name,
        )
        consolidated_state_dict.update(**consolidated_for_pp_rank)

    for key, tensor in consolidated_state_dict.items():
        consolidated_state_dict[key] = tensor

    return consolidated_state_dict


def consolidate_model_parallel_checkpoints_to_unified_checkpoint(
    checkpoint_dir: Union[str, Path],
    output_dir: Union[str, Path],
    save_format: Literal["pytorch", "safetensors"] = "safetensors",
):
    from safetensors.torch import save_file

    # We import here to avoid circular import.
    from ...peft.peft_model import ADAPTER_MODEL_PARALLEL_SHARDS_DIR_NAME

    if not isinstance(checkpoint_dir, Path):
        checkpoint_dir = Path(checkpoint_dir)

    directories = list(checkpoint_dir.iterdir())
    directories_to_consolidate = []
    if checkpoint_dir.name != MODEL_PARALLEL_SHARDS_DIR_NAME:
        if (checkpoint_dir / MODEL_PARALLEL_SHARDS_DIR_NAME).is_dir():
            directories_to_consolidate = [checkpoint_dir / MODEL_PARALLEL_SHARDS_DIR_NAME]
        else:
            for dir in directories:
                if dir.is_dir() and dir.name.startswith("adapter_"):
                    directories_to_consolidate.append(dir / ADAPTER_MODEL_PARALLEL_SHARDS_DIR_NAME)
        if not directories_to_consolidate:
            raise ValueError(f"Could not find the tensor parallel shards from {checkpoint_dir}")
    else:
        directories_to_consolidate = [checkpoint_dir]

    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for checkpoint_dir in directories_to_consolidate:
        # We need to go one level up because the checkpoint directory is at the shards level here.
        parent_dir = checkpoint_dir.parent
        current_output_dir = output_dir
        is_adapter_model = parent_dir.name.startswith("adapter_")
        adapter_name = None
        if is_adapter_model:
            safe_weights_name = PEFT_SAFETENSORS_WEIGHTS_NAME
            weights_name = PEFT_WEIGHTS_NAME
            if parent_dir.name != "adapter_default":
                adapter_name = parent_dir.name.split("_", maxsplit=1)[-1]
                current_output_dir = output_dir / adapter_name
            else:
                adapter_name = "default"
        else:
            safe_weights_name = SAFE_WEIGHTS_NAME
            weights_name = WEIGHTS_NAME

        current_output_dir.mkdir(parents=True, exist_ok=True)

        state_dict = consolidate_model_parallel_checkpoints(checkpoint_dir, adapter_name=adapter_name)
        state_dict_split = split_torch_state_dict_into_shards(
            state_dict, filename_pattern=safe_weights_name if save_format == "safetensors" else weights_name
        )
        # Save index if sharded
        if state_dict_split.is_sharded:
            index = {
                "metadata": state_dict_split.metadata,
                "weight_map": state_dict_split.tensor_to_filename,
            }
            save_index_file = SAFE_WEIGHTS_INDEX_NAME if save_format == "safetensors" else WEIGHTS_INDEX_NAME
            with open(current_output_dir / save_index_file, "w") as fp:
                content = json.dumps(index, indent=2, sort_keys=True) + "\n"
                fp.write(content)
        # Save the model
        filename_to_tensors = state_dict_split.filename_to_tensors.items()
        for shard_file, tensors in filename_to_tensors:
            shard = {}
            for tensor in tensors:
                shard[tensor] = state_dict[tensor].contiguous()
                del state_dict[tensor]
            if save_format == "safetensors":
                save_file(shard, current_output_dir / shard_file, metadata={"format": "pt"})
            else:
                torch.save(shard, current_output_dir / shard_file)
