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
"""Utilities for performing parallelism with `neuronx_distributed`"""

import contextlib
import functools
import os
from pathlib import Path
from typing import Dict, Literal, Optional, Tuple, Union

import torch
from transformers import PretrainedConfig

from ..utils import DynamicPatch, Patcher, is_neuronx_distributed_available
from ..utils.misc import download_checkpoints_in_cache


if is_neuronx_distributed_available():
    from neuronx_distributed.parallel_layers import layers
    from neuronx_distributed.parallel_layers.parallel_state import get_tensor_model_parallel_rank

TENSOR_PARALLEL_SHARDS_DIR_NAME = "tensor_parallel_shards"


def embedding_to_parallel_embedding(
    embedding_layer: "torch.nn.Embedding",
    lm_head_layer: Optional["torch.nn.Linear"] = None,
    orig_to_parallel: Optional[Dict[int, "torch.nn.Parameter"]] = None,
) -> Union["layers.ParallelEmbedding", Tuple["layers.ParallelEmbedding", "layers.ColumnParallelLinear"]]:
    parallel_embedding_layer = layers.ParallelEmbedding(
        embedding_layer.num_embeddings,
        embedding_layer.embedding_dim,
        dtype=embedding_layer.weight.dtype,
    )

    tp_rank = get_tensor_model_parallel_rank()
    row_size, _ = parallel_embedding_layer.weight.shape

    is_tied = False
    if lm_head_layer is not None:
        is_tied = id(embedding_layer.weight.data) == id(lm_head_layer.weight.data)

    with torch.no_grad():
        parallel_embedding_layer.weight.copy_(embedding_layer.weight[tp_rank * row_size : (tp_rank + 1) * row_size, :])
        if lm_head_layer is not None:
            parallel_lm_head_layer = linear_to_parallel_linear(
                lm_head_layer,
                "column",
                gather_output=True,
                orig_to_parallel=orig_to_parallel if not is_tied else None,
            )
            if is_tied:
                parallel_lm_head_layer.weight = parallel_embedding_layer.weight

    if orig_to_parallel:
        orig_to_parallel[id(embedding_layer.weight)] = parallel_embedding_layer.weight

    del embedding_layer.weight

    if lm_head_layer is None:
        return parallel_embedding_layer
    return parallel_embedding_layer, parallel_lm_head_layer


def linear_to_parallel_linear(
    linear_layer: "torch.nn.Linear",
    axis: Union[Literal["row"], Literal["column"]],
    input_is_parallel: bool = False,
    gather_output: bool = True,
    orig_to_parallel: Optional[Dict[int, "torch.nn.Parameter"]] = None,
) -> Union["layers.RowParallelLinear", "layers.ColumnParallelLinear"]:
    if axis not in ["row", "column"]:
        raise ValueError(f'axis must either be "row" or "column", but {axis} was given here.')

    kwargs = {}
    if axis == "row":
        parallel_linear_class = layers.RowParallelLinear
        kwargs["input_is_parallel"] = input_is_parallel
    else:
        parallel_linear_class = layers.ColumnParallelLinear
        kwargs["gather_output"] = gather_output

    kwargs["dtype"] = linear_layer.weight.dtype
    kwargs["bias"] = linear_layer.bias is not None

    parallel_linear_layer = parallel_linear_class(linear_layer.in_features, linear_layer.out_features, **kwargs)

    tp_rank = get_tensor_model_parallel_rank()
    row_size, col_size = parallel_linear_layer.weight.shape

    with torch.no_grad():
        if axis == "row":
            parallel_linear_layer.weight.copy_(linear_layer.weight[:, tp_rank * col_size : (tp_rank + 1) * col_size])
            if linear_layer.bias is not None:
                parallel_linear_layer.bias.copy_(linear_layer.bias)
                if orig_to_parallel is not None:
                    orig_to_parallel[id(linear_layer.bias)] = parallel_linear_layer.bias
        else:
            parallel_linear_layer.weight.copy_(linear_layer.weight[tp_rank * row_size : (tp_rank + 1) * row_size, :])
            if linear_layer.bias is not None:
                parallel_linear_layer.bias.copy_(linear_layer.bias[tp_rank * row_size : (tp_rank + 1) * row_size])
                if orig_to_parallel is not None:
                    orig_to_parallel[id(linear_layer.bias)] = parallel_linear_layer.bias

    if orig_to_parallel is not None:
        orig_to_parallel[id(linear_layer.weight)] = parallel_linear_layer.weight

    return parallel_linear_layer


@classmethod
def from_pretrained_for_tp(
    cls,
    pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
    *model_args,
    config: Optional[Union[PretrainedConfig, str, os.PathLike]] = None,
    cache_dir: Optional[Union[str, os.PathLike]] = None,
    ignore_mismatched_sizes: bool = False,
    force_download: bool = False,
    local_files_only: bool = False,
    token: Optional[Union[str, bool]] = None,
    revision: str = "main",
    use_safetensors: bool = None,
    **kwargs,
):
    kwargs.pop("state_dict", None)
    kwargs.pop("from_tf", False)
    kwargs.pop("from_flax", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)
    kwargs.pop("output_loading_info", False)
    kwargs.pop("use_auth_token", None)
    kwargs.pop("trust_remote_code", None)
    _ = kwargs.pop("mirror", None)
    from_pipeline = kwargs.pop("_from_pipeline", None)
    from_auto_class = kwargs.pop("_from_auto", False)
    _fast_init = kwargs.pop("_fast_init", True)
    kwargs.pop("torch_dtype", None)
    kwargs.pop("low_cpu_mem_usage", None)
    kwargs.pop("device_map", None)
    kwargs.pop("max_memory", None)
    kwargs.pop("offload_folder", None)
    kwargs.pop("offload_state_dict", False)
    kwargs.pop("load_in_8bit", False)
    kwargs.pop("load_in_4bit", False)
    kwargs.pop("quantization_config", None)
    subfolder = kwargs.pop("subfolder", "")
    kwargs.pop("_commit_hash", None)
    kwargs.pop("variant", None)

    filenames, sharded_metadata = download_checkpoints_in_cache(
        pretrained_model_name_or_path,
        cache_dir=cache_dir,
        force_download=force_download,
        local_files_only=local_files_only,
        token=token,
        revision=revision,
        use_safetensors=use_safetensors,
        convert_to_safetensors=True,
        **kwargs,
    )

    if not isinstance(config, PretrainedConfig):
        config_path = config if config is not None else pretrained_model_name_or_path
        config, model_kwargs = cls.config_class.from_pretrained(
            config_path,
            cache_dir=cache_dir,
            return_unused_kwargs=True,
            force_download=force_download,
            resume_download=resume_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            _from_auto=from_auto_class,
            _from_pipeline=from_pipeline,
            **kwargs,
        )
    else:
        model_kwargs = kwargs

    model = cls(config, *model_args, **model_kwargs)

    if sharded_metadata:
        filenames = list(map(Path, filenames))
        safetensors_filename_to_full_filename = {path.name: path for path in filenames}
        weight_map = sharded_metadata["weight_map"]
        for weight_name, safetensors_filename in sharded_metadata.items():
            weight_map[weight_name] = safetensors_filename_to_full_filename[safetensors_filename]
    else:
        filename = Path(filenames)
        weight_map = {name: filename for name, _ in model.named_parameters()}

    model._weight_map = weight_map

    return model


@contextlib.contextmanager
def prepare_for_tp():
    def meta_init(init_fn):
        @functools.wraps(init_fn)
        def wrapper(*args, **kwargs):
            kwargs["device"] = kwargs.pop("device", torch.device("meta"))
            return init_fn(*args, **kwargs)

        return wrapper

    meta_init_patch = DynamicPatch(meta_init)

    patching_specs = [
        ("torch.nn.Embedding.__init__", meta_init_patch),
        ("torch.nn.Linear.__init__", meta_init_patch),
        ("transformers.modeling_utils.PreTrainedModel.from_pretrained", from_pretrained_for_tp),
    ]
    patcher = Patcher(patching_specs=patching_specs)
    with patcher:
        try:
            yield
        finally:
            print("DONE")
