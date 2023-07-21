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
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Literal, Optional, Tuple, Type, Union

import torch
from transformers import PretrainedConfig

from ..utils import DynamicPatch, Patcher, is_neuronx_distributed_available, is_torch_xla_available
from ..utils.misc import download_checkpoints_in_cache
from ..utils.require_utils import requires_safetensors, requires_torch_xla


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm
    from torch_xla.distributed.zero_redundancy_optimizer import ZeroRedundancyOptimizer
else:
    ZeroRedundancyOptimizer = object


if is_neuronx_distributed_available():
    from neuronx_distributed.parallel_layers import layers
    from neuronx_distributed.parallel_layers.parallel_state import (
        get_data_parallel_group,
        get_data_parallel_rank,
        get_data_parallel_size,
        get_tensor_model_parallel_rank,
        model_parallel_is_initialized,
    )

TENSOR_PARALLEL_SHARDS_DIR_NAME = "tensor_parallel_shards"


@dataclass
class WeightInformation:
    """
    Describes the information about the weight of a parameter.

    Attributes:
        - filename (`Union[str, Path]`) -- The name of the `safetensors` checkpoint file containing the weights of the
        parameter.
        - qualified_name (`str`) -- The fully qualified name of the parameter in the model hierarchy.
        - device (`torch.device`) -- The device to put the weight on, defaults to `torch.device("cpu")` if left
        unspecified.
    """

    filename: Union[str, Path]
    qualified_name: str
    device: Optional[torch.device] = None

    def __post_init__(self):
        if self.device is None:
            self.device = torch.device("cpu")


@requires_safetensors
def load_tensor_for_weight(
    weight_info: WeightInformation, tensor_slices: Optional[Tuple[Optional[Tuple[int, ...]], ...]] = None
) -> torch.Tensor:
    """
    Loads potentially sliced weights from a `safetensors` checkpoint.

    Args:
        weight_info (`WeightInformation`):
            The information about the weight, this is used to know where to load the weight from.
        tensor_slices (`Optional[Tuple[Optional[Tuple[int, ...]]]]`, defaults to `None`):
            If specified it will be interpreted as the slices to load from the saved weight, it must be a tuple of either:
                1. A tuple of up to three ints: they will be used to compute a slice.
                2. None: it will be interpreted as taking the whole dimension.
            Example:
                t = torch.rand(4, 3, 2) with tensor_slices = ((2, 3), None, (1,)) will result in t[2:3, :, :1].

    Returns:
        `torch.Tensor`: The loaded tensor.
    """
    from safetensors import safe_open

    device = str(weight_info.device)
    with safe_open(weight_info.filename, framework="pt", device=device) as fp:
        if tensor_slices is None:
            tensor = fp.get_tensor(weight_info.qualified_name)
        else:
            tensor_slice = fp.get_slice(weight_info.qualified_name)
            slices = [slice(*slice_) if slice_ is not None else slice(None, None, None) for slice_ in tensor_slices]
            tensor = tensor_slice[slices].contiguous()
            # This is needed to make sure tensor.numel() == tensor.storage().size().
            tensor = torch.empty_like(tensor).copy_(tensor)

    return tensor


def _validate_weight_info_device_matches_specified_device(device: "torch.device", weight_info: WeightInformation):
    if device != weight_info.device:
        raise ValueError(
            f"The specfified device must match the device in the `WeightInformation` but got {device} and "
            f"{weight_info.device}, the `WeightInformation` object is: {weight_info}."
        )


def embedding_to_parallel_embedding(
    embedding_layer: "torch.nn.Embedding",
    lm_head_layer: Optional["torch.nn.Linear"] = None,
    embedding_weight_info: Optional[WeightInformation] = None,
    lm_head_weight_info: Optional[WeightInformation] = None,
    lm_head_bias_weight_info: Optional[WeightInformation] = None,
    orig_to_parallel: Optional[Dict[int, "torch.nn.Parameter"]] = None,
    device: Optional["torch.device"] = None,
) -> Union["layers.ParallelEmbedding", Tuple["layers.ParallelEmbedding", "layers.ColumnParallelLinear"]]:
    """
    Helper function that creates a `neuronx_distributed.parallel_layers.layers.ParallelEmbedding` from a regular
    `torch.nn.Embedding`.

    It can also handle the case where the embedding layer is tied to a linear layer.

    Args:
        embedding_layer (`torch.nn.Embedding`):
            The embedding layer to parallelize.
        lm_head_layer (`Optional[torch.nn.Linear]`, defaults to `None`):
            If specified, the linear layer tied to the embedding.
        embedding_weight_info (`Optional[WeightInformation]`, defaults to `None`):
            Information about which checkpoint file the embedding weights are stored in.
        lm_head_weight_info (`Optional[WeightInformation]`, defaults to `None`):
            Information about which checkpoint file the tied linear projection weights are stored in.
        lm_head_bias_weight_info (`Optional[WeightInformation]`, defaults to `None`):
            Information about which checkpoint file the tied linear projection bias is stored in.
        orig_to_parallel (`Optional[Dict[int, torch.nn.Parameter]]`, defaults to `None`):
            A dictionary to fill. It maps a former parameter id to its parallel version.
            It might be deprecated soon.
        device (`Optional[torch.device]`, defaults to `None`):
            The device where the new parallel layer should be put.

    Returns:
        `Union[ParallelEmbedding, Tuple[ParallelEmbedding", ColumnParallelLinear]]`: The parallel embedding and the
        parallel linear projection if specified.
    """
    device = device if device is not None else torch.device("cpu")

    for weight_info in [embedding_weight_info, lm_head_weight_info, lm_head_bias_weight_info]:
        if weight_info is None:
            continue
        _validate_weight_info_device_matches_specified_device(device, weight_info)

    parallel_embedding_layer = layers.ParallelEmbedding(
        embedding_layer.num_embeddings,
        embedding_layer.embedding_dim,
        dtype=embedding_layer.weight.dtype,
        device=device,
    )

    tp_rank = get_tensor_model_parallel_rank()
    row_size, _ = parallel_embedding_layer.weight.shape

    is_tied = False
    if lm_head_layer is not None:
        is_tied = id(embedding_layer.weight) == id(lm_head_layer.weight)

    embedding_weight_to_tie = parallel_embedding_layer.weight if is_tied else None

    with torch.no_grad():
        if embedding_weight_info is not None:
            weight_data = load_tensor_for_weight(
                embedding_weight_info,
                tensor_slices=(
                    (tp_rank * row_size, (tp_rank + 1) * row_size),
                    None,
                ),
            )
            parallel_embedding_layer.weight.data = weight_data
        else:
            parallel_embedding_layer.weight.copy_(
                embedding_layer.weight[tp_rank * row_size : (tp_rank + 1) * row_size, :]
            )

        if lm_head_layer is not None:
            parallel_lm_head_layer = linear_to_parallel_linear(
                lm_head_layer,
                "column",
                linear_layer_weight_info=lm_head_weight_info,
                linear_layer_bias_weight_info=lm_head_bias_weight_info,
                embedding_weight_to_tie=embedding_weight_to_tie,
                gather_output=True,
                orig_to_parallel=orig_to_parallel if not is_tied else None,
                device=device,
            )

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
    linear_layer_weight_info: Optional[WeightInformation] = None,
    linear_layer_bias_weight_info: Optional[WeightInformation] = None,
    embedding_weight_to_tie: Optional["torch.nn.Parameter"] = None,
    orig_to_parallel: Optional[Dict[int, "torch.nn.Parameter"]] = None,
    device: Optional["torch.device"] = None,
) -> Union["layers.RowParallelLinear", "layers.ColumnParallelLinear"]:
    """
    Helper function that creates a `neuronx_distributed.parallel_layers.layers.RowParallelLinear` or a
    `neuronx_distributed.parallel_layers.layers.ColumnParallelLinear` from a regular `torch.nn.Linear`.

    Args:
        linear_layer (`torch.nn.Embedding`):
            The linear layer to parallelize.
        axis (`Union[Literal["row"], Literal["column"]]`):
            Either to create a `RowParallelLinear` or a `ColumnParallelLinear`.
        input_is_parallel (`bool`, defaults to `False`):
            Only relevant when `axis="row"`. It means that resulting `RowParallelLinear` must expect a parallelized
            input.
        gather_output (`bool`, defaults to `True`):
            Only relevant when `axis="column"`. It means that the resulting `ColumnParallelLinear` will gather the
            output after its forward. It allows to get a non-parallelized output from a `ColumnParallelLinear` layer.
        linear_layer_weight_info (`Optional[torch.nn.Linear]`, defaults to `None`):
            Information about which checkpoint file the linear layer weights are stored in.
        linear_layer_bias_weight_info (`Optional[WeightInformation]`, defaults to `None`):
            Information about which checkpoint file the linear layer bias is stored in.
        embedding_weight_to_tie (`Optional[torch.nn.Parameter]`, defaults to `None`):
            If specified, will tie the linear layer weights to it.
        orig_to_parallel (`Optional[Dict[int, torch.nn.Parameter]]`, defaults to `None`):
            A dictionary to fill. It maps a former parameter id to its parallel version.
            It might be deprecated soon.
        device (`Optional[torch.device]`, defaults to `None`):
            The device where the new parallel layer should be put.

    Returns:
        `Union[RowParallelLinear, ColumnParallelLinear]`: The parallel linear layer.
    """
    if axis not in ["row", "column"]:
        raise ValueError(f'axis must either be "row" or "column", but {axis} was given here.')

    device = device if device is not None else torch.device("cpu")
    for weight_info in [linear_layer_weight_info, linear_layer_bias_weight_info]:
        if weight_info is None:
            continue
        _validate_weight_info_device_matches_specified_device(device, weight_info)

    kwargs = {}
    if axis == "row":
        parallel_linear_class = layers.RowParallelLinear
        kwargs["input_is_parallel"] = input_is_parallel
    else:
        parallel_linear_class = layers.ColumnParallelLinear
        kwargs["gather_output"] = gather_output

    kwargs["dtype"] = linear_layer.weight.dtype
    kwargs["bias"] = linear_layer.bias is not None
    kwargs["device"] = device

    parallel_linear_layer = parallel_linear_class(linear_layer.in_features, linear_layer.out_features, **kwargs)
    tp_rank = get_tensor_model_parallel_rank()
    row_size, col_size = parallel_linear_layer.weight.shape

    with torch.no_grad():
        if axis == "row":
            if embedding_weight_to_tie is not None:
                parallel_linear_layer.weight = embedding_weight_to_tie
            elif linear_layer_weight_info is not None:
                weight_data = load_tensor_for_weight(
                    linear_layer_weight_info,
                    tensor_slices=(
                        None,
                        (tp_rank * col_size, (tp_rank + 1) * col_size),
                    ),
                )
                parallel_linear_layer.weight.data = weight_data
            else:
                parallel_linear_layer.weight.copy_(
                    linear_layer.weight[:, tp_rank * col_size : (tp_rank + 1) * col_size]
                )

            if linear_layer.bias is not None:
                if linear_layer_bias_weight_info is not None:
                    bias_weight_data = load_tensor_for_weight(linear_layer_bias_weight_info)
                    parallel_linear_layer.bias.data = bias_weight_data
                else:
                    parallel_linear_layer.bias.copy_(linear_layer.bias)

                if orig_to_parallel is not None:
                    orig_to_parallel[id(linear_layer.bias)] = parallel_linear_layer.bias
        else:
            if embedding_weight_to_tie is not None:
                parallel_linear_layer.weight = embedding_weight_to_tie
            elif linear_layer_weight_info is not None:
                weight_data = load_tensor_for_weight(
                    linear_layer_weight_info,
                    tensor_slices=(
                        (tp_rank * row_size, (tp_rank + 1) * row_size),
                        None,
                    ),
                )
                parallel_linear_layer.weight.data = weight_data

            else:
                parallel_linear_layer.weight.copy_(
                    linear_layer.weight[tp_rank * row_size : (tp_rank + 1) * row_size, :]
                )

            if linear_layer.bias is not None:
                if linear_layer_bias_weight_info is not None:
                    if gather_output:
                        tensor_slices = (None,)
                    else:
                        tensor_slices = (
                            (
                                tp_rank * row_size,
                                (tp_rank + 1) * row_size,
                            ),
                        )
                    bias_weight_data = load_tensor_for_weight(
                        linear_layer_bias_weight_info,
                        tensor_slices=tensor_slices,
                    )
                    parallel_linear_layer.bias.data = bias_weight_data

                else:
                    if gather_output:
                        parallel_linear_layer.bias.copy_(linear_layer.bias)
                    else:
                        parallel_linear_layer.bias.copy_(
                            linear_layer.bias[tp_rank * row_size : (tp_rank + 1) * row_size]
                        )

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
    # TODO: `ignore_mismatched_sizes` is not used in the function, figure out if it leads to a bug.
    ignore_mismatched_sizes: bool = False,
    force_download: bool = False,
    local_files_only: bool = False,
    token: Optional[Union[str, bool]] = None,
    revision: str = "main",
    use_safetensors: bool = False,
    **kwargs,
):
    """
    Custom `from_pretrained()` method for tensor parallelism.
    It will download the weights and create the model but not load the weights in the model. Instead it will create a
    weight map, which maps each parameter name to the `safetensors` checkpoint file storing its weights and attach
    this map to the model instance under the `_weight_map` attribute.
    """
    kwargs.pop("state_dict", None)
    kwargs.pop("from_tf", False)
    kwargs.pop("from_flax", False)
    resume_download = kwargs.pop("resume_download", False)
    proxies = kwargs.pop("proxies", None)
    kwargs.pop("output_loading_info", False)
    kwargs.pop("use_auth_token", None)
    kwargs.pop("trust_remote_code", None)
    kwargs.pop("mirror", None)
    from_pipeline = kwargs.pop("_from_pipeline", None)
    from_auto_class = kwargs.pop("_from_auto", False)
    kwargs.pop("_fast_init", True)
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

    xm.rendezvous("waiting after download and conversion")

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
        weight_map = sharded_metadata["weight_map"]
    else:
        filename = Path(filenames)
        # TODO: manage the safetensor check dependency.
        from safetensors import safe_open

        with safe_open(filename, framework="pt", device="cpu") as fp:
            weight_map = {weight_name: filename for weight_name in fp.keys()}

    model._weight_map = weight_map

    return model


@contextlib.contextmanager
def lazy_load_for_parallelism(tensor_parallel_size: int = 1):
    """
    Context manager that makes the loading of a model lazy for model parallelism:

        - Every `torch.nn.Linear` is put on the `torch.device("meta")` device, meaning that it takes no memory to
        instantiate.
        - Every `torch.nn.Embedding` is also put on the `torch.device("meta")` device.
        - No state dict is actually loaded, instead a weight map is created and attached to the model. For more
        information, read the [`optimum.neuron.distributed.utils.from_pretrained_for_tp`] docstring.

    Args:
        tensor_parallel_size (`int`, defaults to 1):
            The parallel size considered for tensor parallel size. If set to 1, no lazy loading is performed.
    """

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
    if tensor_parallel_size > 1:
        patcher = Patcher(patching_specs=patching_specs)
    else:
        patcher = contextlib.nullcontext()
    with patcher:
        try:
            yield
        finally:
            pass


def make_optimizer_constructor_lazy(optimizer_cls: Type["torch.optim.Optimizer"]):
    """
    Transforms an optimizer constructor (optimizer class) to make it lazy by not initializing the parameters.
    This makes the optimizer lightweight and usable to create a "real" optimizer once the model has been
    parallelized.
    """

    def optimizer_constructor(*args, **kwargs):
        optimizer_with_no_parameters = optimizer_cls([torch.nn.Parameter(torch.empty(1))], *args[1:], **kwargs)
        optimizer_with_no_parameters._args_to_recreate = (args, kwargs)
        return optimizer_with_no_parameters

    return optimizer_constructor


@requires_torch_xla
class ZeroRedundancyOptimizerCompatibleWithTensorParallelism(ZeroRedundancyOptimizer):
    def __init__(
        self,
        params: Iterator[torch.Tensor],
        optimizer_class: Type[torch.optim.Optimizer],
        optimizer_dtype: Optional[Any] = None,
        grad_clipping: bool = True,
        max_norm: Optional[float] = None,
        pin_layout: bool = True,
        **defaults: Any,
    ):
        if not is_neuronx_distributed_available() or not model_parallel_is_initialized():
            return super().__init__(
                params,
                optimizer_class,
                optimizer_dtype=optimizer_dtype,
                grad_clipping=grad_clipping,
                max_norm=max_norm,
                pin_layout=pin_layout,
                **defaults,
            )

        self.params = list(params)
        super(ZeroRedundancyOptimizer, self).__init__(self.params, defaults)

        if isinstance(self.params[0], dict):
            self.params = [p for pg in self.params for p in pg["params"]]

        self.device = self.params[0].device

        self.rank = get_data_parallel_rank()
        self.world_size = get_data_parallel_size()
        self.cc_op_groups = get_data_parallel_group(as_list=True)

        self.optimizer_dtype = optimizer_dtype if optimizer_dtype is not None else torch.float32
        self.grad_clipping = grad_clipping
        self.max_norm = max_norm if max_norm is not None else 1.0
        self.pin_layout = pin_layout

        # Shard parameters for use in optimizer
        self.sharded_params = []
        self._shard_parameters()
        # Optimizer initialization
        self.base_optimizer = optimizer_class(iter(self.sharded_params), **defaults)


@dataclass
class ParameterMetadata:
    kind: Literal["tied", "sharded"]
    partition_dim: Optional[int] = None

    def __post_init__(self):
        if self.kind == "sharded":
            if self.partition_dim is None:
                raise ValueError("ParameterMetadata.partion_dim must be specified when the parameter is sharded.")

    @property
    def is_tied(self):
        return self.kind == "tied"

    @property
    def is_sharded(self):
        return self.kind == "sharded"
