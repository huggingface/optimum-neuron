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
"""Base class related to `neuronx_distributed` to perform parallelism."""

import contextlib
import shutil
from abc import ABC, abstractclassmethod
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Mapping, Optional, Tuple, Union

import torch
from transformers.utils import WEIGHTS_NAME

from ...utils import logging
from ..utils import is_neuronx_distributed_available, is_torch_xla_available
from ..utils.deprecate_utils import deprecate
from .parallel_layers import (
    IOSequenceParallelizer,
    LayerNormSequenceParallelizer,
    LayerNormType,
    SequenceCollectiveOpInfo,
)
from .utils import TENSOR_PARALLEL_SHARDS_DIR_NAME, ParameterMetadata, WeightInformation, load_tensor_for_weight


if is_neuronx_distributed_available():
    import neuronx_distributed
    from neuronx_distributed import parallel_layers

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm


if TYPE_CHECKING:
    from transformers import PreTrainedModel


logger = logging.get_logger()


class SavedModelInTemporaryDirectory:
    def __init__(self, model: "PreTrainedModel"):
        self.tmpdir = TemporaryDirectory()
        self.model = model

    def __enter__(self):
        self.model.save_pretrained(self.tmpdir.name)
        return self.tmpdir.name

    def __exit__(self, *exc):
        self.tmpdir.cleanup()


@deprecate(
    "2.0.0",
    package_name="torch",
    reason="torch.nn.Module._named_members takes a `remove_duplicate` parameter starting from 2.0.0",
)
def _named_members(module, get_members_fn, prefix="", recurse=True, remove_duplicate: bool = True):
    r"""Helper method for yielding various names + members of modules."""
    memo = set()
    modules = module.named_modules(prefix=prefix, remove_duplicate=remove_duplicate) if recurse else [(prefix, module)]
    for module_prefix, mod in modules:
        members = get_members_fn(mod)
        for k, v in members:
            if v is None or v in memo:
                continue
            if remove_duplicate:
                memo.add(v)
            name = module_prefix + ("." if module_prefix else "") + k
            yield name, v


def named_parameters(module: "torch.nn.Module", prefix: str = "", recurse: bool = True, remove_duplicate: bool = True):
    gen = _named_members(
        module, lambda mod: mod._parameters.items(), prefix=prefix, recurse=recurse, remove_duplicate=remove_duplicate
    )
    yield from gen


class Parallelizer(ABC):
    """
    Base abstract class that handles model parallelism.
    """

    SEQUENCE_PARALLEL_LAYERNORM_PATTERNS: Optional[List[str]] = None
    LAYERNORM_TYPE: LayerNormType = LayerNormType.REGULAR
    SEQUENCE_COLLECTIVE_OPS_INFOS: Optional[List[SequenceCollectiveOpInfo]] = None

    def __init__(self):
        self._validate_required_libaries_are_available()

    def _validate_required_libaries_are_available(self):
        if not is_neuronx_distributed_available():
            raise RuntimeError(
                "Parallelizer requires the `neuronx_distributed` package. You can install it by running: pip install "
                "neuronx_distributed"
            )
        if not is_torch_xla_available():
            raise RuntimeError(
                "Parallelizer requires the `torch_xla` package. You can install it by running: pip install torch_xla"
            )

    @classmethod
    @contextlib.contextmanager
    def saved_model_in_temporary_directory(cls, model: "PreTrainedModel"):
        tmpdir = TemporaryDirectory()
        path = Path(tmpdir.name) / "pytorch_model.bin"
        torch.save({"model": model.state_dict()}, path.as_posix())
        try:
            yield path
        finally:
            tmpdir.cleanup()

    @abstractclassmethod
    def _parallelize(
        cls,
        model: "PreTrainedModel",
        device: Optional["torch.device"] = None,
        parallelize_embeddings: bool = True,
        sequence_parallel_enabled: bool = False,
    ) -> "PreTrainedModel":
        """
        Parallelizes the model by transforming regular layer into their parallel counterparts.
        Each concrete class must implement it.

        Args:
            model (`PreTrainedModel`):
                The model to parallelize.
            device (`Optional[torch.device]`, defaults to `None`):
                The device where the new parallel layers should be put.
            parallelize_embeddings (`bool`, defaults to `True`):
                Whether or not the embeddings should be parallelized.
                This can be disabled in the case when the TP size does not divide the vocabulary size.
            sequence_parallel_enabled (`bool`, defaults to `False`):
                Whether or not sequence parallelism is enabled.
        Returns:
            `PreTrainedModel`: The parallelized model.
        """

    @classmethod
    def patch_for_sequence_parallelism(cls, model: "PreTrainedModel", sequence_parallel_enabled: bool):
        """
        This method needs to be overriden. It must patch anything model-specfic to make the model compatible with
        sequence parallelism.
        """
        if sequence_parallel_enabled:
            raise NotImplementedError(
                f"No patching for the attention mechanism for sequence parallelism was implemented for {model.__class__}"
            )

    @classmethod
    def parallelize(
        cls,
        model: "PreTrainedModel",
        device: Optional["torch.device"] = None,
        parallelize_embeddings: bool = True,
        sequence_parallel_enabled: bool = False,
    ) -> "PreTrainedModel":
        """
        Parallelizes the model by transforming regular layer into their parallel counterparts using
        `cls._parallelize()`.

        It also makes sure that each parameter has loaded its weights or has been initialized if there is no pre-trained
        weights associated to it.

        Args:
            model (`PreTrainedModel`):
                The model to parallelize.
            device (`Optional[torch.device]`, defaults to `None`):
                The device where the new parallel layers should be put.
            parallelize_embeddings (`bool`, defaults to `True`):
                Whether or not the embeddings should be parallelized.
                This can be disabled in the case when the TP size does not divide the vocabulary size.
            sequence_parallel_enabled (`bool`, defaults to `False`):
                Whether or not sequence parallelism is enabled.

        Returns:
            `PreTrainedModel`: The parallelized model.
        """
        if sequence_parallel_enabled and cls.SEQUENCE_PARALLEL_LAYERNORM_PATTERNS is None:
            raise NotImplementedError(f"Sequence parallelism is not supported for {model.__class__}.")

        # Preparing the model for sequence parallelism:
        # 1. Transforming the LayerNorms.
        layer_norm_qualified_name_patterns = (
            cls.SEQUENCE_PARALLEL_LAYERNORM_PATTERNS if cls.SEQUENCE_PARALLEL_LAYERNORM_PATTERNS is not None else []
        )
        layer_norm_sequence_parallelizer = LayerNormSequenceParallelizer(
            sequence_parallel_enabled, layer_norm_qualified_name_patterns
        )
        layer_norm_sequence_parallelizer.sequence_parallelize(model, cls.LAYERNORM_TYPE)

        # 2. Taking care of scattering / gathering on the sequence axis in the model via the IOSequenceParallelizer.
        io_sequence_parallelizer = IOSequenceParallelizer(
            sequence_parallel_enabled,
            sequence_collective_op_infos=cls.SEQUENCE_COLLECTIVE_OPS_INFOS,
        )
        io_sequence_parallelizer.sequence_parallelize(model)

        # 3. Applying model specific patching for sequence parallelism.
        if sequence_parallel_enabled:
            cls.patch_for_sequence_parallelism(model, sequence_parallel_enabled)

        model = cls._parallelize(
            model,
            device=device,
            parallelize_embeddings=parallelize_embeddings,
            sequence_parallel_enabled=sequence_parallel_enabled,
        )

        weight_map = getattr(model, "_weight_map", None)

        # The model was not loaded lazily, it is already ready.
        if weight_map is None:
            return model

        with torch.no_grad():
            tied_weights = {}
            new_parameters = set()
            modules_to_initialize = []
            for name, parameter in named_parameters(model, remove_duplicate=False):
                split = name.rsplit(".", maxsplit=1)
                module = model.get_submodule(split[0])
                attribute_name = split[1]
                current_weight = getattr(module, attribute_name)

                try:
                    weight_info = WeightInformation(weight_map[name], name, weight_map=weight_map, device=device)
                except KeyError:
                    weight_info = None

                if parameter in new_parameters:
                    # It can be the case if a module is shared in the model.
                    # For example in T5, the embedding layer is shared so after loading the parameter the first time,
                    # it is not needed to do it again, and doing it can cause bugs.
                    continue
                elif parameter in tied_weights:
                    # It can be the case when weights are tied. For example between the embeddings and the LM head.
                    new_parameter = tied_weights[parameter]
                elif weight_info is not None:
                    if getattr(current_weight, "tensor_model_parallel", False):
                        if parameter.device == torch.device("meta"):
                            # This must either be a torch.nn.Embedding or a torch.nn.Linear that was not handled during
                            # parallelization since those are the only classes that we initialize on the `meta` device.
                            num_dims = current_weight.dim()
                            partition_dim = getattr(current_weight, "partition_dim")
                            tp_rank = parallel_layers.parallel_state.get_tensor_model_parallel_rank()
                            size_per_rank = current_weight.size(partition_dim)
                            slices = [
                                None
                                if idx != partition_dim
                                else (size_per_rank * tp_rank, size_per_rank * (tp_rank + 1))
                                for idx in range(num_dims)
                            ]
                        else:
                            # The parameter is not on the `meta` device, it has been loaded from a checkpoint during
                            # parallelization, we can skip.
                            tied_weights[parameter] = parameter
                            new_parameters.add(parameter)
                            continue
                    else:
                        slices = None

                    new_parameter = torch.nn.Parameter(
                        load_tensor_for_weight(weight_info, tensor_slices=slices).to(parameter.dtype)
                    )
                else:
                    # This means that there is no information about where to find the weights for this parameter.
                    device = torch.device("cpu") if device is None else device
                    new_parameter = torch.nn.Parameter(torch.empty_like(current_weight, device=device))
                    modules_to_initialize.append(module)

                setattr(
                    module,
                    attribute_name,
                    new_parameter,
                )
                tied_weights[parameter] = new_parameter
                new_parameters.add(new_parameter)

            for mod in modules_to_initialize:
                # This module has not pre-trained weights, it must be fine-tuned, we initialize it with the
                # `reset_parameters()` method.
                mod.reset_parameters()

        return model

    @classmethod
    def deparallelize(cls, model: "PreTrainedModel") -> "PreTrainedModel":
        raise NotImplementedError

    @classmethod
    def was_parallelized(cls, model: "PreTrainedModel") -> bool:
        parallel_layer_classes = (
            parallel_layers.ParallelEmbedding,
            parallel_layers.ColumnParallelLinear,
            parallel_layers.RowParallelLinear,
        )
        return any(isinstance(mod, parallel_layer_classes) for mod in model.modules())

    @classmethod
    def _check_model_was_parallelized(cls, model: "PreTrainedModel"):
        if not cls.was_parallelized(model):
            raise ValueError("The model needs to be parallelized first.")

    @classmethod
    def optimizer_cpu_params_to_xla_params(
        cls,
        optimizer: "torch.optim.Optimizer",
        orig_param_to_parallel_param_on_xla: Mapping[int, "torch.nn.Parameter"],
    ) -> Tuple[List[Dict[str, Any]], bool]:
        parameters_on_xla = []
        need_to_create_new_optimizer = False
        if hasattr(optimizer, "_args_to_recreate"):
            args, _ = optimizer._args_to_recreate
            parameters = args[0]
            for param in parameters:
                if isinstance(param, dict):
                    new_param = {k: v for k, v in param.items() if k != "params"}
                    params = []
                    for p in param["params"]:
                        params.append(orig_param_to_parallel_param_on_xla[id(p)])
                    new_param["params"] = params
                else:
                    new_param = []
                    for p in param:
                        new_param.append(orig_param_to_parallel_param_on_xla[id(p)])
                parameters_on_xla.append(new_param)
        else:
            for param_group in optimizer.param_groups:
                new_params = []
                params = param_group["params"]
                for idx in range(len(params)):
                    param_on_xla = orig_param_to_parallel_param_on_xla[id(params[idx])]
                    if params[idx] != param_on_xla:
                        need_to_create_new_optimizer = True
                    new_params.append(param_on_xla)
                new_group = {k: v for k, v in param_group.items() if k != "params"}
                new_group["params"] = new_params
                parameters_on_xla.append(new_group)
        return parameters_on_xla, need_to_create_new_optimizer

    @classmethod
    def optimizer_for_tp(
        cls,
        optimizer: "torch.optim.Optimizer",
        orig_param_to_parallel_param_on_xla: Mapping[int, "torch.nn.Parameter"],
    ) -> "torch.optim.Optimizer":
        """
        Creates an optimizer ready for a parallelized model from an existing optimizer.

        There are two cases:
            1. The optimizer has been created via a lazy constructor from
            [`optimum.neuron.distributed.utils.make_optimizer_constructor_lazy`], it which case the exactly intended optimizer is
            created for tensor parallelism.
            2. The optimizer was created with a regular constructor. In this case the optimizer for tensor parallelism
            is created as close as possible to what was intended but that is not guaranteed.

        Args:
            optimizer (`torch.optim.Optimizer`):
                The original optimizer.
            orig_param_to_parallel_param_on_xla (`Mapping[int, torch.nn.Parameter]`):
                A mapping (e.g. dict-like) that maps the id of a parameter in `optimizer` to the id of its
                parallelized counterpart on an XLA device.

        Returns:
            `torch.optim.Optimizer`: The tensor parallelism ready optimizer.
        """
        parallel_parameters, need_to_create_new_optimizer = cls.optimizer_cpu_params_to_xla_params(
            optimizer, orig_param_to_parallel_param_on_xla
        )
        if hasattr(optimizer, "_args_to_recreate"):
            args, kwargs = optimizer._args_to_recreate
            optimizer_for_tp = optimizer.__class__(parallel_parameters, *args[1:], **kwargs)
            del optimizer
        elif need_to_create_new_optimizer:
            optimizer_for_tp = optimizer.__class__(parallel_parameters)
            del optimizer
        else:
            optimizer_for_tp = optimizer
        return optimizer_for_tp

    @classmethod
    def _get_parameters_tp_metadata(cls, named_parameters: Dict[str, "torch.nn.Parameter"]):
        tp_metadata = {}
        for name, param in named_parameters.items():
            if getattr(param, "tensor_model_parallel", False):
                param_metadata = ParameterMetadata(
                    "sharded",
                    partition_dim=param.partition_dim,
                )
            else:
                param_metadata = ParameterMetadata("tied")
            tp_metadata[name] = param_metadata
        return tp_metadata

    @classmethod
    def save_model_checkpoint_as_regular(
        cls,
        model: "PreTrainedModel",
        output_dir: Union[str, Path],
        optimizer: Optional["torch.optim.Optimizer"] = None,
    ):
        cls._check_model_was_parallelized(model)
        data_parallel_rank = parallel_layers.parallel_state.get_data_parallel_rank()
        tensor_parallel_rank = parallel_layers.parallel_state.get_tensor_model_parallel_rank()

        if data_parallel_rank != 0:
            return

        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)

        if optimizer is not None:
            logger.warning(
                "Saving the optimizer state as a regular file under the tensor parallel setting is not supported yet."
            )

        state_dict = {}
        for name, param in model.named_parameters():
            if getattr(param, "tensor_model_parallel", False):
                if param.partition_dim == 1:
                    tensor = neuronx_distributed.utils.gather_from_tensor_model_parallel_region(param)
                else:
                    # Because the gather works only on last dim. Need to make it work for all dims.
                    tensor = neuronx_distributed.utils.gather_from_tensor_model_parallel_region(
                        param.transpose()
                    ).transpose()
            else:
                tensor = param
            state_dict[name] = tensor

        model_state_dict = {"model": state_dict}
        should_save = tensor_parallel_rank == 0
        xm._maybe_convert_to_cpu(model_state_dict, convert=should_save)
        if should_save:
            output_path = output_dir / WEIGHTS_NAME
            torch.save(model_state_dict["model"], output_path.as_posix())
        xm.rendezvous("saving regular checkpoint")

    @classmethod
    def save_model_checkpoint_as_sharded(
        cls,
        model: "PreTrainedModel",
        output_dir: Union[str, Path],
        optimizer: Optional["torch.optim.Optimizer"] = None,
    ):
        cls._check_model_was_parallelized(model)
        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)

        state_dict = {"model": model.state_dict()}
        state_dict["sharded_metadata"] = {
            k: asdict(v) for k, v in cls._get_parameters_tp_metadata(dict(model.named_parameters())).items()
        }

        if optimizer is not None:
            # TODO: have metadata working for the optimizer.
            state_dict["optimizer_state_dict"] = optimizer.state_dict()

        output_path = output_dir / TENSOR_PARALLEL_SHARDS_DIR_NAME
        from neuronx_distributed.parallel_layers.parallel_state import (
            get_data_parallel_rank,
            get_tensor_model_parallel_rank,
        )

        if get_data_parallel_rank() == 0 and get_tensor_model_parallel_rank() == 0:
            if output_path.is_dir():
                shutil.rmtree(output_path, ignore_errors=True)
            output_path.mkdir()
        xm.rendezvous("waiting before saving")
        parallel_layers.save(state_dict, output_path.as_posix())

    @classmethod
    def save_model_checkpoint(
        cls,
        model: "PreTrainedModel",
        output_dir: Union[str, Path],
        as_regular: bool = False,
        as_sharded: bool = True,
        optimizer: Optional["torch.optim.Optimizer"] = None,
    ):
        if not as_regular and not as_sharded:
            raise ValueError("At least as_regular or as_sharded must be True.")
        if as_regular:
            cls.save_model_checkpoint_as_regular(model, output_dir, optimizer=optimizer)
        if as_sharded:
            cls.save_model_checkpoint_as_sharded(model, output_dir, optimizer=optimizer)

    @classmethod
    def load_model_regular_checkpoint(cls, model: "PreTrainedModel", load_dir: Union[str, Path]):
        raise NotImplementedError("This requires being able to deparallelize the model.")

    @classmethod
    def load_model_sharded_checkpoint(cls, model: "PreTrainedModel", load_dir: Union[str, Path]):
        cls._check_model_was_parallelized(model)
        if not isinstance(load_dir, Path):
            load_dir = Path(load_dir)
        parallel_layers.load(load_dir / TENSOR_PARALLEL_SHARDS_DIR_NAME, model=model, sharded=True)

    @classmethod
    def load_model_checkpoint(cls, model: "PreTrainedModel", load_dir: Union[str, Path]):
        if not isinstance(load_dir, Path):
            load_dir = Path(load_dir)

        if (load_dir / TENSOR_PARALLEL_SHARDS_DIR_NAME).is_dir():
            cls.load_model_sharded_checkpoint(model, load_dir)
        elif (load_dir / WEIGHTS_NAME).is_file():
            cls.load_model_regular_checkpoint(model, load_dir)
        else:
            raise FileNotFoundError(f"Could not find a checkpoint file under {load_dir.as_posix()}.")
