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


class Parallelizer(ABC):
    """
    Base abstract class that handles model parallelism.
    """

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
        orig_to_parallel: Optional[Dict[int, "torch.nn.Parameter"]] = None,
        device: Optional["torch.device"] = None,
    ) -> "PreTrainedModel":
        """
        Parallelizes the model by transforming regular layer into their parallel counterparts.
        Each concrete class must implement it.

        Args:
            model (`PreTrainedModel`):
                The model to parallelize.
            orig_to_parallel (`Optional[Dict[int, torch.nn.Parameter]]`, defaults to `None`):
                A dictionary to fill. It maps a former parameter id to its parallel version.
                It might be deprecated soon.
            device (`Optional[torch.device]`, defaults to `None`):
                The device where the new parallel layers should be put.

        Returns:
            `PreTrainedModel`: The parallelized model.
        """

    @classmethod
    def parallelize(
        cls,
        model: "PreTrainedModel",
        orig_to_parallel: Optional[Dict[int, "torch.nn.Parameter"]] = None,
        device: Optional["torch.device"] = None,
    ) -> "PreTrainedModel":
        """
        Parallelizes the model by transforming regular layer into their parallel counterparts using
        `cls._parallelize()`.

        It also makes sure that each parameter has loaded its weights or has been initialized if there is no pre-trained
        weights associated to it.

        Args:
            model (`PreTrainedModel`):
                The model to parallelize.
            orig_to_parallel (`Optional[Dict[int, torch.nn.Parameter]]`, defaults to `None`):
                A dictionary to fill. It maps a former parameter id to its parallel version.
                It might be deprecated soon.
            device (`Optional[torch.device]`, defaults to `None`):
                The device where the new parallel layers should be put.

        Returns:
            `PreTrainedModel`: The parallelized model.
        """
        model = cls._parallelize(model, orig_to_parallel=orig_to_parallel, device=device)
        weight_map = getattr(model, "_weight_map", {})
        with torch.no_grad():
            modules_to_initialize = []
            for name, parameter in model.named_parameters():
                # This must be either a torch.nn.Embedding or a torch.nn.Linear since those are the only
                # classes that we initialize on the `meta` device.
                if parameter.device == torch.device("meta"):
                    if weight_map is None:
                        raise ValueError(
                            f"The parameter called {name} of the model is on the `meta` device and no weight map is "
                            "attached to the model to load the proper weights from file."
                        )
                    split = name.rsplit(".", maxsplit=1)
                    if len(split) == 1:
                        module = model
                        attribute_name = split[0]
                    else:
                        module = model.get_submodule(split[0])
                        attribute_name = split[1]
                    try:
                        weight_info = WeightInformation(weight_map[name], name, device=device)
                        setattr(module, attribute_name, torch.nn.Parameter(load_tensor_for_weight(weight_info)))
                    except KeyError:
                        # This means that there is no information about where to find the weights for this parameter.
                        device = torch.device("cpu") if device is None else device
                        setattr(
                            module,
                            attribute_name,
                            torch.nn.Parameter(torch.empty_like(getattr(module, attribute_name), device=device)),
                        )
                        modules_to_initialize.append(module)
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
        tensor_parallel_rank = parallel_layers.parallel_state.get_tensor_parallel_rank()

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
