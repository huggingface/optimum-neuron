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
from abc import ABC, abstractclassmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Union

import torch
from transformers.utils import WEIGHTS_NAME

from ..utils import is_neuronx_distributed_available, is_torch_xla_available
from .utils import TENSOR_PARALLEL_SHARDS_DIR_NAME


if is_neuronx_distributed_available():
    import neuronx_distributed

if is_torch_xla_available():
    import torch_xla.core.xla_model as xm


if TYPE_CHECKING:
    from transformers import PreTrainedModel


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
    def parallelize(cls, model: "PreTrainedModel") -> "PreTrainedModel":
        pass

    @classmethod
    def deparallelize(cls, model: "PreTrainedModel") -> "PreTrainedModel":
        raise NotImplementedError

    @classmethod
    def was_parallelized(cls, model: "PreTrainedModel") -> bool:
        parallel_layers = (
            neuronx_distributed.parallel_layers.ParallelEmbedding,
            neuronx_distributed.parallel_layers.ColumnParallelLinear,
            neuronx_distributed.parallel_layers.RowParallelLinear,
        )
        return any(isinstance(mod, parallel_layers) for mod in model.modules())

    @classmethod
    def _check_model_was_parallelized(cls, model: "PreTrainedModel"):
        if not cls.was_parallelized(model):
            raise ValueError("The model needs to be parallelized first.")

    @classmethod
    def save_model_checkpoint_as_regular(cls, model: "PreTrainedModel", output_dir: Union[str, Path]):
        cls._check_model_was_parallelized(model)
        data_parallel_rank = neuronx_distributed.parallel_state.get_data_parallel_rank()
        tensor_parallel_rank = neuronx_distributed.parallel_state.get_tensor_parallel_rank()

        if data_parallel_rank != 0:
            return

        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)

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
        xm.rendevous("saving regular checkpoint")

    @classmethod
    def save_model_checkpoint_as_sharded(cls, model: "PreTrainedModel", output_dir: Union[str, Path]):
        cls._check_model_was_parallelized(model)
        data_parallel_rank = neuronx_distributed.parallel_state.get_data_parallel_rank()
        if data_parallel_rank != 0:
            return

        if not isinstance(output_dir, Path):
            output_dir = Path(output_dir)

        output_path = output_dir / TENSOR_PARALLEL_SHARDS_DIR_NAME
        neuronx_distributed.parallel_layers.save({"model": model.state_dict()}, output_path.as_posix())

    @classmethod
    def save_model_checkpoint(
        cls, model: "PreTrainedModel", output_dir: Union[str, Path], as_regular: bool = True, as_sharded: bool = True
    ):
        if not as_regular and not as_sharded:
            raise ValueError("At least as_regular or as_sharded must be True.")
        if as_regular:
            cls.save_model_checkpoint(model, output_dir)
        if as_sharded:
            cls.save_model_checkpoint_as_sharded(model, output_dir)

    @classmethod
    def load_model_regular_checkpoint(cls, model: "PreTrainedModel", load_dir: Union[str, Path]):
        raise NotImplementedError("This requires being able to deparallelize the model.")

    @classmethod
    def load_model_sharded_checkpoint(cls, model: "PreTrainedModel", load_dir: Union[str, Path]):
        cls._check_model_was_parallelized(model)
        if not isinstance(load_dir, Path):
            load_dir = Path(load_dir)
        neuronx_distributed.parallel_layers.load(load_dir / TENSOR_PARALLEL_SHARDS_DIR_NAME, model=model, sharded=True)

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
