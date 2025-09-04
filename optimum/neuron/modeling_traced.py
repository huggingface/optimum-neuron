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
"""NeuronTracedModel base classe for inference on neuron devices using the same API as Transformers."""

import os
import shutil
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Literal

import torch
from huggingface_hub import HfApi, HfFolder, hf_hub_download
from transformers import AutoConfig, AutoModel, GenerationMixin

from optimum.exporters.neuron import main_export
from optimum.exporters.neuron.model_configs import *  # noqa: F403
from optimum.exporters.tasks import TasksManager
from optimum.modeling_base import OptimizedModel
from optimum.utils import logging
from optimum.utils.save_utils import maybe_load_preprocessors

from .cache.entries.single_model import SingleModelCacheEntry
from .cache.hub_cache import create_hub_compile_cache_proxy
from .modeling_base import NeuronModel
from .utils import (
    DTYPE_MAPPER,
    NEURON_FILE_NAME,
    InputShapesArguments,
    check_if_weights_replacable,
    is_neuron_available,
    replace_weights,
    store_compilation_config,
)
from .utils.import_utils import is_neuronx_available
from .utils.version_utils import check_compiler_compatibility, get_neuroncc_version, get_neuronxcc_version


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from optimum.exporters.neuron import NeuronDefaultConfig

if is_neuron_available():
    NEURON_COMPILER_TYPE = "neuron-cc"
    NEURON_COMPILER_VERSION = get_neuroncc_version()

if is_neuronx_available():
    import torch_neuronx
    from torch_neuronx import move_trace_to_device

    NEURON_COMPILER_TYPE = "neuronx-cc"
    NEURON_COMPILER_VERSION = get_neuronxcc_version()

logger = logging.get_logger(__name__)


class NeuronTracedModel(OptimizedModel, NeuronModel):
    """
    Base class running compiled and optimized models on Neuron devices.

    It implements generic methods for interacting with the Hugging Face Hub as well as compiling vanilla
    transformers models to neuron-optimized TorchScript module and export it using `optimum.exporters.neuron` toolchain.

    Class attributes:
        - model_type (`str`, *optional*, defaults to `"neuron_model"`) -- The name of the model type to use when
        registering the NeuronTracedModel classes.
        - auto_model_class (`Type`, *optional*, defaults to `AutoModel`) -- The `AutoModel` class to be represented by the
        current NeuronTracedModel class.

    Common attributes:
        - model (`torch.jit._script.ScriptModule`) -- The loaded `ScriptModule` compiled for neuron devices.
        - config ([`~transformers.PretrainedConfig`]) -- The configuration of the model.
        - model_save_dir (`Path`) -- The directory where a neuron compiled model is saved.
        By default, if the loaded model is local, the directory where the original model will be used. Otherwise, the
        cache directory will be used.
    """

    model_type = "neuron_model"
    auto_model_class = AutoModel
    library_name = "transformers"

    def __init__(
        self,
        model: torch.jit._script.ScriptModule,
        config: "PretrainedConfig",
        model_save_dir: str | Path | TemporaryDirectory | None = None,
        model_file_name: str | None = None,
        preprocessors: list | None = None,
        neuron_config: "NeuronDefaultConfig | None" = None,
        **kwargs,
    ):
        super().__init__(model, config)
        if hasattr(model, "device"):
            self.device = model.device
        else:
            self.device = torch.device("cpu")

        self.model = model
        self.model_file_name = model_file_name or NEURON_FILE_NAME
        self.config = config
        self.neuron_config = neuron_config
        self.input_static_shapes = NeuronTracedModel.get_input_static_shapes(self.neuron_config)
        self._attributes_init(model_save_dir, preprocessors, **kwargs)

    @staticmethod
    def load_model(path: str | Path, to_neuron: bool = False, device_id: int = 0) -> torch.jit._script.ScriptModule:
        """
        Loads a TorchScript module compiled by neuron(x)-cc compiler. It will be first loaded onto CPU and then moved to
        one or multiple [NeuronCore](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/neuroncores-arch.html).

        Args:
            path (`str | Path`):
                Path of the compiled model.
            to_neuron (`bool`, defaults to `False`):
                Whether to move manually the traced model to NeuronCore. It's only needed when `inline_weights_to_neff=False`, otherwise it is loaded automatically to a Neuron device.
            device_id (`int`, defaults to 0):
                Index of NeuronCore to load the traced model to.
        """
        if not isinstance(path, Path):
            path = Path(path)

        if path.is_file():
            model = torch.jit.load(path)
            # For non-inlined models, send the module manually to device. This is important for weights/neff non-inlined module since when loading the module, the neff is automatically moved to Neuron but not the weights. We need to move the weights to Neuron as well manually to avoid great host to device IO penalty.
            if is_neuronx_available():
                torch_neuronx.experimental.set_neuron_cores(
                    model, start_nc=0, nc_count=1
                )  # The inputs are allocated to nc:0 by default, this line ensures both input tensors and the model are on the same core.
                if to_neuron:
                    move_trace_to_device(model, device_id)
            return model

    def replace_weights(self, weights: dict[str, torch.Tensor] | torch.nn.Module | None = None):
        check_if_weights_replacable(self.config, weights)
        if weights is not None:
            replace_weights(self.model, weights)

    def _save_pretrained(self, save_directory: str | Path):
        """
        Saves a model and its configuration file to a directory, so that it can be re-loaded using the
        [`~optimum.neuron.modeling_traced.NeuronTracedModel.from_pretrained`] class method.

        Args:
            save_directory (`str | Path`):
                Directory where to save the model file.
        """
        src_path = self.model_save_dir / self.model_file_name
        dst_path = Path(save_directory) / self.model_file_name

        shutil.copyfile(src_path, dst_path)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: str | Path,
        config: "PretrainedConfig",
        token: bool | str | None = None,
        revision: str | None = None,
        force_download: bool = False,
        cache_dir: str | None = None,
        file_name: str | None = None,
        subfolder: str = "",
        local_files_only: bool = False,
        model_save_dir: str | Path | TemporaryDirectory | None = None,
        neuron_config: "NeuronDefaultConfig | None" = None,
        **kwargs,
    ) -> "NeuronTracedModel":
        model_path = Path(model_id)

        if file_name is None:
            if model_path.is_dir():
                neuron_files = list(model_path.glob("*.neuron"))
            else:
                if isinstance(token, bool):
                    token = HfFolder().get_token()
                else:
                    token = token
                repo_files = map(Path, HfApi().list_repo_files(model_id, revision=revision, token=token))
                pattern = "*.neuron" if subfolder == "" else f"{subfolder}/*.neuron"
                neuron_files = [p for p in repo_files if p.match(pattern)]

            if len(neuron_files) == 0:
                raise FileNotFoundError(f"Could not find any neuron model file in {model_path}")
            elif len(neuron_files) > 1:
                raise RuntimeError(
                    f"Too many neuron model files were found in {model_path}, specify which one to load by using the "
                    "file_name argument."
                )
            else:
                file_name = neuron_files[0].name

        # Check compiler compatibility(compiler type and version) of the saved model vs. system.
        if hasattr(config, "neuron") and "compiler_type" in config.neuron:
            model_compiler_type = config.neuron.get("compiler_type")
            model_compiler_version = config.neuron.get("compiler_version")
            check_compiler_compatibility(model_compiler_type, model_compiler_version)

        # reconstruct neuron config
        neuron_config = cls._neuron_config_init(config) if neuron_config is None else neuron_config
        inline_weights_to_neff = config.neuron.get("inline_weights_to_neff", False)

        preprocessors = None
        if model_path.is_dir():
            model = NeuronTracedModel.load_model(model_path / file_name, to_neuron=not inline_weights_to_neff)
            new_model_save_dir = model_path
        else:
            model_cache_path = hf_hub_download(
                repo_id=model_id,
                filename=file_name,
                subfolder=subfolder,
                token=token,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
            )

            model = NeuronTracedModel.load_model(model_cache_path, to_neuron=not inline_weights_to_neff)
            new_model_save_dir = Path(model_cache_path).parent

        preprocessors = maybe_load_preprocessors(model_id, subfolder=subfolder)

        # model_save_dir can be provided in kwargs as a TemporaryDirectory instance, in which case we want to keep it
        # instead of the path only.
        if model_save_dir is None:
            model_save_dir = new_model_save_dir

        return cls(
            model=model,
            config=config,
            model_save_dir=model_save_dir,
            model_file_name=file_name,
            preprocessors=preprocessors,
            neuron_config=neuron_config,
        )

    @classmethod
    def _from_transformers(cls, *args, **kwargs):
        # Deprecate it when optimum uses `_export` as from_pretrained_method in a stable release.
        return cls._export(*args, **kwargs)

    @classmethod
    def _export(
        cls,
        model_id: str,
        config: "PretrainedConfig",
        token: bool | str | None = None,
        revision: str | None = None,
        force_download: bool = False,
        cache_dir: str | None = None,
        compiler_workdir: str | Path | None = None,
        tensor_parallel_size: int | None = 1,
        disable_neuron_cache: bool = False,
        inline_weights_to_neff: bool = True,
        optlevel: str = "2",
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        task: str | None = None,
        torch_dtype: str | torch.dtype = torch.float32,
        auto_cast: str | None = None,
        auto_cast_type: str | None = None,
        disable_fast_relayout: bool | None = False,
        disable_fallback: bool = False,
        dynamic_batch_size: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        **kwargs_shapes,
    ) -> "NeuronTracedModel":
        """
        Exports a vanilla Transformers model into a neuron-compiled TorchScript Module using `optimum.exporters.neuron.export`.

        Args:
            kwargs_shapes (`dict[str, int]`):
                Shapes to use during inference. This argument allows to override the default shapes used during the export.
        """
        if task is None:
            task = TasksManager.infer_task_from_model(cls.auto_model_class)
        task = TasksManager.map_from_synonym(task)

        # Get compilation arguments
        if is_neuron_available() and dynamic_batch_size is True and "batch_size" in kwargs_shapes:
            kwargs_shapes["batch_size"] = 1
            disable_fallback = True  # Turn off the fallback for neuron, otherwise dynamic batching will still fail
        auto_cast_type = None if auto_cast is None else auto_cast_type
        compiler_kwargs = {
            "auto_cast": auto_cast,
            "auto_cast_type": auto_cast_type,
            "disable_fast_relayout": disable_fast_relayout,
            "disable_fallback": disable_fallback,
        }
        # clean shapes
        commit_hash = kwargs_shapes.pop("_commit_hash", None)

        if not disable_neuron_cache and is_neuronx_available():
            # Check if the cache exists
            compilation_config = store_compilation_config(
                config=config,
                input_shapes=kwargs_shapes,
                compiler_kwargs=compiler_kwargs,
                float_dtype=torch_dtype,
                dynamic_batch_size=dynamic_batch_size,
                tensor_parallel_size=tensor_parallel_size,
                compiler_type=NEURON_COMPILER_TYPE,
                compiler_version=NEURON_COMPILER_VERSION,
                inline_weights_to_neff=inline_weights_to_neff,
                optlevel=optlevel,
                model_type=getattr(config, "model_type", None),
                task=task,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )
            cache_entry = SingleModelCacheEntry(model_id=model_id, task=task, config=compilation_config)
            compile_cache = create_hub_compile_cache_proxy()
            model_cache_dir = compile_cache.default_cache.get_cache_dir_with_cache_key(f"MODULE_{cache_entry.hash}")
            cache_available = compile_cache.download_folder(model_cache_dir, model_cache_dir)
        else:
            cache_available = False

        # load cache
        if cache_available:
            try:
                neuron_model = cls.from_pretrained(model_cache_dir)
                model = TasksManager.get_model_from_task(
                    task=task,
                    model_name_or_path=model_id,
                    subfolder=subfolder,
                    revision=revision,
                    cache_dir=cache_dir,
                    token=token,
                    framework="pt",
                    torch_dtype=torch_dtype,
                    local_files_only=local_files_only,
                    force_download=force_download,
                    trust_remote_code=trust_remote_code,
                    _commit_hash=commit_hash,
                )
                if not inline_weights_to_neff:
                    # replace weights
                    neuron_model.replace_weights(weights=model)
                return neuron_model
            except Exception as e:
                logger.warning(
                    f"Found the cached artifacts but failed to re-load them with error: {e}. \n Falling back to recompilation."
                )
                cache_available = False

        # compile
        if not cache_available:
            # compile
            save_dir = TemporaryDirectory()
            save_dir_path = Path(save_dir.name)
            main_export(
                model_name_or_path=model_id,
                output=save_dir_path,
                compiler_kwargs=compiler_kwargs,
                torch_dtype=torch_dtype,
                task=task,
                dynamic_batch_size=dynamic_batch_size,
                cache_dir=cache_dir,
                disable_neuron_cache=disable_neuron_cache,
                compiler_workdir=compiler_workdir,
                inline_weights_to_neff=inline_weights_to_neff,
                optlevel=optlevel,
                trust_remote_code=trust_remote_code,
                subfolder=subfolder,
                revision=revision,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                do_validation=False,
                library_name=cls.library_name,
                **kwargs_shapes,
            )
            config = AutoConfig.from_pretrained(save_dir_path)

        return cls._from_pretrained(save_dir_path, config, model_save_dir=save_dir)

    def push_to_hub(
        self,
        save_directory: str,
        repository_id: str,
        private: bool | None = None,
        revision: str | None = None,
        token: bool | str | None = None,
        endpoint: str | None = None,
    ) -> str:
        api = HfApi(endpoint=endpoint)

        api.create_repo(
            token=token,
            repo_id=repository_id,
            exist_ok=True,
            private=private,
        )
        for path, subdirs, files in os.walk(save_directory):
            for name in files:
                local_file_path = os.path.join(path, name)
                hub_file_path = os.path.relpath(local_file_path, save_directory)
                api.upload_file(
                    token=token,
                    repo_id=repository_id,
                    path_or_fileobj=os.path.join(os.getcwd(), local_file_path),
                    path_in_repo=hub_file_path,
                    revision=revision,
                )

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def _attributes_init(
        self,
        model_save_dir: str | Path | TemporaryDirectory | None = None,
        preprocessors: list | None = None,
        **kwargs,
    ):
        """
        Initializes attributes.
        """
        self._path_tempdirectory_instance = None
        if isinstance(model_save_dir, TemporaryDirectory):
            self._path_tempdirectory_instance = model_save_dir
            self.model_save_dir = Path(model_save_dir.name)
        elif isinstance(model_save_dir, str):
            self.model_save_dir = Path(model_save_dir)
        else:
            self.model_save_dir = model_save_dir

        self.preprocessors = preprocessors if preprocessors is not None else []

        # Registers the NeuronModelForXXX classes into the transformers AutoModel classes to avoid warnings when creating
        # a pipeline https://github.com/huggingface/transformers/blob/3d3204c025b6b5de013e07dd364208e28b4d9589/src/transformers/pipelines/base.py#L940
        AutoConfig.register(self.model_type, AutoConfig)
        if hasattr(self.auto_model_class, "register"):
            self.auto_model_class.register(AutoConfig, self.__class__)

    @classmethod
    def _neuron_config_init(cls, config: "PretrainedConfig") -> "NeuronDefaultConfig":
        """
        Builds a `NeuronDefaultConfig` with an instance of the `PretrainedConfig` and the task.
        """
        if not hasattr(config, "neuron"):
            raise ValueError(
                "Unable to identify neuron configuration with the keyword `neuron`, make sure that your config file contains necessary information"
            )

        neuron_config = config.neuron
        # Fetch compiler information
        compiler_type = neuron_config.get("compiler_type")
        compiler_version = neuron_config.get("compiler_version")
        tensor_parallel_size = neuron_config.get("tensor_parallel_size", 1)

        # Fetch mandatory shapes from config
        compile_shapes = {
            key.replace("static_", ""): value
            for (key, value) in config.to_diff_dict().get("neuron").items()
            if key.startswith("static_")
        }

        # Neuron config constructor
        task = neuron_config.get("task", None) or TasksManager.infer_task_from_model(cls.auto_model_class)
        task = TasksManager.map_from_synonym(task)
        model_type = neuron_config.get("model_type", None) or config.model_type
        model_type = model_type.replace("_", "-")
        neuron_config_constructor = TasksManager.get_exporter_config_constructor(
            model_type=model_type,
            exporter="neuron",
            task=task,
            library_name=cls.library_name,
        )

        compile_shapes = InputShapesArguments(**compile_shapes)
        return neuron_config_constructor(
            config,
            dynamic_batch_size=neuron_config.get("dynamic_batch_size", False),
            compiler_type=compiler_type,
            compiler_version=compiler_version,
            tensor_parallel_size=tensor_parallel_size,
            input_shapes=compile_shapes,
            output_hidden_states=neuron_config.get("output_hidden_states", False),
            int_dtype=DTYPE_MAPPER.pt(neuron_config.get("int_dtype", "int64")),
            float_dtype=DTYPE_MAPPER.pt(neuron_config.get("float_dtype", "fp32")),
        )

    @classmethod
    def get_input_static_shapes(cls, neuron_config: "NeuronDefaultConfig") -> dict[str, int]:
        """
        Gets a dictionary of inputs with their valid static shapes.
        """
        axes = neuron_config._axes
        input_static_shapes = {
            name: value.shape
            for name, value in neuron_config.generate_dummy_inputs(return_tuple=False, **axes).items()
        }
        return input_static_shapes

    def _validate_static_shape(self, input_shapes: list[int], target_shapes: list[int]) -> bool:
        """
        Checks if a input needs to be padded.
        """
        if self.neuron_config.dynamic_batch_size is True:
            batch_size_check = input_shapes[0] % target_shapes[0] == 0
            other_check = input_shapes[1:] == target_shapes[1:] if len(input_shapes) > 1 else True
            return batch_size_check and other_check
        else:
            return input_shapes == target_shapes

    def _raise_if_invalid_padding(self, input_name, input_tensor, target_shapes, to_pad, dim):
        if to_pad < 0:
            extra = ", unless you set `dynamic_batch_size=True` during the compilation" if dim == 0 else ""
            raise ValueError(
                f"Unable to pad {input_name} with shape: {input_tensor.shape} on dimension {dim} as input shapes must be inferior"
                f" than the static shapes used for compilation: {target_shapes}{extra}."
            )

    def _pad_to_compiled_shape(
        self, inputs: dict[str, "torch.Tensor"], padding_side: Literal["right", "left"] = "right"
    ):
        """
        Pads input tensors if they are not in valid shape.

        Args:
            inputs (`dict[str, "torch.Tensor"]`):
                Dictionary of input torch tensors.
            padding_side (`Literal["right", "left"]`, defaults to "right"):
                The side on which to apply the padding.
        """
        logger.info(f"Padding input tensors, the padding side is: {padding_side}.")
        for input_name, input_tensor in inputs.items():
            target_shapes = self.input_static_shapes[input_name]
            padding = ()
            if self._validate_static_shape(input_tensor.shape, target_shapes):
                continue

            # Dimensions other than 0
            for i in reversed(range(1, input_tensor.dim())):
                to_pad = target_shapes[i] - input_tensor.size(i)

                self._raise_if_invalid_padding(input_name, input_tensor, target_shapes, to_pad, i)
                padding += (0, to_pad) if padding_side == "right" else (to_pad, 0)

            if (
                self.preprocessors is not None
                and len(self.preprocessors) > 0
                and getattr(self.preprocessors[0], "pad_token_id", None)
                and input_name == "input_ids"
            ):
                pad_id = self.preprocessors[0].pad_token_id
            else:
                pad_id = 0

            input_tensor = torch.nn.functional.pad(input_tensor, padding, mode="constant", value=pad_id)

            # Pad to batch size: dimension 0 (pad_token_id can't be 0)
            padding = (0,) * len(padding)
            is_encoder_decoder = getattr(self.config, "is_encoder_decoder", False)
            if (
                not is_encoder_decoder
                and self.neuron_config.dynamic_batch_size is True
                and input_tensor.size(0) % target_shapes[0] == 0
            ):
                inputs[input_name] = input_tensor
                continue
            elif not is_encoder_decoder and self.neuron_config.dynamic_batch_size is True:
                target_shape = (input_tensor.size(0) // target_shapes[0] + 1) * target_shapes[0]
                to_pad = target_shape - input_tensor.size(0)
            else:
                to_pad = target_shapes[0] - input_tensor.size(0)
                self._raise_if_invalid_padding(input_name, input_tensor, target_shapes, to_pad, 0)
            padding += (0, to_pad) if padding_side == "right" else (to_pad, 0)

            pad_id = 1
            inputs[input_name] = torch.nn.functional.pad(input_tensor, padding, mode="constant", value=pad_id)

        return inputs

    @contextmanager
    def neuron_padding_manager(self, inputs: dict[str, "torch.Tensor"]):
        inputs = tuple(self._pad_to_compiled_shape(inputs).values())
        yield inputs

    @staticmethod
    def remove_padding(
        outputs: list[torch.Tensor],
        dims: list[int],
        indices: list[int],
        padding_side: Literal["right", "left"] = "right",
    ) -> list[torch.Tensor]:
        """
        Removes padding from output tensors.

        Args:
            outputs (`list[torch.Tensor]`):
                List of torch tensors which are inference output.
            dims (`list[int]`):
                List of dimensions in which we slice a tensor.
            indices (`list[int]`):
                List of indices in which we slice a tensor along an axis.
            padding_side (`Literal["right", "left"]`, defaults to "right"):
                The side on which the padding has been applied.
        """
        if len(dims) != len(indices):
            raise ValueError(f"The size of `dims`({len(dims)}) and indices`({len(indices)}) must be equal.")

        for dim, indice in zip(dims, indices):
            if padding_side == "right":
                outputs = [
                    torch.index_select(output_tensor, dim, torch.LongTensor(range(indice)))
                    for output_tensor in outputs
                ]
            elif padding_side == "left":
                outputs = [
                    torch.index_select(
                        output_tensor,
                        dim,
                        torch.LongTensor(range(output_tensor.shape[dim] - indice, output_tensor.shape[dim])),
                    )
                    for output_tensor in outputs
                ]

        return outputs

    @property
    def is_weights_neff_separated(self) -> bool:
        """
        Whether the Neuron model has separated weights and neff graph (by setting `inline_weights_to_neff=False` during the compilation).
        """
        return not self.config.neuron.get("inline_weights_to_neff")

    def can_generate(self) -> bool:
        """
        Returns whether this model can generate sequences with `.generate()`.
        """
        return isinstance(self, GenerationMixin)
