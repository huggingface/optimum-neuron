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
"""NeuronModel base classe for inference on neuron devices using the same API as Transformers."""

import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, List, Optional, Union

import torch
from huggingface_hub import HfApi, HfFolder, hf_hub_download
from transformers import AutoConfig, AutoModel

from ..exporters.neuron import export
from ..exporters.neuron.model_configs import *  # noqa: F403
from ..exporters.tasks import TasksManager
from ..modeling_base import OptimizedModel
from ..utils import DEFAULT_DUMMY_SHAPES
from ..utils.save_utils import maybe_load_preprocessors, maybe_save_preprocessors
from .utils import NEURON_FILE_NAME


if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from ..exporters.neuron import NeuronConfig


logger = logging.getLogger(__name__)


class NeuronModel(OptimizedModel):
    """
    Base class running compiled and optimized models on Neuron devices.

    The NeuronModel implements generic methods for interacting with the Hugging Face Hub as well as compiling vanilla
    transformers models to neuron-optimized TorchScript module and export it using `optimum.exporters.neuron` toolchain.

    Class attributes:
        - model_type (`str`, *optional*, defaults to `"neuron_model"`) -- The name of the model type to use when
        registering the NeuronModel classes.
        - auto_model_class (`Type`, *optional*, defaults to `AutoModel`) -- The "AutoModel" class to be represented by the
        current NeuronModel class.

    Common attributes:
        - model (`torch.jit._script.ScriptModule`) -- The loaded `ScriptModule` compiled for neuron devices.
        - config ([`~transformers.PretrainedConfig`] -- The configuration of the model.
        - model_save_dir (`Path`) -- The directory where a neuron compiled model is saved.
        By default, if the loaded model is local, the directory where the original model will be used. Otherwise, the
        cache directory will be used.
    """

    model_type = "neuron_model"
    auto_model_class = AutoModel

    def __init__(
        self,
        model: torch.jit._script.ScriptModule,
        config: "PretrainedConfig",
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        preprocessors: Optional[List] = None,
        neuron_config: "NeuronConfig" = None,
        **kwargs,
    ):
        super().__init__(model, config)

        self.model = model
        self.config = config
        self.model_save_dir = self._normalize_path(model_save_dir)
        self.preprocessors = preprocessors if preprocessors is not None else []
        self.model_file_name = getattr(self.model_save_dir, "name", None)
        self.neuron_config = neuron_config

        # Registers the NeuronModelForXXX classes into the transformers AutoModel classes to avoid warnings when creating
        # a pipeline https://github.com/huggingface/transformers/blob/3d3204c025b6b5de013e07dd364208e28b4d9589/src/transformers/pipelines/base.py#L940
        AutoConfig.register(self.model_type, AutoConfig)
        if hasattr(self.auto_model_class, "register"):
            self.auto_model_class.register(AutoConfig, self.__class__)

    @staticmethod
    def load_model(path: Union[str, Path]) -> torch.jit._script.ScriptModule:
        """
        Loads a TorchScript module compiled by neuron(x)-cc compiler. It will be first loaded onto CPU and then moved to
        one or multiple [NeuronCore](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/neuroncores-arch.html).

        Args:
            path (`Union[str, Path]`):
                Path of the compiled model.
        """
        if not isinstance(path, str):
            path = str(path)

        return torch.jit.load(path)

    def _save_pretrained(self, save_directory: Union[str, Path]):
        """
        Saves a model and its configuration file to a directory, so that it can be re-loaded using the
        [`~optimum.neuron.modeling_base.NeuronModel.from_pretrained`] class method.

        Args:
            save_directory (`Union[str, Path]`):
                Directory where to save the model file.
        """
        dst_path = Path(save_directory) / self.model_file_name

        torch.jit.save(self.model, dst_path)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: "PretrainedConfig",
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        file_name: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        neuron_config: "NeuronConfig" = None,
        **kwargs,
    ) -> "NeuronModel":
        model_path = Path(model_id)

        if file_name is None:
            if model_path.is_dir():
                neuron_files = list(model_path.glob("*.neuron"))
            else:
                if isinstance(use_auth_token, bool):
                    token = HfFolder().get_token()
                else:
                    token = use_auth_token
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

        preprocessors = None
        if model_path.is_dir():
            model = NeuronModel.load_model(model_path / file_name)
            new_model_save_dir = model_path
            preprocessors = maybe_load_preprocessors(model_id)
        else:
            model_cache_path = hf_hub_download(
                repo_id=model_id,
                filename=file_name,
                subfolder=subfolder,
                use_auth_token=use_auth_token,
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
            )

            model = NeuronModel.load_model(model_cache_path)
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
            preprocessors=preprocessors,
            neuron_config=neuron_config,
        )

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        config: "PretrainedConfig",
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        task: Optional[str] = None,
        auto_cast: Optional[str] = None,
        auto_cast_type: Optional[str] = None,
        disable_fast_relayout: Optional[bool] = False,
        **kwargs_shapes,
    ) -> "NeuronModel":
        """
        Export a vanilla Transformers model into a neuron-compiled TorchScript Module using `optimum.exporters.neuron.export`.

        Arguments:
            kwargs_shapes (`Dict`):
                Shapes to use during inference. This argument allows to override the default shapes used during the export.
        """
        if task is None:
            task = cls._auto_model_to_task(cls.auto_model_class)

        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)

        model = TasksManager.get_model_from_task(
            task=task,
            model_name_or_path=model_id,
            subfolder=subfolder,
            revision=revision,
            framework="pt",
            cache_dir=cache_dir,
            use_auth_token=use_auth_token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
        )

        neuron_config_constructor = TasksManager.get_exporter_config_constructor(
            model=model, exporter="neuron", task=task
        )
        input_shapes = {}
        for input_name in DEFAULT_DUMMY_SHAPES.keys():
            input_shapes[input_name] = (
                kwargs_shapes[input_name] if input_name in kwargs_shapes else DEFAULT_DUMMY_SHAPES[input_name]
            )
        neuron_config = neuron_config_constructor(model.config, **input_shapes)

        export(
            model=model,
            config=neuron_config,
            output=save_dir_path / NEURON_FILE_NAME,
            auto_cast=auto_cast,
            auto_cast_type=auto_cast_type,
            disable_fast_relayout=disable_fast_relayout,
        )

        config.save_pretrained(save_dir_path)
        maybe_save_preprocessors(model_id, save_dir_path, src_subfolder=subfolder)

        return cls._from_pretrained(save_dir_path, config, neuron_config=neuron_config)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    @classmethod
    def _auto_model_to_task(cls, auto_model_class):
        """
        Get the task corresponding to a class (for example AutoModelForXXX in transformers).
        """
        return TasksManager.infer_task_from_model(auto_model_class)

    def _normalize_path(self, path: Optional[Union[str, Path, TemporaryDirectory]] = None):
        """
        Convert a path to an instance of `pathlib.Path` or remain None.
        """
        if isinstance(path, TemporaryDirectory):
            return Path(path.name)
        elif isinstance(path, str):
            return Path(path)
        else:
            return path

    def pad_to_compiled_shapes(self, **inputs):
        pass

    def validate_static_shape(self, **kwargs_shapes):
        dim_to_pad = []
        if self.neuron_config is not None:
            for name in kwargs_shapes.keys():
                compiled_shape = getattr(self.neuron_config, name, None)
                rt_shape = kwargs_shapes[name]
                if rt_shape > compiled_shape:
                    raise
                elif rt_shape < compiled_shape:
                    dim_to_pad.append(name)

        return dim_to_pad
