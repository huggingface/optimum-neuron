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
"""Base class for text-generation model architectures on neuron devices."""

import logging
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Optional, Tuple, Union

import torch
from huggingface_hub import HfApi, HfFolder, snapshot_download
from transformers import GenerationConfig

from ..exporters.neuron.model_configs import *  # noqa: F403
from ..exporters.tasks import TasksManager
from ..modeling_base import OptimizedModel
from .utils import is_transformers_neuronx_available
from .utils.version_utils import check_compiler_compatibility, get_neuronxcc_version


if is_transformers_neuronx_available():
    from transformers_neuronx.module import PretrainedModel as NeuronxPretrainedModel
    from transformers_neuronx.module import save_pretrained_split


if TYPE_CHECKING:
    from transformers import PretrainedConfig


logger = logging.getLogger(__name__)


def get_exporter(config, task):
    return TasksManager.get_exporter_config_constructor(model_type=config.model_type, exporter="neuron", task=task)()


class NeuronDecoderModel(OptimizedModel):
    """
    Base class to convert and run pre-trained transformers decoder models on Neuron devices.

    It implements the methods to convert a pre-trained transformers decoder model into a Neuron transformer model by:
    - transferring the checkpoint weights of the original into an optimized neuron graph,
    - compiling the resulting graph using the Neuron compiler.

    Common attributes:
        - model (`torch.nn.Module`) -- The decoder model with a graph optimized for neuron devices.
        - config ([`~transformers.PretrainedConfig`]) -- The configuration of the original model.
        - generation_config ([`~transformers.GenerationConfig`]) -- The generation configuration used by default when calling `generate()`.
    """

    CHECKPOINT_DIR = "checkpoint"
    COMPILED_DIR = "compiled"

    def __init__(
        self,
        model: torch.nn.Module,
        config: "PretrainedConfig",
        model_path: Union[str, Path, TemporaryDirectory],
        generation_config: Optional[GenerationConfig] = None,
    ):
        if not is_transformers_neuronx_available() or not isinstance(model, NeuronxPretrainedModel):
            raise ValueError("The source model must be a transformers_neuronx.PreTrainedModel.")

        super().__init__(model, config)
        self.model_path = model_path
        if generation_config is None:
            generation_config = GenerationConfig.from_model_config(config)
        self.generation_config = generation_config

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
        batch_size: Optional[int] = 1,
        num_cores: Optional[int] = 2,
        auto_cast_type: Optional[str] = "f32",
        **kwargs,
    ) -> "NeuronDecoderModel":
        if not is_transformers_neuronx_available():
            raise ModuleNotFoundError("The transformers_neuronx package is required to export the model.")

        if task is None:
            task = TasksManager.infer_task_from_model(cls.auto_model_class)

        # Instantiate the exporter for the specified configuration and task
        exporter = get_exporter(config, task)

        # Split kwargs between model and neuron args
        model_kwargs, neuron_kwargs = exporter.split_kwargs(**kwargs)

        # Instantiate the transformers model checkpoint
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
            **model_kwargs,
        )

        # Save the model checkpoint in a temporary directory
        checkpoint_dir = TemporaryDirectory()
        save_pretrained_split(model, checkpoint_dir.name)

        # Update the config
        config.neuron = {
            "task": task,
            "batch_size": batch_size,
            "num_cores": num_cores,
            "auto_cast_type": auto_cast_type,
            "neuron_kwargs": neuron_kwargs,
            "compiler_type": "neuronx-cc",
            "compiler_version": get_neuronxcc_version(),
        }

        return cls._from_pretrained(checkpoint_dir, config)

    @classmethod
    def _get_neuron_paths(
        cls, model_dir: Union[str, Path, TemporaryDirectory], token: Optional[str] = None
    ) -> Tuple[str, str, str]:
        if isinstance(model_dir, TemporaryDirectory):
            model_path = model_dir.name
            # We are in the middle of an export: the checkpoint is in the temporary model directory
            checkpoint_path = model_path
            # There are no compiled artifacts yet
            compiled_path = None
        else:
            # The model has already been exported
            if os.path.isdir(model_dir):
                model_path = model_dir
            else:
                # Download the neuron model from the Hub
                model_path = snapshot_download(model_dir, token=token)
            # The checkpoint is in a subdirectory
            checkpoint_path = os.path.join(model_path, cls.CHECKPOINT_DIR)
            # So are the compiled artifacts
            compiled_path = os.path.join(model_path, cls.COMPILED_DIR)
        return model_path, checkpoint_path, compiled_path

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path, TemporaryDirectory],
        config: "PretrainedConfig",
        use_auth_token: Optional[str] = None,
        **kwargs,
    ) -> "NeuronDecoderModel":
        # Verify we are actually trying to load a neuron model
        neuron_config = getattr(config, "neuron", None)
        if neuron_config is None:
            raise ValueError(
                "The specified directory does not contain a neuron model. "
                "Please convert your model to neuron format by passing export=True."
            )

        # Evaluate the configuration passed during export
        task = neuron_config["task"]
        batch_size = neuron_config["batch_size"]
        num_cores = neuron_config["num_cores"]
        auto_cast_type = neuron_config["auto_cast_type"]
        neuron_kwargs = neuron_config["neuron_kwargs"]

        check_compiler_compatibility(neuron_config["compiler_type"], neuron_config["compiler_version"])

        exporter = get_exporter(config, task)

        model_path, checkpoint_path, compiled_path = cls._get_neuron_paths(model_id, use_auth_token)

        neuronx_model = exporter.neuronx_class.from_pretrained(
            checkpoint_path, batch_size=batch_size, tp_degree=num_cores, amp=auto_cast_type, **neuron_kwargs
        )

        if compiled_path is not None:
            # Specify the path where compiled artifacts are stored before conversion
            neuronx_model._load_compiled_artifacts(compiled_path)

        # Compile the Neuron model (if present compiled artifacts will be reloaded instead of compiled)
        os.environ["NEURON_CC_FLAGS"] = "--model-type=transformer-inference"
        neuronx_model.to_neuron()

        # Try to reload the generation config (if any)
        generation_config = None
        try:
            generation_config = GenerationConfig.from_pretrained(model_path)
        except OSError:
            logger.info("Generation config file not found, using a generation config created from the model config.")

        return cls(neuronx_model, config, model_id, generation_config)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def _save_pretrained(self, save_directory: Union[str, Path]):
        _, src_chkpt_path, src_compiled_path = self._get_neuron_paths(self.model_path)
        _, dst_chkpt_path, dst_compiled_path = self._get_neuron_paths(save_directory)

        shutil.copytree(src_chkpt_path, dst_chkpt_path)

        if src_compiled_path is None:
            # The compiled model has never been serialized: do it now
            self.model._save_compiled_artifacts(dst_compiled_path)
        else:
            shutil.copytree(src_compiled_path, dst_compiled_path)

        if isinstance(self.model_path, TemporaryDirectory):
            # Let temporary directory go out-of-scope to release disk space
            self.model_path = save_directory

        self.generation_config.save_pretrained(save_directory)

    def push_to_hub(
        self,
        save_directory: str,
        repository_id: str,
        private: Optional[bool] = None,
        use_auth_token: Union[bool, str] = True,
        endpoint: Optional[str] = None,
    ) -> str:
        if isinstance(use_auth_token, str):
            huggingface_token = use_auth_token
        elif use_auth_token:
            huggingface_token = HfFolder.get_token()
        else:
            raise ValueError("You need to provide `use_auth_token` to be able to push to the hub")
        api = HfApi(endpoint=endpoint)

        user = api.whoami(huggingface_token)
        self.git_config_username_and_email(git_email=user["email"], git_user=user["fullname"])

        api.create_repo(
            token=huggingface_token,
            repo_id=repository_id,
            exist_ok=True,
            private=private,
        )
        for path, subdirs, files in os.walk(save_directory):
            for name in files:
                local_file_path = os.path.join(path, name)
                hub_file_path = os.path.relpath(local_file_path, save_directory)
                api.upload_file(
                    token=huggingface_token,
                    repo_id=repository_id,
                    path_or_fileobj=os.path.join(os.getcwd(), local_file_path),
                    path_in_repo=hub_file_path,
                )
