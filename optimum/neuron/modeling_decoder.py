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

import copy
import logging
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Optional, Tuple, Union

from huggingface_hub import HfApi, get_token, snapshot_download
from huggingface_hub.utils import is_google_colab
from transformers import AutoConfig, AutoModel, GenerationConfig

from ..exporters.neuron.model_configs import *  # noqa: F403
from ..exporters.tasks import TasksManager
from ..modeling_base import OptimizedModel
from .utils import ModelCacheEntry, hub_neuronx_cache, is_transformers_neuronx_available
from .utils.require_utils import requires_transformers_neuronx
from .utils.version_utils import check_compiler_compatibility, get_neuronxcc_version


if is_transformers_neuronx_available():
    from transformers_neuronx.module import save_split


if TYPE_CHECKING:
    from transformers import PretrainedConfig


logger = logging.getLogger(__name__)


def get_exporter(config, task):
    return TasksManager.get_exporter_config_constructor(model_type=config.model_type, exporter="neuron", task=task)()


@requires_transformers_neuronx
def get_available_cores() -> int:
    """A helper to get the number of available cores.

    This number depends first on the actual number of cores, then on the
    content of the NEURON_RT_NUM_CORES and NEURON_RT_VISIBLE_CORES variables.
    """
    max_cores = len(os.listdir("/sys/class/neuron_device/")) * 2
    num_cores = os.environ.get("NEURON_RT_NUM_CORES", max_cores)
    if num_cores != max_cores:
        num_cores = int(num_cores)
    num_cores = min(num_cores, max_cores)
    visible_cores = os.environ.get("NEURON_RT_VISIBLE_CORES", num_cores)
    if visible_cores != num_cores:
        # Assume NEURON_RT_VISIBLE_CORES is in the form '4' or '7-15'
        if "-" in visible_cores:
            start, end = visible_cores.split("-")
            visible_cores = int(end) - int(start) + 1
        else:
            visible_cores = 1
    visible_cores = min(visible_cores, num_cores)
    return visible_cores


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

    model_type = "neuron_model"
    auto_model_class = AutoModel

    CHECKPOINT_DIR = "checkpoint"
    COMPILED_DIR = "compiled"

    @requires_transformers_neuronx
    def __init__(
        self,
        config: "PretrainedConfig",
        checkpoint_dir: Union[str, Path, TemporaryDirectory],
        compiled_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        generation_config: Optional[GenerationConfig] = None,
    ):
        neuron_config = getattr(config, "neuron", None)
        if neuron_config is None:
            raise ValueError(
                "The specified model is not a neuron model."
                "Please convert your model to neuron format by passing export=True."
            )

        self.checkpoint_dir = checkpoint_dir
        self.compiled_dir = compiled_dir
        if generation_config is None:
            logger.info("Generation config file not found, using a generation config created from the model config.")
            generation_config = GenerationConfig.from_model_config(config)
        self.generation_config = generation_config
        # Registers the NeuronModelForXXX classes into the transformers AutoModel classes to avoid warnings when creating
        # a pipeline https://github.com/huggingface/transformers/blob/3d3204c025b6b5de013e07dd364208e28b4d9589/src/transformers/pipelines/base.py#L940
        AutoConfig.register(self.model_type, AutoConfig)
        if hasattr(self.auto_model_class, "register"):
            self.auto_model_class.register(AutoConfig, self.__class__)

        # Evaluate the configuration passed during export
        task = neuron_config["task"]
        batch_size = neuron_config["batch_size"]
        sequence_length = neuron_config["sequence_length"]
        num_cores = neuron_config["num_cores"]
        auto_cast_type = neuron_config["auto_cast_type"]

        check_compiler_compatibility(neuron_config["compiler_type"], neuron_config["compiler_version"])

        exporter = get_exporter(config, task)

        # transformers-neuronx uses f32/f16 instead of fp32/fp16
        auto_cast_type = auto_cast_type.replace("p", "")
        checkpoint_path = checkpoint_dir.name if isinstance(checkpoint_dir, TemporaryDirectory) else checkpoint_dir
        neuronx_model = exporter.neuronx_class.from_pretrained(
            checkpoint_path,
            batch_size=batch_size,
            n_positions=sequence_length,
            tp_degree=num_cores,
            amp=auto_cast_type,
        )

        if compiled_dir is not None:
            # Specify the path where compiled artifacts are stored before conversion
            neuronx_model.load(compiled_dir)

        # When compiling, only create a cache entry if the model comes from the hub
        checkpoint_id = neuron_config.get("checkpoint_id", None)
        cache_entry = None if checkpoint_id is None else ModelCacheEntry(checkpoint_id, config)

        # Export the model using the Optimum Neuron Cache
        with hub_neuronx_cache(entry=cache_entry):
            available_cores = get_available_cores()
            if num_cores > available_cores:
                raise ValueError(
                    f"The specified number of cores ({num_cores}) exceeds the number of cores available ({available_cores})."
                )
            neuron_rt_num_cores = os.environ.get("NEURON_RT_NUM_CORES", None)
            # Restrict the number of cores used to allow multiple models on the same host
            os.environ["NEURON_RT_NUM_CORES"] = str(num_cores)
            # Load the model on neuron cores (if found in cache or compiled directory, the NEFF files
            # will be reloaded instead of compiled)
            neuronx_model.to_neuron()
            if neuron_rt_num_cores is None:
                os.environ.pop("NEURON_RT_NUM_CORES")
            else:
                os.environ["NEURON_RT_NUM_CORES"] = neuron_rt_num_cores

        super().__init__(neuronx_model, config)

    @classmethod
    def _create_checkpoint(
        cls,
        model_id: str,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        task: Optional[str] = None,
        **kwargs,
    ) -> TemporaryDirectory:
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
            **kwargs,
        )

        # Save the model checkpoint in a temporary directory
        checkpoint_dir = TemporaryDirectory()
        model.save_pretrained(
            checkpoint_dir.name, save_function=save_split, safe_serialization=False, max_shard_size="10000GB"
        )
        return checkpoint_dir

    @classmethod
    def get_export_config(
        cls,
        model_id: str,
        config: "PretrainedConfig",
        use_auth_token: Optional[str] = None,
        revision: Optional[str] = None,
        task: Optional[str] = None,
        batch_size: Optional[int] = None,
        sequence_length: Optional[int] = None,
        num_cores: Optional[int] = None,
        auto_cast_type: Optional[str] = None,
    ) -> "PretrainedConfig":
        if task is None:
            task = TasksManager.infer_task_from_model(cls.auto_model_class)

        if os.path.isdir(model_id):
            checkpoint_id = None
            checkpoint_revision = None
        else:
            checkpoint_id = model_id
            # Get the exact checkpoint revision (SHA1)
            api = HfApi(token=use_auth_token)
            model_info = api.repo_info(model_id, revision=revision)
            checkpoint_revision = model_info.sha

        if batch_size is None:
            batch_size = 1
        # If the sequence_length was not specified, deduce it from the model configuration
        if sequence_length is None:
            # Note: for older models, max_position_embeddings is an alias for n_positions
            sequence_length = config.max_position_embeddings
        if num_cores is None:
            # Use all available cores
            num_cores = get_available_cores()
        if auto_cast_type is None:
            auto_cast_type = "fp32"
            if config.torch_dtype == "float16":
                auto_cast_type = "fp16"
            elif config.torch_dtype == "bfloat16":
                auto_cast_type = "bf16"

        new_config = copy.deepcopy(config)
        new_config.neuron = {
            "task": task,
            "batch_size": batch_size,
            "num_cores": num_cores,
            "auto_cast_type": auto_cast_type,
            "sequence_length": sequence_length,
            "compiler_type": "neuronx-cc",
            "compiler_version": get_neuronxcc_version(),
            "checkpoint_id": checkpoint_id,
            "checkpoint_revision": checkpoint_revision,
        }
        return new_config

    @classmethod
    @requires_transformers_neuronx
    def _from_transformers(cls, *args, **kwargs):
        # Deprecate it when optimum uses `_export` as from_pretrained_method in a stable release.
        return cls._export(*args, **kwargs)

    @classmethod
    @requires_transformers_neuronx
    def _export(
        cls,
        model_id: str,
        config: "PretrainedConfig",
        use_auth_token: Optional[str] = None,
        revision: Optional[str] = None,
        task: Optional[str] = None,
        batch_size: Optional[int] = None,
        sequence_length: Optional[int] = None,
        num_cores: Optional[int] = None,
        auto_cast_type: Optional[str] = "fp32",
        **kwargs,
    ) -> "NeuronDecoderModel":
        if not os.path.isdir("/sys/class/neuron_device/"):
            raise SystemError("Decoder models can only be exported on a neuron platform.")

        # Update the config
        new_config = cls.get_export_config(
            model_id,
            config,
            use_auth_token=use_auth_token,
            revision=revision,
            task=task,
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_cores=num_cores,
            auto_cast_type=auto_cast_type,
        )

        # Instantiate the transformers model checkpoint
        checkpoint_dir = cls._create_checkpoint(
            model_id,
            task=new_config.neuron["task"],
            revision=revision,
            **kwargs,
        )

        # Try to reload the generation config (if any)
        generation_config = None
        try:
            generation_config = GenerationConfig.from_pretrained(model_id)
        except OSError:
            pass

        return cls(new_config, checkpoint_dir, generation_config=generation_config)

    @classmethod
    def _get_neuron_dirs(cls, model_path: Union[str, Path]) -> Tuple[str, str]:
        # The checkpoint is in a subdirectory
        checkpoint_dir = os.path.join(model_path, cls.CHECKPOINT_DIR)
        # So are the compiled artifacts
        compiled_dir = os.path.join(model_path, cls.COMPILED_DIR)
        return checkpoint_dir, compiled_dir

    @classmethod
    @requires_transformers_neuronx
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: "PretrainedConfig",
        use_auth_token: Optional[str] = None,
        revision: Optional[str] = None,
        **kwargs,
    ) -> "NeuronDecoderModel":
        # Verify we are actually trying to load a neuron model
        neuron_config = getattr(config, "neuron", None)
        if neuron_config is None:
            raise ValueError(
                "The specified directory does not contain a neuron model."
                "Please convert your model to neuron format by passing export=True."
            )
        check_compiler_compatibility(neuron_config["compiler_type"], neuron_config["compiler_version"])

        model_path = model_id
        if not os.path.isdir(model_id):
            model_path = snapshot_download(model_id, token=use_auth_token, revision=revision)

        checkpoint_dir, compiled_dir = cls._get_neuron_dirs(model_path)
        if not os.path.isdir(checkpoint_dir):
            # Try to recreate checkpoint from neuron config
            task = neuron_config["task"]
            checkpoint_id = neuron_config.get("checkpoint_id", None)
            if checkpoint_id is None:
                raise ValueError("Unable to fetch the neuron model weights files.")
            checkpoint_revision = neuron_config["checkpoint_revision"]
            checkpoint_dir = cls._create_checkpoint(
                checkpoint_id,
                task=task,
                revision=checkpoint_revision,
                use_auth_token=use_auth_token,
                **kwargs,
            )
        assert os.path.isdir(compiled_dir)

        # Try to reload the generation config (if any)
        generation_config = None
        try:
            generation_config = GenerationConfig.from_pretrained(model_id)
        except OSError:
            pass

        return cls(config, checkpoint_dir, compiled_dir=compiled_dir, generation_config=generation_config)

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def _save_pretrained(self, save_directory: Union[str, Path]):
        dst_checkpoint_path, dst_compiled_path = self._get_neuron_dirs(save_directory)

        def copy_dir_to_path(src_dir: Union[str, Path, TemporaryDirectory], dst_path: Union[str, Path]):
            if isinstance(src_dir, TemporaryDirectory):
                shutil.copytree(src_dir.name, dst_path, dirs_exist_ok=True)
            elif not os.path.samefile(src_dir, dst_path):
                os.symlink(dst_path, src_dir)

        # Copy checkpoint directory (it always exists)
        copy_dir_to_path(self.checkpoint_dir, dst_checkpoint_path)
        self.checkpoint_dir = dst_checkpoint_path
        # Save or create compiled directory
        if self.compiled_dir is None:
            # The compilation artifacts have never been saved, do it now
            self.model.save(dst_compiled_path)
        else:
            copy_dir_to_path(self.compiled_dir, dst_compiled_path)
        self.compiled_dir = dst_compiled_path
        self.generation_config.save_pretrained(save_directory)

    def push_to_hub(
        self,
        save_directory: str,
        repository_id: str,
        private: Optional[bool] = None,
        revision: Optional[str] = None,
        use_auth_token: Union[bool, str] = True,
        endpoint: Optional[str] = None,
    ) -> str:
        if isinstance(use_auth_token, str):
            huggingface_token = use_auth_token
        elif use_auth_token:
            huggingface_token = get_token()
        else:
            raise ValueError("You need to provide `use_auth_token` to be able to push to the hub")
        api = HfApi(endpoint=endpoint)

        user = api.whoami(huggingface_token)
        if is_google_colab():
            # Only in Google Colab to avoid the warning message
            self.git_config_username_and_email(git_email=user["email"], git_user=user["fullname"])

        api.create_repo(
            token=huggingface_token,
            repo_id=repository_id,
            exist_ok=True,
            private=private,
        )
        ignore_patterns = []
        neuron_config = getattr(self.config, "neuron")
        checkpoint_id = neuron_config.get("checkpoint_id", None)
        if checkpoint_id is not None:
            # Avoid uploading checkpoints when the original model is available on the hub
            ignore_patterns = [self.CHECKPOINT_DIR + "/*"]
        api.upload_folder(
            repo_id=repository_id,
            folder_path=save_directory,
            token=huggingface_token,
            revision=revision,
            ignore_patterns=ignore_patterns,
        )
