# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team.
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
"""Neuron configuration class and utilities."""

import copy
import enum
import json
import os
from dataclasses import dataclass, is_dataclass
from typing import Any

import torch
from transformers.utils import (
    PushToHubMixin,
    cached_file,
    download_url,
    is_remote_url,
    logging,
)

from .utils.version_utils import get_neuronxcc_version
from .version import __version__


logger = logging.get_logger(__name__)

NEURON_CONFIG_NAME = "neuron_config.json"

_NEURON_CONFIG_FOR_KEY = {}
_KEY_FOR_NEURON_CONFIG = {}


def register_neuron_config(cls):
    key = cls.__name__
    assert issubclass(cls, NeuronConfig)
    _KEY_FOR_NEURON_CONFIG[cls] = key
    _NEURON_CONFIG_FOR_KEY[key] = cls
    return cls


@dataclass
class NeuronConfig(PushToHubMixin):
    # no-format
    """
    Class that holds a configuration for a neuron model.

    Must be subclassed.

    """

    def __hash__(self):
        return hash(self.to_json_string())

    def __eq__(self, other):
        if not isinstance(other, NeuronConfig):
            return False

        self_json = self.to_json_string()
        other_json = other.to_json_string()
        return self_json == other_json

    def __repr__(self):
        return f"{self.__class__.__name__} {self.to_json_string()}"

    def save_pretrained(
        self,
        save_directory: str | os.PathLike,
        config_file_name: str | os.PathLike | None = None,
        push_to_hub: bool = False,
        **kwargs,
    ):
        r"""
        Save a neuron configuration object to the directory `save_directory`, so that it can be re-loaded using the
        [`~NeuronConfig.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the configuration JSON file will be saved (will be created if it does not exist).
            config_file_name (`str` or `os.PathLike`, *optional*, defaults to `"neuron_config.json"`):
                Name of the generation configuration JSON file to be saved in `save_directory`.
            push_to_hub (`bool`, *optional*, defaults to `False`):
                Whether or not to push your model to the Hugging Face model hub after saving it. You can specify the
                repository you want to push to with `repo_id` (will default to the name of `save_directory` in your
                namespace).
            kwargs (`dict[str, Any]`, *optional*):
                Additional key word arguments passed along to the [`~utils.PushToHubMixin.push_to_hub`] method.
        """

        config_file_name = config_file_name if config_file_name is not None else NEURON_CONFIG_NAME

        if os.path.isfile(save_directory):
            raise AssertionError(f"Provided path ({save_directory}) should be a directory, not a file")

        os.makedirs(save_directory, exist_ok=True)

        if push_to_hub:
            commit_message = kwargs.pop("commit_message", None)
            repo_id = kwargs.pop("repo_id", save_directory.split(os.path.sep)[-1])
            repo_id = self._create_repo(repo_id, **kwargs)
            files_timestamps = self._get_files_timestamps(save_directory)

        output_config_file = os.path.join(save_directory, config_file_name)

        self.to_json_file(output_config_file)
        logger.info(f"Configuration saved in {output_config_file}")

        if push_to_hub:
            self._upload_modified_files(
                save_directory,
                repo_id,
                files_timestamps,
                commit_message=commit_message,
                token=kwargs.get("token"),
            )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name: str | os.PathLike,
        config_file_name: str | os.PathLike | None = None,
        token: str | bool | None = None,
        revision: str = "main",
    ) -> "NeuronConfig":
        r"""
        Instantiate a [`NeuronConfig`] from a neuron configuration file.

        Args:
            pretrained_model_name (`str` or `os.PathLike`):
                This can be either:

                - a string, the *model id* of a pretrained model configuration hosted inside a model repo on
                  huggingface.co.
                - a path to a *directory* containing a configuration file saved using the
                  [`~NeuronConfig.save_pretrained`] method, e.g., `./my_model_directory/`.
            config_file_name (`str` or `os.PathLike`, *optional*, defaults to `"neuron_config.json"`):
                Name of the neuron configuration JSON file to be loaded from `pretrained_model_name`.
            token (`str` or `bool`, *optional*):
                The token to use as HTTP bearer authorization for remote files. If `True`, or not specified, will use
                the token generated when running `huggingface-cli login` (stored in `~/.huggingface`).
            revision (`str`, *optional*, defaults to `"main"`):
                The specific model version to use. It can be a branch name, a tag name, or a commit id, since we use a
                git-based system for storing models and other artifacts on huggingface.co, so `revision` can be any
                identifier allowed by git.

        Returns:
            [`NeuronConfig`]: The configuration object instantiated from this pretrained model.

        ```"""
        config_file_name = config_file_name if config_file_name is not None else NEURON_CONFIG_NAME
        config_path = os.path.join(pretrained_model_name, config_file_name)
        config_path = str(config_path)

        is_local = os.path.exists(config_path)
        if os.path.isfile(config_path):
            # Special case when config_path is a local file
            resolved_config_file = config_path
            is_local = True
        elif is_remote_url(config_path):
            configuration_file = config_path
            resolved_config_file = download_url(config_path)
        else:
            configuration_file = config_file_name
            try:
                # Load from local folder or from cache or download from model Hub and cache
                resolved_config_file = cached_file(
                    pretrained_model_name,
                    configuration_file,
                    token=token,
                    revision=revision,
                )
            except EnvironmentError:
                # Raise any environment error raise by `cached_file`. It will have a helpful error message adapted to
                # the original exception.
                raise
            except Exception:
                # For any other exception, we throw a generic error.
                raise EnvironmentError(
                    f"Can't load the configuration of '{pretrained_model_name}'. If you were trying to load it"
                    " from 'https://huggingface.co/models', make sure you don't have a local directory with the same"
                    f" name. Otherwise, make sure '{pretrained_model_name}' is the correct path to a directory"
                    f" containing a {configuration_file} file"
                )

        try:
            # Load config dict
            with open(resolved_config_file, "r", encoding="utf-8") as reader:
                text = reader.read()
            config_dict = json.loads(text)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise EnvironmentError(
                f"It looks like the config file at '{resolved_config_file}' is not a valid JSON file."
            )

        if is_local:
            logger.info(f"loading configuration file {resolved_config_file}")
        else:
            logger.info(f"loading configuration file {configuration_file} from cache at {resolved_config_file}")

        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: dict[str, Any]) -> "NeuronConfig":
        """
        Instantiates a [`NeuronConfig`] from a Python dictionary of parameters.

        Args:
            config_dict (`dict[str, Any]`):
                Dictionary that will be used to instantiate the configuration object.

        Returns:
            [`NeuronConfig`]: The configuration object instantiated from those parameters.
        """
        for key in ["optimum_neuron_version", "_serialized_key", "neuronxcc_version"]:
            if key not in config_dict:
                raise ValueError(f"Invalid neuron configuration: missing mandatory {key} attribute.")
        # Check versions
        version = config_dict.pop("optimum_neuron_version")
        if version != __version__:
            raise ValueError(
                f"This neuron configuration file was generated with optimum-neuron version {version}, but this version is {__version__}."
            )
        cfg_neuronxcc_version = config_dict.pop("neuronxcc_version")
        neuronxcc_version = get_neuronxcc_version()
        if cfg_neuronxcc_version != neuronxcc_version:
            raise RuntimeError(
                "This neuron configuration corresponds to a model exported using version"
                f" {cfg_neuronxcc_version} of the neuronxcc compiler, but {neuronxcc_version} is installed."
            )
        # Retrieve neuron config class from serialized key
        _serialized_key = config_dict.pop("_serialized_key")
        if cls is NeuronConfig:
            # Force the registration of the neuron configuration classes
            from .models import neuron_config  # noqa F401

            # We need to identify the actual neuron configuration class for this serialized key
            if _serialized_key is None:
                raise ValueError("Neuron configuration is invalid: unable to identify the serialized key")
            cls = _NEURON_CONFIG_FOR_KEY.get(_serialized_key, None)
            if cls is None:
                raise ValueError(f"No neuron configuration registered for the {_serialized_key} serialized key")
        else:
            assert _KEY_FOR_NEURON_CONFIG[cls] == _serialized_key
        config = cls(**{**config_dict})
        logger.info(f"Neuron config {config}")
        return config

    def to_dict(self) -> dict[str, Any]:
        """
        Serializes this instance to a Python dictionary.

        Returns:
            `dict[str, Any]`: Dictionary of all the attributes that make up this configuration instance.
        """
        output = copy.deepcopy(self)

        serializable_types = (str, int, float, bool)

        def _to_dict(obj):
            if obj is None or isinstance(obj, serializable_types):
                return obj
            elif isinstance(obj, enum.Enum):
                return obj.value
            elif isinstance(obj, torch.Tensor):
                return obj.tolist()
            elif isinstance(obj, list):
                return [_to_dict(e) for e in obj]
            elif isinstance(obj, dict):
                return {_to_dict(k): _to_dict(v) for k, v in obj.items()}
            elif isinstance(obj, tuple):
                return str(tuple(_to_dict(e) for e in obj))
            elif isinstance(obj, torch.dtype):
                return str(obj).split(".")[1]
            else:
                as_dict = obj.__dict__
                return _to_dict(as_dict)

        output = _to_dict(output)

        # Add serialized key as it is required to identify the NeuronConfig class when deserializing the file
        cls = self.__class__
        _serialized_key = _KEY_FOR_NEURON_CONFIG.get(cls, None)
        if _serialized_key is None:
            raise ValueError(f"Unable to identify the serialized key for {cls.__name__}. Did you register it ?")
        output["_serialized_key"] = _serialized_key
        # Add optimum-neuron version to check compatibility
        output["optimum_neuron_version"] = __version__
        # Also add compiler version
        output["neuronxcc_version"] = get_neuronxcc_version()

        return output

    def to_json_string(self) -> str:
        """
        Serializes this instance to a JSON string.

        Returns:
            `str`: String containing all the attributes that make up this configuration instance in JSON format.
        """
        config_dict = self.to_dict()

        def convert_keys_to_string(obj):
            if isinstance(obj, dict):
                return {str(key): convert_keys_to_string(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_keys_to_string(item) for item in obj]
            else:
                return obj

        def convert_dataclass_to_dict(obj):
            if isinstance(obj, dict):
                return {key: convert_dataclass_to_dict(value) for key, value in obj.items()}
            elif is_dataclass(obj):
                return obj.to_dict()
            else:
                return obj

        config_dict = convert_keys_to_string(config_dict)
        config_dict = convert_dataclass_to_dict(config_dict)

        return json.dumps(config_dict, indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path: Union[str, os.PathLike]):
        """
        Save this instance to a JSON file.

        Args:
            json_file_path (`str` or `os.PathLike`):
                Path to the JSON file in which this configuration instance's parameters will be saved.
        """
        with open(json_file_path, "w", encoding="utf-8") as writer:
            writer.write(self.to_json_string())
