# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
import copy
import logging
import os
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path

import neuronx_distributed.trace.hlo_utils as hlo_utils
import torch
from huggingface_hub import HfApi, snapshot_download
from neuronx_distributed.trace.model_builder import ModelBuilder
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, PretrainedConfig

from ..modeling_utils import NeuronPreTrainedModel
from .config import NxDNeuronConfig
from .graph_builder import NxDGraphBuilder
from .modules.checkpoint import (
    load_state_dict,
)


logger = logging.getLogger("Neuron")


def normalize_path(path):
    """Normalize path separators and ensure path ends with a trailing slash"""
    normalized = os.path.normpath(path)
    return os.path.join(normalized, "")


def get_shards_path(dest_path):
    return os.path.join(dest_path, "weights")


def get_builder(
    neuron_config: NxDNeuronConfig,
    model_wrappers: list[NxDGraphBuilder],
    debug: bool = False,
    checkpoint_loader=None,
    compiler_args: str = None,
):
    """Creates a ModelBuilder instance for the given model wrappers.

    This function initializes a ModelBuilder with the specified Neuron configuration and model wrappers.
    It exists to provide a convenient way to create a ModelBuilder instance, and is called by the
    `NxDPreTrainedModel` class every time a model is compiled or loaded.
    The returned ModelBuilder instances are typically discarded after having been used to save memory.

    Args:
        neuron_config (NxDNeuronConfig): The Neuron configuration.
        model_wrappers (list[NxDGraphBuilder]): The model graphs to be added to the builder.
        debug (bool): Whether to enable debug mode.
        checkpoint_loader (callable): A function to load the model's state dictionary and weights.
        compiler_args (str): Compiler arguments to be passed to the builder.
    Returns:
        ModelBuilder: The ModelBuilder instance.
    """
    base_compile_work_dir = os.environ.get("BASE_COMPILE_WORK_DIR", "/tmp/nxd_model/")

    builder = ModelBuilder(
        router=None,
        tp_degree=neuron_config.tp_degree,
        pp_degree=neuron_config.pp_degree,
        ep_degree=neuron_config.ep_degree,
        world_size=neuron_config.world_size,
        start_rank_id=neuron_config.start_rank_id,
        local_ranks_size=neuron_config.local_ranks_size,
        checkpoint_loader=checkpoint_loader,
        compiler_workdir=base_compile_work_dir,
        debug=debug,
        logical_nc_config=neuron_config.logical_nc_config,
        weights_to_skip_layout_optimization=neuron_config.weights_to_skip_layout_optimization,
    )
    for model in model_wrappers:
        builder.add(
            key=model.tag,
            model_instance=model.get_model_instance(),
            example_inputs=model.input_generator(),
            compiler_args=compiler_args,
            bucket_config=model.get_bucket_config(),
            priority_model_idx=model.priority_model_idx,
        )
    return builder


class NxDPreTrainedModel(NeuronPreTrainedModel, ABC):
    _STATE_DICT_MODEL_PREFIX = "model."
    _NEW_STATE_DICT_MODEL_PREFIX = ""
    _FUSED_PREFIX = ""
    COMPILED_MODEL_FILE_NAME = "model.pt"
    CHECKPOINT_DIR = "checkpoint"

    def __init__(
        self,
        config: PretrainedConfig,
        neuron_config: NxDNeuronConfig,
        traced_model: torch.jit.ScriptModule,
        model_wrappers: list[NxDGraphBuilder],
    ):
        self.config = copy.deepcopy(config)
        self.neuron_config = copy.deepcopy(neuron_config)
        # Override torch_dtype in config as it is used by the neuronx_distributed code to cast weights to the correct type
        self.config.torch_dtype = self.neuron_config.torch_dtype
        self._traced_model = traced_model
        self.model_wrappers = model_wrappers  # Required for loading weights

    # NxDPretrainedModel abstract API
    @abstractmethod
    def forward(self, **kwargs):
        """Forward pass for this model."""
        raise NotImplementedError("forward is not implemented")

    @classmethod
    @abstractmethod
    def get_compiler_args(cls, neuron_config) -> str | None:
        """Gets the Neuron compiler arguments to use when compiling this model."""
        return None

    @staticmethod
    def compile(neuron_config, model_wrappers: list[NxDGraphBuilder], compiler_args: str, debug: bool = False):
        builder = get_builder(neuron_config, model_wrappers, debug=debug, compiler_args=compiler_args)
        return builder.trace(initialize_model_weights=False)

    def save(self, dest_path, weight_path: str | None = None):
        if self._traced_model is None:
            raise ValueError("Model has not been compiled or loaded")
        dest_path = normalize_path(dest_path)
        self.config.save_pretrained(dest_path)
        self.neuron_config.save_pretrained(dest_path)
        torch.jit.save(self._traced_model, dest_path + self.COMPILED_MODEL_FILE_NAME)
        if weight_path is not None:
            self.shard_checkpoint(
                src_path=weight_path,
                dest_path=os.path.join(dest_path, self.CHECKPOINT_DIR),
            )

    def shard_checkpoint(self, src_path, dest_path, debug: bool = False):
        shards_path = get_shards_path(dest_path)
        checkpoint_loader = partial(self.checkpoint_loader_fn, src_path, self.config, self.neuron_config)
        sharder = get_builder(
            self.neuron_config,
            self.model_wrappers,
            debug=debug,
            checkpoint_loader=checkpoint_loader,
            compiler_args=self.get_compiler_args(self.neuron_config),
        )
        sharder.shard_checkpoint(serialize_path=shards_path)

        if hlo_utils.NXD_LAYOUT_TRANSFORMATION_OPTIONS in os.environ:
            sharder.transform_weight_layout_with_overriden_option(sharded_checkpoint_dir=shards_path)

    def _load_weights_from_path(self, weights_path):
        weights_path = normalize_path(weights_path)

        """Loads the model weights to the Neuron device."""
        if self._traced_model is None:
            raise ValueError("Model is not loaded")

        start_rank_id = self.neuron_config.start_rank_id
        local_ranks_size = self.neuron_config.local_ranks_size

        logging.info(f"loading models for ranks {start_rank_id}...{start_rank_id + local_ranks_size - 1}")
        weights = []
        shards_path = get_shards_path(weights_path)

        def get_shard_name(rank):
            return os.path.join(shards_path, f"tp{rank}_sharded_checkpoint.safetensors")

        if os.path.exists(get_shard_name(start_rank_id)):
            # If sharded checkpoints exist, load them
            logger.info(f"Loading sharded checkpoint from {shards_path}")
            for rank in range(start_rank_id, start_rank_id + local_ranks_size):
                ckpt = load_file(get_shard_name(rank))
                weights.append(ckpt)
        else:
            logger.info("There are no saved sharded checkpoints.")
            checkpoint_loader = partial(self.checkpoint_loader_fn, weights_path, self.config, self.neuron_config)
            sharder = get_builder(
                self.neuron_config,
                self.model_wrappers,
                debug=False,
                checkpoint_loader=checkpoint_loader,
                compiler_args=self.get_compiler_args(self.neuron_config),
            )
            weights = sharder.shard_checkpoint()
        start_rank_tensor = torch.tensor([start_rank_id], dtype=torch.int32, device="cpu")
        self._traced_model.nxd_model.initialize(weights, start_rank_tensor)

    def load_weights(
        self,
        model_name_or_path: str | Path,
        token: bool | str | None = None,
        cache_dir: str | None = None,
        force_download: bool | None = False,
        local_files_only: bool | None = False,
    ) -> None:
        """Loads the model weights from the given path."""
        if os.path.exists(model_name_or_path):
            # Look first for pre-sharded weights
            checkpoint_path = os.path.join(model_name_or_path, self.CHECKPOINT_DIR)
            if os.path.exists(checkpoint_path):
                self._load_weights_from_path(checkpoint_path)
                return
            # Fall-back to standard model weights, if any
            try:
                self._load_weights_from_path(model_name_or_path)
                return
            except FileNotFoundError:
                logger.info(f"Checkpoint file not found in {model_name_or_path}, trying to load from HuggingFace Hub.")
        if self.neuron_config.checkpoint_id is not None:
            # Fetch weights from the checkpoint
            checkpoint_path = snapshot_download(
                repo_id=self.neuron_config.checkpoint_id,
                revision=self.neuron_config.checkpoint_revision,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                allow_patterns=["*.safetensors*"],
            )
            self._load_weights_from_path(checkpoint_path)
        else:
            raise ValueError(f"Checkpoint file not found under {model_name_or_path}.")

    def checkpoint_loader_fn(self, checkpoint_path, config, neuron_config):
        """This function loads the model's state dictionary and weights from the hf model"""

        model_sd = self.get_state_dict(checkpoint_path, config, neuron_config)
        if neuron_config.torch_dtype != torch.float32:
            for name, param in model_sd.items():
                if torch.is_floating_point(param) and param.dtype is not neuron_config.torch_dtype:
                    logger.debug(f"Converting {name} to {neuron_config.torch_dtype}")
                    model_sd[name] = param.to(neuron_config.torch_dtype)
        return model_sd

    @classmethod
    def get_state_dict(cls, model_path: str, config: PretrainedConfig, neuron_config: NxDNeuronConfig) -> dict:
        """Gets the state dict for this model."""
        model_sd = load_state_dict(model_path)
        param_name_list = list(model_sd.keys())
        for param_name in param_name_list:
            if param_name.startswith(cls._STATE_DICT_MODEL_PREFIX):
                updated_param_name = param_name.replace(
                    cls._STATE_DICT_MODEL_PREFIX, cls._NEW_STATE_DICT_MODEL_PREFIX, 1
                )
                model_sd[updated_param_name] = model_sd[param_name]
                del model_sd[param_name]
        model_sd = cls.convert_hf_to_neuron_state_dict(model_sd, config, neuron_config)
        if getattr(config, "tie_word_embeddings", False):
            cls.update_state_dict_for_tied_weights(model_sd)

        param_name_list = list(model_sd.keys())
        if cls._FUSED_PREFIX != "":
            for param_name in param_name_list:
                model_sd[f"{cls._FUSED_PREFIX}.{param_name}"] = model_sd[param_name]
                del model_sd[param_name]
        return model_sd

    @staticmethod
    def convert_hf_to_neuron_state_dict(
        state_dict: dict, config: PretrainedConfig, neuron_config: NxDNeuronConfig
    ) -> dict:
        """This function should be over-ridden in child classes as needed"""
        return state_dict

    @staticmethod
    def load_hf_model(model_path):
        """Loads the HuggingFace model from the given checkpoint path."""
        return AutoModelForCausalLM.from_pretrained(model_path)

    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """Implement state_dict update for each model class with tied weights"""
        raise NotImplementedError("State-dict update not implemented")

    @property
    def device(self) -> torch.device:
        """
        `torch.device`: The device on which the module is (assuming that all the module parameters are on the same
        device).
        """
        # We dont want HF to move parameters to device
        return torch.device("cpu")

    def reset(self):
        """Resets the model state. Can be implemented by subclasses."""
        pass

    # NeuronPreTrainedModel methods
    def _save_pretrained(self, save_directory: str | Path, **kwargs):
        model_name_or_path = getattr(self.config, "_name_or_path")
        # If the model was exported from a local path, we need to save the checkpoint (not that we also shard it)
        weight_path = model_name_or_path if os.path.isdir(model_name_or_path) else None
        self.save(save_directory, weight_path=weight_path)

    def push_to_hub(
        self,
        save_directory: str,
        repository_id: str,
        private: bool | None = None,
        revision: str | None = None,
        token: bool | str = True,
        endpoint: str | None = None,
    ) -> str:
        api = HfApi(endpoint=endpoint)

        api.create_repo(
            token=token,
            repo_id=repository_id,
            exist_ok=True,
            private=private,
        )
        ignore_patterns = []
        checkpoint_id = self.neuron_config.checkpoint_id
        if checkpoint_id is not None:
            # Avoid uploading checkpoints when the original model is available on the hub
            ignore_patterns = [self.CHECKPOINT_DIR + "/*"]
        api.upload_folder(
            repo_id=repository_id,
            folder_path=save_directory,
            token=token,
            revision=revision,
            ignore_patterns=ignore_patterns,
        )
