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
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union

import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoConfig, AutoModel, GenerationConfig, GenerationMixin
from transformers.modeling_outputs import ModelOutput

from optimum.exporters.tasks import TasksManager

from .....cache.entries.single_model import SingleModelCacheEntry
from .....cache.hub_cache import hub_neuronx_cache
from .....generation import TokenSelector
from .....modeling_decoder import NeuronModelForCausalLM
from .....utils.system import get_available_cores
from .config import HloNeuronConfig


if TYPE_CHECKING:
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from transformers import GenerationConfig, PretrainedConfig
    from transformers.generation import StoppingCriteriaList


logger = logging.getLogger(__name__)


class HloModelForCausalLM(NeuronModelForCausalLM, GenerationMixin):
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

    # The class used to instantiate the neuron model
    # This attribute must be set by the model subclass
    # (e.g. LlamaHloModelForCausalLM, etc.)
    neuron_model_class = None

    CHECKPOINT_DIR = "checkpoint"
    COMPILED_DIR = "compiled"

    def __init__(
        self,
        config: "PretrainedConfig",
        neuron_config: HloNeuronConfig,
        checkpoint_dir: Union[str, Path, TemporaryDirectory],
        compiled_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        generation_config: Optional[GenerationConfig] = None,
    ):
        if self.neuron_model_class is None:
            raise ValueError(
                f"{self.__class__.__name__} must not be instantiated directly: it must be subclassed by a class specifying a neuron_model_class attribute."
            )

        self.config = config
        self.neuron_config = neuron_config

        self.compiled_dir = compiled_dir
        if generation_config is None:
            logger.info("Generation config file not found, using a generation config created from the model config.")
            generation_config = GenerationConfig.from_model_config(config)
        self.generation_config = generation_config
        # Registers the NeuronModelForXXX classes into the transformers AutoModel classes to avoid warnings when creating
        # a pipeline https://github.com/huggingface/transformers/blob/3d3204c025b6b5de013e07dd364208e28b4d9589/src/transformers/pipelines/base.py#L940
        AutoConfig.register("neuron_model", AutoConfig)
        AutoModel.register(AutoConfig, self.__class__)

        # Instantiate neuronx model
        checkpoint_path = checkpoint_dir.name if isinstance(checkpoint_dir, TemporaryDirectory) else checkpoint_dir
        neuronx_model = self.neuron_model_class.from_pretrained(checkpoint_path, neuron_config=neuron_config)

        if compiled_dir is not None:
            # Specify the path where compiled artifacts are stored before conversion
            neuronx_model.load(compiled_dir)

        # When compiling, only create a cache entry if the model comes from the hub
        checkpoint_id = neuron_config.checkpoint_id
        cache_entry = (
            None
            if checkpoint_id is None
            else SingleModelCacheEntry(
                model_id=checkpoint_id, task="text-generation", config=config, neuron_config=neuron_config
            )
        )

        # Export the model using the Optimum Neuron Cache
        with hub_neuronx_cache(entry=cache_entry):
            available_cores = get_available_cores()
            if neuron_config.tp_degree > available_cores:
                raise ValueError(
                    f"The specified tensor parallelization ({neuron_config.tp_degree}) exceeds the number of cores available ({available_cores})."
                )
            neuron_rt_num_cores = os.environ.get("NEURON_RT_NUM_CORES", None)
            # Restrict the number of cores used to allow multiple models on the same host
            os.environ["NEURON_RT_NUM_CORES"] = str(neuron_config.tp_degree)
            # Load the model on neuron cores (if found in cache or compiled directory, the NEFF files
            # will be reloaded instead of compiled)
            neuronx_model.to_neuron()
            if neuron_rt_num_cores is None:
                os.environ.pop("NEURON_RT_NUM_CORES")
            else:
                os.environ["NEURON_RT_NUM_CORES"] = neuron_rt_num_cores

        self.batch_size = neuron_config.batch_size
        self.max_length = neuron_config.sequence_length
        self.continuous_batching = neuron_config.continuous_batching
        self.model = neuronx_model
        # The generate method from GenerationMixin expects the device attribute to be set
        self.device = torch.device("cpu")

    @classmethod
    def _create_checkpoint(
        cls,
        model_id: str,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        **kwargs,
    ) -> TemporaryDirectory:
        # Instantiate the transformers model checkpoint
        model = TasksManager.get_model_from_task(
            task="text-generation",
            model_name_or_path=model_id,
            subfolder=subfolder,
            revision=revision,
            framework="pt",
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
            torch_dtype="auto",
            **kwargs,
        )

        if model.generation_config is not None:
            with warnings.catch_warnings(record=True) as caught_warnings:
                model.generation_config.validate()
            if len(caught_warnings) > 0:
                logger.warning("Invalid generation config: recreating it from model config.")
                model.generation_config = GenerationConfig.from_model_config(model.config)

        # Save the model checkpoint in a temporary directory
        checkpoint_dir = TemporaryDirectory()
        os.chmod(checkpoint_dir.name, 0o775)
        model.save_pretrained(checkpoint_dir.name)
        return checkpoint_dir

    @classmethod
    def export(
        cls,
        model_id: str,
        config: "PretrainedConfig",
        neuron_config: HloNeuronConfig,
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        load_weights: Optional[bool] = True,
        **kwargs,
    ) -> "HloModelForCausalLM":
        if not os.path.isdir("/sys/class/neuron_device/"):
            raise SystemError("Decoder models can only be exported on a neuron platform.")

        if not load_weights:
            warnings.warn(
                "Ignoring the `load_weights` argument set to False since weights are always loaded for these models."
            )

        if os.path.isdir(model_id):
            checkpoint_dir = model_id
        else:
            # Create the local transformers model checkpoint
            checkpoint_dir = cls._create_checkpoint(
                model_id,
                token=token,
                revision=revision,
                **kwargs,
            )

        # Try to reload the generation config (if any)
        generation_config = None
        try:
            generation_config = GenerationConfig.from_pretrained(model_id, revision=revision)
        except OSError:
            pass

        return cls(config, neuron_config, checkpoint_dir, generation_config=generation_config)

    @classmethod
    def _get_neuron_dirs(cls, model_path: Union[str, Path]) -> Tuple[str, str]:
        # The checkpoint is in a subdirectory
        checkpoint_dir = os.path.join(model_path, cls.CHECKPOINT_DIR)
        # So are the compiled artifacts
        compiled_dir = os.path.join(model_path, cls.COMPILED_DIR)
        return checkpoint_dir, compiled_dir

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: "PretrainedConfig",
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        **kwargs,
    ) -> "HloModelForCausalLM":
        model_path = model_id
        if not os.path.isdir(model_id):
            model_path = snapshot_download(model_id, token=token, revision=revision)

        neuron_config = HloNeuronConfig.from_pretrained(model_path)

        checkpoint_dir, compiled_dir = cls._get_neuron_dirs(model_path)
        if not os.path.isdir(checkpoint_dir):
            # Try to recreate checkpoint from neuron config
            checkpoint_id = neuron_config.checkpoint_id
            if checkpoint_id is None:
                raise ValueError("Unable to fetch the neuron model weights files.")
            checkpoint_revision = neuron_config.checkpoint_revision
            checkpoint_dir = cls._create_checkpoint(
                checkpoint_id,
                revision=checkpoint_revision,
                token=token,
                **kwargs,
            )
        assert os.path.isdir(compiled_dir)

        # Try to reload the generation config (if any)
        generation_config = None
        try:
            generation_config = GenerationConfig.from_pretrained(model_id, revision=revision)
        except OSError:
            pass

        return cls(
            config, neuron_config, checkpoint_dir, compiled_dir=compiled_dir, generation_config=generation_config
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        cache_ids: torch.Tensor,
        start_ids: torch.Tensor = None,
        return_dict: bool = True,
    ):
        # Evaluate the output logits, storing the current key and values at the indices specified by cache_ids
        out_logits = self.model.forward(input_ids, cache_ids, start_ids)
        out_logits = out_logits[:, None, :]
        # Since we are using a static cache, we don't need to return past keys and values
        if return_dict:
            return ModelOutput([("logits", out_logits)])
        return (out_logits,)

    def _save_pretrained(self, save_directory: Union[str, Path]):
        self.neuron_config.save_pretrained(save_directory)
        dst_checkpoint_path, dst_compiled_path = self._get_neuron_dirs(save_directory)

        model_name_or_path = getattr(self.config, "_name_or_path")
        if os.path.isdir(model_name_or_path):
            # Model was exported from a local path, so we need to save the checkpoint
            shutil.copytree(model_name_or_path, dst_checkpoint_path, dirs_exist_ok=True)

        # Save or create compiled directory
        if self.compiled_dir is None:
            # The compilation artifacts have never been saved, do it now
            self.model.save(dst_compiled_path)
        else:
            shutil.copytree(self.compiled_dir, dst_compiled_path)
        self.compiled_dir = dst_compiled_path
        self.generation_config.save_pretrained(save_directory)

    def push_to_hub(
        self,
        save_directory: str,
        repository_id: str,
        private: Optional[bool] = None,
        revision: Optional[str] = None,
        token: Union[bool, str] = True,
        endpoint: Optional[str] = None,
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

    # GenerationMixin implementation below

    def can_generate(self) -> bool:
        """
        Called by the transformers code to identify generation models.

        Returns:
            `bool`: `True` if the model can generate sequences, `False` otherwise.
        """
        return True

    def get_start_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_ids: Optional[torch.Tensor] = None,
    ):
        # The start_ids parameter has different meanings:
        # - for continuous (unpadded) batching it corresponds to the sequence id,
        # - for static batching it corresponds to the start of the padded sequence.
        if self.continuous_batching:
            if seq_ids is None:
                seq_ids = torch.arange(input_ids.shape[0])
            else:
                assert seq_ids.shape[0] == input_ids.shape[0]
            return seq_ids
        start_ids = None
        if attention_mask is not None:
            _, start_ids = attention_mask.max(axis=1)
        return start_ids

    def get_cache_ids(self, attention_mask: torch.tensor, prefill: bool):
        cache_n, cache_len = attention_mask.shape
        if self.continuous_batching:
            # Evaluate the inputs that are not masked for each sequence
            input_length = attention_mask.sum(axis=1)
            if not prefill:
                # When decoding, cache_ids contains a single value per sequence
                return (input_length - 1).unsqueeze(1)
            # When prefilling, cache_ids is an increasing range
            cache_ids = torch.zeros_like(attention_mask)
            for i in range(cache_n):
                cur_length = input_length[i]
                cache_ids[i, :cur_length] = torch.arange(cur_length)
            return cache_ids
        # Static batching
        return None if prefill else torch.tensor([cache_len - 1], dtype=torch.int32)

    def prepare_inputs_for_prefill(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_ids: Optional[List[int]] = None,
        sampling_params: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        start_ids = self.get_start_ids(input_ids, attention_mask, seq_ids=seq_ids)
        cache_ids = self.get_cache_ids(attention_mask, prefill=True)
        if self.continuous_batching and torch.any(attention_mask[:, 0] == 0):
            # Inputs are left padded: we need to invert padding as continuous batching requires right-padding
            batch_size, seq_len = input_ids.shape
            input_length = attention_mask.sum(axis=1)
            new_input_ids = torch.zeros_like(input_ids)
            for i in range(batch_size):
                cur_length = input_length[i]
                new_input_ids[i, :cur_length] = input_ids[i, seq_len - cur_length :]
            input_ids = new_input_ids
        return {
            "input_ids": input_ids,
            "cache_ids": cache_ids,
            "start_ids": start_ids,
        }

    def prepare_inputs_for_decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_ids: Optional[List[int]] = None,
        sampling_params: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        start_ids = self.get_start_ids(input_ids, attention_mask, seq_ids=seq_ids)
        cache_ids = self.get_cache_ids(attention_mask, prefill=False)
        # Only pass the last tokens of each sample
        input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "cache_ids": cache_ids,
            "start_ids": start_ids,
        }

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        generation_config: Optional["GenerationConfig"] = None,
        stopping_criteria: Optional["StoppingCriteriaList"] = None,
        **kwargs,
    ) -> torch.LongTensor:
        # The actual generation configuration is a combination of config and parameters
        generation_config = copy.deepcopy(self.generation_config if generation_config is None else generation_config)
        # Extract tokenizer if any (used only for stop strings)
        tokenizer = kwargs.pop("tokenizer", None)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        # Check model kwargs are actually used by either prepare_inputs_for_generation or forward
        self._validate_model_kwargs(model_kwargs)

        # Instantiate a TokenSelector for the specified configuration
        selector = TokenSelector.create(
            input_ids,
            generation_config,
            self,
            self.max_length,
            stopping_criteria=stopping_criteria,
            tokenizer=tokenizer,
        )

        # Verify that the inputs are compatible with the model static input dimensions
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.max_length:
            raise ValueError(
                f"The input sequence length ({sequence_length}) exceeds the model static sequence length ({self.max_length})"
            )
        padded_input_ids = input_ids
        padded_attention_mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask
        if batch_size > self.batch_size:
            raise ValueError(
                f"The specified batch_size ({batch_size}) exceeds the model static batch size ({self.batch_size})"
            )
        elif batch_size < self.batch_size and not self.continuous_batching:
            logger.warning("Inputs will be padded to match the model static batch size. This will increase latency.")
            padding_shape = [self.batch_size - batch_size, sequence_length]
            pad_token_id = generation_config.pad_token_id
            if pad_token_id is None:
                if isinstance(self.config.eos_token_id, list):
                    pad_token_id = self.config.eos_token_id[0]
                else:
                    pad_token_id = self.config.eos_token_id
            padding = torch.full(padding_shape, fill_value=pad_token_id, dtype=torch.int64)
            padded_input_ids = torch.cat([padded_input_ids, padding])
            padding = torch.zeros(padding_shape, dtype=torch.int64)
            padded_attention_mask = torch.cat([padded_attention_mask, padding])

        output_ids = self.generate_tokens(
            padded_input_ids,
            selector,
            batch_size,
            padded_attention_mask,
            **model_kwargs,
        )
        return output_ids[:batch_size, :]

    def generate_tokens(
        self,
        input_ids: torch.LongTensor,
        selector: TokenSelector,
        batch_size: int,
        attention_mask: torch.Tensor,
        **model_kwargs,
    ) -> torch.LongTensor:
        r"""
        Generate tokens using sampling or greedy search.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            selector (`TokenSelector`):
                The object implementing the generation logic based on transformers processors and stopping criterias.
            batch_size (`int`):
                The actual input batch size. Used to avoid generating tokens for padded inputs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model.

        Return:
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens.

        """
        # keep track of which sequences are already finished
        unfinished_sequences = torch.zeros(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        unfinished_sequences[:batch_size] = 1

        # Prefill and obtain the first token
        model_inputs = self.prepare_inputs_for_prefill(input_ids, attention_mask)
        outputs = self(
            **model_inputs,
            return_dict=True,
        )

        # auto-regressive generation
        while True:
            next_token_logits = outputs.logits[:, -1, :]

            next_tokens = selector.select(input_ids, next_token_logits)

            # finished sentences should have their next token be a padding token
            next_tokens = next_tokens * unfinished_sequences + selector.pad_token_id * (1 - unfinished_sequences)

            # update inputs for the next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)

            unfinished_sequences = unfinished_sequences & ~selector.stopping_criteria(input_ids, None)

            if unfinished_sequences.max() == 0:
                break

            # forward pass to get next token
            model_inputs = self.prepare_inputs_for_decode(input_ids, attention_mask)
            outputs = self(
                **model_inputs,
                return_dict=True,
            )

        return input_ids

    @classmethod
    def _get_neuron_config(
        cls,
        checkpoint_id: str,
        checkpoint_revision: str,
        batch_size: int,
        sequence_length: int,
        auto_cast_type: str,
        tensor_parallel_size: int,
        allow_flash_attention: bool = True,
        continuous_batching: bool = None,
        attention_layout: str = "HSB",
        fuse_qkv=True,
    ):
        if continuous_batching is None:
            continuous_batching = batch_size > 1

        return HloNeuronConfig(
            checkpoint_id=checkpoint_id,
            checkpoint_revision=checkpoint_revision,
            batch_size=batch_size,
            sequence_length=sequence_length,
            tp_degree=tensor_parallel_size,
            auto_cast_type=auto_cast_type,
            attention_layout=attention_layout,
            fuse_qkv=fuse_qkv,
            continuous_batching=continuous_batching,
            allow_flash_attention=allow_flash_attention,
        )
