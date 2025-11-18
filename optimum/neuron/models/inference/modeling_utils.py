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
"""Base classes for neuron model custom modeling for inference."""

import inspect
import logging
import os
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from huggingface_hub import HfApi
from transformers import AutoConfig, GenerationConfig, PretrainedConfig
from transformers.file_utils import add_start_docstrings
from transformers.generation import StoppingCriteriaList

from ...configuration_utils import NeuronConfig
from ...modeling_base import NeuronModel
from ...utils.argument_utils import DTYPE_MAPPER
from ...utils.instance import get_default_compilation_target, normalize_instance_type
from ...utils.system import get_available_cores


logger = logging.getLogger(__name__)


class NeuronPreTrainedModel(NeuronModel, ABC):
    task: str | None = None

    @classmethod
    def _get_neuron_model_class(cls, config: PretrainedConfig):
        """Internal helper to get the actual Neuron model class for the task

        Each subclass of NeuronPreTrainedModel must specify the task is supports.
        """
        if cls.task is None:
            raise SystemError("f{cls} has no associated task. Please specify it in the class declaration.")
        return get_neuron_model_class(config.model_type, task=cls.task, mode="inference")

    @classmethod
    def get_neuron_config(
        cls,
        model_name_or_path: str | Path,
        config: PretrainedConfig | None = None,
        token: bool | str | None = None,
        revision: str | None = None,
        instance_type: str | None = None,
        batch_size: int | None = None,
        sequence_length: int | None = None,
        tensor_parallel_size: int | None = None,
    ) -> NeuronConfig:
        """
        Get the Neuron configuration for the target model class.

        Can be called either from an auto class or from a specific model class.
        In the first case, the actual model class will be deduced from the model configuration.

        Args:
            neuron_model_class (`type`):
                The class of the target neuron model.
            model_name_or_path (`str` or `Path`):
                The model name or path to the model directory.
            config (`PretrainedConfig`, *optional*):
                The model configuration.
            token (`str`, *optional*):
                The token to use for authentication with the Hugging Face Hub.
            revision (`str`, *optional*):
                The revision of the model to use. If not specified, the latest revision will be used.
            instance_type (`str`, *optional*):
                The target Neuron instance type on which the compiled model will be run. If not specified
            batch_size (`int`, *optional*):
                The batch size to use for inference. If not specified, defaults to 1.
            sequence_length (`int`, *optional*):
                The sequence length to use for inference. If not specified, defaults to the model's maximum sequence length.
            tensor_parallel_size (`int`, *optional*):
                The number of cores to use for tensor parallelism. If not specified, all available cores will be used.
        Returns:
            `NeuronConfig`: The Neuron configuration for the model.
        """
        if os.path.isdir(model_name_or_path):
            checkpoint_id = None
            checkpoint_revision = None
        else:
            checkpoint_id = model_name_or_path
            # Get the exact checkpoint revision (SHA1)
            api = HfApi(token=token)
            model_info = api.repo_info(model_name_or_path, revision=revision)
            checkpoint_revision = model_info.sha
        if config is None:
            config = AutoConfig.from_pretrained(
                model_name_or_path,
                revision=checkpoint_revision,
                use_auth_token=token,
            ).get_text_config()

        if instance_type is None:
            instance_type = get_default_compilation_target()
        else:
            instance_type = normalize_instance_type(instance_type)
        if batch_size is None:
            batch_size = 1
        # If the sequence_length was not specified, deduce it from the model configuration
        if sequence_length is None:
            if hasattr(config, "n_positions"):
                sequence_length = config.n_positions
            elif hasattr(config, "max_position_embeddings"):
                sequence_length = config.max_position_embeddings
            else:
                sequence_length = 1024
            # Restrict default sequence length, as some models can have very large position embeddings
            sequence_length = min(sequence_length, 4096)
        if tensor_parallel_size is None:
            # Use all available cores
            tensor_parallel_size = get_available_cores()

        if inspect.isabstract(cls):
            # Instantiation through an abstract class: find the correct model class
            cls = cls._get_neuron_model_class(config)

        # Call the _get_neuron_config method of the specific model class
        return cls._get_neuron_config(
            checkpoint_id=checkpoint_id,
            checkpoint_revision=checkpoint_revision,
            instance_type=instance_type,
            batch_size=batch_size,
            sequence_length=sequence_length,
            tensor_parallel_size=tensor_parallel_size,
            dtype=DTYPE_MAPPER.pt(config.torch_dtype),
        )

    @classmethod
    def export(
        cls,
        model_id: str,
        neuron_config: NeuronConfig,
        config: PretrainedConfig | None = None,
        token: bool | str | None = None,
        revision: str | None = None,
        load_weights: bool | None = False,
        **kwargs,
    ) -> "NeuronPreTrainedModel":
        """Export a Decoder model to Neuron.

        It requires a NeuronConfig object that can be created for instance by the get_neuron_config class method.

        Args:
            model_id (`str`):
                The model ID or path to the model directory.
            neuron_config (`NxDNeuronConfig`):
                The Neuron configuration for the model.
            config (`PretrainedConfig`, *optional*):
                The model configuration.
            token (`str`, *optional*):
                The token to use for authentication with the Hugging Face Hub.
            revision (`str`, *optional*):
                The revision of the model to use. If not specified, the latest revision will be used.
            load_weights (`bool`, *optional*, defaults to `False`):
                Whether to load the model weights after exporting. If `False`, the model will be exported

        Returns:
            `NeuronPreTrainedModel`: The exported Neuron model.
        """
        if config is None:
            config = AutoConfig.from_pretrained(
                model_id,
                revision=revision,
                use_auth_token=token,
            ).get_text_config()
        if inspect.isabstract(cls):
            # Instantiation through an abstract class: find the correct model class
            cls = cls._get_neuron_model_class(config)

        return cls._export(
            model_id,
            config,
            neuron_config,
            token=token,
            revision=revision,
            load_weights=load_weights,
            **kwargs,
        )

    @classmethod
    def from_pretrained(
        cls,
        model_id: "str | Path",
        **kwargs,
    ) -> "NeuronPreTrainedModel":
        config = AutoConfig.from_pretrained(model_id, **kwargs)
        if inspect.isabstract(cls):
            # Instantiation through an abstract class: find the correct model class
            cls = cls._get_neuron_model_class(config)
        return cls._from_pretrained(model_id, config, **kwargs)

    @classmethod
    @abstractmethod
    def _from_pretrained(
        cls,
        model_id: "str | Path",
        config: "PretrainedConfig",
        **kwargs,
    ) -> "NeuronPreTrainedModel":
        raise NotImplementedError("The _from_pretrained method must be implemented in the subclass.")

    @classmethod
    @abstractmethod
    def _get_neuron_config(
        cls,
        checkpoint_id: str,
        checkpoint_revision: str,
        instance_type: str,
        batch_size: int,
        sequence_length: int,
        tensor_parallel_size: int,
        dtype: torch.dtype,
    ):
        raise NotImplementedError("The `_get_neuron_config` method must be implemented in the subclass.")

    @classmethod
    @abstractmethod
    def _export(
        cls,
        model_id: str,
        config: "PretrainedConfig",
        neuron_config: "NeuronConfig",
        token: bool | str | None = None,
        revision: str | None = None,
        cache_dir: str | None = None,
        force_download: bool | None = False,
        local_files_only: bool | None = False,
        trust_remote_code: bool | None = False,
        load_weights: bool | None = False,
        **kwargs,
    ) -> "NeuronPreTrainedModel":
        """Export the model to Neuron format.

        This method must be implemented by the subclass. It should handle the export of the model to Neuron format.
        Args:
            model_id (`str`):
                The model ID or path to the model directory.
            neuron_config (`NeuronConfig`):
                The Neuron configuration for the model.
            config (`PretrainedConfig`, *optional*):
                The model configuration.
            token (`str`, *optional*):
                The token to use for authentication with the Hugging Face Hub.
            revision (`str`, *optional*):
                The revision of the model to use. If not specified, the latest revision will be used.
            load_weights (`bool`, *optional*, defaults to `False`):
                Whether to load the model weights after exporting. If `False`, the model will be exported without weights.
        Returns:
            `NeuronPreTrainedModel`: The exported Neuron model.
        """
        raise NotImplementedError(
            "The `_export` method must be implemented in the subclass. It should handle the export of the model to Neuron format."
        )

    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Save a Neuron model and its configuration file to a directory, so that it can be re-loaded using the
        [`from_pretrained`] class method.

        Args:
            save_directory (`str`):
                The directory where the model and its configuration files will be saved.
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        # Save the Neuron configuration
        neuron_config = self.neuron_config
        neuron_config.save_pretrained(save_directory)
        # Save the model configuration
        self.config.save_pretrained(save_directory)
        logger.info(f"Model and configuration files saved in {save_directory}")
        self._save_pretrained(save_directory, **kwargs)

    @abstractmethod
    def _save_pretrained(self, save_directory: str, **kwargs):
        """
        Save the model to a directory. This method should be implemented by the subclass.

        Args:
            save_directory (`str`):
                The directory where the model weights will be saved.
        """
        raise NotImplementedError("The `_save_pretrained` method must be implemented in the subclass.")

    @abstractmethod
    def push_to_hub(
        self,
        save_directory: str,
        repository_id: str,
        private: bool | None = None,
        revision: str | None = None,
        token: bool | str = True,
        endpoint: str | None = None,
    ) -> str:
        raise NotImplementedError("The `push_to_hub` method must be implemented in the subclass.")


NEURON_CAUSALLM_MODEL_START_DOCSTRING = r"""
    This model inherits from [`~neuron.NeuronModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving)
"""

NEURON_CAUSALLM_MODEL_GENERATE_DOCSTRING = r"""
    A streamlined generate() method overriding the transformers.GenerationMixin.generate() method.

    This method only supports greedy search and sampling.

    It does not support transformers `generate()` advanced options.

    Please refer to https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationMixin.generate
    for details on generation configuration.

    Parameters:
        input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`):
            The sequence used as a prompt for the generation.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices.
        generation_config (`~transformers.generation.GenerationConfig`, *optional*):
            The generation configuration to be used as base parametrization for the generation call. `**kwargs`
            passed to generate matching the attributes of `generation_config` will override them. If
            `generation_config` is not provided, default will be used, which had the following loading
            priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
            configuration. Please note that unspecified parameters will inherit [`~transformers.generation.GenerationConfig`]'s
            default values, whose documentation should be checked to parameterize generation.
        stopping_criteria (`transformers.generation.StoppingCriteriaList | None`, defaults to `None`):
            Custom stopping criteria that complement the default stopping criteria built from arguments and a
            generation config.

    Returns:
        `torch.Tensor`: A  `torch.FloatTensor`.
"""

TEXT_GENERATION_EXAMPLE = r"""
    Example of text generation:

    ```python
    >>> from transformers import AutoTokenizer
    >>> from optimum.neuron import NeuronModelForCausalLM
    >>> import torch

    >>> tokenizer = AutoTokenizer.from_pretrained("{checkpoint}")
    >>> neuron_config = NeuronModelForCausalLM.get_neuron_config("{checkpoint}")
    >>> model = NeuronModelForCausalLM.export("{checkpoint}", neuron_config, load_weights=True)

    >>> inputs = tokenizer("My favorite moment of the day is", return_tensors="pt")

    >>> gen_tokens = model.generate(**inputs, do_sample=True, temperature=0.9, min_length=20, max_length=20)
    >>> tokenizer.batch_decode(gen_tokens)  # doctest: +IGNORE_RESULT
    ```
"""


@add_start_docstrings(
    r"""
    Neuron model with a causal language modeling head for inference on Neuron devices.
    """,
    NEURON_CAUSALLM_MODEL_START_DOCSTRING,
)
class NeuronModelForCausalLM(NeuronPreTrainedModel):
    task = "text-generation"

    @add_start_docstrings(
        NEURON_CAUSALLM_MODEL_GENERATE_DOCSTRING
        + TEXT_GENERATION_EXAMPLE.format(
            checkpoint="Qwen/Qwen2.5-0.5B-Instruct",
        )
    )
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        generation_config: "GenerationConfig | None" = None,
        stopping_criteria: "StoppingCriteriaList | None" = None,
        **kwargs,
    ) -> torch.LongTensor:
        raise NotImplementedError

from ..auto_model import get_neuron_model_class