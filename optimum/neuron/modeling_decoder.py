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
from abc import ABC, abstractmethod
from pathlib import Path

import torch
from huggingface_hub import HfApi
from transformers import GenerationConfig, PretrainedConfig
from transformers.file_utils import add_start_docstrings
from transformers.generation import StoppingCriteriaList

from .configuration_utils import NeuronConfig
from .modeling_base import NeuronModel
from .models.auto_model import get_neuron_model_class
from .utils.system import get_available_cores


logger = logging.getLogger(__name__)


NEURON_CAUSALLM_MODEL_START_DOCSTRING = r"""
    This model inherits from [`~neuron.NeuronModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving)
"""

NEURON_CAUSALLM_MODEL_GENERATE_DOCSTRING = r"""
    A streamlined generate() method overriding the transformers.GenerationMixin.generate() method.

    This method uses the same logits processors/warpers and stopping criterias as the transformers library
    `generate()` method but restricts the generation to greedy search and sampling.

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
    >>> from transformers import {processor_class}
    >>> from optimum.neuron import {model_class}
    >>> import torch

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True)

    >>> inputs = tokenizer("My favorite moment of the day is", return_tensors="pt")

    >>> gen_tokens = model.generate(**inputs, do_sample=True, temperature=0.9, min_length=20, max_length=20)
    >>> tokenizer.batch_decode(gen_tokens)  # doctest: +IGNORE_RESULT
    ```
"""


def get_neuron_causal_lm_model_class(config: PretrainedConfig):
    cls = get_neuron_model_class(config.model_type, task="text-generation", mode="inference")
    if not issubclass(cls, NeuronModelForCausalLM):
        raise ValueError(f"Model {config.model_type} is not a causal language model. Please use another base model.")
    return cls


@add_start_docstrings(
    r"""
    Neuron model with a causal language modeling head for inference on Neuron devices.
    """,
    NEURON_CAUSALLM_MODEL_START_DOCSTRING,
)
class NeuronModelForCausalLM(NeuronModel, ABC):
    preprocessors = []  # Required by optimum OptimizedModel

    @classmethod
    def get_neuron_config(
        cls,
        model_name_or_path: str | Path,
        config: "PretrainedConfig",
        token: bool | str | None = None,
        revision: str | None = None,
        batch_size: int | None = None,
        sequence_length: int | None = None,
        tensor_parallel_size: int | None = None,
        auto_cast_type: str | None = None,
    ):
        """
        Get the Neuron configuration for the target model class.

        Can be called either from the NeuronModelForCausalLM class or from a specific model class.
        In the first case, the actual model class will be deduced from the model configuration.

        Args:
            neuron_model_class (`type`):
                The class of the target neuron model.
            model_name_or_path (`str` or `Path`):
                The model name or path to the model directory.
            config (`PretrainedConfig`):
                The model configuration.
            token (`str`, *optional*):
                The token to use for authentication with the Hugging Face Hub.
            revision (`str`, *optional*):
                The revision of the model to use. If not specified, the latest revision will be used.
            batch_size (`int`, *optional*):
                The batch size to use for inference. If not specified, defaults to 1.
            sequence_length (`int`, *optional*):
                The sequence length to use for inference. If not specified, defaults to the model's maximum sequence length.
            tensor_parallel_size (`int`, *optional*):
                The number of cores to use for tensor parallelism. If not specified, all available cores will be used.
            auto_cast_type (`str`, *optional*):
                The data type to use for automatic casting. If not specified, defaults to the model's data type.
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
        if tensor_parallel_size is None:
            # Use all available cores
            tensor_parallel_size = get_available_cores()
        if auto_cast_type is None:
            auto_cast_type = "fp32"
            if config.torch_dtype == "float16":
                auto_cast_type = "fp16"
            elif config.torch_dtype == "bfloat16":
                auto_cast_type = "bf16"

        if type(cls) is NeuronModelForCausalLM:
            # Instantiation through the abstract class: find the correct model class
            cls = get_neuron_causal_lm_model_class(config)

        # Call the _get_neuron_config method of the specific model class
        return cls._get_neuron_config(
            checkpoint_id=checkpoint_id,
            checkpoint_revision=checkpoint_revision,
            batch_size=batch_size,
            sequence_length=sequence_length,
            tensor_parallel_size=tensor_parallel_size,
            auto_cast_type=auto_cast_type,
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
        batch_size: int | None = None,
        sequence_length: int | None = None,
        num_cores: int | None = None,
        auto_cast_type: str | None = "bf16",
        task: str | None = "text-generation",
        **kwargs,
    ) -> "NeuronModelForCausalLM":
        """Implementation of the `optimum.OptimizedModel._export` method.

        It accepts simplified parameters and converts them to a NeuronConfig object.
        This NeuronConfig object is then passed to the `export` method that is in charge
        of exporting the model to Neuron format.

        Args:
            model_id (`str`):
                The model ID or path to the model directory.
            config (`PretrainedConfig`):
                The model configuration.
            token (`str`, *optional*):
                The token to use for authentication with the Hugging Face Hub.
            revision (`str`, *optional*):
                The revision of the model to use. If not specified, the latest revision will be used.
            batch_size (`int`, *optional*):
                The batch size to use for inference. If not specified, defaults to 1.
            sequence_length (`int`, *optional*):
                The sequence length to use for inference. If not specified, defaults to the model's maximum sequence length.
            num_cores (`int`, *optional*):
                The number of cores to use for tensor parallelism. If not specified, all available cores will be used.
            auto_cast_type (`str`, *optional*):
                The data type to use for automatic casting. If not specified, defaults to the model's data type.
            task (`str`, *optional*):
                The task for which the model is being exported. Defaults to "text-generation".

        Returns:
            `NeuronModelForCausalLM`: The exported Neuron model.
        """
        if task != "text-generation":
            raise ValueError(
                f"Task {task} is not supported for causal language models. Please use another base model."
            )
        if cls is NeuronModelForCausalLM:
            # Instantiation through the abstract class: find the correct model class
            cls = get_neuron_causal_lm_model_class(config)

        # Create the neuron config for the specified parameters
        neuron_config = cls.get_neuron_config(
            model_id,
            config,
            token=token,
            revision=revision,
            batch_size=batch_size,
            sequence_length=sequence_length,
            tensor_parallel_size=num_cores,
            auto_cast_type=auto_cast_type,
        )

        return cls.export(
            model_id,
            config,
            neuron_config,
            token=token,
            revision=revision,
            **kwargs,
        )

    @classmethod
    def _from_pretrained(
        cls,
        model_id: "str | Path",
        config: "PretrainedConfig",
        **kwargs,
    ) -> "NeuronModelForCausalLM":
        # Find the correct model class
        cls = get_neuron_causal_lm_model_class(config)
        return cls._from_pretrained(model_id, config, **kwargs)

    @add_start_docstrings(
        NEURON_CAUSALLM_MODEL_GENERATE_DOCSTRING
        + TEXT_GENERATION_EXAMPLE.format(
            processor_class="AutoTokenizer",
            model_class="NeuronModelForCausalLM",
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

    @classmethod
    @abstractmethod
    def _get_neuron_config(
        cls,
        checkpoint_id: str,
        checkpoint_revision: str,
        batch_size: int,
        sequence_length: int,
        tensor_parallel_size: int,
        auto_cast_type: str,
    ):
        raise NotImplementedError("The `get_neuron_config` method must be implemented in the subclass.")

    @classmethod
    def export(
        cls,
        model_id: str,
        config: "PretrainedConfig",
        neuron_config: "NeuronConfig",
        token: bool | str | None = None,
        revision: str | None = None,
        load_weights: bool | None = True,
        **kwargs,
    ) -> "NeuronModelForCausalLM":
        """Export the model to Neuron format.

        This method must be implemented by the subclass. It should handle the export of the model to Neuron format.
        Args:
            model_id (`str`):
                The model ID or path to the model directory.
            config (`PretrainedConfig`):
                The model configuration.
            neuron_config (`NeuronConfig`):
                The Neuron configuration for the model.
            token (`str`, *optional*):
                The token to use for authentication with the Hugging Face Hub.
            revision (`str`, *optional*):
                The revision of the model to use. If not specified, the latest revision will be used.
            load_weights (`bool`, *optional*, defaults to `True`):
                Whether to load the model weights after exporting. If `False`, the model will be exported without weights.
        Returns:
            `NeuronModelForCausalLM`: The exported Neuron model.
        """
        raise NotImplementedError(
            "The `export` method must be implemented in the subclass. It should handle the export of the model to Neuron format."
        )
