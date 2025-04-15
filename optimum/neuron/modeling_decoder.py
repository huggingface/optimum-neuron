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
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import torch
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, GenerationConfig
from transformers.file_utils import add_start_docstrings
from transformers.generation import GenerationMixin

from optimum.exporters.tasks import TasksManager

from ..exporters.neuron.model_configs import *  # noqa: F403
from .configuration_utils import NeuronConfig
from .modeling_base import NeuronModel
from .models.inference.hlo.backend.config import HloNeuronConfig
from .utils.system import get_available_cores


if TYPE_CHECKING:
    from pathlib import Path

    from transformers import GenerationConfig, PretrainedConfig
    from transformers.generation import StoppingCriteriaList


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
        stopping_criteria (`Optional[transformers.generation.StoppingCriteriaList], defaults to `None`):
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


@add_start_docstrings(
    r"""
    Neuron model with a causal language modeling head for inference on Neuron devices.
    """,
    NEURON_CAUSALLM_MODEL_START_DOCSTRING,
)
class NeuronModelForCausalLM(NeuronModel, GenerationMixin):
    auto_model_class = AutoModelForCausalLM
    main_input_name = "input_ids"
    preprocessors = []  # Required by optimum OptimizedModel

    @staticmethod
    def get_neuron_config(
        model_name_or_path: Union[str, Path],
        config: "PretrainedConfig",
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        batch_size: Optional[int] = None,
        sequence_length: Optional[int] = None,
        tensor_parallel_size: Optional[int] = None,
        auto_cast_type: Optional[str] = None,
    ):
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

        exporter = TasksManager.get_exporter_config_constructor(
            model_type=config.model_type, exporter="neuron", task="text-generation", library_name="transformers"
        )()
        return exporter.get_neuron_config(
            checkpoint_id=checkpoint_id,
            checkpoint_revision=checkpoint_revision,
            batch_size=batch_size,
            sequence_length=sequence_length,
            auto_cast_type=auto_cast_type,
            tensor_parallel_size=tensor_parallel_size,
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
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        batch_size: Optional[int] = None,
        sequence_length: Optional[int] = None,
        num_cores: Optional[int] = None,
        auto_cast_type: Optional[str] = "fp32",
        task: Optional[str] = "text-generation",
        **kwargs,
    ) -> "NeuronModelForCausalLM":
        from .models.inference.hlo.backend.modeling_decoder import HloModelForCausalLM

        # Get the neuron config
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

        if task != "text-generation":
            raise ValueError(
                f"Task {task} is not supported for causal language models. Please use another base model."
            )

        return HloModelForCausalLM._export(
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
        model_id: Union[str, "Path"],
        config: "PretrainedConfig",
        token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        **kwargs,
    ) -> "NeuronModelForCausalLM":
        neuron_config = NeuronConfig.from_pretrained(model_id, token=token, revision=revision)
        if isinstance(neuron_config, HloNeuronConfig):
            from .models.inference.hlo.backend.modeling_decoder import HloModelForCausalLM

            return HloModelForCausalLM._from_pretrained(
                model_id, config, neuron_config, token=token, revision=revision, **kwargs
            )
        raise ValueError(
            "The specified directory does not contain a neuron model."
            "Please convert your model to neuron format by passing export=True."
        )

    def can_generate(self) -> bool:
        """Returns True to validate the check made in `GenerationMixin.generate()`."""
        return True

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
        attention_mask: Optional[torch.Tensor] = None,
        generation_config: Optional["GenerationConfig"] = None,
        stopping_criteria: Optional["StoppingCriteriaList"] = None,
        **kwargs,
    ) -> torch.LongTensor:
        raise NotImplementedError
