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
"""Factory class mapping model architectures to their Parallelizer class."""

from typing import Type, Union

from transformers import PreTrainedModel

from optimum.neuron.utils.training_utils import is_model_officially_supported

from .base import Parallelizer
from .encoder_models import BertParallelizer


class ParallelizersManager:
    _MODEL_TYPE_TO_PARALLEL_MODEL_CLASS = {
        "bert": BertParallelizer,
    }

    @classmethod
    def _get_model_type(cls, model_type_or_model: Union[str, PreTrainedModel]) -> str:
        if isinstance(model_type_or_model, PreTrainedModel):
            model_type = model_type_or_model.config.model_type
        else:
            model_type = model_type_or_model
        return model_type

    @classmethod
    def is_model_supported(cls, model_type_or_model: Union[str, PreTrainedModel]) -> bool:
        model_type = cls._get_model_type(model_type_or_model)
        return model_type in cls._MODEL_TYPE_TO_PARALLEL_MODEL_CLASS

    @classmethod
    def parallelizer_for_model(cls, model_type_or_model: Union[str, PreTrainedModel]) -> Type[Parallelizer]:
        model_type = cls._get_model_type(model_type_or_model)
        if not is_model_officially_supported(model_type_or_model):
            supported_models = ", ".join(cls._MODEL_TYPE_TO_PARALLEL_MODEL_CLASS.keys())
            raise NotImplementedError(
                f"{model_type} is not supported for parallelization, supported model: {supported_models}"
            )
        return cls._MODEL_TYPE_TO_PARALLEL_MODEL_CLASS[model_type]
