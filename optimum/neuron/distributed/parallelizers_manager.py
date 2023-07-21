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

import importlib
from typing import Dict, List, Type, Union

from transformers import PreTrainedModel

from .base import Parallelizer


_PARALLELIZER_CLASSES_MODULE_NAMES = ["encoder_models", "decoder_models"]


def parallelizer_classes_resolver(
    model_type_to_parallelizer_class_name: Dict[str, str]
) -> Dict[str, Type[Parallelizer]]:
    modules = []
    for module_name in _PARALLELIZER_CLASSES_MODULE_NAMES:
        package_name = __name__.rsplit(".", maxsplit=1)[0]
        full_module_name = f"{package_name}.{module_name}"
        modules.append(importlib.import_module(full_module_name))

    resolved = {}
    for model_type, parallelizer_class_name in model_type_to_parallelizer_class_name.items():
        found = False
        for mod in modules:
            cls = getattr(mod, parallelizer_class_name, None)
            if cls is not None:
                found = True
                resolved[model_type] = cls
                break
        if not found:
            raise ValueError(f"Could not resolve the parallelizer class name {parallelizer_class_name}.")
    return resolved


class ParallelizersManager:
    _MODEL_TYPE_TO_PARALLEL_MODEL_CLASS = parallelizer_classes_resolver(
        {
            "bert": "BertParallelizer",
            "roberta": "RobertaParallelizer",
            "gpt_neo": "GPTNeoParallelizer",
            "llama": "LlamaParallelizer",
        }
    )

    @classmethod
    def get_supported_model_types(cls) -> List[str]:
        return list(cls._MODEL_TYPE_TO_PARALLEL_MODEL_CLASS.keys())

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
        if not cls.is_model_supported(model_type_or_model):
            supported_models = ", ".join(cls._MODEL_TYPE_TO_PARALLEL_MODEL_CLASS.keys())
            raise NotImplementedError(
                f"{model_type} is not supported for parallelization, supported models: {supported_models}"
            )
        return cls._MODEL_TYPE_TO_PARALLEL_MODEL_CLASS[model_type]
