# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
"""Pipelines running different Neuron Accelerators."""

import logging
from typing import Any, Dict, Optional, Union

from transformers import (
    AutoConfig,
    FeatureExtractionPipeline,
    FillMaskPipeline,
    Pipeline,
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    QuestionAnsweringPipeline,
    SequenceFeatureExtractor,
    TextClassificationPipeline,
    TextGenerationPipeline,
    TokenClassificationPipeline,
)
from transformers import pipeline as transformers_pipeline
from transformers.feature_extraction_utils import PreTrainedFeatureExtractor
from transformers.onnx.utils import get_preprocessor

from optimum.modeling_base import OptimizedModel
from optimum.neuron.modeling_base import NeuronBaseModel

from ...modeling import (
    NeuronModelForCausalLM,
    NeuronModelForFeatureExtraction,
    NeuronModelForMaskedLM,
    NeuronModelForQuestionAnswering,
    NeuronModelForSequenceClassification,
    NeuronModelForTokenClassification,
)


logger = logging.getLogger(__name__)


NEURONX_SUPPORTED_TASKS = {
    "feature-extraction": {
        "impl": FeatureExtractionPipeline,
        "class": (NeuronModelForFeatureExtraction,),
        "default": "distilbert-base-cased",
        "type": "text",  # feature extraction is only supported for text at the moment
    },
    "fill-mask": {
        "impl": FillMaskPipeline,
        "class": (NeuronModelForMaskedLM,),
        "default": "bert-base-cased",
        "type": "text",
    },
    "question-answering": {
        "impl": QuestionAnsweringPipeline,
        "class": (NeuronModelForQuestionAnswering,),
        "default": "distilbert-base-cased-distilled-squad",
        "type": "text",
    },
    "text-classification": {
        "impl": TextClassificationPipeline,
        "class": (NeuronModelForSequenceClassification,),
        "default": "distilbert-base-uncased-finetuned-sst-2-english",
        "type": "text",
    },
    "token-classification": {
        "impl": TokenClassificationPipeline,
        "class": (NeuronModelForTokenClassification,),
        "default": "dbmdz/bert-large-cased-finetuned-conll03-english",
        "type": "text",
    },
    "text-generation": {
        "impl": TextGenerationPipeline,
        "class": (NeuronModelForCausalLM,),
        "default": "gpt2",
        "type": "text",
    },
}


def load_pipeline(
    model,
    targeted_task,
    load_tokenizer,
    tokenizer,
    feature_extractor,
    load_feature_extractor,
    supported_tasks=NEURONX_SUPPORTED_TASKS,
    input_shapes={},
    export=False,
    subfolder: str = "",
    token: Optional[Union[bool, str]] = None,
    revision: str = "main",
    compiler_args: Optional[Dict[str, Any]] = {},
    config: AutoConfig = None,
    hub_kwargs: Optional[Dict[str, Any]] = {},
    **kwargs,
):
    # loads default model
    if model is None:
        model_id = supported_tasks[targeted_task]["default"]
        model = supported_tasks[targeted_task]["class"][0].from_pretrained(
            model_id, export=True, **compiler_args, **input_shapes, **hub_kwargs, **kwargs
        )
    # loads model from model id and converts it to neuronx optionally
    elif isinstance(model, str):
        model_id = model
        neuronx_model_class = supported_tasks[targeted_task]["class"][0]
        model = neuronx_model_class.from_pretrained(
            model, export=export, **compiler_args, **input_shapes, **hub_kwargs, **kwargs
        )
    # uses neuron model
    elif isinstance(model, (NeuronBaseModel, NeuronModelForCausalLM)):
        if tokenizer is None and load_tokenizer:
            for preprocessor in model.preprocessors:
                if isinstance(preprocessor, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
                    tokenizer = preprocessor
                    break
            if tokenizer is None:
                raise ValueError(
                    "Could not automatically find a tokenizer for the NeuronBaseModel, you must pass a tokenizer explicitly"
                )
        if feature_extractor is None and load_feature_extractor:
            for preprocessor in model.preprocessors:
                if isinstance(preprocessor, SequenceFeatureExtractor):
                    feature_extractor = preprocessor
                    break
            if feature_extractor is None:
                raise ValueError(
                    "Could not automatically find a feature extractor for the NeuronModel, you must pass a "
                    "feature_extractor explictly"
                )
        model_id = None
    else:
        raise ValueError(
            f"""Model {model} is not supported. Please provide a valid model either as string or NeuronModel.
            You can also provide non model then a default one will be used"""
        )
    return model, model_id, tokenizer, feature_extractor


def pipeline(
    task: str = None,
    model: Optional[Union[str, NeuronBaseModel]] = None,
    tokenizer: Optional[Union[str, PreTrainedTokenizer]] = None,
    feature_extractor: Optional[Union[str, PreTrainedFeatureExtractor]] = None,
    use_fast: bool = True,
    export: bool = False,
    input_shapes: Optional[Dict[str, int]] = {},
    compiler_args: Optional[Dict[str, int]] = {},
    token: Optional[Union[str, bool]] = None,
    revision: Optional[str] = None,
    trust_remote_code: Optional[bool] = None,
    **kwargs,
) -> Pipeline:
    if task not in NEURONX_SUPPORTED_TASKS:
        raise ValueError(
            f"Task {task} is not supported for the optimum neuron pipeline. Supported tasks are {list(NEURONX_SUPPORTED_TASKS.keys())}"
        )

    # copied from transformers.pipelines.__init__.py
    hub_kwargs = {
        "revision": revision,
        "token": token,
        "trust_remote_code": trust_remote_code,
        "_commit_hash": None,
    }

    config = kwargs.get("config", None)
    if config is None:
        if isinstance(model, str):
            config = AutoConfig.from_pretrained(model, _from_pipeline=task, **hub_kwargs, **kwargs)
            hub_kwargs["_commit_hash"] = config._commit_hash
        elif isinstance(model, (PreTrainedModel, OptimizedModel)):
            config = model.config

    if export:
        if hasattr(config, "neuron"):
            raise ValueError("This model has already been exported to Neuron format")
        if not input_shapes:
            input_shapes = {"batch_size": 1, "sequence_length": 128}
            logger.warning(f"No input shapes provided, using default shapes, {input_shapes}")
    else:
        if not hasattr(config, "neuron"):
            raise ValueError("The model must be exported to Neuron format first")
        if input_shapes:
            logger.warning("Input shapes can only be set during export")

    no_feature_extractor_tasks = set()
    no_tokenizer_tasks = set()
    for _task, values in NEURONX_SUPPORTED_TASKS.items():
        if values["type"] == "text":
            no_feature_extractor_tasks.add(_task)
        elif values["type"] in {"image", "video"}:
            no_tokenizer_tasks.add(_task)
        elif values["type"] in {"audio"}:
            no_tokenizer_tasks.add(_task)
        elif values["type"] not in ["multimodal", "audio", "video"]:
            raise ValueError(f"SUPPORTED_TASK {_task} contains invalid type {values['type']}")

    # copied from transformers.pipelines.__init__.py l.609
    if task in no_tokenizer_tasks:
        # These will never require a tokenizer.
        # the model on the other hand might have a tokenizer, but
        # the files could be missing from the hub, instead of failing
        # on such repos, we just force to not load it.
        load_tokenizer = False
    else:
        load_tokenizer = True

    if task in no_feature_extractor_tasks:
        load_feature_extractor = False
    else:
        load_feature_extractor = True

    model, model_id, tokenizer, feature_extractor = load_pipeline(
        model,
        task,
        load_tokenizer,
        tokenizer,
        feature_extractor,
        load_feature_extractor,
        export=export,
        input_shapes=input_shapes,
        compiler_args=compiler_args,
        supported_tasks=NEURONX_SUPPORTED_TASKS,
        config=config,
        hub_kwargs=hub_kwargs,
        token=token,
    )

    if tokenizer is None and load_tokenizer:
        tokenizer = get_preprocessor(model_id)
    if feature_extractor is None and load_feature_extractor:
        feature_extractor = get_preprocessor(model_id)

    # If we don't specify a batch_size, the pipeline will assume batch_size 1
    # and it will process the inputs one by one instead of processing them in parallel
    batch_size = 1
    for attr in ["batch_size", "static_batch_size"]:
        if attr in model.config.neuron:
            batch_size = model.config.neuron[attr]
    if batch_size > 1 and tokenizer is not None and tokenizer.pad_token_id is None:
        # The pipeline needs a pad token to be able to batch
        tokenizer.pad_token_id = model.config.eos_token_id

    return transformers_pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        use_fast=use_fast,
        batch_size=batch_size,
        **kwargs,
    )
