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
import warnings
from typing import Any

from transformers import (
    AudioClassificationPipeline,
    AutoConfig,
    AutomaticSpeechRecognitionPipeline,
    BaseImageProcessor,
    FillMaskPipeline,
    ImageClassificationPipeline,
    ImageSegmentationPipeline,
    ObjectDetectionPipeline,
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

from optimum.neuron.modeling_base import NeuronModel
from optimum.neuron.pipelines.transformers.sentence_transformers import (
    FeatureExtractionPipeline,
    is_sentence_transformer_model,
)

from ...configuration_utils import NeuronConfig
from ...modeling import (
    NeuronModelForAudioClassification,
    NeuronModelForCTC,
    NeuronModelForFeatureExtraction,
    NeuronModelForImageClassification,
    NeuronModelForMaskedLM,
    NeuronModelForQuestionAnswering,
    NeuronModelForSemanticSegmentation,
    NeuronModelForSequenceClassification,
    NeuronModelForTokenClassification,
)
from ...models.inference.modeling_utils import NeuronModelForCausalLM
from ...modeling_sentence_transformers import NeuronSentenceTransformers


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
        "default": "Qwen/Qwen2.5-0.5B-Instruct",
        "type": "text",
    },
    "image-classification": {
        "impl": ImageClassificationPipeline,
        "class": (NeuronModelForImageClassification,),
        "default": "microsoft/beit-base-patch16-224-pt22k-ft22k",
        "type": "image",
    },
    "image-segmentation": {
        "impl": ImageSegmentationPipeline,
        "class": (NeuronModelForSemanticSegmentation,),
        "default": "apple/deeplabv3-mobilevit-small",
        "type": "image",
    },
    "object-detection": {
        "impl": ObjectDetectionPipeline,
        "class": (NeuronModelForSemanticSegmentation,),
        "default": "apple/deeplabv3-mobilevit-small",
        "type": "image",
    },
    "automatic-speech-recognition": {
        "impl": AutomaticSpeechRecognitionPipeline,
        "class": (NeuronModelForCTC,),
        "default": "facebook/wav2vec2-large-960h-lv60-self",
        "type": "audio",
    },
    "audio-classification": {
        "impl": AudioClassificationPipeline,
        "class": (NeuronModelForAudioClassification,),
        "default": "facebook/wav2vec2-large-960h-lv60-self",
        "type": "audio",
    },
}


def check_model_type(self, supported_models: list[str, dict]):
    """
    Dummy function to avoid the error logs raised by https://github.com/huggingface/transformers/blob/v4.45.2/src/transformers/pipelines/base.py#L1091
    """
    pass


def load_pipeline(
    model,
    targeted_task,
    tokenizer: str | PreTrainedTokenizer | PreTrainedTokenizerFast | None,
    feature_extractor: "str | PreTrainedFeatureExtractor | None",
    image_processor: str | BaseImageProcessor | None,
    load_tokenizer: bool,
    load_feature_extractor: bool,
    load_image_processor: bool,
    supported_tasks=NEURONX_SUPPORTED_TASKS,
    input_shapes={},
    export=False,
    subfolder: str = "",
    token: bool | str | None = None,
    revision: str = "main",
    compiler_args: dict[str, Any] | None = {},
    hub_kwargs: dict[str, Any] | None = {},
    **export_kwargs,
):
    # loads model from model id and converts it to neuronx optionally
    if isinstance(model, str):
        model_id = model
        neuronx_model_class = supported_tasks[targeted_task]["class"][0]
        # Try to determine the correct feature extraction class to use.
        if targeted_task == "feature-extraction" and is_sentence_transformer_model(
            model, token=token, revision=revision
        ):
            logger.info("Using Sentence Transformers compatible Feature extraction pipeline")
            neuronx_model_class = NeuronSentenceTransformers

        if issubclass(neuronx_model_class, NeuronModelForCausalLM):
            if export:
                neuron_config = neuronx_model_class.get_neuron_config(model, **export_kwargs)
                model = neuronx_model_class.export(model, neuron_config=neuron_config, load_weights=True, **hub_kwargs)
            else:
                model = neuronx_model_class.from_pretrained(model, **hub_kwargs)
        else:
            model = neuronx_model_class.from_pretrained(
                model, export=export, **compiler_args, **input_shapes, **hub_kwargs
            )
    # uses neuron model
    elif isinstance(model, NeuronModel):
        if tokenizer is None and load_tokenizer:
            for preprocessor in model.preprocessors:
                if isinstance(preprocessor, (PreTrainedTokenizer, PreTrainedTokenizerFast)):
                    tokenizer = preprocessor
                    break
            if tokenizer is None:
                raise ValueError(
                    "Could not automatically find a tokenizer for the NeuronModel, you must pass a tokenizer explicitly"
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
        if image_processor is None and load_image_processor:
            for preprocessor in model.preprocessors:
                if isinstance(preprocessor, BaseImageProcessor):
                    image_processor = preprocessor
                    break
            if image_processor is None:
                raise ValueError(
                    "Could not automatically find an image_processor for the NeuronModel, you must pass an image processor explicitly"
                )
        model_id = None
    else:
        raise ValueError(
            f"""Model {model} is not supported. Please provide a valid model either as string or NeuronModel.
            You can also provide non model then a default one will be used"""
        )
    return model, model_id, tokenizer, feature_extractor, image_processor


def pipeline(
    task: str = None,
    model: str | NeuronModel | None = None,
    tokenizer: str | PreTrainedTokenizer | None = None,
    feature_extractor: "str | PreTrainedFeatureExtractor | None" = None,
    image_processor: str | BaseImageProcessor | None = None,
    use_fast: bool = True,
    export: bool = False,
    input_shapes: dict[str, int] | None = {},
    compiler_args: dict[str, int] | None = {},
    token: str | bool | None = None,
    revision: str | None = None,
    trust_remote_code: bool | None = None,
    **kwargs,
) -> Pipeline:
    if task not in NEURONX_SUPPORTED_TASKS:
        raise ValueError(
            f"Task {task} is not supported for the optimum neuron pipeline. Supported tasks are {list(NEURONX_SUPPORTED_TASKS.keys())}"
        )
    # loads default model
    if model is None:
        model = NEURONX_SUPPORTED_TASKS[task]["default"]

    # copied from transformers.pipelines.__init__.py
    commit_hash = kwargs.pop("_commit_hash", None)
    hub_kwargs = {
        "revision": revision,
        "token": token,
        "trust_remote_code": trust_remote_code,
        "_commit_hash": commit_hash,
    }
    batch_size = kwargs.pop("batch_size", None)

    config = kwargs.get("config", None)
    if config is None:
        if isinstance(model, str):
            config = AutoConfig.from_pretrained(model, _from_pipeline=task, **hub_kwargs, **kwargs)
            hub_kwargs["_commit_hash"] = config._commit_hash
        elif isinstance(model, (PreTrainedModel, NeuronModel)):
            if hasattr(model, "encoder"):
                config = model.encoder.config
            else:
                config = model.config

    neuron_config = getattr(config, "neuron", None)
    if neuron_config is None:
        if isinstance(model, str):
            try:
                neuron_config = NeuronConfig.from_pretrained(model, token=token, revision=revision)
            except EnvironmentError:
                # If the model is not a Neuron model, we will just ignore the error
                pass
        elif isinstance(model, NeuronModel):
            neuron_config = getattr(model, "neuron_config", None)

    export_kwargs = {}
    if export:
        if neuron_config is not None:
            raise ValueError("This model has already been exported to Neuron format")
        # Decoder models can select default input shapes from the config
        if task != "text-generation" and not input_shapes:
            input_shapes = {"batch_size": 1, "sequence_length": 128}
            logger.warning(f"No input shapes provided, using default shapes, {input_shapes}")
        else:
            num_cores = None
            # We still support this deprecated argument for backward compatibility
            if "num_cores" in kwargs:
                warnings.warn(
                    "The num_cores argument is deprecated and will be removed in a future version. Please use tensor_parallel_size instead.",
                    category=DeprecationWarning,
                )
                num_cores = kwargs.pop("num_cores")
            export_kwargs = {
                "batch_size": batch_size,
                "sequence_length": kwargs.pop("sequence_length", None),
                "tensor_parallel_size": kwargs.pop("tensor_parallel_size", num_cores),
            }
    else:
        if neuron_config is None:
            raise ValueError("The model must be exported to Neuron format first")
        if input_shapes:
            logger.warning("Input shapes can only be set during export")

    no_feature_extractor_tasks = set()
    no_tokenizer_tasks = set()
    no_image_processor_tasks = set()
    for _task, values in NEURONX_SUPPORTED_TASKS.items():
        if values["type"] == "text":
            no_feature_extractor_tasks.add(_task)
            no_image_processor_tasks.add(_task)
        elif values["type"] in {"image", "video"}:
            no_tokenizer_tasks.add(_task)
            no_feature_extractor_tasks.add(_task)
        elif values["type"] in {"audio"}:
            no_tokenizer_tasks.add(_task)
            no_image_processor_tasks.add(_task)
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

    if task in no_image_processor_tasks:
        load_image_processor = False
    else:
        load_image_processor = True

    model, model_id, tokenizer, feature_extractor, image_processor = load_pipeline(
        model=model,
        targeted_task=task,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        image_processor=image_processor,
        load_tokenizer=load_tokenizer,
        load_feature_extractor=load_feature_extractor,
        load_image_processor=load_image_processor,
        export=export,
        input_shapes=input_shapes,
        compiler_args=compiler_args,
        supported_tasks=NEURONX_SUPPORTED_TASKS,
        hub_kwargs=hub_kwargs,
        token=token,
        **export_kwargs,
    )

    if tokenizer is None and load_tokenizer:
        tokenizer = get_preprocessor(model_id)
    if feature_extractor is None and load_feature_extractor:
        feature_extractor = get_preprocessor(model_id)
    if image_processor is None and load_image_processor:
        image_processor = get_preprocessor(model_id)

    if batch_size is None:
        # If we don't specify a batch_size, the pipeline will assume batch_size 1
        # and it will process the inputs one by one instead of processing them in parallel
        neuron_config = (
            getattr(config, "neuron", None)
            or getattr(model.config, "neuron", None)
            or getattr(model, "neuron_config", None)
        )
        if isinstance(neuron_config, NeuronConfig):
            batch_size = neuron_config.batch_size
        elif isinstance(neuron_config, dict):
            for attr in ["batch_size", "static_batch_size"]:
                batch_size = neuron_config.get(attr, batch_size)
        else:
            batch_size = 1
    if batch_size > 1 and tokenizer is not None and tokenizer.pad_token_id is None:
        # The pipeline needs a pad token to be able to batch
        if isinstance(model.config.eos_token_id, list):
            tokenizer.pad_token_id = model.config.eos_token_id[0]
        else:
            tokenizer.pad_token_id = model.config.eos_token_id

    if hasattr(NEURONX_SUPPORTED_TASKS[task]["impl"], "check_model_type"):
        NEURONX_SUPPORTED_TASKS[task]["impl"].check_model_type = check_model_type

    return transformers_pipeline(
        task,
        model=model,
        tokenizer=tokenizer,
        feature_extractor=feature_extractor,
        image_processor=image_processor,
        use_fast=use_fast,
        batch_size=batch_size,
        pipeline_class=NEURONX_SUPPORTED_TASKS[task]["impl"],
        device=model.device,
        **kwargs,
    )
