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

import os
from typing import TYPE_CHECKING

from .utils.training_utils import patch_transformers_for_neuron_sdk


if not os.environ.get("DISABLE_TRANSFORMERS_PATCHING", False):
    patch_transformers_for_neuron_sdk()

from transformers.utils import _LazyModule


_import_structure = {
    "hf_argparser": ["NeuronHfArgumentParser"],
    "trainers": [
        "NeuronTrainer",
        "NeuronSFTTrainer",
        "NeuronTrainingArguments",
        "NeuronSFTConfig",
    ],
    "modeling_traced": ["NeuronTracedModel"],
    "modeling": [
        "NeuronModelForFeatureExtraction",
        "NeuronModelForSentenceTransformers",
        "NeuronModelForMaskedLM",
        "NeuronModelForQuestionAnswering",
        "NeuronModelForSequenceClassification",
        "NeuronModelForTokenClassification",
        "NeuronModelForMultipleChoice",
        "NeuronModelForImageClassification",
        "NeuronModelForSemanticSegmentation",
        "NeuronModelForObjectDetection",
        "NeuronModelForCTC",
        "NeuronModelForAudioClassification",
        "NeuronModelForAudioFrameClassification",
        "NeuronModelForXVector",
    ],
    "modeling_diffusion": [
        "NeuronDiffusionPipelineBase",
        "NeuronStableDiffusionPipeline",
        "NeuronStableDiffusionImg2ImgPipeline",
        "NeuronStableDiffusionInpaintPipeline",
        "NeuronStableDiffusionInstructPix2PixPipeline",
        "NeuronLatentConsistencyModelPipeline",
        "NeuronStableDiffusionXLPipeline",
        "NeuronStableDiffusionXLImg2ImgPipeline",
        "NeuronStableDiffusionXLInpaintPipeline",
        "NeuronStableDiffusionControlNetPipeline",
        "NeuronStableDiffusionXLControlNetPipeline",
        "NeuronPixArtAlphaPipeline",
        "NeuronPixArtSigmaPipeline",
        "NeuronFluxPipeline",
        "NeuronFluxInpaintPipeline",
    ],
    "modeling_decoder": ["NeuronModelForCausalLM"],
    "modeling_seq2seq": [
        "NeuronModelForSeq2SeqLM",
    ],
    "models": [],
    "accelerate": [
        "NeuronAccelerator",
        "NeuronAcceleratorState",
        "NeuronPartialState",
    ],
    "pipelines": ["pipeline"],
}

# Model structures
_import_structure["models.inference.bert"] = [
    "NeuronBertModel",
    "NeuronBertForMaskedLM",
    "NeuronBertForQuestionAnswering",
    "NeuronBertForSequenceClassification",
    "NeuronBertForTokenClassification",
    "NeuronBertForMultipleChoice",
]
_import_structure["models.inference.clip"] = ["NeuronCLIPModel", "NeuronCLIPForImageClassification"]
_import_structure["models.inference.whisper"] = ["NeuronWhisperForConditionalGeneration"]
_import_structure["models.inference.yolos"] = ["NeuronYolosForObjectDetection"]

if TYPE_CHECKING:
    from .accelerate import NeuronAccelerator, NeuronAcceleratorState, NeuronPartialState
    from .hf_argparser import NeuronHfArgumentParser
    from .modeling import (
        NeuronModelForAudioClassification,
        NeuronModelForAudioFrameClassification,
        NeuronModelForCTC,
        NeuronModelForFeatureExtraction,
        NeuronModelForImageClassification,
        NeuronModelForMaskedLM,
        NeuronModelForMultipleChoice,
        NeuronModelForObjectDetection,
        NeuronModelForQuestionAnswering,
        NeuronModelForSemanticSegmentation,
        NeuronModelForSentenceTransformers,
        NeuronModelForSequenceClassification,
        NeuronModelForTokenClassification,
        NeuronModelForXVector,
    )
    from .modeling_decoder import NeuronModelForCausalLM
    from .modeling_diffusion import (
        NeuronDiffusionPipelineBase,
        NeuronFluxInpaintPipeline,
        NeuronFluxPipeline,
        NeuronLatentConsistencyModelPipeline,
        NeuronPixArtAlphaPipeline,
        NeuronPixArtSigmaPipeline,
        NeuronStableDiffusionControlNetPipeline,
        NeuronStableDiffusionImg2ImgPipeline,
        NeuronStableDiffusionInpaintPipeline,
        NeuronStableDiffusionInstructPix2PixPipeline,
        NeuronStableDiffusionPipeline,
        NeuronStableDiffusionXLControlNetPipeline,
        NeuronStableDiffusionXLImg2ImgPipeline,
        NeuronStableDiffusionXLInpaintPipeline,
        NeuronStableDiffusionXLPipeline,
    )
    from .modeling_seq2seq import NeuronModelForSeq2SeqLM
    from .modeling_traced import NeuronTracedModel

    # Modeling
    from .models.inference.bert import (
        NeuronBertForMaskedLM,
        NeuronBertForMultipleChoice,
        NeuronBertForQuestionAnswering,
        NeuronBertForSequenceClassification,
        NeuronBertForTokenClassification,
        NeuronBertModel,
    )
    from .models.inference.clip import NeuronCLIPForImageClassification, NeuronCLIPModel
    from .models.inference.whisper import NeuronWhisperForConditionalGeneration
    from .models.inference.yolos import NeuronYolosForObjectDetection
    from .pipelines import pipeline
    from .trainers import (
        NeuronSFTConfig,
        NeuronSFTTrainer,
        NeuronTrainer,
        NeuronTrainingArguments,
    )

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )


from .utils import is_neuron_available, is_neuronx_available, patch_transformers_for_neuron_sdk
from .version import __version__
