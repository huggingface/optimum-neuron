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

from typing import TYPE_CHECKING

from transformers.utils import _LazyModule


_import_structure = {
    "hf_argparser": ["NeuronHfArgumentParser"],
    "trainers": ["NeuronTrainer", "Seq2SeqNeuronTrainer"],
    "training_args": ["NeuronTrainingArguments", "Seq2SeqNeuronTrainingArguments"],
    "modeling_base": ["NeuronBaseModel"],
    "modeling": [
        "NeuronModelForFeatureExtraction",
        "NeuronModelForSentenceTransformers",
        "NeuronModelForMaskedLM",
        "NeuronModelForQuestionAnswering",
        "NeuronModelForSequenceClassification",
        "NeuronModelForTokenClassification",
        "NeuronModelForMultipleChoice",
        "NeuronModelForCausalLM",
    ],
    "modeling_diffusion": [
        "NeuronStableDiffusionPipeline",
        "NeuronStableDiffusionImg2ImgPipeline",
        "NeuronStableDiffusionInpaintPipeline",
        "NeuronLatentConsistencyModelPipeline",
        "NeuronStableDiffusionXLPipeline",
        "NeuronStableDiffusionXLImg2ImgPipeline",
        "NeuronStableDiffusionXLInpaintPipeline",
    ],
    "modeling_decoder": ["NeuronDecoderModel"],
    "modeling_seq2seq": ["NeuronModelForSeq2SeqLM"],
    "accelerate": [
        "NeuronAccelerator",
        "NeuronAcceleratorState",
        "NeuronPartialState",
        "ModelParallelismPlugin",
    ],
    "pipelines": ["pipeline"],
}

if TYPE_CHECKING:
    from .accelerate import ModelParallelismPlugin, NeuronAccelerator, NeuronAcceleratorState, NeuronPartialState
    from .hf_argparser import NeuronHfArgumentParser
    from .modeling import (
        NeuronModelForCausalLM,
        NeuronModelForFeatureExtraction,
        NeuronModelForMaskedLM,
        NeuronModelForMultipleChoice,
        NeuronModelForQuestionAnswering,
        NeuronModelForSentenceTransformers,
        NeuronModelForSequenceClassification,
        NeuronModelForTokenClassification,
    )
    from .modeling_base import NeuronBaseModel
    from .modeling_decoder import NeuronDecoderModel
    from .modeling_diffusion import (
        NeuronLatentConsistencyModelPipeline,
        NeuronStableDiffusionImg2ImgPipeline,
        NeuronStableDiffusionInpaintPipeline,
        NeuronStableDiffusionPipeline,
        NeuronStableDiffusionXLImg2ImgPipeline,
        NeuronStableDiffusionXLInpaintPipeline,
        NeuronStableDiffusionXLPipeline,
    )
    from .modeling_seq2seq import NeuronModelForSeq2SeqLM
    from .pipelines import pipeline
    from .trainers import NeuronTrainer, Seq2SeqNeuronTrainer
    from .training_args import NeuronTrainingArguments, Seq2SeqNeuronTrainingArguments

else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )


import os

from .utils import is_neuron_available, is_neuronx_available, patch_transformers_for_neuron_sdk
from .version import __version__


if not os.environ.get("DISABLE_TRANSFORMERS_PATCHING", False):
    patch_transformers_for_neuron_sdk()
