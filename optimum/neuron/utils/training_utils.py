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
"""Training utilities"""

import os
from typing import TYPE_CHECKING, List, Optional, Union

import torch
import transformers
from accelerate import skip_first_batches as accelerate_skip_first_batches
from torch.utils._pytree import tree_map
from torch.utils.data import DataLoader, Dataset, IterableDataset
from transformers import GenerationMixin
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_BACKBONE_MAPPING_NAMES,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_CTC_MAPPING_NAMES,
    MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES,
    MODEL_FOR_MASKED_LM_MAPPING_NAMES,
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES,
    MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES,
    MODEL_FOR_PRETRAINING_MAPPING_NAMES,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
    MODEL_MAPPING_NAMES,
)
from transformers.trainer_pt_utils import get_model_param_count as transformers_get_model_param_count
from transformers.utils.logging import set_verbosity as set_verbosity_transformers

from ...utils.logging import set_verbosity as set_verbosity_optimum
from ..generation import NeuronGenerationMixin
from . import is_torch_xla_available


if TYPE_CHECKING:
    from transformers import PreTrainedModel

if is_torch_xla_available():
    import torch_xla.distributed.parallel_loader as pl

TRANSFORMERS_MIN_VERSION_FOR_XLA_FSDP = "4.30.0.dev0"
TRANSFORMERS_MIN_VERSION_USE_ACCELERATE = "4.30.0.dev0"


def _generate_supported_model_class_names(
    model_type: str,
    supported_tasks: Optional[Union[str, List[str]]] = None,
) -> List[str]:
    task_mapping = {
        "default": MODEL_MAPPING_NAMES,
        "pretraining": MODEL_FOR_PRETRAINING_MAPPING_NAMES,
        "next-sentence-prediction": MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES,
        "masked-lm": MODEL_FOR_MASKED_LM_MAPPING_NAMES,
        "causal-lm": MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
        "seq2seq-lm": MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
        "speech-seq2seq": MODEL_FOR_SPEECH_SEQ_2_SEQ_MAPPING_NAMES,
        "multiple-choice": MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES,
        "document-question-answering": MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES,
        "question-answering": MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
        "sequence-classification": MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
        "token-classification": MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
        "masked-image-modeling": MODEL_FOR_MASKED_IMAGE_MODELING_MAPPING_NAMES,
        "image-classification": MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
        "ctc": MODEL_FOR_CTC_MAPPING_NAMES,
        "audio-classification": MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,
        "semantic-segmentation": MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES,
        "backbone": MODEL_FOR_BACKBONE_MAPPING_NAMES,
    }

    if supported_tasks is None:
        supported_tasks = task_mapping.keys()

    if isinstance(supported_tasks, str):
        supported_tasks = [supported_tasks]

    model_class_names = []
    for task in supported_tasks:
        class_name = task_mapping[task].get(model_type, None)
        if class_name:
            model_class_names.append(class_name)

    return model_class_names


_SUPPORTED_MODEL_TYPES = [
    ("albert", ("sequence-classification", "token-classification", "question-answering")),
    "bart",
    "bert",
    "camembert",
    "distilbert",
    "electra",
    "gpt-2",
    "gpt_neo",
    "marian",
    "roberta",
    "t5",
    "vit",
    ("xlm-roberta", ("sequence-classification", "token-classification", "question-answering")),
]

_SUPPORTED_MODEL_NAMES = set()
for model_type in _SUPPORTED_MODEL_TYPES:
    if isinstance(model_type, str):
        model_type = (model_type, None)
    _SUPPORTED_MODEL_NAMES.update(_generate_supported_model_class_names(*model_type))


def is_precompilation() -> bool:
    return os.environ.get("NEURON_PARALLEL_COMPILE") == "1"


def is_model_officially_supported(model: "PreTrainedModel") -> bool:
    # In theory the type annotation is not correct since we can have also a XlaFullyShardedDataParallel
    # but let's ignore it here.
    if not is_torch_xla_available():
        raise RuntimeError(
            "is_model_officially_supported requires torch_xla to run, please install it by running: "
            "pip install torch_xla"
        )
    from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel

    if isinstance(model, XlaFullyShardedDataParallel):
        class_name = model.module.__class__.__name__
    else:
        class_name = model.__class__.__name__
    return class_name in _SUPPORTED_MODEL_NAMES


class FirstAndLastDataset(Dataset):
    def __init__(
        self, dataloader: DataLoader, num_repeat: int = 10, gradient_accumulation_steps: int = 1, world_size: int = 1
    ):
        self.dataloader = dataloader
        self.num_repeat = num_repeat * gradient_accumulation_steps * world_size
        self.samples = self.create_samples()

    def _create_samples_for_map_style_dataset(self):
        samples = []
        num_samples = len(self.dataloader.dataset)
        batch_size = self.dataloader.batch_size
        if batch_size is None and self.dataloader.batch_sampler is not None:
            batch_size = self.dataloader.batch_sampler.batch_size

        # TODO: validate that.
        if batch_size is None:
            samples = [self.dataloader.dataset[0]] * self.num_repeat + [self.dataloader.dataset[-1]] * self.num_repeat
            return samples

        num_batches = num_samples // batch_size
        remaining = num_samples % batch_size

        iterator = iter(self.dataloader)
        first_batch = next(iterator)
        samples = [first_batch] * self.num_repeat

        if num_batches >= 1 and remaining != 0:

            def map_fn(example):
                if isinstance(example, torch.Tensor):
                    return example[:remaining]
                else:
                    return example

            last_batch = tree_map(map_fn, first_batch)
            samples += [last_batch] * self.num_repeat

        return samples

    def _create_samples_for_iterable_dataset(self):
        # Will not work if the iterable dataset yields dynamic batch sizes.
        iterator = iter(self.dataloader)
        first_batch = next(iterator)
        samples = [first_batch] * self.num_repeat
        last_batch = None
        while True:
            try:
                last_batch = next(iterator)
            except StopIteration:
                if last_batch is not None:
                    samples += [last_batch] * self.num_repeat
                break
        return samples

    def create_samples(self):
        if isinstance(self.dataloader.dataset, IterableDataset):
            return self._create_samples_for_iterable_dataset()
        else:
            return self._create_samples_for_map_style_dataset()

    def __getitem__(self, idx: int):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


orig_finfo = torch.finfo


def patched_finfo(dtype):
    if dtype is torch.float32:
        return orig_finfo(torch.bfloat16)
    return orig_finfo(dtype)


def patch_generation_mixin_to_neuron_generation_mixin(model: "PreTrainedModel"):
    """
    Changes the vanilla `GenerationMixin` class from Transformers to `NeuronGenerationMixin` in the model's
    inheritance. This allows to make the model Neuron-compatible for generation without much hassle.
    """
    to_visit = [model.__class__]
    should_stop = False
    while to_visit and not should_stop:
        cls = to_visit.pop(0)
        if cls is object:
            continue
        bases = cls.__bases__
        new_bases = []
        for base in bases:
            to_visit.append(base)
            if base == GenerationMixin:
                new_bases.append(NeuronGenerationMixin)
                should_stop = True
            elif base == NeuronGenerationMixin:
                should_stop = True
                new_bases.append(base)
            else:
                new_bases.append(base)
        cls.__bases__ = tuple(new_bases)


def prepare_environment_for_neuron():
    """
    Prepares the system environment for Transformers models training on AWS Neuron.
    """
    # Set compiler flag to compile for transformer model type
    os.environ["NEURON_CC_FLAGS"] = (
        os.environ.get("NEURON_CC_FLAGS", "") + " --model-type=transformer --enable-experimental-O1"
    )


def set_verbosity(verbosity: int):
    set_verbosity_transformers(verbosity)
    set_verbosity_optimum(verbosity)


def patch_transformers_for_neuron_sdk():
    """
    Patches the Transformers library if needed to make it work with AWS Neuron.
    """
    transformers.utils.logging.set_verbosity = set_verbosity


def skip_first_batches(dataloader, num_batches=0):
    """
    Wrapper around `accelerate.data_loader.skip_first_batches` to handle `pl.ParallelLoader` when using
    `torch_xla.distributed`, for XLA FSDP for instance.
    """
    if isinstance(dataloader, (pl.ParallelLoader, pl.PerDeviceLoader)):
        dataloader._loader = skip_first_batches(dataloader._loader, num_batches=num_batches)
    else:
        dataloader = accelerate_skip_first_batches(dataloader, num_batches=num_batches)
    return dataloader


def get_model_param_count(model, trainable_only=False):
    """Wrapper around `transformers.trainer_pt_utils.get_model_param_count` to handle tensor parallelism."""
    # TODO: make it work for TP
    return transformers_get_model_param_count(model, trainable_only=trainable_only)
