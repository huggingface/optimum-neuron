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

from typing import TYPE_CHECKING, List, Optional, Type, Union

import torch
import transformers
from accelerate import skip_first_batches as accelerate_skip_first_batches
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
from transformers.utils.logging import set_verbosity as set_verbosity_transformers

from ...utils.logging import set_verbosity as set_verbosity_optimum
from ..generation import GeneralNeuronGenerationMixin, NeuronGenerationMixin
from . import is_neuronx_distributed_available
from .patching import replace_class_in_inheritance_hierarchy
from .require_utils import requires_neuronx_distributed, requires_torch_xla


if is_neuronx_distributed_available():
    from neuronx_distributed.pipeline import NxDPPModel


if TYPE_CHECKING:
    from transformers import PreTrainedModel


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


def is_model_officially_supported(model: "PreTrainedModel") -> bool:
    class_name = model.__class__.__name__
    return class_name in _SUPPORTED_MODEL_NAMES


@requires_torch_xla
def is_topology_supported() -> bool:
    import torch_xla.core.xla_model as xm

    num_devices = xm.xrt_world_size()
    allowed_number_of_devices = [1, 2, 8]
    return num_devices in allowed_number_of_devices or num_devices % 32 == 0


def patch_generation_mixin_to_neuron_generation_mixin(
    model: "PreTrainedModel", neuron_generation_mixin_cls: Type = NeuronGenerationMixin
):
    """
    Changes the vanilla `GenerationMixin` class from Transformers to `neuron_generation_mixin_cls` in the model's
    inheritance. This allows to make the model Neuron-compatible for generation without much hassle.
    """
    return replace_class_in_inheritance_hierarchy(model, GenerationMixin, neuron_generation_mixin_cls)


def patch_generation_mixin_to_general_neuron_generation_mixin(model: "PreTrainedModel"):
    """
    Changes the vanilla `GenerationMixin` class from Transformers to `GeneralNeuronGenerationMixin` in the model's
    inheritance. This allows to make the model Neuron-compatible for generation without much hassle.
    """
    return patch_generation_mixin_to_neuron_generation_mixin(
        model, neuron_generation_mixin_cls=GeneralNeuronGenerationMixin
    )


def set_verbosity(verbosity: int):
    set_verbosity_transformers(verbosity)
    set_verbosity_optimum(verbosity)


def patch_transformers_for_neuron_sdk():
    """
    Patches the Transformers library if needed to make it work with AWS Neuron.
    """
    transformers.utils.logging.set_verbosity = set_verbosity


@requires_torch_xla
def skip_first_batches(dataloader, num_batches=0):
    """
    Wrapper around `accelerate.data_loader.skip_first_batches` to handle `pl.ParallelLoader` when using
    `torch_xla.distributed`, for XLA FSDP for instance.
    """
    import torch_xla.distributed.parallel_loader as pl

    if isinstance(dataloader, (pl.ParallelLoader, pl.PerDeviceLoader, pl.MpDeviceLoader)):
        dataloader._loader = skip_first_batches(dataloader._loader, num_batches=num_batches)
    else:
        dataloader = accelerate_skip_first_batches(dataloader, num_batches=num_batches)
    return dataloader


@requires_neuronx_distributed
def _get_model_param_count(model: Union[torch.nn.Module, "NxDPPModel"]):
    """Counts the number of parameters of the model."""
    import torch_xla.core.xla_model as xm
    from neuronx_distributed.parallel_layers.parallel_state import (
        get_pipeline_model_parallel_group,
        get_pipeline_model_parallel_rank,
        get_pipeline_model_parallel_size,
        get_tensor_model_parallel_size,
        model_parallel_is_initialized,
    )
    from neuronx_distributed.pipeline import NxDPPModel
    from neuronx_distributed.pipeline.partition import analyze_shared_weights_across_stages

    if isinstance(model, NxDPPModel):
        named_parameters = model.local_named_parameters()
        shared = analyze_shared_weights_across_stages(model.traced_model, model.partitions)
        shared_parameters_across_pipeline_stages = {
            t[0]: t[1] for shared_parameter_info in shared for t in shared_parameter_info
        }
    else:
        named_parameters = model.named_parameters()
        shared_parameters_across_pipeline_stages = {}

    # We make sure `named_parameters` is not an iterator because we are going to iterate over it twice.
    named_parameters = list(named_parameters)

    if torch.distributed.is_initialized() and model_parallel_is_initialized():
        tp_size = get_tensor_model_parallel_size()
        pp_size = get_pipeline_model_parallel_size()
        pp_rank = get_pipeline_model_parallel_rank()
    else:
        tp_size = 1
        pp_size = 1
        pp_rank = 0

    def numel(parameter_name, parameter) -> int:
        should_count_param = shared_parameters_across_pipeline_stages.get(parameter_name, pp_rank) == pp_rank

        num_elements = parameter.numel()
        if getattr(parameter, "tensor_model_parallel", False):
            num_elements *= tp_size

        if parameter.__class__.__name__ == "Params4bit":
            if hasattr(parameter, "element_size"):
                num_bytes = parameter.element_size()
            elif not hasattr(parameter, "quant_storage"):
                num_bytes = 1
            else:
                num_bytes = parameter.quant_storage.itemsize
            num_elements = num_elements * 2 * num_bytes

        return num_elements if should_count_param else 0

    def reduce_param_count_over_pp_ranks(param_count: int):
        param_count = torch.tensor(param_count, dtype=torch.float32).to(xm.xla_device())
        param_count = xm.all_reduce(xm.REDUCE_SUM, param_count, groups=get_pipeline_model_parallel_group(as_list=True))
        xm.mark_step()
        param_count = int(param_count.detach().cpu().item())
        return param_count

    all_param_count = sum(numel(n, p) for n, p in named_parameters)
    trainable_param_count = sum(numel(n, p) for n, p in named_parameters if p.requires_grad)
    if pp_size > 1:
        all_param_count = reduce_param_count_over_pp_ranks(all_param_count)
        trainable_param_count = reduce_param_count_over_pp_ranks(trainable_param_count)

    return trainable_param_count, all_param_count


@requires_neuronx_distributed
def get_model_param_count(model: Union[torch.nn.Module, "NxDPPModel"], trainable_only: bool = False) -> int:
    trainable_param_count, all_param_count = _get_model_param_count(model)
    if trainable_only:
        output = trainable_param_count
    else:
        output = all_param_count
    return output


@requires_neuronx_distributed
def is_main_worker_for_metrics() -> bool:
    from neuronx_distributed.parallel_layers.parallel_state import (
        get_data_parallel_rank,
        get_pipeline_model_parallel_rank,
        get_pipeline_model_parallel_size,
        get_tensor_model_parallel_rank,
    )

    if not torch.distributed.is_initialized():
        return True

    dp_rank = get_data_parallel_rank()
    tp_rank = get_tensor_model_parallel_rank()
    pp_rank = get_pipeline_model_parallel_rank()
    pp_size = get_pipeline_model_parallel_size()

    can_log_loss = dp_rank == tp_rank == 0 and pp_rank == pp_size - 1

    return can_log_loss


def is_main_worker_for_metrics_method(self) -> bool:
    """
    Method version of `is_main_worker_for_metrics`, useful when this is used to patch a method from the Trainer class.
    """
    return is_main_worker_for_metrics()
