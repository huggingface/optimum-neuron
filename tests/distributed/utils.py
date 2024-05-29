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
"""Utilities for tests distributed."""

import inspect
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import torch
from transformers import AutoTokenizer
from transformers.models.auto import get_values
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_BACKBONE_MAPPING_NAMES,
    MODEL_FOR_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_CTC_MAPPING_NAMES,
    MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_MASKED_LM_MAPPING_NAMES,
    MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES,
    MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES,
    MODEL_FOR_PRETRAINING_MAPPING_NAMES,
    MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES,
    MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES,
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES,
    MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES,
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
)

from optimum.neuron.utils.require_utils import requires_neuronx_distributed, requires_torch_xla


if TYPE_CHECKING:
    from transformers import PreTrainedModel


@requires_neuronx_distributed
def generate_dummy_labels(
    model: "PreTrainedModel",
    shape: List[int],
    vocab_size: Optional[int] = None,
    seed: Optional[int] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> Dict[str, torch.Tensor]:
    """Generates dummy labels."""
    from neuronx_distributed.pipeline import NxDPPModel

    if isinstance(model, NxDPPModel):
        model_class_name = model.original_torch_module.__class__.__name__
    else:
        model_class_name = model.__class__.__name__

    labels = {}

    batch_size = shape[0]

    if model_class_name in [
        *get_values(MODEL_FOR_NEXT_SENTENCE_PREDICTION_MAPPING_NAMES),
        *get_values(MODEL_FOR_MULTIPLE_CHOICE_MAPPING_NAMES),
        *get_values(MODEL_FOR_IMAGE_CLASSIFICATION_MAPPING_NAMES),
        *get_values(MODEL_FOR_BACKBONE_MAPPING_NAMES),
        *get_values(MODEL_FOR_AUDIO_CLASSIFICATION_MAPPING_NAMES),
    ]:
        labels["labels"] = torch.zeros(batch_size, dtype=torch.long, device=device)
    elif model_class_name in [
        *get_values(MODEL_FOR_QUESTION_ANSWERING_MAPPING_NAMES),
        *get_values(MODEL_FOR_DOCUMENT_QUESTION_ANSWERING_MAPPING_NAMES),
        "XLNetForQuestionAnswering",
    ]:
        labels["start_positions"] = torch.zeros(batch_size, dtype=torch.long, device=device)
        labels["end_positions"] = torch.zeros(batch_size, dtype=torch.long, device=device)
    elif model_class_name in get_values(MODEL_FOR_SEQUENCE_CLASSIFICATION_MAPPING_NAMES):
        if not hasattr(model.config, "problem_type") or model.config.problem_type is None:
            raise ValueError(
                "Could not retrieve the problem type for the sequence classification task, please set "
                'model.config.problem_type to one of the following values: "regression", '
                '"single_label_classification", or "multi_label_classification".'
            )

        if model.config.problem_type == "regression":
            labels_shape = (batch_size, model.config.num_labels)
            labels_dtype = torch.float32
        elif model.config.problem_type == "single_label_classification":
            labels_shape = (batch_size,)
            labels_dtype = torch.long
        elif model.config.problem_type == "multi_label_classification":
            labels_shape = (batch_size, model.config.num_labels)
            labels_dtype = torch.float32
        else:
            raise ValueError(
                'Expected model.config.problem_type to be either: "regression", "single_label_classification"'
                f', or "multi_label_classification", but "{model.config.problem_type}" was provided.'
            )
        labels["labels"] = torch.zeros(*labels_shape, dtype=labels_dtype, device=device)
    elif model_class_name in [
        *get_values(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES),
        *get_values(MODEL_FOR_PRETRAINING_MAPPING_NAMES),
        *get_values(MODEL_FOR_CAUSAL_LM_MAPPING_NAMES),
        *get_values(MODEL_FOR_MASKED_LM_MAPPING_NAMES),
        *get_values(MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING_NAMES),
        *get_values(MODEL_FOR_SEMANTIC_SEGMENTATION_MAPPING_NAMES),
        "GPT2DoubleHeadsModel",
        "PeftModelForCausalLM",
        "PeftModelForSeq2SeqLM",
    ]:
        if vocab_size is None:
            raise ValueError(
                "The vocabulary size needs to be specified to generate dummy labels for language-modeling tasks."
            )
        if seed is not None:
            orig_seed = torch.seed()
            torch.manual_seed(seed)
        if model_class_name in get_values(MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES):
            max_value = model.config.num_labels
        else:
            max_value = vocab_size
        random_labels = torch.randint(0, max_value, shape, dtype=torch.long)
        if device is not None:
            random_labels = random_labels.to(device)
        labels["labels"] = random_labels
        if seed is not None:
            torch.manual_seed(orig_seed)
    elif model_class_name in [*get_values(MODEL_FOR_CTC_MAPPING_NAMES)]:
        labels["labels"] = torch.zeros(shape, dtype=torch.float32, device=device)
    else:
        raise NotImplementedError(f"Generating the dummy input named for {model_class_name} is not supported yet.")
    return labels


@requires_torch_xla
@requires_neuronx_distributed
def gather_along_last_dim(input_: torch.Tensor) -> torch.Tensor:
    import torch_xla.core.xla_model as xm
    from neuronx_distributed.parallel_layers.parallel_state import (
        get_tensor_model_parallel_group,
        get_tensor_model_parallel_rank,
        get_tensor_model_parallel_size,
    )

    """Gathers tensors and concatenate along the last dimension."""

    world_size = get_tensor_model_parallel_size()
    # Bypass the function if we are using only 1 device.
    if world_size == 1:
        return input_

    # Size and dimension.
    last_dim = input_.dim() - 1
    rank = get_tensor_model_parallel_rank()

    tensor_list = [torch.empty_like(input_) for _ in range(world_size)]
    tensor_list[rank] = input_
    torch.distributed.all_gather(tensor_list, input_, group=get_tensor_model_parallel_group())

    # Compilation fails when not synchronizing here.
    xm.mark_step()

    # Note: torch.cat already creates a contiguous tensor.
    output = torch.cat(tensor_list, dim=last_dim).contiguous()

    xm.mark_step()

    return output


@requires_torch_xla
@requires_neuronx_distributed
def gather_along_first_dim(input_: torch.Tensor) -> torch.Tensor:
    import torch_xla.core.xla_model as xm
    from neuronx_distributed.parallel_layers.parallel_state import (
        get_tensor_model_parallel_group,
        get_tensor_model_parallel_size,
    )

    world_size = get_tensor_model_parallel_size()
    # Bypass the function if we are using only 1 GPU.
    if world_size == 1:
        return input_

    output = xm.all_gather(input_, groups=get_tensor_model_parallel_group(as_list=True), pin_layout=False)

    return output


@requires_torch_xla
@requires_neuronx_distributed
def gather_along_dim(input_: torch.Tensor, dim: int) -> torch.Tensor:
    input_ = input_.clone().contiguous()
    if dim == 0:
        return gather_along_first_dim(input_)
    elif dim in [-1, input_.dim() - 1]:
        return gather_along_last_dim(input_)
    else:
        t = input_.transpose(0, dim).contiguous()
        gathered_t = gather_along_first_dim(t)
        return gathered_t.transpose(0, dim).contiguous()


def get_model_inputs(
    model: "PreTrainedModel",
    model_name_or_path: str,
    include_labels: bool = True,
    random_labels: bool = True,
    batch_size: int = 1,
    pad_to_multiple_of: Optional[int] = None,
):
    input_str = "Hello there, I'm Michael and I live in Paris!"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    inputs = tokenizer(input_str, return_tensors="pt")

    if model.config.is_encoder_decoder:
        sig = inspect.signature(model.forward)
        for input_name in inputs:
            decoder_input_name = f"decoder_{input_name}"
            if decoder_input_name in sig.parameters:
                inputs[decoder_input_name] = inputs[input_name].clone()

    if include_labels:
        if random_labels:
            labels = generate_dummy_labels(model, inputs["input_ids"].shape, vocab_size=model.config.vocab_size)
            inputs.update(**labels)
        else:
            labels = tokenizer(input_str, return_tensors="pt")["input_ids"]
            inputs["labels"] = labels

    if batch_size > 1:
        for name, tensor in inputs.items():
            repeat = [batch_size] + [1] * (tensor.dim() - 1)
            tensor = tensor.repeat(*repeat)
            inputs[name] = tensor

    if pad_to_multiple_of is not None:
        pad_token_id = getattr(model.config, "pad_token_id", 1)
        for name, tensor in inputs.items():
            if tensor.dim() == 2 and tensor.shape[1] % pad_to_multiple_of != 0:
                if "attention_mask" not in name:
                    pad_value = pad_token_id
                else:
                    pad_value = 1
                tensor = torch.nn.functional.pad(
                    tensor,
                    pad=(0, pad_to_multiple_of - tensor.shape[1] % pad_to_multiple_of),
                    value=pad_value,
                )
                inputs[name] = tensor
    return inputs
