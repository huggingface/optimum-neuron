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

import inspect
from typing import TYPE_CHECKING, Type

import torch
import transformers
from accelerate import skip_first_batches as accelerate_skip_first_batches
from neuronx_distributed.pipeline import NxDPPModel
from optimum.utils.logging import set_verbosity as set_verbosity_optimum
from transformers import GenerationMixin
from transformers.utils.logging import set_verbosity as set_verbosity_transformers

from ...generation import GeneralNeuronGenerationMixin, NeuronGenerationMixin
from ...utils.patching import replace_class_in_inheritance_hierarchy


if TYPE_CHECKING:
    from transformers import PreTrainedModel


def is_topology_supported() -> bool:
    import torch_xla.runtime as xr

    num_devices = xr.world_size()
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


def _get_model_param_count(model: "torch.nn.Module | NxDPPModel"):
    """Counts the number of parameters of the model."""
    import torch_xla
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
        torch_xla.sync()
        param_count = int(param_count.detach().cpu().item())
        return param_count

    all_param_count = sum(numel(n, p) for n, p in named_parameters)
    trainable_param_count = sum(numel(n, p) for n, p in named_parameters if p.requires_grad)
    if pp_size > 1:
        all_param_count = reduce_param_count_over_pp_ranks(all_param_count)
        trainable_param_count = reduce_param_count_over_pp_ranks(trainable_param_count)

    torch_xla.sync()
    return trainable_param_count, all_param_count


def get_model_param_count(model: "torch.nn.Module | NxDPPModel", trainable_only: bool = False) -> int:
    trainable_param_count, all_param_count = _get_model_param_count(model)
    if trainable_only:
        output = trainable_param_count
    else:
        output = all_param_count
    return output


def is_logging_process() -> bool:
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

    return dp_rank == tp_rank == 0 and pp_rank == pp_size - 1


def is_logging_process_method(self) -> bool:
    """
    Method version of `is_logging_process`, useful when this is used to patch a method from the Trainer class.
    """
    return is_logging_process()


def is_custom_modeling_model(model) -> bool:
    from peft import PeftModel

    model_to_consider = model
    if isinstance(model, PeftModel):
        model_to_consider = model.get_base_model()
    return inspect.getmodule(model_to_consider.__class__).__name__.startswith("optimum.neuron.models.training")


def checkpoint_with_kwargs(fn, *args, **kwargs):
    """XLA-compatible gradient checkpointing that accepts keyword arguments via functools.partial."""
    from functools import partial

    from torch_xla.utils.checkpoint import checkpoint

    fn_with_kwargs = partial(fn, **kwargs)
    return checkpoint(fn_with_kwargs, *args)
