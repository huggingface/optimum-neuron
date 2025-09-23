# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""
Utilities for Pipeline Parallelism model setup and parameter management.
"""

import contextlib
import functools
import logging as python_logging
from collections.abc import Iterable
from types import ModuleType
from typing import Any, Callable

import neuronx_distributed
import torch
from neuronx_distributed.parallel_layers.parallel_state import get_pipeline_model_parallel_size
from neuronx_distributed.pipeline import NxDPPModel
from neuronx_distributed.pipeline.trace import HFTracerWrapper, NxDTracer
from neuronx_distributed.pipeline.trace import trace_model as orig_trace_model
from torch import nn
from transformers.utils.fx import HFTracer, create_wrapper

from .transformations_utils import get_tensor_model_parallel_attributes


class OptimumNeuronFXTracer(HFTracerWrapper):
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        return NxDTracer.is_leaf_module(self, m, module_qualified_name) or HFTracer.is_leaf_module(
            self, m, module_qualified_name
        )


class NxDPPModelWithPeftSupport(NxDPPModel):
    """
    Subclass of NxDPPModel that adds support for PEFT models by handling prefixed module names.
    """

    def cut_pipeline_stage(self, cut_point):
        from ...peft import NeuronPeftModel

        prefix = ""
        if isinstance(self.original_torch_module, NeuronPeftModel):
            prefix = self.original_torch_module.get_base_model_prefix()

        cut_point = cut_point[len(prefix) + 1 :] if prefix and cut_point.startswith(prefix + ".") else cut_point

        return super().cut_pipeline_stage(cut_point)

    def _build_parameter_buffer_name_mapping(self, qualname_map):
        from ...peft import NeuronPeftModel

        # Temporarily set original_torch_module to the base model for correct name mapping
        orig_module = self.original_torch_module
        module = orig_module
        if isinstance(module, NeuronPeftModel):
            module = orig_module.get_base_model()
        self.original_torch_module = module
        super()._build_parameter_buffer_name_mapping(qualname_map)

        # Restore original_torch_module
        self.original_torch_module = orig_module


def trace_model(
    model: nn.Module,
    args: list[Any] | None = None,
    kwargs: dict[Any, Any] | None = None,
    input_names: list[str] | None = None,
    tracer_cls: Any | str = None,
    leaf_modules: list[Any] | None = None,
    autowrap_functions: list[Callable] | None = None,
    autowrap_modules: list[ModuleType] | None = None,
    autowrap_obj_methods: dict[Any, list[Callable]] | None = None,
):
    from ...peft import NeuronPeftModel

    if isinstance(model, NeuronPeftModel):
        model = model.get_base_model()
    return orig_trace_model(
        model,
        args=args,
        kwargs=kwargs,
        input_names=input_names,
        tracer_cls=tracer_cls,
        leaf_modules=leaf_modules,
        autowrap_functions=autowrap_functions,
        autowrap_modules=autowrap_modules,
        autowrap_obj_methods=autowrap_obj_methods,
    )


class MetaParametersOnly:
    """
    Context manager that forces all nn.Parameter creations to use the meta device while leaving buffers on the CPU
    device.
    """

    def __init__(self):
        self.original_parameter_new = nn.Parameter.__new__

        @functools.wraps(self.original_parameter_new)
        def patched_parameter_new(cls, data=None, requires_grad=True):
            with torch.device("meta"):
                return self.original_parameter_new(cls, data, requires_grad)

        self.patched_parameter_new = patched_parameter_new

    def __enter__(self):
        nn.Parameter.__new__ = self.patched_parameter_new
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        nn.Parameter.__new__ = self.original_parameter_new


def create_nxdpp_model(model) -> NxDPPModel:
    """
    Creates an NxDPPModel wrapper for pipeline parallelism.

    Args:
        model: The model to wrap for pipeline parallelism

    Returns:
        NxDPPModel: The wrapped model ready for pipeline parallelism
    """

    if not model.supports_pipeline_parallelism():
        raise NotImplementedError(f"The model {model.__class__.__name__} does not support pipeline parallelism.")

    model.config.use_cache = False
    model.config.output_attentions = False
    model.config.output_hidden_states = False

    orig_class_forward = model.__class__.forward
    if hasattr(orig_class_forward, "__wrapped__"):
        # If the forward method is wrapped, it was wrapped by the `can_return_tuple` decorator, we need to
        # unwrap it first.
        model.__class__.forward = orig_class_forward.__wrapped__

    # It is important to use this modified version for PEFT models.
    neuronx_distributed.pipeline.model.trace_model = trace_model

    model = NxDPPModelWithPeftSupport(
        model,
        transformer_layer_cls=model.PIPELINE_TRANSFORMER_LAYER_CLS,
        num_microbatches=model.trn_config.pipeline_parallel_num_microbatches,
        virtual_pipeline_size=model.trn_config.virtual_pipeline_parallel_size,
        output_loss_value_spec=(True, False),
        input_names=model.PIPELINE_INPUT_NAMES,
        leaf_module_cls=model.PIPELINE_LEAF_MODULE_CLASSE_NAMES,
        use_zero1_optimizer=model.trn_config.pipeline_parallel_use_zero1_optimizer,
        tracer_cls=OptimumNeuronFXTracer,
        auto_partition=True,
        # By default it is set to True to create less graphs, but it complicates things when reducing the
        # loss for logging.
        return_loss_on_cpu=False,
    )

    # Setting it back to the original forward.
    model.__class__.forward = orig_class_forward
    return model


@contextlib.contextmanager
def suppress_logging(logger_names=None):
    """
    Context manager to suppress logging from specified loggers or all loggers.
    """
    if logger_names is None:
        # Suppress all logging
        original_level = python_logging.root.level
        python_logging.root.setLevel(python_logging.CRITICAL + 1)
        try:
            yield
        finally:
            python_logging.root.setLevel(original_level)
    else:
        # Suppress specific loggers
        original_levels = {}
        loggers = []

        for logger_name in logger_names:
            logger_obj = python_logging.getLogger(logger_name)
            loggers.append(logger_obj)
            original_levels[logger_name] = logger_obj.level
            logger_obj.setLevel(python_logging.CRITICAL + 1)

        try:
            yield
        finally:
            for logger_name, logger_obj in zip(logger_names, loggers):
                logger_obj.setLevel(original_levels[logger_name])


def get_pipeline_parameters_for_current_stage(model) -> set[str]:
    """
    Determines which parameters are needed for the current pipeline stage.

    Uses a meta device model wrapped with NxDPPModel to determine parameter
    assignment across pipeline stages, then returns the parameter names
    needed for the current stage.

    Args:
        model: The model to analyze for pipeline parameter assignment

    Returns:
        Set of parameter names needed for the current pipeline stage
    """
    from ...peft import NeuronPeftModel

    with suppress_logging():
        if get_pipeline_model_parallel_size() <= 1 or not model.supports_pipeline_parallelism():
            # Return all parameters if no pipeline parallelism
            parameter_names = set(model.state_dict().keys())
        elif isinstance(model, NeuronPeftModel):
            base_model = model.get_base_model()
            with torch.device("meta"):
                meta_model = base_model.__class__(base_model.config, base_model.trn_config)
                meta_nxdpp_model = create_nxdpp_model(meta_model)

            # Get local parameter substrings from the base model's NxDPPModel
            local_parameter_substrings = list(meta_nxdpp_model.local_state_dict().keys())
            for idx, name in enumerate(local_parameter_substrings):
                local_parameter_substrings[idx] = name.rsplit(".", 1)[0]

            # Match local parameter substrings to full parameter names in the PEFT model
            parameter_names = []
            for name in model.state_dict().keys():
                if any(substring in name for substring in local_parameter_substrings):
                    parameter_names.append(name)
            parameter_names = set(parameter_names)
        else:
            with torch.device("meta"):
                meta_model = model.__class__(model.config, model.trn_config)
                meta_nxdpp_model = create_nxdpp_model(meta_model)
            parameter_names = set(meta_nxdpp_model.local_state_dict().keys())

    return parameter_names


def move_params_to_cpu(model: nn.Module, param_names: Iterable[str]):
    """
    Moves specified model parameters to CPU while preserving tensor model parallel attributes.

    Args:
        model: The model containing the parameters to move
        param_names: Iterable of parameter names to move to CPU
    """
    param_names_set = set(param_names)

    for name, param in model.named_parameters():
        if name in param_names_set:
            cpu_tensor = torch.empty_like(param, device="cpu")
            cpu_param = nn.Parameter(cpu_tensor)
            tensor_model_parallel_attributes = get_tensor_model_parallel_attributes(param)
            for attr_name, attr in tensor_model_parallel_attributes.items():
                setattr(cpu_param, attr_name, attr)
            module = model
            parts = name.split(".")
            for part in parts[:-1]:
                module = getattr(module, part)
            setattr(module, parts[-1], cpu_param)


def dynamic_torch_fx_wrap(func):
    """
    Wraps a function dynamically (does not need to be done at the top of the module like with `torch.fx.wrap`).
    This is useful for functions that fail to be traced by the HF tracer during pipeline parallelism setup.
    """
    return create_wrapper(func, "call_function")
