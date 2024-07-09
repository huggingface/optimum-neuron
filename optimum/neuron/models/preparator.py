# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Implements the NeuronPreparator class, which transforms a model with custom modules defined for Neuron."""

import contextlib
import importlib
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedModel
from transformers.utils import is_peft_available

from ..accelerate.state import NeuronAcceleratorState
from ..accelerate.utils import AutocastBackend
from ..utils import (
    DynamicPatch,
    ModelPatcher,
    NeuronPeftModel,
    patch_within_function,
    replace_class_in_inheritance_hierarchy,
)
from .core import NeuronAttention, create_patched_finfo, create_patched_save_pretrained


DEFAULT_MODEL_PATCHING_SPECS = [
    ("config.layerdrop", 0),
    ("no_sync", lambda: contextlib.nullcontext()),
]


class NeuronPreparator:
    _TRANSFORMERS_TO_NEURON_CLASSES: Dict[str, Dict[str, str]] = {
        "llama": {
            "LlamaAttention": "NeuronLlamaAttention",
            "LlamaModel": "NeuronLlamaModel",
        },
        "mistral": {
            "MistralAttention": "NeuronMistralAttention",
            "MistralModel": "NeuronMistralModel",
        },
        "gpt_neox": {
            "GPTNeoXAttention": "NeuronGPTNeoXAttention",
            "GPTNeoXModel": "NeuronGPTNeoXModel",
        },
    }

    @classmethod
    def prepare_modeling(cls, model: PreTrainedModel, **options):
        """
        Prepares the modeling of a model by potentially changing some of its modules with Neuron optimized versions of
        them.
        """
        if model.config.model_type not in cls._TRANSFORMERS_TO_NEURON_CLASSES:
            return

        patches = cls._TRANSFORMERS_TO_NEURON_CLASSES[model.config.model_type]
        module = importlib.import_module(f"..modeling_{model.config.model_type}.py")
        for name, mod in model.modules():
            replacement_cls_name = patches.get(mod.__class__.__name__, "")
            if replacement_cls_name:
                names = name.rsplit(".", maxsplit=1)
                if len(names) == 1:
                    parent, attr_name = model, names[0]
                else:
                    parent, attr_name = model.get_submodule(names[0]), names[1]
                replacement_cls = getattr(module, replacement_cls_name)
                setattr(parent, attr_name, replacement_cls.from_original(mod, **options))

        # Flash attention
        attn_implementation = getattr(model.config, "_attn_implementation", None)
        flash_attention_enabled = options.get("flash_attention_enabled", None)
        enabled = False
        if attn_implementation is not None:
            enabled = attn_implementation == "flash_attention_v2"
            model.config._attn_implementation = "eager"
        if flash_attention_enabled is not None:
            enabled = flash_attention_enabled

        for mod in model.modules():
            if isinstance(mod, NeuronAttention):
                mod.flash_attention_enabled = enabled

    @classmethod
    def patch_model_for_neuron(
        cls,
        model: "torch.nn.Module",
        patching_specs: Optional[List[Tuple[str, Any]]] = None,
    ) -> "torch.nn.Module":
        """
        Patches the model in various ways to make sure it works properly on Neuron devices.
        """
        if patching_specs is None:
            patching_specs = DEFAULT_MODEL_PATCHING_SPECS

        # Working on a copy for safety.
        patching_specs = list(patching_specs)

        accelerator_state = NeuronAcceleratorState()

        mixed_precision_is_bf16 = accelerator_state.mixed_precision == "bf16"
        patched_finfo = create_patched_finfo(
            xla_downcast_bf16=mixed_precision_is_bf16 and accelerator_state.downcast_bfloat,
            use_amp=mixed_precision_is_bf16 and accelerator_state.autocast_backend is AutocastBackend.AMP,
            xla_use_bf16=mixed_precision_is_bf16 and not accelerator_state.downcast_bfloat,
        )
        patching_specs.append(
            (
                "forward",
                DynamicPatch(patch_within_function(("torch.finfo", patched_finfo))),
            ),
        )

        if isinstance(model, PreTrainedModel):
            patching_specs.append(
                (
                    "save_pretrained",
                    DynamicPatch(create_patched_save_pretrained),
                ),
            )

        # TODO: @michaelbenayoun generalize an implementation of gradient checkpointing working for:
        #   - DDP
        #   - TP
        #   - PP
        # if hasattr(model, "gradient_checkpointing_enable"):
        #     patching_specs.append(
        #         (
        #             "gradient_checkpointing_enable",
        #             patched_gradient_checkpointing_enable,
        #         ),
        #     )

        prepared_patching_specs = []
        for spec in patching_specs:
            prepared_patching_specs.append((model,) + spec)

        model_patcher = ModelPatcher(prepared_patching_specs, ignore_missing_attributes=True)
        model_patcher.patch()

        if is_peft_available():
            from peft import PeftModel
            from peft.tuners.tuners_utils import BaseTunerLayer
            from peft.utils import ModulesToSaveWrapper

            if isinstance(model, PeftModel):
                replace_class_in_inheritance_hierarchy(model, PeftModel, NeuronPeftModel)
            else:
                for _, module in model.named_modules():
                    if isinstance(module, (BaseTunerLayer, ModulesToSaveWrapper)):
                        raise ValueError(
                            "It appears that the model is using a PEFT method, please wrap your model with `PeftModel` "
                            "to make it work with `optimum-neuron`"
                        )
