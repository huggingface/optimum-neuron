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
    "__main__": [
        "infer_shapes_of_diffusers",
        "main_export",
        "normalize_diffusers_input_shapes",
        "get_submodels_and_neuron_configs",
        "load_models_and_neuron_configs",
    ],
    "base": ["NeuronDefaultConfig"],
    "convert": ["export", "export_models", "validate_model_outputs", "validate_models_outputs"],
    "utils": [
        "build_stable_diffusion_components_mandatory_shapes",
        "get_diffusion_models_for_export",
        "replace_stable_diffusion_submodels",
        "get_submodels_for_export_diffusion",
    ],
}

if TYPE_CHECKING:
    from .__main__ import (
        get_submodels_and_neuron_configs,
        infer_shapes_of_diffusers,
        load_models_and_neuron_configs,
        main_export,
        normalize_diffusers_input_shapes,
    )
    from .base import NeuronDefaultConfig
    from .convert import export, export_models, validate_model_outputs, validate_models_outputs
    from .utils import (
        build_stable_diffusion_components_mandatory_shapes,
        get_diffusion_models_for_export,
        get_submodels_for_export_diffusion,
        replace_stable_diffusion_submodels,
    )
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
