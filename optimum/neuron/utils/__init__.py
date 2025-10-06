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

from typing import TYPE_CHECKING

from transformers.utils import _LazyModule


_import_structure = {
    "argument_utils": [
        "LoRAAdapterArguments",
        "IPAdapterArguments",
        "ImageEncoderArguments",
        "InputShapesArguments",
        "NeuronArgumentParser",
        "convert_neuronx_compiler_args_to_neuron",
        "store_compilation_config",
    ],
    "constant": [
        "DECODER_NAME",
        "DIFFUSION_MODEL_TEXT_ENCODER_2_NAME",
        "DIFFUSION_MODEL_TEXT_ENCODER_NAME",
        "DIFFUSION_MODEL_IMAGE_ENCODER_NAME",
        "DIFFUSION_MODEL_UNET_NAME",
        "DIFFUSION_MODEL_TRANSFORMER_NAME",
        "DIFFUSION_MODEL_VAE_DECODER_NAME",
        "DIFFUSION_MODEL_VAE_ENCODER_NAME",
        "DIFFUSION_MODEL_CONTROLNET_NAME",
        "WEIGHTS_INDEX_NAME",
        "SAFE_WEIGHTS_INDEX_NAME",
        "ENCODER_NAME",
        "NEURON_FILE_NAME",
    ],
    "import_utils": [
        "is_accelerate_available",
        "is_neuron_available",
        "is_neuronx_available",
        "is_torch_neuronx_available",
        "is_trl_available",
    ],
    "input_generators": [
        "DTYPE_MAPPER",
        "DummyBeamValuesGenerator",
        "DummyMaskedPosGenerator",
        "DummyControNetInputGenerator",
        "ASTDummyAudioInputGenerator",
        "DummyIPAdapterInputGenerator",
        "DummyFluxTransformerRotaryEmbGenerator",
        "DummyFluxKontextTransformerRotaryEmbGenerator",
        "DummyTimestepInputGenerator",
        "WhisperDummyTextInputGenerator",
    ],
    "misc": [
        "DiffusersPretrainedConfig",
        "check_if_weights_replacable",
        "get_stable_diffusion_configs",
        "is_main_worker",
        "is_precompilation",
        "replace_weights",
        "get_checkpoint_shard_files",
    ],
    "model_utils": ["get_tied_parameters_dict", "tie_parameters", "saved_model_in_temporary_directory"],
    "optimization_utils": [
        "get_attention_scores_sd",
        "get_attention_scores_sdxl",
        "neuron_scaled_dot_product_attention",
    ],
    "patching": [
        "DynamicPatch",
        "ModelPatcher",
        "Patcher",
        "patch_everywhere",
        "patch_within_function",
        "replace_class_in_inheritance_hierarchy",
    ],
    "instance": [
        "SUPPORTED_INSTANCE_TYPES",
        "current_instance_type",
        "normalize_instance_type",
        "define_target_instance_type",
    ],
    "trl_utils": ["NeuronSFTConfig", "NeuronORPOConfig"],
}

if TYPE_CHECKING:
    from .argument_utils import (
        ImageEncoderArguments,
        InputShapesArguments,
        IPAdapterArguments,
        LoRAAdapterArguments,
        NeuronArgumentParser,
        convert_neuronx_compiler_args_to_neuron,
        store_compilation_config,
    )
    from .constant import (
        DECODER_NAME,
        DIFFUSION_MODEL_CONTROLNET_NAME,
        DIFFUSION_MODEL_IMAGE_ENCODER_NAME,
        DIFFUSION_MODEL_TEXT_ENCODER_2_NAME,
        DIFFUSION_MODEL_TEXT_ENCODER_NAME,
        DIFFUSION_MODEL_TRANSFORMER_NAME,
        DIFFUSION_MODEL_UNET_NAME,
        DIFFUSION_MODEL_VAE_DECODER_NAME,
        DIFFUSION_MODEL_VAE_ENCODER_NAME,
        ENCODER_NAME,
        NEURON_FILE_NAME,
        SAFE_WEIGHTS_INDEX_NAME,
        WEIGHTS_INDEX_NAME,
    )
    from .import_utils import (
        is_accelerate_available,
        is_neuron_available,
        is_neuronx_available,
        is_torch_neuronx_available,
        is_trl_available,
    )
    from .input_generators import (
        DTYPE_MAPPER,
        ASTDummyAudioInputGenerator,
        DummyBeamValuesGenerator,
        DummyControNetInputGenerator,
        DummyFluxKontextTransformerRotaryEmbGenerator,
        DummyFluxTransformerRotaryEmbGenerator,
        DummyIPAdapterInputGenerator,
        DummyMaskedPosGenerator,
        DummyTimestepInputGenerator,
        WhisperDummyTextInputGenerator,
    )
    from .instance import (
        SUPPORTED_INSTANCE_TYPES,
        current_instance_type,
        define_target_instance_type,
        normalize_instance_type,
    )
    from .misc import (
        DiffusersPretrainedConfig,
        check_if_weights_replacable,
        get_checkpoint_shard_files,
        get_stable_diffusion_configs,
        is_main_worker,
        is_precompilation,
        replace_weights,
    )
    from .model_utils import get_tied_parameters_dict, saved_model_in_temporary_directory, tie_parameters
    from .optimization_utils import (
        get_attention_scores_sd,
        get_attention_scores_sdxl,
        neuron_scaled_dot_product_attention,
    )
    from .patching import (
        DynamicPatch,
        ModelPatcher,
        Patcher,
        patch_everywhere,
        patch_within_function,
        replace_class_in_inheritance_hierarchy,
    )
    from .trl_utils import NeuronORPOConfig, NeuronSFTConfig
else:
    import sys

    sys.modules[__name__] = _LazyModule(
        __name__,
        globals()["__file__"],
        _import_structure,
        module_spec=__spec__,
    )
