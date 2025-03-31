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
        "ENCODER_NAME",
        "NEURON_FILE_NAME",
    ],
    "hub_cache_utils": [
        "ModelCacheEntry",
        "get_hub_cached_entries",
        "get_hub_cached_models",
        "hub_neuronx_cache",
        "synchronize_hub_cache",
    ],
    "import_utils": [
        "is_accelerate_available",
        "is_neuron_available",
        "is_neuronx_available",
        "is_neuronx_distributed_available",
        "is_torch_neuronx_available",
        "is_torch_xla_available",
        "is_transformers_neuronx_available",
        "is_trl_available",
    ],
    "input_generators": [
        "DummyBeamValuesGenerator",
        "DummyMaskedPosGenerator",
        "DummyControNetInputGenerator",
        "ASTDummyAudioInputGenerator",
        "DummyIPAdapterInputGenerator",
        "WhisperDummyTextInputGenerator",
    ],
    "misc": [
        "DiffusersPretrainedConfig",
        "check_if_weights_replacable",
        "get_stable_diffusion_configs",
        "is_main_worker",
        "is_precompilation",
        "replace_weights",
        "map_torch_dtype",
    ],
    "model_utils": ["get_tied_parameters_dict", "tie_parameters"],
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
    "peft_utils": ["NeuronPeftModel", "get_peft_model"],
    "training_utils": [
        "is_model_officially_supported",
        "patch_transformers_for_neuron_sdk",
    ],
    "trl_utils": ["NeuronSFTConfig", "NeuronORPOConfig"],
    "doc": [
        "_TOKENIZER_FOR_DOC",
        "_PROCESSOR_FOR_IMAGE",
        "_GENERIC_PROCESSOR",
        "NEURON_FEATURE_EXTRACTION_EXAMPLE",
        "NEURON_MASKED_LM_EXAMPLE",
        "NEURON_SEQUENCE_CLASSIFICATION_EXAMPLE",
        "NEURON_TOKEN_CLASSIFICATION_EXAMPLE",
        "NEURON_QUESTION_ANSWERING_EXAMPLE",
        "NEURON_MULTIPLE_CHOICE_EXAMPLE",
        "NEURON_IMAGE_CLASSIFICATION_EXAMPLE",
        "NEURON_IMAGE_CLASSIFICATION_PIPELINE_EXAMPLE",
        "NEURON_SEMANTIC_SEGMENTATION_EXAMPLE",
        "NEURON_SEMANTIC_SEGMENTATION_PIPELINE_EXAMPLE",
        "NEURON_OBJECT_DETECTION_EXAMPLE",
        "NEURON_OBJECT_DETECTION_PIPELINE_EXAMPLE",
        "NEURON_AUDIO_CLASSIFICATION_EXAMPLE",
        "NEURON_AUDIO_CLASSIFICATION_PIPELINE_EXAMPLE",
        "NEURON_AUDIO_FRAME_CLASSIFICATION_EXAMPLE",
        "NEURON_CTC_EXAMPLE",
        "NEURON_CTC_PIPELINE_EXAMPLE",
        "NEURON_AUDIO_XVECTOR_EXAMPLE",
        "NEURON_SENTENCE_TRANSFORMERS_EXAMPLE",
        "NEURON_TEXT_GENERATION_EXAMPLE",
        "NEURON_TEXT_INPUTS_DOCSTRING",
        "NEURON_IMAGE_INPUTS_DOCSTRING",
        "NEURON_AUDIO_INPUTS_DOCSTRING",
        "NEURON_CAUSALLM_INPUTS_DOCSTRING",
        "NEURON_MODEL_START_DOCSTRING",
        "NEURON_CAUSALLM_MODEL_START_DOCSTRING",
    ],
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
    )
    from .doc import (
        _GENERIC_PROCESSOR,
        _PROCESSOR_FOR_IMAGE,
        _TOKENIZER_FOR_DOC,
        NEURON_AUDIO_CLASSIFICATION_EXAMPLE,
        NEURON_AUDIO_CLASSIFICATION_PIPELINE_EXAMPLE,
        NEURON_AUDIO_FRAME_CLASSIFICATION_EXAMPLE,
        NEURON_AUDIO_INPUTS_DOCSTRING,
        NEURON_AUDIO_XVECTOR_EXAMPLE,
        NEURON_CAUSALLM_INPUTS_DOCSTRING,
        NEURON_CAUSALLM_MODEL_START_DOCSTRING,
        NEURON_CTC_EXAMPLE,
        NEURON_CTC_PIPELINE_EXAMPLE,
        NEURON_FEATURE_EXTRACTION_EXAMPLE,
        NEURON_IMAGE_CLASSIFICATION_EXAMPLE,
        NEURON_IMAGE_CLASSIFICATION_PIPELINE_EXAMPLE,
        NEURON_IMAGE_INPUTS_DOCSTRING,
        NEURON_MASKED_LM_EXAMPLE,
        NEURON_MODEL_START_DOCSTRING,
        NEURON_MULTIPLE_CHOICE_EXAMPLE,
        NEURON_OBJECT_DETECTION_EXAMPLE,
        NEURON_OBJECT_DETECTION_PIPELINE_EXAMPLE,
        NEURON_QUESTION_ANSWERING_EXAMPLE,
        NEURON_SEMANTIC_SEGMENTATION_EXAMPLE,
        NEURON_SEMANTIC_SEGMENTATION_PIPELINE_EXAMPLE,
        NEURON_SENTENCE_TRANSFORMERS_EXAMPLE,
        NEURON_SEQUENCE_CLASSIFICATION_EXAMPLE,
        NEURON_TEXT_GENERATION_EXAMPLE,
        NEURON_TEXT_INPUTS_DOCSTRING,
        NEURON_TOKEN_CLASSIFICATION_EXAMPLE,
    )
    from .hub_cache_utils import (
        ModelCacheEntry,
        get_hub_cached_entries,
        get_hub_cached_models,
        hub_neuronx_cache,
        synchronize_hub_cache,
    )
    from .import_utils import (
        is_accelerate_available,
        is_neuron_available,
        is_neuronx_available,
        is_neuronx_distributed_available,
        is_torch_neuronx_available,
        is_torch_xla_available,
        is_transformers_neuronx_available,
        is_trl_available,
    )
    from .input_generators import (
        ASTDummyAudioInputGenerator,
        DummyBeamValuesGenerator,
        DummyControNetInputGenerator,
        DummyIPAdapterInputGenerator,
        DummyMaskedPosGenerator,
        WhisperDummyTextInputGenerator,
    )
    from .misc import (
        DiffusersPretrainedConfig,
        check_if_weights_replacable,
        get_stable_diffusion_configs,
        is_main_worker,
        is_precompilation,
        map_torch_dtype,
        replace_weights,
    )
    from .model_utils import get_tied_parameters_dict, tie_parameters
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
    from .peft_utils import NeuronPeftModel, get_peft_model
    from .training_utils import (
        is_model_officially_supported,
        patch_transformers_for_neuron_sdk,
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
