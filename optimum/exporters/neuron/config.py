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
"""
Common Neuron configuration classes that handle most of the features for building model specific
configurations.
"""
import torch
from neuronx_distributed.trace.model_builder import BaseModelInstance
from ...utils import (
    DummyAudioInputGenerator,
    DummyBboxInputGenerator,
    DummyInputGenerator,
    DummySeq2SeqDecoderTextInputGenerator,
    DummySeq2SeqPastKeyValuesGenerator,
    DummyTextInputGenerator,
    DummyVisionInputGenerator,
    logging,
)
from .base import NeuronDefaultConfig


logger = logging.get_logger(__name__)


class TextEncoderNeuronConfig(NeuronDefaultConfig):
    """
    Handles encoder-based text architectures.
    """

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator,)
    INPUT_ARGS = ("batch_size", "sequence_length", ("multiple-choice", "num_choices"))


class VisionNeuronConfig(NeuronDefaultConfig):
    """
    Handles vision architectures.
    """

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVisionInputGenerator,)
    INPUT_ARGS = ("batch_size", "num_channels", "width", "height")


class TextAndVisionNeuronConfig(NeuronDefaultConfig):
    """
    Handles multi-modal text and vision architectures.
    """

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyTextInputGenerator, DummyVisionInputGenerator, DummyBboxInputGenerator)


class AudioNeuronConfig(NeuronDefaultConfig):
    """
    Handles audio architectures.
    """

    DUMMY_INPUT_GENERATOR_CLASSES = (DummyAudioInputGenerator, DummyTextInputGenerator)
    INPUT_ARGS = ("batch_size", "audio_sequence_length")


class TextSeq2SeqNeuronConfig(NeuronDefaultConfig):
    """
    Handles encoder-decoder-based text architectures.
    """

    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyTextInputGenerator,
        DummySeq2SeqDecoderTextInputGenerator,
        DummySeq2SeqPastKeyValuesGenerator,
    )

    def _create_dummy_input_generator_classes(self, **kwargs) -> list["DummyInputGenerator"]:
        dummy_text_input_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[0](
            self.task, self._normalized_config, **kwargs
        )
        dummy_decoder_text_input_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[1](
            self.task,
            self._normalized_config,
            **kwargs,
        )
        dummy_seq2seq_past_key_values_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[2](
            self.task,
            self._normalized_config,
            encoder_sequence_length=dummy_text_input_generator.sequence_length,
            **kwargs,
        )
        dummy_inputs_generators = [
            dummy_text_input_generator,
            dummy_decoder_text_input_generator,
            dummy_seq2seq_past_key_values_generator,
        ]

        return dummy_inputs_generators


class NxDNeuronConfig:
    _STATE_DICT_MODEL_PREFIX = "model."
    _NEW_STATE_DICT_MODEL_PREFIX = ""
    _FUSED_PREFIX = ""
    
    def patch_model_and_prepare_aliases(self, model_or_path, *args):
        base_model_instance = BaseModelInstance(
            self.get_parallel_callable,
            input_output_aliases={},
        )
        return base_model_instance, None

    def get_parallel_callable(self):
        raise NotImplementedError("State-dict update not implemented")

    def checkpoint_loader_fn(self, mmap: bool = False):
        """This function loads the model's state dictionary and weights from the hf model"""

        model_path = getattr(self._config, "_name_or_path")

        def _cast_helper(_model_sd):
            for name, param in _model_sd.items():
                if torch.is_floating_point(param) and param.dtype not in [torch.float8_e4m3fn]:
                    current_dtype = self.float_dtype
                    # only cast floating types
                    if name.endswith("scale"):
                        logger.warning(
                            f"Found {param.dtype} scales, skip converting to {current_dtype}"
                        )
                    elif param.dtype != current_dtype:
                        logger.warning(
                            f"Found {param.dtype} weights in checkpoint: {name}. Will convert to {current_dtype}"
                        )
                        _model_sd[name] = param.to(current_dtype)

        model_sd = self.get_state_dict(model_path)

        if self.float_dtype != torch.float32:
            _cast_helper(model_sd)

        return model_sd
    
    def get_state_dict(self, model_path: str) -> dict:
        """Gets the state dict for this model."""
        from optimum.neuron.models.inference.backend.modules.checkpoint import load_state_dict
        model_sd = load_state_dict(model_path)
        param_name_list = list(model_sd.keys())
        for param_name in param_name_list:
            updated_param_name = param_name
            if param_name.startswith(self._STATE_DICT_MODEL_PREFIX):
                updated_param_name = param_name.replace(
                    self._STATE_DICT_MODEL_PREFIX, self._NEW_STATE_DICT_MODEL_PREFIX, 1
                )
            if param_name.endswith(".weight_scale"):
                updated_param_name = updated_param_name.replace(".weight_scale", ".scale")
            if updated_param_name != param_name:
                model_sd[updated_param_name] = model_sd[param_name]
                del model_sd[param_name]

        model_sd = self.convert_hf_to_neuron_state_dict(model_sd)
        if getattr(self._config, "tie_word_embeddings", False):
            self.update_state_dict_for_tied_weights(model_sd)

        param_name_list = list(model_sd.keys())
        if self._FUSED_PREFIX != "":
            for param_name in param_name_list:
                model_sd[f"{self._FUSED_PREFIX}.{param_name}"] = model_sd[param_name]
                del model_sd[param_name]
        return model_sd
    
    @staticmethod
    def convert_hf_to_neuron_state_dict(state_dict: dict) -> dict:
        """This function should be over-ridden in child classes as needed"""
        return state_dict
    
    @staticmethod
    def update_state_dict_for_tied_weights(state_dict):
        """Implement state_dict update for each model class with tied weights"""
        raise NotImplementedError("State-dict update not implemented")
    
    def get_compiler_args(self) -> str:
        """Gets the Neuron compiler arguments to use when compiling this model."""
        return None
