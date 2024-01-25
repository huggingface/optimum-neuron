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
from typing import List

from ...utils import (
    DummyBboxInputGenerator,
    DummyInputGenerator,
    DummySeq2SeqDecoderTextInputGenerator,
    DummySeq2SeqPastKeyValuesGenerator,
    DummyTextInputGenerator,
    DummyVisionInputGenerator,
    logging,
)
from .base import NeuronDecoderConfig, NeuronDefaultConfig


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


class TextNeuronDecoderConfig(NeuronDecoderConfig):
    """
    Handles text decoder architectures.
    """

    pass


class TextSeq2SeqNeuronConfig(NeuronDefaultConfig):
    """
    Handles encoder-decoder-based text architectures.
    """

    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyTextInputGenerator,
        DummySeq2SeqDecoderTextInputGenerator,
        DummySeq2SeqPastKeyValuesGenerator,
    )

    @property
    def inputs(self) -> List[str]:
        common_inputs = []
        # encoder + decoder without past
        if "encoder" in self.MODEL_TYPE:
            common_inputs = ["input_ids", "attention_mask"]
        # decoder with past
        if "decoder" in self.MODEL_TYPE:
            common_inputs = [
                "decoder_input_ids",
                "decoder_attention_mask",
                "encoder_hidden_states",
                "attention_mask",  # TODO: replace with `encoder_attention_mask` after optimum 1.14 release
            ]

        return common_inputs

    @property
    def outputs(self) -> List[str]:
        common_outputs = []
        # encoder + decoder without past
        if "encoder" in self.MODEL_TYPE:
            common_outputs = (
                [f"present.{idx}.self.key" for idx in range(self._config.num_decoder_layers)]
                + [f"present.{idx}.self.value" for idx in range(self._config.num_decoder_layers)]
                + [f"present.{idx}.cross.key" for idx in range(self._config.num_decoder_layers)]
                + [f"present.{idx}.cross.value" for idx in range(self._config.num_decoder_layers)]
            )
        # decoder with past
        if "decoder" in self.MODEL_TYPE:
            beam_outputs = (
                ["next_token_scores", "next_tokens", "next_indices"] if self.num_beams > 1 else ["next_tokens"]
            )
            common_outputs = (
                beam_outputs
                + [f"past.{idx}.self.key" for idx in range(self._config.num_decoder_layers)]
                + [f"past.{idx}.self.value" for idx in range(self._config.num_decoder_layers)]
                + [f"past.{idx}.cross.key" for idx in range(self._config.num_decoder_layers)]
                + [f"past.{idx}.cross.value" for idx in range(self._config.num_decoder_layers)]
            )

            if self.output_hidden_states:
                # Flatten hidden states of all layers
                common_outputs += [
                    f"decoder_hidden_state.{idx}" for idx in range(self._config.num_decoder_layers + 1)
                ]  # +1 for the embedding layer

            if self.output_attentions:
                # Flatten attentions tensors of all attention layers
                common_outputs += [f"decoder_attention.{idx}" for idx in range(self._config.num_decoder_layers)]
                if getattr(self._config, "is_encoder_decoder", False) is True:
                    common_outputs += [f"cross_attention.{idx}" for idx in range(self._config.num_decoder_layers)]

        return common_outputs

    def _create_dummy_input_generator_classes(self, **kwargs) -> List["DummyInputGenerator"]:
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
