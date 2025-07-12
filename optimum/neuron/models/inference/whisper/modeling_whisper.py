# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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
"""Whisper model on Neuron devices."""

import logging
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any

import torch
from transformers import GenerationConfig, WhisperForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput
from transformers.utils import ModelOutput

from ....exporters.neuron import (
    NeuronDefaultConfig,
)
from ...modeling_seq2seq import NeuronModelForConditionalGeneration, _NeuronSeq2SeqModelPart
from ...modeling_traced import NeuronTracedModel
from ...utils import (
    NEURON_FILE_NAME,
    is_neuronx_available,
)
from ...utils.doc import (
    NEURON_AUDIO_SEQ2SEQ_INPUTS_DOCSTRING,
    NEURON_SEQ2SEQ_MODEL_START_DOCSTRING,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)


if TYPE_CHECKING:
    from transformers import PretrainedConfig

if is_neuronx_available():
    pass


logger = logging.getLogger(__name__)


class DummyLayer:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __call__(self, x):
        return x


class NeuronWhisperEncoder(_NeuronSeq2SeqModelPart):
    """
    Encoder and the 1st forward of decoder+language head.
    """

    main_input_name = "input_features"

    def __init__(
        self,
        model: torch.jit._script.ScriptModule,
        parent_model: NeuronTracedModel,
        config: "PretrainedConfig | None" = None,
        neuron_config: dict[str, str] | None = None,
    ):
        super().__init__(model, parent_model, config, neuron_config, "encoder")
        stride = getattr(self.config, "stride", [1, 2])
        self.conv1 = DummyLayer(stride=[stride[0]])
        self.conv2 = DummyLayer(stride=[stride[1]])

    def forward(
        self,
        input_features: torch.FloatTensor,
        decoder_input_ids: torch.LongTensor | None = None,
        **kwargs,
    ):
        prepare_encoder_decoder_kwargs_for_generation = False
        if decoder_input_ids is None:
            decoder_input_ids = torch.full(
                (self.neuron_config.batch_size, 1), self.config.decoder_start_token_id, dtype=torch.long
            )
            prepare_encoder_decoder_kwargs_for_generation = True
        outputs = self.model(input_features, decoder_input_ids)
        if prepare_encoder_decoder_kwargs_for_generation:
            return BaseModelOutput(last_hidden_state=outputs[1])
        else:
            return outputs


class NeuronWhisperDecoder(_NeuronSeq2SeqModelPart):
    """
    Decoder with output embedding of the whisper model for Neuron inference.
    """

    def __init__(
        self,
        model: torch.jit._script.ScriptModule,
        parent_model: NeuronTracedModel,
        config: "PretrainedConfig | None" = None,
        neuron_config: dict[str, str] | None = None,
    ):
        super().__init__(model, parent_model, config, neuron_config, "decoder")

    def forward(
        self,
        decoder_input_ids: torch.LongTensor | None,
        encoder_hidden_states: torch.FloatTensor | None,
        **kwargs,
    ):
        inputs = (
            decoder_input_ids,
            encoder_hidden_states,
        )
        outputs = self.model(*inputs)
        return (outputs, encoder_hidden_states)


class NeuronWhisperModel:
    def __init__(self, encoder: NeuronWhisperEncoder, decoder: NeuronWhisperDecoder):
        self.encoder = encoder
        self.decoder = decoder


@add_start_docstrings(
    """
    Whisper Neuron model with a language modeling head that can be used for automatic speech recognition.
    """,
    NEURON_SEQ2SEQ_MODEL_START_DOCSTRING,
)
class NeuronWhisperForConditionalGeneration(NeuronModelForConditionalGeneration, WhisperForConditionalGeneration):
    auto_model_class = WhisperForConditionalGeneration
    main_input_name = "input_features"
    encoder_class = NeuronWhisperEncoder
    decoder_class = NeuronWhisperDecoder

    def __init__(
        self,
        encoder: torch.jit._script.ScriptModule,
        decoder: torch.jit._script.ScriptModule,
        config: "PretrainedConfig",
        model_save_dir: str | Path | TemporaryDirectory | None = None,
        encoder_file_name: str | None = NEURON_FILE_NAME,
        decoder_file_name: str | None = NEURON_FILE_NAME,
        preprocessors: list | None = None,
        neuron_configs: dict[str, "NeuronDefaultConfig"] | None = None,
        configs: dict[str, "PretrainedConfig"] | None = None,
        generation_config: GenerationConfig | None = None,
        **kwargs,
    ):
        super().__init__(
            encoder,
            decoder,
            config,
            model_save_dir,
            encoder_file_name,
            decoder_file_name,
            preprocessors,
            neuron_configs,
            configs,
            generation_config,
            **kwargs,
        )
        self.model = NeuronWhisperModel(self.encoder, self.decoder)

    @property
    def device(self):
        return torch.device("cpu")

    def get_encoder(self) -> "NeuronWhisperEncoder":
        return self.encoder

    def _update_model_kwargs_for_generation(
        self,
        outputs: ModelOutput,
        model_kwargs: dict[str, Any],
        is_encoder_decoder: bool = False,
        num_new_tokens: int = 1,
    ) -> dict[str, Any]:
        # Override "use_cache" to False, since whisper with cache is not yet supported for neuron.
        model_kwargs["use_cache"] = False

        model_kwargs = super()._update_model_kwargs_for_generation(
            outputs, model_kwargs, is_encoder_decoder, num_new_tokens
        )

        return model_kwargs

    @add_start_docstrings_to_model_forward(NEURON_AUDIO_SEQ2SEQ_INPUTS_DOCSTRING)
    def forward(
        self,
        input_features: torch.FloatTensor | None = None,
        decoder_input_ids: torch.LongTensor | None = None,
        encoder_outputs: tuple[torch.FloatTensor] | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor] | Seq2SeqLMOutput:
        if encoder_outputs is None:
            lm_logits, encoder_last_hidden_state = self.encoder(
                input_features=input_features, decoder_input_ids=decoder_input_ids
            )
        else:
            # pad `decoder_input_ids` to the sequence length of the compilation
            decoder_input_ids_length = decoder_input_ids.shape[1]
            pad_size = torch.as_tensor(self.neuron_configs["decoder"].sequence_length - decoder_input_ids_length)
            decoder_input_ids = torch.nn.functional.pad(
                decoder_input_ids, (0, pad_size), "constant", self.preprocessors[0].pad_token_id
            )

            lm_logits, encoder_last_hidden_state = self.decoder(
                decoder_input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs[0],
            )
            # unpad
            lm_logits = lm_logits[:, :decoder_input_ids_length, :]

        return Seq2SeqLMOutput(
            logits=lm_logits,
            encoder_last_hidden_state=encoder_last_hidden_state,
        )
