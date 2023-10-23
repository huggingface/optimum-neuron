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
"""NeuroModelForXXX classes for seq2seq models' inference on neuron devices."""
from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import torch
from transformers import AutoModelForSeq2SeqLM

from .generation import NeuronGenerationMixin
from .modeling_base import NeuronBaseModel, NeuronConfig
from .utils import (
    NEURON_FILE_NAME,
    is_neuronx_available,
)


if TYPE_CHECKING:
    from transformers import PretrainedConfig

if is_neuronx_available():
    pass


class NeuronModelForConditionalGeneration(NeuronBaseModel):
    base_model_prefix = "neuron_model"

    def __init__(
        self,
        encoder: torch.jit._script.ScriptModule,
        decoder: torch.jit._script.ScriptModule,
        config: "PretrainedConfig",
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        model_file_name: Optional[str] = None,
        preprocessors: Optional[List] = None,
        neuron_config: Optional["NeuronConfig"] = None,
        **kwargs,
    ):
        pass

    @staticmethod
    def load_model(
        encoder_path: Union[str, Path],
        decoder_path: Optional[Union[str, Path]] = None,
        device_ids: Optional[List[int]] = None,
        dynamic_batch_size: bool = False,
    ):
        pass

    def _save_pretrained(
        self,
        save_directory: Union[str, Path],
        encoder_file_name: str = NEURON_FILE_NAME,
        decoder_file_name: str = NEURON_FILE_NAME,
    ):
        """
        Saves the model encoder and decoder as well as their configuration files to a
        directory, so that it can be re-loaded using the
        [`~optimum.neuron.modeling_seq2seq.NeuronModelForSeq2SeqLM.from_pretrained`] class method.

        Args:
            save_directory (`Union[str, Path`]):
                The directory where to save the model files.
        """
        pass

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: Dict[str, Any],
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        encoder_file_name: Optional[str] = NEURON_FILE_NAME,
        decoder_file_name: Optional[str] = NEURON_FILE_NAME,
        subfolder: str = "",
        local_files_only: bool = False,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        device_ids: Optional[List[int]] = None,
        **kwargs,
    ):
        pass

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        config: "PretrainedConfig",
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: str = "main",
        force_download: bool = True,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        task: Optional[str] = None,
        auto_cast: Optional[str] = "matmul",
        auto_cast_type: Optional[str] = "bf16",
        disable_fast_relayout: Optional[bool] = False,
        disable_fallback: bool = False,
        dynamic_batch_size: bool = False,
        device_ids: Optional[List[int]] = None,
    ) -> "NeuronModelForConditionalGeneration":
        pass


class NeuronModelForSeq2SeqLM(NeuronModelForConditionalGeneration, NeuronGenerationMixin):
    auto_model_class = AutoModelForSeq2SeqLM
    main_input_name = "input_ids"


class _NeuronSeq2SeqModelPart:
    """
    For Seq2Seq architecture, we usually compile it to multiple neuron models. Each represents a part of the model.
    """

    def __init__(
        self,
        model: torch.jit._script.ScriptModule,
        parent_model: NeuronBaseModel,
        config: Optional["PretrainedConfig"] = None,
        neuron_config: Optional["NeuronConfig"] = None,
        model_type: str = "encoder",
        device: Optional[int] = None,
    ):
        self.model = model
        self.parent_model = parent_model
        self.config = config
        self.neuron_config = neuron_config
        self.model_type = model_type
        self.device = device

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class NeuronEncoder(_NeuronSeq2SeqModelPart):
    """
    Encoder part of the encoder-decoder model for Neuron inference. (Actually it's a monolith of encoder + decoder without past_key_values to workaround the control flow in the decoder).
    """

    def __init__(
        self,
        model: torch.jit._script.ScriptModule,
        parent_model: NeuronBaseModel,
        config: Optional["PretrainedConfig"] = None,
        neuron_config: Optional[Dict[str, str]] = None,
    ):
        super().__init__(model, parent_model, config, neuron_config, "encoder")

    def forward(self, input_ids: torch.LongTensor, attention_mask: torch.FloatTensor):
        inputs = (
            input_ids,
            attention_mask,
        )
        outputs = self.model(*inputs)
        return outputs


class NeuronDecoder(_NeuronSeq2SeqModelPart):
    """
    Decoder part of the encoder-decoder model for Neuron inference. (Actually it's decoder with past_key_values).
    """

    def __init__(
        self,
        model: torch.jit._script.ScriptModule,
        parent_model: NeuronBaseModel,
        config: Optional["PretrainedConfig"] = None,
        neuron_config: Optional[Dict[str, str]] = None,
    ):
        super().__init__(model, parent_model, config, neuron_config, "decoder")

    def forward(
        self,
        input_ids: torch.LongTensor,
        decoder_attention_mask: torch.FloatTensor,
        encoder_hidden_states: torch.FloatTensor,
        encoder_attention_mask: torch.FloatTensor,
        beam_idx: torch.LongTensor,
        beam_scores: torch.FloatTensor,
    ):
        inputs = (
            input_ids,
            decoder_attention_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            beam_idx,
            beam_scores,
        )
        outputs = self.model(*inputs)
        return outputs
