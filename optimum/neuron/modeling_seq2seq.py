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
import os
import shutil
import logging
from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModelForSeq2SeqLM, GenerationConfig

from ..exporters.neuron import (
    NeuronConfig,
    main_export,
)
from ..exporters.neuron.model_configs import *  # noqa: F403
from ..exporters.tasks import TasksManager
from ..utils.save_utils import maybe_load_preprocessors
from .generation import NeuronGenerationMixin
from .modeling_base import NeuronBaseModel
from .utils import (
    DECODER_NAME,
    ENCODER_NAME,
    NEURON_FILE_NAME,
    is_neuronx_available,
)


if TYPE_CHECKING:
    from transformers import PretrainedConfig

if is_neuronx_available():
    pass

logger = logging.getLogger(__name__)


class NeuronModelForConditionalGeneration(NeuronBaseModel):
    base_model_prefix = "neuron_model"
    config_name = "config.json"

    def __init__(
        self,
        encoder: torch.jit._script.ScriptModule,
        decoder: torch.jit._script.ScriptModule,
        configs: Optional[Dict[str, "PretrainedConfig"]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        encoder_file_name: Optional[str] = NEURON_FILE_NAME,
        decoder_file_name: Optional[str] = NEURON_FILE_NAME,
        preprocessors: Optional[List] = None,
        neuron_configs: Optional[Dict[str, "NeuronConfig"]] = None,
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ):
        self.encoder = NeuronEncoder(
            encoder,
            self,
            self.configs[ENCODER_NAME],
            self.neuron_configs[ENCODER_NAME],
        )
        self.decoder = NeuronEncoder(
            decoder,
            self,
            self.configs[DECODER_NAME],
            self.neuron_configs[DECODER_NAME],
        )
        self.configs = configs
        self.neuron_configs = neuron_configs
        self.dynamic_batch_size = all(
            neuron_config._config.neuron["dynamic_batch_size"] for neuron_config in self.neuron_configs.values()
        )
        self._attributes_init(model_save_dir, preprocessors, **kwargs)
        self.encoder_file_name = encoder_file_name
        self.decoder_file_name = decoder_file_name
        
        if generation_config is None:
            generation_config = GenerationConfig.from_model_config(self.configs[DECODER_NAME])
        self.generation_config = generation_config

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
        if self.model_and_config_save_paths is None:
            logger.warning(
                "`model_save_paths` is None which means that no path of Neuron model is defined. Nothing will be saved."
            )
            return
        
        save_directory = Path(save_directory)
        if not self.model_and_config_save_paths.get(ENCODER_NAME)[0].is_file():
            self.model_and_config_save_paths.pop(ENCODER_NAME)

        if not self.model_and_config_save_paths.get(DECODER_NAME)[0].is_file():
            self.model_and_config_save_paths.pop(DECODER_NAME)
        
        dst_paths = {
            ENCODER_NAME: save_directory / ENCODER_NAME / encoder_file_name,
            DECODER_NAME: save_directory / DECODER_NAME / decoder_file_name,
        }
        
        model_src_to_dst_path = {
            self.model_and_config_save_paths[model_name][0]: dst_paths[model_name]
            for model_name in set(self.model_and_config_save_paths.keys()).intersection(dst_paths.keys())
        }
        # save
        config_src_to_dst_path = {
            self.model_and_config_save_paths[model_name][1]: dst_paths[model_name].parent / self.config_name
            for model_name in set(self.model_and_config_save_paths.keys()).intersection(dst_paths.keys())
        }
        
        src_paths = list(model_src_to_dst_path.keys()) + list(config_src_to_dst_path.keys())
        dst_paths = list(model_src_to_dst_path.values()) + list(config_src_to_dst_path.values())

        for src_path, dst_path in zip(src_paths, dst_paths):
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            if src_path.is_file():
                shutil.copyfile(src_path, dst_path)
            
        src_paths = [Path(path) for path in self.onnx_paths]
        dst_paths = [save_directory / path.name for path in src_paths]
        
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory)

        self.generation_config.save_pretrained(save_directory)

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: "PretrainedConfig",
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        force_download: bool = False,
        cache_dir: Optional[str] = None,
        encoder_file_name: Optional[str] = NEURON_FILE_NAME,
        decoder_file_name: Optional[str] = NEURON_FILE_NAME,
        subfolder: str = "",
        local_files_only: bool = False,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        model_id = str(model_id)

        if not os.path.isdir(model_id):
            # Downloads all repo's files matching the allowed patterns
            model_id = snapshot_download(
                model_id,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                force_download=force_download,
                ignore_patterns=["*.msgpack", "*.safetensors", "*.bin"],  # only download *.neuron artifacts
            )

        preprocessors = maybe_load_preprocessors(model_id, subfolder=subfolder)

        new_model_save_dir = Path(model_id)

        model_and_config_save_paths = {
            "encoder": (
                new_model_save_dir / ENCODER_NAME / encoder_file_name,
                new_model_save_dir / ENCODER_NAME / cls.config_name,
            ),
            "decoder": (
                new_model_save_dir / DECODER_NAME / decoder_file_name,
                new_model_save_dir / DECODER_NAME / cls.config_name,
            ),
        }

        # Re-build pretrained configs and neuron configs
        configs, neuron_configs = {}, {}
        for name, file_paths in model_and_config_save_paths.items():
            if file_paths[1].is_file():
                model_config = AutoConfig.from_pretrained(file_paths[1])
                configs[name] = model_config
                neuron_configs[name] = cls._neuron_config_init(model_config)

        encoder = cls.load_model(model_and_config_save_paths["encoder"][0])
        decoder = cls.load_model(model_and_config_save_paths["decoder"][0])
        
        # TODO: Debug num_beams unmatched issue
        import pdb
        pdb.set_trace()
        
        if model_save_dir is None:
            model_save_dir = new_model_save_dir
        
        generation_config = None
        try:
            generation_config = GenerationConfig.from_pretrained(
                model_id,
                cache_dir=cache_dir,
                force_download=force_download,
                local_files_only=local_files_only,
                use_auth_token=use_auth_token,
                revision=revision,
                subfolder=os.path.join(subfolder, DECODER_NAME),
            )
        except OSError:
            logger.info("Generation config file not found, using a generation config created from the model config.")

        return cls(
            encoder=encoder,
            decoder=decoder,
            configs=configs,
            model_save_dir=model_save_dir,
            encoder_file_name=encoder_file_name,
            decoder_file_name=decoder_file_name,
            preprocessors=preprocessors,
            neuron_configs=neuron_configs,
            generation_config=generation_config,
        )

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
        **kwargs_shapes,
    ) -> "NeuronModelForConditionalGeneration":
        if task is None:
            task = TasksManager.infer_task_from_model(cls.auto_model_class)

        # Get compilation arguments
        auto_cast_type = None if auto_cast is None else auto_cast_type
        compiler_kwargs = {
            "auto_cast": auto_cast,
            "auto_cast_type": auto_cast_type,
            "disable_fast_relayout": disable_fast_relayout,
            "disable_fallback": disable_fallback,
        }

        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)

        main_export(
            model_name_or_path=model_id,
            output=save_dir_path,
            compiler_kwargs=compiler_kwargs,
            task=task,
            dynamic_batch_size=dynamic_batch_size,
            cache_dir=cache_dir,
            trust_remote_code=trust_remote_code,
            subfolder=subfolder,
            revision=revision,
            force_download=force_download,
            local_files_only=local_files_only,
            use_auth_token=use_auth_token,
            do_validation=False,
            **kwargs_shapes,
        )

        return cls._from_pretrained(
            model_id=save_dir_path,
            config=config,
            model_save_dir=save_dir,
        )


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
