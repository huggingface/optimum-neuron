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
"""NeuroStableDiffusionPipeline class for inference of diffusion models on neuron devices."""
from abc import abstractmethod
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, Union, List

import torch
from diffusers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler, StableDiffusionPipeline
from transformers import CLIPFeatureExtractor, CLIPTokenizer

from ..pipelines.diffusers.pipeline_stable_diffusion import StableDiffusionPipelineMixin
from ..utils import (
    DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER,
    DIFFUSION_MODEL_UNET_SUBFOLDER,
    DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER,
)

from ..exporters.neuron import NeuronConfig
from .modeling_base import NeuronBaseModel
from .utils import NEURON_FILE_NAME


DIFFUSION_MODEL_VAE_POST_QUANT_CONV_SUBFOLDER = ""


class NeuronStableDiffusionPipeline(NeuronBaseModel, StableDiffusionPipelineMixin):
    auto_model_class = StableDiffusionPipeline
    main_input_name = "input_ids"
    config_name = "model_index.json"

    def __init__(
        self,
        text_encoder: torch.jit._script.ScriptModule,
        vae_decoder: torch.jit._script.ScriptModule,
        unet: torch.jit._script.ScriptModule,
        vae_post_quant_conv: torch.jit._script.ScriptModule,
        config: Dict[str, Any],
        tokenizer: CLIPTokenizer,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        feature_extractor: Optional[CLIPFeatureExtractor] = None,
        neuron_config: Optional[List["NeuronConfig"]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        model_file_paths: Optional[List[str, Path, TemporaryDirectory]] = None,
    ):
        """
        Args:
            text_encoder (`torch.jit._script.ScriptModule`):
                The Neuron TorchScript module associated to the text encoder.
            vae_decoder (`torch.jit._script.ScriptModule`):
                The Neuron TorchScript module associated to the VAE decoder.
            unet (`torch.jit._script.ScriptModule`):
                The Neuron TorchScript module associated to the U-NET.
            vae_post_quant_conv (`torch.jit._script.ScriptModule`):
                The Neuron TorchScript module associated to the VAE post quant convolutional layer.
            config (`Dict[str, Any]`):
                A config dictionary from which the model components will be instantiated. Make sure to only load
                configuration files of compatible classes.
            tokenizer (`CLIPTokenizer`):
                Tokenizer of class
                [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
            scheduler (`Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler]`):
                A scheduler to be used in combination with the U-NET component to denoise the encoded image latents.
            feature_extractor (`Optional[CLIPFeatureExtractor]`, defaults to `None`):
                A model extracting features from generated images to be used as inputs for the `safety_checker`
            neuron_config  (Optional["NeuronConfig"], defaults to `None`):
                A list of Neuron configurations.
            model_save_dir (`Optional[Union[str, Path, TemporaryDirectory]]`, defaults to `None`):
                The directory under which the exported Neuron models were saved.
        """

        self._internal_dict = config
        self.neuron_config = self._neuron_config_init(self.config) if neuron_config is None else neuron_config

        self.text_encoder = NeuronModelTextEncoder(text_encoder, self)
        self.vae_decoder = NeuronModelVaeDecoder(vae_decoder, self)
        self.unet = NeuronModelUnet(unet, self)
        self.vae_post_quant_conv = NeuronModelUnet(vae_post_quant_conv, self)

        self.text_encoder_model_path = Path(model_file_paths[0]) 
        self.vae_decoder_model_path = Path(model_file_paths[1])      
        self.unet_model_path = Path(model_file_paths[2])
        self.vae_post_quant_conv_model_path = Path(model_file_paths[3])  


        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.feature_extractor = feature_extractor
        self.safety_checker = None
        sub_models = {
            DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER: self.text_encoder,
            DIFFUSION_MODEL_UNET_SUBFOLDER: self.unet,
            DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER: self.vae_decoder,
            DIFFUSION_MODEL_VAE_POST_QUANT_CONV_SUBFOLDER: self.vae_post_quant_conv,
        }
        for name in sub_models.keys():
            self._internal_dict[name] = ("optimum", sub_models[name].__class__.__name__)
        self._internal_dict.pop("vae", None)
    

    def _save_pretrained(
        self,
        save_directory: Union[str, Path],
        text_encoder_file_name: str = NEURON_FILE_NAME,
        vae_decoder_file_name: str = NEURON_FILE_NAME,
        unet_file_name: str = NEURON_FILE_NAME,
        vae_post_quant_conv_file_name: str = NEURON_FILE_NAME,
        **kwargs,
    ):
        """
        Saves the model to the serialized format optimized for Neuron devices.
        """
        save_directory = Path(save_directory)
        src_to_dst_path = {
            self.text_encoder_model_path: save_directory / DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER / text_encoder_file_name,
            self.vae_decoder_model_path: save_directory / DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER / vae_decoder_file_name,
            self.unet_model_path: save_directory / DIFFUSION_MODEL_UNET_SUBFOLDER / unet_file_name,
            self.vae_post_quant_conv_model_path: save_directory / DIFFUSION_MODEL_VAE_POST_QUANT_CONV_SUBFOLDER / vae_post_quant_conv_file_name,
        }

        src_paths = list(src_to_dst_path.keys())
        dst_paths = list(src_to_dst_path.values())

        for src_path, dst_path in zip(src_paths, dst_paths):
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_path, dst_path)

        self.tokenizer.save_pretrained(save_directory.joinpath("tokenizer"))
        self.scheduler.save_pretrained(save_directory.joinpath("scheduler"))
        if self.feature_extractor is not None:
            self.feature_extractor.save_pretrained(save_directory.joinpath("feature_extractor"))

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: Dict[str, Any],
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        text_encoder_file_name: Optional[str] = None,
        vae_decoder_file_name: Optional[str] = None,
        unet_file_name: Optional[str] = None,
        vae_post_quant_conv_file_name: Optional[str] = None,
        local_files_only: bool = False,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        model_id = str(model_id)
    
    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        config: Optional[str] = None,
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: str = "main",
        force_download: bool = True,
        cache_dir: Optional[str] = None,
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        task: Optional[str] = None,
    ) -> "NeuronStableDiffusionPipeline":
        if task is None:
            task = cls._auto_model_to_task(cls.auto_model_class)

        save_dir = TemporaryDirectory()
        save_dir_path = Path(save_dir.name)
    

    def __call__(self, *args, **kwargs):
        return StableDiffusionPipelineMixin.__call__(self, *args, **kwargs)

    @classmethod
    def _load_config(cls, config_name_or_path: Union[str, os.PathLike], **kwargs):
        return cls.load_config(config_name_or_path, **kwargs)

    def _save_config(self, save_directory):
        self.save_config(save_directory)
        

class NeuronDiffusionModelPart:
    """
    For multi-file Neuron models, represents a part of the model.
    """

    def __init__(
        self,
        model: torch.jit._script.ScriptModule,
        parent_model: NeuronBaseModel,
        neuron_config: Optional["NeuronConfig"] = None,
        model_type: str = "encoder",
    ):
        self.model = model
        self.parent_model = parent_model
        self.input_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.inputs)}
        self.output_names = {key.get_any_name(): idx for idx, key in enumerate(self.model.outputs)}

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class NeuronModelTextEncoder(NeuronDiffusionModelPart):
    def __init__(
        self,
        model: torch.jit._script.ScriptModule,
        parent_model: NeuronBaseModel,
        neuron_config: Optional[Dict[str, str]] = None,
    ):
        super().__init__(model, parent_model, neuron_config, "text_encoder")

    def forward(self, input_ids: torch.Tensor):
        inputs = (input_ids,)
        outputs = self.model(*inputs)
        return outputs


class NeuronModelVaeDecoder(NeuronDiffusionModelPart):
    def __init__(
        self,
        model: torch.jit._script.ScriptModule,
        parent_model: NeuronBaseModel,
        neuron_config: Optional[Dict[str, str]] = None,
    ):
        super().__init__(model, parent_model, neuron_config, "vae_decoder")

    def forward(self, latent_sample: torch.Tensor):
        inputs = (latent_sample,)
        outputs = self.model(*inputs)
        return outputs


class NeuronModelUnet(NeuronDiffusionModelPart):
    def __init__(
        self,
        model: torch.jit._script.ScriptModule,
        parent_model: NeuronBaseModel,
        neuron_config: Optional[Dict[str, str]] = None,
    ):
        super().__init__(model, parent_model, neuron_config, "unet")

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor):
        inputs = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }
        outputs = self.model(*inputs)
        return outputs

class NeuronModelVaePostQuantConv(NeuronDiffusionModelPart):
    def __init__(
        self,
        model: torch.jit._script.ScriptModule,
        parent_model: NeuronBaseModel,
        neuron_config: Optional[Dict[str, str]] = None,
    ):
        super().__init__(model, parent_model, neuron_config, "vae_decoder")

    def forward(self, latent_sample: torch.Tensor):
        inputs = (latent_sample,)
        outputs = self.model(*inputs)
        return outputs