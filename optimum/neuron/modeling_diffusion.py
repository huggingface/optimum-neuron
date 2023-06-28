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
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, Optional, Union

import torch
from diffusers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler, StableDiffusionPipeline
from transformers import CLIPFeatureExtractor, CLIPTokenizer

from optimum.pipelines.diffusers.pipeline_stable_diffusion import StableDiffusionPipelineMixin

from ..exporters.neuron import NeuronConfig
from .modeling_base import NeuronBaseModel


class NeuronStableDiffusionPipeline(NeuronBaseModel, StableDiffusionPipelineMixin):
    auto_model_class = StableDiffusionPipeline
    main_input_name = "input_ids"
    config_name = "model_index.json"

    def __init__(
        self,
        vae_decoder: torch.jit._script.ScriptModule,
        text_encoder: torch.jit._script.ScriptModule,
        unet: torch.jit._script.ScriptModule,
        config: Dict[str, Any],
        tokenizer: CLIPTokenizer,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        feature_extractor: Optional[CLIPFeatureExtractor] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
    ):
        """
        Args:
            vae_decoder (`torch.jit._script.ScriptModule`):
                The TorchScript module associated to the VAE decoder.
            text_encoder (`torch.jit._script.ScriptModule`):
                The TorchScript module associated to the text encoder.
            unet (`torch.jit._script.ScriptModule`):
                The TorchScript module associated to the U-NET.
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
            model_save_dir (`Optional[str]`, defaults to `None`):
                The directory under which the model exported to ONNX was saved.
        """
        self.shared_attributes_init(
            vae_decoder,
            model_save_dir=model_save_dir,
        )
        self._internal_dict = config
        self.vae_decoder = ORTModelVaeDecoder(vae_decoder_session, self)
        self.vae_decoder_model_path = Path(vae_decoder_session._model_path)
        self.text_encoder = ORTModelTextEncoder(text_encoder_session, self)
        self.text_encoder_model_path = Path(text_encoder_session._model_path)
        self.unet = ORTModelUnet(unet_session, self)
        self.unet_model_path = Path(unet_session._model_path)
        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.feature_extractor = feature_extractor
        self.safety_checker = None
        sub_models = {
            DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER: self.text_encoder,
            DIFFUSION_MODEL_UNET_SUBFOLDER: self.unet,
            DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER: self.vae_decoder,
        }
        for name in sub_models.keys():
            self._internal_dict[name] = ("optimum", sub_models[name].__class__.__name__)
        self._internal_dict.pop("vae", None)


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
