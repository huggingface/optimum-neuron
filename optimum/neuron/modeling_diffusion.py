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
"""NeuronDiffusionPipelineBase class for inference of diffusion models on neuron devices."""

import copy
import importlib
import inspect
import logging
import os
import shutil
from abc import abstractmethod
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Literal

import torch
from huggingface_hub import snapshot_download
from torch.nn import ModuleList
from transformers import CLIPFeatureExtractor, CLIPTokenizer, PretrainedConfig, T5Tokenizer
from transformers.modeling_outputs import ModelOutput

from ..exporters.neuron import (
    load_models_and_neuron_configs,
    main_export,
    normalize_stable_diffusion_input_shapes,
    replace_stable_diffusion_submodels,
)
from ..exporters.neuron.model_configs import *  # noqa: F403
from ..exporters.tasks import TasksManager
from ..utils import is_diffusers_available
from .cache.entries.multi_model import MultiModelCacheEntry
from .cache.hub_cache import create_hub_compile_cache_proxy
from .modeling_traced import NeuronTracedModel
from .utils import (
    DIFFUSION_MODEL_CONTROLNET_NAME,
    DIFFUSION_MODEL_IMAGE_ENCODER_NAME,
    DIFFUSION_MODEL_TEXT_ENCODER_2_NAME,
    DIFFUSION_MODEL_TEXT_ENCODER_NAME,
    DIFFUSION_MODEL_TRANSFORMER_NAME,
    DIFFUSION_MODEL_UNET_NAME,
    DIFFUSION_MODEL_VAE_DECODER_NAME,
    DIFFUSION_MODEL_VAE_ENCODER_NAME,
    NEURON_FILE_NAME,
    DiffusersPretrainedConfig,
    NeuronArgumentParser,
    check_if_weights_replacable,
    is_neuronx_available,
    replace_weights,
    store_compilation_config,
)
from .utils.require_utils import requires_torch_neuronx
from .utils.version_utils import get_neuronxcc_version


if is_neuronx_available():
    import torch_neuronx

    NEURON_COMPILER_TYPE = "neuronx-cc"
    NEURON_COMPILER_VERSION = get_neuronxcc_version()


if is_diffusers_available():
    from diffusers import (
        ControlNetModel,
        LatentConsistencyModelPipeline,
        LCMScheduler,
        PixArtAlphaPipeline,
        PixArtSigmaPipeline,
        StableDiffusionControlNetPipeline,
        StableDiffusionImg2ImgPipeline,
        StableDiffusionInpaintPipeline,
        StableDiffusionInstructPix2PixPipeline,
        StableDiffusionPipeline,
        StableDiffusionXLControlNetPipeline,
        StableDiffusionXLImg2ImgPipeline,
        StableDiffusionXLInpaintPipeline,
        StableDiffusionXLPipeline,
    )
    from diffusers.configuration_utils import FrozenDict
    from diffusers.image_processor import PixArtImageProcessor, VaeImageProcessor
    from diffusers.models.autoencoders.vae import DecoderOutput, DiagonalGaussianDistribution
    from diffusers.models.controlnet import ControlNetOutput
    from diffusers.models.embeddings import ImageProjection, IPAdapterFullImageProjection
    from diffusers.models.modeling_outputs import AutoencoderKLOutput
    from diffusers.pipelines.controlnet import MultiControlNetModel
    from diffusers.pipelines.pipeline_utils import DiffusionPipeline
    from diffusers.schedulers import SchedulerMixin
    from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
    from diffusers.utils import CONFIG_NAME

    from .pipelines import (
        NeuronStableDiffusionControlNetPipelineMixin,
        NeuronStableDiffusionXLControlNetPipelineMixin,
        NeuronStableDiffusionXLPipelineMixin,
    )
else:
    raise ModuleNotFoundError("`diffusers` python package is not installed.")


if TYPE_CHECKING:
    from ..exporters.neuron import NeuronDefaultConfig


logger = logging.getLogger(__name__)


class NeuronDiffusionPipelineBase(NeuronTracedModel):
    auto_model_class = DiffusionPipeline
    task = None
    library_name = "diffusers"
    base_model_prefix = "neuron_model"
    config_name = "model_index.json"
    sub_component_config_name = "config.json"
    _optional_components = [
        "tokenizer",
        "tokenizer_2",
        "text_encoder",
        "text_encoder_2",
        "vae_encoder",
        "image_encoder",
        "unet",
        "transformer",
        "feature_extractor",
    ]
    encoder_hid_proj = None  # A dummy module of Unet/transformer when they take the outputs of image encoder.

    def __init__(
        self,
        config: dict[str, Any],
        configs: dict[str, "PretrainedConfig"],
        neuron_configs: dict[str, "NeuronDefaultConfig"],
        data_parallel_mode: Literal["none", "unet", "transformer", "all"],
        scheduler: SchedulerMixin | None,
        vae_decoder: "torch.jit._script.ScriptModule | NeuronModelVaeDecoder",
        text_encoder: "torch.jit._script.ScriptModule | NeuronModelTextEncoder | None" = None,
        text_encoder_2: "torch.jit._script.ScriptModule | NeuronModelTextEncoder | None" = None,
        unet: "torch.jit._script.ScriptModule | NeuronModelUnet | None" = None,
        transformer: "torch.jit._script.ScriptModule | NeuronModelTransformer | None" = None,
        vae_encoder: "torch.jit._script.ScriptModule | NeuronModelVaeEncoder | None" = None,
        image_encoder: torch.jit._script.ScriptModule | None = None,
        safety_checker: torch.jit._script.ScriptModule | None = None,
        tokenizer: CLIPTokenizer | T5Tokenizer | None = None,
        tokenizer_2: CLIPTokenizer | None = None,
        feature_extractor: CLIPFeatureExtractor | None = None,
        controlnet: "torch.jit._script.ScriptModule | list[torch.jit._script.ScriptModule]| NeuronControlNetModel | NeuronMultiControlNetModel | None" = None,
        # stable diffusion xl specific arguments
        requires_aesthetics_score: bool = False,
        force_zeros_for_empty_prompt: bool = True,
        add_watermarker: bool | None = None,
        model_save_dir: str | Path | TemporaryDirectory | None = None,
        model_and_config_save_paths: dict[str, tuple[str, Path]] | None = None,
    ):
        """
        Args:
            config (`dict[str, Any]`):
                A config dictionary from which the model components will be instantiated. Make sure to only load
                configuration files of compatible classes.
            configs (dict[str, "PretrainedConfig"], defaults to `None`):
                A dictionary configurations for components of the pipeline.
            neuron_configs (dict[str, "NeuronDefaultConfig"], defaults to `None`):
                A list of Neuron configurations related to the compilation.
            data_parallel_mode (`Literal["none", "unet", "all"]`):
                Mode to decide what components to load into both NeuronCores of a Neuron device. Can be "none"(no data parallel), "unet"(only
                load unet into both cores of each device), "all"(load the whole pipeline into both cores).
            scheduler (`SchedulerMixin | None`):
                A scheduler to be used in combination with the U-NET component to denoise the encoded image latents.
            vae_decoder (`torch.jit._script.ScriptModule | "NeuronModelVaeDecoder"`):
                The Neuron TorchScript module associated to the VAE decoder.
            text_encoder (`torch.jit._script.ScriptModule | "NeuronModelTextEncoder" | None`, defaults to `None`):
                The Neuron TorchScript module associated to the text encoder.
            text_encoder_2 (`torch.jit._script.ScriptModule | "NeuronModelTextEncoder" | None`, defaults to `None`):
                The Neuron TorchScript module associated to the second frozen text encoder. Stable Diffusion XL uses the text and pool portion of [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection), specifically the [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k) variant.
            unet (`torch.jit._script.ScriptModule | "NeuronModelUnet" | None`, defaults to `None`):
                The Neuron TorchScript module associated to the U-NET.
            transformer (`torch.jit._script.ScriptModule | "NeuronModelTransformer" | None`, defaults to `None`):
                The Neuron TorchScript module associated to the diffuser transformer.
            vae_encoder (`torch.jit._script.ScriptModule | "NeuronModelVaeEncoder" | None`, defaults to `None`):
                The Neuron TorchScript module associated to the VAE encoder.
            image_encoder (`torch.jit._script.ScriptModule | None`, defaults to `None`):
                The Neuron TorchScript module associated to the frozen CLIP image-encoder.
            safety_checker (`torch.jit._script.ScriptModule | None`, defaults to `None`):
                The Neuron TorchScript module associated to the Classification module that estimates whether generated images could be considered offensive or harmful.
            tokenizer (`CLIPTokenizer | T5Tokenizer | None`, defaults to `None`):
                Tokenizer of class
                [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer) for stable diffusion models,
                or tokenizer of class [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer) for diffusion transformers.
            tokenizer_2 (`CLIPTokenizer | None`, defaults to `None`):
                Second tokenizer of class
                [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
            feature_extractor (`CLIPFeatureExtractor | None`, defaults to `None`):
                A model extracting features from generated images to be used as inputs for the `safety_checker`
            controlnet (`torch.jit._script.ScriptModule | list[torch.jit._script.ScriptModule | None | "NeuronControlNetModel" | "NeuronMultiControlNetModel"`, defaults to `None`):
                The Neuron TorchScript module(s) associated to the ControlNet(s).
            requires_aesthetics_score (`bool`, defaults to `False`):
                Whether the `unet` requires an `aesthetic_score` condition to be passed during inference. Also see the
                config of `stabilityai/stable-diffusion-xl-refiner-1-0`.
            force_zeros_for_empty_prompt (`bool`, defaults to `True`):
                Whether the negative prompt embeddings shall be forced to always be set to 0. Also see the config of
                `stabilityai/stable-diffusion-xl-base-1-0`.
            add_watermarker (`bool | None`, defaults to `None`):
                Whether to use the [invisible_watermark library](https://github.com/ShieldMnt/invisible-watermark/) to
                watermark output images. If not defined, it will default to True if the package is installed, otherwise no
                watermarker will be used.
            model_save_dir (`str | Path | TemporaryDirectory | None`, defaults to `None`):
                The directory under which the exported Neuron models were saved.
            model_and_config_save_paths (`dict[str, tuple[str, Path]] | None`, defaults to `None`):
                The paths where exported Neuron models were saved.
        """

        # configurations
        self._internal_dict = config
        self.data_parallel_mode = data_parallel_mode
        self.configs = configs
        self.neuron_configs = neuron_configs
        self.dynamic_batch_size = all(
            neuron_config._config.neuron["dynamic_batch_size"] for neuron_config in self.neuron_configs.values()
        )

        # pipeline components
        self.text_encoder = (
            NeuronModelTextEncoder(
                text_encoder,
                self,
                self.configs[DIFFUSION_MODEL_TEXT_ENCODER_NAME],
                self.neuron_configs[DIFFUSION_MODEL_TEXT_ENCODER_NAME],
            )
            if text_encoder is not None and not isinstance(text_encoder, NeuronModelTextEncoder)
            else text_encoder
        )
        self.text_encoder_2 = (
            NeuronModelTextEncoder(
                text_encoder_2,
                self,
                self.configs[DIFFUSION_MODEL_TEXT_ENCODER_2_NAME],
                self.neuron_configs[DIFFUSION_MODEL_TEXT_ENCODER_2_NAME],
            )
            if text_encoder_2 is not None and not isinstance(text_encoder_2, NeuronModelTextEncoder)
            else text_encoder_2
        )
        self.unet = (
            NeuronModelUnet(
                unet, self, self.configs[DIFFUSION_MODEL_UNET_NAME], self.neuron_configs[DIFFUSION_MODEL_UNET_NAME]
            )
            if unet is not None and not isinstance(unet, NeuronModelUnet)
            else unet
        )
        self.transformer = (
            NeuronModelTransformer(
                transformer,
                self,
                self.configs[DIFFUSION_MODEL_TRANSFORMER_NAME],
                self.neuron_configs[DIFFUSION_MODEL_TRANSFORMER_NAME],
            )
            if transformer is not None and not isinstance(transformer, NeuronModelTransformer)
            else transformer
        )
        self.vae_encoder = (
            NeuronModelVaeEncoder(
                vae_encoder,
                self,
                self.configs[DIFFUSION_MODEL_VAE_ENCODER_NAME],
                self.neuron_configs[DIFFUSION_MODEL_VAE_ENCODER_NAME],
            )
            if vae_encoder is not None and not isinstance(vae_encoder, NeuronModelVaeEncoder)
            else vae_encoder
        )
        self.vae_decoder = (
            NeuronModelVaeDecoder(
                vae_decoder,
                self,
                self.configs[DIFFUSION_MODEL_VAE_DECODER_NAME],
                self.neuron_configs[DIFFUSION_MODEL_VAE_DECODER_NAME],
            )
            if vae_decoder is not None and not isinstance(vae_decoder, NeuronModelVaeDecoder)
            else vae_decoder
        )
        self.vae = NeuronModelVae(self.vae_encoder, self.vae_decoder)

        if (
            controlnet
            and not isinstance(controlnet, NeuronControlNetModel)
            and not isinstance(controlnet, NeuronMultiControlNetModel)
        ):
            controlnet_cls = (
                NeuronMultiControlNetModel
                if isinstance(controlnet, list) and len(controlnet) > 1
                else NeuronControlNetModel
            )
            self.controlnet = controlnet_cls(
                controlnet,
                self,
                self.configs[DIFFUSION_MODEL_CONTROLNET_NAME],
                self.neuron_configs[DIFFUSION_MODEL_CONTROLNET_NAME],
            )
        else:
            self.controlnet = controlnet

        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.scheduler = scheduler

        # change lcm scheduler which extends the denoising procedure
        self.is_lcm = False
        if self.unet and NeuronDiffusionPipelineBase.is_lcm(self.unet.config):
            self.is_lcm = True
            self.scheduler = LCMScheduler.from_config(self.scheduler.config)

        self.feature_extractor = feature_extractor
        self.image_encoder = (
            NeuronModelImageEncoder(
                image_encoder,
                self,
                self.configs[DIFFUSION_MODEL_IMAGE_ENCODER_NAME],
                self.neuron_configs[DIFFUSION_MODEL_IMAGE_ENCODER_NAME],
            )
            if image_encoder is not None and not isinstance(image_encoder, NeuronModelImageEncoder)
            else image_encoder
        )
        self.safety_checker = safety_checker  # TODO: implement the class `NeuronStableDiffusionSafetyChecker`.

        all_possible_init_args = {
            "vae": self.vae,
            "unet": self.unet,
            "transformer": self.transformer,
            "text_encoder": self.text_encoder,
            "text_encoder_2": self.text_encoder_2,
            "controlnet": self.controlnet,
            "image_encoder": self.image_encoder,
            "safety_checker": self.safety_checker,
            "scheduler": self.scheduler,
            "tokenizer": self.tokenizer,
            "tokenizer_2": self.tokenizer_2,
            "feature_extractor": self.feature_extractor,
            "requires_aesthetics_score": requires_aesthetics_score,
            "force_zeros_for_empty_prompt": force_zeros_for_empty_prompt,
            "add_watermarker": add_watermarker,
        }
        diffusers_pipeline_args = {}
        for key in inspect.signature(self.auto_model_class).parameters.keys():
            if key in all_possible_init_args:
                diffusers_pipeline_args[key] = all_possible_init_args[key]
        self.auto_model_class.__init__(self, **diffusers_pipeline_args)

        self._attributes_init(model_save_dir)
        self.model_and_config_save_paths = model_and_config_save_paths if model_and_config_save_paths else None
        self.register_to_config(force_zeros_for_empty_prompt=force_zeros_for_empty_prompt)
        self.register_to_config(requires_aesthetics_score=requires_aesthetics_score)

        # Calculate static shapes
        if hasattr(self.vae.config, "block_out_channels"):
            self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        else:
            self.vae_scale_factor = 8

        unet_or_transformer = "transformer" if self.transformer else "unet"
        unet_or_transformer_batch_size = self.neuron_configs[unet_or_transformer].batch_size
        if "text_encoder" in self.neuron_configs:
            text_encoder_batch_size = self.neuron_configs["text_encoder"].batch_size
            self.num_images_per_prompt = unet_or_transformer_batch_size // text_encoder_batch_size
        elif "text_encoder_2" in self.neuron_configs:
            text_encoder_batch_size = self.neuron_configs["text_encoder_2"].batch_size
            self.num_images_per_prompt = unet_or_transformer_batch_size // text_encoder_batch_size
        else:
            self.num_images_per_prompt = 1

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        self.control_image_processor = VaeImageProcessor(
            vae_scale_factor=self.vae_scale_factor, do_convert_rgb=True, do_normalize=False
        )
        # create dummy objects for inference with ip adapters
        self._maybe_create_dummy_image_proj_layers()

    @staticmethod
    def is_lcm(unet_config):
        patterns = ["lcm", "latent-consistency"]
        unet_name_or_path = getattr(unet_config, "_name_or_path", "").lower()
        return any(pattern in unet_name_or_path for pattern in patterns)

    @staticmethod
    @requires_torch_neuronx
    def load_model(
        data_parallel_mode: Literal["none", "unet", "transformer", "all"] | None,
        text_encoder_path: str | Path | None = None,
        text_encoder_2_path: str | Path | None = None,
        image_encoder_path: str | Path | None = None,
        unet_path: str | Path | None = None,
        transformer_path: str | Path | None = None,
        vae_encoder_path: str | Path | None = None,
        vae_decoder_path: str | Path | None = None,
        controlnet_paths: list[Path] | None = None,
        dynamic_batch_size: bool = False,
        to_neuron: bool = False,
    ):
        """
        Loads Stable Diffusion TorchScript modules compiled by neuron(x)-cc compiler. It will be first loaded onto CPU and then moved to
        one or multiple [NeuronCore](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/neuroncores-arch.html).

        Args:
            data_parallel_mode (`Literal["none", "unet", "all"] | None`):
                Mode to decide what components to load into both NeuronCores of a Neuron device. Can be "none"(no data parallel), "unet"(only
                load unet into both cores of each device), "all"(load the whole pipeline into both cores).
            text_encoder_path (`str | Path`, defaults to `None`):
                Path of the compiled text encoder.
            text_encoder_2_path (`str | Path | None`, defaults to `None`):
                Path of the compiled second frozen text encoder. SDXL only.
            image_encoder_path (`str | Path | None`, defaults to `None`):
                Path of the compiled image encoder.
            unet_path (`str | Path | None`, defaults to `None`):
                Path of the compiled U-NET.
            transformer_path (`str | Path | None`, defaults to `None`):
                Path of the compiled diffusion transformer.
            vae_encoder_path (`str | Path | None`, defaults to `None`):
                Path of the compiled VAE encoder. It is optional, only used for tasks taking images as input.
            vae_decoder_path (`str | Path | None`, defaults to `None`):
                Path of the compiled VAE decoder.
            controlnet_paths (`list[Path] | None`, defaults to `None`):
                Path of the compiled controlnets.
            dynamic_batch_size (`bool`, defaults to `False`):
                Whether enable dynamic batch size for neuron compiled model. If `True`, the input batch size can be a multiple of the batch size during the compilation.
            to_neuron (`bool`, defaults to `False`):
                Whether to move manually the traced model to NeuronCore. It's only needed when `inline_weights_to_neff=False`, otherwise it is loaded automatically to a Neuron device.
        """
        submodels = {
            # Load the UNet/Diffusion transformer first to avoid CPU OOM
            "unet": unet_path,
            "transformer": transformer_path,
            "text_encoder": text_encoder_path,
            "text_encoder_2": text_encoder_2_path,
            "vae_encoder": vae_encoder_path,
            "vae_decoder": vae_decoder_path,
            "controlnet": controlnet_paths,
            "image_encoder": image_encoder_path,
        }

        def _load_models_to_neuron(submodels, models_on_both_cores=None, models_on_a_single_core=None):
            # loading models to both cores, eg. unet, transformer.
            if models_on_both_cores:
                for model_name in models_on_both_cores:
                    submodel_paths = submodels[model_name]
                    # for the case of multiple controlnets the path could be a list
                    if not isinstance(submodel_paths, list):
                        submodel_paths = [submodel_paths]
                    submodels_list = []
                    for submodel_path in submodel_paths:
                        if submodel_path is not None and submodel_path.is_file():
                            submodel = NeuronTracedModel.load_model(
                                submodel_path, to_neuron=False
                            )  # No need to load to neuron manually when dp
                            submodel = torch_neuronx.DataParallel(
                                submodel,
                                [0, 1],
                                set_dynamic_batching=dynamic_batch_size,
                            )
                            submodels_list.append(submodel)
                    if submodels_list:
                        submodels[model_name] = submodels_list if len(submodels_list) > 1 else submodels_list[0]
                    else:
                        submodels[model_name] = None
            # loading models to a single core, eg. text encoders, vae.
            if models_on_a_single_core:
                for model_name in models_on_a_single_core:
                    submodel_paths = submodels[model_name]
                    # for the case of multiple controlnets the path could be a list
                    if not isinstance(submodel_paths, list):
                        submodel_paths = [submodel_paths]
                    submodels_list = []
                    for submodel_path in submodel_paths:
                        if submodel_path is not None and submodel_path.is_file():
                            submodel = NeuronTracedModel.load_model(submodel_path, to_neuron=to_neuron)
                            submodels_list.append(submodel)
                    if submodels_list:
                        submodels[model_name] = submodels_list if len(submodels_list) > 1 else submodels_list[0]
                    else:
                        submodels[model_name] = None
            return submodels

        if data_parallel_mode == "all":
            logger.info("Loading the whole pipeline into both Neuron Cores...")
            submodels = _load_models_to_neuron(submodels=submodels, models_on_both_cores=list(submodels))
        elif data_parallel_mode == "unet":
            logger.info("Loading only U-Net into both Neuron Cores...")
            models_on_a_single_core = list(submodels)
            models_on_a_single_core.remove("unet")
            models_on_a_single_core.remove(
                "controlnet"
            )  # controlnet takes inputs with the same batch_size as the unet
            submodels = _load_models_to_neuron(
                submodels=submodels,
                models_on_both_cores=["unet", "controlnet"],
                models_on_a_single_core=models_on_a_single_core,
            )
        elif data_parallel_mode == "transformer":
            logger.info("Loading only diffusion transformer into both Neuron Cores...")
            models_on_a_single_core = list(submodels)
            models_on_a_single_core.remove("transformer")
            models_on_a_single_core.remove(
                "controlnet"
            )  # controlnet takes inputs with the same batch_size as the transformer
            submodels = _load_models_to_neuron(
                submodels=submodels,
                models_on_both_cores=["transformer", "controlnet"],
                models_on_a_single_core=models_on_a_single_core,
            )
        elif data_parallel_mode == "none":
            logger.info("Loading the pipeline without any data parallelism...")
            submodels = _load_models_to_neuron(submodels=submodels, models_on_a_single_core=list(submodels))
        else:
            raise ValueError("You need to pass `data_parallel_mode` to define Neuron Core allocation.")

        return submodels

    def replace_weights(self, weights: dict[str, torch.Tensor] | None | torch.nn.Module = None):
        check_if_weights_replacable(self.configs, weights)
        model_names = [
            "text_encoder",
            "text_encoder_2",
            "unet",
            "transformer",
            "vae_decoder",
            "vae_encoder",
            "image_encoder",
        ]
        for name in model_names:
            model = getattr(self, name, None)
            weight = getattr(weights, name, None)
            if model is not None and weight is not None:
                model = replace_weights(model.model, weight)

    @staticmethod
    def set_default_dp_mode(configs: dict):
        if "unet" in configs:
            unet_config = configs["unet"]
            if NeuronDiffusionPipelineBase.is_lcm(unet_config) is True:
                # LCM applies guidance using guidance embeddings, so we can load the whole pipeline into both cores.
                return "all"
            else:
                # Load U-Net into both cores for classifier-free guidance which doubles batch size of inputs passed to the U-Net.
                return "unet"
        elif "transformer" in configs:
            return "transformer"
        else:
            logger.warning(
                "There is no unet nor transformer in your pipeline, the data parallelism will be disabled, make sure that you are loading the model correctly!"
            )
            return "none"

    def _save_pretrained(
        self,
        save_directory: str | Path,
        text_encoder_file_name: str = NEURON_FILE_NAME,
        text_encoder_2_file_name: str = NEURON_FILE_NAME,
        unet_file_name: str = NEURON_FILE_NAME,
        transformer_file_name: str = NEURON_FILE_NAME,
        vae_encoder_file_name: str = NEURON_FILE_NAME,
        vae_decoder_file_name: str = NEURON_FILE_NAME,
        controlnet_file_name: str = NEURON_FILE_NAME,
        image_encoder_file_name: str = NEURON_FILE_NAME,
    ):
        """
        Saves the model to the serialized format optimized for Neuron devices.
        """
        if self.model_and_config_save_paths is None:
            logger.warning(
                "`model_save_paths` is None which means that no path of Neuron model is defined. Nothing will be saved."
            )
            return

        save_directory = Path(save_directory)

        def _remove_submodel_if_non_exist(model_names):
            for model_name in model_names:
                if not self.model_and_config_save_paths.get(model_name)[0].is_file():
                    self.model_and_config_save_paths.pop(model_name)

        _remove_submodel_if_non_exist(
            [
                DIFFUSION_MODEL_TEXT_ENCODER_NAME,
                DIFFUSION_MODEL_TEXT_ENCODER_2_NAME,
                DIFFUSION_MODEL_UNET_NAME,
                DIFFUSION_MODEL_TRANSFORMER_NAME,
                DIFFUSION_MODEL_VAE_ENCODER_NAME,
                DIFFUSION_MODEL_IMAGE_ENCODER_NAME,
            ]
        )

        if not self.model_and_config_save_paths.get(DIFFUSION_MODEL_CONTROLNET_NAME)[0]:
            self.model_and_config_save_paths.pop(DIFFUSION_MODEL_CONTROLNET_NAME)
            num_controlnet = 0
        else:
            num_controlnet = len(self.model_and_config_save_paths.get(DIFFUSION_MODEL_CONTROLNET_NAME)[0])

        logger.info(f"Saving the {tuple(self.model_and_config_save_paths.keys())}...")

        dst_paths = {
            DIFFUSION_MODEL_TEXT_ENCODER_NAME: save_directory
            / DIFFUSION_MODEL_TEXT_ENCODER_NAME
            / text_encoder_file_name,
            DIFFUSION_MODEL_TEXT_ENCODER_2_NAME: save_directory
            / DIFFUSION_MODEL_TEXT_ENCODER_2_NAME
            / text_encoder_2_file_name,
            DIFFUSION_MODEL_UNET_NAME: save_directory / DIFFUSION_MODEL_UNET_NAME / unet_file_name,
            DIFFUSION_MODEL_TRANSFORMER_NAME: save_directory
            / DIFFUSION_MODEL_TRANSFORMER_NAME
            / transformer_file_name,
            DIFFUSION_MODEL_VAE_ENCODER_NAME: save_directory
            / DIFFUSION_MODEL_VAE_ENCODER_NAME
            / vae_encoder_file_name,
            DIFFUSION_MODEL_VAE_DECODER_NAME: save_directory
            / DIFFUSION_MODEL_VAE_DECODER_NAME
            / vae_decoder_file_name,
            DIFFUSION_MODEL_IMAGE_ENCODER_NAME: save_directory
            / DIFFUSION_MODEL_IMAGE_ENCODER_NAME
            / image_encoder_file_name,
        }
        dst_paths[DIFFUSION_MODEL_CONTROLNET_NAME] = [
            save_directory / (DIFFUSION_MODEL_CONTROLNET_NAME + f"_{str(idx)}") / controlnet_file_name
            for idx in range(num_controlnet)
        ]

        src_paths_list = []
        dst_paths_list = []
        for model_name in set(self.model_and_config_save_paths.keys()).intersection(dst_paths.keys()):
            model_src_path = self.model_and_config_save_paths[model_name][0]
            if isinstance(model_src_path, list):
                # neuron model
                src_paths_list += model_src_path
                dst_paths_list += dst_paths[model_name]

                # config
                src_paths_list += self.model_and_config_save_paths[model_name][1]
                dst_paths_list += [model_path.parent / CONFIG_NAME for model_path in dst_paths[model_name]]

            else:
                # neuron model
                src_paths_list.append(model_src_path)
                dst_paths_list.append(dst_paths[model_name])

                # config
                src_paths_list.append(self.model_and_config_save_paths[model_name][1])
                dst_paths_list.append(dst_paths[model_name].parent / CONFIG_NAME)

        for src_path, dst_path in zip(src_paths_list, dst_paths_list):
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            if src_path.is_file():
                shutil.copyfile(src_path, dst_path)

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(save_directory.joinpath("tokenizer"))
        if self.tokenizer_2 is not None:
            self.tokenizer_2.save_pretrained(save_directory.joinpath("tokenizer_2"))
        self.scheduler.save_pretrained(save_directory.joinpath("scheduler"))
        if self.feature_extractor is not None:
            self.feature_extractor.save_pretrained(save_directory.joinpath("feature_extractor"))

    @classmethod
    @requires_torch_neuronx
    def _from_pretrained(
        cls,
        model_id: str | Path,
        config: dict[str, Any],
        token: bool | str | None = None,
        revision: str | None = None,
        force_download: bool = False,
        cache_dir: str | None = None,
        text_encoder_file_name: str | None = NEURON_FILE_NAME,
        text_encoder_2_file_name: str | None = NEURON_FILE_NAME,
        unet_file_name: str | None = NEURON_FILE_NAME,
        transformer_file_name: str | None = NEURON_FILE_NAME,
        vae_encoder_file_name: str | None = NEURON_FILE_NAME,
        vae_decoder_file_name: str | None = NEURON_FILE_NAME,
        controlnet_file_name: str | None = NEURON_FILE_NAME,
        image_encoder_file_name: str | None = NEURON_FILE_NAME,
        local_files_only: bool = False,
        model_save_dir: str | Path | TemporaryDirectory | None = None,
        data_parallel_mode: Literal["none", "unet", "transformer", "all"] | None = None,
        **kwargs,  # To share kwargs only available for `_from_transformers`
    ):
        model_id = str(model_id)
        patterns = set(config.keys())
        processors_to_load = patterns.intersection({"feature_extractor", "tokenizer", "tokenizer_2", "scheduler"})

        if not os.path.isdir(model_id):
            patterns.update({DIFFUSION_MODEL_VAE_ENCODER_NAME, DIFFUSION_MODEL_VAE_DECODER_NAME})
            allow_patterns = {os.path.join(k, "*") for k in patterns if not k.startswith("_")}
            allow_patterns.update(
                {
                    text_encoder_file_name,
                    text_encoder_2_file_name,
                    unet_file_name,
                    transformer_file_name,
                    vae_encoder_file_name,
                    vae_decoder_file_name,
                    controlnet_file_name,
                    image_encoder_file_name,
                    SCHEDULER_CONFIG_NAME,
                    CONFIG_NAME,
                    cls.config_name,
                }
            )
            # Downloads all repo's files matching the allowed patterns
            model_id = snapshot_download(
                model_id,
                cache_dir=cache_dir,
                local_files_only=local_files_only,
                token=token,
                revision=revision,
                force_download=force_download,
                allow_patterns=allow_patterns,
                ignore_patterns=["*.msgpack", "*.safetensors", "*.bin"],
            )

        new_model_save_dir = Path(model_id)
        sub_models = {}
        for name in processors_to_load:
            library_name, library_classes = config[name]
            if library_classes is not None:
                library = importlib.import_module(library_name)
                class_obj = getattr(library, library_classes)
                load_method = getattr(class_obj, "from_pretrained")
                # Check if the module is in a subdirectory
                if (new_model_save_dir / name).is_dir():
                    sub_models[name] = load_method(new_model_save_dir / name)
                else:
                    sub_models[name] = load_method(new_model_save_dir)

        model_and_config_save_paths = {
            "text_encoder": (
                new_model_save_dir / DIFFUSION_MODEL_TEXT_ENCODER_NAME / text_encoder_file_name,
                new_model_save_dir / DIFFUSION_MODEL_TEXT_ENCODER_NAME / cls.sub_component_config_name,
            ),
            "text_encoder_2": (
                new_model_save_dir / DIFFUSION_MODEL_TEXT_ENCODER_2_NAME / text_encoder_2_file_name,
                new_model_save_dir / DIFFUSION_MODEL_TEXT_ENCODER_2_NAME / cls.sub_component_config_name,
            ),
            "image_encoder": (
                new_model_save_dir / DIFFUSION_MODEL_IMAGE_ENCODER_NAME / image_encoder_file_name,
                new_model_save_dir / DIFFUSION_MODEL_IMAGE_ENCODER_NAME / cls.sub_component_config_name,
            ),
            "unet": (
                new_model_save_dir / DIFFUSION_MODEL_UNET_NAME / unet_file_name,
                new_model_save_dir / DIFFUSION_MODEL_UNET_NAME / cls.sub_component_config_name,
            ),
            "transformer": (
                new_model_save_dir / DIFFUSION_MODEL_TRANSFORMER_NAME / transformer_file_name,
                new_model_save_dir / DIFFUSION_MODEL_TRANSFORMER_NAME / cls.sub_component_config_name,
            ),
            "vae_encoder": (
                new_model_save_dir / DIFFUSION_MODEL_VAE_ENCODER_NAME / vae_encoder_file_name,
                new_model_save_dir / DIFFUSION_MODEL_VAE_ENCODER_NAME / cls.sub_component_config_name,
            ),
            "vae_decoder": (
                new_model_save_dir / DIFFUSION_MODEL_VAE_DECODER_NAME / vae_decoder_file_name,
                new_model_save_dir / DIFFUSION_MODEL_VAE_DECODER_NAME / cls.sub_component_config_name,
            ),
        }

        # Add ControlNet paths
        controlnet_model_paths = []
        controlnet_config_paths = []
        for path in new_model_save_dir.iterdir():
            if path.is_dir() and path.name.startswith("controlnet"):
                controlnet_model_paths.append(path / controlnet_file_name)
                controlnet_config_paths.append(path / cls.sub_component_config_name)
        model_and_config_save_paths["controlnet"] = (controlnet_model_paths, controlnet_config_paths)

        # Re-build pretrained configs and neuron configs
        configs, neuron_configs = {}, {}
        inline_weights_to_neff = True
        for name, (_, config_paths) in model_and_config_save_paths.items():
            if not isinstance(config_paths, list):
                config_paths = [config_paths]
            sub_model_configs = []
            sub_neuron_configs = []
            for config_path in config_paths:
                if config_path.is_file():
                    model_config = DiffusersPretrainedConfig.from_json_file(config_path)
                    neuron_config = cls._neuron_config_init(model_config)
                    inline_weights_to_neff = inline_weights_to_neff and neuron_config._config.neuron.get(
                        "inline_weights_to_neff", True
                    )
                    sub_model_configs.append(model_config)
                    sub_neuron_configs.append(neuron_config)
            if sub_model_configs and sub_neuron_configs:
                configs[name] = sub_model_configs if len(sub_model_configs) > 1 else sub_model_configs[0]
                neuron_configs[name] = sub_neuron_configs if len(sub_neuron_configs) > 1 else sub_neuron_configs[0]

        if data_parallel_mode is None:
            data_parallel_mode = cls.set_default_dp_mode(configs)

        pipe = cls.load_model(
            data_parallel_mode=data_parallel_mode,
            text_encoder_path=model_and_config_save_paths["text_encoder"][0],
            unet_path=model_and_config_save_paths["unet"][0],
            transformer_path=model_and_config_save_paths["transformer"][0],
            vae_decoder_path=model_and_config_save_paths["vae_decoder"][0],
            vae_encoder_path=model_and_config_save_paths["vae_encoder"][0],
            text_encoder_2_path=model_and_config_save_paths["text_encoder_2"][0],
            image_encoder_path=model_and_config_save_paths["image_encoder"][0],
            controlnet_paths=model_and_config_save_paths["controlnet"][0],
            dynamic_batch_size=neuron_configs[DIFFUSION_MODEL_TEXT_ENCODER_NAME].dynamic_batch_size,
            to_neuron=not inline_weights_to_neff,
        )

        if model_save_dir is None:
            model_save_dir = new_model_save_dir

        return cls(
            text_encoder=pipe.get("text_encoder"),
            text_encoder_2=pipe.get("text_encoder_2"),
            unet=pipe.get("unet"),
            transformer=pipe.get("transformer"),
            vae_encoder=pipe.get("vae_encoder"),
            vae_decoder=pipe.get("vae_decoder"),
            controlnet=pipe.get("controlnet"),
            image_encoder=pipe.get("image_encoder"),
            config=config,
            tokenizer=sub_models.get("tokenizer", None),
            tokenizer_2=sub_models.get("tokenizer_2", None),
            scheduler=sub_models.get("scheduler"),
            feature_extractor=sub_models.get("feature_extractor", None),
            data_parallel_mode=data_parallel_mode,
            configs=configs,
            neuron_configs=neuron_configs,
            model_save_dir=model_save_dir,
            model_and_config_save_paths=model_and_config_save_paths,
        )

    @classmethod
    @requires_torch_neuronx
    def _from_transformers(cls, *args, **kwargs):
        # Deprecate it when optimum uses `_export` as from_pretrained_method in a stable release.
        return cls._export(*args, **kwargs)

    @classmethod
    @requires_torch_neuronx
    def _export(
        cls,
        model_id: str | Path,
        config: dict[str, Any],
        torch_dtype: str | torch.dtype | None = None,
        unet_id: str | Path | None = None,
        token: bool | str | None = None,
        revision: str = "main",
        force_download: bool = True,
        cache_dir: str | None = None,
        compiler_workdir: str | None = None,
        tensor_parallel_size: int | None = 1,
        disable_neuron_cache: bool = False,
        inline_weights_to_neff: bool = True,
        optlevel: str = "2",
        subfolder: str = "",
        local_files_only: bool = False,
        trust_remote_code: bool = False,
        task: str | None = None,
        auto_cast: str | None = "matmul",
        auto_cast_type: str | None = "bf16",
        dynamic_batch_size: bool = False,
        output_attentions: bool = False,
        output_hidden_states: bool = False,
        data_parallel_mode: Literal["none", "unet", "transformer", "all"] | None = None,
        controlnet_ids: str | list[str] | None = None,
        **kwargs,
    ) -> "NeuronDiffusionPipelineBase":
        """
        Args:
            model_id (`str | Path`):
                Can be either:
                - A string, the *model id* of a pretrained model hosted inside a model repo on huggingface.co.
                    Valid model ids can be located at the root-level, like `bert-base-uncased`, or namespaced under a
                    user or organization name, like `dbmdz/bert-base-german-cased`.
                - A path to a *directory* containing a model saved using [`~OptimizedModel.save_pretrained`],
                    e.g., `./my_model_directory/`.
            config (`dict[str, Any]`):
                A config dictionary from which the model components will be instantiated. Make sure to only load
                configuration files of compatible classes.
            torch_dtype (`str | torch.dtype | None`, defaults to `None`):
                Override the default `torch.dtype` and load the model under this dtype. If `auto` is passed, the dtype will be automatically derived from the model's weights.
            unet_id (`str | Path | None`, defaults to `None`):
                A string or a path point to the U-NET model to replace the one in the original pipeline.
            token (`bool | str | None`, defaults to `None`):
                The token to use as HTTP bearer authorization for remote files. If `True`, will use the token generated
                when running `huggingface-cli login` (stored in `huggingface_hub.constants.HF_TOKEN_PATH`).
            revision (`str`, defaults to `"main"`):
                The specific model version to use (can be a branch name, tag name or commit id).
            force_download (`bool`, defaults to `True`):
                Whether or not to force the (re-)download of the model weights and configuration files, overriding the
                cached versions if they exist.
            cache_dir (`str | None`, defaults to `None`):
                Path to a directory in which a downloaded pretrained model configuration should be cached if the
                standard cache should not be used.
            compiler_workdir (`str | None`, defaults to `None`):
                Path to a directory in which the neuron compiler will store all intermediary files during the compilation(neff, weight, hlo graph...).
            disable_neuron_cache (`bool`, defaults to `False`):
                Whether to disable automatic caching of compiled models. If set to True, will not load neuron cache nor cache the compiled artifacts.
            inline_weights_to_neff (`bool`, defaults to `True`):
                Whether to inline the weights to the neff graph. If set to False, weights will be separated from the neff.
            optlevel (`str`, defaults to `"2"`):
            The level of optimization the compiler should perform. Can be `"1"`, `"2"` or `"3"`, defaults to "2".
                1: enables the core performance optimizations in the compiler, while also minimizing compile time.
                2: provides the best balance between model performance and compile time.
                3: may provide additional model execution performance but may incur longer compile times and higher host memory usage during model compilation.
            subfolder (`str`, defaults to `""`):
                In case the relevant files are located inside a subfolder of the model repo either locally or on huggingface.co, you can
                specify the folder name here.
            local_files_only (`bool`, defaults to `False`):
                Whether or not to only look at local files (i.e., do not try to download the model).
            trust_remote_code (`bool`, defaults to `False`):
                Whether or not to allow for custom code defined on the Hub in their own modeling. This option should only be set
                to `True` for repositories you trust and in which you have read the code, as it will execute code present on
                the Hub on your local machine.
            task (`str | None`, defaults to `None`):
                The task to export the model for. If not specified, the task will be auto-inferred based on the model.
            auto_cast (`str | None`, defaults to `"matmul"`):
                Whether to cast operations from FP32 to lower precision to speed up the inference. Can be `"none"`, `"matmul"` or `"all"`.
            auto_cast_type (`str | None`, defaults to `"bf16"`):
                The data type to cast FP32 operations to when auto-cast mode is enabled. Can be `"bf16"`, `"fp16"` or `"tf32"`.
            dynamic_batch_size (`bool`, defaults to `False`):
                Whether to enable dynamic batch size for neuron compiled model. If this option is enabled, the input batch size can be a multiple of the
                batch size during the compilation, but it comes with a potential tradeoff in terms of latency.
            output_hidden_states (`bool`, defaults to `False`):
                Whether or not for the traced text encoders to return the hidden states of all layers.
            data_parallel_mode (`Literal["none", "unet", "transformer", "all"] | None`, defaults to `None`):
                Mode to decide what components to load into both NeuronCores of a Neuron device. Can be "none"(no data parallel), "unet"(only
                load unet into both cores of each device), "all"(load the whole pipeline into both cores).
            lora_model_ids (`str | list[str] | None`, defaults to `None`):
                Lora model local paths or repo ids (eg. `ostris/super-cereal-sdxl-lora`) on the Hugginface Hub.
            lora_weight_names (`str | list[str] | None`, defaults to `None`):
                Lora weights file names.
            lora_adapter_names (`str | list[str] | None`, defaults to `None`):
                Adapter names to be used for referencing the loaded adapter models.
            lora_scales (`list[float] | None`, defaults to `None`):
                Lora adapters scaling factors.
            controlnet_ids (`str | list[str] | None`, defaults to `None`):
                List of ControlNet model ids (eg. `thibaud/controlnet-openpose-sdxl-1.0`)."
            ip_adapter_ids (`str | list[str] | None`, defaults to `None`):
                Model ids (eg. `h94/IP-Adapter`) of IP-Adapter models hosted on the Hub or paths to local directories containing the IP-Adapter weights.
            ip_adapter_subfolders (`str | list[str | None]`, defaults to `None`):
                The subfolder location of a model file within a larger model repository on the Hub or locally. If a list is passed, it should have the same length as `ip_adapter_weight_names`.
            ip_adapter_weight_names (`str | list[str] | None`, defaults to `None`):
                The name of the weight file to load. If a list is passed, it should have the same length as `ip_adapter_subfolders`.
            ip_adapter_scales (`float | list[float] | None`, defaults to `None`):
                Scaling factors for the IP-Adapters.
        """
        # Parse kwargs to their dataclass
        parser = NeuronArgumentParser(**kwargs)
        lora_args = parser.lora_args
        ip_adapter_args = parser.ip_adapter_args
        kwargs_shapes = asdict(parser.input_shapes)

        if task is None:
            if cls.task is not None:
                task = cls.task
            else:
                task = TasksManager.infer_task_from_model(cls.auto_model_class)

        # mandatory shapes
        input_shapes = normalize_stable_diffusion_input_shapes(kwargs_shapes)

        # Get compilation arguments
        auto_cast_type = None if auto_cast is None else auto_cast_type
        compiler_kwargs = {
            "auto_cast": auto_cast,
            "auto_cast_type": auto_cast_type,
        }

        pipe = TasksManager.get_model_from_task(
            task=task,
            model_name_or_path=model_id,
            subfolder=subfolder,
            revision=revision,
            framework="pt",
            torch_dtype=torch_dtype,
            library_name=cls.library_name,
            cache_dir=cache_dir,
            token=token,
            local_files_only=local_files_only,
            force_download=force_download,
            trust_remote_code=trust_remote_code,
        )
        submodels = {"unet": unet_id}
        pipe = replace_stable_diffusion_submodels(pipe, submodels)

        # Check if the cache exists
        if not disable_neuron_cache:
            save_dir = TemporaryDirectory()
            save_dir_path = Path(save_dir.name)
            # 1. Fetch all model configs
            input_shapes_copy = copy.deepcopy(input_shapes)
            models_and_neuron_configs, _ = load_models_and_neuron_configs(
                model_name_or_path=model_id,
                output=save_dir_path,
                model=pipe,
                task=task,
                dynamic_batch_size=dynamic_batch_size,
                cache_dir=cache_dir,
                trust_remote_code=trust_remote_code,
                subfolder=subfolder,
                revision=revision,
                library_name=cls.library_name,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                submodels=submodels,
                lora_args=lora_args,
                ip_adapter_args=ip_adapter_args,
                output_hidden_states=output_hidden_states,
                torch_dtype=torch_dtype,
                controlnet_ids=controlnet_ids,
                **input_shapes_copy,
            )

            # 2. Build compilation config
            compilation_configs = {}
            for name, (model, neuron_config) in models_and_neuron_configs.items():
                if "vae" in name:  # vae configs are not cached.
                    continue
                model_config = model.config
                if isinstance(model_config, FrozenDict):
                    model_config = OrderedDict(model_config)
                    model_config = DiffusersPretrainedConfig.from_dict(model_config)

                compilation_config = store_compilation_config(
                    config=model_config,
                    input_shapes=neuron_config.input_shapes,
                    compiler_kwargs=compiler_kwargs,
                    input_names=neuron_config.inputs,
                    output_names=neuron_config.outputs,
                    dynamic_batch_size=neuron_config.dynamic_batch_size,
                    tensor_parallel_size=tensor_parallel_size,
                    compiler_type=NEURON_COMPILER_TYPE,
                    compiler_version=NEURON_COMPILER_VERSION,
                    inline_weights_to_neff=inline_weights_to_neff,
                    optlevel=optlevel,
                    model_type=getattr(neuron_config, "MODEL_TYPE", None),
                    task=getattr(neuron_config, "task", None),
                    output_attentions=output_attentions,
                    output_hidden_states=getattr(neuron_config, "output_hidden_states", False),
                )
                compilation_configs[name] = compilation_config

            # 3. Lookup cached config
            cache_entry = MultiModelCacheEntry(model_id=model_id, configs=compilation_configs)
            compile_cache = create_hub_compile_cache_proxy()
            model_cache_dir = compile_cache.default_cache.get_cache_dir_with_cache_key(f"MODULE_{cache_entry.hash}")
            cache_exist = compile_cache.download_folder(model_cache_dir, model_cache_dir)
        else:
            cache_exist = False

        if cache_exist:
            # load cache
            neuron_model = cls.from_pretrained(model_cache_dir, data_parallel_mode=data_parallel_mode)
            # replace weights
            if not inline_weights_to_neff:
                neuron_model.replace_weights(weights=pipe)
            return neuron_model
        else:
            # compile
            save_dir = TemporaryDirectory()
            save_dir_path = Path(save_dir.name)

            main_export(
                model_name_or_path=model_id,
                output=save_dir_path,
                compiler_kwargs=compiler_kwargs,
                lora_args=lora_args,
                ip_adapter_args=ip_adapter_args,
                torch_dtype=torch_dtype,
                task=task,
                dynamic_batch_size=dynamic_batch_size,
                cache_dir=cache_dir,
                disable_neuron_cache=disable_neuron_cache,
                compiler_workdir=compiler_workdir,
                inline_weights_to_neff=inline_weights_to_neff,
                optlevel=optlevel,
                trust_remote_code=trust_remote_code,
                subfolder=subfolder,
                revision=revision,
                force_download=force_download,
                local_files_only=local_files_only,
                token=token,
                do_validation=False,
                submodels={"unet": unet_id},
                output_hidden_states=output_hidden_states,
                controlnet_ids=controlnet_ids,
                library_name=cls.library_name,
                **input_shapes,
            )

        return cls._from_pretrained(
            model_id=save_dir_path,
            config=config,
            model_save_dir=save_dir,
            data_parallel_mode=data_parallel_mode,
        )

    @classmethod
    def _load_config(cls, config_name_or_path: str | os.PathLike, **kwargs):
        return cls.load_config(config_name_or_path, **kwargs)

    def _save_config(self, save_directory):
        self.save_config(save_directory)

    @property
    def components(self) -> dict[str, Any]:
        components = {
            "vae_encoder": self.vae_encoder,
            "vae_decoder": self.vae_decoder,
            "unet": self.unet,
            "transformer": self.transformer,
            "text_encoder": self.text_encoder,
            "text_encoder_2": self.text_encoder_2,
            "image_encoder": self.image_encoder,
            "safety_checker": self.safety_checker,
            "neuron_configs": self.neuron_configs,
            "data_parallel_mode": self.data_parallel_mode,
            "feature_extractor": self.feature_extractor,
            "configs": self.configs,
            "config": self.config,
            "tokenizer": self.tokenizer,
            "tokenizer_2": self.tokenizer_2,
            "scheduler": self.scheduler,
        }
        return components

    @property
    def do_classifier_free_guidance(self):
        return (
            self._guidance_scale > 1
            and self.unet.config.time_cond_proj_dim is None
            and (
                self.dynamic_batch_size
                or self.data_parallel_mode == "unet"
                or self.data_parallel_mode == "transformer"
            )
        )

    def _maybe_create_dummy_image_proj_layers(self):
        if all([self.image_encoder, self.encoder_hid_proj]):
            self.unet.encoder_hid_proj = self.encoder_hid_proj

    def __call__(self, *args, **kwargs):
        # Height and width to unet/transformer (static shapes)
        unet_or_transformer = self.unet or self.transformer
        height = unet_or_transformer.config.neuron["static_height"] * self.vae_scale_factor
        width = unet_or_transformer.config.neuron["static_width"] * self.vae_scale_factor
        kwargs.pop("height", None)
        kwargs.pop("width", None)
        if kwargs.get("image", None):
            kwargs["image"] = self.image_processor.preprocess(kwargs["image"], height=height, width=width)
        # Override default `max_sequence_length`, eg. pixart
        if "max_sequence_length" in inspect.signature(self.auto_model_class.__call__).parameters:
            kwargs["max_sequence_length"] = self.text_encoder.config.neuron.get("static_sequence_length", None)
        return self.auto_model_class.__call__(self, height=height, width=width, *args, **kwargs)


class _NeuronDiffusionModelPart:
    """
    For multi-file Neuron models, represents a part / a model in the pipeline.
    """

    def __init__(
        self,
        model: torch.jit._script.ScriptModule,
        parent_pipeline: NeuronDiffusionPipelineBase,
        config: DiffusersPretrainedConfig | PretrainedConfig | None = None,
        neuron_config: "NeuronDefaultConfig | None" = None,
        model_type: str = "unet",
        device: int | None = None,
    ):
        self.model = model
        self.parent_pipeline = parent_pipeline
        self.config = config
        self.neuron_config = neuron_config
        self.model_type = model_type
        self.device = device

    @abstractmethod
    def forward(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    @property
    def dtype(self):
        return None

    def to(self, *args, **kwargs):
        pass


class NeuronModelTextEncoder(_NeuronDiffusionModelPart):
    def __init__(
        self,
        model: torch.jit._script.ScriptModule,
        parent_pipeline: NeuronDiffusionPipelineBase,
        config: DiffusersPretrainedConfig | None = None,
        neuron_config: dict[str, str] | None = None,
    ):
        super().__init__(model, parent_pipeline, config, neuron_config, DIFFUSION_MODEL_TEXT_ENCODER_NAME)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = True,
    ):
        if output_hidden_states:
            assert (
                self.config.output_hidden_states or self.config.neuron.get("output_hidden_states")
            ) == output_hidden_states, (
                "output_hidden_states is expected to be False since the model was compiled without hidden_states as output."
            )

        input_ids = input_ids.to(torch.long)  # dummy generator uses long int for tracing
        inputs = (input_ids,)
        if attention_mask is not None and not torch.all(attention_mask == 1):
            inputs += (attention_mask,)

        outputs = self.model(*inputs)

        if return_dict:
            outputs = ModelOutput(dict(zip(self.neuron_config.outputs, outputs)))

        return outputs

    def modules(self):
        # dummy func for passing `unscale_lora_layers`.
        return []


class NeuronModelImageEncoder(_NeuronDiffusionModelPart):
    def __init__(
        self,
        model: torch.jit._script.ScriptModule,
        parent_pipeline: NeuronDiffusionPipelineBase,
        config: DiffusersPretrainedConfig | None = None,
        neuron_config: dict[str, str] | None = None,
    ):
        super().__init__(model, parent_pipeline, config, neuron_config, DIFFUSION_MODEL_IMAGE_ENCODER_NAME)

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        interpolate_pos_encoding: bool = False,
        return_dict: bool | None = True,
    ):
        inputs = (pixel_values,)

        outputs = self.model(*inputs)

        if return_dict:
            outputs = ModelOutput(dict(zip(self.neuron_config.outputs, outputs)))

        return outputs

    # Create a dummy parameters to be compatible with `https://github.com/huggingface/diffusers/blob/c14057c8dbc32847bac9082bcc0ae00c9a19357d/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L514`
    def parameters(self):
        class DummyObject:
            def __init__(self):
                self.dtype = None

        return iter([DummyObject()])


class NeuronModelUnet(_NeuronDiffusionModelPart):
    def __init__(
        self,
        model: torch.jit._script.ScriptModule,
        parent_pipeline: NeuronDiffusionPipelineBase,
        config: DiffusersPretrainedConfig | None = None,
        neuron_config: dict[str, str] | None = None,
    ):
        super().__init__(model, parent_pipeline, config, neuron_config, DIFFUSION_MODEL_UNET_NAME)
        if hasattr(self.model, "device"):
            self.device = self.model.device

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs: dict[str, torch.Tensor] | None = None,
        down_block_additional_residuals: tuple[torch.Tensor] | None = None,
        mid_block_additional_residual: torch.Tensor | None = None,
        timestep_cond: torch.Tensor | None = None,
        cross_attention_kwargs: dict[str, Any] | None = None,
        return_dict: bool = True,
    ):
        if cross_attention_kwargs is not None:
            logger.warning("`cross_attention_kwargs` is not yet supported during the tracing and it will be ignored.")
        timestep = timestep.float().expand((sample.shape[0],))
        inputs = (sample, timestep, encoder_hidden_states)
        if timestep_cond is not None:
            inputs = inputs + (timestep_cond,)
        if mid_block_additional_residual is not None:
            inputs = inputs + (mid_block_additional_residual,)
        if down_block_additional_residuals is not None:
            for idx in range(len(down_block_additional_residuals)):
                inputs = inputs + (down_block_additional_residuals[idx],)
        if added_cond_kwargs:
            optional_inputs_names = ["text_embeds", "time_ids", "image_embeds"]
            for optional_input_name in optional_inputs_names:
                optional_input = added_cond_kwargs.get(optional_input_name, None)
                if isinstance(optional_input, list):
                    optional_input = (
                        torch.stack(optional_input, dim=0) if len(optional_input) > 1 else optional_input[0]
                    )
                if optional_input is not None:
                    inputs = inputs + (optional_input,)

        outputs = self.model(*inputs)
        if return_dict:
            outputs = ModelOutput(dict(zip(self.neuron_config.outputs, outputs)))
        return outputs


class NeuronModelTransformer(_NeuronDiffusionModelPart):
    def __init__(
        self,
        model: torch.jit._script.ScriptModule,
        parent_pipeline: NeuronDiffusionPipelineBase,
        config: DiffusersPretrainedConfig | None = None,
        neuron_config: dict[str, str] | None = None,
    ):
        super().__init__(model, parent_pipeline, config, neuron_config, DIFFUSION_MODEL_TRANSFORMER_NAME)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        timestep: torch.LongTensor | None = None,
        added_cond_kwargs: dict[str, torch.Tensor] | None = None,
        cross_attention_kwargs: dict[str, Any] | None = None,
        attention_mask: torch.Tensor | None = None,
        encoder_attention_mask: torch.Tensor | None = None,
        return_dict: bool = True,
    ):
        inputs = (hidden_states, encoder_hidden_states, timestep, encoder_attention_mask)
        outputs = self.model(*inputs)
        if return_dict:
            outputs = ModelOutput(dict(zip(self.neuron_config.outputs, outputs)))
        return outputs


class NeuronModelVaeEncoder(_NeuronDiffusionModelPart):
    def __init__(
        self,
        model: torch.jit._script.ScriptModule,
        parent_pipeline: NeuronDiffusionPipelineBase,
        config: DiffusersPretrainedConfig | None = None,
        neuron_config: dict[str, str] | None = None,
    ):
        super().__init__(model, parent_pipeline, config, neuron_config, DIFFUSION_MODEL_VAE_ENCODER_NAME)

    def forward(self, sample: torch.Tensor, return_dict: bool = True):
        inputs = (sample,)
        outputs = self.model(*inputs)

        if "latent_parameters" in outputs:
            outputs["latent_dist"] = DiagonalGaussianDistribution(parameters=outputs.pop("latent_parameters"))

        if not return_dict:
            return tuple(output for output in outputs.values())
        else:
            return AutoencoderKLOutput(latent_dist=outputs["latent_dist"])


class NeuronModelVaeDecoder(_NeuronDiffusionModelPart):
    def __init__(
        self,
        model: torch.jit._script.ScriptModule,
        parent_pipeline: NeuronDiffusionPipelineBase,
        config: DiffusersPretrainedConfig | None = None,
        neuron_config: dict[str, str | None] = None,
    ):
        super().__init__(model, parent_pipeline, config, neuron_config, DIFFUSION_MODEL_VAE_DECODER_NAME)

    def forward(
        self,
        latent_sample: torch.Tensor,
        image: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        return_dict: bool = True,
        generator=None,
    ):
        inputs = (latent_sample,)
        if image is not None:
            inputs += (image,)
        if mask is not None:
            inputs += (mask,)
        outputs = self.model(*inputs)

        if not return_dict:
            return tuple(output for output in outputs.values())
        else:
            return DecoderOutput(**outputs)


class NeuronModelVae(_NeuronDiffusionModelPart):
    def __init__(
        self,
        encoder: NeuronModelVaeEncoder | None,
        decoder: NeuronModelVaeDecoder,
    ):
        self.encoder = encoder
        self.decoder = decoder

    @property
    def config(self):
        return self.decoder.config

    def encode(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)

    def decode(self, *args, **kwargs):
        return self.decoder(*args, **kwargs)


class NeuronControlNetModel(_NeuronDiffusionModelPart):
    auto_model_class = ControlNetModel
    library_name = "diffusers"
    base_model_prefix = "neuron_model"
    config_name = "model_index.json"
    sub_component_config_name = "config.json"

    def __init__(
        self,
        model: torch.jit._script.ScriptModule,
        parent_pipeline: NeuronDiffusionPipelineBase,
        config: DiffusersPretrainedConfig | None = None,
        neuron_config: dict[str, str | None] = None,
    ):
        super().__init__(model, parent_pipeline, config, neuron_config, DIFFUSION_MODEL_CONTROLNET_NAME)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        guess_mode: bool = False,
        added_cond_kwargs: dict | None = None,
        return_dict: bool = True,
    ) -> "ControlNetOutput | tuple[tuple[torch.Tensor, ...]]":
        timestep = timestep.expand((sample.shape[0],)).to(torch.long)
        inputs = (sample, timestep, encoder_hidden_states, controlnet_cond, conditioning_scale)
        if added_cond_kwargs:
            text_embeds = added_cond_kwargs.pop("text_embeds", None)
            time_ids = added_cond_kwargs.pop("time_ids", None)
            inputs += (text_embeds, time_ids)
        outputs = self.model(*inputs)

        if guess_mode:
            logger.info(
                "Guess mode is not yet supported. File us an issue on: https://github.com/huggingface/optimum-neuron/issues."
            )

        if return_dict:
            outputs = ControlNetOutput(dict(zip(self.neuron_config.outputs, outputs)))

        return outputs

    @property
    def __class__(self):
        return ControlNetModel


class NeuronMultiControlNetModel(_NeuronDiffusionModelPart):
    auto_model_class = MultiControlNetModel
    library_name = "diffusers"
    base_model_prefix = "neuron_model"
    config_name = "model_index.json"
    sub_component_config_name = "config.json"

    def __init__(
        self,
        models: list[torch.jit._script.ScriptModule],
        parent_pipeline: NeuronTracedModel,
        config: DiffusersPretrainedConfig | None = None,
        neuron_config: dict[str, str] | None = None,
    ):
        self.nets = models
        self.parent_pipeline = parent_pipeline
        self.config = config
        self.neuron_config = neuron_config
        self.model_type = DIFFUSION_MODEL_CONTROLNET_NAME
        self.device = None

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor | float | int,
        encoder_hidden_states: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: list[float],
        guess_mode: bool = False,
        return_dict: bool = True,
    ) -> "ControlNetOutput | tuple[tuple[torch.Tensor, ...]]":
        if guess_mode:
            logger.info(
                "Guess mode is not yet supported. File us an issue on: https://github.com/huggingface/optimum-neuron/issues."
            )
        for i, (image, scale, controlnet) in enumerate(zip(controlnet_cond, conditioning_scale, self.nets)):
            inputs = (sample, timestep, encoder_hidden_states, image, scale)
            down_samples, mid_sample = controlnet(*inputs)

            # merge samples
            if i == 0:
                down_block_res_samples, mid_block_res_sample = down_samples, mid_sample
            else:
                down_block_res_samples = [
                    samples_prev + samples_curr
                    for samples_prev, samples_curr in zip(down_block_res_samples, down_samples)
                ]
                mid_block_res_sample += mid_sample

        if return_dict:
            return ControlNetOutput(
                down_block_res_samples=down_block_res_samples, mid_block_res_sample=mid_block_res_sample
            )

        return down_block_res_samples, mid_block_res_sample

    @property
    def __class__(self):
        return MultiControlNetModel


class NeuronStableDiffusionPipeline(NeuronDiffusionPipelineBase, StableDiffusionPipeline):
    main_input_name = "prompt"
    auto_model_class = StableDiffusionPipeline

    class DummyEncoderHidProj:
        def __init__(self):
            self.image_projection_layers = ModuleList(
                [IPAdapterFullImageProjection()]
            )  # TODO: support multiple IP adapters

    encoder_hid_proj = DummyEncoderHidProj()


class NeuronStableDiffusionImg2ImgPipeline(NeuronDiffusionPipelineBase, StableDiffusionImg2ImgPipeline):
    main_input_name = "image"
    auto_model_class = StableDiffusionImg2ImgPipeline


class NeuronStableDiffusionInpaintPipeline(NeuronDiffusionPipelineBase, StableDiffusionInpaintPipeline):
    main_input_name = "prompt"
    auto_model_class = StableDiffusionInpaintPipeline


class NeuronStableDiffusionInstructPix2PixPipeline(
    NeuronDiffusionPipelineBase, StableDiffusionInstructPix2PixPipeline
):
    main_input_name = "prompt"
    task = "text-to-image"
    auto_model_class = StableDiffusionInstructPix2PixPipeline


class NeuronLatentConsistencyModelPipeline(NeuronDiffusionPipelineBase, LatentConsistencyModelPipeline):
    main_input_name = "prompt"
    auto_model_class = LatentConsistencyModelPipeline


class NeuronStableDiffusionControlNetPipeline(
    NeuronStableDiffusionControlNetPipelineMixin, NeuronDiffusionPipelineBase, StableDiffusionControlNetPipeline
):
    main_input_name = "prompt"
    auto_model_class = StableDiffusionControlNetPipeline


class NeuronPixArtAlphaPipeline(NeuronDiffusionPipelineBase, PixArtAlphaPipeline):
    main_input_name = "prompt"
    auto_model_class = PixArtAlphaPipeline

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_processor = PixArtImageProcessor(vae_scale_factor=self.vae_scale_factor)


class NeuronPixArtSigmaPipeline(NeuronDiffusionPipelineBase, PixArtSigmaPipeline):
    main_input_name = "prompt"
    auto_model_class = PixArtSigmaPipeline

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.image_processor = PixArtImageProcessor(vae_scale_factor=self.vae_scale_factor)


class NeuronStableDiffusionXLPipeline(
    NeuronStableDiffusionXLPipelineMixin, NeuronDiffusionPipelineBase, StableDiffusionXLPipeline
):
    main_input_name = "prompt"
    auto_model_class = StableDiffusionXLPipeline

    class DummyEncoderHidProj:
        def __init__(self):
            self.image_projection_layers = ModuleList([ImageProjection()])  # TODO: support multiple IP adapters

    encoder_hid_proj = DummyEncoderHidProj()


class NeuronStableDiffusionXLImg2ImgPipeline(
    NeuronStableDiffusionXLPipelineMixin, NeuronDiffusionPipelineBase, StableDiffusionXLImg2ImgPipeline
):
    main_input_name = "prompt"
    auto_model_class = StableDiffusionXLImg2ImgPipeline


class NeuronStableDiffusionXLInpaintPipeline(
    NeuronStableDiffusionXLPipelineMixin, NeuronDiffusionPipelineBase, StableDiffusionXLInpaintPipeline
):
    main_input_name = "image"
    auto_model_class = StableDiffusionXLInpaintPipeline


class NeuronStableDiffusionXLControlNetPipeline(
    NeuronStableDiffusionXLControlNetPipelineMixin, NeuronDiffusionPipelineBase, StableDiffusionXLControlNetPipeline
):
    main_input_name = "prompt"
    auto_model_class = StableDiffusionXLControlNetPipeline
