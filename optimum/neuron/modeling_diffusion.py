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

import importlib
import logging
import os
import shutil
from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import torch
from huggingface_hub import snapshot_download
from transformers import CLIPFeatureExtractor, CLIPTokenizer, PretrainedConfig

from ..exporters.neuron import DiffusersPretrainedConfig, main_export, normalize_stable_diffusion_input_shapes
from ..exporters.neuron.model_configs import *  # noqa: F403
from ..exporters.tasks import TasksManager
from ..utils import is_diffusers_available
from .modeling_base import NeuronBaseModel
from .utils import (
    DIFFUSION_MODEL_TEXT_ENCODER_2_NAME,
    DIFFUSION_MODEL_TEXT_ENCODER_NAME,
    DIFFUSION_MODEL_UNET_NAME,
    DIFFUSION_MODEL_VAE_DECODER_NAME,
    DIFFUSION_MODEL_VAE_ENCODER_NAME,
    NEURON_FILE_NAME,
    is_neuronx_available,
)


if is_neuronx_available():
    import torch_neuronx


if is_diffusers_available():
    from diffusers import (
        DDIMScheduler,
        LMSDiscreteScheduler,
        PNDMScheduler,
        StableDiffusionPipeline,
        StableDiffusionXLImg2ImgPipeline,
    )
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
    from diffusers.utils import CONFIG_NAME, is_invisible_watermark_available

    from .pipelines import (
        NeuronStableDiffusionImg2ImgPipelineMixin,
        NeuronStableDiffusionInpaintPipelineMixin,
        NeuronStableDiffusionPipelineMixin,
        NeuronStableDiffusionXLImg2ImgPipelineMixin,
        NeuronStableDiffusionXLInpaintPipelineMixin,
        NeuronStableDiffusionXLPipelineMixin,
    )


if TYPE_CHECKING:
    from ..exporters.neuron import NeuronConfig


logger = logging.getLogger(__name__)


class NeuronStableDiffusionPipelineBase(NeuronBaseModel):
    auto_model_class = StableDiffusionPipeline
    base_model_prefix = "neuron_model"
    config_name = "model_index.json"
    sub_component_config_name = "config.json"

    def __init__(
        self,
        text_encoder: torch.jit._script.ScriptModule,
        unet: torch.jit._script.ScriptModule,
        vae_decoder: Union[torch.jit._script.ScriptModule, "NeuronModelVaeDecoder"],
        config: Dict[str, Any],
        tokenizer: CLIPTokenizer,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        vae_encoder: Optional[Union[torch.jit._script.ScriptModule, "NeuronModelVaeEncoder"]] = None,
        text_encoder_2: Optional[Union[torch.jit._script.ScriptModule, "NeuronModelTextEncoder"]] = None,
        tokenizer_2: Optional[CLIPTokenizer] = None,
        feature_extractor: Optional[CLIPFeatureExtractor] = None,
        device_ids: Optional[List[int]] = None,
        configs: Optional[Dict[str, "PretrainedConfig"]] = None,
        neuron_configs: Optional[Dict[str, "NeuronConfig"]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        model_and_config_save_paths: Optional[Dict[str, Tuple[str, Path]]] = None,
    ):
        """
        Args:
            text_encoder (`torch.jit._script.ScriptModule`):
                The Neuron TorchScript module associated to the text encoder.
            unet (`torch.jit._script.ScriptModule`):
                The Neuron TorchScript module associated to the U-NET.
            vae_decoder (`torch.jit._script.ScriptModule`):
                The Neuron TorchScript module associated to the VAE decoder.
            config (`Dict[str, Any]`):
                A config dictionary from which the model components will be instantiated. Make sure to only load
                configuration files of compatible classes.
            tokenizer (`CLIPTokenizer`):
                Tokenizer of class
                [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
            scheduler (`Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler]`):
                A scheduler to be used in combination with the U-NET component to denoise the encoded image latents.
            vae_encoder (`Optional[torch.jit._script.ScriptModule]`, defaults to `None`):
                The Neuron TorchScript module associated to the VAE encoder.
            text_encoder_2 (`Optional[torch.jit._script.ScriptModule]`, defaults to `None`):
                The Neuron TorchScript module associated to the second frozen text encoder. Stable Diffusion XL uses the text and pool portion of [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModelWithProjection), specifically the [laion/CLIP-ViT-bigG-14-laion2B-39B-b160k](https://huggingface.co/laion/CLIP-ViT-bigG-14-laion2B-39B-b160k) variant.
            tokenizer_2 (`Optional[CLIPTokenizer]`, defaults to `None`):
                Second tokenizer of class
                [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
            feature_extractor (`Optional[CLIPFeatureExtractor]`, defaults to `None`):
                A model extracting features from generated images to be used as inputs for the `safety_checker`
            device_ids (Optional[List[int]], defaults to `None`):
                A list of integers that specify the NeuronCores to use for parallelization
            configs (Optional[Dict[str, "PretrainedConfig"]], defaults to `None`):
                A dictionary configurations for components of the pipeline.
            neuron_configs (Optional["NeuronConfig"], defaults to `None`):
                A list of Neuron configurations.
            model_save_dir (`Optional[Union[str, Path, TemporaryDirectory]]`, defaults to `None`):
                The directory under which the exported Neuron models were saved.
            model_and_config_save_paths (`Optional[Dict[str, Tuple[str, Path]]]`, defaults to `None`):
                The paths where exported Neuron models were saved.
        """

        self._internal_dict = config
        self.device_ids = device_ids
        self.configs = configs
        self.neuron_configs = neuron_configs
        self.dynamic_batch_size = all(
            neuron_config._config.neuron["dynamic_batch_size"] for neuron_config in self.neuron_configs.values()
        )

        self.text_encoder = (
            NeuronModelTextEncoder(
                text_encoder,
                self,
                self.configs[DIFFUSION_MODEL_TEXT_ENCODER_NAME],
                self.neuron_configs[DIFFUSION_MODEL_TEXT_ENCODER_NAME],
            )
            if text_encoder is not None
            else None
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
        self.unet = NeuronModelUnet(
            unet, self, self.configs[DIFFUSION_MODEL_UNET_NAME], self.neuron_configs[DIFFUSION_MODEL_UNET_NAME]
        )
        if vae_encoder is not None and not isinstance(vae_encoder, NeuronModelVaeEncoder):
            self.vae_encoder = NeuronModelVaeEncoder(
                vae_encoder,
                self,
                self.configs[DIFFUSION_MODEL_VAE_ENCODER_NAME],
                self.neuron_configs[DIFFUSION_MODEL_VAE_ENCODER_NAME],
            )
        else:
            self.vae_encoder = vae_encoder

        if vae_decoder is not None and not isinstance(vae_decoder, NeuronModelVaeDecoder):
            self.vae_decoder = NeuronModelVaeDecoder(
                vae_decoder,
                self,
                self.configs[DIFFUSION_MODEL_VAE_DECODER_NAME],
                self.neuron_configs[DIFFUSION_MODEL_VAE_DECODER_NAME],
            )
        else:
            self.vae_decoder = vae_decoder

        self.tokenizer = tokenizer
        self.tokenizer_2 = tokenizer_2
        self.scheduler = scheduler
        self.feature_extractor = feature_extractor
        self.safety_checker = None
        sub_models = {
            DIFFUSION_MODEL_TEXT_ENCODER_NAME: self.text_encoder,
            DIFFUSION_MODEL_UNET_NAME: self.unet,
            DIFFUSION_MODEL_VAE_DECODER_NAME: self.vae_decoder,
        }
        if self.text_encoder_2 is not None:
            sub_models[DIFFUSION_MODEL_TEXT_ENCODER_2_NAME] = self.text_encoder_2
        if self.vae_encoder is not None:
            sub_models[DIFFUSION_MODEL_VAE_ENCODER_NAME] = self.vae_encoder

        for name in sub_models.keys():
            self._internal_dict[name] = ("optimum", sub_models[name].__class__.__name__)
        self._internal_dict.pop("vae", None)

        self._attributes_init(model_save_dir)
        self.model_and_config_save_paths = model_and_config_save_paths if model_and_config_save_paths else None

        if hasattr(self.vae_decoder.config, "block_out_channels"):
            self.vae_scale_factor = 2 ** (len(self.vae_decoder.config.block_out_channels) - 1)
        else:
            self.vae_scale_factor = 8

        unet_batch_size = self.neuron_configs["unet"].batch_size
        if "text_encoder" in self.neuron_configs:
            text_encoder_batch_size = self.neuron_configs["text_encoder"].batch_size
            self.num_images_per_prompt = unet_batch_size // text_encoder_batch_size
        elif "text_encoder_2" in self.neuron_configs:
            text_encoder_batch_size = self.neuron_configs["text_encoder_2"].batch_size
            self.num_images_per_prompt = unet_batch_size // text_encoder_batch_size
        else:
            self.num_images_per_prompt = 1

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    @staticmethod
    def load_model(
        text_encoder_path: Union[str, Path],
        unet_path: Union[str, Path],
        vae_decoder_path: Optional[Union[str, Path]] = None,
        vae_encoder_path: Optional[Union[str, Path]] = None,
        text_encoder_2_path: Optional[Union[str, Path]] = None,
        device_ids: Optional[List[int]] = None,
        dynamic_batch_size: bool = False,
    ):
        """
        Loads Stable Diffusion TorchScript modules compiled by neuron(x)-cc compiler. It will be first loaded onto CPU and then moved to
        one or multiple [NeuronCore](https://awsdocs-neuron.readthedocs-hosted.com/en/latest/general/arch/neuron-hardware/neuroncores-arch.html).

        Args:
            text_encoder_path (`Union[str, Path]`):
                Path of the compiled text encoder.
            unet_path (`Union[str, Path]`):
                Path of the compiled U-NET.
            vae_decoder_path (`Optional[Union[str, Path]]`, defaults to `None`):
                Path of the compiled VAE decoder.
            vae_encoder_path (`Optional[Union[str, Path]]`, defaults to `None`):
                Path of the compiled VAE encoder. It is optional, only used for tasks taking images as input.
            text_encoder_2_path (`Optional[Union[str, Path]]`, defaults to `None`):
                Path of the compiled second frozen text encoder. SDXL only.
            device_ids (`Optional[List[int]]`, defaults to `None`):
                The ID of neuron cores to load a model, in the case of stable diffusion, it is only used for loading unet, and by default unet will be loaded onto both neuron cores of a device.
            dynamic_batch_size (`bool`, defaults to `False`):
                Whether enable dynamic batch size for neuron compiled model. If `True`, the input batch size can be a multiple of the batch size during the compilation.
        """
        if device_ids is None:
            device_ids = [0, 1]

        text_encoder = NeuronBaseModel.load_model(text_encoder_path)
        if len(device_ids) > 1:
            unet = torch_neuronx.DataParallel(
                torch.jit.load(unet_path),
                device_ids,
                set_dynamic_batching=dynamic_batch_size,
            )
        else:
            unet = NeuronBaseModel.load_model(unet_path)

        vae_decoder = NeuronBaseModel.load_model(vae_decoder_path) if vae_decoder_path is not None else None
        vae_encoder = NeuronBaseModel.load_model(vae_encoder_path) if vae_encoder_path is not None else None
        text_encoder_2 = NeuronBaseModel.load_model(text_encoder_2_path) if text_encoder_2_path is not None else None

        return {
            "text_encoder": text_encoder,
            "unet": unet,
            "vae_decoder": vae_decoder,
            "vae_encoder": vae_encoder,
            "text_encoder_2": text_encoder_2,
        }

    def _save_pretrained(
        self,
        save_directory: Union[str, Path],
        text_encoder_file_name: str = NEURON_FILE_NAME,
        text_encoder_2_file_name: str = NEURON_FILE_NAME,
        unet_file_name: str = NEURON_FILE_NAME,
        vae_encoder_file_name: str = NEURON_FILE_NAME,
        vae_decoder_file_name: str = NEURON_FILE_NAME,
    ):
        """
        Saves the model to the serialized format optimized for Neuron devices.
        """
        save_directory = Path(save_directory)
        if not self.model_and_config_save_paths.get(DIFFUSION_MODEL_VAE_ENCODER_NAME)[0].is_file():
            self.model_and_config_save_paths.pop(DIFFUSION_MODEL_VAE_ENCODER_NAME)

        if not self.model_and_config_save_paths.get(DIFFUSION_MODEL_TEXT_ENCODER_NAME)[0].is_file():
            self.model_and_config_save_paths.pop(DIFFUSION_MODEL_TEXT_ENCODER_NAME)

        if not self.model_and_config_save_paths.get(DIFFUSION_MODEL_TEXT_ENCODER_2_NAME)[0].is_file():
            self.model_and_config_save_paths.pop(DIFFUSION_MODEL_TEXT_ENCODER_2_NAME)

        if self.model_and_config_save_paths is None:
            logger.warning(
                "`model_save_paths` is None which means that no path of Neuron model is defined. Nothing will be saved."
            )
            return
        else:
            logger.info(f"Saving the {tuple(self.model_and_config_save_paths.keys())}...")

        dst_paths = {
            DIFFUSION_MODEL_TEXT_ENCODER_NAME: save_directory
            / DIFFUSION_MODEL_TEXT_ENCODER_NAME
            / text_encoder_file_name,
            DIFFUSION_MODEL_TEXT_ENCODER_2_NAME: save_directory
            / DIFFUSION_MODEL_TEXT_ENCODER_2_NAME
            / text_encoder_2_file_name,
            DIFFUSION_MODEL_UNET_NAME: save_directory / DIFFUSION_MODEL_UNET_NAME / unet_file_name,
            DIFFUSION_MODEL_VAE_ENCODER_NAME: save_directory
            / DIFFUSION_MODEL_VAE_ENCODER_NAME
            / vae_encoder_file_name,
            DIFFUSION_MODEL_VAE_DECODER_NAME: save_directory
            / DIFFUSION_MODEL_VAE_DECODER_NAME
            / vae_decoder_file_name,
        }
        model_src_to_dst_path = {
            self.model_and_config_save_paths[model_name][0]: dst_paths[model_name]
            for model_name in set(self.model_and_config_save_paths.keys()).intersection(dst_paths.keys())
        }
        # save
        config_src_to_dst_path = {
            self.model_and_config_save_paths[model_name][1]: dst_paths[model_name].parent / CONFIG_NAME
            for model_name in set(self.model_and_config_save_paths.keys()).intersection(dst_paths.keys())
        }

        src_paths = list(model_src_to_dst_path.keys()) + list(config_src_to_dst_path.keys())
        dst_paths = list(model_src_to_dst_path.values()) + list(config_src_to_dst_path.values())

        for src_path, dst_path in zip(src_paths, dst_paths):
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
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: Dict[str, Any],
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        text_encoder_file_name: Optional[str] = NEURON_FILE_NAME,
        text_encoder_2_file_name: Optional[str] = NEURON_FILE_NAME,
        unet_file_name: Optional[str] = NEURON_FILE_NAME,
        vae_encoder_file_name: Optional[str] = NEURON_FILE_NAME,
        vae_decoder_file_name: Optional[str] = NEURON_FILE_NAME,
        text_encoder_2: Optional["NeuronModelTextEncoder"] = None,
        vae_encoder: Optional["NeuronModelVaeEncoder"] = None,
        vae_decoder: Optional["NeuronModelVaeDecoder"] = None,
        local_files_only: bool = False,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        device_ids: Optional[List[int]] = None,
        **kwargs,  # To share kwargs only available for `_from_transformers`
    ):
        model_id = str(model_id)
        patterns = set(config.keys())
        sub_models_to_load = patterns.intersection({"feature_extractor", "tokenizer", "tokenizer_2", "scheduler"})

        if not os.path.isdir(model_id):
            patterns.update({DIFFUSION_MODEL_VAE_ENCODER_NAME, DIFFUSION_MODEL_VAE_DECODER_NAME})
            allow_patterns = {os.path.join(k, "*") for k in patterns if not k.startswith("_")}
            allow_patterns.update(
                {
                    text_encoder_file_name,
                    text_encoder_2_file_name,
                    unet_file_name,
                    vae_encoder_file_name,
                    vae_decoder_file_name,
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
                use_auth_token=use_auth_token,
                revision=revision,
                allow_patterns=allow_patterns,
                ignore_patterns=["*.msgpack", "*.safetensors", "*.bin"],
            )

        new_model_save_dir = Path(model_id)
        sub_models = {}
        for name in sub_models_to_load:
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
            "unet": (
                new_model_save_dir / DIFFUSION_MODEL_UNET_NAME / unet_file_name,
                new_model_save_dir / DIFFUSION_MODEL_UNET_NAME / cls.sub_component_config_name,
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

        # Re-build pretrained configs and neuron configs
        configs, neuron_configs = {}, {}
        for name, file_paths in model_and_config_save_paths.items():
            if file_paths[1].is_file():
                model_config = DiffusersPretrainedConfig.from_json_file(file_paths[1])
                configs[name] = model_config
                neuron_configs[name] = cls._neuron_config_init(model_config)

        pipe = cls.load_model(
            text_encoder_path=model_and_config_save_paths["text_encoder"][0],
            unet_path=model_and_config_save_paths["unet"][0],
            vae_decoder_path=model_and_config_save_paths["vae_decoder"][0] if vae_decoder is None else None,
            vae_encoder_path=model_and_config_save_paths["vae_encoder"][0] if vae_encoder is None else None,
            text_encoder_2_path=model_and_config_save_paths["text_encoder_2"][0] if text_encoder_2 is None else None,
            device_ids=device_ids,
            dynamic_batch_size=neuron_configs[DIFFUSION_MODEL_UNET_NAME].dynamic_batch_size,
        )

        if model_save_dir is None:
            model_save_dir = new_model_save_dir

        return cls(
            text_encoder=pipe.get("text_encoder"),
            unet=pipe.get("unet"),
            vae_decoder=vae_decoder or pipe.get("vae_decoder"),
            config=config,
            tokenizer=sub_models.get("tokenizer", None),
            scheduler=sub_models.get("scheduler"),
            vae_encoder=vae_encoder or pipe.get("vae_encoder"),
            text_encoder_2=text_encoder_2 or pipe.get("text_encoder_2"),
            tokenizer_2=sub_models.get("tokenizer_2", None),
            feature_extractor=sub_models.get("feature_extractor", None),
            device_ids=device_ids,
            configs=configs,
            neuron_configs=neuron_configs,
            model_save_dir=model_save_dir,
            model_and_config_save_paths=model_and_config_save_paths,
        )

    @classmethod
    def _from_transformers(
        cls,
        model_id: str,
        config: Dict[str, Any],
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
        **kwargs_shapes,
    ) -> "NeuronStableDiffusionPipelineBase":
        if task is None:
            task = TasksManager.infer_task_from_model(cls.auto_model_class)

        # mandatory shapes
        input_shapes = normalize_stable_diffusion_input_shapes(kwargs_shapes)

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
            **input_shapes,
        )

        return cls._from_pretrained(
            model_id=save_dir_path,
            config=config,
            model_save_dir=save_dir,
            device_ids=device_ids,
        )

    @classmethod
    def _load_config(cls, config_name_or_path: Union[str, os.PathLike], **kwargs):
        return cls.load_config(config_name_or_path, **kwargs)

    def _save_config(self, save_directory):
        self.save_config(save_directory)


class _NeuronDiffusionModelPart:
    """
    For multi-file Neuron models, represents a part of the model.
    """

    def __init__(
        self,
        model: torch.jit._script.ScriptModule,
        parent_model: NeuronBaseModel,
        config: Optional[Union[DiffusersPretrainedConfig, PretrainedConfig]] = None,
        neuron_config: Optional["NeuronConfig"] = None,
        model_type: str = "unet",
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


class NeuronModelTextEncoder(_NeuronDiffusionModelPart):
    def __init__(
        self,
        model: torch.jit._script.ScriptModule,
        parent_model: NeuronBaseModel,
        config: Optional[DiffusersPretrainedConfig] = None,
        neuron_config: Optional[Dict[str, str]] = None,
    ):
        super().__init__(model, parent_model, config, neuron_config, DIFFUSION_MODEL_TEXT_ENCODER_NAME)

    def forward(self, input_ids: torch.Tensor):
        inputs = (input_ids,)
        outputs = self.model(*inputs)
        return outputs


class NeuronModelUnet(_NeuronDiffusionModelPart):
    def __init__(
        self,
        model: torch.jit._script.ScriptModule,
        parent_model: NeuronBaseModel,
        config: Optional[DiffusersPretrainedConfig] = None,
        neuron_config: Optional[Dict[str, str]] = None,
    ):
        super().__init__(model, parent_model, config, neuron_config, DIFFUSION_MODEL_UNET_NAME)
        if hasattr(self.model, "device"):
            self.device = self.model.device

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        added_cond_kwargs: Optional[Dict[str, Any]] = None,
    ):
        timestep = timestep.float().expand((sample.shape[0],))
        inputs = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }
        if added_cond_kwargs is not None:
            inputs["text_embeds"] = added_cond_kwargs.pop("text_embeds", None)
            inputs["time_ids"] = added_cond_kwargs.pop("time_ids", None)

        outputs = self.model(*tuple(inputs.values()))
        return outputs


class NeuronModelVaeEncoder(_NeuronDiffusionModelPart):
    def __init__(
        self,
        model: torch.jit._script.ScriptModule,
        parent_model: NeuronBaseModel,
        config: Optional[DiffusersPretrainedConfig] = None,
        neuron_config: Optional[Dict[str, str]] = None,
    ):
        super().__init__(model, parent_model, config, neuron_config, DIFFUSION_MODEL_VAE_ENCODER_NAME)

    def forward(self, sample: torch.Tensor):
        inputs = (sample,)
        outputs = self.model(*inputs)

        return tuple(output for output in outputs.values())


class NeuronModelVaeDecoder(_NeuronDiffusionModelPart):
    def __init__(
        self,
        model: torch.jit._script.ScriptModule,
        parent_model: NeuronBaseModel,
        config: Optional[DiffusersPretrainedConfig] = None,
        neuron_config: Optional[Dict[str, str]] = None,
    ):
        super().__init__(model, parent_model, config, neuron_config, DIFFUSION_MODEL_VAE_DECODER_NAME)

    def forward(
        self,
        latent_sample: torch.Tensor,
        image: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ):
        inputs = (latent_sample,)
        if image is not None:
            inputs += (image,)
        if mask is not None:
            inputs += (mask,)
        outputs = self.model(*inputs)

        return tuple(output for output in outputs.values())


class NeuronStableDiffusionPipeline(NeuronStableDiffusionPipelineBase, NeuronStableDiffusionPipelineMixin):
    __call__ = NeuronStableDiffusionPipelineMixin.__call__


class NeuronStableDiffusionImg2ImgPipeline(
    NeuronStableDiffusionPipelineBase, NeuronStableDiffusionImg2ImgPipelineMixin
):
    __call__ = NeuronStableDiffusionImg2ImgPipelineMixin.__call__


class NeuronStableDiffusionInpaintPipeline(
    NeuronStableDiffusionPipelineBase, NeuronStableDiffusionInpaintPipelineMixin
):
    __call__ = NeuronStableDiffusionInpaintPipelineMixin.__call__


class NeuronStableDiffusionXLPipelineBase(NeuronStableDiffusionPipelineBase):
    # `TasksManager` registered img2ime pipeline for `stable-diffusion-xl`: https://github.com/huggingface/optimum/blob/v1.12.0/optimum/exporters/tasks.py#L174
    auto_model_class = StableDiffusionXLImg2ImgPipeline

    def __init__(
        self,
        text_encoder: torch.jit._script.ScriptModule,
        unet: torch.jit._script.ScriptModule,
        vae_decoder: torch.jit._script.ScriptModule,
        config: Dict[str, Any],
        tokenizer: CLIPTokenizer,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        vae_encoder: Optional[torch.jit._script.ScriptModule] = None,
        text_encoder_2: Optional[torch.jit._script.ScriptModule] = None,
        tokenizer_2: Optional[CLIPTokenizer] = None,
        feature_extractor: Optional[CLIPFeatureExtractor] = None,
        device_ids: Optional[List[int]] = None,
        configs: Optional[Dict[str, "PretrainedConfig"]] = None,
        neuron_configs: Optional[Dict[str, "NeuronConfig"]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        model_and_config_save_paths: Optional[Dict[str, Tuple[str, Path]]] = None,
        add_watermarker: Optional[bool] = None,
    ):
        super().__init__(
            text_encoder=text_encoder,
            unet=unet,
            vae_decoder=vae_decoder,
            config=config,
            tokenizer=tokenizer,
            scheduler=scheduler,
            vae_encoder=vae_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer_2=tokenizer_2,
            feature_extractor=feature_extractor,
            device_ids=device_ids,
            configs=configs,
            neuron_configs=neuron_configs,
            model_save_dir=model_save_dir,
            model_and_config_save_paths=model_and_config_save_paths,
        )

        add_watermarker = add_watermarker if add_watermarker is not None else is_invisible_watermark_available()

        if add_watermarker:
            if not is_invisible_watermark_available():
                raise ImportError(
                    "`add_watermarker` requires invisible-watermark to be installed, which can be installed with `pip install invisible-watermark`."
                )
            from diffusers.pipelines.stable_diffusion_xl.watermark import StableDiffusionXLWatermarker

            self.watermark = StableDiffusionXLWatermarker()
        else:
            self.watermark = None


class NeuronStableDiffusionXLPipeline(NeuronStableDiffusionXLPipelineBase, NeuronStableDiffusionXLPipelineMixin):
    __call__ = NeuronStableDiffusionXLPipelineMixin.__call__


class NeuronStableDiffusionXLImg2ImgPipeline(
    NeuronStableDiffusionXLPipelineBase, NeuronStableDiffusionXLImg2ImgPipelineMixin
):
    __call__ = NeuronStableDiffusionXLImg2ImgPipelineMixin.__call__


class NeuronStableDiffusionXLInpaintPipeline(
    NeuronStableDiffusionXLPipelineBase, NeuronStableDiffusionXLInpaintPipelineMixin
):
    __call__ = NeuronStableDiffusionXLInpaintPipelineMixin.__call__
