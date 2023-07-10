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

import os
import logging
import importlib
import shutil
from abc import abstractmethod
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Dict, List, Optional, Union, Tuple, TYPE_CHECKING

import torch
import torch_neuronx
from diffusers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler, StableDiffusionPipeline
from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from diffusers.utils import CONFIG_NAME
from huggingface_hub import snapshot_download
from transformers import CLIPFeatureExtractor, CLIPTokenizer, PretrainedConfig

from ..exporters.neuron import NeuronConfig, main_export, normalize_input_shapes
from ..exporters.neuron.model_configs import *  # noqa: F403
from ..pipelines.diffusers.pipeline_stable_diffusion import StableDiffusionPipelineMixin
# from ..pipelines.diffusers.pipeline_stable_diffusion_img2img import StableDiffusionImg2ImgPipelineMixin
# from ..pipelines.diffusers.pipeline_stable_diffusion_inpaint import StableDiffusionInpaintPipelineMixin
from .modeling_base import NeuronBaseModel
from .utils import (
    NEURON_FILE_NAME,
    DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER,
    DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER,
    DIFFUSION_MODEL_UNET_SUBFOLDER,
    DIFFUSION_MODEL_VAE_POST_QUANT_CONV_SUBFOLDER,
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
        vae_decoder: torch.jit._script.ScriptModule,
        unet: torch.jit._script.ScriptModule,
        vae_post_quant_conv: torch.jit._script.ScriptModule,
        config: Dict[str, Any],
        tokenizer: CLIPTokenizer,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        feature_extractor: Optional[CLIPFeatureExtractor] = None,
        neuron_configs: Optional[Dict[str, "NeuronConfig"]] = None,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        model_and_config_save_paths: Optional[Dict[str, Tuple[str, Path]]] = None,
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
            model_and_config_save_paths (`Optional[Dict[str, Tuple[str, Path]]]`, defaults to `None`):
                The paths where exported Neuron models were saved.
        """

        self._internal_dict = config

        # Re-build pretrained configs and neuron configs 
        if neuron_configs is None:
            self.configs = {name: PretrainedConfig.from_json_file(model_config[1]) for name, model_config in model_and_config_save_paths.items()}
            self.neuron_configs = {name: self._neuron_config_init(model_config) for name, model_config in self.configs.items()}
        else:
            self.configs = None
            self.neuron_configs = neuron_configs
        import pdb
        pdb.set_trace()

        self.text_encoder = NeuronModelTextEncoder(text_encoder, self, self.configs[DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER], self.neuron_configs[DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER])
        self.vae_decoder = NeuronModelVaeDecoder(vae_decoder, self, self.configs[DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER], self.neuron_configs[DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER])
        self.unet = NeuronModelUnet(unet, self, self.configs[DIFFUSION_MODEL_UNET_SUBFOLDER], self.neuron_configs[DIFFUSION_MODEL_UNET_SUBFOLDER])
        self.vae_post_quant_conv = NeuronModelVaePostQuantConv(vae_post_quant_conv, self, self.configs[DIFFUSION_MODEL_VAE_POST_QUANT_CONV_SUBFOLDER], self.neuron_configs[DIFFUSION_MODEL_VAE_POST_QUANT_CONV_SUBFOLDER])

        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.feature_extractor = feature_extractor
        self.safety_checker = None
        sub_models = {
            DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER: self.text_encoder,
            DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER: self.vae_decoder,
            DIFFUSION_MODEL_UNET_SUBFOLDER: self.unet,
            DIFFUSION_MODEL_VAE_POST_QUANT_CONV_SUBFOLDER: self.vae_post_quant_conv,
        }
        for name in sub_models.keys():
            self._internal_dict[name] = ("optimum", sub_models[name].__class__.__name__)
        self._internal_dict.pop("vae", None)

        self._attributes_init(model_save_dir)
        self.model_and_config_save_paths = model_and_config_save_paths if model_and_config_save_paths else None
        

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
        if self.model_and_config_save_paths is None:
            logger.warning("`model_save_paths` is None which means that no path of Neuron model is defined. Nothing will be saved.")
            return
        else:
            logger.info(f"Saving the {self.model_and_config_save_paths.keys()}...")

        dst_paths = {
            "text_encoder": save_directory / DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER / text_encoder_file_name,
            "vae_decoder": save_directory / DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER / vae_decoder_file_name,
            "unet": save_directory / DIFFUSION_MODEL_UNET_SUBFOLDER / unet_file_name,
            "vae_post_quant_conv": save_directory / DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER / vae_post_quant_conv_file_name,
        }
        model_src_to_dst_path = {
            self.model_and_config_save_paths[model_name][0]: dst_paths[model_name] for model_name in set(self.model_and_config_save_paths.keys()).intersection(dst_paths.keys())
        }
        # save
        config_src_to_dst_path = {
            self.model_and_config_save_paths[model_name][1]: dst_paths[model_name].parent for model_name in set(self.model_and_config_save_paths.keys()).intersection(dst_paths.keys())
        }

        src_paths = list(model_src_to_dst_path.keys()) + list(config_src_to_dst_path.keys())
        dst_paths = list(model_src_to_dst_path.values()) + list(config_src_to_dst_path.values())

        import pdb
        pdb.set_trace()
        for src_path, dst_path in zip(src_paths, dst_paths):
            dst_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src_path, dst_path)

        self.tokenizer.save_pretrained(save_directory.joinpath("tokenizer"))
        self.scheduler.save_pretrained(save_directory.joinpath("scheduler"))
        if self.feature_extractor is not None:
            self.feature_extractor.save_pretrained(save_directory.joinpath("feature_extractor"))
        import pdb
        pdb.set_trace()

    @classmethod
    def _from_pretrained(
        cls,
        model_id: Union[str, Path],
        config: Dict[str, Any],
        use_auth_token: Optional[Union[bool, str]] = None,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        text_encoder_file_name: Optional[str] = NEURON_FILE_NAME,
        vae_decoder_file_name: Optional[str] = NEURON_FILE_NAME,
        unet_file_name: Optional[str] = NEURON_FILE_NAME,
        vae_post_quant_conv_file_name: Optional[str] = NEURON_FILE_NAME,
        local_files_only: bool = False,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        **kwargs,
    ):
        model_id = str(model_id)
        sub_models_to_load, _, _ = cls.extract_init_dict(config)
        sub_models_names = set(sub_models_to_load.keys()).intersection({"feature_extractor", "tokenizer", "scheduler"})
        sub_models = {}

        if not os.path.isdir(model_id):
            patterns = set(config.keys())
            allow_patterns = {os.path.join(k, "*") for k in patterns if not k.startswith("_")}
            allow_patterns.update(
                {
                    text_encoder_file_name,
                    vae_decoder_file_name,
                    unet_file_name,
                    vae_post_quant_conv_file_name,
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
        for name in sub_models_names:
            library_name, library_classes = sub_models_to_load[name]
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
            "text_encoder": (new_model_save_dir / DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER / text_encoder_file_name, new_model_save_dir / DIFFUSION_MODEL_TEXT_ENCODER_SUBFOLDER / cls.sub_component_config_name),
            "vae_decoder": (new_model_save_dir / DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER / vae_decoder_file_name, new_model_save_dir / DIFFUSION_MODEL_VAE_DECODER_SUBFOLDER / cls.sub_component_config_name),
            "unet": (new_model_save_dir / DIFFUSION_MODEL_UNET_SUBFOLDER / unet_file_name, new_model_save_dir / DIFFUSION_MODEL_UNET_SUBFOLDER / cls.sub_component_config_name),
            "vae_post_quant_conv": (new_model_save_dir / DIFFUSION_MODEL_VAE_POST_QUANT_CONV_SUBFOLDER / vae_post_quant_conv_file_name, new_model_save_dir / DIFFUSION_MODEL_VAE_POST_QUANT_CONV_SUBFOLDER / cls.sub_component_config_name),
        }
        text_encoder = cls.load_model(model_and_config_save_paths["text_encoder"][0])
        vae_decoder = cls.load_model(model_and_config_save_paths["vae_decoder"][0])
        unet = cls.load_model(model_and_config_save_paths["unet"][0])
        vae_post_quant_conv = cls.load_model(model_and_config_save_paths["vae_post_quant_conv"][0])

        if model_save_dir is None:
            model_save_dir = new_model_save_dir

        return cls(
            text_encoder=text_encoder,
            vae_decoder=vae_decoder,
            unet=unet,
            vae_post_quant_conv=vae_post_quant_conv,
            config=config,
            tokenizer=sub_models["tokenizer"],
            scheduler=sub_models["scheduler"],
            feature_extractor=sub_models.pop("feature_extractor", None),
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
        auto_cast: Optional[str] = None,
        auto_cast_type: Optional[str] = None,
        disable_fast_relayout: Optional[bool] = False,
        disable_fallback: bool = False,
        dynamic_batch_size: bool = False,
        **kwargs_shapes,
    ) -> "NeuronStableDiffusionPipelineBase":
        if task is None:
            task = cls._auto_model_to_task(cls.auto_model_class)
        
        # mandatory shapes
        input_shapes = normalize_input_shapes(task, kwargs_shapes)

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
        )

    def __call__(self, *args, **kwargs):
        return StableDiffusionPipelineMixin.__call__(self, *args, **kwargs)

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
        config: Optional["PretrainedConfig"] = None,
        neuron_config: Optional["NeuronConfig"] = None,
        model_type: str = "unet",
        device: Optional[int] = None,
    ):
        self.model = model
        self.parent_model = parent_model
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
        config: Optional["PretrainedConfig"] = None,
        neuron_config: Optional[Dict[str, str]] = None,
    ):
        super().__init__(model, parent_model, neuron_config, "text_encoder")

    def forward(self, input_ids: torch.Tensor):
        inputs = (input_ids,)
        outputs = self.model(*inputs)
        return outputs


class NeuronModelVaeDecoder(_NeuronDiffusionModelPart):
    def __init__(
        self,
        model: torch.jit._script.ScriptModule,
        parent_model: NeuronBaseModel,
        config: Optional["PretrainedConfig"] = None,
        neuron_config: Optional[Dict[str, str]] = None,
    ):
        super().__init__(model, parent_model, neuron_config, "vae_decoder")

    def forward(self, latent_sample: torch.Tensor):

        inputs = (latent_sample,)
        outputs = self.model(*inputs)
        return outputs


class NeuronModelUnet(_NeuronDiffusionModelPart):
    def __init__(
        self,
        model: torch.jit._script.ScriptModule,
        parent_model: NeuronBaseModel,
        config: Optional["PretrainedConfig"] = None,
        neuron_config: Optional[Dict[str, str]] = None,
        device_ids: Optional[List[int]] = None
    ):
        super().__init__(model, parent_model, neuron_config, "unet")
        if device_ids:
            self.wrapped_model = torch_neuronx.DataParallel(self.model, device_ids, set_dynamic_batching=False)

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor):
        timestep = timestep.float().expand((sample.shape[0],))
        inputs = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }
        outputs = self.model(*inputs)
        return outputs


class NeuronModelVaePostQuantConv(_NeuronDiffusionModelPart):
    def __init__(
        self,
        model: torch.jit._script.ScriptModule,
        parent_model: NeuronBaseModel,
        config: Optional["PretrainedConfig"] = None,
        neuron_config: Optional[Dict[str, str]] = None,
    ):
        super().__init__(model, parent_model, neuron_config, "vae_post_quant_conv")

    def forward(self, latent_sample: torch.Tensor):
        inputs = (latent_sample,)
        outputs = self.model(*inputs)
        return outputs
    
class NeuronStableDiffusionPipeline(NeuronStableDiffusionPipelineBase, StableDiffusionPipelineMixin):
    def __call__(self, *args, **kwargs):
        return StableDiffusionPipelineMixin.__call__(self, *args, **kwargs)


# class NeuronStableDiffusionImg2ImgPipeline(NeuronStableDiffusionPipelineBase, StableDiffusionImg2ImgPipelineMixin):
#     def __call__(self, *args, **kwargs):
#         return StableDiffusionImg2ImgPipelineMixin.__call__(self, *args, **kwargs)


# class NeuronStableDiffusionInpaintPipeline(NeuronStableDiffusionPipelineBase, StableDiffusionInpaintPipelineMixin):
#     def __call__(self, *args, **kwargs):
#         return StableDiffusionInpaintPipelineMixin.__call__(self, *args, **kwargs)
