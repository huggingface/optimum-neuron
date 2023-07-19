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
from .pipelines.diffusers.pipeline_stable_diffusion import StableDiffusionPipelineMixin
from .utils import (
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
    from diffusers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler, StableDiffusionPipeline
    from diffusers.image_processor import VaeImageProcessor
    from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
    from diffusers.utils import CONFIG_NAME


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
        vae_encoder: torch.jit._script.ScriptModule,
        vae_decoder: torch.jit._script.ScriptModule,
        config: Dict[str, Any],
        tokenizer: CLIPTokenizer,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        feature_extractor: Optional[CLIPFeatureExtractor] = None,
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
            vae_encoder (`torch.jit._script.ScriptModule`):
                The Neuron TorchScript module associated to the VAE encoder.
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
            feature_extractor (`Optional[CLIPFeatureExtractor]`, defaults to `None`):
                A model extracting features from generated images to be used as inputs for the `safety_checker`
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
        self.configs = configs
        self.neuron_configs = neuron_configs

        self.text_encoder = NeuronModelTextEncoder(
            text_encoder,
            self,
            self.configs[DIFFUSION_MODEL_TEXT_ENCODER_NAME],
            self.neuron_configs[DIFFUSION_MODEL_TEXT_ENCODER_NAME],
        )
        self.unet = NeuronModelUnet(
            unet, self, self.configs[DIFFUSION_MODEL_UNET_NAME], self.neuron_configs[DIFFUSION_MODEL_UNET_NAME]
        )
        self.vae_encoder = NeuronModelVaeEncoder(
            vae_encoder,
            self,
            self.configs[DIFFUSION_MODEL_VAE_ENCODER_NAME],
            self.neuron_configs[DIFFUSION_MODEL_VAE_ENCODER_NAME],
        )
        self.vae_decoder = NeuronModelVaeDecoder(
            vae_decoder,
            self,
            self.configs[DIFFUSION_MODEL_VAE_DECODER_NAME],
            self.neuron_configs[DIFFUSION_MODEL_VAE_DECODER_NAME],
        )

        self.tokenizer = tokenizer
        self.scheduler = scheduler
        self.feature_extractor = feature_extractor
        self.safety_checker = None
        sub_models = {
            DIFFUSION_MODEL_TEXT_ENCODER_NAME: self.text_encoder,
            DIFFUSION_MODEL_UNET_NAME: self.unet,
            DIFFUSION_MODEL_VAE_ENCODER_NAME: self.vae_encoder,
            DIFFUSION_MODEL_VAE_DECODER_NAME: self.vae_decoder,
        }
        for name in sub_models.keys():
            self._internal_dict[name] = ("optimum", sub_models[name].__class__.__name__)
        self._internal_dict.pop("vae", None)

        self._attributes_init(model_save_dir)
        self.model_and_config_save_paths = model_and_config_save_paths if model_and_config_save_paths else None

        if hasattr(self.vae_decoder.config, "block_out_channels"):
            self.vae_scale_factor = 2 ** (
                len(self.vae_decoder.config.block_out_channels) - 1
            )  # not working for tiny test models, need to remove `block_out_channels` in `config.json`.
        else:
            self.vae_scale_factor = 8

        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)

    def _save_pretrained(
        self,
        save_directory: Union[str, Path],
        text_encoder_file_name: str = NEURON_FILE_NAME,
        unet_file_name: str = NEURON_FILE_NAME,
        vae_encoder_file_name: str = NEURON_FILE_NAME,
        vae_decoder_file_name: str = NEURON_FILE_NAME,
    ):
        """
        Saves the model to the serialized format optimized for Neuron devices.
        """
        save_directory = Path(save_directory)
        if self.model_and_config_save_paths is None:
            logger.warning(
                "`model_save_paths` is None which means that no path of Neuron model is defined. Nothing will be saved."
            )
            return
        else:
            logger.info(f"Saving the {tuple(self.model_and_config_save_paths.keys())}...")

        dst_paths = {
            "text_encoder": save_directory / DIFFUSION_MODEL_TEXT_ENCODER_NAME / text_encoder_file_name,
            "unet": save_directory / DIFFUSION_MODEL_UNET_NAME / unet_file_name,
            "vae_encoder": save_directory / DIFFUSION_MODEL_VAE_ENCODER_NAME / vae_encoder_file_name,
            "vae_decoder": save_directory / DIFFUSION_MODEL_VAE_DECODER_NAME / vae_decoder_file_name,
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
        text_encoder_file_name: Optional[str] = NEURON_FILE_NAME,
        unet_file_name: Optional[str] = NEURON_FILE_NAME,
        vae_encoder_file_name: Optional[str] = NEURON_FILE_NAME,
        vae_decoder_file_name: Optional[str] = NEURON_FILE_NAME,
        local_files_only: bool = False,
        model_save_dir: Optional[Union[str, Path, TemporaryDirectory]] = None,
        device_ids: Optional[List[int]] = None,
        **kwargs,  # To share kwargs only available for `_from_transformers`
    ):
        model_id = str(model_id)
        sub_models_to_load, _, _ = cls.extract_init_dict(config)
        sub_models_names = set(sub_models_to_load.keys()).intersection({"feature_extractor", "tokenizer", "scheduler"})
        sub_models = {}

        if not os.path.isdir(model_id):
            patterns = set(config.keys())
            patterns.update({DIFFUSION_MODEL_VAE_ENCODER_NAME, DIFFUSION_MODEL_VAE_DECODER_NAME})
            allow_patterns = {os.path.join(k, "*") for k in patterns if not k.startswith("_")}
            allow_patterns.update(
                {
                    text_encoder_file_name,
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
            "text_encoder": (
                new_model_save_dir / DIFFUSION_MODEL_TEXT_ENCODER_NAME / text_encoder_file_name,
                new_model_save_dir / DIFFUSION_MODEL_TEXT_ENCODER_NAME / cls.sub_component_config_name,
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
        configs = {
            name: DiffusersPretrainedConfig.from_json_file(model_config[1])
            for name, model_config in model_and_config_save_paths.items()
        }
        neuron_configs = {name: cls._neuron_config_init(model_config) for name, model_config in configs.items()}

        text_encoder = cls.load_model(model_and_config_save_paths["text_encoder"][0])
        if device_ids:
            # Load the compiled UNet onto multiple neuron cores
            unet = torch_neuronx.DataParallel(
                torch.jit.load(model_and_config_save_paths["unet"][0]),
                device_ids,
                set_dynamic_batching=neuron_configs[DIFFUSION_MODEL_UNET_NAME].dynamic_batch_size,
            )
        else:
            unet = cls.load_model(model_and_config_save_paths["unet"][0])
        vae_encoder = cls.load_model(model_and_config_save_paths["vae_encoder"][0])
        vae_decoder = cls.load_model(model_and_config_save_paths["vae_decoder"][0])

        if model_save_dir is None:
            model_save_dir = new_model_save_dir

        return cls(
            text_encoder=text_encoder,
            unet=unet,
            vae_encoder=vae_encoder,
            vae_decoder=vae_decoder,
            config=config,
            tokenizer=sub_models["tokenizer"],
            scheduler=sub_models["scheduler"],
            feature_extractor=sub_models.pop("feature_extractor", None),
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
        auto_cast: Optional[str] = None,
        auto_cast_type: Optional[str] = None,
        disable_fast_relayout: Optional[bool] = False,
        disable_fallback: bool = False,
        dynamic_batch_size: bool = False,
        device_ids: Optional[List[int]] = None,
        **kwargs_shapes,
    ) -> "NeuronStableDiffusionPipelineBase":
        if task is None:
            task = TasksManager.infer_task_from_model(cls.auto_model_class)

        # mandatory shapes
        input_shapes = normalize_stable_diffusion_input_shapes(task, kwargs_shapes)

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

    def forward(self, sample: torch.Tensor, timestep: torch.Tensor, encoder_hidden_states: torch.Tensor):
        timestep = timestep.float().expand((sample.shape[0],))
        inputs = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
        }
        outputs = self.model(*tuple(inputs.values()))

        return tuple(output for output in outputs.values())


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

    def forward(self, latent_sample: torch.Tensor):
        inputs = (latent_sample,)
        outputs = self.model(*inputs)

        return tuple(output for output in outputs.values())


class NeuronStableDiffusionPipeline(NeuronStableDiffusionPipelineBase, StableDiffusionPipelineMixin):
    def __call__(self, *args, **kwargs):
        return StableDiffusionPipelineMixin.__call__(self, *args, **kwargs)
