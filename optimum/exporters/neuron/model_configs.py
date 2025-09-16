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
"""Model specific Neuron configurations."""

import copy
import inspect
import os
from functools import partial
from pathlib import Path
from typing import Any

import neuronx_distributed
import torch
from neuronx_distributed.trace.model_builder import BaseModelInstance
from optimum.exporters.tasks import TasksManager
from optimum.neuron.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    ASTDummyAudioInputGenerator,
    DummyBeamValuesGenerator,
    DummyControNetInputGenerator,
    DummyTransformerRotaryEmbGenerator,
    DummyQwenImageTransformerInputGenerator,
    DummyIPAdapterInputGenerator,
    DummyMaskedPosGenerator,
    DummyTimestepInputGenerator,
    WhisperDummyTextInputGenerator,
    get_checkpoint_shard_files,
    saved_model_in_temporary_directory,
)
from optimum.utils import (
    DummyFluxTransformerTextInputGenerator,
    DummyFluxTransformerVisionInputGenerator,
    DummyInputGenerator,
    DummySeq2SeqDecoderTextInputGenerator,
    DummyTextInputGenerator,
    DummyTransformerTimestepInputGenerator,
    DummyVisionInputGenerator,
    NormalizedConfig,
    NormalizedConfigManager,
    NormalizedSeq2SeqConfig,
    NormalizedTextAndVisionConfig,
    NormalizedTextConfig,
    NormalizedVisionConfig,
    is_diffusers_available,
    logging,
)
from safetensors.torch import load_file

from optimum.neuron.utils import (
    SAFE_WEIGHTS_INDEX_NAME,
    ASTDummyAudioInputGenerator,
    DummyBeamValuesGenerator,
    DummyControNetInputGenerator,
    DummyFluxKontextTransformerRotaryEmbGenerator,
    DummyTransformerRotaryEmbGenerator,
    DummyIPAdapterInputGenerator,
    DummyMaskedPosGenerator,
    DummyTimestepInputGenerator,
    WhisperDummyTextInputGenerator,
    get_checkpoint_shard_files,
    saved_model_in_temporary_directory,
)

from .config import (
    AudioNeuronConfig,
    TextAndVisionNeuronConfig,
    TextEncoderNeuronConfig,
    TextSeq2SeqNeuronConfig,
    VisionNeuronConfig,
)
from .model_wrappers import (
    CLIPVisionWithProjectionNeuronWrapper,
    ControlNetNeuronWrapper,
    FluxTransformerNeuronWrapper,
    QwenImageTransformerNeuronWrapper,
    NoCacheModelWrapper,
    PixartTransformerNeuronWrapper,
    SentenceTransformersCLIPNeuronWrapper,
    SentenceTransformersTransformerNeuronWrapper,
    T5DecoderWrapper,
    T5EncoderForSeq2SeqLMWrapper,
    T5EncoderWrapper,
    UnetNeuronWrapper,
    WhisperDecoderWrapper,
    WhisperEncoderWrapper,
)


if is_diffusers_available():
    from diffusers.models.autoencoders.vae import Decoder as VaeDecoder
    from diffusers.models.model_loading_utils import _get_model_file

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

COMMON_TEXT_TASKS = [
    "feature-extraction",
    "fill-mask",
    "multiple-choice",
    "question-answering",
    "text-classification",
    "token-classification",
]
register_in_tasks_manager = TasksManager.create_register("neuron")


@register_in_tasks_manager("bert", *COMMON_TEXT_TASKS)
class BertNeuronConfig(TextEncoderNeuronConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfigManager.get_normalized_config_class("bert")
    ATOL_FOR_VALIDATION = 1e-3

    @property
    def inputs(self) -> list[str]:
        return ["input_ids", "attention_mask", "token_type_ids"]


@register_in_tasks_manager("albert", *COMMON_TEXT_TASKS)
class AlbertNeuronConfig(BertNeuronConfig):
    pass


@register_in_tasks_manager("convbert", *COMMON_TEXT_TASKS)
class ConvBertNeuronConfig(BertNeuronConfig):
    ATOL_FOR_VALIDATION = 1e-1  # TODO: why accuracy more off than other arch

    @property
    def outputs(self) -> list[str]:
        if self.task == "feature-extraction":
            return ["last_hidden_state"]
        return self._TASK_TO_COMMON_OUTPUTS[self.task]


@register_in_tasks_manager("electra", *COMMON_TEXT_TASKS)
class ElectraNeuronConfig(BertNeuronConfig):
    @property
    def outputs(self) -> list[str]:
        if self.task == "feature-extraction":
            return ["last_hidden_state"]
        return self._TASK_TO_COMMON_OUTPUTS[self.task]


@register_in_tasks_manager("esm", *["feature-extraction", "fill-mask", "text-classification", "token-classification"])
class EsmNeuronConfig(TextEncoderNeuronConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfigManager.get_normalized_config_class("bert")
    ATOL_FOR_VALIDATION = 1e-3

    @property
    def inputs(self) -> list[str]:
        return ["input_ids", "attention_mask"]


@register_in_tasks_manager("flaubert", *COMMON_TEXT_TASKS)
class FlaubertNeuronConfig(ElectraNeuronConfig):
    pass


@register_in_tasks_manager("mobilebert", *COMMON_TEXT_TASKS)
class MobileBertNeuronConfig(BertNeuronConfig):
    pass


@register_in_tasks_manager(
    "modernbert", *["feature-extraction", "fill-mask", "text-classification", "token-classification"]
)
class ModernBertNeuronConfig(BertNeuronConfig):
    @property
    def inputs(self) -> list[str]:
        return ["input_ids", "attention_mask"]

    @property
    def outputs(self) -> list[str]:
        if self.task == "feature-extraction":
            return ["last_hidden_state"]
        return self._TASK_TO_COMMON_OUTPUTS[self.task]


@register_in_tasks_manager("phi", *["feature-extraction", "text-classification", "token-classification"])
class PhiNeuronConfig(ElectraNeuronConfig):
    CUSTOM_MODEL_WRAPPER = NoCacheModelWrapper

    @property
    def inputs(self) -> list[str]:
        return ["input_ids", "attention_mask"]


@register_in_tasks_manager("roformer", *COMMON_TEXT_TASKS)
class RoFormerNeuronConfig(ElectraNeuronConfig):
    pass


@register_in_tasks_manager("xlm", *COMMON_TEXT_TASKS)
class XLMNeuronConfig(ElectraNeuronConfig):
    pass


@register_in_tasks_manager("distilbert", *COMMON_TEXT_TASKS)
class DistilBertNeuronConfig(BertNeuronConfig):
    ATOL_FOR_VALIDATION = 1e-3

    @property
    def inputs(self) -> list[str]:
        return ["input_ids", "attention_mask"]

    @property
    def outputs(self) -> list[str]:
        if self.task == "feature-extraction":
            return ["last_hidden_state"]
        return self._TASK_TO_COMMON_OUTPUTS[self.task]


@register_in_tasks_manager("camembert", *COMMON_TEXT_TASKS)
class CamembertNeuronConfig(BertNeuronConfig):
    ATOL_FOR_VALIDATION = 1e-3

    @property
    def inputs(self) -> list[str]:
        return ["input_ids", "attention_mask"]


@register_in_tasks_manager("mpnet", *COMMON_TEXT_TASKS)
class MPNetNeuronConfig(CamembertNeuronConfig):
    pass


@register_in_tasks_manager("roberta", *COMMON_TEXT_TASKS)
class RobertaNeuronConfig(CamembertNeuronConfig):
    pass


@register_in_tasks_manager("xlm-roberta", *COMMON_TEXT_TASKS)
class XLMRobertaNeuronConfig(CamembertNeuronConfig):
    pass


# https://github.com/aws-neuron/aws-neuron-sdk/issues/642
# Failed only for INF1: 'XSoftmax'
@register_in_tasks_manager("deberta", *([task for task in COMMON_TEXT_TASKS if task != "multiple-choice"]))
class DebertaNeuronConfig(ElectraNeuronConfig):
    @property
    def inputs(self) -> list[str]:
        common_inputs = super().inputs
        if self._config.type_vocab_size == 0:
            # We remove token type ids.
            common_inputs.pop(-1)
        return common_inputs


# https://github.com/aws-neuron/aws-neuron-sdk/issues/642
# Failed only for INF1: 'XSoftmax'
@register_in_tasks_manager("deberta-v2", *([task for task in COMMON_TEXT_TASKS if task != "multiple-choice"]))
class DebertaV2NeuronConfig(ElectraNeuronConfig):
    pass


@register_in_tasks_manager(
    "transformer", *["feature-extraction", "sentence-similarity"], library_name="sentence_transformers"
)
class SentenceTransformersTransformerNeuronConfig(TextEncoderNeuronConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig
    CUSTOM_MODEL_WRAPPER = SentenceTransformersTransformerNeuronWrapper
    ATOL_FOR_VALIDATION = 1e-3

    @property
    def inputs(self) -> list[str]:
        return ["input_ids", "attention_mask"]

    @property
    def outputs(self) -> list[str]:
        return ["token_embeddings", "sentence_embedding"]


class CLIPNormalizedConfig(NormalizedTextAndVisionConfig):
    TEXT_CONFIG = "text_config"
    VISION_CONFIG = "vision_config"


@register_in_tasks_manager("clip-vision-with-projection", *["feature-extraction"], library_name="diffusers")
class CLIPVisionWithProjectionNeuronConfig(VisionNeuronConfig):
    MODEL_TYPE = "clip-vision-with-projection"
    NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig
    CUSTOM_MODEL_WRAPPER = CLIPVisionWithProjectionNeuronWrapper

    @property
    def inputs(self) -> list[str]:
        return ["pixel_values"]

    @property
    def outputs(self) -> list[str]:
        common_outputs = ["image_embeds", "last_hidden_state"]
        if self.output_hidden_states:
            common_outputs.append("hidden_states")
        return common_outputs


@register_in_tasks_manager("clip", *["feature-extraction", "zero-shot-image-classification", "image-classification"])
class CLIPNeuronConfig(TextAndVisionNeuronConfig):
    NORMALIZED_CONFIG_CLASS = CLIPNormalizedConfig
    INPUT_ARGS = ("text_batch_size", "image_batch_size", "sequence_length", "num_channels", "width", "height")

    @property
    def inputs(self) -> list[str]:
        if self.task == "image-classification":
            return ["pixel_values"]
        else:
            return ["input_ids", "pixel_values", "attention_mask"]

    @property
    def outputs(self) -> list[str]:
        if self.task == "image-classification":
            return ["logits"]
        else:
            return [
                "logits_per_image",
                "logits_per_text",
                "text_embeds",
                "image_embeds",
                "text_model_output",
                "vision_model_output",
            ]

    def _create_dummy_input_generator_classes(self, **kwargs) -> list["DummyInputGenerator"]:
        for name, axis_dim in self._axes.items():
            self._axes[name] = kwargs.pop(name, axis_dim)

        self._validate_mandatory_axes()

        other_axes = copy.deepcopy(self._axes)
        text_batch_size = other_axes.pop("text_batch_size")
        images_batch_size = other_axes.pop("image_batch_size")

        return [
            DummyTextInputGenerator(self.task, self._normalized_config, batch_size=text_batch_size, **other_axes),
            DummyVisionInputGenerator(self.task, self._normalized_config, batch_size=images_batch_size, **other_axes),
        ]


@register_in_tasks_manager("clip-text-with-projection", *["feature-extraction"], library_name="diffusers")
class CLIPTextWithProjectionNeuronConfig(TextEncoderNeuronConfig):
    MODEL_TYPE = "clip-text-with-projection"
    ATOL_FOR_VALIDATION = 1e-3

    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        vocab_size="vocab_size",
        sequence_length="max_position_embeddings",
        num_layers="num_hidden_layers",
        allow_new=True,
    )

    @property
    def inputs(self) -> list[str]:
        return ["input_ids"]

    @property
    def outputs(self) -> list[str]:
        common_outputs = ["text_embeds", "last_hidden_state"]

        if self._normalized_config.output_hidden_states:
            common_outputs.append("hidden_states")

        return common_outputs

    @property
    def values_override(self) -> dict[str, Any] | None:
        return {"return_dict": False}


@register_in_tasks_manager("clip-text-model", *["feature-extraction"], library_name="diffusers")
class CLIPTextNeuronConfig(CLIPTextWithProjectionNeuronConfig):
    MODEL_TYPE = "clip-text-model"

    @property
    def outputs(self) -> list[str]:
        common_outputs = ["last_hidden_state", "pooler_output"]

        if self._normalized_config.output_hidden_states:
            common_outputs.append("hidden_states")

        return common_outputs


# TODO: We should decouple clip text and vision, this would need fix on Optimum main. For the current workaround
# users can pass dummy text inputs when encoding image, vice versa.
@register_in_tasks_manager(
    "clip", *["feature-extraction", "sentence-similarity"], library_name="sentence_transformers"
)
class SentenceTransformersCLIPNeuronConfig(CLIPNeuronConfig):
    CUSTOM_MODEL_WRAPPER = SentenceTransformersCLIPNeuronWrapper
    ATOL_FOR_VALIDATION = 1e-3

    @property
    def outputs(self) -> list[str]:
        return ["text_embeds", "image_embeds"]


@register_in_tasks_manager("vit", *["feature-extraction", "image-classification"])
class ViTNeuronConfig(VisionNeuronConfig):
    ATOL_FOR_VALIDATION = 1e-3
    NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVisionInputGenerator, DummyMaskedPosGenerator)
    INPUT_ARGS = ("batch_size",)  # `num_channels` and `image_size` are inferred from the config

    @property
    def inputs(self) -> list[str]:
        common_inputs = ["pixel_values"]
        if self.task == "masked-im":
            common_inputs.append("bool_masked_pos")
        return common_inputs


@register_in_tasks_manager("beit", *["feature-extraction", "image-classification"])
class BeitNeuronConfig(ViTNeuronConfig):
    pass


@register_in_tasks_manager("convnext", *["feature-extraction", "image-classification"])
class ConvNextNeuronConfig(ViTNeuronConfig):
    pass


@register_in_tasks_manager("convnextv2", *["feature-extraction", "image-classification"])
class ConvNextV2NeuronConfig(ViTNeuronConfig):
    pass


@register_in_tasks_manager("cvt", *["feature-extraction", "image-classification"])
class CvTNeuronConfig(ViTNeuronConfig):
    MODEL_TYPE = "cvt"

    @property
    def outputs(self) -> list[str]:
        common_outputs = super().outputs
        if self.task == "feature-extraction":
            return ["last_hidden_state", "cls_token_value"]
        else:
            return common_outputs


@register_in_tasks_manager("deit", *["feature-extraction", "image-classification"])
class DeiTNeuronConfig(ViTNeuronConfig):
    pass


@register_in_tasks_manager("donut-swin", *["feature-extraction"])
class DonutSwinNeuronConfig(ViTNeuronConfig):
    pass


@register_in_tasks_manager("dpt", *["feature-extraction"])
class DptNeuronConfig(ViTNeuronConfig):
    pass


@register_in_tasks_manager("levit", *["feature-extraction", "image-classification"])
class LevitNeuronConfig(ViTNeuronConfig):
    MODEL_TYPE = "levit"
    pass


@register_in_tasks_manager(
    "mobilenet-v2", *["feature-extraction", "image-classification", "semantic-segmentation", "image-segmentation"]
)
class MobileNetV2NeuronConfig(ViTNeuronConfig):
    MODEL_TYPE = "mobilenet-v2"
    pass


@register_in_tasks_manager(
    "mobilevit", *["feature-extraction", "image-classification", "semantic-segmentation", "image-segmentation"]
)
class MobileViTNeuronConfig(ViTNeuronConfig):
    MODEL_TYPE = "mobilevit"
    pass


@register_in_tasks_manager("swin", *["feature-extraction", "image-classification"])
class SwinNeuronConfig(ViTNeuronConfig):
    pass


@register_in_tasks_manager("yolos", *["feature-extraction", "object-detection"])
class YolosTNeuronConfig(ViTNeuronConfig):
    @property
    def outputs(self) -> list[str]:
        common_outputs = super().outputs
        if self.task == "object-detection":
            common_outputs.append("last_hidden_state")
        return common_outputs


@register_in_tasks_manager(
    "wav2vec2",
    *[
        "feature-extraction",
        "automatic-speech-recognition",
        "audio-classification",
        "audio-frame-classification",
        "audio-xvector",
    ],
)
class Wav2Vec2NeuronConfig(AudioNeuronConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfig
    MODEL_TYPE = "wav2vec2"

    @property
    def inputs(self) -> list[str]:
        return ["input_values"]

    @property
    def outputs(self) -> list[str]:
        common_outputs = super().outputs
        if self.task == "feature-extraction":
            common_outputs = ["last_hidden_state", "extract_features"]
        if self.task == "audio-xvector":
            common_outputs.append("embeddings")
        return common_outputs


@register_in_tasks_manager(
    "audio-spectrogram-transformer",
    *[
        "feature-extraction",
        "audio-classification",
    ],
)
class ASTNeuronConfig(AudioNeuronConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        num_mel_bins="num_mel_bins", max_length="max_length", allow_new=True
    )
    DUMMY_INPUT_GENERATOR_CLASSES = (ASTDummyAudioInputGenerator,)

    @property
    def inputs(self) -> list[str]:
        return ["input_values"]


@register_in_tasks_manager(
    "hubert",
    *[
        "feature-extraction",
        "automatic-speech-recognition",
        "audio-classification",
    ],
)
class HubertNeuronConfig(Wav2Vec2NeuronConfig):
    MODEL_TYPE = "hubert"

    @property
    def outputs(self) -> list[str]:
        common_outputs = super().outputs
        if self.task == "feature-extraction":
            common_outputs = ["last_hidden_state"]
        return common_outputs


# TODO: compilation failed due to a bug in xla: https://github.com/pytorch/xla/issues/6398.
# @register_in_tasks_manager(
#     "sew",
#     *[
#         "feature-extraction",
#         "automatic-speech-recognition",
#         "audio-classification",
#     ],
# )
# class SEWNeuronConfig(Wav2Vec2NeuronConfig):
#     pass


# TODO: compilation failed due to a bug in xla: https://github.com/pytorch/xla/issues/6398.
# @register_in_tasks_manager(
#     "sew-d",
#     *[
#         "feature-extraction",
#         "automatic-speech-recognition",
#         "audio-classification",
#     ],
# )
# class SEWDNeuronConfig(Wav2Vec2NeuronConfig):
#     pass


@register_in_tasks_manager(
    "unispeech",
    *[
        "feature-extraction",
        "automatic-speech-recognition",
        "audio-classification",
    ],
)
class UniSpeechNeuronConfig(Wav2Vec2NeuronConfig):
    MODEL_TYPE = "unispeech"
    pass


@register_in_tasks_manager(
    "unispeech-sat",
    *[
        "feature-extraction",
        "automatic-speech-recognition",
        "audio-classification",
        "audio-frame-classification",
        "audio-xvector",
    ],
)
class UniSpeechSATNeuronConfig(Wav2Vec2NeuronConfig):
    MODEL_TYPE = "unispeech-sat"
    pass


# TODO: compilation failed due to a bug in xla: https://github.com/pytorch/xla/issues/6398.
# @register_in_tasks_manager(
#     "wav2vec2-bert",
#     *[
#         "feature-extraction",
#         "automatic-speech-recognition",
#         "audio-classification",
#         "audio-frame-classification",
#         "audio-xvector",
#     ],
# )
# class Wav2Vec2BertNeuronConfig(Wav2Vec2NeuronConfig):
#     pass


# TODO: compilation failed due to a bug in xla: https://github.com/pytorch/xla/issues/6398.
# @register_in_tasks_manager(
#     "wav2vec2-conformer",
#     *[
#         "feature-extraction",
#         "automatic-speech-recognition",
#         "audio-classification",
#         "audio-frame-classification",
#         "audio-xvector",
#     ],
# )
# class Wav2Vec2ConformerNeuronConfig(Wav2Vec2NeuronConfig):
#     pass


@register_in_tasks_manager(
    "wavlm",
    *[
        "feature-extraction",
        "automatic-speech-recognition",
        "audio-classification",
        "audio-frame-classification",
        "audio-xvector",
    ],
)
class WavLMNeuronConfig(Wav2Vec2NeuronConfig):
    MODEL_TYPE = "wavlm"
    pass


@register_in_tasks_manager("unet", *["semantic-segmentation"], library_name="diffusers")
class UNetNeuronConfig(VisionNeuronConfig):
    ATOL_FOR_VALIDATION = 1e-3
    INPUT_ARGS = ("batch_size", "sequence_length", "num_channels", "width", "height", "vae_scale_factor")
    MODEL_TYPE = "unet"
    CUSTOM_MODEL_WRAPPER = UnetNeuronWrapper
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        height="height",
        width="width",
        num_channels="in_channels",
        hidden_size="cross_attention_dim",
        vocab_size="norm_num_groups",
        allow_new=True,
    )

    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyVisionInputGenerator,
        DummySeq2SeqDecoderTextInputGenerator,
        DummyTimestepInputGenerator,
        DummyControNetInputGenerator,
        DummyIPAdapterInputGenerator,
    )

    @property
    def inputs(self) -> list[str]:
        common_inputs = ["sample", "timestep", "encoder_hidden_states"]

        # TODO : add text_image, image and image_embeds
        if getattr(self._normalized_config, "addition_embed_type", None) == "text_time":
            common_inputs.append("text_embeds")
            common_inputs.append("time_ids")

        if getattr(self._normalized_config, "time_cond_proj_dim", None) is not None:
            common_inputs.append("timestep_cond")

        if self.with_controlnet:
            # outputs of controlnet
            common_inputs += ["down_block_additional_residuals", "mid_block_additional_residual"]

        if self.with_ip_adapter:
            # add output of image encoder
            if self.image_encoder_output_hidden_states:
                common_inputs += ["image_enc_hidden_states"]
            else:
                common_inputs += ["image_embeds"]

        return common_inputs

    @property
    def outputs(self) -> list[str]:
        return ["sample"]

    def generate_dummy_inputs(self, return_tuple: bool = False, **kwargs):
        dummy_inputs = super().generate_dummy_inputs(**kwargs)
        dummy_inputs["encoder_hidden_states"] = dummy_inputs["encoder_hidden_states"][0]

        # break down down_block_additional_residuals
        num_down_block_outputs = len(self._normalized_config.down_block_types) * (
            self._normalized_config.layers_per_block + 1
        )
        down_block_additional_residuals = dummy_inputs.pop("down_block_additional_residuals", None)

        if down_block_additional_residuals:
            for idx in range(num_down_block_outputs):
                dummy_inputs[f"down_block_additional_residuals_{idx}"] = down_block_additional_residuals[idx]

        if getattr(self._normalized_config, "addition_embed_type", None) == "text_time":
            dummy_inputs["added_cond_kwargs"] = {
                "text_embeds": dummy_inputs.pop("text_embeds"),
                "time_ids": dummy_inputs.pop("time_ids"),
            }

        if return_tuple is True:
            return tuple(dummy_inputs.values())
        else:
            return dummy_inputs

    @property
    def is_sdxl(self) -> bool:
        return self._is_sdxl

    @is_sdxl.setter
    def is_sdxl(self, is_sdxl: bool):
        self._is_sdxl = is_sdxl

    @property
    def with_controlnet(self) -> bool:
        return self._with_controlnet

    @with_controlnet.setter
    def with_controlnet(self, with_controlnet: bool):
        self._with_controlnet = with_controlnet

    @property
    def with_ip_adapter(self) -> bool:
        return self._with_ip_adapter

    @with_ip_adapter.setter
    def with_ip_adapter(self, with_ip_adapter: bool):
        self._with_ip_adapter = with_ip_adapter
        if with_ip_adapter:
            self.mandatory_axes += ("image_encoder_shapes",)
            setattr(self, "image_encoder_shapes", self.input_shapes["image_encoder_shapes"])


@register_in_tasks_manager("pixart-transformer-2d", *["semantic-segmentation"], library_name="diffusers")
class PixartTransformerNeuronConfig(VisionNeuronConfig):
    ATOL_FOR_VALIDATION = 1e-3
    INPUT_ARGS = (
        "batch_size",
        "sequence_length",
        "num_channels",
        "width",
        "height",
        "vae_scale_factor",
        "encoder_hidden_size",
    )
    MODEL_TYPE = "pixart-transformer-2d"
    CUSTOM_MODEL_WRAPPER = PixartTransformerNeuronWrapper
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        height="height",
        width="width",
        num_channels="in_channels",
        hidden_size="cross_attention_dim",
        vocab_size="norm_num_groups",
        allow_new=True,
    )

    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyVisionInputGenerator,
        DummyTimestepInputGenerator,
        DummyControNetInputGenerator,
        DummyTextInputGenerator,
        DummySeq2SeqDecoderTextInputGenerator,
    )

    @property
    def inputs(self) -> list[str]:
        common_inputs = ["sample", "encoder_hidden_states", "timestep", "encoder_attention_mask"]
        return common_inputs

    @property
    def outputs(self) -> list[str]:
        return ["out_hidden_states"]


@register_in_tasks_manager("flux-transformer-2d", *["semantic-segmentation"], library_name="diffusers")
class FluxTransformerNeuronConfig(VisionNeuronConfig):
    ATOL_FOR_VALIDATION = 1e-3
    INPUT_ARGS = (
        "batch_size",
        "sequence_length",
        "num_channels",
        "width",
        "height",
        "vae_scale_factor",
        "encoder_hidden_size",
        "rotary_axes_dim",
    )
    MODEL_TYPE = "flux-transformer-2d"
    CUSTOM_MODEL_WRAPPER = FluxTransformerNeuronWrapper
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        height="height",
        width="width",
        num_channels="in_channels",
        vocab_size="attention_head_dim",
        hidden_size="joint_attention_dim",
        projection_size="pooled_projection_dim",
        allow_new=True,
    )

    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyTransformerTimestepInputGenerator,
        DummyFluxTransformerVisionInputGenerator,
        DummyFluxTransformerTextInputGenerator,
        DummyTransformerRotaryEmbGenerator,
    )

    @property
    def inputs(self) -> list[str]:
        common_inputs = [
            "hidden_states",
            "encoder_hidden_states",
            "pooled_projections",
            "timestep",
            # Q: Why `image_rotary_emb` but not `txt_ids` and `img_ids`? We compute the rotary positional embeddings in CPU to save Neuron memory.
            # shape: [txt_ids.shape(0)+img_ids.shape(0), sum(axes_dim), 2]
            "image_rotary_emb",
        ]
        if getattr(self._config, "guidance_embeds", False):
            common_inputs.append("guidance")

        return common_inputs

    @property
    def outputs(self) -> list[str]:
        return ["out_hidden_states"]

    def patch_model_and_prepare_aliases(self, model_or_path, *args):
        base_model_instance = BaseModelInstance(
            partial(self.get_parallel_callable, self._config),
            input_output_aliases={},
        )
        return base_model_instance, None

    def get_parallel_callable(self, config):
        from optimum.neuron.models.inference.flux.modeling_flux import NeuronFluxTransformer2DModel

        # Parallelize Flux transformer with NxD backend modeling
        valid_params = inspect.signature(NeuronFluxTransformer2DModel.__init__).parameters
        model_config = {k: v for k, v in config.items() if k in valid_params and k != "self"}
        model = NeuronFluxTransformer2DModel(**model_config)
        model.eval()
        if self.float_dtype == torch.bfloat16:
            model.bfloat16()

        return model

    # Adapted from diffusers.models.modeling_utils.ModelMixin.from_pretrained, this is a helper function for loading checkpoints required by `ModelBuilder`.
    def get_checkpoint_loader_fn(self):
        is_local = os.path.isdir(self.pretrained_model_name_or_path)
        subfolder = getattr(self, "subfolder", "transformer")
        if is_local:
            index_file = Path(
                self.pretrained_model_name_or_path,
                subfolder or "",
                SAFE_WEIGHTS_INDEX_NAME,
            )
        else:
            index_file_in_repo = Path(
                subfolder or "",
                SAFE_WEIGHTS_INDEX_NAME,
            ).as_posix()
            index_file = _get_model_file(
                self.pretrained_model_name_or_path,
                weights_name=index_file_in_repo,
                # TODO: add extra args, eg. revision, trust_remote_code, etc.
            )

        model_shards_file_paths, _ = get_checkpoint_shard_files(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            index_filename=index_file,
            subfolder=subfolder,
        )

        merged_state_dict = {}
        for shard_file in model_shards_file_paths:
            state_dict = load_file(shard_file)
            merged_state_dict.update(state_dict)

        inner_dim = self._config.num_attention_heads * self._config.attention_head_dim
        for i in range(self._config.num_single_layers):
            merged_state_dict[f"single_transformer_blocks.{i}.proj_out_attn.weight"] = merged_state_dict[
                f"single_transformer_blocks.{i}.proj_out.weight"
            ][:, :inner_dim].contiguous()
            merged_state_dict[f"single_transformer_blocks.{i}.proj_out_attn.bias"] = (
                merged_state_dict[f"single_transformer_blocks.{i}.proj_out.bias"].clone().detach().contiguous()
            )
            merged_state_dict[f"single_transformer_blocks.{i}.proj_out_mlp.weight"] = merged_state_dict[
                f"single_transformer_blocks.{i}.proj_out.weight"
            ][:, inner_dim:].contiguous()

        return merged_state_dict


@register_in_tasks_manager("qwen-image-transformer-2d", *["semantic-segmentation"], library_name="diffusers")
class QwenImageTransformerNeuronConfig(FluxTransformerNeuronConfig):
    ATOL_FOR_VALIDATION = 1e-3
    INPUT_ARGS = (
        "batch_size",
        "sequence_length",
        "num_channels",
        "width",
        "height",
        "vae_scale_factor",
        "encoder_hidden_size",
        "rotary_axes_dim",
    )
    MODEL_TYPE = "qwen-image-transformer-2d"
    CUSTOM_MODEL_WRAPPER = QwenImageTransformerNeuronWrapper
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        height="height",
        width="width",
        num_channels="in_channels",
        vocab_size="attention_head_dim",
        hidden_size="joint_attention_dim",
        projection_size="pooled_projection_dim",
        allow_new=True,
    )

    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyTransformerTimestepInputGenerator,
        DummyQwenImageTransformerInputGenerator,
        DummyTransformerRotaryEmbGenerator,
    )

    @property
    def inputs(self) -> list[str]:
        common_inputs = [
            "hidden_states",
            "encoder_hidden_states",
            "pooled_projections",
            "timestep",
            # Q: Why `image_rotary_emb` but not `txt_ids` and `img_ids`? We compute the rotary positional embeddings in CPU to save Neuron memory.
            # shape: [txt_ids.shape(0)+img_ids.shape(0), sum(axes_dim), 2]
            "image_rotary_emb",
        ]
        if getattr(self._config, "guidance_embeds", False):
            common_inputs.append("guidance")

        return common_inputs

    @property
    def outputs(self) -> list[str]:
        return ["out_hidden_states"]

    def patch_model_and_prepare_aliases(self, model_or_path, *args):
        base_model_instance = BaseModelInstance(
            partial(self.get_parallel_callable, self._config),
            input_output_aliases={},
        )
        return base_model_instance, None

    def get_parallel_callable(self, config):
        from optimum.neuron.models.inference.flux.modeling_flux import NeuronFluxTransformer2DModel

        # Parallelize Flux transformer with NxD backend modeling
        valid_params = inspect.signature(NeuronFluxTransformer2DModel.__init__).parameters
        model_config = {k: v for k, v in config.items() if k in valid_params and k != "self"}
        model = NeuronFluxTransformer2DModel(**model_config)
        model.eval()
        if self.float_dtype == torch.bfloat16:
            model.bfloat16()

        return model

    # Adapted from diffusers.models.modeling_utils.ModelMixin.from_pretrained, this is a helper function for loading checkpoints required by `ModelBuilder`.
    def get_checkpoint_loader_fn(self):
        is_local = os.path.isdir(self.pretrained_model_name_or_path)
        subfolder = getattr(self, "subfolder", "transformer")
        if is_local:
            index_file = Path(
                self.pretrained_model_name_or_path,
                subfolder or "",
                SAFE_WEIGHTS_INDEX_NAME,
            )
        else:
            index_file_in_repo = Path(
                subfolder or "",
                SAFE_WEIGHTS_INDEX_NAME,
            ).as_posix()
            index_file = _get_model_file(
                self.pretrained_model_name_or_path,
                weights_name=index_file_in_repo,
                # TODO: add extra args, eg. revision, trust_remote_code, etc.
            )

        model_shards_file_paths, _ = get_checkpoint_shard_files(
            pretrained_model_name_or_path=self.pretrained_model_name_or_path,
            index_filename=index_file,
            subfolder=subfolder,
        )

        merged_state_dict = {}
        for shard_file in model_shards_file_paths:
            state_dict = load_file(shard_file)
            merged_state_dict.update(state_dict)

        inner_dim = self._config.num_attention_heads * self._config.attention_head_dim
        for i in range(self._config.num_single_layers):
            merged_state_dict[f"single_transformer_blocks.{i}.proj_out_attn.weight"] = merged_state_dict[
                f"single_transformer_blocks.{i}.proj_out.weight"
            ][:, :inner_dim].contiguous()
            merged_state_dict[f"single_transformer_blocks.{i}.proj_out_attn.bias"] = (
                merged_state_dict[f"single_transformer_blocks.{i}.proj_out.bias"].clone().detach().contiguous()
            )
            merged_state_dict[f"single_transformer_blocks.{i}.proj_out_mlp.weight"] = merged_state_dict[
                f"single_transformer_blocks.{i}.proj_out.weight"
            ][:, inner_dim:].contiguous()

        return merged_state_dict

    def generate_dummy_inputs(self, return_tuple: bool = False, **kwargs):
        if self.is_flux_kontext:
            self.DUMMY_INPUT_GENERATOR_CLASSES = self.DUMMY_INPUT_GENERATOR_CLASSES + (
                DummyFluxKontextTransformerRotaryEmbGenerator,
            )
            dummy_inputs = super().generate_dummy_inputs(**kwargs)
            dummy_inputs["hidden_states"] = torch.cat(
                [dummy_inputs["hidden_states"], dummy_inputs["hidden_states"]], dim=1
            )
        else:
            self.DUMMY_INPUT_GENERATOR_CLASSES = self.DUMMY_INPUT_GENERATOR_CLASSES + (
                DummyTransformerRotaryEmbGenerator,
            )
            dummy_inputs = super().generate_dummy_inputs(**kwargs)

        if return_tuple is True:
            return tuple(dummy_inputs.values())
        else:
            return dummy_inputs

    @property
    def is_flux_kontext(self) -> bool:
        return self._is_flux_kontext

    @is_flux_kontext.setter
    def is_flux_kontext(self, is_flux_kontext: bool):
        self._is_flux_kontext = is_flux_kontext


@register_in_tasks_manager("controlnet", *["semantic-segmentation"], library_name="diffusers")
class ControlNetNeuronConfig(VisionNeuronConfig):
    ATOL_FOR_VALIDATION = 1e-3
    INPUT_ARGS = (
        "batch_size",
        "sequence_length",
        "num_channels",
        "height",
        "width",
        "vae_scale_factor",
        "encoder_hidden_size",
    )
    MODEL_TYPE = "controlnet"
    CUSTOM_MODEL_WRAPPER = ControlNetNeuronWrapper
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        height="height",
        width="width",
        num_channels="in_channels",
        hidden_size="cross_attention_dim",
        vocab_size="norm_num_groups",
        allow_new=True,
    )

    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyVisionInputGenerator,
        DummyTimestepInputGenerator,
        DummyControNetInputGenerator,  # Instead of `encoder_hidden_states` generated by `DummySeq2SeqDecoderTextInputGenerator`
        DummySeq2SeqDecoderTextInputGenerator,
    )

    @property
    def inputs(self) -> list[str]:
        common_inputs = ["sample", "timestep", "encoder_hidden_states", "controlnet_cond", "conditioning_scale"]

        if getattr(self._normalized_config, "addition_embed_type", None) == "text_time":
            common_inputs.append("text_embeds")
            common_inputs.append("time_ids")

        return common_inputs

    @property
    def outputs(self) -> list[str]:
        return ["down_block_res_samples", "mid_block_res_sample"]


@register_in_tasks_manager("vae-encoder", *["semantic-segmentation"], library_name="diffusers")
class VaeEncoderNeuronConfig(VisionNeuronConfig):
    ATOL_FOR_VALIDATION = 1e-3
    MODEL_TYPE = "vae-encoder"

    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        num_channels="in_channels",
        allow_new=True,
    )

    @property
    def inputs(self) -> list[str]:
        return ["sample"]

    @property
    def outputs(self) -> list[str]:
        return ["latent_parameters"]

    def generate_dummy_inputs(self, return_tuple: bool = False, **kwargs):
        dummy_inputs = super().generate_dummy_inputs(**kwargs)

        if return_tuple is True:
            return tuple(dummy_inputs.values())
        else:
            return dummy_inputs


@register_in_tasks_manager("vae-decoder", *["semantic-segmentation"], library_name="diffusers")
class VaeDecoderNeuronConfig(VisionNeuronConfig):
    ATOL_FOR_VALIDATION = 1e-3
    MODEL_TYPE = "vae-decoder"

    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        num_channels="latent_channels",
        allow_new=True,
    )

    @property
    def inputs(self) -> list[str]:
        return ["latent_sample"]

    @property
    def outputs(self) -> list[str]:
        return ["sample"]

    def patch_model_and_prepare_aliases(
        self,
        model: "VaeDecoder",
        input_names: list[str] = None,
        **kwargs,
    ):
        return super().patch_model_and_prepare_aliases(model=model, input_names=input_names, forward_with_tuple=True)


@register_in_tasks_manager("qwen-image-vae-encoder", *["semantic-segmentation"], library_name="diffusers")
class QwenImageVaeEncoderNeuronConfig(VaeEncoderNeuronConfig):
    pass


@register_in_tasks_manager("qwen-image-vae-decoder", *["semantic-segmentation"], library_name="diffusers")
class QwenImageVaeDecoderNeuronConfig(VaeDecoderNeuronConfig):
    pass


@register_in_tasks_manager("qwen2-5-vl", *["feature-extraction"], library_name="diffusers")
class Qwen2_5_VLEncoderNeuronConfig(TextEncoderNeuronConfig):
    ATOL_FOR_VALIDATION = 1e-3
    NORMALIZED_CONFIG_CLASS = NormalizedTextConfig

    @property
    def inputs(self) -> list[str]:
        return ["input_ids", "attention_mask"]
    

class T5EncoderBaseNeuronConfig(TextSeq2SeqNeuronConfig):
    ATOL_FOR_VALIDATION = 1e-3
    NORMALIZED_CONFIG_CLASS = NormalizedSeq2SeqConfig.with_args(
        hidden_size="d_model",
        num_attention_heads="num_heads",
        encoder_num_layers="num_layers",
        decoder_num_layers="num_decoder_layers",
        key_value_dim="d_kv",
        allow_new=True,
    )

    @property
    def inputs(self) -> list[str]:
        return ["input_ids", "attention_mask"]


@register_in_tasks_manager("t5-encoder", *["feature-extraction"], library_name="diffusers")
class T5EncoderForDiffusersNeuronConfig(T5EncoderBaseNeuronConfig):
    CUSTOM_MODEL_WRAPPER = T5EncoderWrapper
    INPUT_ARGS = ("batch_size", "sequence_length")
    MODEL_TYPE = "t5-encoder"
    LIBRARY_NAME = "diffusers"

    @property
    def inputs(self) -> list[str]:
        return ["input_ids"]

    @property
    def outputs(self) -> list[str]:
        return ["last_hidden_state"]

    @property
    def is_encoder_decoder(self) -> bool:
        return True

    def patch_model_and_prepare_aliases(self, model_or_path, device="cpu", **input_shapes):
        batch_size = input_shapes.pop("batch_size", None)
        sequence_length = input_shapes.pop("sequence_length", None)
        if self.tensor_parallel_size > 1:
            # `torch.nn.modules` objects not eligible for pickling, the model needs to be loaded within the func.
            return partial(
                self.get_parallel_callable,
                model_or_path,
                sequence_length,
                batch_size,
                device,
                self.tensor_parallel_size,
            ), None
        else:
            return self.CUSTOM_MODEL_WRAPPER(
                model_or_path,
                sequence_length=sequence_length,
                batch_size=batch_size,
                device=device,
                tensor_parallel_size=self.tensor_parallel_size,
            ), {}

    def get_parallel_callable(self, model_name_or_path, sequence_length, batch_size, device, tensor_parallel_size):
        """Unlike `torch_neuronx.trace`, `parallel_model_trace` requires a function returning a model object and a dictionary of states."""

        pipe = TasksManager.get_model_from_task(
            model_name_or_path=model_name_or_path,
            task=self.task,
            torch_dtype=torch.bfloat16,
            framework="pt",
            library_name="diffusers",
        )  # TODO: add extra args, eg. revision, trust_remote_code, etc.
        text_encoder = pipe.text_encoder_2
        text_encoder.eval()

        # Parallelize the encoder with its custom wrapper
        sharded_text_encoder = self.CUSTOM_MODEL_WRAPPER(
            text_encoder,
            sequence_length=sequence_length,
            batch_size=batch_size,
            device=device,
            tensor_parallel_size=tensor_parallel_size,
        )

        return sharded_text_encoder, {}


@register_in_tasks_manager("t5-encoder", *["text2text-generation"])
class T5EncoderForTransformersNeuronConfig(T5EncoderBaseNeuronConfig):
    CUSTOM_MODEL_WRAPPER = T5EncoderForSeq2SeqLMWrapper
    INPUT_ARGS = ("batch_size", "sequence_length", "num_beams")
    MODEL_TYPE = "t5-encoder"

    @property
    def outputs(self) -> list[str]:
        common_outputs = (
            [f"present.{idx}.self.key" for idx in range(self._config.num_decoder_layers)]
            + [f"present.{idx}.self.value" for idx in range(self._config.num_decoder_layers)]
            + [f"present.{idx}.cross.key" for idx in range(self._config.num_decoder_layers)]
            + [f"present.{idx}.cross.value" for idx in range(self._config.num_decoder_layers)]
        )
        return common_outputs

    @property
    def is_encoder_decoder(self) -> bool:
        return True

    def patch_model_and_prepare_aliases(self, model_or_path, device="xla", **kwargs):
        num_beams = kwargs.pop("num_beams", 1)
        sequence_length = kwargs.pop("sequence_length", None)
        batch_size = kwargs.pop("batch_size", None)

        if self.tensor_parallel_size > 1:
            # `torch.nn.modules` objects not eligible for pickling, the model needs to be loaded within the func.
            return partial(
                self.get_parallel_callable,
                model_or_path,
                sequence_length,
                batch_size,
                num_beams,
                device,
                self.tensor_parallel_size,
            ), None
        else:
            # Override T5 encoder and build aliases
            checked_model = self.CUSTOM_MODEL_WRAPPER(
                model_or_path,
                sequence_length=sequence_length,
                batch_size=batch_size,
                num_beams=num_beams,
                device=device,
                tensor_parallel_size=self.tensor_parallel_size,
            )
            aliases = self.generate_io_aliases(checked_model)

            return checked_model, aliases

    def get_parallel_callable(
        self, model_name_or_path, sequence_length, batch_size, num_beams, device, tensor_parallel_size
    ):
        """Unlike `torch_neuronx.trace`, `parallel_model_trace` requires a function returning a model object and a dictionary of states."""
        model = TasksManager.get_model_from_task(
            model_name_or_path=model_name_or_path,
            task=self.task,
            framework="pt",
            library_name="transformers",
        )  # TODO: add extra args, eg. revision, trust_remote_code, etc.
        model.config.use_cache = True
        with saved_model_in_temporary_directory(model) as ckpt_path:
            # Plug in parallel layers
            from optimum.neuron.models.inference.t5.modeling_t5 import parallelize

            parallel_model = parallelize(model)
            # Load the weights into the parallel layers
            neuronx_distributed.parallel_layers.load(ckpt_path, parallel_model, sharded=False)
        encoder = self.CUSTOM_MODEL_WRAPPER(
            parallel_model,
            sequence_length=sequence_length,
            batch_size=batch_size,
            num_beams=num_beams,
            device=device,
            tensor_parallel_size=tensor_parallel_size,
        )
        encoder.eval()
        aliases = self.generate_io_aliases(encoder)
        return encoder, aliases

    def generate_io_aliases(self, encoder=None):
        aliases = {}
        if self.tensor_parallel_size > 1:
            for i in range(len(encoder.past_key_values_sa)):
                aliases[encoder.past_key_values_sa[i]] = i
            for i in range(len(encoder.past_key_values_ca)):
                aliases[encoder.past_key_values_ca[i]] = len(encoder.past_key_values_sa) + i
        return aliases


@register_in_tasks_manager("t5-decoder", "text2text-generation")
class T5DecoderNeuronConfig(TextSeq2SeqNeuronConfig):
    ATOL_FOR_VALIDATION = 1e-3
    DUMMY_INPUT_GENERATOR_CLASSES = TextSeq2SeqNeuronConfig.DUMMY_INPUT_GENERATOR_CLASSES + (DummyBeamValuesGenerator,)
    INPUT_ARGS = ("batch_size", "sequence_length", "num_beams")
    MODEL_TYPE = "t5-decoder"
    CUSTOM_MODEL_WRAPPER = T5DecoderWrapper
    NORMALIZED_CONFIG_CLASS = NormalizedSeq2SeqConfig

    @property
    def inputs(self) -> list[str]:
        common_inputs = [
            "decoder_input_ids",
            "decoder_attention_mask",
            "encoder_hidden_states",
            "attention_mask",  # TODO: replace with `encoder_attention_mask` after optimum 1.14 release
            "beam_idx",
            "beam_scores",
        ]
        return common_inputs

    @property
    def outputs(self) -> list[str]:
        beam_outputs = ["next_token_scores", "next_tokens", "next_indices"] if self.num_beams > 1 else ["next_tokens"]
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

    @property
    def is_encoder_decoder(self) -> bool:
        return True

    def generate_dummy_inputs(self, **kwargs):
        batch_size = kwargs.pop("batch_size") * kwargs.get("num_beams")
        dummy_inputs = super().generate_dummy_inputs(batch_size=batch_size, **kwargs)
        dummy_inputs["decoder_input_ids"] = dummy_inputs["decoder_input_ids"][:, :1]  # sequence_length = 1
        dummy_inputs["encoder_hidden_states"] = dummy_inputs["encoder_hidden_states"][0]

        return dummy_inputs

    def _create_dummy_input_generator_classes(self, **kwargs) -> list["DummyInputGenerator"]:
        dummy_inputs_generators = super()._create_dummy_input_generator_classes(**kwargs)
        dummy_beam_values_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[-1](
            self.task,
            self._normalized_config,
            num_beams=kwargs.pop("num_beams", 1),
            **kwargs,
        )
        dummy_inputs_generators.append(dummy_beam_values_generator)
        return dummy_inputs_generators

    def patch_model_and_prepare_aliases(self, model, device="xla", **kwargs):
        batch_size = kwargs.pop("batch_size", 1)
        sequence_length = kwargs.pop("sequence_length", 1)
        num_beams = kwargs.pop("num_beams", 1)

        trace_args = {
            "model": model,
            "batch_size": batch_size,
            "sequence_length": sequence_length,
            "num_beams": num_beams,
            "output_hidden_states": self.output_hidden_states,
            "output_attentions": self.output_attentions,
            "device": device,
            "tensor_parallel_size": self.tensor_parallel_size,
        }
        if self.tensor_parallel_size > 1:
            return partial(
                self.get_parallel_callable,
                model,
                batch_size,
                sequence_length,
                num_beams,
                self.output_hidden_states,
                self.output_attentions,
                device,
                self.tensor_parallel_size,
            ), None
        else:
            # Override T5 encoder and build aliases
            checked_model = self.CUSTOM_MODEL_WRAPPER(**trace_args)
            aliases = self.generate_io_aliases(checked_model)

            return checked_model, aliases

    def get_parallel_callable(
        self,
        model_name_or_path,
        batch_size,
        sequence_length,
        num_beams,
        output_hidden_states,
        output_attentions,
        device,
        tensor_parallel_size,
    ):
        """Unlike `torch_neuronx.trace`, `parallel_model_trace` requires a function returning a model object and a dictionary of states."""
        model = TasksManager.get_model_from_task(
            model_name_or_path=model_name_or_path,
            task=self.task,
            framework="pt",
            library_name="transformers",
        )  # TODO: add extra args, eg. revision, trust_remote_code, etc.
        model.config.use_cache = True
        with saved_model_in_temporary_directory(model) as ckpt_path:
            # Plug in parallel layers
            from optimum.neuron.models.inference.t5.modeling_t5 import parallelize

            parallel_model = parallelize(model)
            # Load the weights into the parallel layers
            neuronx_distributed.parallel_layers.load(ckpt_path, parallel_model, sharded=False)
        decoder = self.CUSTOM_MODEL_WRAPPER(
            parallel_model,
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_beams=num_beams,
            output_hidden_states=output_hidden_states,
            output_attentions=output_attentions,
            device=device,
            tensor_parallel_size=tensor_parallel_size,
        )
        decoder.eval()
        aliases = self.generate_io_aliases(decoder)
        return decoder, aliases

    def generate_io_aliases(self, decoder):
        num_outputs_from_trace = 3 if decoder.num_beams > 1 else 1
        aliases = {}
        for i in range(len(decoder.past_key_values_sa)):
            aliases[decoder.past_key_values_sa[i]] = i + num_outputs_from_trace
        for i in range(len(decoder.past_key_values_ca)):
            aliases[decoder.past_key_values_ca[i]] = len(decoder.past_key_values_sa) + i + num_outputs_from_trace

        return aliases


@register_in_tasks_manager("whisper-encoder", *["automatic-speech-recognition"])
class WhisperEncoderNeuronConfig(AudioNeuronConfig):
    ATOL_FOR_VALIDATION = 1e-3
    MODEL_TYPE = "whisper-encoder"
    CUSTOM_MODEL_WRAPPER = WhisperEncoderWrapper
    INPUT_ARGS = ("batch_size", "sequence_length")
    DUMMY_INPUT_GENERATOR_CLASSES = AudioNeuronConfig.DUMMY_INPUT_GENERATOR_CLASSES + (WhisperDummyTextInputGenerator,)
    NORMALIZED_CONFIG_CLASS = NormalizedSeq2SeqConfig.with_args(
        encoder_num_layers="encoder_layers",
        decoder_num_layers="decoder_layers",
        feature_size="num_mel_bins",
        allow_new=True,
    )

    @property
    def inputs(self) -> list[str]:
        return ["input_features", "decoder_input_ids"]

    @property
    def outputs(self) -> list[str]:
        return ["lm_logits", "encoder_last_hidden_state"]

    @property
    def is_encoder_decoder(self) -> bool:
        return True

    def generate_dummy_inputs(self, return_tuple: bool = False, **kwargs):
        kwargs["sequence_length"] = 1  # only `decoder_start_token_id`
        return super().generate_dummy_inputs(return_tuple=return_tuple, **kwargs)

    def patch_model_and_prepare_aliases(self, model_or_path, **input_shapes):
        return self.CUSTOM_MODEL_WRAPPER(model_or_path, **input_shapes), {}


@register_in_tasks_manager("whisper-decoder", *["automatic-speech-recognition"])
class WhisperDecoderNeuronConfig(AudioNeuronConfig):
    ATOL_FOR_VALIDATION = 1e-3
    MODEL_TYPE = "whisper-decoder"
    DUMMY_INPUT_GENERATOR_CLASSES = (WhisperDummyTextInputGenerator,)
    INPUT_ARGS = ("batch_size", "sequence_length")
    CUSTOM_MODEL_WRAPPER = WhisperDecoderWrapper
    NORMALIZED_CONFIG_CLASS = NormalizedSeq2SeqConfig.with_args(
        encoder_num_layers="encoder_layers",
        decoder_num_layers="decoder_layers",
        feature_size="num_mel_bins",
        hidden_size="d_model",
        allow_new=True,
    )

    @property
    def inputs(self) -> list[str]:
        return ["decoder_input_ids", "encoder_hidden_states"]

    @property
    def outputs(self) -> list[str]:
        return ["lm_logits"]

    @property
    def is_encoder_decoder(self) -> bool:
        return True

    def patch_model_and_prepare_aliases(self, model_or_path, **input_shapes):
        return self.CUSTOM_MODEL_WRAPPER(model_or_path, **input_shapes), {}
