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
from functools import partial
from typing import TYPE_CHECKING, Dict, List

import torch

from optimum.exporters.tasks import TasksManager
from optimum.utils import (
    DummyInputGenerator,
    DummySeq2SeqDecoderTextInputGenerator,
    DummyTextInputGenerator,
    DummyTimestepInputGenerator,
    DummyVisionInputGenerator,
    NormalizedConfig,
    NormalizedConfigManager,
    NormalizedSeq2SeqConfig,
    NormalizedTextAndVisionConfig,
    NormalizedTextConfig,
    NormalizedVisionConfig,
    is_diffusers_available,
)

from ....neuron.distributed import ParallelizersManager
from ....neuron.utils import (
    ASTDummyAudioInputGenerator,
    DummyBeamValuesGenerator,
    DummyControNetInputGenerator,
    DummyMaskedPosGenerator,
    is_neuronx_distributed_available,
)
from ..config import (
    AudioNeuronConfig,
    TextAndVisionNeuronConfig,
    TextEncoderNeuronConfig,
    TextSeq2SeqNeuronConfig,
    VisionNeuronConfig,
)
from ..model_wrappers import (
    ControlNetNeuronWrapper,
    NoCacheModelWrapper,
    SentenceTransformersCLIPNeuronWrapper,
    SentenceTransformersTransformerNeuronWrapper,
    T5DecoderWrapper,
    T5EncoderWrapper,
    UnetNeuronWrapper,
)


if is_neuronx_distributed_available():
    import neuronx_distributed

if TYPE_CHECKING:
    if is_diffusers_available():
        from diffusers.models.vae import Decoder as VaeDecoder


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
    def inputs(self) -> List[str]:
        return ["input_ids", "attention_mask", "token_type_ids"]


@register_in_tasks_manager("albert", *COMMON_TEXT_TASKS)
class AlbertNeuronConfig(BertNeuronConfig):
    pass


@register_in_tasks_manager("convbert", *COMMON_TEXT_TASKS)
class ConvBertNeuronConfig(BertNeuronConfig):
    ATOL_FOR_VALIDATION = 1e-1  # TODO: why accuracy more off than other arch

    @property
    def outputs(self) -> List[str]:
        if self.task == "feature-extraction":
            return ["last_hidden_state"]
        return self._TASK_TO_COMMON_OUTPUTS[self.task]


@register_in_tasks_manager("electra", *COMMON_TEXT_TASKS)
class ElectraNeuronConfig(BertNeuronConfig):
    @property
    def outputs(self) -> List[str]:
        if self.task == "feature-extraction":
            return ["last_hidden_state"]
        return self._TASK_TO_COMMON_OUTPUTS[self.task]


@register_in_tasks_manager("esm", *["feature-extraction", "fill-mask", "text-classification", "token-classification"])
class EsmNeuronConfig(TextEncoderNeuronConfig):
    NORMALIZED_CONFIG_CLASS = NormalizedConfigManager.get_normalized_config_class("bert")
    ATOL_FOR_VALIDATION = 1e-3

    @property
    def inputs(self) -> List[str]:
        return ["input_ids", "attention_mask"]


@register_in_tasks_manager("flaubert", *COMMON_TEXT_TASKS)
class FlaubertNeuronConfig(ElectraNeuronConfig):
    pass


@register_in_tasks_manager("mobilebert", *COMMON_TEXT_TASKS)
class MobileBertNeuronConfig(BertNeuronConfig):
    pass


@register_in_tasks_manager("phi", *["feature-extraction", "text-classification", "token-classification"])
class PhiNeuronConfig(ElectraNeuronConfig):
    CUSTOM_MODEL_WRAPPER = NoCacheModelWrapper

    @property
    def inputs(self) -> List[str]:
        return ["input_ids", "attention_mask"]

    def patch_model_for_export(self, model, dummy_inputs):
        return self.CUSTOM_MODEL_WRAPPER(model, list(dummy_inputs.keys()))


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
    def inputs(self) -> List[str]:
        return ["input_ids", "attention_mask"]

    @property
    def outputs(self) -> List[str]:
        if self.task == "feature-extraction":
            return ["last_hidden_state"]
        return self._TASK_TO_COMMON_OUTPUTS[self.task]


@register_in_tasks_manager("camembert", *COMMON_TEXT_TASKS)
class CamembertNeuronConfig(BertNeuronConfig):
    ATOL_FOR_VALIDATION = 1e-3

    @property
    def inputs(self) -> List[str]:
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
    def inputs(self) -> List[str]:
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
    def inputs(self) -> List[str]:
        return ["input_ids", "attention_mask"]

    @property
    def outputs(self) -> List[str]:
        return ["token_embeddings", "sentence_embedding"]

    def patch_model_for_export(self, model, dummy_inputs):
        return self.CUSTOM_MODEL_WRAPPER(model, list(dummy_inputs.keys()))


class CLIPNormalizedConfig(NormalizedTextAndVisionConfig):
    TEXT_CONFIG = "text_config"
    VISION_CONFIG = "vision_config"


@register_in_tasks_manager("clip", *["feature-extraction", "zero-shot-image-classification"])
class CLIPNeuronConfig(TextAndVisionNeuronConfig):
    NORMALIZED_CONFIG_CLASS = CLIPNormalizedConfig

    @property
    def inputs(self) -> List[str]:
        return ["input_ids", "pixel_values", "attention_mask"]

    @property
    def outputs(self) -> List[str]:
        return ["logits_per_image", "logits_per_text", "text_embeds", "image_embeds"]


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
    def inputs(self) -> List[str]:
        return ["input_ids"]

    @property
    def outputs(self) -> List[str]:
        common_outputs = ["text_embeds", "last_hidden_state"]

        if self._normalized_config.output_hidden_states:
            common_outputs.append("hidden_states")

        return common_outputs


@register_in_tasks_manager("clip-text-model", *["feature-extraction"], library_name="diffusers")
class CLIPTextNeuronConfig(CLIPTextWithProjectionNeuronConfig):
    MODEL_TYPE = "clip-text-model"

    @property
    def outputs(self) -> List[str]:
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
    INPUT_ARGS = ("text_batch_size", "image_batch_size", "sequence_length", "num_channels", "width", "height")

    @property
    def outputs(self) -> List[str]:
        return ["text_embeds", "image_embeds"]

    def patch_model_for_export(self, model, dummy_inputs):
        return self.CUSTOM_MODEL_WRAPPER(model, list(dummy_inputs.keys()))

    def _create_dummy_input_generator_classes(self, **kwargs) -> List["DummyInputGenerator"]:
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


@register_in_tasks_manager("vit", *["feature-extraction", "image-classification"])
class ViTNeuronConfig(VisionNeuronConfig):
    ATOL_FOR_VALIDATION = 1e-3
    NORMALIZED_CONFIG_CLASS = NormalizedVisionConfig
    DUMMY_INPUT_GENERATOR_CLASSES = (DummyVisionInputGenerator, DummyMaskedPosGenerator)
    INPUT_ARGS = ("batch_size",)  # `num_channels` and `image_size` are inferred from the config

    @property
    def inputs(self) -> List[str]:
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
    @property
    def outputs(self) -> List[str]:
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
    pass


@register_in_tasks_manager(
    "mobilenet-v2", *["feature-extraction", "image-classification", "semantic-segmentation", "image-segmentation"]
)
class MobileNetV2NeuronConfig(ViTNeuronConfig):
    pass


@register_in_tasks_manager(
    "mobilevit", *["feature-extraction", "image-classification", "semantic-segmentation", "image-segmentation"]
)
class MobileViTNeuronConfig(ViTNeuronConfig):
    pass


@register_in_tasks_manager("swin", *["feature-extraction", "image-classification"])
class SwinNeuronConfig(ViTNeuronConfig):
    pass


@register_in_tasks_manager("yolos", *["feature-extraction", "object-detection"])
class YolosTNeuronConfig(ViTNeuronConfig):
    @property
    def outputs(self) -> List[str]:
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

    @property
    def inputs(self) -> List[str]:
        return ["input_values"]

    @property
    def outputs(self) -> List[str]:
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
    def inputs(self) -> List[str]:
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
    @property
    def outputs(self) -> List[str]:
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
        DummyTimestepInputGenerator,
        DummySeq2SeqDecoderTextInputGenerator,
        DummyControNetInputGenerator,
    )

    @property
    def inputs(self) -> List[str]:
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

        return common_inputs

    @property
    def outputs(self) -> List[str]:
        return ["sample"]

    def generate_dummy_inputs(self, return_tuple: bool = False, **kwargs):
        dummy_inputs = super().generate_dummy_inputs(**kwargs)
        dummy_inputs["timestep"] = dummy_inputs["timestep"].float()
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

    def patch_model_for_export(self, model, dummy_inputs):
        return self.CUSTOM_MODEL_WRAPPER(model, list(dummy_inputs.keys()))

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
        DummyControNetInputGenerator,  # Instead of `encoder_hidden_states` generated by `DummySeq2SeqDecoderTextInputGenerator`
        DummyTimestepInputGenerator,
        DummySeq2SeqDecoderTextInputGenerator,
    )

    @property
    def inputs(self) -> List[str]:
        common_inputs = ["sample", "timestep", "encoder_hidden_states", "controlnet_cond", "conditioning_scale"]

        if getattr(self._normalized_config, "addition_embed_type", None) == "text_time":
            common_inputs.append("text_embeds")
            common_inputs.append("time_ids")

        return common_inputs

    @property
    def outputs(self) -> List[str]:
        return ["down_block_res_samples", "mid_block_res_sample"]

    def patch_model_for_export(self, model, dummy_inputs):
        return self.CUSTOM_MODEL_WRAPPER(model, list(dummy_inputs.keys()))


@register_in_tasks_manager("vae-encoder", *["semantic-segmentation"], library_name="diffusers")
class VaeEncoderNeuronConfig(VisionNeuronConfig):
    ATOL_FOR_VALIDATION = 1e-3
    MODEL_TYPE = "vae-encoder"

    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        num_channels="in_channels",
        allow_new=True,
    )

    @property
    def inputs(self) -> List[str]:
        return ["sample"]

    @property
    def outputs(self) -> List[str]:
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
    def inputs(self) -> List[str]:
        return ["latent_sample"]

    @property
    def outputs(self) -> List[str]:
        return ["sample"]

    def patch_model_for_export(
        self,
        model: "VaeDecoder",
        dummy_inputs: Dict[str, torch.Tensor],
        **kwargs,
    ):
        return super().patch_model_for_export(model=model, dummy_inputs=dummy_inputs, forward_with_tuple=True)


@register_in_tasks_manager("t5-encoder", "text2text-generation")
class T5EncoderNeuronConfig(TextSeq2SeqNeuronConfig):
    ATOL_FOR_VALIDATION = 1e-3
    INPUT_ARGS = ("batch_size", "sequence_length", "num_beams")
    MODEL_TYPE = "t5-encoder"
    CUSTOM_MODEL_WRAPPER = T5EncoderWrapper
    NORMALIZED_CONFIG_CLASS = NormalizedSeq2SeqConfig.with_args(
        hidden_size="d_model",
        num_attention_heads="num_heads",
        encoder_num_layers="num_layers",
        decoder_num_layers="num_decoder_layers",
        key_value_dim="d_kv",
        allow_new=True,
    )

    @property
    def is_decoder(self) -> bool:
        return False

    def patch_model_for_export(self, model_or_path, device="xla", **kwargs):
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
            )
        else:
            return self.CUSTOM_MODEL_WRAPPER(
                model_or_path,
                sequence_length=sequence_length,
                batch_size=batch_size,
                num_beams=num_beams,
                device=device,
                tensor_parallel_size=self.tensor_parallel_size,
            )

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
        parallelizer = ParallelizersManager.parallelizer_for_model(model)
        with parallelizer.saved_model_in_temporary_directory(model) as ckpt_path:
            # Replace parallel layers
            parallel_model = parallelizer._parallelize(model, parallelize_embeddings=False)
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
    def is_decoder(self) -> bool:
        return True

    @property
    def inputs(self) -> List[str]:
        common_inputs = super().inputs + ["beam_idx", "beam_scores"]
        return common_inputs

    def generate_dummy_inputs(self, **kwargs):
        batch_size = kwargs.pop("batch_size") * kwargs.get("num_beams")
        dummy_inputs = super().generate_dummy_inputs(batch_size=batch_size, **kwargs)
        dummy_inputs["decoder_input_ids"] = dummy_inputs["decoder_input_ids"][:, :1]  # sequence_length = 1
        dummy_inputs["encoder_hidden_states"] = dummy_inputs["encoder_hidden_states"][0]

        return dummy_inputs

    def _create_dummy_input_generator_classes(self, **kwargs) -> List["DummyInputGenerator"]:
        dummy_inputs_generators = super()._create_dummy_input_generator_classes(**kwargs)
        dummy_beam_values_generator = self.DUMMY_INPUT_GENERATOR_CLASSES[-1](
            self.task,
            self._normalized_config,
            num_beams=kwargs.pop("num_beams", 1),
            **kwargs,
        )
        dummy_inputs_generators.append(dummy_beam_values_generator)
        return dummy_inputs_generators

    def patch_model_for_export(self, model, device="xla", **kwargs):
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
            )
        else:
            return self.CUSTOM_MODEL_WRAPPER(**trace_args)

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
        parallelizer = ParallelizersManager.parallelizer_for_model(model)
        with parallelizer.saved_model_in_temporary_directory(model) as ckpt_path:
            # Replace parallel layers
            parallel_model = parallelizer._parallelize(model, parallelize_embeddings=False)
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
