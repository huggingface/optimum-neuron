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


from typing import TYPE_CHECKING, Dict, List

import torch

from ...neuron.utils import DummyBeamValuesGenerator
from ...utils import (
    DummyInputGenerator,
    DummySeq2SeqDecoderTextInputGenerator,
    DummyTimestepInputGenerator,
    DummyVisionInputGenerator,
    NormalizedConfig,
    NormalizedConfigManager,
    NormalizedSeq2SeqConfig,
    NormalizedTextAndVisionConfig,
    NormalizedTextConfig,
    is_diffusers_available,
)
from ...utils.normalized_config import T5LikeNormalizedTextConfig
from ..tasks import TasksManager
from .config import (
    TextAndVisionNeuronConfig,
    TextEncoderNeuronConfig,
    TextNeuronDecoderConfig,
    TextSeq2SeqNeuronConfig,
    VisionNeuronConfig,
)
from .model_wrappers import (
    SentenceTransformersCLIPNeuronWrapper,
    SentenceTransformersTransformerNeuronWrapper,
    T5DecoderWrapper,
    T5EncoderWrapper,
    UnetNeuronWrapper,
)


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


@register_in_tasks_manager("flaubert", *COMMON_TEXT_TASKS)
class FlaubertNeuronConfig(ElectraNeuronConfig):
    pass


@register_in_tasks_manager("mobilebert", *COMMON_TEXT_TASKS)
class MobileBertNeuronConfig(BertNeuronConfig):
    pass


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


@register_in_tasks_manager("sentence-transformers-transformer", *["feature-extraction", "sentence-similarity"])
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


@register_in_tasks_manager("clip-text-with-projection", *["feature-extraction"])
class CLIPTextWithProjectionNeuronConfig(TextEncoderNeuronConfig):
    MODEL_TYPE = "clip-text-model"
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


@register_in_tasks_manager("clip-text-model", *["feature-extraction"])
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
@register_in_tasks_manager("sentence-transformers-clip", *["feature-extraction", "sentence-similarity"])
class SentenceTransformersCLIPNeuronConfig(CLIPNeuronConfig):
    CUSTOM_MODEL_WRAPPER = SentenceTransformersCLIPNeuronWrapper
    ATOL_FOR_VALIDATION = 1e-3
    INPUT_ARGS = ("batch_size", "sequence_length", "num_channels", "width", "height")

    @property
    def outputs(self) -> List[str]:
        return ["text_embeds", "image_embeds"]

    def patch_model_for_export(self, model, dummy_inputs):
        return self.CUSTOM_MODEL_WRAPPER(model, list(dummy_inputs.keys()))


@register_in_tasks_manager("unet", *["semantic-segmentation"])
class UNetNeuronConfig(VisionNeuronConfig):
    ATOL_FOR_VALIDATION = 1e-3
    INPUT_ARGS = ("batch_size", "sequence_length", "num_channels", "width", "height")
    MODEL_TYPE = "unet"
    CUSTOM_MODEL_WRAPPER = UnetNeuronWrapper
    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        image_size="sample_size",
        num_channels="in_channels",
        hidden_size="cross_attention_dim",
        vocab_size="norm_num_groups",
        allow_new=True,
    )

    DUMMY_INPUT_GENERATOR_CLASSES = (
        DummyVisionInputGenerator,
        DummyTimestepInputGenerator,
        DummySeq2SeqDecoderTextInputGenerator,
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

        return common_inputs

    @property
    def outputs(self) -> List[str]:
        return ["sample"]

    def generate_dummy_inputs(self, return_tuple: bool = False, **kwargs):
        # For neuron, we use static shape for compiling the unet. Unlike `optimum`, we use the given `height` and `width` instead of the `sample_size`.
        # TODO: Modify optimum.utils.DummyVisionInputGenerator to enable unequal height and width (it prioritize `image_size` to custom h/w now)
        if self.height == self.width:
            self._normalized_config.image_size = self.height
        else:
            raise ValueError(
                "You need to input the same value for `self.height({self.height})` and `self.width({self.width})`."
            )
        dummy_inputs = super().generate_dummy_inputs(**kwargs)
        dummy_inputs["timestep"] = dummy_inputs["timestep"].float()
        dummy_inputs["encoder_hidden_states"] = dummy_inputs["encoder_hidden_states"][0]

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


@register_in_tasks_manager("vae-encoder", *["semantic-segmentation"])
class VaeEncoderNeuronConfig(VisionNeuronConfig):
    ATOL_FOR_VALIDATION = 1e-3
    MODEL_TYPE = "vae-encoder"

    NORMALIZED_CONFIG_CLASS = NormalizedConfig.with_args(
        num_channels="in_channels",
        image_size="sample_size",
        allow_new=True,
    )

    @property
    def inputs(self) -> List[str]:
        return ["sample"]

    @property
    def outputs(self) -> List[str]:
        return ["latent_sample"]

    def generate_dummy_inputs(self, return_tuple: bool = False, **kwargs):
        # For neuron, we use static shape for compiling the unet. Unlike `optimum`, we use the given `height` and `width` instead of the `sample_size`.
        # TODO: Modify optimum.utils.DummyVisionInputGenerator to enable unequal height and width (it prioritize `image_size` to custom h/w now)
        if self.height == self.width:
            self._normalized_config.image_size = self.height
        else:
            raise ValueError(
                "You need to input the same value for `self.height({self.height})` and `self.width({self.width})`."
            )
        dummy_inputs = super().generate_dummy_inputs(**kwargs)

        if return_tuple is True:
            return tuple(dummy_inputs.values())
        else:
            return dummy_inputs


@register_in_tasks_manager("vae-decoder", *["semantic-segmentation"])
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


@register_in_tasks_manager("gpt2", "text-generation")
class GPT2NeuronConfig(TextNeuronDecoderConfig):
    NEURONX_CLASS = "gpt2.model.GPT2ForSampling"


@register_in_tasks_manager("llama", "text-generation")
class LLamaNeuronConfig(TextNeuronDecoderConfig):
    NEURONX_CLASS = "llama.model.LlamaForSampling"


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

    def patch_model_for_export(self, model, device="xla", **kwargs):
        num_beams = kwargs.pop("num_beams", 1)
        return self.CUSTOM_MODEL_WRAPPER(model, num_beams=num_beams, device=device)


@register_in_tasks_manager("opt", "text-generation")
class OPTNeuronConfig(TextNeuronDecoderConfig):
    NEURONX_CLASS = "opt.model.OPTForSampling"


@register_in_tasks_manager("bloom", "text-generation")
class BloomNeuronConfig(TextNeuronDecoderConfig):
    NEURONX_CLASS = "bloom.model.BloomForSampling"


@register_in_tasks_manager("t5-decoder", "text2text-generation")
class T5DecoderNeuronConfig(TextSeq2SeqNeuronConfig):
    ATOL_FOR_VALIDATION = 1e-3
    DUMMY_INPUT_GENERATOR_CLASSES = TextSeq2SeqNeuronConfig.DUMMY_INPUT_GENERATOR_CLASSES + (DummyBeamValuesGenerator,)
    INPUT_ARGS = ("batch_size", "sequence_length", "num_beams")
    MODEL_TYPE = "t5-decoder"
    CUSTOM_MODEL_WRAPPER = T5DecoderWrapper
    NORMALIZED_CONFIG_CLASS = T5LikeNormalizedTextConfig

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

        return self.CUSTOM_MODEL_WRAPPER(
            model,
            batch_size=batch_size,
            sequence_length=sequence_length,
            num_beams=num_beams,
            output_hidden_states=self.output_hidden_states,
            output_attentions=self.output_attentions,
            device=device,
        )

    def generate_io_aliases(self, model):
        num_outputs_from_trace = 3 if model.num_beams > 1 else 1
        aliases = {}
        for i in range(len(model.past_key_values_sa)):
            aliases[model.past_key_values_sa[i]] = i + num_outputs_from_trace
        for i in range(len(model.past_key_values_ca)):
            aliases[model.past_key_values_ca[i]] = len(model.past_key_values_sa) + i + num_outputs_from_trace

        return aliases


@register_in_tasks_manager("mistral", "text-generation")
class MistralNeuronConfig(TextNeuronDecoderConfig):
    NEURONX_CLASS = "mistral.model.MistralForSampling"
