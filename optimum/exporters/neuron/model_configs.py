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

from ...utils import (
    DummySeq2SeqDecoderTextInputGenerator,
    DummyTimestepInputGenerator,
    DummyVisionInputGenerator,
    NormalizedConfig,
    NormalizedConfigManager,
    NormalizedTextAndVisionConfig,
    is_diffusers_available,
)
from ..tasks import TasksManager
from .config import (
    TextAndVisionNeuronConfig,
    TextEncoderNeuronConfig,
    TextNeuronDecoderConfig,
    VisionNeuronConfig,
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
    ATOL_FOR_VALIDATION = 1e-4

    @property
    def inputs(self) -> List[str]:
        return ["input_ids", "attention_mask", "token_type_ids"]


@register_in_tasks_manager("albert", *COMMON_TEXT_TASKS)
class AlbertNeuronConfig(BertNeuronConfig):
    pass


@register_in_tasks_manager("convbert", *COMMON_TEXT_TASKS)
class ConvBertNeuronConfig(BertNeuronConfig):
    @property
    def outputs(self) -> List[str]:
        if self.task == "feature-extraction":
            return ["last_hidden_state"]
        return self._TASK_TO_COMMON_OUTPUTS[self.task]


@register_in_tasks_manager("electra", *COMMON_TEXT_TASKS)
class ElectraNeuronConfig(ConvBertNeuronConfig):
    pass


@register_in_tasks_manager("flaubert", *COMMON_TEXT_TASKS)
class FlaubertNeuronConfig(ConvBertNeuronConfig):
    pass


@register_in_tasks_manager("mobilebert", *COMMON_TEXT_TASKS)
class MobileBertNeuronConfig(BertNeuronConfig):
    pass


@register_in_tasks_manager("roformer", *COMMON_TEXT_TASKS)
class RoFormerNeuronConfig(ConvBertNeuronConfig):
    pass


@register_in_tasks_manager("xlm", *COMMON_TEXT_TASKS)
class XLMNeuronConfig(ConvBertNeuronConfig):
    pass


@register_in_tasks_manager("distilbert", *COMMON_TEXT_TASKS)
class DistilBertNeuronConfig(BertNeuronConfig):
    ATOL_FOR_VALIDATION = 1e-4

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
    ATOL_FOR_VALIDATION = 1e-4

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
@register_in_tasks_manager("deberta", *COMMON_TEXT_TASKS)
class DebertaNeuronConfig(BertNeuronConfig):
    @property
    def inputs(self) -> List[str]:
        common_inputs = super().inputs
        if self._config.type_vocab_size == 0:
            # We remove token type ids.
            common_inputs.pop(-1)
        return common_inputs


# https://github.com/aws-neuron/aws-neuron-sdk/issues/642
# Failed only for INF1: 'XSoftmax'
@register_in_tasks_manager("deberta-v2", *COMMON_TEXT_TASKS)
class DebertaV2NeuronConfig(DebertaNeuronConfig):
    pass


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


@register_in_tasks_manager("clip-text-model", *["feature-extraction"])
class CLIPTextWithProjectionNeuronConfig(TextEncoderNeuronConfig):
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
            for i in range(self._normalized_config.num_layers + 1):
                common_outputs.append(f"hidden_states.{i}")

        return common_outputs


@register_in_tasks_manager("clip-text-model", *["stable-diffusion", "feature-extraction"])
class CLIPTextNeuronConfig(CLIPTextWithProjectionNeuronConfig):
    MODEL_TYPE = "clip-text-model"

    @property
    def outputs(self) -> List[str]:
        common_outputs = ["last_hidden_state", "pooler_output"]

        if self._normalized_config.output_hidden_states:
            for i in range(self._normalized_config.num_layers + 1):
                common_outputs.append(f"hidden_states.{i}")

        return common_outputs

    def generate_dummy_inputs(self, return_tuple: bool = False, **kwargs):
        dummy_inputs = super().generate_dummy_inputs(**kwargs)
        dummy_inputs["input_ids"] = dummy_inputs["input_ids"]

        if return_tuple is True:
            return tuple(dummy_inputs.values())
        else:
            return dummy_inputs

    def check_model_inputs_order(self, model, dummy_inputs, forward_with_tuple=False):
        return super().check_model_inputs_order(model, dummy_inputs, forward_with_tuple, eligible_outputs=[0])


@register_in_tasks_manager("unet", *["stable-diffusion", "semantic-segmentation"])
class UNetNeuronConfig(VisionNeuronConfig):
    ATOL_FOR_VALIDATION = 1e-3
    MANDATORY_AXES = ("batch_size", "sequence_length", "num_channels", "width", "height")
    MODEL_TYPE = "unet"

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

        return common_inputs

    @property
    def outputs(self) -> List[str]:
        return ["sample"]

    def generate_dummy_inputs(self, return_tuple: bool = False, **kwargs):
        dummy_inputs = super().generate_dummy_inputs(**kwargs)
        dummy_inputs["timestep"] = dummy_inputs["timestep"].float()
        dummy_inputs["encoder_hidden_states"] = dummy_inputs["encoder_hidden_states"][0]

        if return_tuple is True:
            return tuple(dummy_inputs.values())
        else:
            return dummy_inputs


@register_in_tasks_manager("vae-encoder", *["stable-diffusion", "semantic-segmentation"])
class VaeEncoderNeuronConfig(VisionNeuronConfig):
    ATOL_FOR_VALIDATION = 1e-2
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


@register_in_tasks_manager("vae-decoder", *["stable-diffusion", "semantic-segmentation"])
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

    def check_model_inputs_order(
        self,
        model: "VaeDecoder",
        dummy_inputs: Dict[str, torch.Tensor],
        **kwargs,
    ):
        return super().check_model_inputs_order(model=model, dummy_inputs=dummy_inputs, forward_with_tuple=True)


@register_in_tasks_manager("gpt2", "text-generation")
class GPT2NeuronConfig(TextNeuronDecoderConfig):
    NEURONX_ARGS = ["n_positions"]
    NEURONX_CLASS = "gpt2.model.GPT2ForSampling"
