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
"""NeuronModelForXXX classes for inference on neuron devices using the same API as Transformers."""

import logging
from typing import TYPE_CHECKING

import torch
from transformers import (
    AutoModel,
    AutoModelForAudioClassification,
    AutoModelForAudioFrameClassification,
    AutoModelForAudioXVector,
    AutoModelForCTC,
    AutoModelForImageClassification,
    AutoModelForMaskedLM,
    AutoModelForMultipleChoice,
    AutoModelForObjectDetection,
    AutoModelForQuestionAnswering,
    AutoModelForSemanticSegmentation,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    CausalLMOutput,
    ImageClassifierOutput,
    MaskedLMOutput,
    ModelOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SemanticSegmenterOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    XVectorOutput,
)

from .modeling_traced import NeuronTracedModel
from .utils.doc import (
    _GENERIC_PROCESSOR,
    _PROCESSOR_FOR_IMAGE,
    _TOKENIZER_FOR_DOC,
    NEURON_AUDIO_CLASSIFICATION_EXAMPLE,
    NEURON_AUDIO_FRAME_CLASSIFICATION_EXAMPLE,
    NEURON_AUDIO_INPUTS_DOCSTRING,
    NEURON_AUDIO_XVECTOR_EXAMPLE,
    NEURON_CTC_EXAMPLE,
    NEURON_FEATURE_EXTRACTION_EXAMPLE,
    NEURON_IMAGE_CLASSIFICATION_EXAMPLE,
    NEURON_IMAGE_INPUTS_DOCSTRING,
    NEURON_MASKED_LM_EXAMPLE,
    NEURON_MODEL_START_DOCSTRING,
    NEURON_MULTIPLE_CHOICE_EXAMPLE,
    NEURON_OBJECT_DETECTION_EXAMPLE,
    NEURON_QUESTION_ANSWERING_EXAMPLE,
    NEURON_SEMANTIC_SEGMENTATION_EXAMPLE,
    NEURON_SEQUENCE_CLASSIFICATION_EXAMPLE,
    NEURON_TEXT_INPUTS_DOCSTRING,
    NEURON_TOKEN_CLASSIFICATION_EXAMPLE,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)


if TYPE_CHECKING:
    pass


logger = logging.getLogger(__name__)


@add_start_docstrings(
    """
    Neuron Model with a BaseModelOutput for feature-extraction tasks.
    """,
    NEURON_MODEL_START_DOCSTRING,
)
class NeuronModelForFeatureExtraction(NeuronTracedModel):
    """
    Feature Extraction model on Neuron devices.
    """

    auto_model_class = AutoModel

    @add_start_docstrings_to_model_forward(
        NEURON_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + NEURON_FEATURE_EXTRACTION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="NeuronModelForFeatureExtraction",
            checkpoint="optimum/all-MiniLM-L6-v2-neuronx",
        )
    )
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        **kwargs,
    ):
        neuron_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            neuron_inputs["token_type_ids"] = token_type_ids

        with self.neuron_padding_manager(neuron_inputs) as inputs:
            outputs = self.model(*inputs)
            # last_hidden_state -> (batch_size, sequencen_len, hidden_size)
            last_hidden_state = self.remove_padding(
                [outputs[0]], dims=[0, 1], indices=[input_ids.shape[0], input_ids.shape[1]]
            )[0]  # Remove padding on batch_size(0), and sequence_length(1)
            if len(outputs) > 1:
                # pooler_output -> (batch_size, hidden_size)
                pooler_output = self.remove_padding([outputs[1]], dims=[0], indices=[input_ids.shape[0]])[
                    0
                ]  # Remove padding on batch_size(0)
            else:
                pooler_output = None

        return BaseModelOutputWithPooling(last_hidden_state=last_hidden_state, pooler_output=pooler_output)


@add_start_docstrings(
    """
    Neuron Model with a MaskedLMOutput for masked language modeling tasks.
    """,
    NEURON_MODEL_START_DOCSTRING,
)
class NeuronModelForMaskedLM(NeuronTracedModel):
    """
    Masked language model for on Neuron devices.
    """

    auto_model_class = AutoModelForMaskedLM

    @add_start_docstrings_to_model_forward(
        NEURON_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + NEURON_MASKED_LM_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="NeuronModelForMaskedLM",
            checkpoint="optimum/legal-bert-base-uncased-neuronx",
        )
    )
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        **kwargs,
    ):
        neuron_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            neuron_inputs["token_type_ids"] = token_type_ids

        with self.neuron_padding_manager(neuron_inputs) as inputs:
            outputs = self.model(*inputs)  # shape: (batch_size, sequencen_len, vocab_size)
            outputs = self.remove_padding(
                outputs, dims=[0, 1], indices=[input_ids.shape[0], input_ids.shape[1]]
            )  # Remove padding on batch_size(0), and sequence_length(1)

        logits = outputs[0]

        return MaskedLMOutput(logits=logits)


@add_start_docstrings(
    """
    Neuron Model with a QuestionAnsweringModelOutput for extractive question-answering tasks like SQuAD.
    """,
    NEURON_MODEL_START_DOCSTRING,
)
class NeuronModelForQuestionAnswering(NeuronTracedModel):
    """
    Question Answering model on Neuron devices.
    """

    auto_model_class = AutoModelForQuestionAnswering

    @add_start_docstrings_to_model_forward(
        NEURON_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + NEURON_QUESTION_ANSWERING_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="NeuronModelForQuestionAnswering",
            checkpoint="optimum/roberta-base-squad2-neuronx",
        )
    )
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        **kwargs,
    ):
        neuron_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            neuron_inputs["token_type_ids"] = token_type_ids

        with self.neuron_padding_manager(neuron_inputs) as inputs:
            outputs = self.model(*inputs)  # shape: [batch_size, sequence_length]
            outputs = self.remove_padding(
                outputs, dims=[0, 1], indices=[input_ids.shape[0], input_ids.shape[1]]
            )  # Remove padding on batch_size(0), and sequence_length(1)

        start_logits = outputs[0]
        end_logits = outputs[1]

        return QuestionAnsweringModelOutput(start_logits=start_logits, end_logits=end_logits)


@add_start_docstrings(
    """
    Neuron Model with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    NEURON_MODEL_START_DOCSTRING,
)
class NeuronModelForSequenceClassification(NeuronTracedModel):
    """
    Sequence Classification model on Neuron devices.
    """

    auto_model_class = AutoModelForSequenceClassification

    @add_start_docstrings_to_model_forward(
        NEURON_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + NEURON_SEQUENCE_CLASSIFICATION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="NeuronModelForSequenceClassification",
            checkpoint="optimum/distilbert-base-uncased-finetuned-sst-2-english-neuronx",
        )
    )
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        **kwargs,
    ):
        neuron_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            neuron_inputs["token_type_ids"] = token_type_ids

        with self.neuron_padding_manager(neuron_inputs) as inputs:
            outputs = self.model(*inputs)  # shape: [batch_size, num_labels]
            outputs = self.remove_padding(
                outputs, dims=[0], indices=[input_ids.shape[0]]
            )  # Remove padding on batch_size(0)

        logits = outputs[0]

        return SequenceClassifierOutput(logits=logits)


@add_start_docstrings(
    """
    Neuron Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    NEURON_MODEL_START_DOCSTRING,
)
class NeuronModelForTokenClassification(NeuronTracedModel):
    """
    Token Classification model on Neuron devices.
    """

    auto_model_class = AutoModelForTokenClassification

    @add_start_docstrings_to_model_forward(
        NEURON_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + NEURON_TOKEN_CLASSIFICATION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="NeuronModelForTokenClassification",
            checkpoint="optimum/bert-base-NER-neuronx",
        )
    )
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        **kwargs,
    ):
        neuron_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            neuron_inputs["token_type_ids"] = token_type_ids

        # run inference
        with self.neuron_padding_manager(neuron_inputs) as inputs:
            outputs = self.model(*inputs)  # shape: [batch_size, sequence_length, num_labels]
            outputs = self.remove_padding(
                outputs, dims=[0, 1], indices=[input_ids.shape[0], input_ids.shape[1]]
            )  # Remove padding on batch_size(0), and sequence_length(-1)

        logits = outputs[0]

        return TokenClassifierOutput(logits=logits)


@add_start_docstrings(
    """
    Neuron Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    NEURON_MODEL_START_DOCSTRING,
)
class NeuronModelForMultipleChoice(NeuronTracedModel):
    """
    Multiple choice model on Neuron devices.
    """

    auto_model_class = AutoModelForMultipleChoice

    @add_start_docstrings_to_model_forward(
        NEURON_TEXT_INPUTS_DOCSTRING.format("batch_size, num_choices, sequence_length")
        + NEURON_MULTIPLE_CHOICE_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="NeuronModelForMultipleChoice",
            checkpoint="optimum/bert-base-uncased_SWAG-neuronx",
        )
    )
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
        **kwargs,
    ):
        neuron_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            neuron_inputs["token_type_ids"] = token_type_ids

        # run inference
        with self.neuron_padding_manager(neuron_inputs) as inputs:
            outputs = self.model(*inputs)  # shape: [batch_size, num_choices]
            outputs = self.remove_padding(
                outputs, dims=[0, -1], indices=[input_ids.shape[0], input_ids.shape[1]]
            )  # Remove padding on batch_size(0), and num_choices(-1)

        logits = outputs[0]

        return MultipleChoiceModelOutput(logits=logits)


@add_start_docstrings(
    """
    Neuron Model with an image classification head on top (a linear layer on top of the final hidden state of the [CLS] token) e.g. for ImageNet.
    """,
    NEURON_MODEL_START_DOCSTRING,
)
class NeuronModelForImageClassification(NeuronTracedModel):
    """
    Neuron Model for image-classification tasks. This class officially supports beit, convnext, convnextv2, deit, levit, mobilenet_v2, mobilevit, vit, etc.
    """

    auto_model_class = AutoModelForImageClassification

    @property
    def dtype(self) -> "torch.dtype | None":
        """
        Torch dtype of the inputs to avoid error in transformers on casting a BatchFeature to type None.
        """
        return getattr(self.config.neuron, "input_dtype", torch.float32)

    @add_start_docstrings_to_model_forward(
        NEURON_IMAGE_INPUTS_DOCSTRING.format("batch_size, num_channels, height, width")
        + NEURON_IMAGE_CLASSIFICATION_EXAMPLE.format(
            processor_class=_PROCESSOR_FOR_IMAGE,
            model_class="NeuronModelForImageClassification",
            checkpoint="optimum/vit-base-patch16-224-neuronx",
        )
    )
    def forward(
        self,
        pixel_values: torch.Tensor,
        **kwargs,
    ):
        neuron_inputs = {"pixel_values": pixel_values}

        # run inference
        with self.neuron_padding_manager(neuron_inputs) as inputs:
            outputs = self.model(*inputs)  # shape: [batch_size, num_channels, image_size, image_size]
            outputs = self.remove_padding(
                outputs, dims=[0], indices=[pixel_values.shape[0]]
            )  # Remove padding on batch_size(0)

        logits = outputs[0]

        return ImageClassifierOutput(logits=logits)


@add_start_docstrings(
    """
    Neuron Model with a semantic segmentation head on top, e.g. for Pascal VOC.
    """,
    NEURON_MODEL_START_DOCSTRING,
)
class NeuronModelForSemanticSegmentation(NeuronTracedModel):
    """
    Neuron Model for semantic-segmentation, with an all-MLP decode head on top e.g. for ADE20k, CityScapes. This class officially supports mobilevit, mobilenet-v2, etc.
    """

    auto_model_class = AutoModelForSemanticSegmentation

    @property
    def dtype(self) -> "torch.dtype | None":
        """
        Torch dtype of the inputs to avoid error in transformers on casting a BatchFeature to type None.
        """
        return getattr(self.config.neuron, "input_dtype", torch.float32)

    @add_start_docstrings_to_model_forward(
        NEURON_IMAGE_INPUTS_DOCSTRING.format("batch_size, num_channels, height, width")
        + NEURON_SEMANTIC_SEGMENTATION_EXAMPLE.format(
            processor_class=_PROCESSOR_FOR_IMAGE,
            model_class="NeuronModelForSemanticSegmentation",
            checkpoint="optimum/deeplabv3-mobilevit-small-neuronx",
        )
    )
    def forward(
        self,
        pixel_values: torch.Tensor,
        **kwargs,
    ):
        neuron_inputs = {"pixel_values": pixel_values}

        # run inference
        with self.neuron_padding_manager(neuron_inputs) as inputs:
            outputs = self.model(*inputs)  # shape: [batch_size, num_channels, image_size, image_size]
            outputs = self.remove_padding(
                outputs, dims=[0], indices=[pixel_values.shape[0]]
            )  # Remove padding on batch_size(0)

        logits = outputs[0]

        return SemanticSegmenterOutput(logits=logits)


@add_start_docstrings(
    """
    Neuron Model with object detection heads on top, for tasks such as COCO detection.
    """,
    NEURON_MODEL_START_DOCSTRING,
)
class NeuronModelForObjectDetection(NeuronTracedModel):
    """
    Neuron Model for object-detection, with object detection heads on top, for tasks such as COCO detection.
    """

    auto_model_class = AutoModelForObjectDetection

    @property
    def dtype(self) -> "torch.dtype | None":
        """
        Torch dtype of the inputs to avoid error in transformers on casting a BatchFeature to type None.
        """
        return getattr(self.config.neuron, "input_dtype", torch.float32)

    @add_start_docstrings_to_model_forward(
        NEURON_IMAGE_INPUTS_DOCSTRING.format("batch_size, num_channels, height, width")
        + NEURON_OBJECT_DETECTION_EXAMPLE.format(
            processor_class=_PROCESSOR_FOR_IMAGE,
            model_class="NeuronModelForObjectDetection",
            checkpoint="hustvl/yolos-tiny",
        )
    )
    def forward(
        self,
        pixel_values: torch.Tensor,
        **kwargs,
    ):
        neuron_inputs = {"pixel_values": pixel_values}

        # run inference
        with self.neuron_padding_manager(neuron_inputs) as inputs:
            outputs = self.model(*inputs)  # shape: [batch_size, num_channels, image_size, image_size]
            outputs = self.remove_padding(
                outputs, dims=[0], indices=[pixel_values.shape[0]]
            )  # Remove padding on batch_size(0)

        logits = outputs[0]
        pred_boxes = outputs[1]
        last_hidden_state = outputs[2]

        return ModelOutput(logits=logits, pred_boxes=pred_boxes, last_hidden_state=last_hidden_state)


@add_start_docstrings(
    """
    Neuron Model with an audio classification head.
    """,
    NEURON_MODEL_START_DOCSTRING,
)
class NeuronModelForAudioClassification(NeuronTracedModel):
    """
    Neuron Model for audio-classification, with a sequence classification head on top (a linear layer over the pooled output) for tasks like
    SUPERB Keyword Spotting.
    """

    auto_model_class = AutoModelForAudioClassification

    @add_start_docstrings_to_model_forward(
        NEURON_AUDIO_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + NEURON_AUDIO_CLASSIFICATION_EXAMPLE.format(
            processor_class=_GENERIC_PROCESSOR,
            model_class="NeuronModelForAudioClassification",
            checkpoint="Jingya/wav2vec2-large-960h-lv60-self-neuronx-audio-classification",
        )
    )
    def forward(
        self,
        input_values: torch.Tensor,
        **kwargs,
    ):
        neuron_inputs = {"input_values": input_values}

        # run inference
        with self.neuron_padding_manager(neuron_inputs) as inputs:
            outputs = self.model(*inputs)  # shape: [batch_size, num_labels]
            outputs = self.remove_padding(
                outputs, dims=[0], indices=[input_values.shape[0]]
            )  # Remove padding on batch_size(0)

        logits = outputs[0]

        return SequenceClassifierOutput(logits=logits)


@add_start_docstrings(
    """
    Neuron Model with an audio frame classification head.
    """,
    NEURON_MODEL_START_DOCSTRING,
)
class NeuronModelForAudioFrameClassification(NeuronTracedModel):
    """
    Neuron Model with a frame classification head on top for tasks like Speaker Diarization.
    """

    auto_model_class = AutoModelForAudioFrameClassification

    @add_start_docstrings_to_model_forward(
        NEURON_AUDIO_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + NEURON_AUDIO_FRAME_CLASSIFICATION_EXAMPLE.format(
            processor_class=_GENERIC_PROCESSOR,
            model_class="NeuronModelForAudioFrameClassification",
            checkpoint="Jingya/wav2vec2-base-superb-sd-neuronx",
        )
    )
    def forward(
        self,
        input_values: torch.Tensor,
        **kwargs,
    ):
        neuron_inputs = {"input_values": input_values}

        # run inference
        with self.neuron_padding_manager(neuron_inputs) as inputs:
            outputs = self.model(*inputs)  # shape: [batch_size, num_labels]
            outputs = self.remove_padding(
                outputs, dims=[0], indices=[input_values.shape[0]]
            )  # Remove padding on batch_size(0)

        logits = outputs[0]

        return TokenClassifierOutput(logits=logits)


@add_start_docstrings(
    """
    Neuron Model with a connectionist temporal classification head.
    """,
    NEURON_MODEL_START_DOCSTRING,
)
class NeuronModelForCTC(NeuronTracedModel):
    """
    Neuron Model with a language modeling head on top for Connectionist Temporal Classification (CTC).
    """

    auto_model_class = AutoModelForCTC
    main_input_name = "input_values"

    @add_start_docstrings_to_model_forward(
        NEURON_AUDIO_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + NEURON_CTC_EXAMPLE.format(
            processor_class=_GENERIC_PROCESSOR,
            model_class="NeuronModelForCTC",
            checkpoint="Jingya/wav2vec2-large-960h-lv60-self-neuronx-ctc",
        )
    )
    def forward(
        self,
        input_values: torch.Tensor,
        **kwargs,
    ):
        neuron_inputs = {"input_values": input_values}

        # run inference
        with self.neuron_padding_manager(neuron_inputs) as inputs:
            outputs = self.model(*inputs)  # shape: [batch_size, sequence_length]
            outputs = self.remove_padding(
                outputs, dims=[0], indices=[input_values.shape[0]]
            )  # Remove padding on batch_size(0)

        logits = outputs[0]

        return CausalLMOutput(logits=logits)


@add_start_docstrings(
    """
    Neuron Model with an XVector feature extraction head on top for tasks like Speaker Verification.
    """,
    NEURON_MODEL_START_DOCSTRING,
)
class NeuronModelForXVector(NeuronTracedModel):
    """
    Neuron Model with an XVector feature extraction head on top for tasks like Speaker Verification.
    """

    auto_model_class = AutoModelForAudioXVector

    @add_start_docstrings_to_model_forward(
        NEURON_AUDIO_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + NEURON_AUDIO_XVECTOR_EXAMPLE.format(
            processor_class=_GENERIC_PROCESSOR,
            model_class="NeuronModelForXVector",
            checkpoint="Jingya/wav2vec2-base-superb-sv-neuronx",
        )
    )
    def forward(
        self,
        input_values: torch.Tensor,
        **kwargs,
    ):
        neuron_inputs = {"input_values": input_values}

        # run inference
        with self.neuron_padding_manager(neuron_inputs) as inputs:
            outputs = self.model(*inputs)  # shape: [batch_size, num_labels]
            outputs = self.remove_padding(
                outputs, dims=[0], indices=[input_values.shape[0]]
            )  # Remove padding on batch_size(0)

        logits = outputs[0]
        embeddings = outputs[1]

        return XVectorOutput(logits=logits, embeddings=embeddings)
