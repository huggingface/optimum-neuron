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

import copy
import logging
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import torch
from transformers import (
    AutoModel,
    AutoModelForAudioClassification,
    AutoModelForAudioFrameClassification,
    AutoModelForAudioXVector,
    AutoModelForCausalLM,
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
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.generation import (
    GenerationMixin,
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

from .generation import TokenSelector
from .modeling_decoder import NeuronDecoderModel
from .modeling_traced import NeuronTracedModel


if TYPE_CHECKING:
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from transformers import GenerationConfig, PretrainedConfig
    from transformers.generation import StoppingCriteriaList


logger = logging.getLogger(__name__)


_TOKENIZER_FOR_DOC = "AutoTokenizer"
_PROCESSOR_FOR_IMAGE = "AutoImageProcessor"
_GENERIC_PROCESSOR = "AutoProcessor"

NEURON_MODEL_START_DOCSTRING = r"""
    This model inherits from [`~neuron.modeling.NeuronTracedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving)

    Args:
        config (`transformers.PretrainedConfig`): [PretrainedConfig](https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig) is the Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`optimum.neuron.modeling.NeuronTracedModel.from_pretrained`] method to load the model weights.
        model (`torch.jit._script.ScriptModule`): [torch.jit._script.ScriptModule](https://pytorch.org/docs/stable/generated/torch.jit.ScriptModule.html) is the TorchScript module with embedded NEFF(Neuron Executable File Format) compiled by neuron(x) compiler.
"""

NEURON_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.Tensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using [`AutoTokenizer`](https://huggingface.co/docs/transformers/autoclass_tutorial#autotokenizer).
            See [`PreTrainedTokenizer.encode`](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizerBase.encode) and
            [`PreTrainedTokenizer.__call__`](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizerBase.__call__) for details.
            [What are input IDs?](https://huggingface.co/docs/transformers/glossary#input-ids)
        attention_mask (`Union[torch.Tensor, None]` of shape `({0})`, defaults to `None`):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](https://huggingface.co/docs/transformers/glossary#attention-mask)
        token_type_ids (`Union[torch.Tensor, None]` of shape `({0})`, defaults to `None`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
            - 1 for tokens that are **sentence A**,
            - 0 for tokens that are **sentence B**.
            [What are token type IDs?](https://huggingface.co/docs/transformers/glossary#token-type-ids)
"""

NEURON_IMAGE_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`Union[torch.Tensor, None]` of shape `({0})`, defaults to `None`):
            Pixel values corresponding to the images in the current batch.
            Pixel values can be obtained from encoded images using [`AutoImageProcessor`](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoImageProcessor).
"""

NEURON_AUDIO_INPUTS_DOCSTRING = r"""
    Args:
        input_values (`torch.Tensor` of shape `({0})`):
            Float values of input raw speech waveform..
            Input values can be obtained from audio file loaded into an array using [`AutoProcessor`](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoProcessor).
"""

FEATURE_EXTRACTION_EXAMPLE = r"""
    Example of feature extraction:
    *(Following model is compiled with neuronx compiler and can only be run on INF2. Replace "neuronx" with "neuron" if you are using INF1.)*

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.neuron import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Dear Evan Hansen is the winner of six Tony Awards.", return_tensors="pt")

    >>> outputs = model(**inputs)
    >>> last_hidden_state = outputs.last_hidden_state
    >>> list(last_hidden_state.shape)
    [1, 13, 384]
    ```
"""


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
        + FEATURE_EXTRACTION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="NeuronModelForFeatureExtraction",
            checkpoint="optimum/all-MiniLM-L6-v2-neuronx",
        )
    )
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
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


SENTENCE_TRANSFORMERS_EXAMPLE = r"""
    Example of TEXT Sentence Transformers:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.neuron import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("In the smouldering promise of the fall of Troy, a mythical world of gods and mortals rises from the ashes.", return_tensors="pt")

    >>> outputs = model(**inputs)
    >>> token_embeddings = outputs.token_embeddings
    >>> sentence_embedding = = outputs.sentence_embedding
    ```
"""


@add_start_docstrings(
    """
    Neuron Model for Sentence Transformers.
    """,
    NEURON_MODEL_START_DOCSTRING,
)
class NeuronModelForSentenceTransformers(NeuronTracedModel):
    """
    Sentence Transformers model on Neuron devices.
    """

    auto_model_class = AutoModel
    library_name = "sentence_transformers"

    @add_start_docstrings_to_model_forward(
        NEURON_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + SENTENCE_TRANSFORMERS_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="NeuronModelForSentenceTransformers",
            checkpoint="optimum/bge-base-en-v1.5-neuronx",
        )
    )
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        model_type = self.config.neuron["model_type"]
        neuron_inputs = {"input_ids": input_ids}
        if pixel_values is not None:
            neuron_inputs["pixel_values"] = pixel_values
        neuron_inputs["attention_mask"] = (
            attention_mask  # The input order for clip is: input_ids, pixel_values, attention_mask.
        )

        with self.neuron_padding_manager(neuron_inputs) as inputs:
            outputs = self.model(*inputs)
            if "clip" in model_type:
                text_embeds = self.remove_padding([outputs[0]], dims=[0], indices=[input_ids.shape[0]])[
                    0
                ]  # Remove padding on batch_size(0)
                image_embeds = self.remove_padding([outputs[1]], dims=[0], indices=[pixel_values.shape[0]])[
                    0
                ]  # Remove padding on batch_size(0)
                return ModelOutput(text_embeds=text_embeds, image_embeds=image_embeds)
            else:
                # token_embeddings -> (batch_size, sequencen_len, hidden_size)
                token_embeddings = self.remove_padding(
                    [outputs[0]], dims=[0, 1], indices=[input_ids.shape[0], input_ids.shape[1]]
                )[0]  # Remove padding on batch_size(0), and sequence_length(1)
                # sentence_embedding -> (batch_size, hidden_size)
                sentence_embedding = self.remove_padding([outputs[1]], dims=[0], indices=[input_ids.shape[0]])[
                    0
                ]  # Remove padding on batch_size(0)

                return ModelOutput(token_embeddings=token_embeddings, sentence_embedding=sentence_embedding)


MASKED_LM_EXAMPLE = r"""
    Example of fill mask:
    *(Following model is compiled with neuronx compiler and can only be run on INF2. Replace "neuronx" with "neuron" if you are using INF1.)*

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.neuron import {model_class}
    >>> import torch

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("This [MASK] Agreement is between General Motors and John Murray.", return_tensors="pt")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    >>> list(logits.shape)
    [1, 13, 30522]
    ```
"""


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
        + MASKED_LM_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="NeuronModelForMaskedLM",
            checkpoint="optimum/legal-bert-base-uncased-neuronx",
        )
    )
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
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


QUESTION_ANSWERING_EXAMPLE = r"""
    Example of question answering:
    *(Following model is compiled with neuronx compiler and can only be run on INF2.)*

    ```python
    >>> import torch
    >>> from transformers import {processor_class}
    >>> from optimum.neuron import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> question, text = "Are there wheelchair spaces in the theatres?", "Yes, we have reserved wheelchair spaces with a good view."
    >>> inputs = tokenizer(question, text, return_tensors="pt")
    >>> start_positions = torch.tensor([1])
    >>> end_positions = torch.tensor([12])

    >>> outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
    >>> start_scores = outputs.start_logits
    >>> end_scores = outputs.end_logits
    ```
"""


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
        + QUESTION_ANSWERING_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="NeuronModelForQuestionAnswering",
            checkpoint="optimum/roberta-base-squad2-neuronx",
        )
    )
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
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


SEQUENCE_CLASSIFICATION_EXAMPLE = r"""
    Example of single-label classification:
    *(Following model is compiled with neuronx compiler and can only be run on INF2.)*

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.neuron import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hamilton is considered to be the best musical of human history.", return_tensors="pt")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    >>> list(logits.shape)
    [1, 2]
    ```
"""


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
        + SEQUENCE_CLASSIFICATION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="NeuronModelForSequenceClassification",
            checkpoint="optimum/distilbert-base-uncased-finetuned-sst-2-english-neuronx",
        )
    )
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
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


TOKEN_CLASSIFICATION_EXAMPLE = r"""
    Example of token classification:
    *(Following model is compiled with neuronx compiler and can only be run on INF2.)*

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.neuron import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Lin-Manuel Miranda is an American songwriter, actor, singer, filmmaker, and playwright.", return_tensors="pt")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    >>> list(logits.shape)
    [1, 20, 9]
    ```
"""


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
        + TOKEN_CLASSIFICATION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="NeuronModelForTokenClassification",
            checkpoint="optimum/bert-base-NER-neuronx",
        )
    )
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
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


MULTIPLE_CHOICE_EXAMPLE = r"""
    Example of mutliple choice:
    *(Following model is compiled with neuronx compiler and can only be run on INF2.)*

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.neuron import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> num_choices = 4
    >>> first_sentence = ["Members of the procession walk down the street holding small horn brass instruments."] * num_choices
    >>> second_sentence = [
    ...     "A drum line passes by walking down the street playing their instruments.",
    ...     "A drum line has heard approaching them.",
    ...     "A drum line arrives and they're outside dancing and asleep.",
    ...     "A drum line turns the lead singer watches the performance."
    ... ]
    >>> inputs = tokenizer(first_sentence, second_sentence, truncation=True, padding=True)

    # Unflatten the inputs values expanding it to the shape [batch_size, num_choices, seq_length]
    >>> for k, v in inputs.items():
    ...     inputs[k] = [v[i: i + num_choices] for i in range(0, len(v), num_choices)]
    >>> inputs = dict(inputs.convert_to_tensors(tensor_type="pt"))
    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    >>> logits.shape
    [1, 4]
    ```
"""


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
        + MULTIPLE_CHOICE_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="NeuronModelForMultipleChoice",
            checkpoint="optimum/bert-base-uncased_SWAG-neuronx",
        )
    )
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
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


IMAGE_CLASSIFICATION_EXAMPLE = r"""
    Example of image classification:

    ```python
    >>> import requests
    >>> from PIL import Image
    >>> from optimum.neuron import {model_class}
    >>> from transformers import {processor_class}

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> preprocessor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = preprocessor(images=image, return_tensors="pt")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    >>> predicted_label = logits.argmax(-1).item()
    ```
    Example using `optimum.neuron.pipeline`:

    ```python
    >>> import requests
    >>> from PIL import Image
    >>> from transformers import {processor_class}
    >>> from optimum.neuron import {model_class}, pipeline

    >>> preprocessor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> pipe = pipeline("image-classification", model=model, feature_extractor=preprocessor)

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> pred = pipe(url)
    ```
"""


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
    def dtype(self) -> Optional["torch.dtype"]:
        """
        Torch dtype of the inputs to avoid error in transformers on casting a BatchFeature to type None.
        """
        return getattr(self.config.neuron, "input_dtype", torch.float32)

    @add_start_docstrings_to_model_forward(
        NEURON_IMAGE_INPUTS_DOCSTRING.format("batch_size, num_channels, height, width")
        + IMAGE_CLASSIFICATION_EXAMPLE.format(
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


SEMANTIC_SEGMENTATION_EXAMPLE = r"""
    Example of semantic segmentation:

    ```python
    >>> import requests
    >>> from PIL import Image
    >>> from optimum.neuron import {model_class}
    >>> from transformers import {processor_class}

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> preprocessor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = preprocessor(images=image, return_tensors="pt")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    ```

    Example using `optimum.neuron.pipeline`:

    ```python
    >>> import requests
    >>> from PIL import Image
    >>> from transformers import {processor_class}
    >>> from optimum.neuron import {model_class}, pipeline

    >>> preprocessor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> pipe = pipeline("image-segmentation", model=model, feature_extractor=preprocessor)

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> pred = pipe(url)
    ```
"""


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
    def dtype(self) -> Optional["torch.dtype"]:
        """
        Torch dtype of the inputs to avoid error in transformers on casting a BatchFeature to type None.
        """
        return getattr(self.config.neuron, "input_dtype", torch.float32)

    @add_start_docstrings_to_model_forward(
        NEURON_IMAGE_INPUTS_DOCSTRING.format("batch_size, num_channels, height, width")
        + SEMANTIC_SEGMENTATION_EXAMPLE.format(
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


OBJECT_DETECTION_EXAMPLE = r"""
    Example of object detection:

    ```python
    >>> import requests
    >>> from PIL import Image
    >>> from optimum.neuron import {model_class}
    >>> from transformers import {processor_class}

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)

    >>> preprocessor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True, batch_size=1)

    >>> inputs = preprocessor(images=image, return_tensors="pt")

    >>> outputs = model(**inputs)
    >>> target_sizes = torch.tensor([image.size[::-1]])
    >>> results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
    ```

    Example using `optimum.neuron.pipeline`:

    ```python
    >>> import requests
    >>> from PIL import Image
    >>> from transformers import {processor_class}
    >>> from optimum.neuron import {model_class}, pipeline

    >>> preprocessor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> pipe = pipeline("object-detection", model=model, feature_extractor=preprocessor)

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> pred = pipe(url)
    ```
"""


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
    def dtype(self) -> Optional["torch.dtype"]:
        """
        Torch dtype of the inputs to avoid error in transformers on casting a BatchFeature to type None.
        """
        return getattr(self.config.neuron, "input_dtype", torch.float32)

    @add_start_docstrings_to_model_forward(
        NEURON_IMAGE_INPUTS_DOCSTRING.format("batch_size, num_channels, height, width")
        + OBJECT_DETECTION_EXAMPLE.format(
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


AUDIO_CLASSIFICATION_EXAMPLE = r"""
    Example of audio classification:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.neuron import {model_class}
    >>> from datasets import load_dataset
    >>> import torch

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> feature_extractor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> # audio file is decoded on the fly
    >>> inputs = feature_extractor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")

    >>> logits = model(**inputs).logits
    >>> predicted_class_ids = torch.argmax(logits, dim=-1).item()
    >>> predicted_label = model.config.id2label[predicted_class_ids]
    ```
    Example using `optimum.neuron.pipeline`:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.neuron import {model_class}, pipeline

    >>> feature_extractor = {processor_class}.from_pretrained("{checkpoint}")
    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")

    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> ac = pipeline("audio-classification", model=model, feature_extractor=feature_extractor)

    >>> pred = ac(dataset[0]["audio"]["array"])
    ```
"""


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
        + AUDIO_CLASSIFICATION_EXAMPLE.format(
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


AUDIO_FRAME_CLASSIFICATION_EXAMPLE = r"""
    Example of audio frame classification:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.neuron import {model_class}
    >>> from datasets import load_dataset
    >>> import torch

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> feature_extractor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model =  {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = feature_extractor(dataset[0]["audio"]["array"], return_tensors="pt", sampling_rate=sampling_rate)
    >>> logits = model(**inputs).logits

    >>> probabilities = torch.sigmoid(logits[0])
    >>> labels = (probabilities > 0.5).long()
    >>> labels[0].tolist()
    ```
"""


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
        + AUDIO_FRAME_CLASSIFICATION_EXAMPLE.format(
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


CTC_EXAMPLE = r"""
    Example of CTC:

    ```python
    >>> from transformers import {processor_class}, Wav2Vec2ForCTC
    >>> from optimum.neuron import {model_class}
    >>> from datasets import load_dataset
    >>> import torch

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> processor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> # audio file is decoded on the fly
    >>> inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
    >>> logits = model(**inputs).logits
    >>> predicted_ids = torch.argmax(logits, dim=-1)

    >>> transcription = processor.batch_decode(predicted_ids)
    ```
    Example using `optimum.neuron.pipeline`:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.neuron import {model_class}, pipeline

    >>> processor = {processor_class}.from_pretrained("{checkpoint}")
    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")

    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> asr = pipeline("automatic-speech-recognition", model=model, feature_extractor=processor.feature_extractor, tokenizer=processor.tokenizer)
    ```
"""


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
        + CTC_EXAMPLE.format(
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


AUDIO_XVECTOR_EXAMPLE = r"""
    Example of Audio XVector:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.neuron import {model_class}
    >>> from datasets import load_dataset
    >>> import torch

    >>> dataset = load_dataset("hf-internal-testing/librispeech_asr_demo", "clean", split="validation")
    >>> dataset = dataset.sort("id")
    >>> sampling_rate = dataset.features["audio"].sampling_rate

    >>> feature_extractor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = feature_extractor(
    ...     [d["array"] for d in dataset[:2]["audio"]], sampling_rate=sampling_rate, return_tensors="pt", padding=True
    ... )
    >>> embeddings = model(**inputs).embeddings

    >>> embeddings = torch.nn.functional.normalize(embeddings, dim=-1)

    >>> cosine_sim = torch.nn.CosineSimilarity(dim=-1)
    >>> similarity = cosine_sim(embeddings[0], embeddings[1])
    >>> threshold = 0.7
    >>> if similarity < threshold:
    ...     print("Speakers are not the same!")
    >>> round(similarity.item(), 2)
    ```
"""


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
        + AUDIO_XVECTOR_EXAMPLE.format(
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


NEURON_CAUSALLM_MODEL_START_DOCSTRING = r"""
    This model inherits from [`~neuron.modeling.NeuronDecoderModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving)

    Args:
        model (`torch.nn.Module`): [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) is the neuron decoder graph.
        config (`transformers.PretrainedConfig`): [PretrainedConfig](https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig) is the Model configuration class with all the parameters of the model.
        model_path (`Path`): The directory where the compiled artifacts for the model are stored.
            It can be a temporary directory if the model has never been saved locally before.
        generation_config (`transformers.GenerationConfig`): [GenerationConfig](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig) holds the configuration for the model generation task.
"""

NEURON_CAUSALLM_MODEL_FORWARD_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, sequence_length)`.
        cache_ids (`torch.LongTensor`): The indices at which the cached key and value for the current inputs need to be stored.
        start_ids (`torch.LongTensor`): The indices of the first tokens to be processed, deduced form the attention masks.
"""

TEXT_GENERATION_EXAMPLE = r"""
    Example of text generation:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.neuron import {model_class}
    >>> import torch

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True)

    >>> inputs = tokenizer("My favorite moment of the day is", return_tensors="pt")

    >>> gen_tokens = model.generate(**inputs, do_sample=True, temperature=0.9, min_length=20, max_length=20)
    >>> tokenizer.batch_decode(gen_tokens)  # doctest: +IGNORE_RESULT
    ```
"""


@add_start_docstrings(
    r"""
    Neuron model with a causal language modeling head for inference on Neuron devices.
    """,
    NEURON_CAUSALLM_MODEL_START_DOCSTRING,
)
class NeuronModelForCausalLM(NeuronDecoderModel, GenerationMixin):
    auto_model_class = AutoModelForCausalLM
    main_input_name = "input_ids"

    def __init__(
        self,
        config: "PretrainedConfig",
        checkpoint_dir: Union[str, "Path", "TemporaryDirectory"],
        compiled_dir: Optional[Union[str, "Path", "TemporaryDirectory"]] = None,
        generation_config: Optional["GenerationConfig"] = None,
    ):
        super().__init__(config, checkpoint_dir, compiled_dir=compiled_dir, generation_config=generation_config)
        self.batch_size = self.config.neuron["batch_size"]
        self.max_length = self.config.neuron["sequence_length"]
        self.continuous_batching = self.model.neuron_config and self.model.neuron_config.continuous_batching
        # The generate method from GenerationMixin expects the device attribute to be set
        self.device = torch.device("cpu")

    def reset_generation(self):
        pass

    @add_start_docstrings_to_model_forward(
        NEURON_CAUSALLM_MODEL_FORWARD_DOCSTRING
        + TEXT_GENERATION_EXAMPLE.format(
            processor_class="AutoTokenizer",
            model_class="NeuronModelForCausalLM",
            checkpoint="gpt2",
        )
    )
    def forward(
        self,
        input_ids: torch.Tensor,
        cache_ids: torch.Tensor,
        start_ids: torch.Tensor = None,
        return_dict: bool = True,
    ):
        # Evaluate the output logits, storing the current key and values at the indices specified by cache_ids
        out_logits = self.model.forward(input_ids, cache_ids, start_ids)
        out_logits = out_logits[:, None, :]
        # Since we are using a static cache, we don't need to return past keys and values
        if return_dict:
            return ModelOutput([("logits", out_logits)])
        return (out_logits,)

    def get_start_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_ids: Optional[torch.Tensor] = None,
    ):
        # The start_ids parameter has different meanings:
        # - for continuous (unpadded) batching it corresponds to the sequence id,
        # - for static batching it corresponds to the start of the padded sequence.
        if self.continuous_batching:
            if seq_ids is None:
                seq_ids = torch.arange(input_ids.shape[0])
            else:
                assert seq_ids.shape[0] == input_ids.shape[0]
            return seq_ids
        start_ids = None
        if attention_mask is not None:
            _, start_ids = attention_mask.max(axis=1)
        return start_ids

    def get_cache_ids(self, attention_mask: torch.tensor, prefill: bool):
        cache_n, cache_len = attention_mask.shape
        if self.continuous_batching:
            # Evaluate the inputs that are not masked for each sequence
            input_length = attention_mask.sum(axis=1)
            if not prefill:
                # When decoding, cache_ids contains a single value per sequence
                return (input_length - 1).unsqueeze(1)
            # When prefilling, cache_ids is an increasing range
            cache_ids = torch.zeros_like(attention_mask)
            for i in range(cache_n):
                cur_length = input_length[i]
                cache_ids[i, :cur_length] = torch.arange(cur_length)
            return cache_ids
        # Static batching
        return None if prefill else torch.tensor([cache_len - 1], dtype=torch.int32)

    def prepare_inputs_for_prefill(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, seq_ids: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        start_ids = self.get_start_ids(input_ids, attention_mask, seq_ids=seq_ids)
        cache_ids = self.get_cache_ids(attention_mask, prefill=True)
        if self.continuous_batching and torch.any(attention_mask[:, 0] == 0):
            # Inputs are left padded: we need to invert padding as continuous batching requires right-padding
            batch_size, seq_len = input_ids.shape
            input_length = attention_mask.sum(axis=1)
            new_input_ids = torch.zeros_like(input_ids)
            for i in range(batch_size):
                cur_length = input_length[i]
                new_input_ids[i, :cur_length] = input_ids[i, seq_len - cur_length :]
            input_ids = new_input_ids
        return {
            "input_ids": input_ids,
            "cache_ids": cache_ids,
            "start_ids": start_ids,
        }

    def prepare_inputs_for_decode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        seq_ids: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        start_ids = self.get_start_ids(input_ids, attention_mask, seq_ids=seq_ids)
        cache_ids = self.get_cache_ids(attention_mask, prefill=False)
        # Only pass the last tokens of each sample
        input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "cache_ids": cache_ids,
            "start_ids": start_ids,
        }

    def can_generate(self) -> bool:
        """Returns True to validate the check made in `GenerationMixin.generate()`."""
        return True

    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        generation_config: Optional["GenerationConfig"] = None,
        stopping_criteria: Optional["StoppingCriteriaList"] = None,
        **kwargs,
    ) -> torch.LongTensor:
        r"""
        A streamlined generate() method overriding the transformers.GenerationMixin.generate() method.

        This method uses the same logits processors/warpers and stopping criterias as the transformers library
        `generate()` method but restricts the generation to greedy search and sampling.

        It does not support transformers `generate()` advanced options.

        Please refer to https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationMixin.generate
        for details on generation configuration.

        Parameters:
            input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.
            generation_config (`~transformers.generation.GenerationConfig`, *optional*):
                The generation configuration to be used as base parametrization for the generation call. `**kwargs`
                passed to generate matching the attributes of `generation_config` will override them. If
                `generation_config` is not provided, default will be used, which had the following loading
                priority: 1) from the `generation_config.json` model file, if it exists; 2) from the model
                configuration. Please note that unspecified parameters will inherit [`~transformers.generation.GenerationConfig`]'s
                default values, whose documentation should be checked to parameterize generation.
            stopping_criteria (`Optional[transformers.generation.StoppingCriteriaList], defaults to `None`):
                Custom stopping criteria that complement the default stopping criteria built from arguments and a
                generation config.

        Returns:
            `torch.Tensor`: A  `torch.FloatTensor`.
        """
        # The actual generation configuration is a combination of config and parameters
        generation_config = copy.deepcopy(self.generation_config if generation_config is None else generation_config)
        # Extract tokenizer if any (used only for stop strings)
        tokenizer = kwargs.pop("tokenizer", None)
        model_kwargs = generation_config.update(**kwargs)  # All unused kwargs must be model kwargs
        # Check model kwargs are actually used by either prepare_inputs_for_generation or forward
        self._validate_model_kwargs(model_kwargs)

        # Instantiate a TokenSelector for the specified configuration
        selector = TokenSelector.create(
            input_ids,
            generation_config,
            self,
            self.max_length,
            stopping_criteria=stopping_criteria,
            tokenizer=tokenizer,
        )

        # Verify that the inputs are compatible with the model static input dimensions
        batch_size, sequence_length = input_ids.shape
        if sequence_length > self.max_length:
            raise ValueError(
                f"The input sequence length ({sequence_length}) exceeds the model static sequence length ({self.max_length})"
            )
        padded_input_ids = input_ids
        padded_attention_mask = torch.ones_like(input_ids) if attention_mask is None else attention_mask
        if batch_size > self.batch_size:
            raise ValueError(
                f"The specified batch_size ({batch_size}) exceeds the model static batch size ({self.batch_size})"
            )
        elif batch_size < self.batch_size and not self.continuous_batching:
            logger.warning("Inputs will be padded to match the model static batch size. This will increase latency.")
            padding_shape = [self.batch_size - batch_size, sequence_length]
            pad_token_id = generation_config.pad_token_id
            if pad_token_id is None:
                if isinstance(self.config.eos_token_id, list):
                    pad_token_id = self.config.eos_token_id[0]
                else:
                    pad_token_id = self.config.eos_token_id
            padding = torch.full(padding_shape, fill_value=pad_token_id, dtype=torch.int64)
            padded_input_ids = torch.cat([padded_input_ids, padding])
            padding = torch.zeros(padding_shape, dtype=torch.int64)
            padded_attention_mask = torch.cat([padded_attention_mask, padding])

        output_ids = self.generate_tokens(
            padded_input_ids,
            selector,
            batch_size,
            padded_attention_mask,
            **model_kwargs,
        )
        return output_ids[:batch_size, :]

    def generate_tokens(
        self,
        input_ids: torch.LongTensor,
        selector: TokenSelector,
        batch_size: int,
        attention_mask: torch.Tensor,
        **model_kwargs,
    ) -> torch.LongTensor:
        r"""
        Generate tokens using sampling or greedy search.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            selector (`TokenSelector`):
                The object implementing the generation logic based on transformers processors and stopping criterias.
            batch_size (`int`):
                The actual input batch size. Used to avoid generating tokens for padded inputs.
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model.

        Return:
            `torch.LongTensor`: A `torch.LongTensor` containing the generated tokens.

        """
        # keep track of which sequences are already finished
        unfinished_sequences = torch.zeros(input_ids.shape[0], dtype=torch.long, device=input_ids.device)
        unfinished_sequences[:batch_size] = 1

        # Prefill and obtain the first token
        model_inputs = self.prepare_inputs_for_prefill(input_ids, attention_mask)
        outputs = self(
            **model_inputs,
            return_dict=True,
        )

        # auto-regressive generation
        while True:
            next_token_logits = outputs.logits[:, -1, :]

            next_tokens = selector.select(input_ids, next_token_logits)

            # finished sentences should have their next token be a padding token
            next_tokens = next_tokens * unfinished_sequences + selector.pad_token_id * (1 - unfinished_sequences)

            # update inputs for the next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            attention_mask = torch.cat([attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1)

            unfinished_sequences = unfinished_sequences & ~selector.stopping_criteria(input_ids, None)

            if unfinished_sequences.max() == 0:
                break

            # forward pass to get next token
            model_inputs = self.prepare_inputs_for_decode(input_ids, attention_mask)
            outputs = self(
                **model_inputs,
                return_dict=True,
            )

        return input_ids
