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
from typing import Optional

import torch
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

from .modeling_base import NeuronBaseModel


logger = logging.getLogger(__name__)


_TOKENIZER_FOR_DOC = "AutoTokenizer"

NEURON_MODEL_START_DOCSTRING = r"""
    This model inherits from [`~neuron.modeling.NeuronBaseModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving)

    Args:
        config (`transformers.PretrainedConfig`): [PretrainedConfig](https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig) is the Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`optimum.neuron.modeling.NeuronBaseModel.from_pretrained`] method to load the model weights.
        model (`torch.jit._script.ScriptModule`): [torch.jit._script.ScriptModule](https://pytorch.org/docs/stable/generated/torch.jit.ScriptModule.html) is the TorchScript graph compiled by neuron(x) compiler.
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
class NeuronModelForFeatureExtraction(NeuronBaseModel):
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
            outputs = self.model(
                *inputs
            )  # last_hidden_state: (batch_size, sequencen_len, hidden_size), pooler_output: (batch_size, hidden_size)
            last_hidden_state = self.remove_padding(
                [outputs[0]], dims=[0, 1], indices=[input_ids.shape[0], input_ids.shape[1]]
            )[
                0
            ]  # Remove padding on batch_size(0), and sequence_length(1)
            if len(outputs) > 1:
                pooler_output = self.remove_padding([outputs[1]], dims=[0], indices=[input_ids.shape[0]])[
                    0
                ]  # Remove padding on batch_size(0)
            else:
                pooler_output = None

        return BaseModelOutputWithPooling(last_hidden_state=last_hidden_state, pooler_output=pooler_output)


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
class NeuronModelForMaskedLM(NeuronBaseModel):
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
class NeuronModelForQuestionAnswering(NeuronBaseModel):
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
class NeuronModelForSequenceClassification(NeuronBaseModel):
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
class NeuronModelForTokenClassification(NeuronBaseModel):
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
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True)

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
class NeuronModelForMultipleChoice(NeuronBaseModel):
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
