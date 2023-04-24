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
    BaseModelOutput,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

from .modeling_base import NeuronModel


logger = logging.getLogger(__name__)


_TOKENIZER_FOR_DOC = "AutoTokenizer"

NEURON_MODEL_START_DOCSTRING = r"""
    This model inherits from [`~neuron.modeling.NeuronModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving)

    Args:
        config (`transformers.PretrainedConfig`): [PretrainedConfig](https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig) is the Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`~neuron.modeling.NeuronModel.from_pretrained`] method to load the model weights.
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

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.neuron import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("My name is Philipp and I live in Germany.", return_tensors="pt")

    >>> outputs = model(**inputs)
    >>> last_hidden_state = outputs.last_hidden_state
    >>> list(last_hidden_state.shape)
    [2, 12, 384]
    ```

    Example using `transformers.pipeline`:

    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.neuron import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> neuron_extractor = pipeline("feature-extraction", model=model, tokenizer=tokenizer)

    >>> text = "My name is Philipp and I live in Germany."
    >>> pred = neuron_extractor(text)
    ```
"""


@add_start_docstrings(
    """
    Neuron Model with a BaseModelOutput for feature-extraction tasks.
    """,
    NEURON_MODEL_START_DOCSTRING,
)
class NeuronModelForFeatureExtraction(NeuronModel):
    """
    Feature Extraction model on Neuron devices.
    """

    auto_model_class = AutoModel

    @add_start_docstrings_to_model_forward(
        NEURON_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + FEATURE_EXTRACTION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="NeuronModelForFeatureExtraction",
            checkpoint="optimum/all-MiniLM-L6-v2",
        )
    )
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        neuron_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            neuron_inputs["token_type_ids"] = token_type_ids

        outputs = self.model(*tuple(neuron_inputs.values()))

        last_hidden_state = outputs[0]
        pooler_output = outputs[1] if len(outputs) > 1 else None

        # converts output to namedtuple for pipelines post-processing
        return BaseModelOutput(last_hidden_state=last_hidden_state, pooler_output=pooler_output)


MASKED_LM_EXAMPLE = r"""
    Example of feature extraction:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.neuron import {model_class}
    >>> import torch

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("The capital of France is [MASK].", return_tensors="pt")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    >>> list(logits.shape)
    [2, 8, 28996]
    ```

    Example using `transformers.pipeline`:

    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.neuron import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> fill_masker = pipeline("fill-mask", model=model, tokenizer=tokenizer)

    >>> text = "The capital of France is [MASK]."
    >>> pred = fill_masker(text)
    ```
"""


@add_start_docstrings(
    """
    Neuron Model with a MaskedLMOutput for masked language modeling tasks.
    """,
    NEURON_MODEL_START_DOCSTRING,
)
class NeuronModelForMaskedLM(NeuronModel):
    """
    Masked language model for on Neuron devices.
    """

    auto_model_class = AutoModelForMaskedLM

    @add_start_docstrings_to_model_forward(
        NEURON_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + MASKED_LM_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="NeuronModelForMaskedLM",
            checkpoint="optimum/bert-base-uncased-for-fill-mask",
        )
    )
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        neuron_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            neuron_inputs["token_type_ids"] = token_type_ids

        outputs = self.model(*tuple(neuron_inputs.values()))

        logits = outputs[0]

        # converts output to namedtuple for pipelines post-processing
        return MaskedLMOutput(logits=logits)


QUESTION_ANSWERING_EXAMPLE = r"""
    Example of question answering:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.neuron import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    >>> inputs = tokenizer(question, text, return_tensors="pt")
    >>> start_positions = torch.tensor([1])
    >>> end_positions = torch.tensor([3])

    >>> outputs = model(**inputs, start_positions=start_positions, end_positions=end_positions)
    >>> start_scores = outputs.start_logits
    >>> end_scores = outputs.end_logits
    ```
    Example using `transformers.pipeline`:

    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.neuron import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> neuron_qa = pipeline("question-answering", model=model, tokenizer=tokenizer)

    >>> question, text = "Who was Jim Henson?", "Jim Henson was a nice puppet"
    >>> pred = neuron_qa(question, text)
    ```
"""


@add_start_docstrings(
    """
    Neuron Model with a QuestionAnsweringModelOutput for extractive question-answering tasks like SQuAD.
    """,
    NEURON_MODEL_START_DOCSTRING,
)
class NeuronModelForQuestionAnswering(NeuronModel):
    """
    Question Answering model on Neuron devices.
    """

    auto_model_class = AutoModelForQuestionAnswering

    @add_start_docstrings_to_model_forward(
        NEURON_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + QUESTION_ANSWERING_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="NeuronModelForQuestionAnswering",
            checkpoint="optimum/roberta-base-squad2",
        )
    )
    def forward(
        self,
        input_ids: torch.Tensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        neuron_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            neuron_inputs["token_type_ids"] = token_type_ids

        outputs = self.model(*tuple(neuron_inputs.values()))
        start_logits = outputs[0]
        end_logits = outputs[1]

        # converts output to namedtuple for pipelines post-processing
        return QuestionAnsweringModelOutput(start_logits=start_logits, end_logits=end_logits)


SEQUENCE_CLASSIFICATION_EXAMPLE = r"""
    Example of single-label classification:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.neuron import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("Hello, my cats are much cuter than your dogs", return_tensors="pt")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    >>> list(logits.shape)
    [2, 2]
    ```

    Example using `transformers.pipelines`:

    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.neuron import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> neuron_classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

    >>> text = "Hello, my cats are much cuter than your dogs"
    >>> pred = neuron_classifier(text)
    ```

    Example using zero-shot-classification `transformers.pipelines`:

    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.neuron import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("optimum/distilbert-base-uncased-mnli")
    >>> model = {model_class}.from_pretrained("optimum/distilbert-base-uncased-mnli")
    >>> neuron_z0 = pipeline("zero-shot-classification", model=model, tokenizer=tokenizer)

    >>> sequence_to_classify = "Who are you voting for in 2023?"
    >>> candidate_labels = ["Europe", "public health", "politics", "elections"]
    >>> pred = neuron_z0(sequence_to_classify, candidate_labels, multi_label=True)
    ```
"""


@add_start_docstrings(
    """
    Neuron Model with a sequence classification/regression head on top (a linear layer on top of the
    pooled output) e.g. for GLUE tasks.
    """,
    NEURON_MODEL_START_DOCSTRING,
)
class NeuronModelForSequenceClassification(NeuronModel):
    """
    Sequence Classification model on Neuron devices.
    """

    auto_model_class = AutoModelForSequenceClassification

    @add_start_docstrings_to_model_forward(
        NEURON_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + SEQUENCE_CLASSIFICATION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="NeuronModelForSequenceClassification",
            checkpoint="optimum/distilbert-base-uncased-finetuned-sst-2-english",
        )
    )
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        neuron_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            neuron_inputs["token_type_ids"] = token_type_ids

        outputs = self.model(*tuple(neuron_inputs.values()))

        logits = outputs[0]

        # converts output to namedtuple for pipelines post-processing
        return SequenceClassifierOutput(logits=logits)


TOKEN_CLASSIFICATION_EXAMPLE = r"""
    Example of token classification:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.neuron import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("My name is Philipp and I live in Germany.", return_tensors="pt")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    >>> list(logits.shape)
    [2, 12, 9]
    ```

    Example using `transformers.pipelines`:

    ```python
    >>> from transformers import {processor_class}, pipeline
    >>> from optimum.neuron import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> neuron_ner = pipeline("token-classification", model=model, tokenizer=tokenizer)

    >>> text = "My name is Philipp and I live in Germany."
    >>> pred = neuron_ner(text)
    ```
"""


@add_start_docstrings(
    """
    Neuron Model with a token classification head on top (a linear layer on top of the hidden-states output) e.g.
    for Named-Entity-Recognition (NER) tasks.
    """,
    NEURON_MODEL_START_DOCSTRING,
)
class NeuronModelForTokenClassification(NeuronModel):
    """
    Token Classification model on Neuron devices.
    """

    auto_model_class = AutoModelForTokenClassification

    @add_start_docstrings_to_model_forward(
        NEURON_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + TOKEN_CLASSIFICATION_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="NeuronModelForTokenClassification",
            checkpoint="optimum/bert-base-NER",
        )
    )
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
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
        outputs = self.model(*tuple(neuron_inputs.values()))
        logits = outputs[0]

        # converts output to namedtuple for pipelines post-processing
        return TokenClassifierOutput(logits=logits)


MULTIPLE_CHOICE_EXAMPLE = r"""
    Example of mutliple choice:

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
    ```
"""


@add_start_docstrings(
    """
    Neuron Model with a multiple choice classification head on top (a linear layer on top of the pooled output and a
    softmax) e.g. for RocStories/SWAG tasks.
    """,
    NEURON_MODEL_START_DOCSTRING,
)
class NeuronModelForMultipleChoice(NeuronModel):
    """
    Multiple choice model on Neuron devices.
    """

    auto_model_class = AutoModelForMultipleChoice

    @add_start_docstrings_to_model_forward(
        NEURON_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + MULTIPLE_CHOICE_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="NeuronModelForMultipleChoice",
            checkpoint="ehdwns1516/bert-base-uncased_SWAG",
        )
    )
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
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
        outputs = self.model(*tuple(neuron_inputs.values()))
        logits = outputs[0]

        # converts output to namedtuple for pipelines post-processing
        return MultipleChoiceModelOutput(logits=logits)
