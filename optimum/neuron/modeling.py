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
import warnings
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
)
from transformers.file_utils import add_start_docstrings, add_start_docstrings_to_model_forward
from transformers.generation import (
    GenerationMixin,
    LogitsProcessorList,
    LogitsWarper,
    SampleDecoderOnlyOutput,
    StoppingCriteriaList,
    TopKLogitsWarper,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from transformers.utils import ModelOutput

from .modeling_base import NeuronBaseModel
from .modeling_decoder import NeuronDecoderModel


if TYPE_CHECKING:
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from transformers import BaseStreamer, GenerationConfig, PretrainedConfig


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
        model: torch.nn.Module,
        config: "PretrainedConfig",
        model_path: Union[str, "Path", "TemporaryDirectory"],
        generation_config: Optional["GenerationConfig"] = None,
    ):
        super().__init__(model, config, model_path, generation_config)
        self.cur_len = 0
        self.batch_size = model.config.batch_size
        self.max_length = model.config.n_positions
        # The generate method from GenerationMixin expects the device attribute to be set
        self.device = torch.device("cpu")

    def reset_generation(self):
        self.cur_len = 0

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
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        attention_mask: torch.Tensor = None,
        return_dict: bool = True,
    ):
        if output_hidden_states or output_attentions or attention_mask is not None:
            warnings.warn(
                "Warning: These arguments are not used by forward(): \
                (output_hidden_states, output_attentions, attention_mask)"
            )
        # Evaluate the output logits, storing the current key and values at the indices specified by cache_ids
        out_logits = self.model.forward(input_ids, cache_ids, start_ids)
        out_logits = out_logits[:, None, :]
        # Since we are using a static cache, we don't need to return past keys and values
        if return_dict:
            return ModelOutput([("logits", out_logits)])
        return (out_logits,)

    def prepare_inputs_for_generation(self, input_ids: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        # Sanity checks
        if kwargs.get("past_key_values", None) is not None:
            raise ValueError("This model does not support dynamic key, value cache.")
        batch_size, sequence_length = input_ids.shape
        if batch_size != self.batch_size:
            raise ValueError(
                f"The specified batch_size ({batch_size}) does not match the model static batch size ({self.batch_size})"
            )
        if sequence_length > self.max_length:
            raise ValueError(
                f"The current sequence length ({sequence_length}) exceeds the model static sequence length ({self.max_length})"
            )
        # convert attention_mask to start_ids
        attention_mask = None
        start_ids = None
        if "attention_mask" in kwargs:
            attention_mask = kwargs["attention_mask"]

        if attention_mask is not None:
            _, start_ids = attention_mask.max(axis=1)

        if self.cur_len > 0:
            # Only pass the last tokens of each sample
            input_ids = input_ids[:, -1:]
            # Specifiy the single index at which the new keys and values need to be stored
            cache_ids = torch.as_tensor([self.cur_len], dtype=torch.int32)
        else:
            # All tokens will be processed: initialize the cached with all intermediate keys and values
            cache_ids = torch.arange(input_ids.shape[-1], dtype=torch.int32)

        # Increment the current cache index
        self.cur_len += input_ids.shape[-1]
        model_inputs = {
            "input_ids": input_ids,
            "cache_ids": cache_ids,
            "start_ids": start_ids,
        }

        return model_inputs

    def can_generate(self) -> bool:
        """Returns True to validate the check made in `GenerationMixin.generate()`."""
        return True

    class FastTopKLogitsWarper(LogitsWarper):
        r"""Returns [batch_size, top_k] scores and indices instead of [batch_size, vocab_size] scores."""

        def __init__(self, top_k: int, filter_value: float):
            self.top_k = top_k
            self.filter_value = filter_value

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
            top_k = min(self.top_k, scores.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            return torch.topk(scores, top_k)

    # Adapted from transformers.generation.utils.GenerationMixin.sample
    def sample(
        self,
        input_ids: torch.LongTensor,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        logits_warper: Optional[LogitsProcessorList] = None,
        max_length: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[Union[int, List[int]]] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        output_scores: Optional[bool] = None,
        return_dict_in_generate: Optional[bool] = None,
        synced_gpus: bool = False,
        streamer: Optional["BaseStreamer"] = None,
        **model_kwargs,
    ) -> Union[SampleDecoderOnlyOutput, torch.LongTensor]:
        r"""
        This is a simplified version of the transformers `GenerationMixin.sample()` method that is optimized for neuron inference.

        It generates sequences of token ids for models with a language modeling head using **multinomial sampling** and
        can be used for text-decoder, text-to-text, speech-to-text, and vision-to-text models.

        Please refer to https://huggingface.co/docs/transformers/en/main_classes/text_generation#transformers.GenerationMixin.sample.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                The sequence used as a prompt for the generation.
            logits_processor (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsProcessor`]
                used to modify the prediction scores of the language modeling head applied at each generation step.
            stopping_criteria (`StoppingCriteriaList`, *optional*):
                An instance of [`StoppingCriteriaList`]. List of instances of class derived from [`StoppingCriteria`]
                used to tell if the generation loop should stop.
            logits_warper (`LogitsProcessorList`, *optional*):
                An instance of [`LogitsProcessorList`]. List of instances of class derived from [`LogitsWarper`] used
                to warp the prediction score distribution of the language modeling head applied before multinomial
                sampling at each generation step.
            pad_token_id (`int`, *optional*):
                The id of the *padding* token.
            eos_token_id (`Union[int, List[int]]`, *optional*):
                The id of the *end-of-sequence* token. Optionally, use a list to set multiple *end-of-sequence* tokens.
            output_attentions (`bool`, *optional*, defaults to `False`):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more details.
            output_hidden_states (`bool`, *optional*, defaults to `False`):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more details.
            output_scores (`bool`, *optional*, defaults to `False`):
                Whether or not to return the prediction scores. See `scores` under returned tensors for more details.
            return_dict_in_generate (`bool`, *optional*, defaults to `False`):
                Whether or not to return a [`~utils.ModelOutput`] instead of a plain tuple.
            synced_gpus (`bool`, *optional*, defaults to `False`):
                Whether to continue running the while loop until max_length (needed for ZeRO stage 3)
            streamer (`BaseStreamer`, *optional*):
                Streamer object that will be used to stream the generated sequences. Generated tokens are passed
                through `streamer.put(token_ids)` and the streamer is responsible for any further processing.
            model_kwargs:
                Additional model specific kwargs will be forwarded to the `forward` function of the model. If model is
                an encoder-decoder model the kwargs should include `encoder_outputs`.

        Return:
            [`~generation.SampleDecoderOnlyOutput`], [`~generation.SampleEncoderDecoderOutput`] or `torch.LongTensor`:
            A `torch.LongTensor` containing the generated tokens (default behaviour) or a
            [`~generation.SampleDecoderOnlyOutput`] if `model.config.is_encoder_decoder=False` and
            `return_dict_in_generate=True` or a [`~generation.SampleEncoderDecoderOutput`] if
            `model.config.is_encoder_decoder=True`.

        """
        # We don't support all parameters
        if synced_gpus:
            raise ValueError("Neuron models cannot run on GPUs.")
        # init values
        logits_processor = logits_processor if logits_processor is not None else LogitsProcessorList()
        stopping_criteria = stopping_criteria if stopping_criteria is not None else StoppingCriteriaList()
        if max_length is not None:
            warnings.warn(
                "`max_length` is deprecated in this function, use"
                " `stopping_criteria=StoppingCriteriaList(MaxLengthCriteria(max_length=max_length))` instead.",
                UserWarning,
            )
            stopping_criteria = self.validate_stopping_criteria(stopping_criteria, max_length)
        # Modifications to the original algorithm are all conditioned by the fast_topk boolean
        fast_topk = False
        if logits_warper is None:
            logits_warper = LogitsProcessorList()
        else:
            last_warper = logits_warper[-1]
            fast_topk = isinstance(last_warper, TopKLogitsWarper)
            if fast_topk:
                # Replace the last warping operation by a faster alternative
                logits_warper[-1] = self.FastTopKLogitsWarper(last_warper.top_k, last_warper.filter_value)

        pad_token_id = pad_token_id if pad_token_id is not None else self.generation_config.pad_token_id
        eos_token_id = eos_token_id if eos_token_id is not None else self.generation_config.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        output_scores = output_scores if output_scores is not None else self.generation_config.output_scores
        output_attentions = (
            output_attentions if output_attentions is not None else self.generation_config.output_attentions
        )
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.generation_config.output_hidden_states
        )
        return_dict_in_generate = (
            return_dict_in_generate
            if return_dict_in_generate is not None
            else self.generation_config.return_dict_in_generate
        )

        # init attention / hidden states / scores tuples
        scores = () if (return_dict_in_generate and output_scores) else None
        decoder_attentions = () if (return_dict_in_generate and output_attentions) else None
        cross_attentions = () if (return_dict_in_generate and output_attentions) else None
        decoder_hidden_states = () if (return_dict_in_generate and output_hidden_states) else None

        # keep track of which sequences are already finished
        unfinished_sequences = torch.ones(input_ids.shape[0], dtype=torch.long, device=input_ids.device)

        # This is specific to Neuron models
        self.reset_generation()

        # auto-regressive generation
        while True:
            # prepare model inputs
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)

            # forward pass to get next token
            outputs = self(
                **model_inputs,
                return_dict=True,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
            )

            next_token_logits = outputs.logits[:, -1, :]

            # pre-process distribution
            next_token_scores = logits_processor(input_ids, next_token_logits)
            if fast_topk:
                # Get [batch_size, top_k] scores and indices instead of [batch_size, vocab_size] scores
                next_token_scores, next_token_indices = logits_warper(input_ids, next_token_scores)
            else:
                next_token_scores = logits_warper(input_ids, next_token_scores)

            # Store scores, attentions and hidden_states when required
            if return_dict_in_generate:
                if output_scores:
                    if fast_topk:
                        # Expand the [batch_size, top_k] scores to [batch_size, vocab_size]
                        expanded_scores = torch.full(next_token_logits.shape, last_warper.filter_value)
                        expanded_scores.scatter(1, next_token_indices, next_token_scores)
                        scores += (expanded_scores,)
                    else:
                        scores += (next_token_scores,)
                if output_attentions:
                    decoder_attentions += (
                        (outputs.decoder_attentions,) if self.config.is_encoder_decoder else (outputs.attentions,)
                    )
                    if self.config.is_encoder_decoder:
                        cross_attentions += (outputs.cross_attentions,)

                if output_hidden_states:
                    decoder_hidden_states += (
                        (outputs.decoder_hidden_states,)
                        if self.config.is_encoder_decoder
                        else (outputs.hidden_states,)
                    )

            # sample
            probs = torch.nn.functional.softmax(next_token_scores, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)
            if fast_topk:
                # Convert the topk relative tokens to actual vocabulary tokens
                next_tokens = torch.gather(next_token_indices, 1, next_tokens)
            next_tokens = next_tokens.squeeze(1)

            # finished sentences should have their next token be a padding token
            if eos_token_id is not None:
                if pad_token_id is None:
                    raise ValueError("If `eos_token_id` is defined, make sure that `pad_token_id` is defined.")
                next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # update generated ids, model inputs, and length for next step
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            if streamer is not None:
                streamer.put(next_tokens.cpu())
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )

            # if eos_token was found in one sentence, set sentence to finished
            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    break

            # stop if we exceed the maximum length
            if stopping_criteria(input_ids, scores):
                break

        if streamer is not None:
            streamer.end()

        if return_dict_in_generate:
            return SampleDecoderOnlyOutput(
                sequences=input_ids,
                scores=scores,
                attentions=decoder_attentions,
                hidden_states=decoder_hidden_states,
            )
        else:
            return input_ids
