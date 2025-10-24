# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
"""NeuronModelForSentenceTransformers classe for inference on neuron devices of sentence transformers."""

import logging
from typing import Literal

import numpy as np
import torch
from sentence_transformers.similarity_functions import SimilarityFunction
from tqdm.autonotebook import trange
from transformers import AutoModel
from transformers.modeling_outputs import ModelOutput

from .modeling_traced import NeuronTracedModel
from .utils.doc import (
    _GENERIC_PROCESSOR,
    _TOKENIZER_FOR_DOC,
    NEURON_MODEL_START_DOCSTRING,
    NEURON_SENTENCE_TRANSFORMERS_IMAGE_EXAMPLE,
    NEURON_SENTENCE_TRANSFORMERS_TEXT_EXAMPLE,
    NEURON_TEXT_INPUTS_DOCSTRING,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
)


logger = logging.getLogger(__name__)


@add_start_docstrings(
    """
    Neuron Model for Sentence Transformers.
    """,
    NEURON_MODEL_START_DOCSTRING,
)
class NeuronSentenceTransformers(NeuronTracedModel):
    """
    Sentence Transformers model on Neuron devices.
    """

    auto_model_class = AutoModel
    library_name = "sentence_transformers"

    def __init__(
        self,
        **kwargs,
    ):
        self.prompts = {"query": "", "document": ""}
        prompts = kwargs.pop("prompts", None)
        if prompts:
            self.prompts.update(prompts)
        self.default_prompt_name = kwargs.pop("default_prompt_nam", None)
        self._prompt_length_mapping = {}
        self.truncate_dim = kwargs.pop("truncate_dim", None)
        self.similarity_fn_name = kwargs.pop("similarity_fn_name", None)

        super().__init__(**kwargs)

    @add_start_docstrings_to_model_forward(
        NEURON_TEXT_INPUTS_DOCSTRING.format("batch_size, sequence_length")
        + NEURON_SENTENCE_TRANSFORMERS_TEXT_EXAMPLE.format(
            processor_class=_TOKENIZER_FOR_DOC,
            model_class="NeuronModelForSentenceTransformers",
            checkpoint="optimum/bge-base-en-v1.5-neuronx",
        )
        + NEURON_SENTENCE_TRANSFORMERS_IMAGE_EXAMPLE.format(
            processor_class=_GENERIC_PROCESSOR,
            model_class="NeuronModelForSentenceTransformers",
            checkpoint="optimum/clip_vit_emb_neuronx",
        )
    )
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        pixel_values: torch.Tensor | None = None,
        token_type_ids: torch.Tensor | None = None,
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

    def tokenize(self, texts: list[str] | list[dict] | list[tuple[str, str]], **kwargs) -> dict[str, torch.Tensor]:
        """
        Tokenizes the texts.

        Args:
            texts (Union[List[str], List[Dict], List[Tuple[str, str]]]): A list of texts to be tokenized.

        Returns:
            Dict[str, Tensor]: A dictionary of tensors with the tokenized texts. Common keys are "input_ids",
                "attention_mask", and "token_type_ids".
        """
        return self.preprocessors[0](texts, **kwargs)

    def _get_prompt_length(self, prompt: str, **kwargs) -> int:
        """
        Return the length of the prompt in tokens, including the BOS token
        """
        if (prompt, *kwargs.values()) in self._prompt_length_mapping:
            return self._prompt_length_mapping[(prompt, *kwargs.values())]

        tokenized_prompt = self.tokenize([prompt], return_tensors="pt", **kwargs)
        if "input_ids" not in tokenized_prompt:
            # If the tokenizer does not return input_ids, we cannot determine the prompt length.
            # This can happen with some tokenizers that do not use input_ids.
            return None
        prompt_length = tokenized_prompt["input_ids"].shape[-1]
        # If the tokenizer adds a special EOS token, we do not count it as part of the prompt length.
        # This is to ensure that the prompt length does not include the EOS token.
        last_token = tokenized_prompt["input_ids"][..., -1].item()
        if hasattr(self.tokenizer, "all_special_ids") and last_token in self.tokenizer.all_special_ids:
            prompt_length -= 1
        self._prompt_length_mapping[(prompt, *kwargs.values())] = prompt_length
        return prompt_length

    def _text_length(self, text: list[int] | list[list[int]]) -> int:
        """
        Help function to get the length for the input text. Text can be either
        a list of ints (which means a single text as input), or a tuple of list of ints
        (representing several text inputs to the model).
        """

        if isinstance(text, dict):  # {key: value} case
            return len(next(iter(text.values())))
        elif not hasattr(text, "__len__"):  # Object has no len() method
            return 1
        elif len(text) == 0 or isinstance(text[0], int):  # Empty string or list of ints
            return len(text)
        else:
            return sum([len(t) for t in text])  # Sum of length of individual strings

    @property
    def similarity_fn_name(self) -> Literal["cosine", "dot", "euclidean", "manhattan"]:
        """Return the name of the similarity function used by :meth:`SentenceTransformer.similarity` and :meth:`SentenceTransformer.similarity_pairwise`.

        Returns:
            Optional[str]: The name of the similarity function. Can be None if not set, in which case it will
                default to "cosine" when first called.

        Example:
            >>> model = SentenceTransformer("multi-qa-mpnet-base-dot-v1")
            >>> model.similarity_fn_name
            'dot'
        """
        if self._similarity_fn_name is None:
            self.similarity_fn_name = SimilarityFunction.COSINE
        return self._similarity_fn_name

    @similarity_fn_name.setter
    def similarity_fn_name(
        self, value: Literal["cosine", "dot", "euclidean", "manhattan"] | SimilarityFunction
    ) -> None:
        if isinstance(value, SimilarityFunction):
            value = value.value
        self._similarity_fn_name = value

        if value is not None:
            self._similarity = SimilarityFunction.to_similarity_fn(value)
            self._similarity_pairwise = SimilarityFunction.to_similarity_pairwise_fn(value)

    def encode(
        self,
        sentences: str | list[str],
        prompt_name: str | None = None,
        prompt: str | None = None,
        show_progress_bar: bool | None = None,
        output_value: Literal["sentence_embedding", "token_embeddings"] | None = "sentence_embedding",
        normalize_embeddings: bool = False,
        **kwargs,
    ):
        if show_progress_bar is None:
            show_progress_bar = logger.getEffectiveLevel() in (logging.INFO, logging.DEBUG)

        # Cast an individual input to a list with length 1
        input_was_string = False
        if isinstance(sentences, str) or not hasattr(sentences, "__len__"):
            sentences = [sentences]
            input_was_string = True

        if prompt is None:
            if prompt_name is not None:
                try:
                    prompt = self.prompts[prompt_name]
                except KeyError:
                    raise ValueError(
                        f"Prompt name '{prompt_name}' not found in the configured prompts dictionary with keys {list(self.prompts.keys())!r}."
                    )
            elif self.default_prompt_name is not None:
                prompt = self.prompts.get(self.default_prompt_name, None)
        else:
            if prompt_name is not None:
                logger.warning(
                    "Encode with either a `prompt`, a `prompt_name`, or neither, but not both. "
                    "Ignoring the `prompt_name` in favor of `prompt`."
                )

        extra_features = {}
        if prompt is not None and len(prompt) > 0:
            sentences = [prompt + sentence for sentence in sentences]

            # Some models (e.g. INSTRUCTOR, GRIT) require removing the prompt before pooling
            # Tracking the prompt length allow us to remove the prompt during pooling
            length = self._get_prompt_length(prompt, **kwargs)
            if length is not None:
                extra_features["prompt_length"] = length

        all_embeddings = []
        length_sorted_idx = np.argsort([-self._text_length(sen) for sen in sentences])
        sentences_sorted = [sentences[int(idx)] for idx in length_sorted_idx]

        batch_size = self.neuron_config.batch_size
        for start_index in trange(0, len(sentences), batch_size, desc="Batches", disable=not show_progress_bar):
            sentences_batch = sentences_sorted[start_index : start_index + batch_size]
            features = self.tokenize(sentences_batch, return_tensors="pt", **kwargs)

            features.update(extra_features)
            out_features = self.forward(**features)

            if output_value == "token_embeddings":
                embeddings = []
                for token_emb, attention in zip(out_features[output_value], out_features["attention_mask"]):
                    last_mask_id = len(attention) - 1
                    while last_mask_id > 0 and attention[last_mask_id].item() == 0:
                        last_mask_id -= 1

                    embeddings.append(token_emb[0 : last_mask_id + 1])
            elif output_value is None:  # Return all outputs
                embeddings = []
                for idx in range(len(out_features["sentence_embedding"])):
                    batch_item = {}
                    for name, value in out_features.items():
                        try:
                            batch_item[name] = value[idx]
                        except TypeError:
                            # Handle non-indexable values (like prompt_length)
                            batch_item[name] = value
                    embeddings.append(batch_item)
            else:  # Sentence embeddings
                embeddings = out_features[output_value]
                embeddings = embeddings.detach()
                if normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

                all_embeddings.extend(embeddings)

        all_embeddings = [all_embeddings[idx] for idx in np.argsort(length_sorted_idx)]

        if len(all_embeddings):
            if isinstance(all_embeddings, np.ndarray):
                all_embeddings = torch.from_numpy(all_embeddings)
            else:
                all_embeddings = torch.stack(all_embeddings)
        else:
            all_embeddings = torch.tensor([], device=self.device)

        if input_was_string:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    @property
    def similarity(self):
        if self.similarity_fn_name is None:
            self.similarity_fn_name = SimilarityFunction.COSINE

        return self._similarity
