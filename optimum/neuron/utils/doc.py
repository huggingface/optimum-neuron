# coding=utf-8
# Copyright 2025 The HuggingFace Team. All rights reserved.
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

import inspect
import textwrap


_TOKENIZER_FOR_DOC = "AutoTokenizer"
_PROCESSOR_FOR_IMAGE = "AutoImageProcessor"
_GENERIC_PROCESSOR = "AutoProcessor"

# ==============================================================================
# MODEL CODE EXAMPLES
# ==============================================================================

NEURON_FEATURE_EXTRACTION_EXAMPLE = r"""
    Example:

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

NEURON_MULTIMODAL_FEATURE_EXTRACTION_EXAMPLE = r"""
    Example:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.neuron import {model_class}

    >>> processor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
    >>> image = Image.open(requests.get(url, stream=True).raw)
    >>> inputs = processor(text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True)

    >>> outputs = model(**inputs)
    >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
    >>> probs = logits_per_image.softmax(dim=1)
    ```
"""

NEURON_MASKED_LM_EXAMPLE = r"""
    Example:

    ```python
    >>> from transformers import {processor_class}
    >>> from optimum.neuron import {model_class}

    >>> tokenizer = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")

    >>> inputs = tokenizer("This [MASK] Agreement is between General Motors and John Murray.", return_tensors="pt")

    >>> outputs = model(**inputs)
    >>> logits = outputs.logits
    >>> list(logits.shape)
    [1, 13, 30522]
    ```
"""

NEURON_SEQUENCE_CLASSIFICATION_EXAMPLE = r"""
    Example:

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

NEURON_TOKEN_CLASSIFICATION_EXAMPLE = r"""
    Example:

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

NEURON_QUESTION_ANSWERING_EXAMPLE = r"""
    Example:

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

NEURON_MULTIPLE_CHOICE_EXAMPLE = r"""
    Example:

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

NEURON_IMAGE_CLASSIFICATION_EXAMPLE = r"""
    Example:

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
"""

NEURON_SEMANTIC_SEGMENTATION_EXAMPLE = r"""
    Example:

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
"""

NEURON_OBJECT_DETECTION_EXAMPLE = r"""
    Example:

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
    >>> target_sizes = torch.tensor([image.size[::-1]])
    >>> results = image_processor.post_process_object_detection(outputs, threshold=0.9, target_sizes=target_sizes)[0]
    ```
"""

NEURON_AUDIO_CLASSIFICATION_EXAMPLE = r"""
    Example:

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
"""

NEURON_AUDIO_FRAME_CLASSIFICATION_EXAMPLE = r"""
    Example:

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

NEURON_CTC_EXAMPLE = r"""
    Example:

    ```python
    >>> from transformers import {processor_class}
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

NEURON_CTC_PIPELINE_EXAMPLE = r"""
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

NEURON_AUDIO_XVECTOR_EXAMPLE = r"""
    Example:

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

NEURON_SENTENCE_TRANSFORMERS_TEXT_EXAMPLE = r"""
    Text Example:

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

NEURON_SENTENCE_TRANSFORMERS_IMAGE_EXAMPLE = r"""
    Image Example:

    ```python
    >>> from PIL import Image
    >>> from transformers import {processor_class}
    >>> from sentence_transformers import util
    >>> from optimum.neuron import {model_class}

    >>> processor = {processor_class}.from_pretrained("{checkpoint}")
    >>> model = {model_class}.from_pretrained("{checkpoint}")
    >>> util.http_get("https://github.com/UKPLab/sentence-transformers/raw/master/examples/sentence_transformer/applications/image-search/two_dogs_in_snow.jpg", "two_dogs_in_snow.jpg")
    >>> inputs = processor(
    >>>     text=["Two dogs in the snow", 'A cat on a table', 'A picture of London at night'], images=Image.open("two_dogs_in_snow.jpg"), return_tensors="pt", padding=True
    >>> )

    >>> outputs = model(**inputs)
    >>> cos_scores = util.cos_sim(outputs.image_embeds, outputs.text_embeds)  # Compute cosine similarities
    ```
"""

NEURON_TEXT_GENERATION_EXAMPLE = r"""
    Example:

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

NEURON_TRANSLATION_EXAMPLE = r"""
    *(Following models are compiled with neuronx compiler and can only be run on INF2.)*
    Example of text-to-text generation:

    ```python
    from transformers import {processor_class}
    from optimum.neuron import {model_class}
    # export
    neuron_model = {model_class}.from_pretrained({checkpoint}, export=True, dynamic_batch_size=False, batch_size=1, sequence_length=64, num_beams=4)
    neuron_model.save_pretrained("{save_dir}")
    del neuron_model

    # inference
    neuron_model = {model_class}.from_pretrained("{save_dir}")
    tokenizer = {processor_class}.from_pretrained("{save_dir}")
    inputs = tokenizer("translate English to German: Lets eat good food.", return_tensors="pt")

    output = neuron_model.generate(
        **inputs,
        num_return_sequences=1,
    )
    results = [tokenizer.decode(t, skip_special_tokens=True) for t in output]
    ```
"""

NEURON_TRANSLATION_TP_EXAMPLE = r"""
    *(For large models, in order to fit into Neuron cores, we need to apply tensor parallelism. Here below is an example ran on `inf2.24xlarge`.)*
    Example of text-to-text generation with tensor parallelism:
    ```python
    from transformers import {processor_class}
    from optimum.neuron import {model_class}
    # export
    if __name__ == "__main__":  # compulsory for parallel tracing since the API will spawn multiple processes.
        neuron_model = {model_class}.from_pretrained(
            {checkpoint}, export=True, tensor_parallel_size=8, dynamic_batch_size=False, batch_size=1, sequence_length=128, num_beams=4,
        )
        neuron_model.save_pretrained("{save_dir}")
        del neuron_model
    # inference
    neuron_model = {model_class}.from_pretrained("{save_dir}")
    tokenizer = {processor_class}.from_pretrained("{save_dir}")
    inputs = tokenizer("translate English to German: Lets eat good food.", return_tensors="pt")

    output = neuron_model.generate(
        **inputs,
        num_return_sequences=1,
    )
    results = [tokenizer.decode(t, skip_special_tokens=True) for t in output]
    ```
"""

# ==============================================================================
# PIPELINE CODE EXAMPLES
# ==============================================================================

NEURON_IMAGE_CLASSIFICATION_PIPELINE_EXAMPLE = r"""
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

NEURON_SEMANTIC_SEGMENTATION_PIPELINE_EXAMPLE = r"""
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

NEURON_OBJECT_DETECTION_PIPELINE_EXAMPLE = r"""
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

NEURON_AUDIO_CLASSIFICATION_PIPELINE_EXAMPLE = r"""
    Example:

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

# ==============================================================================
# MODEL INPUTS
# ==============================================================================

NEURON_TEXT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.Tensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using [`AutoTokenizer`](https://huggingface.co/docs/transformers/autoclass_tutorial#autotokenizer).
            See [`PreTrainedTokenizer.encode`](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizerBase.encode) and
            [`PreTrainedTokenizer.__call__`](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizerBase.__call__) for details.
            [What are input IDs?](https://huggingface.co/docs/transformers/glossary#input-ids)
        attention_mask (`torch.Tensor | None` of shape `({0})`, defaults to `None`):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](https://huggingface.co/docs/transformers/glossary#attention-mask)
        token_type_ids (`torch.Tensor | None` of shape `({0})`, defaults to `None`):
            Segment token indices to indicate first and second portions of the inputs. Indices are selected in `[0, 1]`:
            - 1 for tokens that are **sentence A**,
            - 0 for tokens that are **sentence B**.
            [What are token type IDs?](https://huggingface.co/docs/transformers/glossary#token-type-ids)
"""

NEURON_IMAGE_INPUTS_DOCSTRING = r"""
    Args:
        pixel_values (`torch.Tensor | None` of shape `({0})`, defaults to `None`):
            Pixel values corresponding to the images in the current batch.
            Pixel values can be obtained from encoded images using [`AutoImageProcessor`](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoImageProcessor).
"""

NEURON_AUDIO_INPUTS_DOCSTRING = r"""
    Args:
        input_values (`torch.Tensor` of shape `({0})`):
            Float values of input raw speech waveform..
            Input values can be obtained from audio file loaded into an array using [`AutoProcessor`](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoProcessor).
"""

NEURON_TEXT_IMAGE_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.Tensor` of shape `(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using [`AutoTokenizer`](https://huggingface.co/docs/transformers/autoclass_tutorial#autotokenizer).
            See [`PreTrainedTokenizer.encode`](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizerBase.encode) and
            [`PreTrainedTokenizer.__call__`](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizerBase.__call__) for details.
            [What are input IDs?](https://huggingface.co/docs/transformers/glossary#input-ids)
        attention_mask (`torch.Tensor | None` of shape `(batch_size, sequence_length)`):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](https://huggingface.co/docs/transformers/glossary#attention-mask)
        pixel_values (`torch.Tensor | None` of shape `(batch_size, num_channels, height, width)`):
            Pixel values corresponding to the images in the current batch.
            Pixel values can be obtained from encoded images using [`AutoImageProcessor`](https://huggingface.co/docs/transformers/en/model_doc/auto#transformers.AutoImageProcessor).
"""

NEURON_CAUSALLM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, sequence_length)`.
        cache_ids (`torch.LongTensor`): The indices at which the cached key and value for the current inputs need to be stored.
        start_ids (`torch.LongTensor`): The indices of the first tokens to be processed, deduced form the attention masks.
"""

NEURON_SEQ2SEQ_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.Tensor` of shape `({0})`):
            Indices of input sequence tokens in the vocabulary.
            Indices can be obtained using [`AutoTokenizer`](https://huggingface.co/docs/transformers/autoclass_tutorial#autotokenizer).
            See [`PreTrainedTokenizer.encode`](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizerBase.encode) and
            [`PreTrainedTokenizer.__call__`](https://huggingface.co/docs/transformers/main_classes/tokenizer#transformers.PreTrainedTokenizerBase.__call__) for details.
            [What are input IDs?](https://huggingface.co/docs/transformers/glossary#input-ids)
        attention_mask (`torch.Tensor | None` of shape `({0})`, defaults to `None`):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:
            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.
            [What are attention masks?](https://huggingface.co/docs/transformers/glossary#attention-mask)
"""

NEURON_AUDIO_SEQ2SEQ_INPUTS_DOCSTRING = r"""
    Args:
        input_features (`torch.FloatTensor | None` of shape `(batch_size, feature_size, sequence_length)`):
            Float values mel features extracted from the raw speech waveform. Raw speech waveform can be obtained by
            loading a `.flac` or `.wav` audio file into an array of type `list[float]` or a `numpy.ndarray`, *e.g.* via
            the soundfile library (`pip install soundfile`). To prepare the array into `input_features`, the
            [`AutoFeatureExtractor`] should be used for extracting the mel features, padding and conversion into a
            tensor of type `torch.FloatTensor`. See [`~WhisperFeatureExtractor.__call__`]
        decoder_input_ids (`torch.LongTensor | None` of shape `(batch_size, max_sequence_length)`):
            Indices of decoder input sequence tokens in the vocabulary. Indices can be obtained using [`WhisperTokenizer`].
            See [`PreTrainedTokenizer.encode`] and [`PreTrainedTokenizer.__call__`] for details. Since the cache is not yet
            supported for Whisper, it needs to be padded to the `sequence_length` used for the compilation.
        encoder_outputs (`tuple[torch.FloatTensor | None]`):
            Tuple consists of `last_hidden_state` of shape `(batch_size, sequence_length, hidden_size)`) is a sequence of
            hidden-states at the output of the last layer of the encoder. Used in the cross-attention of the decoder.
"""

# ==============================================================================
# MODEL START DOCSTRING
# ==============================================================================

NEURON_MODEL_START_DOCSTRING = r"""
    This model inherits from [`~neuron.modeling.NeuronTracedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving)

    Args:
        config (`transformers.PretrainedConfig`): [PretrainedConfig](https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig) is the Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`optimum.neuron.modeling.NeuronTracedModel.from_pretrained`] method to load the model weights.
        model (`torch.jit._script.ScriptModule`): [torch.jit._script.ScriptModule](https://pytorch.org/docs/stable/generated/torch.jit.ScriptModule.html) is the TorchScript module with embedded NEFF(Neuron Executable File Format) compiled by neuron(x) compiler.
"""

NEURON_CAUSALLM_MODEL_START_DOCSTRING = r"""
    This model inherits from [`~neuron.modeling.NeuronDecoderModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving)

    Args:
        model (`(`torch.jit._script.ScriptModule`)`): [(`torch.jit._script.ScriptModule`)](https://pytorch.org/docs/stable/generated/torch.jit.ScriptModule.html) is the TorchScript module with embedded NEFF(Neuron Executable File Format) of the decoder compiled by neuron(x) compiler.
        config (`transformers.PretrainedConfig`): [PretrainedConfig](https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig) is the Model configuration class with all the parameters of the model.
        model_path (`Path`): The directory where the compiled artifacts for the model are stored.
            It can be a temporary directory if the model has never been saved locally before.
        generation_config (`transformers.GenerationConfig`): [GenerationConfig](https://huggingface.co/docs/transformers/main_classes/text_generation#transformers.GenerationConfig) holds the configuration for the model generation task.
"""

NEURON_SEQ2SEQ_MODEL_START_DOCSTRING = r"""
    This model inherits from [`~neuron.modeling.NeuronTracedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving)

    Args:
        encoder (`torch.jit._script.ScriptModule`): [torch.jit._script.ScriptModule](https://pytorch.org/docs/stable/generated/torch.jit.ScriptModule.html) is the TorchScript module of the encoder with embedded NEFF(Neuron Executable File Format) compiled by neuron(x) compiler.
        decoder (`torch.jit._script.ScriptModule`): [torch.jit._script.ScriptModule](https://pytorch.org/docs/stable/generated/torch.jit.ScriptModule.html) is the TorchScript module of the decoder with embedded NEFF(Neuron Executable File Format) compiled by neuron(x) compiler.
        config (`transformers.PretrainedConfig`): [PretrainedConfig](https://huggingface.co/docs/transformers/main_classes/configuration#transformers.PretrainedConfig) is the Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the
            configuration. Check out the [`optimum.neuron.modeling.NeuronTracedModel.from_pretrained`] method to load the model weights.
"""


# Copied from https://github.com/huggingface/transformers/blob/257bc670fb0eb5e118468f0adfb6e011ddd96782/src/transformers/utils/doc.py#L25
def get_docstring_indentation_level(func):
    """Return the indentation level of the start of the docstring of a class or function (or method)."""
    # We assume classes are always defined in the global scope
    if inspect.isclass(func):
        return 4
    source = inspect.getsource(func)
    first_line = source.splitlines()[0]
    function_def_level = len(first_line) - len(first_line.lstrip())
    return 4 + function_def_level


# Copied from https://github.com/huggingface/transformers/blob/257bc670fb0eb5e118468f0adfb6e011ddd96782/src/transformers/utils/doc.py#L36
def add_start_docstrings(*docstr):
    def docstring_decorator(fn):
        fn.__doc__ = "".join(docstr) + (fn.__doc__ if fn.__doc__ is not None else "")
        return fn

    return docstring_decorator


# Adapted from https://github.com/huggingface/transformers/blob/897ff9af0e8892167af1eb4ec58677001c3a0041/src/transformers/utils/doc.py#L44
def add_start_docstrings_to_model_forward(*docstr):
    def docstring_decorator(fn):
        class_name = f"`{fn.__qualname__.split('.')[0]}`"
        intro = rf"""    The {class_name} forward method, overrides the `__call__` special method. Accepts only the inputs traced during the compilation step. Any additional inputs provided during inference will be ignored. To include extra inputs, recompile the model with those inputs specified."""

        correct_indentation = get_docstring_indentation_level(fn)
        current_doc = fn.__doc__ if fn.__doc__ is not None else ""
        try:
            first_non_empty = next(line for line in current_doc.splitlines() if line.strip() != "")
            doc_indentation = len(first_non_empty) - len(first_non_empty.lstrip())
        except StopIteration:
            doc_indentation = correct_indentation

        docs = docstr
        # In this case, the correct indentation level (class method, 2 Python levels) was respected, and we should
        # correctly reindent everything. Otherwise, the doc uses a single indentation level
        if doc_indentation == 4 + correct_indentation:
            docs = [textwrap.indent(textwrap.dedent(doc), " " * correct_indentation) for doc in docstr]
            intro = textwrap.indent(textwrap.dedent(intro), " " * correct_indentation)

        docstring = "".join(docs) + current_doc
        fn.__doc__ = intro + docstring
        return fn

    return docstring_decorator
