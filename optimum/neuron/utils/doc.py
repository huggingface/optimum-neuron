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

NEURON_MASKED_LM_EXAMPLE = r"""
    Example:

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
    >>> model = {model_class}.from_pretrained("{checkpoint}", export=True, batch_size=1)

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

NEURON_SENTENCE_TRANSFORMERS_EXAMPLE = r"""
    Example:

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

# ==============================================================================
# MODEL CODE EXAMPLES
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

NEURON_CAUSALLM_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (`torch.LongTensor`):
            Indices of decoder input sequence tokens in the vocabulary of shape `(batch_size, sequence_length)`.
        cache_ids (`torch.LongTensor`): The indices at which the cached key and value for the current inputs need to be stored.
        start_ids (`torch.LongTensor`): The indices of the first tokens to be processed, deduced form the attention masks.
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
