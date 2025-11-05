# coding=utf-8
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
import gc
import os
import shutil
import tempfile
import warnings

import requests
import torch
from datasets import load_dataset
from huggingface_hub.constants import default_cache_path
from optimum.utils import CONFIG_NAME, logging
from parameterized import parameterized
from PIL import Image
from sentence_transformers import SentenceTransformer, util
from transformers import (
    AutoFeatureExtractor,
    AutoImageProcessor,
    AutoModel,
    AutoModelForAudioClassification,
    AutoModelForCTC,
    AutoModelForImageClassification,
    AutoModelForMaskedLM,
    AutoModelForMultipleChoice,
    AutoModelForObjectDetection,
    AutoModelForQuestionAnswering,
    AutoModelForSemanticSegmentation,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoProcessor,
    AutoTokenizer,
    PretrainedConfig,
    set_seed,
)
from transformers import (
    __version__ as transformers_version,
)

from optimum.neuron import (
    NeuronModelForAudioClassification,
    NeuronModelForAudioFrameClassification,
    NeuronModelForCTC,
    NeuronModelForFeatureExtraction,
    NeuronModelForImageClassification,
    NeuronModelForMaskedLM,
    NeuronModelForMultipleChoice,
    NeuronModelForObjectDetection,
    NeuronModelForQuestionAnswering,
    NeuronModelForSemanticSegmentation,
    NeuronModelForSequenceClassification,
    NeuronModelForTokenClassification,
    NeuronModelForXVector,
    NeuronSentenceTransformers,
    NeuronTracedModel,
    pipeline,
)
from optimum.neuron.utils import NEURON_FILE_NAME
from optimum.neuron.utils.testing_utils import is_inferentia_test, requires_neuronx

from ..inference_utils import (
    MODEL_NAMES,
    SEED,
    SENTENCE_TRANSFORMERS_MODEL_NAMES,
    NeuronModelIntegrationTestMixin,
    NeuronModelTestMixin,
)


logger = logging.get_logger()


@is_inferentia_test
class NeuronModelIntegrationTest(NeuronModelIntegrationTestMixin):
    MODEL_ID = MODEL_NAMES["bert"]
    NEURON_MODEL_REPO = "tiny_random_bert_neuronx"
    NEURON_MODEL_CLASS = NeuronModelForSequenceClassification
    STATIC_INPUTS_SHAPES = {"batch_size": 1, "sequence_length": 32}

    TINY_SUBFOLDER_MODEL_ID = "fxmarty/tiny-bert-sst2-distilled-subfolder"
    FAIL_NEURON_MODEL_ID = "sshleifer/tiny-distilbert-base-cased-distilled-squad"
    TINY_MODEL_REMOTE = "Jingya/tiny-random-bert-remote-code"

    def test_load_model_from_local_path(self):
        model = NeuronTracedModel.from_pretrained(self.local_model_path)
        self.assertIsInstance(model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_hub(self):
        model = NeuronTracedModel.from_pretrained(self.neuron_model_id)
        self.assertIsInstance(model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_hub_subfolder(self):
        model = NeuronModelForSequenceClassification.from_pretrained(
            self.TINY_SUBFOLDER_MODEL_ID,
            subfolder="my_subfolder",
            export=True,
            **self.STATIC_INPUTS_SHAPES,
        )
        self.assertIsInstance(model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_cache(self):
        _ = NeuronTracedModel.from_pretrained(self.neuron_model_id)  # caching

        model = NeuronTracedModel.from_pretrained(self.neuron_model_id, local_files_only=True)

        self.assertIsInstance(model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_empty_cache(self):
        dirpath = os.path.join(default_cache_path, "models--" + self.neuron_model_id.replace("/", "--"))

        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        with self.assertRaises(Exception):
            _ = NeuronTracedModel.from_pretrained(self.neuron_model_id, local_files_only=True)

    def test_load_model_from_hub_without_neuron_model(self):
        with self.assertRaises(FileNotFoundError):
            NeuronTracedModel.from_pretrained(self.FAIL_NEURON_MODEL_ID)

    def test_save_model(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model = NeuronTracedModel.from_pretrained(self.local_model_path)
            model.save_pretrained(tmpdirname)
            # folder contains all config files and neuron exported model
            folder_contents = os.listdir(tmpdirname)
            self.assertTrue(NEURON_FILE_NAME in folder_contents)
            self.assertTrue(CONFIG_NAME in folder_contents)

    def test_trust_remote_code(self):
        model = NeuronModelForSequenceClassification.from_pretrained(
            self.TINY_MODEL_REMOTE, export=True, trust_remote_code=True, **self.STATIC_INPUTS_SHAPES
        )
        self.assertIsInstance(model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(model.config, PretrainedConfig)

    @requires_neuronx
    def test_save_compiler_intermediary_files(self):
        with tempfile.TemporaryDirectory() as tempdir:
            save_path = f"{tempdir}/neff"
            neff_path = os.path.join(save_path, "graph.neff")
            _ = NeuronModelForSequenceClassification.from_pretrained(
                self.MODEL_ID,
                export=True,
                compiler_workdir=save_path,
                disable_neuron_cache=True,
                **self.STATIC_INPUTS_SHAPES,
            )
            self.assertTrue(os.path.isdir(save_path))
            os.listdir(save_path)
            self.assertTrue(os.path.exists(neff_path))

    @requires_neuronx
    def test_decouple_weights_neff_and_replace_weight(self):
        with tempfile.TemporaryDirectory() as tempdir:
            # compile
            save_path = f"{tempdir}/neff"
            neuron_model = NeuronModelForSequenceClassification.from_pretrained(
                self.MODEL_ID,
                export=True,
                compiler_workdir=save_path,
                inline_weights_to_neff=False,
                **self.STATIC_INPUTS_SHAPES,
            )
            self.assertFalse(neuron_model.config.neuron.get("inline_weights_to_neff"))

            # replace weights
            model = AutoModelForSequenceClassification.from_pretrained(self.MODEL_ID)
            neuron_model.replace_weights(weights=model)

            self.assertIsInstance(neuron_model.model, torch.jit._script.ScriptModule)


@is_inferentia_test
class NeuronModelForFeatureExtractionIntegrationTest(NeuronModelTestMixin):
    NEURON_MODEL_CLASS = NeuronModelForFeatureExtraction
    TASK = "feature-extraction"
    ATOL_FOR_VALIDATION = 1e-3
    SUPPORTED_ARCHITECTURES = [
        "albert",
        "bert",
        "camembert",
        "convbert",
        "deberta",
        "deberta-v2",
        "distilbert",
        "electra",
        "flaubert",
        "mobilebert",
        "roberta",
        "roformer",
        "xlm",
        "xlm-roberta",
    ]

    def _load_neuron_model_and_processor(self, model_arch, suffix):
        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForFeatureExtraction.from_pretrained(self.neuron_model_dirs[model_arch + suffix])
        preprocessor = AutoTokenizer.from_pretrained(model_id)
        self.assertIsInstance(neuron_model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(neuron_model.config, PretrainedConfig)
        return neuron_model, preprocessor

    def _load_transformers_model(self, model_arch):
        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        set_seed(SEED)
        transformers_model = AutoModel.from_pretrained(model_id)
        return transformers_model

    def _prepare_inputs(self, preprocessor, batch_size):
        text = ["This is a sample output"] * batch_size
        inputs = preprocessor(text, return_tensors="pt")
        return inputs

    def _validate_outputs(self, model_arch, suffix, batch_size):
        neuron_model, preprocessor = self._load_neuron_model_and_processor(model_arch, suffix)
        transformers_model = self._load_transformers_model(model_arch)
        inputs = self._prepare_inputs(preprocessor, batch_size)

        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs)

        # Numeric validation
        atol = neuron_model.neuron_config.ATOL_FOR_VALIDATION or self.ATOL_FOR_VALIDATION
        neuron_outputs = neuron_model(**inputs)
        self.assertIn("last_hidden_state", neuron_outputs)
        self.assertIsInstance(neuron_outputs.last_hidden_state, torch.Tensor)
        result_close_last_hidden_state = torch.allclose(
            neuron_outputs.last_hidden_state,
            transformers_outputs.last_hidden_state,
            atol=atol,
        )
        if not result_close_last_hidden_state:
            warnings.warn(
                f"`last_hidden_state` between pytorch model and neuron model of {model_arch} not close enough."
            )

        if "pooler_output" in neuron_outputs:
            self.assertIsInstance(neuron_outputs.pooler_output, torch.Tensor)
            result_close_pooler_output = torch.allclose(
                neuron_outputs.pooler_output,
                transformers_outputs.pooler_output,
                atol=atol,
            )
            if not result_close_pooler_output:
                warnings.warn(
                    f"`pooler_output` between pytorch model and neuron model of {model_arch} not close enough."
                )

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_compare_to_transformers_non_dyn_bs(self, model_arch):
        model_args = {
            "test_name": model_arch + "_dyn_bs_false",
            "model_arch": model_arch,
            "dynamic_batch_size": False,
        }
        self._setup(model_args)
        self._validate_outputs(model_arch, "_dyn_bs_false", batch_size=1)

        gc.collect()

    def test_compare_to_transformers_dyn_bs(self):
        model_arch = "albert"
        # Neuron model with dynamic batching
        model_args = {
            "test_name": model_arch + "_dyn_bs_true",
            "model_arch": model_arch,
            "dynamic_batch_size": True,
        }
        self._setup(model_args)
        self._validate_outputs(model_arch, "_dyn_bs_true", batch_size=2)

        gc.collect()

    def test_pipeline_model(self):
        model_arch = "albert"
        model_args = {"test_name": model_arch + "_dyn_bs_false", "model_arch": model_arch}
        self._setup(model_args)
        neuron_model, tokenizer = self._load_neuron_model_and_processor(model_arch, "_dyn_bs_false")
        pipe = pipeline(self.TASK, model=neuron_model, tokenizer=tokenizer)
        text = "My Name is Alexander Hamilton."
        outputs = pipe(text)

        self.assertTrue(all(all(isinstance(item, float) for item in row) for row in outputs[0]))

        gc.collect()


@is_inferentia_test
class NeuronSentenceTransformersIntegrationTest(NeuronModelTestMixin):
    NEURON_MODEL_CLASS = NeuronSentenceTransformers
    TASK = "feature-extraction"
    ATOL_FOR_VALIDATION = 1e-2
    SUPPORTED_ARCHITECTURES = ["transformer", "clip"]

    @parameterized.expand(["transformer"], skip_on_empty=True)
    @requires_neuronx
    def test_sentence_transformers_dyn_bs(self, model_arch):
        # Neuron model with dynamic batching
        model_args = {
            "test_name": model_arch + "_dyn_bs_true",
            "model_arch": model_arch,
            "dynamic_batch_size": True,
        }
        self._setup(model_args)

        model_id = SENTENCE_TRANSFORMERS_MODEL_NAMES[model_arch]

        neuron_model_dyn = self.NEURON_MODEL_CLASS.from_pretrained(self.neuron_model_dirs[model_arch + "_dyn_bs_true"])
        self.assertIsInstance(neuron_model_dyn.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(neuron_model_dyn.config, PretrainedConfig)

        set_seed(SEED)
        sentence_transformers_model = SentenceTransformer(model_id)

        text = ["This is a sample output"] * 2
        tokens = neuron_model_dyn.tokenize(text, return_tensors="pt")
        with torch.no_grad():
            sentence_transformers_outputs = sentence_transformers_model(tokens)

        neuron_outputs_dyn = neuron_model_dyn(**tokens)

        # Validate token_embeddings
        atol = neuron_model_dyn.neuron_config.ATOL_FOR_VALIDATION or self.ATOL_FOR_VALIDATION
        self.assertIn("token_embeddings", neuron_outputs_dyn)
        self.assertIsInstance(neuron_outputs_dyn.token_embeddings, torch.Tensor)
        self.assertTrue(
            torch.allclose(
                neuron_outputs_dyn.token_embeddings,
                sentence_transformers_outputs.token_embeddings,
                atol=atol,
            )
        )

        # Validate sentence_embedding
        self.assertIn("sentence_embedding", neuron_outputs_dyn)
        self.assertIsInstance(neuron_outputs_dyn.sentence_embedding, torch.Tensor)
        self.assertTrue(
            torch.allclose(
                neuron_outputs_dyn.sentence_embedding,
                sentence_transformers_outputs.sentence_embedding,
                atol=atol,
            )
        )

        # Encode + Similarity
        sentences_1 = ["Life is pain au chocolat", "Life is galette des rois"]
        sentences_2 = ["Life is eclaire au cafe", "Life is mille feuille"]
        embeddings_1 = neuron_model_dyn.encode(sentences_1, normalize_embeddings=True)
        embeddings_2 = neuron_model_dyn.encode(sentences_2, normalize_embeddings=True)
        similarity = neuron_model_dyn.similarity(embeddings_1, embeddings_2)
        self.assertIsInstance(similarity, torch.Tensor)

        gc.collect()

    @parameterized.expand(["clip"], skip_on_empty=True)
    @requires_neuronx
    def test_sentence_transformers_clip(self, model_arch):
        # Neuron model with dynamic batching
        model_id = SENTENCE_TRANSFORMERS_MODEL_NAMES[model_arch]
        input_shapes = {
            "num_channels": 3,
            "height": 224,
            "width": 224,
            "text_batch_size": 3,
            "image_batch_size": 1,
            "sequence_length": 16,
        }

        neuron_model = self.NEURON_MODEL_CLASS.from_pretrained(
            model_id, subfolder="0_CLIPModel", export=True, **input_shapes
        )
        self.assertIsInstance(neuron_model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(neuron_model.config, PretrainedConfig)

        texts = ["Two dogs in the snow", "A cat on a table", "A picture of London at night"]
        util.http_get(
            "https://github.com/UKPLab/sentence-transformers/raw/master/examples/sentence_transformer/applications/image-search/two_dogs_in_snow.jpg",
            "two_dogs_in_snow.jpg",
        )

        processor = AutoProcessor.from_pretrained(model_id, subfolder="0_CLIPModel")
        inputs = processor(text=texts, images=Image.open("two_dogs_in_snow.jpg"), return_tensors="pt", padding=True)
        outputs = neuron_model(**inputs)
        self.assertIn("image_embeds", outputs)
        self.assertIn("text_embeds", outputs)

        gc.collect()

    def test_pipeline_model(self):
        input_shapes = {
            "batch_size": 1,
            "sequence_length": 16,
        }
        model_id = SENTENCE_TRANSFORMERS_MODEL_NAMES["transformer"]
        neuron_model = self.NEURON_MODEL_CLASS.from_pretrained(model_id, export=True, **input_shapes)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        pipe = pipeline(self.TASK, model=neuron_model, tokenizer=tokenizer)
        text = "My Name is Alexander Hamilton."
        outputs = pipe(text)

        self.assertTrue(all(isinstance(item, float) for item in outputs))

        gc.collect()


@is_inferentia_test
class NeuronModelForMaskedLMIntegrationTest(NeuronModelTestMixin):
    NEURON_MODEL_CLASS = NeuronModelForMaskedLM
    TASK = "fill-mask"
    ATOL_FOR_VALIDATION = 1e-3
    SUPPORTED_ARCHITECTURES = [
        "albert",
        "bert",
        "camembert",
        "convbert",
        "deberta",
        "deberta-v2",
        "distilbert",
        "electra",
        "flaubert",
        "mobilebert",
        "roberta",
        "roformer",
        "xlm",
        "xlm-roberta",
    ]

    def _load_neuron_model_and_processor(self, model_arch, suffix):
        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForMaskedLM.from_pretrained(self.neuron_model_dirs[model_arch + suffix])
        preprocessor = AutoTokenizer.from_pretrained(model_id)
        self.assertIsInstance(neuron_model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(neuron_model.config, PretrainedConfig)
        return neuron_model, preprocessor

    def _load_transformers_model(self, model_arch):
        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        set_seed(SEED)
        transformers_model = AutoModelForMaskedLM.from_pretrained(model_id)
        return transformers_model

    def _prepare_inputs(self, preprocessor, batch_size):
        text = [f"The capital of France is {preprocessor.mask_token}."] * batch_size
        inputs = preprocessor(text, return_tensors="pt")
        return inputs

    def _validate_outputs(self, model_arch, suffix, batch_size):
        neuron_model, preprocessor = self._load_neuron_model_and_processor(model_arch, suffix)
        transformers_model = self._load_transformers_model(model_arch)
        inputs = self._prepare_inputs(preprocessor, batch_size)

        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs)

        # Numeric validation
        atol = neuron_model.neuron_config.ATOL_FOR_VALIDATION or self.ATOL_FOR_VALIDATION
        neuron_outputs = neuron_model(**inputs)
        self.assertIn("logits", neuron_outputs)
        self.assertIsInstance(neuron_outputs.logits, torch.Tensor)
        result_close = torch.allclose(
            neuron_outputs.logits,
            transformers_outputs.logits,
            atol=atol,
        )
        if not result_close:
            warnings.warn(
                f"Inference results between pytorch model and neuron model of {model_arch} not close enough."
            )

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = NeuronModelForMaskedLM.from_pretrained(
                "hf-internal-testing/tiny-random-t5", from_transformers=True, **self.STATIC_INPUTS_SHAPES
            )

        self.assertIn("Unrecognized configuration class", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_compare_to_transformers_non_dyn_bs(self, model_arch):
        model_args = {
            "test_name": model_arch + "_dyn_bs_false",
            "model_arch": model_arch,
            "dynamic_batch_size": False,
        }
        self._setup(model_args)
        self._validate_outputs(model_arch, "_dyn_bs_false", batch_size=1)

        gc.collect()

    def test_compare_to_transformers_dyn_bs(self):
        model_arch = "albert"
        # Neuron model with dynamic batching
        model_args = {
            "test_name": model_arch + "_dyn_bs_true",
            "model_arch": model_arch,
            "dynamic_batch_size": True,
        }
        self._setup(model_args)
        self._validate_outputs(model_arch, "_dyn_bs_true", batch_size=2)

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_pipeline_model(self, model_arch):
        model_args = {"test_name": model_arch + "_dyn_bs_false", "model_arch": model_arch}
        self._setup(model_args)
        neuron_model, tokenizer = self._load_neuron_model_and_processor(model_arch, "_dyn_bs_false")
        MASK_TOKEN = tokenizer.mask_token
        pipe = pipeline(self.TASK, model=neuron_model, tokenizer=tokenizer)
        text = f"The capital of France is {MASK_TOKEN}."
        outputs = pipe(text)

        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["token_str"], str))

        gc.collect()


@is_inferentia_test
class NeuronModelForQuestionAnsweringIntegrationTest(NeuronModelTestMixin):
    NEURON_MODEL_CLASS = NeuronModelForQuestionAnswering
    TASK = "question-answering"
    ATOL_FOR_VALIDATION = 1e-3
    SUPPORTED_ARCHITECTURES = [
        "albert",
        "bert",
        "camembert",
        "convbert",
        "deberta",
        "deberta-v2",
        "distilbert",
        "electra",
        "flaubert",
        "mobilebert",
        "roberta",
        "roformer",
        "xlm",
        "xlm-roberta",
    ]

    def _load_neuron_model_and_processor(self, model_arch, suffix):
        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForQuestionAnswering.from_pretrained(self.neuron_model_dirs[model_arch + suffix])
        preprocessor = AutoTokenizer.from_pretrained(model_id)
        self.assertIsInstance(neuron_model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(neuron_model.config, PretrainedConfig)
        return neuron_model, preprocessor

    def _load_transformers_model(self, model_arch):
        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        set_seed(SEED)
        transformers_model = AutoModelForQuestionAnswering.from_pretrained(model_id)
        return transformers_model

    def _prepare_inputs(self, preprocessor, batch_size):
        text = ["This is a sample output"] * batch_size
        inputs = preprocessor(text, return_tensors="pt")
        return inputs

    def _validate_outputs(self, model_arch, suffix, batch_size):
        neuron_model, preprocessor = self._load_neuron_model_and_processor(model_arch, suffix)
        transformers_model = self._load_transformers_model(model_arch)
        inputs = self._prepare_inputs(preprocessor, batch_size)

        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs)

        # Numeric validation
        atol = neuron_model.neuron_config.ATOL_FOR_VALIDATION or self.ATOL_FOR_VALIDATION
        neuron_outputs = neuron_model(**inputs)
        self.assertIn("start_logits", neuron_outputs)
        self.assertIn("end_logits", neuron_outputs)

        # Compare tensor outputs
        result_close_start_logits = torch.allclose(
            torch.Tensor(neuron_outputs.start_logits),
            transformers_outputs.start_logits,
            atol=atol,
        )
        result_close_end_logits = torch.allclose(
            torch.Tensor(neuron_outputs.end_logits),
            transformers_outputs.end_logits,
            atol=atol,
        )
        if not result_close_start_logits:
            warnings.warn(f"Start logits between pytorch model and neuron model of {model_arch} not close enough.")
        if not result_close_end_logits:
            warnings.warn(f"End logits between pytorch model and neuron model of {model_arch} not close enough.")

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = NeuronModelForQuestionAnswering.from_pretrained(
                "hf-internal-testing/tiny-random-t5", from_transformers=True, **self.STATIC_INPUTS_SHAPES
            )

        assert ("doesn't support" in str(context.exception)) or ("is not supported" in str(context.exception))

    def test_compare_to_transformers_dyn_bs(self):
        model_arch = "albert"
        # Neuron model with dynamic batching
        model_args = {
            "test_name": model_arch + "_dyn_bs_true",
            "model_arch": model_arch,
            "dynamic_batch_size": True,
        }
        self._setup(model_args)
        self._validate_outputs(model_arch, "_dyn_bs_true", batch_size=2)

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_compare_to_transformers_non_dyn_bs(self, model_arch):
        model_args = {
            "test_name": model_arch + "_dyn_bs_false",
            "model_arch": model_arch,
            "dynamic_batch_size": False,
        }
        self._setup(model_args)
        self._validate_outputs(model_arch, "_dyn_bs_false", batch_size=1)

        gc.collect()

    def test_pipeline_model(self):
        model_arch = "albert"
        model_args = {"test_name": model_arch + "_dyn_bs_false", "model_arch": model_arch}
        self._setup(model_args)

        neuron_model, tokenizer = self._load_neuron_model_and_processor(model_arch, "_dyn_bs_false")
        pipe = pipeline(self.TASK, model=neuron_model, tokenizer=tokenizer)
        question = "Whats your name?"
        context = "My Name is Alexander Hamilton."
        outputs = pipe(question, context)

        self.assertGreaterEqual(outputs["score"], 0.0)
        self.assertIsInstance(outputs["answer"], str)

        gc.collect()


@is_inferentia_test
class NeuronModelForSequenceClassificationIntegrationTest(NeuronModelTestMixin):
    NEURON_MODEL_CLASS = NeuronModelForSequenceClassification
    TASK = "text-classification"
    ATOL_FOR_VALIDATION = 1e-3
    SUPPORTED_ARCHITECTURES = [
        "albert",
        "bert",
        "camembert",
        "convbert",
        "deberta",
        "deberta-v2",
        "distilbert",
        "electra",
        "flaubert",
        "mobilebert",
        "roberta",
        "roformer",
        # "xlm",  # accuracy off compared to pytorch (not due to the padding)
        "xlm-roberta",
    ]

    def _load_neuron_model_and_processor(self, model_arch, suffix):
        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForSequenceClassification.from_pretrained(
            self.neuron_model_dirs[model_arch + suffix]
        )
        preprocessor = AutoTokenizer.from_pretrained(model_id)
        self.assertIsInstance(neuron_model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(neuron_model.config, PretrainedConfig)
        return neuron_model, preprocessor

    def _load_transformers_model(self, model_arch):
        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        set_seed(SEED)
        transformers_model = AutoModelForSequenceClassification.from_pretrained(model_id)
        return transformers_model

    def _prepare_inputs(self, preprocessor, batch_size):
        text = ["This is a sample output"] * batch_size
        inputs = preprocessor(text, return_tensors="pt")
        return inputs

    def _validate_outputs(self, model_arch, suffix, batch_size):
        neuron_model, preprocessor = self._load_neuron_model_and_processor(model_arch, suffix)
        transformers_model = self._load_transformers_model(model_arch)
        inputs = self._prepare_inputs(preprocessor, batch_size)

        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs)

        # Numeric validation
        atol = neuron_model.neuron_config.ATOL_FOR_VALIDATION or self.ATOL_FOR_VALIDATION
        neuron_outputs = neuron_model(**inputs)
        self.assertIn("logits", neuron_outputs)
        self.assertIsInstance(neuron_outputs.logits, torch.Tensor)
        result_close = torch.allclose(
            neuron_outputs.logits,
            transformers_outputs.logits,
            atol=atol,
        )
        if not result_close:
            warnings.warn(
                f"Inference results between pytorch model and neuron model of {model_arch} not close enough."
            )

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = NeuronModelForSequenceClassification.from_pretrained(
                "hf-internal-testing/tiny-random-t5", from_transformers=True, **self.STATIC_INPUTS_SHAPES
            )

        assert ("doesn't support" in str(context.exception)) or ("is not supported" in str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_compare_to_transformers_non_dyn_bs(self, model_arch):
        model_args = {
            "test_name": model_arch + "_dyn_bs_false",
            "model_arch": model_arch,
            "dynamic_batch_size": False,
        }
        self._setup(model_args)
        self._validate_outputs(model_arch, "_dyn_bs_false", batch_size=1)

        gc.collect()

    def test_compare_to_transformers_dyn_bs(self):
        model_arch = "albert"
        # Neuron model with dynamic batching
        model_args = {
            "test_name": model_arch + "_dyn_bs_true",
            "model_arch": model_arch,
            "dynamic_batch_size": True,
        }
        self._setup(model_args)
        self._validate_outputs(model_arch, "_dyn_bs_true", batch_size=2)

        gc.collect()

    def test_pipeline_model(self):
        model_arch = "albert"
        model_args = {"test_name": model_arch + "_dyn_bs_false", "model_arch": model_arch}
        self._setup(model_args)
        neuron_model, tokenizer = self._load_neuron_model_and_processor(model_arch, "_dyn_bs_false")
        pipe = pipeline(self.TASK, model=neuron_model, tokenizer=tokenizer)
        text = "I like apple."
        outputs = pipe(text)

        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertIsInstance(outputs[0]["label"], str)

        gc.collect()


@is_inferentia_test
class NeuronModelForTokenClassificationIntegrationTest(NeuronModelTestMixin):
    NEURON_MODEL_CLASS = NeuronModelForTokenClassification
    TASK = "token-classification"
    ATOL_FOR_VALIDATION = 1e-3
    SUPPORTED_ARCHITECTURES = [
        "albert",
        "bert",
        "camembert",
        "convbert",
        "deberta",
        "deberta-v2",
        "distilbert",
        "electra",
        "flaubert",
        "mobilebert",
        "roberta",
        "roformer",
        "xlm",
        "xlm-roberta",
    ]

    def _load_neuron_model_and_processor(self, model_arch, suffix):
        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForTokenClassification.from_pretrained(self.neuron_model_dirs[model_arch + suffix])
        preprocessor = AutoTokenizer.from_pretrained(model_id)
        self.assertIsInstance(neuron_model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(neuron_model.config, PretrainedConfig)
        return neuron_model, preprocessor

    def _load_transformers_model(self, model_arch):
        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        set_seed(SEED)
        transformers_model = AutoModelForTokenClassification.from_pretrained(model_id)
        return transformers_model

    def _prepare_inputs(self, preprocessor, batch_size):
        text = ["This is a sample output"] * batch_size
        inputs = preprocessor(text, return_tensors="pt")
        return inputs

    def _validate_outputs(self, model_arch, suffix, batch_size):
        neuron_model, preprocessor = self._load_neuron_model_and_processor(model_arch, suffix)
        transformers_model = self._load_transformers_model(model_arch)
        inputs = self._prepare_inputs(preprocessor, batch_size)

        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs)

        # Numeric validation
        atol = neuron_model.neuron_config.ATOL_FOR_VALIDATION or self.ATOL_FOR_VALIDATION
        neuron_outputs = neuron_model(**inputs)
        self.assertIn("logits", neuron_outputs)
        self.assertIsInstance(neuron_outputs.logits, torch.Tensor)
        result_close = torch.allclose(
            neuron_outputs.logits,
            transformers_outputs.logits,
            atol=atol,
        )
        if not result_close:
            warnings.warn(
                f"Inference results between pytorch model and neuron model of {model_arch} not close enough."
            )

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = NeuronModelForTokenClassification.from_pretrained(
                "hf-internal-testing/tiny-random-t5", from_transformers=True, **self.STATIC_INPUTS_SHAPES
            )

        assert ("doesn't support" in str(context.exception)) or ("is not supported" in str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_compare_to_transformers_non_dyn_bs(self, model_arch):
        model_args = {
            "test_name": model_arch + "_dyn_bs_false",
            "model_arch": model_arch,
            "dynamic_batch_size": False,
        }
        self._setup(model_args)
        self._validate_outputs(model_arch, "_dyn_bs_false", batch_size=1)

        gc.collect()

    def test_compare_to_transformers_dyn_bs(self):
        model_arch = "albert"
        # Neuron model with dynamic batching
        model_args = {
            "test_name": model_arch + "_dyn_bs_true",
            "model_arch": model_arch,
            "dynamic_batch_size": True,
        }
        self._setup(model_args)
        self._validate_outputs(model_arch, "_dyn_bs_true", batch_size=2)

        gc.collect()

    def test_pipeline_model(self):
        model_arch = "albert"
        model_args = {"test_name": model_arch + "_dyn_bs_false", "model_arch": model_arch}
        self._setup(model_args)
        neuron_model, tokenizer = self._load_neuron_model_and_processor(model_arch, "_dyn_bs_false")
        pipe = pipeline(self.TASK, model=neuron_model, tokenizer=tokenizer)
        text = "My Name is Hamilton."
        outputs = pipe(text)

        self.assertTrue(all(item["score"] > 0.0 for item in outputs))

        gc.collect()


@is_inferentia_test
class NeuronModelForMultipleChoiceIntegrationTest(NeuronModelTestMixin):
    NEURON_MODEL_CLASS = NeuronModelForMultipleChoice
    TASK = "multiple-choice"
    STATIC_INPUTS_SHAPES = {"batch_size": 1, "num_choices": 4, "sequence_length": 128}
    ATOL_FOR_VALIDATION = 1e-3
    SUPPORTED_ARCHITECTURES = [
        "albert",
        "bert",
        "camembert",
        # "convbert",  # accuracy off compared to pytorch: atol=1e-2
        "distilbert",
        "electra",
        "flaubert",
        "mobilebert",
        "roberta",
        # "roformer",  # accuracy off compared to pytorch: atol=1e-1
        # "xlm",  # accuracy off compared to pytorch (not due to the padding)
        # "xlm-roberta",  # Aborted (core dumped)
    ]

    def _load_neuron_model_and_processor(self, model_arch, suffix):
        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForMultipleChoice.from_pretrained(self.neuron_model_dirs[model_arch + suffix])
        preprocessor = AutoTokenizer.from_pretrained(model_id)
        self.assertIsInstance(neuron_model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(neuron_model.config, PretrainedConfig)
        return neuron_model, preprocessor

    def _load_transformers_model(self, model_arch):
        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        set_seed(SEED)
        transformers_model = AutoModelForMultipleChoice.from_pretrained(model_id)
        return transformers_model

    def _prepare_inputs(self, preprocessor, num_choices=4):
        first_sentence = ["The sky is blue due to the shorter wavelength of blue light."] * num_choices
        start = "The color of the sky is"
        second_sentence = [start + "blue", start + "green", start + "red", start + "yellow"]
        inputs = preprocessor(first_sentence, second_sentence, truncation=True, padding=True)

        # Unflatten the tokenized inputs values expanding it to the shape [batch_size, num_choices, seq_length]
        for k, v in inputs.items():
            inputs[k] = [v[i : i + num_choices] for i in range(0, len(v), num_choices)]

        inputs = dict(inputs.convert_to_tensors(tensor_type="pt"))
        return inputs

    def _validate_outputs(self, model_arch, suffix):
        neuron_model, preprocessor = self._load_neuron_model_and_processor(model_arch, suffix)
        transformers_model = self._load_transformers_model(model_arch)
        inputs = self._prepare_inputs(preprocessor)

        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs)

        # Numeric validation
        atol = neuron_model.neuron_config.ATOL_FOR_VALIDATION or self.ATOL_FOR_VALIDATION
        neuron_outputs = neuron_model(**inputs)
        self.assertIn("logits", neuron_outputs)
        self.assertIsInstance(neuron_outputs.logits, torch.Tensor)
        result_close = torch.allclose(
            neuron_outputs.logits,
            transformers_outputs.logits,
            atol=atol,
        )
        if not result_close:
            warnings.warn(
                f"Inference results between pytorch model and neuron model of {model_arch} not close enough."
            )

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_compare_to_transformers_non_dyn_bs(self, model_arch):
        model_args = {
            "test_name": model_arch + "_dyn_bs_false",
            "model_arch": model_arch,
            "dynamic_batch_size": False,
        }
        self._setup(model_args)
        self._validate_outputs(model_arch, "_dyn_bs_false")

        gc.collect()

    def test_compare_to_transformers_dyn_bs(self):
        # Neuron model with dynamic batching
        model_arch = "albert"
        model_args = {
            "test_name": model_arch + "_dyn_bs_true",
            "model_arch": model_arch,
            "dynamic_batch_size": True,
        }
        self._setup(model_args)
        self._validate_outputs(model_arch, "_dyn_bs_true")

        gc.collect()


@is_inferentia_test
class NeuronModelForImageClassificationIntegrationTest(NeuronModelTestMixin):
    NEURON_MODEL_CLASS = NeuronModelForImageClassification
    TASK = "image-classification"
    ATOL_FOR_VALIDATION = 1e-3
    SUPPORTED_ARCHITECTURES = [
        "beit",
        "convnext",
        "convnextv2",
        "cvt",
        "deit",
        "levit",
        "mobilenet-v2",
        "mobilevit",
        "swin",
        "vit",
    ]

    def _load_neuron_model_and_processor(self, model_arch, suffix):
        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForImageClassification.from_pretrained(self.neuron_model_dirs[model_arch + suffix])
        preprocessor = AutoImageProcessor.from_pretrained(model_id)
        self.assertIsInstance(neuron_model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(neuron_model.config, PretrainedConfig)
        return neuron_model, preprocessor

    def _load_transformers_model(self, model_arch):
        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        set_seed(SEED)
        transformers_model = AutoModelForImageClassification.from_pretrained(model_id)
        return transformers_model

    def _prepare_inputs(self, preprocessor, batch_size):
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = preprocessor(images=image, return_tensors="pt")
        if batch_size > 1:
            for name, tensor in inputs.items():
                inputs[name] = torch.cat(batch_size * [tensor])
        return inputs

    def _validate_outputs(self, model_arch, suffix, batch_size):
        neuron_model, preprocessor = self._load_neuron_model_and_processor(model_arch, suffix)
        transformers_model = self._load_transformers_model(model_arch)
        inputs = self._prepare_inputs(preprocessor, batch_size)

        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs)

        # Numeric validation
        atol = neuron_model.neuron_config.ATOL_FOR_VALIDATION or self.ATOL_FOR_VALIDATION
        neuron_outputs = neuron_model(**inputs)
        self.assertIn("logits", neuron_outputs)
        self.assertIsInstance(neuron_outputs.logits, torch.Tensor)
        result_close = torch.allclose(
            neuron_outputs.logits,
            transformers_outputs.logits,
            atol=atol,
        )
        if not result_close:
            warnings.warn(
                f"Inference results between pytorch model and neuron model of {model_arch} not close enough."
            )

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_compare_to_transformers_non_dyn_bs(self, model_arch):
        model_args = {
            "test_name": model_arch + "_dyn_bs_false",
            "model_arch": model_arch,
            "dynamic_batch_size": False,
        }
        # REMOVEME: convnextv2 contains a bug in the GRN layer, which is used in the convnextv2 model, but the bug has
        # been fixed in the transformers library on newer versions. For more info see:
        # https://github.com/huggingface/transformers/issues/38015
        if model_arch == "convnextv2" and transformers_version.startswith("4.51"):
            self.skipTest("convnextv2 contains a bug in this version of transformers.")
        self._setup(model_args)
        self._validate_outputs(model_arch, "_dyn_bs_false", batch_size=1)

        gc.collect()

    def test_compare_to_transformers_dyn_bs(self):
        # Neuron model with dynamic batching
        model_arch = "vit"
        model_args = {
            "test_name": model_arch + "_dyn_bs_true",
            "model_arch": model_arch,
            "dynamic_batch_size": True,
        }
        # REMOVEME: convnextv2 contains a bug in the GRN layer, which is used in the convnextv2 model, but the bug has
        # been fixed in the transformers library on newer versions. For more info see:
        # https://github.com/huggingface/transformers/issues/38015
        if model_arch == "convnextv2" and transformers_version.startswith("4.51"):
            self.skipTest("convnextv2 contains a bug in this version of transformers.")
        self._setup(model_args)
        self._validate_outputs(model_arch, "_dyn_bs_true", batch_size=2)

        gc.collect()

    def test_pipeline_model(self):
        model_arch = "vit"
        model_args = {"test_name": model_arch + "_dyn_bs_false", "model_arch": model_arch}
        self._setup(model_args)

        neuron_model, preprocessor = self._load_neuron_model_and_processor(model_arch, "_dyn_bs_false")

        pipe = pipeline("image-classification", model=neuron_model, feature_extractor=preprocessor)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        outputs = pipe(url)

        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))

        gc.collect()


@is_inferentia_test
class NeuronModelForSemanticSegmentationIntegrationTest(NeuronModelTestMixin):
    NEURON_MODEL_CLASS = NeuronModelForSemanticSegmentation
    TASK = "semantic-segmentation"
    ATOL_FOR_VALIDATION = 1e-3
    SUPPORTED_ARCHITECTURES = [
        "mobilenet-v2",
        "mobilevit",
    ]

    def _load_neuron_model_and_processor(self, model_arch, suffix):
        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForSemanticSegmentation.from_pretrained(self.neuron_model_dirs[model_arch + suffix])
        preprocessor = AutoImageProcessor.from_pretrained(model_id)
        self.assertIsInstance(neuron_model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(neuron_model.config, PretrainedConfig)
        return neuron_model, preprocessor

    def _load_transformers_model(self, model_arch):
        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        set_seed(SEED)
        transformers_model = AutoModelForSemanticSegmentation.from_pretrained(model_id)
        return transformers_model

    def _prepare_inputs(self, preprocessor, batch_size):
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = preprocessor(images=image, return_tensors="pt")
        if batch_size > 1:
            for name, tensor in inputs.items():
                inputs[name] = torch.cat(batch_size * [tensor])
        return inputs

    def _validate_outputs(self, model_arch, suffix, batch_size):
        neuron_model, preprocessor = self._load_neuron_model_and_processor(model_arch, suffix)
        transformers_model = self._load_transformers_model(model_arch)
        inputs = self._prepare_inputs(preprocessor, batch_size)

        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs)

        # Numeric validation
        atol = neuron_model.neuron_config.ATOL_FOR_VALIDATION or self.ATOL_FOR_VALIDATION
        neuron_outputs = neuron_model(**inputs)
        self.assertIn("logits", neuron_outputs)
        self.assertIsInstance(neuron_outputs.logits, torch.Tensor)
        result_close = torch.allclose(
            neuron_outputs.logits,
            transformers_outputs.logits,
            atol=atol,
        )
        if not result_close:
            warnings.warn(
                f"Inference results between pytorch model and neuron model of {model_arch} not close enough."
            )

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_compare_to_transformers_non_dyn_bs(self, model_arch):
        model_args = {
            "test_name": model_arch + "_dyn_bs_false",
            "model_arch": model_arch,
            "dynamic_batch_size": False,
        }
        self._setup(model_args)
        self._validate_outputs(model_arch, "_dyn_bs_false", batch_size=1)

        gc.collect()

    def test_compare_to_transformers_dyn_bs(self):
        # Neuron model with dynamic batching
        model_arch = "mobilevit"
        model_args = {
            "test_name": model_arch + "_dyn_bs_true",
            "model_arch": model_arch,
            "dynamic_batch_size": True,
        }
        self._setup(model_args)
        self._validate_outputs(model_arch, "_dyn_bs_true", batch_size=2)

        gc.collect()

    def test_pipeline_model(self):
        model_arch = "mobilevit"
        model_args = {"test_name": model_arch + "_dyn_bs_false", "model_arch": model_arch}
        self._setup(model_args)

        neuron_model, preprocessor = self._load_neuron_model_and_processor(model_arch, "_dyn_bs_false")

        pipe = pipeline("image-segmentation", model=neuron_model, feature_extractor=preprocessor)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        outputs = pipe(url)

        self.assertTrue(outputs[0]["mask"] is not None)
        self.assertTrue(isinstance(outputs[0]["label"], str))

        gc.collect()


@is_inferentia_test
class NeuronModelForObjectDetectionIntegrationTest(NeuronModelTestMixin):
    NEURON_MODEL_CLASS = NeuronModelForObjectDetection
    TASK = "object-detection"
    ATOL_FOR_VALIDATION = 1e-3
    SUPPORTED_ARCHITECTURES = ["yolos"]

    def _load_neuron_model_and_processor(self, model_arch, suffix):
        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForObjectDetection.from_pretrained(self.neuron_model_dirs[model_arch + suffix])
        preprocessor = AutoImageProcessor.from_pretrained(model_id)
        self.assertIsInstance(neuron_model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(neuron_model.config, PretrainedConfig)
        return neuron_model, preprocessor

    def _load_transformers_model(self, model_arch):
        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        set_seed(SEED)
        transformers_model = AutoModelForObjectDetection.from_pretrained(model_id)
        return transformers_model

    def _prepare_inputs(self, preprocessor, batch_size):
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        image = Image.open(requests.get(url, stream=True).raw)
        inputs = preprocessor(images=image, return_tensors="pt")
        if batch_size > 1:
            for name, tensor in inputs.items():
                inputs[name] = torch.cat(batch_size * [tensor])
        return inputs

    def _validate_outputs(self, model_arch, suffix, batch_size):
        neuron_model, preprocessor = self._load_neuron_model_and_processor(model_arch, suffix)
        transformers_model = self._load_transformers_model(model_arch)
        inputs = self._prepare_inputs(preprocessor, batch_size)

        with torch.no_grad():
            transformers_outputs = transformers_model(**inputs)

        # Numeric validation
        atol = neuron_model.neuron_config.ATOL_FOR_VALIDATION or self.ATOL_FOR_VALIDATION
        neuron_outputs = neuron_model(**inputs)
        self.assertIn("logits", neuron_outputs)
        self.assertIn("pred_boxes", neuron_outputs)
        self.assertIsInstance(neuron_outputs.logits, torch.Tensor)
        result_close = torch.allclose(
            neuron_outputs.logits,
            transformers_outputs.logits,
            atol=atol,
        )
        if not result_close:
            warnings.warn(
                f"Inference results between pytorch model and neuron model of {model_arch} not close enough."
            )

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_compare_to_transformers_non_dyn_bs(self, model_arch):
        model_args = {
            "test_name": model_arch + "_dyn_bs_false",
            "model_arch": model_arch,
            "dynamic_batch_size": False,
        }
        self._setup(model_args)
        self._validate_outputs(model_arch, "_dyn_bs_false", batch_size=1)

        gc.collect()

    def test_compare_to_transformers_dyn_bs(self):
        # Neuron model with dynamic batching
        model_arch = "yolos"
        model_args = {
            "test_name": model_arch + "_dyn_bs_true",
            "model_arch": model_arch,
            "dynamic_batch_size": True,
        }
        self._setup(model_args)
        self._validate_outputs(model_arch, "_dyn_bs_true", batch_size=2)

        gc.collect()

    def test_pipeline_model(self):
        model_arch = "yolos"
        model_args = {"test_name": model_arch + "_dyn_bs_false", "model_arch": model_arch}
        self._setup(model_args)

        neuron_model, preprocessor = self._load_neuron_model_and_processor(model_arch, "_dyn_bs_false")

        pipe = pipeline("object-detection", model=neuron_model, feature_extractor=preprocessor)
        url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        _ = pipe(url)

        gc.collect()


@is_inferentia_test
class NeuronModelForCTCIntegrationTest(NeuronModelTestMixin):
    NEURON_MODEL_CLASS = NeuronModelForCTC
    TASK = "automatic-speech-recognition"
    STATIC_INPUTS_SHAPES = {"batch_size": 1, "audio_sequence_length": 100000}
    ATOL_FOR_VALIDATION = 1e-3
    SUPPORTED_ARCHITECTURES = ["wav2vec2"]

    def _load_neuron_model_and_processor(self, model_arch, suffix):
        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForCTC.from_pretrained(self.neuron_model_dirs[model_arch + suffix])
        preprocessor = AutoProcessor.from_pretrained(model_id)
        self.assertIsInstance(neuron_model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(neuron_model.config, PretrainedConfig)
        return neuron_model, preprocessor

    def _load_transformers_model(self, model_arch):
        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        set_seed(SEED)
        transformers_model = AutoModelForCTC.from_pretrained(model_id)
        return transformers_model

    def _prepare_inputs(self, processor, batch_size):
        dataset = load_dataset(
            "hf-internal-testing/librispeech_asr_demo", "clean", split="validation", trust_remote_code=True
        )
        dataset = dataset.sort("id")
        sampling_rate = dataset.features["audio"].sampling_rate
        inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
        if batch_size > 1:
            for name, tensor in inputs.items():
                inputs[name] = torch.cat(batch_size * [tensor])
        return inputs

    def _validate_outputs(self, model_arch, suffix, batch_size):
        neuron_model, preprocessor = self._load_neuron_model_and_processor(model_arch, suffix)
        inputs = self._prepare_inputs(preprocessor, batch_size)
        neuron_outputs = neuron_model(**inputs)
        self.assertIn("logits", neuron_outputs)
        self.assertIsInstance(neuron_outputs.logits, torch.Tensor)

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_compare_to_transformers_non_dyn_bs(self, model_arch):
        model_args = {
            "test_name": model_arch + "_dyn_bs_false",
            "model_arch": model_arch,
            "dynamic_batch_size": False,
        }
        self._setup(model_args)
        self._validate_outputs(model_arch, "_dyn_bs_false", batch_size=1)

        gc.collect()

    def test_compare_to_transformers_dyn_bs(self):
        # Neuron model with dynamic batching
        model_arch = "wav2vec2"
        model_args = {
            "test_name": model_arch + "_dyn_bs_true",
            "model_arch": model_arch,
            "dynamic_batch_size": True,
        }
        self._setup(model_args)
        self._validate_outputs(model_arch, "_dyn_bs_true", batch_size=2)

        gc.collect()

    def test_pipeline_model(self):
        model_arch = "wav2vec2"
        model_args = {"test_name": model_arch + "_dyn_bs_false", "model_arch": model_arch}
        self._setup(model_args)

        neuron_model, processor = self._load_neuron_model_and_processor(model_arch, "_dyn_bs_false")

        pipe = pipeline(
            "automatic-speech-recognition",
            model=neuron_model,
            feature_extractor=processor.feature_extractor,
            tokenizer=processor.tokenizer,
        )
        dataset = load_dataset(
            "hf-internal-testing/librispeech_asr_demo", "clean", split="validation", trust_remote_code=True
        )
        dataset = dataset.sort("id")
        _ = pipe(dataset[0]["audio"]["array"])

        gc.collect()


@is_inferentia_test
class NeuronModelForAudioClassificationIntegrationTest(NeuronModelTestMixin):
    NEURON_MODEL_CLASS = NeuronModelForAudioClassification
    TASK = "audio-classification"
    STATIC_INPUTS_SHAPES = {"batch_size": 1, "audio_sequence_length": 100000}
    ATOL_FOR_VALIDATION = 1e-3
    SUPPORTED_ARCHITECTURES = ["wav2vec2"]

    def _load_neuron_model_and_processor(self, model_arch, suffix):
        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForAudioClassification.from_pretrained(self.neuron_model_dirs[model_arch + suffix])
        preprocessor = AutoProcessor.from_pretrained(model_id)
        self.assertIsInstance(neuron_model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(neuron_model.config, PretrainedConfig)
        return neuron_model, preprocessor

    def _load_transformers_model(self, model_arch):
        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        set_seed(SEED)
        transformers_model = AutoModelForAudioClassification.from_pretrained(model_id)
        return transformers_model

    def _prepare_inputs(self, processor, batch_size):
        dataset = load_dataset(
            "hf-internal-testing/librispeech_asr_demo", "clean", split="validation", trust_remote_code=True
        )
        dataset = dataset.sort("id")
        sampling_rate = dataset.features["audio"].sampling_rate
        inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
        if batch_size > 1:
            for name, tensor in inputs.items():
                inputs[name] = torch.cat(batch_size * [tensor])
        return inputs

    def _validate_outputs(self, model_arch, suffix, batch_size):
        neuron_model, preprocessor = self._load_neuron_model_and_processor(model_arch, suffix)
        inputs = self._prepare_inputs(preprocessor, batch_size)
        neuron_outputs = neuron_model(**inputs)
        self.assertIn("logits", neuron_outputs)
        self.assertIsInstance(neuron_outputs.logits, torch.Tensor)

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_compare_to_transformers_non_dyn_bs(self, model_arch):
        model_args = {
            "test_name": model_arch + "_dyn_bs_false",
            "model_arch": model_arch,
            "dynamic_batch_size": False,
        }
        self._setup(model_args)
        self._validate_outputs(model_arch, "_dyn_bs_false", batch_size=1)

        gc.collect()

    def test_compare_to_transformers_dyn_bs(self):
        # Neuron model with dynamic batching
        model_arch = "wav2vec2"
        model_args = {
            "test_name": model_arch + "_dyn_bs_true",
            "model_arch": model_arch,
            "dynamic_batch_size": True,
        }
        self._setup(model_args)
        self._validate_outputs(model_arch, "_dyn_bs_true", batch_size=2)

        gc.collect()

    def test_pipeline_model(self):
        model_arch = "wav2vec2"
        model_args = {"test_name": model_arch + "_dyn_bs_false", "model_arch": model_arch}
        self._setup(model_args)

        neuron_model, processor = self._load_neuron_model_and_processor(model_arch, "_dyn_bs_false")

        pipe = pipeline(
            "audio-classification",
            model=neuron_model,
            feature_extractor=processor.feature_extractor,
            tokenizer=processor.tokenizer,
        )
        dataset = load_dataset(
            "hf-internal-testing/librispeech_asr_demo", "clean", split="validation", trust_remote_code=True
        )
        dataset = dataset.sort("id")
        _ = pipe(dataset[0]["audio"]["array"])

        gc.collect()


@is_inferentia_test
class NeuronModelForAudioFrameClassificationIntegrationTest(NeuronModelTestMixin):
    NEURON_MODEL_CLASS = NeuronModelForAudioFrameClassification
    TASK = "audio-frame-classification"
    STATIC_INPUTS_SHAPES = {"batch_size": 1, "audio_sequence_length": 100000}
    ATOL_FOR_VALIDATION = 1e-3
    SUPPORTED_ARCHITECTURES = ["wav2vec2"]

    def _load_neuron_model_and_processor(self, model_arch, suffix):
        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForAudioFrameClassification.from_pretrained(
            self.neuron_model_dirs[model_arch + suffix]
        )
        preprocessor = AutoFeatureExtractor.from_pretrained(model_id)
        self.assertIsInstance(neuron_model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(neuron_model.config, PretrainedConfig)
        return neuron_model, preprocessor

    def _load_transformers_model(self, model_arch):
        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        set_seed(SEED)
        transformers_model = NeuronModelForAudioFrameClassification.from_pretrained(model_id)
        return transformers_model

    def _prepare_inputs(self, processor, batch_size):
        dataset = load_dataset(
            "hf-internal-testing/librispeech_asr_demo", "clean", split="validation", trust_remote_code=True
        )
        dataset = dataset.sort("id")
        sampling_rate = dataset.features["audio"].sampling_rate
        inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
        if batch_size > 1:
            for name, tensor in inputs.items():
                inputs[name] = torch.cat(batch_size * [tensor])
        return inputs

    def _validate_outputs(self, model_arch, suffix, batch_size):
        neuron_model, preprocessor = self._load_neuron_model_and_processor(model_arch, suffix)
        inputs = self._prepare_inputs(preprocessor, batch_size)
        neuron_outputs = neuron_model(**inputs)
        self.assertIn("logits", neuron_outputs)
        self.assertIsInstance(neuron_outputs.logits, torch.Tensor)

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_compare_to_transformers_non_dyn_bs(self, model_arch):
        model_args = {
            "test_name": model_arch + "_dyn_bs_false",
            "model_arch": model_arch,
            "dynamic_batch_size": False,
        }
        self._setup(model_args)
        self._validate_outputs(model_arch, "_dyn_bs_false", batch_size=1)

        gc.collect()

    def test_compare_to_transformers_dyn_bs(self):
        # Neuron model with dynamic batching
        model_arch = "wav2vec2"
        model_args = {
            "test_name": model_arch + "_dyn_bs_true",
            "model_arch": model_arch,
            "dynamic_batch_size": True,
        }
        self._setup(model_args)
        self._validate_outputs(model_arch, "_dyn_bs_true", batch_size=2)

        gc.collect()


@is_inferentia_test
class NeuronModelForXVectorIntegrationTest(NeuronModelTestMixin):
    NEURON_MODEL_CLASS = NeuronModelForXVector
    TASK = "audio-xvector"
    STATIC_INPUTS_SHAPES = {"batch_size": 1, "audio_sequence_length": 100000}
    ATOL_FOR_VALIDATION = 1e-3
    SUPPORTED_ARCHITECTURES = ["wav2vec2"]

    def _load_neuron_model_and_processor(self, model_arch, suffix):
        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForXVector.from_pretrained(self.neuron_model_dirs[model_arch + suffix])
        preprocessor = AutoFeatureExtractor.from_pretrained(model_id)
        self.assertIsInstance(neuron_model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(neuron_model.config, PretrainedConfig)
        return neuron_model, preprocessor

    def _load_transformers_model(self, model_arch):
        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        set_seed(SEED)
        transformers_model = NeuronModelForXVector.from_pretrained(model_id)
        return transformers_model

    def _prepare_inputs(self, processor, batch_size):
        dataset = load_dataset(
            "hf-internal-testing/librispeech_asr_demo", "clean", split="validation", trust_remote_code=True
        )
        dataset = dataset.sort("id")
        sampling_rate = dataset.features["audio"].sampling_rate
        inputs = processor(dataset[0]["audio"]["array"], sampling_rate=sampling_rate, return_tensors="pt")
        if batch_size > 1:
            for name, tensor in inputs.items():
                inputs[name] = torch.cat(batch_size * [tensor])
        return inputs

    def _validate_outputs(self, model_arch, suffix, batch_size):
        neuron_model, preprocessor = self._load_neuron_model_and_processor(model_arch, suffix)
        inputs = self._prepare_inputs(preprocessor, batch_size)
        neuron_outputs = neuron_model(**inputs)
        self.assertIn("logits", neuron_outputs)
        self.assertIsInstance(neuron_outputs.logits, torch.Tensor)

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_compare_to_transformers_non_dyn_bs(self, model_arch):
        model_args = {
            "test_name": model_arch + "_dyn_bs_false",
            "model_arch": model_arch,
            "dynamic_batch_size": False,
        }
        self._setup(model_args)
        self._validate_outputs(model_arch, "_dyn_bs_false", batch_size=1)

        gc.collect()

    def test_compare_to_transformers_dyn_bs(self):
        # Neuron model with dynamic batching
        model_arch = "wav2vec2"
        model_args = {
            "test_name": model_arch + "_dyn_bs_true",
            "model_arch": model_arch,
            "dynamic_batch_size": True,
        }
        self._setup(model_args)
        self._validate_outputs(model_arch, "_dyn_bs_true", batch_size=2)

        gc.collect()
