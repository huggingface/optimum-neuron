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
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from typing import Dict

import pytest
import requests
import torch
from huggingface_hub.constants import default_cache_path
from parameterized import parameterized
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoModel,
    AutoModelForAudioClassification,
    AutoModelForAudioFrameClassification,
    AutoModelForAudioXVector,
    AutoModelForCausalLM,
    AutoModelForCTC,
    AutoModelForImageClassification,
    AutoModelForMaskedLM,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoModelForSemanticSegmentation,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForSpeechSeq2Seq,
    AutoModelForTokenClassification,
    AutoModelForVision2Seq,
    AutoTokenizer,
    MBartForConditionalGeneration,
    PretrainedConfig,
    set_seed,
)
from transformers.modeling_utils import no_init_weights
from transformers.neuron.utils import get_preprocessor
from transformers.testing_utils import get_gpu_count, require_torch_gpu
from utils_neuronruntime_tests import MODEL_NAMES, SEED

from optimum.exporters import TasksManager
from optimum.neuron import (
    NEURON_FILE_NAME,
    NeuronModel,
    NeuronModelForFeatureExtraction,
    NeuronModelForMaskedLM,
    NeuronModelForMultipleChoice,
    NeuronModelForQuestionAnswering,
    NeuronModelForSequenceClassification,
    NeuronModelForTokenClassification,
)
from optimum.pipelines import pipeline
from optimum.utils import (
    CONFIG_NAME,
    logging,
)
from optimum.utils.testing_utils import grid_parameters, require_hf_token


logger = logging.get_logger()


class NeuronModelTestMixin(unittest.TestCase):
    ARCH_MODEL_MAP = {}

    @classmethod
    def setUpClass(cls):
        cls.neuron_model_dirs = {}

    def _setup(self, model_args: Dict):
        """
        Exports the PyTorch models to Neuron model ahead of time to avoid multiple exports during the tests.
        We don't use unittest setUpClass, in order to still be able to run individual tests.
        """
        model_arch = model_args["model_arch"]
        model_arch_and_params = model_args["test_name"]

        if "use_cache" in model_args and task not in TasksManager.get_supported_tasks_for_model_type(
            model_arch.replace("_", "-"), exporter="neuron"
        ):
            self.skipTest("Unsupported export case")

        if model_arch_and_params not in self.neuron_model_dirs:
            # model_args will contain kwargs to pass to NeuronModel.from_pretrained()
            model_args.pop("test_name")
            model_args.pop("model_arch")

            model_id = (
                self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
            )
            set_seed(SEED)
            neuron_model = self.NEURONMODEL_CLASS.from_pretrained(model_id, **model_args, export=True)

            model_dir = tempfile.mkdtemp(prefix=f"{model_arch_and_params}_{self.TASK}_")
            neuron_model.save_pretrained(model_dir)
            self.neuron_model_dirs[model_arch_and_params] = model_dir

    @classmethod
    def tearDownClass(cls):
        for _, dir_path in cls.neuron_model_dirs.items():
            shutil.rmtree(dir_path)


class NeuronModelIntegrationTest(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.TEST_MODEL_ID = "sshleifer/tiny-distilbert-base-cased-distilled-squad"
        self.LOCAL_MODEL_PATH = "assets/neuron"
        self.NEURON_MODEL_ID = "philschmid/distilbert-neuron"
        self.TINY_NEURON_MODEL_ID = "fxmarty/resnet-tiny-beans"
        self.FAIL_NEURON_MODEL_ID = "sshleifer/tiny-distilbert-base-cased-distilled-squad"

    def test_load_model_from_local_path(self):
        model = NeuronModel.from_pretrained(self.LOCAL_MODEL_PATH)
        self.assertIsInstance(model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_hub(self):
        model = NeuronModel.from_pretrained(self.NEURON_MODEL_ID)
        self.assertIsInstance(model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_hub_subfolder(self):
        # does not pass with NeuronModel as it does not have export_feature attribute
        model = NeuronModelForSequenceClassification.from_pretrained(
            "fxmarty/tiny-bert-sst2-distilled-subfolder", subfolder="my_subfolder", from_transformers=True
        )
        self.assertIsInstance(model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(model.config, PretrainedConfig)

        model = NeuronModel.from_pretrained("fxmarty/tiny-bert-sst2-distilled-neuron-subfolder", subfolder="my_subfolder")
        self.assertIsInstance(model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_cache(self):
        _ = NeuronModel.from_pretrained(self.TINY_NEURON_MODEL_ID)  # caching

        model = NeuronModel.from_pretrained(self.TINY_NEURON_MODEL_ID, local_files_only=True)

        self.assertIsInstance(model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_empty_cache(self):
        dirpath = os.path.join(default_cache_path, "models--" + self.TINY_neuron_MODEL_ID.replace("/", "--"))

        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        with self.assertRaises(Exception):
            _ = NeuronModel.from_pretrained(self.TINY_NEURON_MODEL_ID, local_files_only=True)  

    def test_load_model_from_hub_without_neuron_model(self):
        with self.assertRaises(FileNotFoundError):
            NeuronModel.from_pretrained(self.FAIL_NEURON_MODEL_ID)

    @require_hf_token
    def test_load_model_from_hub_private(self):
        model = NeuronModel.from_pretrained(self.NEURON_MODEL_ID, use_auth_token=os.environ.get("HF_AUTH_TOKEN", None))
        self.assertIsInstance(model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_save_model(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model = NeuronModel.from_pretrained(self.LOCAL_MODEL_PATH)
            model.save_pretrained(tmpdirname)
            # folder contains all config files and neuron exported model
            folder_contents = os.listdir(tmpdirname)
            self.assertTrue(NEURON_FILE_NAME in folder_contents)
            self.assertTrue(CONFIG_NAME in folder_contents)

    @require_hf_token
    def test_save_model_from_hub(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model = NeuronModel.from_pretrained(self.LOCAL_MODEL_PATH)
            model.save_pretrained(
                tmpdirname,
                use_auth_token=os.environ.get("HF_AUTH_TOKEN", None),
                push_to_hub=True,
                repository_id=self.HUB_REPOSITORY,
                private=True,
            )

    def test_trust_remote_code(self):
        model_id = "fxmarty/tiny-testing-gpt2-remote-code"
        neuron_model = NeuronModelForCausalLM.from_pretrained(model_id, export=True, trust_remote_code=True)
        pt_model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True)

        tokenizer = AutoTokenizer.from_pretrained(model_id)

        inputs = tokenizer("My name is", return_tensors="pt")

        with torch.inference_mode():
            pt_logits = pt_model(**inputs).logits

        neuron_logits = neuron_model(**inputs).logits

        self.assertTrue(
            torch.allclose(pt_logits, neuron_logits, atol=1e-4), f" Maxdiff: {torch.abs(pt_logits - neuron_logits).max()}"
        )


class NeuronModelForQuestionAnsweringIntegrationTest(NeuronModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "albert",
        "bart",
        "bert",
        # "big_bird",
        # "bigbird_pegasus",
        "camembert",
        "convbert",
        "data2vec_text",
        "deberta",
        "deberta_v2",
        "distilbert",
        "electra",
        "flaubert",
        "gptj",
        "ibert",
        # TODO: these two should be supported, but require image inputs not supported in NeuronModel
        # "layoutlm"
        # "layoutlmv3",
        "mbart",
        "mobilebert",
        "nystromformer",
        "roberta",
        "roformer",
        "squeezebert",
        "xlm",
        "xlm_roberta",
    ]

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    NeuronModel_CLASS = NeuronModelForQuestionAnswering
    TASK = "question-answering"

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = NeuronModelForQuestionAnswering.from_pretrained(MODEL_NAMES["t5"], from_transformers=True)

        self.assertIn("Unrecognized configuration class", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForQuestionAnswering.from_pretrained(self.neuron_model_dirs[model_arch])

        self.assertIsInstance(neuron_model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(neuron_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForQuestionAnswering.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)

        tokens = tokenizer("This is a sample output", return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        for input_type in ["pt", "np"]:
            tokens = tokenizer("This is a sample output", return_tensors=input_type)
            neuron_outputs = neuron_model(**tokens)

            self.assertIn("start_logits", neuron_outputs)
            self.assertIn("end_logits", neuron_outputs)
            self.assertIsInstance(neuron_outputs.start_logits, self.TENSOR_ALIAS_TO_TYPE[input_type])
            self.assertIsInstance(neuron_outputs.end_logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # Compare tensor outputs
            self.assertTrue(
                torch.allclose(torch.Tensor(neuron_outputs.start_logits), transformers_outputs.start_logits, atol=1e-4)
            )
            self.assertTrue(
                torch.allclose(torch.Tensor(neuron_outputs.end_logits), transformers_outputs.end_logits, atol=1e-4)
            )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_neuron_model(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForQuestionAnswering.from_pretrained(self.neuron_model_dirs[model_arch])
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("question-answering", model=neuron_model, tokenizer=tokenizer)
        question = "Whats my name?"
        context = "My Name is Philipp and I live in Nuremberg."
        outputs = pipe(question, context)

        self.assertEqual(pipe.device, pipe.model.device)
        self.assertGreaterEqual(outputs["score"], 0.0)
        self.assertIsInstance(outputs["answer"], str)

        gc.collect()

    @pytest.mark.run_in_series
    def test_pipeline_model_is_none(self):
        pipe = pipeline("question-answering")
        question = "Whats my name?"
        context = "My Name is Philipp and I live in Nuremberg."
        outputs = pipe(question, context)

        # compare model output class
        self.assertGreaterEqual(outputs["score"], 0.0)
        self.assertIsInstance(outputs["answer"], str)


class NeuronModelForMaskedLMIntegrationTest(NeuronModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "albert",
        "bert",
        "camembert",
        # "convbert",
        # "deberta",
        # "deberta_v2",
        "distilbert",
        "electra",
        "flaubert",
        "mobilebert",
        "mpnet",
        "roberta",
        "roformer",
        "xlm",
        "xlm_roberta",
    ]

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    NeuronModel_CLASS = NeuronModelForMaskedLM
    TASK = "fill-mask"

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = NeuronModelForMaskedLM.from_pretrained(MODEL_NAMES["t5"], from_transformers=True)

        self.assertIn("Unrecognized configuration class", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForMaskedLM.from_pretrained(self.neuron_model_dirs[model_arch])

        self.assertIsInstance(neuron_model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(neuron_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForMaskedLM.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)

        text = f"The capital of France is {tokenizer.mask_token}."
        tokens = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        for input_type in ["pt", "np"]:
            tokens = tokenizer(text, return_tensors=input_type)
            neuron_outputs = neuron_model(**tokens)

            self.assertIn("logits", neuron_outputs)
            self.assertIsInstance(neuron_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # compare tensor outputs
            self.assertTrue(torch.allclose(torch.Tensor(neuron_outputs.logits), transformers_outputs.logits, atol=1e-4))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_ort_model(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForMaskedLM.from_pretrained(self.neuron_model_dirs[model_arch])
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("fill-mask", model=neuron_model, tokenizer=tokenizer)
        MASK_TOKEN = tokenizer.mask_token
        text = f"The capital of France is {MASK_TOKEN}."
        outputs = pipe(text)

        self.assertEqual(pipe.device, neuron_model.device)
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertIsInstance(outputs[0]["token_str"], str)

        gc.collect()

    @pytest.mark.run_in_series
    def test_pipeline_model_is_none(self):
        pipe = pipeline("fill-mask")
        text = "The capital of France is [MASK]."
        outputs = pipe(text)

        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertIsInstance(outputs[0]["token_str"], str)

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @pytest.mark.gpu_test
    def test_pipeline_on_gpu(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForMaskedLM.from_pretrained(self.neuron_model_dirs[model_arch])
        tokenizer = get_preprocessor(model_id)
        MASK_TOKEN = tokenizer.mask_token
        pipe = pipeline("fill-mask", model=neuron_model, tokenizer=tokenizer, device=0)
        text = f"The capital of France is {MASK_TOKEN}."
        outputs = pipe(text)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["token_str"], str))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @pytest.mark.gpu_test
    def test_compare_to_io_binding(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForMaskedLM.from_pretrained(self.neuron_model_dirs[model_arch], use_io_binding=False).to(
            "cuda"
        )
        io_model = NeuronModelForMaskedLM.from_pretrained(self.neuron_model_dirs[model_arch], use_io_binding=True).to(
            "cuda"
        )

        tokenizer = get_preprocessor(model_id)
        MASK_TOKEN = tokenizer.mask_token
        tokens = tokenizer([f"The capital of France is {MASK_TOKEN}."] * 2, return_tensors="pt")
        neuron_outputs = neuron_model(**tokens)
        io_outputs = io_model(**tokens)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        self.assertTrue(torch.equal(neuron_outputs.logits, io_outputs.logits))

        gc.collect()


class NeuronModelForSequenceClassificationIntegrationTest(NeuronModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "albert",
        "bart",
        "bert",
        # "big_bird",
        # "bigbird_pegasus",
        "bloom",
        "camembert",
        "convbert",
        "data2vec_text",
        "deberta",
        "deberta_v2",
        "distilbert",
        "electra",
        "flaubert",
        "gpt2",
        "gpt_neo",
        "gptj",
        "ibert",
        # TODO: these two should be supported, but require image inputs not supported in NeuronModel
        # "layoutlm"
        # "layoutlmv3",
        "mbart",
        "mobilebert",
        "nystromformer",
        # "perceiver",
        "roberta",
        "roformer",
        "squeezebert",
        "xlm",
        "xlm_roberta",
    ]

    ARCH_MODEL_MAP = {
        # TODO: fix non passing test
        # "perceiver": "hf-internal-testing/tiny-random-language_perceiver",
    }

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    NeuronModel_CLASS = NeuronModelForSequenceClassification
    TASK = "text-classification"

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = NeuronModelForSequenceClassification.from_pretrained(MODEL_NAMES["t5"], from_transformers=True)

        self.assertIn("Unrecognized configuration class", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForSequenceClassification.from_pretrained(self.neuron_model_dirs[model_arch])

        self.assertIsInstance(neuron_model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(neuron_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForSequenceClassification.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)

        text = "This is a sample output"
        tokens = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        for input_type in ["pt", "np"]:
            tokens = tokenizer(text, return_tensors=input_type)
            neuron_outputs = neuron_model(**tokens)

            self.assertIn("logits", neuron_outputs)
            self.assertIsInstance(neuron_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # compare tensor outputs
            self.assertTrue(torch.allclose(torch.Tensor(neuron_outputs.logits), transformers_outputs.logits, atol=1e-4))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_ort_model(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForSequenceClassification.from_pretrained(self.neuron_model_dirs[model_arch])
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("text-classification", model=neuron_model, tokenizer=tokenizer)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)

        self.assertEqual(pipe.device, neuron_model.device)
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertIsInstance(outputs[0]["label"], str)

        gc.collect()

    @pytest.mark.run_in_series
    def test_pipeline_model_is_none(self):
        pipe = pipeline("text-classification")
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)

        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertIsInstance(outputs[0]["label"], str)

    @parameterized.expand(
        grid_parameters(
            {"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["CUDAExecutionProvider", "TensorrtExecutionProvider"]}
        )
    )
    @require_torch_gpu
    @pytest.mark.gpu_test
    def test_pipeline_on_gpu(self, test_name: str, model_arch: str, provider: str):
        if provider == "TensorrtExecutionProvider" and model_arch != self.__class__.SUPPORTED_ARCHITECTURES[0]:
            self.skipTest("testing a single arch for TensorrtExecutionProvider")

        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForSequenceClassification.from_pretrained(
            self.neuron_model_dirs[model_arch], provider=provider
        )
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("text-classification", model=neuron_model, tokenizer=tokenizer, device=0)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertTrue(isinstance(outputs[0]["label"], str))

        gc.collect()

    def test_pipeline_zero_shot_classification(self):
        neuron_model = NeuronModelForSequenceClassification.from_pretrained(
            "typeform/distilbert-base-uncased-mnli", from_transformers=True
        )
        tokenizer = get_preprocessor("typeform/distilbert-base-uncased-mnli")
        pipe = pipeline("zero-shot-classification", model=neuron_model, tokenizer=tokenizer)
        sequence_to_classify = "Who are you voting for in 2020?"
        candidate_labels = ["Europe", "public health", "politics", "elections"]
        hypothesis_template = "This text is about {}."
        outputs = pipe(
            sequence_to_classify, candidate_labels, multi_class=True, hypothesis_template=hypothesis_template
        )

        # compare model output class
        self.assertTrue(all(score > 0.0 for score in outputs["scores"]))
        self.assertTrue(all(isinstance(label, str) for label in outputs["labels"]))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @pytest.mark.gpu_test
    def test_compare_to_io_binding(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForSequenceClassification.from_pretrained(
            self.neuron_model_dirs[model_arch], use_io_binding=False
        ).to("cuda")
        io_model = NeuronModelForSequenceClassification.from_pretrained(
            self.neuron_model_dirs[model_arch], use_io_binding=True
        ).to("cuda")

        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer(["This is a sample output"] * 2, return_tensors="pt")
        neuron_outputs = neuron_model(**tokens)
        io_outputs = io_model(**tokens)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        self.assertTrue(torch.equal(neuron_outputs.logits, io_outputs.logits))

        gc.collect()


class NeuronModelForTokenClassificationIntegrationTest(NeuronModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "albert",
        "bert",
        # "big_bird",
        "bloom",
        "camembert",
        "convbert",
        "data2vec_text",
        "deberta",
        "deberta_v2",
        "distilbert",
        "electra",
        "flaubert",
        "gpt2",
        "ibert",
        # TODO: these two should be supported, but require image inputs not supported in NeuronModel
        # "layoutlm"
        # "layoutlmv3",
        "mobilebert",
        "roberta",
        "roformer",
        "squeezebert",
        "xlm",
        "xlm_roberta",
    ]

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    NeuronModel_CLASS = NeuronModelForTokenClassification
    TASK = "token-classification"

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = NeuronModelForTokenClassification.from_pretrained(MODEL_NAMES["t5"], from_transformers=True)

        self.assertIn("Unrecognized configuration class", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForTokenClassification.from_pretrained(self.neuron_model_dirs[model_arch])

        self.assertIsInstance(neuron_model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(neuron_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForTokenClassification.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)

        text = "This is a sample output"
        tokens = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        for input_type in ["pt", "np"]:
            tokens = tokenizer(text, return_tensors=input_type)
            neuron_outputs = neuron_model(**tokens)

            self.assertIn("logits", neuron_outputs)
            self.assertIsInstance(neuron_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # compare tensor outputs
            self.assertTrue(torch.allclose(torch.Tensor(neuron_outputs.logits), transformers_outputs.logits, atol=1e-4))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_ort_model(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForTokenClassification.from_pretrained(self.neuron_model_dirs[model_arch])
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("token-classification", model=neuron_model, tokenizer=tokenizer)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)

        self.assertEqual(pipe.device, neuron_model.device)
        self.assertTrue(all(item["score"] > 0.0 for item in outputs))

        gc.collect()

    @pytest.mark.run_in_series
    def test_pipeline_model_is_none(self):
        pipe = pipeline("token-classification")
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)

        # compare model output class
        self.assertTrue(all(item["score"] > 0.0 for item in outputs))

    @parameterized.expand(
        grid_parameters(
            {"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["CUDAExecutionProvider", "TensorrtExecutionProvider"]}
        )
    )
    @require_torch_gpu
    @pytest.mark.gpu_test
    def test_pipeline_on_gpu(self, test_name: str, model_arch: str, provider: str):
        if provider == "TensorrtExecutionProvider" and model_arch != self.__class__.SUPPORTED_ARCHITECTURES[0]:
            self.skipTest("testing a single arch for TensorrtExecutionProvider")

        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForTokenClassification.from_pretrained(
            self.neuron_model_dirs[model_arch], provider=provider
        )
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("token-classification", model=neuron_model, tokenizer=tokenizer, device=0)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertTrue(all(item["score"] > 0.0 for item in outputs))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @pytest.mark.gpu_test
    def test_compare_to_io_binding(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForTokenClassification.from_pretrained(
            self.neuron_model_dirs[model_arch], use_io_binding=False
        ).to("cuda")
        io_model = NeuronModelForTokenClassification.from_pretrained(
            self.neuron_model_dirs[model_arch], use_io_binding=True
        ).to("cuda")

        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer(["This is a sample output"] * 2, return_tensors="pt")
        neuron_outputs = neuron_model(**tokens)
        io_outputs = io_model(**tokens)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        self.assertTrue(torch.equal(neuron_outputs.logits, io_outputs.logits))

        gc.collect()


class NeuronModelForFeatureExtractionIntegrationTest(NeuronModelTestMixin):
    SUPPORTED_ARCHITECTURES = ["albert", "bert", "camembert", "distilbert", "electra", "roberta", "xlm_roberta"]

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    NeuronModel_CLASS = NeuronModelForFeatureExtraction
    TASK = "feature-extraction"

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForFeatureExtraction.from_pretrained(self.neuron_model_dirs[model_arch])

        self.assertIsInstance(neuron_model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(neuron_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModel.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        text = "This is a sample output"
        tokens = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        for input_type in ["pt", "np"]:
            tokens = tokenizer(text, return_tensors=input_type)
            neuron_outputs = neuron_model(**tokens)

            self.assertIn("last_hidden_state", neuron_outputs)
            self.assertIsInstance(neuron_outputs.last_hidden_state, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # compare tensor outputs
            self.assertTrue(
                torch.allclose(
                    torch.Tensor(neuron_outputs.last_hidden_state), transformers_outputs.last_hidden_state, atol=1e-4
                )
            )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_pipeline_ort_model(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForFeatureExtraction.from_pretrained(self.neuron_model_dirs[model_arch])
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("feature-extraction", model=neuron_model, tokenizer=tokenizer)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)

        # compare model output class
        self.assertEqual(pipe.device, neuron_model.device)
        self.assertTrue(all(all(isinstance(item, float) for item in row) for row in outputs[0]))

        gc.collect()

    @pytest.mark.run_in_series
    def test_pipeline_model_is_none(self):
        pipe = pipeline("feature-extraction")
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)

        # compare model output class
        self.assertTrue(all(all(isinstance(item, float) for item in row) for row in outputs[0]))

    @parameterized.expand(
        grid_parameters(
            {"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["CUDAExecutionProvider", "TensorrtExecutionProvider"]}
        )
    )
    @require_torch_gpu
    @pytest.mark.gpu_test
    def test_pipeline_on_gpu(self, test_name: str, model_arch: str, provider: str):
        if provider == "TensorrtExecutionProvider" and model_arch != self.__class__.SUPPORTED_ARCHITECTURES[0]:
            self.skipTest("testing a single arch for TensorrtExecutionProvider")

        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForFeatureExtraction.from_pretrained(self.neuron_model_dirs[model_arch], provider=provider)
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline("feature-extraction", model=neuron_model, tokenizer=tokenizer, device=0)
        text = "My Name is Philipp and i live in Germany."
        outputs = pipe(text)
        # check model device
        self.assertEqual(pipe.model.device.type.lower(), "cuda")
        # compare model output class
        self.assertTrue(all(all(isinstance(item, float) for item in row) for row in outputs[0]))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @pytest.mark.gpu_test
    def test_compare_to_io_binding(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForFeatureExtraction.from_pretrained(
            self.neuron_model_dirs[model_arch], use_io_binding=False
        ).to("cuda")
        io_model = NeuronModelForFeatureExtraction.from_pretrained(
            self.neuron_model_dirs[model_arch], use_io_binding=True
        ).to("cuda")

        tokenizer = get_preprocessor(model_id)
        tokens = tokenizer(["This is a sample output"] * 2, return_tensors="pt")
        neuron_outputs = neuron_model(**tokens)
        io_outputs = io_model(**tokens)

        self.assertTrue("last_hidden_state" in io_outputs)
        self.assertIsInstance(io_outputs.last_hidden_state, torch.Tensor)

        # compare tensor outputs
        self.assertTrue(torch.equal(neuron_outputs.last_hidden_state, io_outputs.last_hidden_state))

        gc.collect()


class NeuronModelForMultipleChoiceIntegrationTest(NeuronModelTestMixin):
    # Multiple Choice tests are conducted on different models due to mismatch size in model's classifier
    SUPPORTED_ARCHITECTURES = [
        "albert",
        "bert",
        # "big_bird",
        "camembert",
        "convbert",
        "data2vec_text",
        "deberta_v2",
        "distilbert",
        "electra",
        "flaubert",
        "ibert",
        "mobilebert",
        "nystromformer",
        "roberta",
        "roformer",
        "squeezebert",
        "xlm",
        "xlm_roberta",
    ]

    FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
    NeuronModel_CLASS = NeuronModelForMultipleChoice
    TASK = "multiple-choice"

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForMultipleChoice.from_pretrained(self.neuron_model_dirs[model_arch])

        self.assertIsInstance(neuron_model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(neuron_model.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForMultipleChoice.from_pretrained(model_id)
        tokenizer = get_preprocessor(model_id)
        num_choices = 4
        first_sentence = ["The sky is blue due to the shorter wavelength of blue light."] * num_choices
        start = "The color of the sky is"
        second_sentence = [start + "blue", start + "green", start + "red", start + "yellow"]
        inputs = tokenizer(first_sentence, second_sentence, truncation=True, padding=True)

        # Unflatten the tokenized inputs values expanding it to the shape [batch_size, num_choices, seq_length]
        for k, v in inputs.items():
            inputs[k] = [v[i : i + num_choices] for i in range(0, len(v), num_choices)]

        pt_inputs = dict(inputs.convert_to_tensors(tensor_type="pt"))
        with torch.no_grad():
            transformers_outputs = transformers_model(**pt_inputs)

        for input_type in ["pt", "np"]:
            inps = dict(inputs.convert_to_tensors(tensor_type=input_type))
            neuron_outputs = neuron_model(**inps)

            self.assertTrue("logits" in neuron_outputs)
            self.assertIsInstance(neuron_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

            # Compare tensor outputs
            self.assertTrue(torch.allclose(torch.Tensor(neuron_outputs.logits), transformers_outputs.logits, atol=1e-4))

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    @require_torch_gpu
    @pytest.mark.gpu_test
    def test_compare_to_io_binding(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForMultipleChoice.from_pretrained(
            self.neuron_model_dirs[model_arch], use_io_binding=False
        ).to("cuda")
        io_model = NeuronModelForMultipleChoice.from_pretrained(self.neuron_model_dirs[model_arch], use_io_binding=True).to(
            "cuda"
        )

        tokenizer = get_preprocessor(model_id)
        num_choices = 4
        first_sentence = ["The sky is blue due to the shorter wavelength of blue light."] * num_choices
        start = "The color of the sky is"
        second_sentence = [start + "blue", start + "green", start + "red", start + "yellow"]
        inputs = tokenizer(first_sentence, second_sentence, truncation=True, padding=True)

        # Unflatten the tokenized inputs values expanding it to the shape [batch_size, num_choices, seq_length]
        for k, v in inputs.items():
            inputs[k] = [v[i : i + num_choices] for i in range(0, len(v), num_choices)]

        inputs = dict(inputs.convert_to_tensors(tensor_type="pt"))

        neuron_outputs = neuron_model(**inputs)
        io_outputs = io_model(**inputs)

        self.assertTrue("logits" in io_outputs)
        self.assertIsInstance(io_outputs.logits, torch.Tensor)

        # compare tensor outputs
        self.assertTrue(torch.equal(neuron_outputs.logits, io_outputs.logits))

        gc.collect()
