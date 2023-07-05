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
import unittest
from typing import Dict

import torch
from huggingface_hub.constants import default_cache_path
from parameterized import parameterized
from transformers import (
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForMultipleChoice,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PretrainedConfig,
    set_seed,
)
from transformers.onnx.utils import get_preprocessor

from optimum.neuron import (
    NeuronBaseModel,
    NeuronModelForFeatureExtraction,
    NeuronModelForMaskedLM,
    NeuronModelForMultipleChoice,
    NeuronModelForQuestionAnswering,
    NeuronModelForSequenceClassification,
    NeuronModelForTokenClassification,
    pipeline,
)
from optimum.neuron.utils import NEURON_FILE_NAME, is_neuron_available, is_neuronx_available
from optimum.neuron.utils.testing_utils import is_inferentia_test
from optimum.utils import (
    CONFIG_NAME,
    logging,
)
from optimum.utils.testing_utils import require_hf_token

from .exporters.exporters_utils import EXPORT_MODELS_TINY as MODEL_NAMES
from .exporters.exporters_utils import SEED


logger = logging.get_logger()


class NeuronModelTestMixin(unittest.TestCase):
    ARCH_MODEL_MAP = {}
    STATIC_INPUTS_SHAPES = {"batch_size": 1, "sequence_length": 32}

    @classmethod
    def setUpClass(cls):
        cls.neuron_model_dirs = {}

    def _setup(self, model_args: Dict):
        """
        Exports the PyTorch models to Neuron models ahead of time to avoid multiple exports during the tests.
        We don't use unittest setUpClass, in order to still be able to run individual tests.
        """
        model_arch = model_args["model_arch"]
        model_arch_and_params = model_args["test_name"]
        dynamic_batch_size = getattr(model_args, "dynamic_batch_size", False)

        if model_arch_and_params not in self.neuron_model_dirs:
            # model_args will contain kwargs to pass to NeuronBaseModel.from_pretrained()
            model_args.pop("test_name")
            model_args.pop("model_arch")
            model_args.pop("dynamic_batch_size", None)

            model_id = (
                self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
            )
            set_seed(SEED)
            neuron_model = self.NEURON_MODEL_CLASS.from_pretrained(
                model_id, **model_args, export=True, dynamic_batch_size=dynamic_batch_size, **self.STATIC_INPUTS_SHAPES
            )

            model_dir = tempfile.mkdtemp(prefix=f"{model_arch_and_params}_{self.TASK}_")
            neuron_model.save_pretrained(model_dir)
            self.neuron_model_dirs[model_arch_and_params] = model_dir

    @classmethod
    def tearDownClass(cls):
        for _, dir_path in cls.neuron_model_dirs.items():
            shutil.rmtree(dir_path)


@is_inferentia_test
class NeuronModelIntegrationTest(unittest.TestCase):
    if is_neuron_available():
        LOCAL_MODEL_PATH = "tests/assets/neuron"
        NEURON_MODEL_ID = "optimum/tiny_random_bert_neuron"
        PRIVATE_NEURON_MODEL_ID = "Jingya/tiny-random-BertModel-neuron-private"
    elif is_neuronx_available():
        LOCAL_MODEL_PATH = "tests/assets/neuronx"
        NEURON_MODEL_ID = "optimum/tiny_random_bert_neuronx"
        PRIVATE_NEURON_MODEL_ID = "Jingya/tiny-random-BertModel-neuronx-private"

    TINY_SUBFOLDER_MODEL_ID = "fxmarty/tiny-bert-sst2-distilled-subfolder"
    FAIL_NEURON_MODEL_ID = "sshleifer/tiny-distilbert-base-cased-distilled-squad"
    TINY_MODEL_REMOTE = "Jingya/tiny-random-bert-remote-code"
    INPUTS_SHAPES = {"batch_size": 3, "sequence_length": 64}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_load_model_from_local_path(self):
        model = NeuronBaseModel.from_pretrained(self.LOCAL_MODEL_PATH)
        self.assertIsInstance(model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_hub(self):
        model = NeuronBaseModel.from_pretrained(self.NEURON_MODEL_ID)
        self.assertIsInstance(model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_hub_subfolder(self):
        model = NeuronModelForSequenceClassification.from_pretrained(
            self.TINY_SUBFOLDER_MODEL_ID, subfolder="my_subfolder", export=True, **self.INPUTS_SHAPES
        )
        self.assertIsInstance(model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_cache(self):
        _ = NeuronBaseModel.from_pretrained(self.NEURON_MODEL_ID)  # caching

        model = NeuronBaseModel.from_pretrained(self.NEURON_MODEL_ID, local_files_only=True)

        self.assertIsInstance(model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_load_model_from_empty_cache(self):
        dirpath = os.path.join(default_cache_path, "models--" + self.NEURON_MODEL_ID.replace("/", "--"))

        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)
        with self.assertRaises(Exception):
            _ = NeuronBaseModel.from_pretrained(self.NEURON_MODEL_ID, local_files_only=True)

    def test_load_model_from_hub_without_neuron_model(self):
        with self.assertRaises(FileNotFoundError):
            NeuronBaseModel.from_pretrained(self.FAIL_NEURON_MODEL_ID)

    @require_hf_token
    def test_load_model_from_hub_private(self):
        model = NeuronBaseModel.from_pretrained(
            self.PRIVATE_NEURON_MODEL_ID, use_auth_token=os.environ.get("HF_AUTH_TOKEN", None)
        )
        self.assertIsInstance(model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(model.config, PretrainedConfig)

    def test_save_model(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model = NeuronBaseModel.from_pretrained(self.LOCAL_MODEL_PATH)
            model.save_pretrained(tmpdirname)
            # folder contains all config files and neuron exported model
            folder_contents = os.listdir(tmpdirname)
            self.assertTrue(NEURON_FILE_NAME in folder_contents)
            self.assertTrue(CONFIG_NAME in folder_contents)

    @require_hf_token
    def test_push_model_to_hub(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            model = NeuronBaseModel.from_pretrained(self.LOCAL_MODEL_PATH)
            model.save_pretrained(
                tmpdirname,
                use_auth_token=os.environ.get("HF_AUTH_TOKEN", None),
                push_to_hub=True,
                repository_id=self.PRIVATE_NEURON_MODEL_ID,
                private=True,
            )

    def test_trust_remote_code(self):
        model = NeuronModelForSequenceClassification.from_pretrained(
            self.TINY_MODEL_REMOTE, export=True, trust_remote_code=True, **self.INPUTS_SHAPES
        )
        self.assertIsInstance(model.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(model.config, PretrainedConfig)


@is_inferentia_test
class NeuronModelForFeatureExtractionIntegrationTest(NeuronModelTestMixin):
    NEURON_MODEL_CLASS = NeuronModelForFeatureExtraction
    TASK = "feature-extraction"
    if is_neuron_available():
        ATOL_FOR_VALIDATION = 1e-2  # Not good enough, needs to be further investigate
        SUPPORTED_ARCHITECTURES = [
            "albert",
            "bert",
            "camembert",
            # "convbert",  # accuracy off compared to pytorch: atol=1e-1
            # "deberta",  # INF2 only
            # "deberta_v2",  # INF2 only
            # "distilbert",  # accuracy off compared to pytorch: atol=1e-1
            "electra",
            # "flaubert",  # accuracy off compared to pytorch (not due to the padding)
            "mobilebert",
            # "roberta",  # accuracy off compared to pytorch: atol=1e-1
            "roformer",
            # "xlm",  # accuracy off compared to pytorch (not due to the padding)
            "xlm-roberta",
        ]
    elif is_neuronx_available():
        ATOL_FOR_VALIDATION = 1e-3
        SUPPORTED_ARCHITECTURES = [
            "albert",
            "bert",
            "camembert",
            # "convbert",  # accuracy off compared to pytorch: atol=1e-2
            # "deberta",  # INF2 only
            # "deberta_v2",  # INF2 only
            "distilbert",
            "electra",
            # "flaubert",  # accuracy off compared to pytorch (not due to the padding)
            "mobilebert",
            "roberta",
            "roformer",
            # "xlm",  # accuracy off compared to pytorch (not due to the padding)
            "xlm-roberta",
        ]
    else:
        ATOL_FOR_VALIDATION = 1e-5
        SUPPORTED_ARCHITECTURES = []

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_compare_to_transformers_dyn_bs(self, model_arch):
        # Neuron model with dynamic batching
        model_args_dyn = {
            "test_name": model_arch + "_dyn_bs_true",
            "model_arch": model_arch,
            "dynamic_batch_size": True,
        }
        self._setup(model_args_dyn)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]

        neuron_model_dyn = NeuronModelForFeatureExtraction.from_pretrained(
            self.neuron_model_dirs[model_arch + "_dyn_bs_true"]
        )
        self.assertIsInstance(neuron_model_dyn.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(neuron_model_dyn.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModel.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        text = "This is a sample output"
        tokens = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        # Numeric validation
        neuron_outputs_dyn = neuron_model_dyn(**tokens)
        self.assertIn("last_hidden_state", neuron_outputs_dyn)
        self.assertIsInstance(neuron_outputs_dyn.last_hidden_state, torch.Tensor)
        self.assertTrue(
            torch.allclose(
                neuron_outputs_dyn.last_hidden_state,
                transformers_outputs.last_hidden_state,
                atol=self.ATOL_FOR_VALIDATION,
            )
        )

        if "pooler_output" in neuron_outputs_dyn:
            self.assertIsInstance(neuron_outputs_dyn.pooler_output, torch.Tensor)
            self.assertTrue(
                torch.allclose(
                    neuron_outputs_dyn.pooler_output, transformers_outputs.pooler_output, atol=self.ATOL_FOR_VALIDATION
                )
            )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_compare_to_transformers_non_dyn_bs(self, model_arch):
        model_args_non_dyn = {
            "test_name": model_arch + "_dyn_bs_false",
            "model_arch": model_arch,
            "dynamic_batch_size": False,
        }
        self._setup(model_args_non_dyn)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]

        neuron_model_non_dyn = NeuronModelForFeatureExtraction.from_pretrained(
            self.neuron_model_dirs[model_arch + "_dyn_bs_false"]
        )
        self.assertIsInstance(neuron_model_non_dyn.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(neuron_model_non_dyn.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModel.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        text = "This is a sample output"
        tokens = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        # Numeric validation
        neuron_outputs_non_dyn = neuron_model_non_dyn(**tokens)
        self.assertIn("last_hidden_state", neuron_outputs_non_dyn)
        self.assertIsInstance(neuron_outputs_non_dyn.last_hidden_state, torch.Tensor)
        self.assertTrue(
            torch.allclose(
                neuron_outputs_non_dyn.last_hidden_state,
                transformers_outputs.last_hidden_state,
                atol=self.ATOL_FOR_VALIDATION,
            )
        )

        if "pooler_output" in neuron_outputs_non_dyn:
            self.assertIsInstance(neuron_outputs_non_dyn.pooler_output, torch.Tensor)
            self.assertTrue(
                torch.allclose(
                    neuron_outputs_non_dyn.pooler_output,
                    transformers_outputs.pooler_output,
                    atol=self.ATOL_FOR_VALIDATION,
                )
            )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_pipeline_model(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForSequenceClassification.from_pretrained(self.neuron_model_dirs[model_arch])
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline(self.TASK, model=neuron_model, tokenizer=tokenizer)
        text = "My Name is Philipp."
        outputs = pipe(text)

        self.assertTrue(all(all(isinstance(item, float) for item in row) for row in outputs[0]))

        gc.collect()


@is_inferentia_test
class NeuronModelForMaskedLMIntegrationTest(NeuronModelTestMixin):
    NEURON_MODEL_CLASS = NeuronModelForMaskedLM
    TASK = "fill-mask"
    if is_neuron_available():
        ATOL_FOR_VALIDATION = 1e-2  # Not good enough, needs to be further investigate
        SUPPORTED_ARCHITECTURES = [
            "albert",
            "bert",
            "camembert",
            # "convbert",  # accuracy off compared to pytorch: atol=1e-1
            # "deberta",  # INF2 only
            # "deberta_v2",  # INF2 only
            # "distilbert",  # accuracy off compared to pytorch: atol=1e-1
            "electra",
            # "flaubert",  # accuracy off compared to pytorch (not due to the padding)
            "mobilebert",
            # "roberta",  # accuracy off compared to pytorch: atol=1e-1
            "roformer",
            # "xlm",  # accuracy off compared to pytorch (not due to the padding)
            "xlm-roberta",
        ]
    elif is_neuronx_available():
        ATOL_FOR_VALIDATION = 1e-3
        SUPPORTED_ARCHITECTURES = [
            "albert",
            "bert",
            "camembert",
            # "convbert",  # accuracy off compared to pytorch: atol=1e-2
            # "deberta",  # INF2 only
            # "deberta_v2",  # INF2 only
            "distilbert",
            "electra",
            # "flaubert",  # accuracy off compared to pytorch (not due to the padding)
            "mobilebert",
            "roberta",
            "roformer",
            # "xlm",  # accuracy off compared to pytorch (not due to the padding)
            "xlm-roberta",
        ]
    else:
        ATOL_FOR_VALIDATION = 1e-5
        SUPPORTED_ARCHITECTURES = []

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = NeuronModelForMaskedLM.from_pretrained(
                "hf-internal-testing/tiny-random-t5", from_transformers=True, **self.STATIC_INPUTS_SHAPES
            )

        self.assertIn("Unrecognized configuration class", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_compare_to_transformers_dyn_bs(self, model_arch):
        # Neuron model with dynamic batching
        model_args_dyn = {
            "test_name": model_arch + "_dyn_bs_true",
            "model_arch": model_arch,
            "dynamic_batch_size": True,
        }
        self._setup(model_args_dyn)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]

        neuron_model_dyn = NeuronModelForMaskedLM.from_pretrained(self.neuron_model_dirs[model_arch + "_dyn_bs_true"])
        self.assertIsInstance(neuron_model_dyn.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(neuron_model_dyn.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForMaskedLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        text = f"The capital of France is {tokenizer.mask_token}."
        tokens = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        # Numeric validation
        neuron_outputs_dyn = neuron_model_dyn(**tokens)
        self.assertIn("logits", neuron_outputs_dyn)
        self.assertIsInstance(neuron_outputs_dyn.logits, torch.Tensor)
        self.assertTrue(
            torch.allclose(neuron_outputs_dyn.logits, transformers_outputs.logits, atol=self.ATOL_FOR_VALIDATION)
        )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_compare_to_transformers_non_dyn_bs(self, model_arch):
        model_args_non_dyn = {
            "test_name": model_arch + "_dyn_bs_false",
            "model_arch": model_arch,
            "dynamic_batch_size": False,
        }
        self._setup(model_args_non_dyn)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]

        neuron_model_non_dyn = NeuronModelForMaskedLM.from_pretrained(
            self.neuron_model_dirs[model_arch + "_dyn_bs_false"]
        )
        self.assertIsInstance(neuron_model_non_dyn.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(neuron_model_non_dyn.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForMaskedLM.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        text = f"The capital of France is {tokenizer.mask_token}."
        tokens = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        # Numeric validation
        neuron_outputs_non_dyn = neuron_model_non_dyn(**tokens)
        self.assertIn("logits", neuron_outputs_non_dyn)
        self.assertIsInstance(neuron_outputs_non_dyn.logits, torch.Tensor)
        self.assertTrue(
            torch.allclose(neuron_outputs_non_dyn.logits, transformers_outputs.logits, atol=self.ATOL_FOR_VALIDATION)
        )

        gc.collect()

    def test_non_dyn_bs_neuron_model_on_false_batch_size(self):
        model_arch = "albert"
        model_args = {
            "test_name": model_arch + "_dyn_bs_false",
            "model_arch": model_arch,
            "dynamic_batch_size": False,
        }
        self._setup(model_args)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        text = (f"The capital of France is {tokenizer.mask_token}.",) * 2
        tokens = tokenizer(text, return_tensors="pt")

        neuron_model_non_dyn = NeuronModelForMaskedLM.from_pretrained(
            self.neuron_model_dirs[model_arch + "_dyn_bs_false"]
        )

        with self.assertRaises(Exception) as context:
            _ = neuron_model_non_dyn(**tokens)

        self.assertIn("set `dynamic_batch_size=True` during the compilation", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_pipeline_model(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForSequenceClassification.from_pretrained(self.neuron_model_dirs[model_arch])
        tokenizer = get_preprocessor(model_id)
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
    if is_neuron_available():
        ATOL_FOR_VALIDATION = 1e-2  # Not good enough, needs to be further investigate
        SUPPORTED_ARCHITECTURES = [
            "albert",
            "bert",
            "camembert",
            # "convbert",  # accuracy off compared to pytorch: atol=1e-1
            # "deberta",  # INF2 only
            # "deberta_v2",  # INF2 only
            # "distilbert",  # accuracy off compared to pytorch: atol=1e-1
            "electra",
            # "flaubert",  # accuracy off compared to pytorch (not due to the padding)
            "mobilebert",
            # "roberta",  # accuracy off compared to pytorch: atol=1e-1
            "roformer",
            # "xlm",  # accuracy off compared to pytorch (not due to the padding)
            "xlm-roberta",
        ]
    elif is_neuronx_available():
        ATOL_FOR_VALIDATION = 1e-3
        SUPPORTED_ARCHITECTURES = [
            "albert",
            "bert",
            "camembert",
            # "convbert",  # accuracy off compared to pytorch: atol=1e-2
            # "deberta",  # INF2 only
            # "deberta_v2",  # INF2 only
            "distilbert",
            "electra",
            # "flaubert",  # accuracy off compared to pytorch (not due to the padding)
            "mobilebert",
            "roberta",
            "roformer",
            # "xlm",  # accuracy off compared to pytorch (not due to the padding)
            "xlm-roberta",
        ]
    else:
        ATOL_FOR_VALIDATION = 1e-5
        SUPPORTED_ARCHITECTURES = []

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = NeuronModelForQuestionAnswering.from_pretrained(
                "hf-internal-testing/tiny-random-t5", from_transformers=True, **self.STATIC_INPUTS_SHAPES
            )

        self.assertIn("Unrecognized configuration class", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_compare_to_transformers_dyn_bs(self, model_arch):
        # Neuron model with dynamic batching
        model_args_dyn = {
            "test_name": model_arch + "_dyn_bs_true",
            "model_arch": model_arch,
            "dynamic_batch_size": True,
        }
        self._setup(model_args_dyn)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]

        neuron_model_dyn = NeuronModelForQuestionAnswering.from_pretrained(
            self.neuron_model_dirs[model_arch + "_dyn_bs_true"]
        )
        self.assertIsInstance(neuron_model_dyn.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(neuron_model_dyn.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForQuestionAnswering.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        text = "This is a sample output"
        tokens = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        # Numeric validation
        neuron_outputs_dyn = neuron_model_dyn(**tokens)
        self.assertIn("start_logits", neuron_outputs_dyn)
        self.assertIn("end_logits", neuron_outputs_dyn)
        self.assertIsInstance(neuron_outputs_dyn.start_logits, torch.Tensor)
        self.assertIsInstance(neuron_outputs_dyn.end_logits, torch.Tensor)

        # Compare tensor outputs
        self.assertTrue(
            torch.allclose(
                torch.Tensor(neuron_outputs_dyn.start_logits),
                transformers_outputs.start_logits,
                atol=self.ATOL_FOR_VALIDATION,
            )
        )
        self.assertTrue(
            torch.allclose(
                torch.Tensor(neuron_outputs_dyn.end_logits),
                transformers_outputs.end_logits,
                atol=self.ATOL_FOR_VALIDATION,
            )
        )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_compare_to_transformers_non_dyn_bs(self, model_arch):
        model_args_non_dyn = {
            "test_name": model_arch + "_dyn_bs_false",
            "model_arch": model_arch,
            "dynamic_batch_size": False,
        }
        self._setup(model_args_non_dyn)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]

        neuron_model_non_dyn = NeuronModelForQuestionAnswering.from_pretrained(
            self.neuron_model_dirs[model_arch + "_dyn_bs_false"]
        )
        self.assertIsInstance(neuron_model_non_dyn.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(neuron_model_non_dyn.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForQuestionAnswering.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        text = "This is a sample output"
        tokens = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        # Numeric validation
        neuron_outputs_non_dyn = neuron_model_non_dyn(**tokens)
        self.assertIn("start_logits", neuron_outputs_non_dyn)
        self.assertIn("end_logits", neuron_outputs_non_dyn)
        self.assertIsInstance(neuron_outputs_non_dyn.start_logits, torch.Tensor)
        self.assertIsInstance(neuron_outputs_non_dyn.end_logits, torch.Tensor)

        # Compare tensor outputs
        self.assertTrue(
            torch.allclose(
                torch.Tensor(neuron_outputs_non_dyn.start_logits),
                transformers_outputs.start_logits,
                atol=self.ATOL_FOR_VALIDATION,
            )
        )
        self.assertTrue(
            torch.allclose(
                torch.Tensor(neuron_outputs_non_dyn.end_logits),
                transformers_outputs.end_logits,
                atol=self.ATOL_FOR_VALIDATION,
            )
        )

        gc.collect()

    def test_non_dyn_bs_neuron_model_on_false_batch_size(self):
        model_arch = "albert"
        model_args = {
            "test_name": model_arch + "_dyn_bs_false",
            "model_arch": model_arch,
            "dynamic_batch_size": False,
        }
        self._setup(model_args)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        text = ("This is a sample output",) * 2
        tokens = tokenizer(text, return_tensors="pt")

        neuron_model_non_dyn = NeuronModelForQuestionAnswering.from_pretrained(
            self.neuron_model_dirs[model_arch + "_dyn_bs_false"]
        )

        with self.assertRaises(Exception) as context:
            _ = neuron_model_non_dyn(**tokens)

        self.assertIn("set `dynamic_batch_size=True` during the compilation", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_pipeline_model(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForQuestionAnswering.from_pretrained(self.neuron_model_dirs[model_arch])
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline(self.TASK, model=neuron_model, tokenizer=tokenizer)
        question = "Whats my name?"
        context = "My Name is Philipp."
        outputs = pipe(question, context)

        self.assertGreaterEqual(outputs["score"], 0.0)
        self.assertIsInstance(outputs["answer"], str)

        gc.collect()


@is_inferentia_test
class NeuronModelForSequenceClassificationIntegrationTest(NeuronModelTestMixin):
    NEURON_MODEL_CLASS = NeuronModelForSequenceClassification
    TASK = "text-classification"
    if is_neuron_available():
        ATOL_FOR_VALIDATION = 1e-2  # Not good enough, needs to be further investigate
        SUPPORTED_ARCHITECTURES = [
            "albert",
            "bert",
            "camembert",
            # "convbert",  # accuracy off compared to pytorch: atol=1e-1
            # "deberta",  # INF2 only
            # "deberta_v2",  # INF2 only
            # "distilbert",  # accuracy off compared to pytorch: atol=1e-1
            "electra",
            # "flaubert",  # accuracy off compared to pytorch (not due to the padding)
            "mobilebert",
            # "roberta",  # accuracy off compared to pytorch: atol=1e-1
            "roformer",
            # "xlm",  # accuracy off compared to pytorch (not due to the padding)
            "xlm-roberta",
        ]
    elif is_neuronx_available():
        ATOL_FOR_VALIDATION = 1e-3
        SUPPORTED_ARCHITECTURES = [
            "albert",
            "bert",
            "camembert",
            # "convbert",  # accuracy off compared to pytorch: atol=1e-2
            # "deberta",  # INF2 only
            # "deberta_v2",  # INF2 only
            "distilbert",
            "electra",
            # "flaubert",  # accuracy off compared to pytorch (not due to the padding)
            "mobilebert",
            "roberta",
            "roformer",
            # "xlm",  # accuracy off compared to pytorch (not due to the padding)
            "xlm-roberta",
        ]
    else:
        ATOL_FOR_VALIDATION = 1e-5
        SUPPORTED_ARCHITECTURES = []

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = NeuronModelForSequenceClassification.from_pretrained(
                "hf-internal-testing/tiny-random-t5", from_transformers=True, **self.STATIC_INPUTS_SHAPES
            )

        self.assertIn("Unrecognized configuration class", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_compare_to_transformers_dyn_bs(self, model_arch):
        # Neuron model with dynamic batching
        model_args_dyn = {
            "test_name": model_arch + "_dyn_bs_true",
            "model_arch": model_arch,
            "dynamic_batch_size": True,
        }
        self._setup(model_args_dyn)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]

        neuron_model_dyn = NeuronModelForSequenceClassification.from_pretrained(
            self.neuron_model_dirs[model_arch + "_dyn_bs_true"]
        )
        self.assertIsInstance(neuron_model_dyn.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(neuron_model_dyn.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForSequenceClassification.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        text = "This is a sample output"
        tokens = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        # Numeric validation
        neuron_outputs_dyn = neuron_model_dyn(**tokens)
        self.assertIn("logits", neuron_outputs_dyn)
        self.assertIsInstance(neuron_outputs_dyn.logits, torch.Tensor)
        self.assertTrue(
            torch.allclose(neuron_outputs_dyn.logits, transformers_outputs.logits, atol=self.ATOL_FOR_VALIDATION)
        )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_compare_to_transformers_non_dyn_bs(self, model_arch):
        model_args_non_dyn = {
            "test_name": model_arch + "_dyn_bs_false",
            "model_arch": model_arch,
            "dynamic_batch_size": False,
        }
        self._setup(model_args_non_dyn)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]

        neuron_model_non_dyn = NeuronModelForSequenceClassification.from_pretrained(
            self.neuron_model_dirs[model_arch + "_dyn_bs_false"]
        )
        self.assertIsInstance(neuron_model_non_dyn.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(neuron_model_non_dyn.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForSequenceClassification.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        text = "This is a sample output"
        tokens = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        # Numeric validation
        neuron_outputs_non_dyn = neuron_model_non_dyn(**tokens)
        self.assertIn("logits", neuron_outputs_non_dyn)
        self.assertIsInstance(neuron_outputs_non_dyn.logits, torch.Tensor)
        self.assertTrue(
            torch.allclose(neuron_outputs_non_dyn.logits, transformers_outputs.logits, atol=self.ATOL_FOR_VALIDATION)
        )

        gc.collect()

    def test_non_dyn_bs_neuron_model_on_false_batch_size(self):
        model_arch = "albert"
        model_args = {
            "test_name": model_arch + "_dyn_bs_false",
            "model_arch": model_arch,
            "dynamic_batch_size": False,
        }
        self._setup(model_args)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        text = ("This is a sample output",) * 2
        tokens = tokenizer(text, return_tensors="pt")

        neuron_model_non_dyn = NeuronModelForSequenceClassification.from_pretrained(
            self.neuron_model_dirs[model_arch + "_dyn_bs_false"]
        )

        with self.assertRaises(Exception) as context:
            _ = neuron_model_non_dyn(**tokens)

        self.assertIn("set `dynamic_batch_size=True` during the compilation", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_pipeline_model(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForSequenceClassification.from_pretrained(self.neuron_model_dirs[model_arch])
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline(self.TASK, model=neuron_model, tokenizer=tokenizer)
        text = "I like you."
        outputs = pipe(text)

        self.assertGreaterEqual(outputs[0]["score"], 0.0)
        self.assertIsInstance(outputs[0]["label"], str)

        gc.collect()


@is_inferentia_test
class NeuronModelForTokenClassificationIntegrationTest(NeuronModelTestMixin):
    NEURON_MODEL_CLASS = NeuronModelForTokenClassification
    TASK = "token-classification"
    if is_neuron_available():
        ATOL_FOR_VALIDATION = 1e-2  # Not good enough, needs to be further investigate
        SUPPORTED_ARCHITECTURES = [
            "albert",
            "bert",
            "camembert",
            # "convbert",  # accuracy off compared to pytorch: atol=1e-1
            # "deberta",  # INF2 only
            # "deberta_v2",  # INF2 only
            # "distilbert",  # accuracy off compared to pytorch: atol=1e-1
            "electra",
            # "flaubert",  # accuracy off compared to pytorch (not due to the padding)
            "mobilebert",
            # "roberta",  # accuracy off compared to pytorch: atol=1e-1
            "roformer",
            # "xlm",  # accuracy off compared to pytorch (not due to the padding)
            "xlm-roberta",
        ]
    elif is_neuronx_available():
        ATOL_FOR_VALIDATION = 1e-3
        SUPPORTED_ARCHITECTURES = [
            "albert",
            "bert",
            "camembert",
            # "convbert",  # accuracy off compared to pytorch: atol=1e-2
            # "deberta",  # INF2 only
            # "deberta_v2",  # INF2 only
            "distilbert",
            "electra",
            # "flaubert",  # accuracy off compared to pytorch (not due to the padding)
            "mobilebert",
            "roberta",
            "roformer",
            # "xlm",  # accuracy off compared to pytorch (not due to the padding)
            "xlm-roberta",
        ]
    else:
        ATOL_FOR_VALIDATION = 1e-5
        SUPPORTED_ARCHITECTURES = []

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = NeuronModelForTokenClassification.from_pretrained(
                "hf-internal-testing/tiny-random-t5", from_transformers=True, **self.STATIC_INPUTS_SHAPES
            )

        self.assertIn("Unrecognized configuration class", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_compare_to_transformers_dyn_bs(self, model_arch):
        # Neuron model with dynamic batching
        model_args_dyn = {
            "test_name": model_arch + "_dyn_bs_true",
            "model_arch": model_arch,
            "dynamic_batch_size": True,
        }
        self._setup(model_args_dyn)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]

        neuron_model_dyn = NeuronModelForTokenClassification.from_pretrained(
            self.neuron_model_dirs[model_arch + "_dyn_bs_true"]
        )
        self.assertIsInstance(neuron_model_dyn.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(neuron_model_dyn.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForTokenClassification.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        text = "This is a sample output"
        tokens = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        # Numeric validation
        neuron_outputs_dyn = neuron_model_dyn(**tokens)
        self.assertIn("logits", neuron_outputs_dyn)
        self.assertIsInstance(neuron_outputs_dyn.logits, torch.Tensor)
        self.assertTrue(
            torch.allclose(neuron_outputs_dyn.logits, transformers_outputs.logits, atol=self.ATOL_FOR_VALIDATION)
        )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_compare_to_transformers_non_dyn_bs(self, model_arch):
        model_args_non_dyn = {
            "test_name": model_arch + "_dyn_bs_false",
            "model_arch": model_arch,
            "dynamic_batch_size": False,
        }
        self._setup(model_args_non_dyn)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]

        neuron_model_non_dyn = NeuronModelForTokenClassification.from_pretrained(
            self.neuron_model_dirs[model_arch + "_dyn_bs_false"]
        )
        self.assertIsInstance(neuron_model_non_dyn.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(neuron_model_non_dyn.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForTokenClassification.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)

        text = "This is a sample output"
        tokens = tokenizer(text, return_tensors="pt")
        with torch.no_grad():
            transformers_outputs = transformers_model(**tokens)

        # Numeric validation
        neuron_outputs_non_dyn = neuron_model_non_dyn(**tokens)
        self.assertIn("logits", neuron_outputs_non_dyn)
        self.assertIsInstance(neuron_outputs_non_dyn.logits, torch.Tensor)
        self.assertTrue(
            torch.allclose(neuron_outputs_non_dyn.logits, transformers_outputs.logits, atol=self.ATOL_FOR_VALIDATION)
        )

        gc.collect()

    def test_non_dyn_bs_neuron_model_on_false_batch_size(self):
        model_arch = "albert"
        model_args = {
            "test_name": model_arch + "_dyn_bs_false",
            "model_arch": model_arch,
            "dynamic_batch_size": False,
        }
        self._setup(model_args)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        text = ("This is a sample output",) * 2
        tokens = tokenizer(text, return_tensors="pt")

        neuron_model_non_dyn = NeuronModelForTokenClassification.from_pretrained(
            self.neuron_model_dirs[model_arch + "_dyn_bs_false"]
        )

        with self.assertRaises(Exception) as context:
            _ = neuron_model_non_dyn(**tokens)

        self.assertIn("set `dynamic_batch_size=True` during the compilation", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_pipeline_model(self, model_arch):
        model_args = {"test_name": model_arch, "model_arch": model_arch}
        self._setup(model_args)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
        neuron_model = NeuronModelForTokenClassification.from_pretrained(self.neuron_model_dirs[model_arch])
        tokenizer = get_preprocessor(model_id)
        pipe = pipeline(self.TASK, model=neuron_model, tokenizer=tokenizer)
        text = "My Name is Philipp."
        outputs = pipe(text)

        self.assertTrue(all(item["score"] > 0.0 for item in outputs))

        gc.collect()


@is_inferentia_test
class NeuronModelForMultipleChoiceIntegrationTest(NeuronModelTestMixin):
    NEURON_MODEL_CLASS = NeuronModelForMultipleChoice
    TASK = "multiple-choice"
    STATIC_INPUTS_SHAPES = {"batch_size": 1, "num_choices": 4, "sequence_length": 128}
    if is_neuron_available():
        ATOL_FOR_VALIDATION = 1e-2  # Not good enough, needs to be further investigate
        SUPPORTED_ARCHITECTURES = [
            "albert",
            "bert",
            "camembert",
            # "convbert",  # accuracy off compared to pytorch: atol=1e-1
            # "deberta",  # INF2 only
            # "deberta_v2",  # INF2 only
            # "distilbert",  # accuracy off compared to pytorch: atol=1e-1
            "electra",
            # "flaubert",  # accuracy off compared to pytorch (not due to the padding)
            "mobilebert",
            # "roberta",  # accuracy off compared to pytorch: atol=1e-1
            # "roformer",
            # "xlm",  # accuracy off compared to pytorch (not due to the padding)
            # "xlm-roberta",
        ]
    elif is_neuronx_available():
        ATOL_FOR_VALIDATION = 1e-3
        SUPPORTED_ARCHITECTURES = [
            "albert",
            "bert",
            "camembert",
            # "convbert",  # accuracy off compared to pytorch: atol=1e-2
            # "deberta",  # INF2 only
            # "deberta_v2",  # INF2 only
            "distilbert",
            "electra",
            # "flaubert",  # accuracy off compared to pytorch (not due to the padding)
            "mobilebert",
            "roberta",
            # "roformer",  # accuracy off compared to pytorch: atol=1e-1
            # "xlm",  # accuracy off compared to pytorch (not due to the padding)
            # "xlm-roberta",  # Aborted (core dumped)
        ]
    else:
        ATOL_FOR_VALIDATION = 1e-5
        SUPPORTED_ARCHITECTURES = []

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_compare_to_transformers_dyn_bs(self, model_arch):
        # Neuron model with dynamic batching
        model_args_dyn = {
            "test_name": model_arch + "_dyn_bs_true",
            "model_arch": model_arch,
            "dynamic_batch_size": True,
        }
        self._setup(model_args_dyn)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]

        neuron_model_dyn = NeuronModelForMultipleChoice.from_pretrained(
            self.neuron_model_dirs[model_arch + "_dyn_bs_true"]
        )
        self.assertIsInstance(neuron_model_dyn.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(neuron_model_dyn.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForMultipleChoice.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
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

        # Numeric validation
        neuron_outputs_dyn = neuron_model_dyn(**pt_inputs)
        self.assertIn("logits", neuron_outputs_dyn)
        self.assertIsInstance(neuron_outputs_dyn.logits, torch.Tensor)
        self.assertTrue(
            torch.allclose(neuron_outputs_dyn.logits, transformers_outputs.logits, atol=self.ATOL_FOR_VALIDATION)
        )

        gc.collect()

    @parameterized.expand(SUPPORTED_ARCHITECTURES, skip_on_empty=True)
    def test_compare_to_transformers_non_dyn_bas(self, model_arch):
        model_args_non_dyn = {
            "test_name": model_arch + "_dyn_bs_false",
            "model_arch": model_arch,
            "dynamic_batch_size": False,
        }
        self._setup(model_args_non_dyn)

        model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]

        neuron_model_non_dyn = NeuronModelForMultipleChoice.from_pretrained(
            self.neuron_model_dirs[model_arch + "_dyn_bs_false"]
        )
        self.assertIsInstance(neuron_model_non_dyn.model, torch.jit._script.ScriptModule)
        self.assertIsInstance(neuron_model_non_dyn.config, PretrainedConfig)

        set_seed(SEED)
        transformers_model = AutoModelForMultipleChoice.from_pretrained(model_id)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
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

        # Numeric validation
        neuron_outputs_non_dyn = neuron_model_non_dyn(**pt_inputs)
        self.assertIn("logits", neuron_outputs_non_dyn)
        self.assertIsInstance(neuron_outputs_non_dyn.logits, torch.Tensor)
        self.assertTrue(
            torch.allclose(neuron_outputs_non_dyn.logits, transformers_outputs.logits, atol=self.ATOL_FOR_VALIDATION)
        )

        gc.collect()
