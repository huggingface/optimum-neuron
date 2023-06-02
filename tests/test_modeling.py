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
import os
import gc
import shutil
import tempfile
import unittest
from typing import Dict

import torch
from huggingface_hub.constants import default_cache_path
from parameterized import parameterized
from transformers import (
    AutoConfig,
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

from optimum.neuron import (
    NeuronModel,
    NeuronModelForSequenceClassification,
)
from optimum.neuron.utils import NEURON_FILE_NAME
from optimum.utils import (
    CONFIG_NAME,
    logging,
)
from optimum.utils.testing_utils import require_hf_token

from .exporters.exporters_utils import EXPORT_MODELS_TINY as MODEL_NAMES, SEED


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
            # model_args will contain kwargs to pass to NeuronModel.from_pretrained()
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


# class NeuronModelIntegrationTest(unittest.TestCase):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         self.LOCAL_MODEL_PATH = "assets/neuron"
#         self.NEURON_MODEL_ID = "optimum/tiny-random-BertModel-neuron"
#         self.TINY_SUBFOLDER_MODEL_ID = "fxmarty/tiny-bert-sst2-distilled-subfolder"
#         self.FAIL_NEURON_MODEL_ID = "sshleifer/tiny-distilbert-base-cased-distilled-squad"
#         self.PRIVATE_NEURON_MODEL_ID = "Jingya/tiny-random-BertModel-neuron-private"
#         self.TINY_MODEL_REMOTE = "Jingya/tiny-random-bert-remote-code"
#         self.INPUTS_SHAPES = {"batch_size": 3, "sequence_length": 64}

#     def test_load_model_from_local_path(self):
#         model = NeuronModel.from_pretrained(self.LOCAL_MODEL_PATH)
#         self.assertIsInstance(model.model, torch.jit._script.ScriptModule)
#         self.assertIsInstance(model.config, PretrainedConfig)

#     def test_load_model_from_hub(self):
#         model = NeuronModel.from_pretrained(self.NEURON_MODEL_ID)
#         self.assertIsInstance(model.model, torch.jit._script.ScriptModule)
#         self.assertIsInstance(model.config, PretrainedConfig)

#     def test_load_model_from_hub_subfolder(self):
#         model = NeuronModelForSequenceClassification.from_pretrained(
#             self.TINY_SUBFOLDER_MODEL_ID, subfolder="my_subfolder", export=True, **self.INPUTS_SHAPES
#         )
#         self.assertIsInstance(model.model, torch.jit._script.ScriptModule)
#         self.assertIsInstance(model.config, PretrainedConfig)

#     def test_load_model_from_cache(self):
#         _ = NeuronModel.from_pretrained(self.NEURON_MODEL_ID)  # caching

#         model = NeuronModel.from_pretrained(self.NEURON_MODEL_ID, local_files_only=True)

#         self.assertIsInstance(model.model, torch.jit._script.ScriptModule)
#         self.assertIsInstance(model.config, PretrainedConfig)

#     def test_load_model_from_empty_cache(self):
#         dirpath = os.path.join(default_cache_path, "models--" + self.NEURON_MODEL_ID.replace("/", "--"))

#         if os.path.exists(dirpath) and os.path.isdir(dirpath):
#             shutil.rmtree(dirpath)
#         with self.assertRaises(Exception):
#             _ = NeuronModel.from_pretrained(self.NEURON_MODEL_ID, local_files_only=True)

#     def test_load_model_from_hub_without_neuron_model(self):
#         with self.assertRaises(FileNotFoundError):
#             NeuronModel.from_pretrained(self.FAIL_NEURON_MODEL_ID)

#     @require_hf_token
#     def test_load_model_from_hub_private(self):
#         model = NeuronModel.from_pretrained(
#             self.PRIVATE_NEURON_MODEL_ID, use_auth_token=os.environ.get("HF_AUTH_TOKEN", None)
#         )
#         self.assertIsInstance(model.model, torch.jit._script.ScriptModule)
#         self.assertIsInstance(model.config, PretrainedConfig)

#     def test_save_model(self):
#         with tempfile.TemporaryDirectory() as tmpdirname:
#             model = NeuronModel.from_pretrained(self.LOCAL_MODEL_PATH)
#             model.save_pretrained(tmpdirname)
#             # folder contains all config files and neuron exported model
#             folder_contents = os.listdir(tmpdirname)
#             self.assertTrue(NEURON_FILE_NAME in folder_contents)
#             self.assertTrue(CONFIG_NAME in folder_contents)

#     @require_hf_token
#     def test_push_model_to_hub(self):
#         with tempfile.TemporaryDirectory() as tmpdirname:
#             model = NeuronModel.from_pretrained(self.LOCAL_MODEL_PATH)
#             model.save_pretrained(
#                 tmpdirname,
#                 use_auth_token=os.environ.get("HF_AUTH_TOKEN", None),
#                 push_to_hub=True,
#                 repository_id=self.PRIVATE_NEURON_MODEL_ID,
#                 private=True,
#             )

#     def test_trust_remote_code(self):
#         model = NeuronModelForSequenceClassification.from_pretrained(
#             self.TINY_MODEL_REMOTE, export=True, trust_remote_code=True, **self.INPUTS_SHAPES
#         )
#         self.assertIsInstance(model.model, torch.jit._script.ScriptModule)
#         self.assertIsInstance(model.config, PretrainedConfig)


# class NeuronModelForQuestionAnsweringIntegrationTest(NeuronModelTestMixin):
#     SUPPORTED_ARCHITECTURES = [
#         "albert",
#         "bart",
#         "bert",
#         # "big_bird",
#         # "bigbird_pegasus",
#         "camembert",
#         "convbert",
#         "data2vec_text",
#         "deberta",
#         "deberta_v2",
#         "distilbert",
#         "electra",
#         "flaubert",
#         "gptj",
#         "ibert",
#         # TODO: these two should be supported, but require image inputs not supported in NeuronModel
#         # "layoutlm"
#         # "layoutlmv3",
#         "mbart",
#         "mobilebert",
#         "nystromformer",
#         "roberta",
#         "roformer",
#         "squeezebert",
#         "xlm",
#         "xlm_roberta",
#     ]

#     FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
#     NeuronModel_CLASS = NeuronModelForQuestionAnswering
#     TASK = "question-answering"

#     def test_load_vanilla_transformers_which_is_not_supported(self):
#         with self.assertRaises(Exception) as context:
#             _ = NeuronModelForQuestionAnswering.from_pretrained(MODEL_NAMES["t5"], from_transformers=True)

#         self.assertIn("Unrecognized configuration class", str(context.exception))

#     @parameterized.expand(SUPPORTED_ARCHITECTURES)
#     def test_compare_to_transformers(self, model_arch):
#         model_args = {"test_name": model_arch, "model_arch": model_arch}
#         self._setup(model_args)

#         model_id = MODEL_NAMES[model_arch]
#         neuron_model = NeuronModelForQuestionAnswering.from_pretrained(self.neuron_model_dirs[model_arch])

#         self.assertIsInstance(neuron_model.model, torch.jit._script.ScriptModule)
#         self.assertIsInstance(neuron_model.config, PretrainedConfig)

#         set_seed(SEED)
#         transformers_model = AutoModelForQuestionAnswering.from_pretrained(model_id)
#         tokenizer = get_preprocessor(model_id)

#         tokens = tokenizer("This is a sample output", return_tensors="pt")
#         with torch.no_grad():
#             transformers_outputs = transformers_model(**tokens)

#         for input_type in ["pt", "np"]:
#             tokens = tokenizer("This is a sample output", return_tensors=input_type)
#             neuron_outputs = neuron_model(**tokens)

#             self.assertIn("start_logits", neuron_outputs)
#             self.assertIn("end_logits", neuron_outputs)
#             self.assertIsInstance(neuron_outputs.start_logits, self.TENSOR_ALIAS_TO_TYPE[input_type])
#             self.assertIsInstance(neuron_outputs.end_logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

#             # Compare tensor outputs
#             self.assertTrue(
#                 torch.allclose(torch.Tensor(neuron_outputs.start_logits), transformers_outputs.start_logits, atol=1e-4)
#             )
#             self.assertTrue(
#                 torch.allclose(torch.Tensor(neuron_outputs.end_logits), transformers_outputs.end_logits, atol=1e-4)
#             )

#         gc.collect()

#     @parameterized.expand(SUPPORTED_ARCHITECTURES)
#     def test_pipeline_neuron_model(self, model_arch):
#         model_args = {"test_name": model_arch, "model_arch": model_arch}
#         self._setup(model_args)

#         model_id = MODEL_NAMES[model_arch]
#         neuron_model = NeuronModelForQuestionAnswering.from_pretrained(self.neuron_model_dirs[model_arch])
#         tokenizer = get_preprocessor(model_id)
#         pipe = pipeline("question-answering", model=neuron_model, tokenizer=tokenizer)
#         question = "Whats my name?"
#         context = "My Name is Philipp and I live in Nuremberg."
#         outputs = pipe(question, context)

#         self.assertEqual(pipe.device, pipe.model.device)
#         self.assertGreaterEqual(outputs["score"], 0.0)
#         self.assertIsInstance(outputs["answer"], str)

#         gc.collect()

#     @pytest.mark.run_in_series
#     def test_pipeline_model_is_none(self):
#         pipe = pipeline("question-answering")
#         question = "Whats my name?"
#         context = "My Name is Philipp and I live in Nuremberg."
#         outputs = pipe(question, context)

#         # compare model output class
#         self.assertGreaterEqual(outputs["score"], 0.0)
#         self.assertIsInstance(outputs["answer"], str)


# class NeuronModelForMaskedLMIntegrationTest(NeuronModelTestMixin):
#     SUPPORTED_ARCHITECTURES = [
#         "albert",
#         "bert",
#         "camembert",
#         # "convbert",
#         # "deberta",
#         # "deberta_v2",
#         "distilbert",
#         "electra",
#         "flaubert",
#         "mobilebert",
#         "mpnet",
#         "roberta",
#         "roformer",
#         "xlm",
#         "xlm_roberta",
#     ]

#     FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
#     NeuronModel_CLASS = NeuronModelForMaskedLM
#     TASK = "fill-mask"

#     def test_load_vanilla_transformers_which_is_not_supported(self):
#         with self.assertRaises(Exception) as context:
#             _ = NeuronModelForMaskedLM.from_pretrained(MODEL_NAMES["t5"], from_transformers=True)

#         self.assertIn("Unrecognized configuration class", str(context.exception))

#     @parameterized.expand(SUPPORTED_ARCHITECTURES)
#     def test_compare_to_transformers(self, model_arch):
#         model_args = {"test_name": model_arch, "model_arch": model_arch}
#         self._setup(model_args)

#         model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
#         neuron_model = NeuronModelForMaskedLM.from_pretrained(self.neuron_model_dirs[model_arch])

#         self.assertIsInstance(neuron_model.model, torch.jit._script.ScriptModule)
#         self.assertIsInstance(neuron_model.config, PretrainedConfig)

#         set_seed(SEED)
#         transformers_model = AutoModelForMaskedLM.from_pretrained(model_id)
#         tokenizer = get_preprocessor(model_id)

#         text = f"The capital of France is {tokenizer.mask_token}."
#         tokens = tokenizer(text, return_tensors="pt")
#         with torch.no_grad():
#             transformers_outputs = transformers_model(**tokens)

#         for input_type in ["pt", "np"]:
#             tokens = tokenizer(text, return_tensors=input_type)
#             neuron_outputs = neuron_model(**tokens)

#             self.assertIn("logits", neuron_outputs)
#             self.assertIsInstance(neuron_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

#             # compare tensor outputs
#             self.assertTrue(
#                 torch.allclose(torch.Tensor(neuron_outputs.logits), transformers_outputs.logits, atol=1e-4)
#             )

#         gc.collect()

#     @parameterized.expand(SUPPORTED_ARCHITECTURES)
#     def test_pipeline_ort_model(self, model_arch):
#         model_args = {"test_name": model_arch, "model_arch": model_arch}
#         self._setup(model_args)

#         model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
#         neuron_model = NeuronModelForMaskedLM.from_pretrained(self.neuron_model_dirs[model_arch])
#         tokenizer = get_preprocessor(model_id)
#         pipe = pipeline("fill-mask", model=neuron_model, tokenizer=tokenizer)
#         MASK_TOKEN = tokenizer.mask_token
#         text = f"The capital of France is {MASK_TOKEN}."
#         outputs = pipe(text)

#         self.assertEqual(pipe.device, neuron_model.device)
#         self.assertGreaterEqual(outputs[0]["score"], 0.0)
#         self.assertIsInstance(outputs[0]["token_str"], str)

#         gc.collect()

#     @pytest.mark.run_in_series
#     def test_pipeline_model_is_none(self):
#         pipe = pipeline("fill-mask")
#         text = "The capital of France is [MASK]."
#         outputs = pipe(text)

#         # compare model output class
#         self.assertGreaterEqual(outputs[0]["score"], 0.0)
#         self.assertIsInstance(outputs[0]["token_str"], str)

#     @parameterized.expand(SUPPORTED_ARCHITECTURES)
#     @require_torch_gpu
#     @pytest.mark.gpu_test
#     def test_pipeline_on_gpu(self, model_arch):
#         model_args = {"test_name": model_arch, "model_arch": model_arch}
#         self._setup(model_args)

#         model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
#         neuron_model = NeuronModelForMaskedLM.from_pretrained(self.neuron_model_dirs[model_arch])
#         tokenizer = get_preprocessor(model_id)
#         MASK_TOKEN = tokenizer.mask_token
#         pipe = pipeline("fill-mask", model=neuron_model, tokenizer=tokenizer, device=0)
#         text = f"The capital of France is {MASK_TOKEN}."
#         outputs = pipe(text)
#         # check model device
#         self.assertEqual(pipe.model.device.type.lower(), "cuda")
#         # compare model output class
#         self.assertGreaterEqual(outputs[0]["score"], 0.0)
#         self.assertTrue(isinstance(outputs[0]["token_str"], str))

#         gc.collect()

#     @parameterized.expand(SUPPORTED_ARCHITECTURES)
#     @require_torch_gpu
#     @pytest.mark.gpu_test
#     def test_compare_to_io_binding(self, model_arch):
#         model_args = {"test_name": model_arch, "model_arch": model_arch}
#         self._setup(model_args)

#         model_id = self.ARCH_MODEL_MAP[model_arch] if model_arch in self.ARCH_MODEL_MAP else MODEL_NAMES[model_arch]
#         neuron_model = NeuronModelForMaskedLM.from_pretrained(
#             self.neuron_model_dirs[model_arch], use_io_binding=False
#         ).to("cuda")
#         io_model = NeuronModelForMaskedLM.from_pretrained(self.neuron_model_dirs[model_arch], use_io_binding=True).to(
#             "cuda"
#         )

#         tokenizer = get_preprocessor(model_id)
#         MASK_TOKEN = tokenizer.mask_token
#         tokens = tokenizer([f"The capital of France is {MASK_TOKEN}."] * 2, return_tensors="pt")
#         neuron_outputs = neuron_model(**tokens)
#         io_outputs = io_model(**tokens)

#         self.assertTrue("logits" in io_outputs)
#         self.assertIsInstance(io_outputs.logits, torch.Tensor)

#         # compare tensor outputs
#         self.assertTrue(torch.equal(neuron_outputs.logits, io_outputs.logits))

#         gc.collect()


class NeuronModelForSequenceClassificationIntegrationTest(NeuronModelTestMixin):
    SUPPORTED_ARCHITECTURES = [
        "albert",
        "bert",
        "camembert",
        "convbert",
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

    NEURON_MODEL_CLASS = NeuronModelForSequenceClassification
    TASK = "text-classification"

    def test_load_vanilla_transformers_which_is_not_supported(self):
        with self.assertRaises(Exception) as context:
            _ = NeuronModelForSequenceClassification.from_pretrained("hf-internal-testing/tiny-random-t5", from_transformers=True, **self.STATIC_INPUTS_SHAPES)

        self.assertIn("Unrecognized configuration class", str(context.exception))

    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers_dyn_bas(self, model_arch):
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
        self.assertTrue(torch.allclose(neuron_outputs_dyn.logits, transformers_outputs.logits, atol=1e-3))

        gc.collect()
    
    @parameterized.expand(SUPPORTED_ARCHITECTURES)
    def test_compare_to_transformers_non_dyn_bas(self, model_arch):
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
        self.assertTrue(torch.allclose(neuron_outputs_non_dyn.logits, transformers_outputs.logits, atol=1e-3))

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
        text = ("This is a sample output", )*2
        tokens = tokenizer(text, return_tensors="pt")

        neuron_model_non_dyn = NeuronModelForSequenceClassification.from_pretrained(
            self.neuron_model_dirs[model_arch + "_dyn_bs_false"]
        )

        with self.assertRaises(Exception) as context:
            _ = neuron_model_non_dyn(**tokens)

        self.assertIn("set `dynamic_batch_size=True` during the compilation", str(context.exception))


# class NeuronModelForTokenClassificationIntegrationTest(NeuronModelTestMixin):
#     SUPPORTED_ARCHITECTURES = [
#         "albert",
#         "bert",
#         # "big_bird",
#         "bloom",
#         "camembert",
#         "convbert",
#         "data2vec_text",
#         "deberta",
#         "deberta_v2",
#         "distilbert",
#         "electra",
#         "flaubert",
#         "gpt2",
#         "ibert",
#         # TODO: these two should be supported, but require image inputs not supported in NeuronModel
#         # "layoutlm"
#         # "layoutlmv3",
#         "mobilebert",
#         "roberta",
#         "roformer",
#         "squeezebert",
#         "xlm",
#         "xlm_roberta",
#     ]

#     FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
#     NeuronModel_CLASS = NeuronModelForTokenClassification
#     TASK = "token-classification"

#     def test_load_vanilla_transformers_which_is_not_supported(self):
#         with self.assertRaises(Exception) as context:
#             _ = NeuronModelForTokenClassification.from_pretrained(MODEL_NAMES["t5"], from_transformers=True)

#         self.assertIn("Unrecognized configuration class", str(context.exception))

#     @parameterized.expand(SUPPORTED_ARCHITECTURES)
#     def test_compare_to_transformers(self, model_arch):
#         model_args = {"test_name": model_arch, "model_arch": model_arch}
#         self._setup(model_args)

#         model_id = MODEL_NAMES[model_arch]
#         neuron_model = NeuronModelForTokenClassification.from_pretrained(self.neuron_model_dirs[model_arch])

#         self.assertIsInstance(neuron_model.model, torch.jit._script.ScriptModule)
#         self.assertIsInstance(neuron_model.config, PretrainedConfig)

#         set_seed(SEED)
#         transformers_model = AutoModelForTokenClassification.from_pretrained(model_id)
#         tokenizer = get_preprocessor(model_id)

#         text = "This is a sample output"
#         tokens = tokenizer(text, return_tensors="pt")
#         with torch.no_grad():
#             transformers_outputs = transformers_model(**tokens)

#         for input_type in ["pt", "np"]:
#             tokens = tokenizer(text, return_tensors=input_type)
#             neuron_outputs = neuron_model(**tokens)

#             self.assertIn("logits", neuron_outputs)
#             self.assertIsInstance(neuron_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

#             # compare tensor outputs
#             self.assertTrue(
#                 torch.allclose(torch.Tensor(neuron_outputs.logits), transformers_outputs.logits, atol=1e-4)
#             )

#         gc.collect()

#     @parameterized.expand(SUPPORTED_ARCHITECTURES)
#     def test_pipeline_ort_model(self, model_arch):
#         model_args = {"test_name": model_arch, "model_arch": model_arch}
#         self._setup(model_args)

#         model_id = MODEL_NAMES[model_arch]
#         neuron_model = NeuronModelForTokenClassification.from_pretrained(self.neuron_model_dirs[model_arch])
#         tokenizer = get_preprocessor(model_id)
#         pipe = pipeline("token-classification", model=neuron_model, tokenizer=tokenizer)
#         text = "My Name is Philipp and i live in Germany."
#         outputs = pipe(text)

#         self.assertEqual(pipe.device, neuron_model.device)
#         self.assertTrue(all(item["score"] > 0.0 for item in outputs))

#         gc.collect()

#     @pytest.mark.run_in_series
#     def test_pipeline_model_is_none(self):
#         pipe = pipeline("token-classification")
#         text = "My Name is Philipp and i live in Germany."
#         outputs = pipe(text)

#         # compare model output class
#         self.assertTrue(all(item["score"] > 0.0 for item in outputs))

#     @parameterized.expand(
#         grid_parameters(
#             {"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["CUDAExecutionProvider", "TensorrtExecutionProvider"]}
#         )
#     )
#     @require_torch_gpu
#     @pytest.mark.gpu_test
#     def test_pipeline_on_gpu(self, test_name: str, model_arch: str, provider: str):
#         if provider == "TensorrtExecutionProvider" and model_arch != self.__class__.SUPPORTED_ARCHITECTURES[0]:
#             self.skipTest("testing a single arch for TensorrtExecutionProvider")

#         model_args = {"test_name": model_arch, "model_arch": model_arch}
#         self._setup(model_args)

#         model_id = MODEL_NAMES[model_arch]
#         neuron_model = NeuronModelForTokenClassification.from_pretrained(
#             self.neuron_model_dirs[model_arch], provider=provider
#         )
#         tokenizer = get_preprocessor(model_id)
#         pipe = pipeline("token-classification", model=neuron_model, tokenizer=tokenizer, device=0)
#         text = "My Name is Philipp and i live in Germany."
#         outputs = pipe(text)
#         # check model device
#         self.assertEqual(pipe.model.device.type.lower(), "cuda")
#         # compare model output class
#         self.assertTrue(all(item["score"] > 0.0 for item in outputs))

#         gc.collect()

#     @parameterized.expand(SUPPORTED_ARCHITECTURES)
#     @require_torch_gpu
#     @pytest.mark.gpu_test
#     def test_compare_to_io_binding(self, model_arch):
#         model_args = {"test_name": model_arch, "model_arch": model_arch}
#         self._setup(model_args)

#         model_id = MODEL_NAMES[model_arch]
#         neuron_model = NeuronModelForTokenClassification.from_pretrained(
#             self.neuron_model_dirs[model_arch], use_io_binding=False
#         ).to("cuda")
#         io_model = NeuronModelForTokenClassification.from_pretrained(
#             self.neuron_model_dirs[model_arch], use_io_binding=True
#         ).to("cuda")

#         tokenizer = get_preprocessor(model_id)
#         tokens = tokenizer(["This is a sample output"] * 2, return_tensors="pt")
#         neuron_outputs = neuron_model(**tokens)
#         io_outputs = io_model(**tokens)

#         self.assertTrue("logits" in io_outputs)
#         self.assertIsInstance(io_outputs.logits, torch.Tensor)

#         # compare tensor outputs
#         self.assertTrue(torch.equal(neuron_outputs.logits, io_outputs.logits))

#         gc.collect()


# class NeuronModelForFeatureExtractionIntegrationTest(NeuronModelTestMixin):
#     SUPPORTED_ARCHITECTURES = ["albert", "bert", "camembert", "distilbert", "electra", "roberta", "xlm_roberta"]

#     FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
#     NeuronModel_CLASS = NeuronModelForFeatureExtraction
#     TASK = "feature-extraction"

#     @parameterized.expand(SUPPORTED_ARCHITECTURES)
#     def test_compare_to_transformers(self, model_arch):
#         model_args = {"test_name": model_arch, "model_arch": model_arch}
#         self._setup(model_args)

#         model_id = MODEL_NAMES[model_arch]
#         neuron_model = NeuronModelForFeatureExtraction.from_pretrained(self.neuron_model_dirs[model_arch])

#         self.assertIsInstance(neuron_model.model, torch.jit._script.ScriptModule)
#         self.assertIsInstance(neuron_model.config, PretrainedConfig)

#         set_seed(SEED)
#         transformers_model = AutoModel.from_pretrained(model_id)
#         tokenizer = get_preprocessor(model_id)
#         text = "This is a sample output"
#         tokens = tokenizer(text, return_tensors="pt")
#         with torch.no_grad():
#             transformers_outputs = transformers_model(**tokens)

#         for input_type in ["pt", "np"]:
#             tokens = tokenizer(text, return_tensors=input_type)
#             neuron_outputs = neuron_model(**tokens)

#             self.assertIn("last_hidden_state", neuron_outputs)
#             self.assertIsInstance(neuron_outputs.last_hidden_state, self.TENSOR_ALIAS_TO_TYPE[input_type])

#             # compare tensor outputs
#             self.assertTrue(
#                 torch.allclose(
#                     torch.Tensor(neuron_outputs.last_hidden_state), transformers_outputs.last_hidden_state, atol=1e-4
#                 )
#             )

#         gc.collect()

#     @parameterized.expand(SUPPORTED_ARCHITECTURES)
#     def test_pipeline_ort_model(self, model_arch):
#         model_args = {"test_name": model_arch, "model_arch": model_arch}
#         self._setup(model_args)

#         model_id = MODEL_NAMES[model_arch]
#         neuron_model = NeuronModelForFeatureExtraction.from_pretrained(self.neuron_model_dirs[model_arch])
#         tokenizer = get_preprocessor(model_id)
#         pipe = pipeline("feature-extraction", model=neuron_model, tokenizer=tokenizer)
#         text = "My Name is Philipp and i live in Germany."
#         outputs = pipe(text)

#         # compare model output class
#         self.assertEqual(pipe.device, neuron_model.device)
#         self.assertTrue(all(all(isinstance(item, float) for item in row) for row in outputs[0]))

#         gc.collect()

#     @pytest.mark.run_in_series
#     def test_pipeline_model_is_none(self):
#         pipe = pipeline("feature-extraction")
#         text = "My Name is Philipp and i live in Germany."
#         outputs = pipe(text)

#         # compare model output class
#         self.assertTrue(all(all(isinstance(item, float) for item in row) for row in outputs[0]))

#     @parameterized.expand(
#         grid_parameters(
#             {"model_arch": SUPPORTED_ARCHITECTURES, "provider": ["CUDAExecutionProvider", "TensorrtExecutionProvider"]}
#         )
#     )
#     @require_torch_gpu
#     @pytest.mark.gpu_test
#     def test_pipeline_on_gpu(self, test_name: str, model_arch: str, provider: str):
#         if provider == "TensorrtExecutionProvider" and model_arch != self.__class__.SUPPORTED_ARCHITECTURES[0]:
#             self.skipTest("testing a single arch for TensorrtExecutionProvider")

#         model_args = {"test_name": model_arch, "model_arch": model_arch}
#         self._setup(model_args)

#         model_id = MODEL_NAMES[model_arch]
#         neuron_model = NeuronModelForFeatureExtraction.from_pretrained(
#             self.neuron_model_dirs[model_arch], provider=provider
#         )
#         tokenizer = get_preprocessor(model_id)
#         pipe = pipeline("feature-extraction", model=neuron_model, tokenizer=tokenizer, device=0)
#         text = "My Name is Philipp and i live in Germany."
#         outputs = pipe(text)
#         # check model device
#         self.assertEqual(pipe.model.device.type.lower(), "cuda")
#         # compare model output class
#         self.assertTrue(all(all(isinstance(item, float) for item in row) for row in outputs[0]))

#         gc.collect()

#     @parameterized.expand(SUPPORTED_ARCHITECTURES)
#     @require_torch_gpu
#     @pytest.mark.gpu_test
#     def test_compare_to_io_binding(self, model_arch):
#         model_args = {"test_name": model_arch, "model_arch": model_arch}
#         self._setup(model_args)

#         model_id = MODEL_NAMES[model_arch]
#         neuron_model = NeuronModelForFeatureExtraction.from_pretrained(
#             self.neuron_model_dirs[model_arch], use_io_binding=False
#         ).to("cuda")
#         io_model = NeuronModelForFeatureExtraction.from_pretrained(
#             self.neuron_model_dirs[model_arch], use_io_binding=True
#         ).to("cuda")

#         tokenizer = get_preprocessor(model_id)
#         tokens = tokenizer(["This is a sample output"] * 2, return_tensors="pt")
#         neuron_outputs = neuron_model(**tokens)
#         io_outputs = io_model(**tokens)

#         self.assertTrue("last_hidden_state" in io_outputs)
#         self.assertIsInstance(io_outputs.last_hidden_state, torch.Tensor)

#         # compare tensor outputs
#         self.assertTrue(torch.equal(neuron_outputs.last_hidden_state, io_outputs.last_hidden_state))

#         gc.collect()


# class NeuronModelForMultipleChoiceIntegrationTest(NeuronModelTestMixin):
#     # Multiple Choice tests are conducted on different models due to mismatch size in model's classifier
#     SUPPORTED_ARCHITECTURES = [
#         "albert",
#         "bert",
#         # "big_bird",
#         "camembert",
#         "convbert",
#         "data2vec_text",
#         "deberta_v2",
#         "distilbert",
#         "electra",
#         "flaubert",
#         "ibert",
#         "mobilebert",
#         "nystromformer",
#         "roberta",
#         "roformer",
#         "squeezebert",
#         "xlm",
#         "xlm_roberta",
#     ]

#     FULL_GRID = {"model_arch": SUPPORTED_ARCHITECTURES}
#     NeuronModel_CLASS = NeuronModelForMultipleChoice
#     TASK = "multiple-choice"

#     @parameterized.expand(SUPPORTED_ARCHITECTURES)
#     def test_compare_to_transformers(self, model_arch):
#         model_args = {"test_name": model_arch, "model_arch": model_arch}
#         self._setup(model_args)

#         model_id = MODEL_NAMES[model_arch]
#         neuron_model = NeuronModelForMultipleChoice.from_pretrained(self.neuron_model_dirs[model_arch])

#         self.assertIsInstance(neuron_model.model, torch.jit._script.ScriptModule)
#         self.assertIsInstance(neuron_model.config, PretrainedConfig)

#         set_seed(SEED)
#         transformers_model = AutoModelForMultipleChoice.from_pretrained(model_id)
#         tokenizer = get_preprocessor(model_id)
#         num_choices = 4
#         first_sentence = ["The sky is blue due to the shorter wavelength of blue light."] * num_choices
#         start = "The color of the sky is"
#         second_sentence = [start + "blue", start + "green", start + "red", start + "yellow"]
#         inputs = tokenizer(first_sentence, second_sentence, truncation=True, padding=True)

#         # Unflatten the tokenized inputs values expanding it to the shape [batch_size, num_choices, seq_length]
#         for k, v in inputs.items():
#             inputs[k] = [v[i : i + num_choices] for i in range(0, len(v), num_choices)]

#         pt_inputs = dict(inputs.convert_to_tensors(tensor_type="pt"))
#         with torch.no_grad():
#             transformers_outputs = transformers_model(**pt_inputs)

#         for input_type in ["pt", "np"]:
#             inps = dict(inputs.convert_to_tensors(tensor_type=input_type))
#             neuron_outputs = neuron_model(**inps)

#             self.assertTrue("logits" in neuron_outputs)
#             self.assertIsInstance(neuron_outputs.logits, self.TENSOR_ALIAS_TO_TYPE[input_type])

#             # Compare tensor outputs
#             self.assertTrue(
#                 torch.allclose(torch.Tensor(neuron_outputs.logits), transformers_outputs.logits, atol=1e-4)
#             )

#         gc.collect()

#     @parameterized.expand(SUPPORTED_ARCHITECTURES)
#     @require_torch_gpu
#     @pytest.mark.gpu_test
#     def test_compare_to_io_binding(self, model_arch):
#         model_args = {"test_name": model_arch, "model_arch": model_arch}
#         self._setup(model_args)

#         model_id = MODEL_NAMES[model_arch]
#         neuron_model = NeuronModelForMultipleChoice.from_pretrained(
#             self.neuron_model_dirs[model_arch], use_io_binding=False
#         ).to("cuda")
#         io_model = NeuronModelForMultipleChoice.from_pretrained(
#             self.neuron_model_dirs[model_arch], use_io_binding=True
#         ).to("cuda")

#         tokenizer = get_preprocessor(model_id)
#         num_choices = 4
#         first_sentence = ["The sky is blue due to the shorter wavelength of blue light."] * num_choices
#         start = "The color of the sky is"
#         second_sentence = [start + "blue", start + "green", start + "red", start + "yellow"]
#         inputs = tokenizer(first_sentence, second_sentence, truncation=True, padding=True)

#         # Unflatten the tokenized inputs values expanding it to the shape [batch_size, num_choices, seq_length]
#         for k, v in inputs.items():
#             inputs[k] = [v[i : i + num_choices] for i in range(0, len(v), num_choices)]

#         inputs = dict(inputs.convert_to_tensors(tensor_type="pt"))

#         neuron_outputs = neuron_model(**inputs)
#         io_outputs = io_model(**inputs)

#         self.assertTrue("logits" in io_outputs)
#         self.assertIsInstance(io_outputs.logits, torch.Tensor)

#         # compare tensor outputs
#         self.assertTrue(torch.equal(neuron_outputs.logits, io_outputs.logits))

#         gc.collect()
