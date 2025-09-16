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

import shutil
import tempfile
import unittest
from typing import Dict

import huggingface_hub
import torch
from transformers import set_seed


SEED = 42


MODEL_NAMES = {
    "albert": "hf-internal-testing/tiny-random-AlbertModel",
    "audio-spectrogram-transformer": "Ericwang/tiny-random-ast",
    "beit": "hf-internal-testing/tiny-random-BeitForImageClassification",
    "bert": "hf-internal-testing/tiny-random-BertModel",
    "camembert": "hf-internal-testing/tiny-random-camembert",
    "convbert": "hf-internal-testing/tiny-random-ConvBertModel",
    "convnext": "hf-internal-testing/tiny-random-convnext",
    "convnextv2": "hf-internal-testing/tiny-random-ConvNextV2Model",
    "cvt": "hf-internal-testing/tiny-random-CvTModel",
    "deberta": "hf-internal-testing/tiny-random-DebertaModel",  # Failed for INF1: 'XSoftmax'
    "deberta-v2": "hf-internal-testing/tiny-random-DebertaV2Model",  # Failed for INF1: 'XSoftmax'
    "deit": "hf-internal-testing/tiny-random-DeiTModel",
    "distilbert": "hf-internal-testing/tiny-random-DistilBertModel",
    "donut-swin": "hf-internal-testing/tiny-random-DonutSwinModel",
    "dpt": "hf-internal-testing/tiny-random-DPTModel",
    "electra": "hf-internal-testing/tiny-random-ElectraModel",
    "flaubert": "flaubert/flaubert_small_cased",
    "hubert": "hf-internal-testing/tiny-random-HubertModel",
    "gpt2": "hf-internal-testing/tiny-random-gpt2",
    "levit": "hf-internal-testing/tiny-random-LevitModel",
    "mobilebert": "hf-internal-testing/tiny-random-MobileBertModel",
    "mobilenet-v2": "hf-internal-testing/tiny-random-MobileNetV2Model",
    "mobilevit": "hf-internal-testing/tiny-random-mobilevit",
    "modernbert": "hf-internal-testing/tiny-random-ModernBertModel",
    "mpnet": "hf-internal-testing/tiny-random-MPNetModel",
    "phi": "bumblebee-testing/tiny-random-PhiModel",
    "roberta": "hf-internal-testing/tiny-random-RobertaModel",
    "roformer": "hf-internal-testing/tiny-random-RoFormerModel",
    # "sew": "hf-internal-testing/tiny-random-SEWModel",  # blocked
    # "sew-d": "hf-internal-testing/tiny-random-SEWDModel",  # blocked
    "swin": "hf-internal-testing/tiny-random-SwinModel",
    "unispeech": "hf-internal-testing/tiny-random-unispeech",
    "unispeech-sat": "hf-internal-testing/tiny-random-unispeech-sat",
    "vit": "hf-internal-testing/tiny-random-vit",
    "wav2vec2": "hf-internal-testing/tiny-random-Wav2Vec2Model",
    # "wav2vec2-conformer": "hf-internal-testing/tiny-random-wav2vec2-conformer",  # blocked
    "wavlm": "hf-internal-testing/tiny-random-wavlm",
    "xlm": "hf-internal-testing/tiny-random-XLMModel",
    "xlm-roberta": "hf-internal-testing/tiny-xlm-roberta",
    "yolos": "hf-internal-testing/tiny-random-YolosModel",
}

SENTENCE_TRANSFORMERS_MODEL_NAMES = {
    "transformer": "sentence-transformers/all-MiniLM-L6-v2",
    "clip": "sentence-transformers/clip-ViT-B-32",
}


class NeuronModelIntegrationTestMixin(unittest.TestCase):
    USER = "optimum-internal-testing"
    MODEL_ID = None
    NEURON_MODEL_REPO = None
    NEURON_MODEL_CLASS = None
    STATIC_INPUTS_SHAPES = {}

    @classmethod
    def setUpClass(cls):
        cls._token = huggingface_hub.get_token()

        model_name = cls.MODEL_ID.split("/")[-1]
        model_dir = tempfile.mkdtemp(prefix=f"{model_name}_")

        # Export model to local path
        neuron_model = cls.NEURON_MODEL_CLASS.from_pretrained(cls.MODEL_ID, export=True, **cls.STATIC_INPUTS_SHAPES)
        neuron_model.save_pretrained(model_dir)
        cls.local_model_path = model_dir

        # Upload to the hub
        cls.neuron_model_id = f"{cls.USER}/{cls.NEURON_MODEL_REPO}"

        if cls._token:
            neuron_model.push_to_hub(model_dir, repository_id=cls.neuron_model_id, token=cls._token)

    @classmethod
    def tearDownClass(cls):
        if cls.local_model_path is not None:
            shutil.rmtree(cls.local_model_path)


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
        dynamic_batch_size = model_args.get("dynamic_batch_size", False)

        if model_arch_and_params not in self.neuron_model_dirs:
            # model_args will contain kwargs to pass to NeuronTracedModel.from_pretrained()
            model_args.pop("test_name")
            model_args.pop("model_arch")
            model_args.pop("dynamic_batch_size", None)

            if model_arch in self.ARCH_MODEL_MAP:
                model_id = self.ARCH_MODEL_MAP[model_arch]
            elif model_arch in SENTENCE_TRANSFORMERS_MODEL_NAMES:
                model_id = SENTENCE_TRANSFORMERS_MODEL_NAMES[model_arch]
            else:
                model_id = MODEL_NAMES[model_arch]

            set_seed(SEED)
            neuron_model = self.NEURON_MODEL_CLASS.from_pretrained(
                model_id,
                **model_args,
                export=True,
                torch_dtype=torch.float32,
                dynamic_batch_size=dynamic_batch_size,
                **self.STATIC_INPUTS_SHAPES,
            )

            model_dir = tempfile.mkdtemp(prefix=f"{model_arch_and_params}_{self.TASK}_")
            neuron_model.save_pretrained(model_dir)
            self.neuron_model_dirs[model_arch_and_params] = model_dir

    @classmethod
    def tearDownClass(cls):
        for _, dir_path in cls.neuron_model_dirs.items():
            shutil.rmtree(dir_path)
