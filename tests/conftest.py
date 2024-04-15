# coding=utf-8
# Copyright 2023 The HuggingFace Team. All rights reserved.
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
import pytest


# Inferentia fixtures
ENCODER_ARCHITECTURES = [
    "albert",
    "bert",
    "camembert",
    "convbert",
    "distilbert",
    "electra",
    "flaubert",
    "mobilebert",
    "mpnet",
    "roberta",
    "roformer",
    "xlm",
    "roberta",
]
DECODER_ARCHITECTURES = ["gpt2", "llama", "mixtral"]
DIFFUSER_ARCHITECTURES = ["stable-diffusion", "stable-diffusion-xl"]

INFERENTIA_MODEL_NAMES = {
    "albert": "hf-internal-testing/tiny-random-AlbertModel",
    "bert": "hf-internal-testing/tiny-random-BertModel",
    "camembert": "hf-internal-testing/tiny-random-camembert",
    "convbert": "hf-internal-testing/tiny-random-ConvBertModel",
    # "deberta": "hf-internal-testing/tiny-random-DebertaModel",  # Failed for INF1: 'XSoftmax'
    # "deberta-v2": "hf-internal-testing/tiny-random-DebertaV2Model",  # Failed for INF1: 'XSoftmax'
    "distilbert": "hf-internal-testing/tiny-random-DistilBertModel",
    "electra": "hf-internal-testing/tiny-random-ElectraModel",
    "flaubert": "flaubert/flaubert_small_cased",
    "gpt2": "hf-internal-testing/tiny-random-gpt2",
    "llama": "dacorvo/tiny-random-llama",
    "mixtral": "dacorvo/Mixtral-tiny",
    "mobilebert": "hf-internal-testing/tiny-random-MobileBertModel",
    "mpnet": "hf-internal-testing/tiny-random-MPNetModel",
    "roberta": "hf-internal-testing/tiny-random-RobertaModel",
    "roformer": "hf-internal-testing/tiny-random-RoFormerModel",
    "stable-diffusion": "hf-internal-testing/tiny-stable-diffusion-torch",
    "stable-diffusion-xl": "echarlaix/tiny-random-stable-diffusion-xl",
    "xlm": "hf-internal-testing/tiny-random-XLMModel",
    "xlm-roberta": "hf-internal-testing/tiny-xlm-roberta",
}


@pytest.fixture(scope="module", params=[INFERENTIA_MODEL_NAMES[model_arch] for model_arch in ENCODER_ARCHITECTURES])
def inf_encoder_model(request):
    return request.param


@pytest.fixture(scope="module", params=[INFERENTIA_MODEL_NAMES[model_arch] for model_arch in DECODER_ARCHITECTURES])
def inf_decoder_model(request):
    return request.param


@pytest.fixture(scope="module", params=[INFERENTIA_MODEL_NAMES[model_arch] for model_arch in DIFFUSER_ARCHITECTURES])
def inf_diffuser_model(request):
    return request.param
