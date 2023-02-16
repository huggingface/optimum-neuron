# coding=utf-8
# Copyright 2023 HuggingFace Inc.
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


MODELS_TO_TEST_MAPPING = {
    "albert": "albert-base-v2",
    "bart": "facebook/bart-base",
    "bert": "bert-base-uncased",
    "camembert": "camembert-base",
    "distilbert": "distilbert-base-uncased",
    "electra": "google/electra-base-discriminator",
    "gpt2": "gpt2",
    "gpt_neo": "EleutherAI/gpt-neo-125M",
    "marian": "Helsinki-NLP/opus-mt-en-ro",
    "roberta": "roberta-base",
    "t5": "t5-small",
    "vit": "google/vit-base-patch16-224-in21k",
    "xlm-roberta": "xlm-roberta-base",
    # TODO: issue with this model for now.
    # "m2m_100": "facebook/m2m100_418M",
    # "wav2vec2": "facebook/wav2vec2-base",
    # Remaning: XLNet, Deberta-v2, MPNet, CLIP
}
