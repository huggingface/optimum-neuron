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


EXPORT_MODELS_TINY = {
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
    # "esm": "hf-internal-testing/tiny-random-EsmModel",  # TODO: put the test back, when https://github.com/aws-neuron/aws-neuron-sdk/issues/1081 is solved.
    "flaubert": "flaubert/flaubert_small_cased",
    "hubert": "hf-internal-testing/tiny-random-HubertModel",
    "levit": "hf-internal-testing/tiny-random-LevitModel",
    "mobilebert": "hf-internal-testing/tiny-random-MobileBertModel",
    "mobilenet-v2": "hf-internal-testing/tiny-random-MobileNetV2Model",
    "mobilevit": "hf-internal-testing/tiny-random-mobilevit",
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

ENCODER_DECODER_MODELS_TINY = {
    "t5": "hf-internal-testing/tiny-random-t5",
}

STABLE_DIFFUSION_MODELS_TINY = {
    "stable-diffusion": "hf-internal-testing/tiny-stable-diffusion-torch",
    "stable-diffusion-xl": "echarlaix/tiny-random-stable-diffusion-xl",
    "latent-consistency": "echarlaix/tiny-random-latent-consistency",
}

LORA_WEIGHTS_TINY = {
    "stable-diffusion": ("Jingya/tiny-stable-diffusion-lora-64", "pytorch_lora_weights.safetensors", "pokemon"),
}

SENTENCE_TRANSFORMERS_MODELS = {
    "transformer": "sentence-transformers/all-MiniLM-L6-v2",
    "clip": "sentence-transformers/clip-ViT-B-32",
}

EXTREA_DEFAULT_DUMMY_SHAPES = {
    "text_batch_size": 1,
    "image_batch_size": 1,
}

WEIGHTS_NEFF_SEPARATION_UNSUPPORTED_ARCH = ["camembert", "roberta", "mobilenet-v2"]

SEED = 42
