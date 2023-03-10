# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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
"""Tools that fills the neuron cache with common models for the supported tasks."""



ARCHITECTURES_TO_COMMON_PRETRAINED_WEIGHTS = {
    "albert": {
        "albert-base-v2": {
            "default": {"batch_size": 16, "sequence_length": 128}, 
            "token-classification":  {"batch_size": 2, "sequence_length": 512},
            "multiple-choice":  {"batch_size": 2, "sequence_length": 512},
        },
        "albert-large-v2": {
            "default": {"batch_size": 16, "sequence_length": 128}, 
            "token-classification":  {"batch_size": 2, "sequence_length": 512},
            "multiple-choice":  {"batch_size": 2, "sequence_length": 512},
        },
    },
    "bart": {
        "facebook/bart-base": {
            "translation": {"batch_size": 8, "source_sequence_length": 512, "target_sequence_length": 512},
            "summarization": {"batch_size": 8, "source_sequence_length": 200, "target_sequence_length": 1024},
        },
        "facebook/bart-large": {
            "translation": {"batch_size": 8, "source_sequence_length": 512, "target_sequence_length": 512},
            "summarization": {"batch_size": 8, "source_sequence_length": 200, "target_sequence_length": 1024},
        },

    },
    "bert": {
        "bert-base-uncased": {
            "default": {"batch_size": 16, "sequence_length": 128}, 
            "token-classification":  {"batch_size": 2, "sequence_length": 512},
            "multiple-choice":  {"batch_size": 2, "sequence_length": 512},
        },
        "bert-large-uncased": {
            "default": {"batch_size": 16, "sequence_length": 128}, 
            "token-classification":  {"batch_size": 2, "sequence_length": 512},
            "multiple-choice":  {"batch_size": 2, "sequence_length": 512},
        },
    },
    "camembert": {
        "camembert-base": {
            "default": {"batch_size": 16, "sequence_length": 128}, 
            "token-classification":  {"batch_size": 2, "sequence_length": 512},
            "multiple-choice":  {"batch_size": 2, "sequence_length": 512},
        },
        "camembert/camembert-large": {
            "default": {"batch_size": 16, "sequence_length": 128}, 
            "token-classification":  {"batch_size": 2, "sequence_length": 512},
            "multiple-choice":  {"batch_size": 2, "sequence_length": 512},
        },
    },
    "distilbert": {
        "distilbert-base-uncased": {
            "default": {"batch_size": 16, "sequence_length": 128}, 
            "token-classification":  {"batch_size": 2, "sequence_length": 512},
            "multiple-choice":  {"batch_size": 2, "sequence_length": 512},
        },
    },
    "electra": {
        "google/electra-small-discriminator": {
            "default": {"batch_size": 16, "sequence_length": 128}, 
            "token-classification":  {"batch_size": 2, "sequence_length": 512},
            "multiple-choice":  {"batch_size": 2, "sequence_length": 512},
        },
        "google/electra-base-discriminator": {
            "default": {"batch_size": 16, "sequence_length": 128}, 
            "token-classification":  {"batch_size": 2, "sequence_length": 512},
            "multiple-choice":  {"batch_size": 2, "sequence_length": 512},
        },
        "google/electra-large-discriminator": {
            "default": {"batch_size": 16, "sequence_length": 128}, 
            "token-classification":  {"batch_size": 2, "sequence_length": 512},
            "multiple-choice":  {"batch_size": 2, "sequence_length": 512},
        },
    },
    "gpt2": {
        "gpt2": {
            "default": {"batch_size": 16, "sequence_length": 128}, 
        },
        # "gpt2-large": {
        #     "default": {"batch_size": 16, "sequence_length": 128}, 
        # },
    },
    "gpt-neo": {
        "EleutherAI/gpt-neo-125M": {
            "default": {"batch_size": 16, "sequence_length": 128}, 
        },
    },
    "marian": {
        "Helsinki-NLP/opus-mt-en-es": {
            "translation": {"batch_size": 8, "source_sequence_length": 512, "target_sequence_length": 512},
        },
        "Helsinki-NLP/opus-mt-en-hi": {
            "translation": {"batch_size": 8, "source_sequence_length": 512, "target_sequence_length": 512},
        },
        "Helsinki-NLP/opus-mt-es-en": {
            "translation": {"batch_size": 8, "source_sequence_length": 512, "target_sequence_length": 512},
        },
    },
    "roberta": {
        "roberta-base": {
            "default": {"batch_size": 16, "sequence_length": 128}, 
            "token-classification":  {"batch_size": 2, "sequence_length": 512},
            "multiple-choice":  {"batch_size": 2, "sequence_length": 512},
        },
        "roberta-large": {
            "default": {"batch_size": 16, "sequence_length": 128}, 
            "token-classification":  {"batch_size": 2, "sequence_length": 512},
            "multiple-choice":  {"batch_size": 2, "sequence_length": 512},
        },
    },
    "t5": {
        "t5-small": {
            "translation": {"batch_size": 8, "source_sequence_length": 512, "target_sequence_length": 512},
            "summarization": {"batch_size": 8, "source_sequence_length": 200, "target_sequence_length": 1024},
        },
        "t5-base": {
            "translation": {"batch_size": 8, "source_sequence_length": 512, "target_sequence_length": 512},
            "summarization": {"batch_size": 8, "source_sequence_length": 200, "target_sequence_length": 1024},
        },

    },
    "vit": {
        "google/vit-base-patch16-224": {"default": {"batch_size": 16}},
        "google/vit-base-patch16-224-in21k": {"default": {"batch_size": 16}},
        "google/vit-large-patch16-224-in21k": {"default": {"batch_size": 8}},
            
    },
    "xlm-roberta": {
        "xlm-roberta-base": {
            "default": {"batch_size": 16, "sequence_length": 128}, 
            "token-classification":  {"batch_size": 2, "sequence_length": 512},
            "multiple-choice":  {"batch_size": 2, "sequence_length": 512},
        },
        "xlm-roberta-large": {
            "default": {"batch_size": 16, "sequence_length": 128}, 
            "token-classification":  {"batch_size": 2, "sequence_length": 512},
            "multiple-choice":  {"batch_size": 2, "sequence_length": 512},
        },
    },

}
