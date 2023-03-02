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
import os

from .hf_argparser import TrainiumHfArgumentParser
from .trainers import Seq2SeqTrainiumTrainer, TrainiumTrainer
from .utils.training_utils import patch_transformers_for_neuron_sdk


if not os.environ.get("DISABLE_TRANSFORMERS_PATCHING", False):
    patch_transformers_for_neuron_sdk()
