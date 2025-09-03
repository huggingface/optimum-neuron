# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""Utilities related to the TRL library and support."""

from dataclasses import dataclass

from ..utils.import_utils import is_trl_available
from .training_args import NeuronTrainingArguments


TRL_VERSION = "0.11.4"


if is_trl_available():
    from trl import ORPOConfig, SFTConfig
else:

    @dataclass
    class SFTConfig:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(f"You need to install the `trl=={TRL_VERSION}` library to use the `NeuronSFTConfig`.")

    @dataclass
    class ORPOConfig:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(f"You need to install the `trl=={TRL_VERSION}` library to use the `NeuronORPOConfig`.")


@dataclass
class NeuronSFTConfig(NeuronTrainingArguments, SFTConfig):
    pass


@dataclass
class NeuronORPOConfig(NeuronTrainingArguments, ORPOConfig):
    @property
    def neuron_cc_flags_model_type(self) -> str | None:
        return None
