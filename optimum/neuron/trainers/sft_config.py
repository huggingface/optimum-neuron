# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from dataclasses import dataclass

from ..utils.import_utils import is_trl_available
from .training_args import NeuronTrainingArguments
from .trl_utils import TRL_VERSION


if is_trl_available():
    from trl import SFTConfig
else:

    @dataclass
    class SFTConfig:
        def __init__(self, *args, **kwargs):
            raise RuntimeError(f"You need to install the `trl=={TRL_VERSION}` library to use the `NeuronSFTConfig`.")


@dataclass
class NeuronSFTConfig(NeuronTrainingArguments, SFTConfig):
    def __post_init__(self):
        # Handle max_seq_length -> max_length migration for backward compatibility
        if hasattr(self, "max_seq_length") and self.max_seq_length is not None:
            if self.max_length == 1024:  # 1024 is the default
                self.max_length = self.max_seq_length

        # Force padding_free to False for Neuron - critical for avoiding recompilation
        self.padding_free = False

        super().__post_init__()
