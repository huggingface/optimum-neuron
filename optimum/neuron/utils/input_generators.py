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
# limitations under the License.
"""Dummy input generation classes."""

import torch

from ...utils import DTYPE_MAPPER, DummyInputGenerator, NormalizedTextConfig


class DummyBeamValuesGenerator(DummyInputGenerator):
    """
    Generates dummy beam search inputs.
    """

    SUPPORTED_INPUT_NAMES = (
        "beam_idx",
        "beam_scores",
    )

    def __init__(
        self,
        task: str,
        normalized_config: NormalizedTextConfig,
        num_beams: int = 1,
        **kwargs,
    ):
        self.task = task
        self.num_beams = num_beams

    def generate(self, input_name: str, framework: str = "pt", int_dtype: str = "int64", float_dtype: str = "fp32"):
        if input_name == "beam_idx":
            return torch.arange(0, self.num_beams, dtype=DTYPE_MAPPER.pt(int_dtype))
        elif input_name == "beam_scores":
            return torch.zeros((self.num_beams,), dtype=DTYPE_MAPPER.pt(float_dtype))
