# Copyright Amazon Web Services and its Affiliates. All Rights Reserved.
# Copyright 2025 The HuggingFace Team. All rights reserved.
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
# ==============================================================================
import torch


def to_torch_dtype(dtype):
    mapping = {
        "f32": torch.float32,
        "f16": torch.float16,
        "bf16": torch.bfloat16,
        "s8": torch.int8,
        "f8e4m3fn": torch.float8_e4m3fn if hasattr(torch, "float8_e4m3fn") else torch.int8,
    }
    return mapping[dtype]


def to_pyhlo_type(scribe, dtype):
    """
    Map a torch dtype to the corresponding scribe dtype object.
    """
    mapping = {
        "float32": scribe.f32,
        "float16": scribe.f16,
        "bfloat16": scribe.bf16,
    }
    return mapping[dtype]
