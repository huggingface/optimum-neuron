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
""" """

import enum

class NeuronDistributedType(str, enum.Enum):
    """
    Represents a type of distributed environment.

    Values:

        - **NO** -- Not a distributed environment, just a single process.
        - **MULTI_CPU** -- Distributed on multiple CPU nodes.
        - **MULTI_GPU** -- Distributed on multiple GPUs.
        - **MULTI_XPU** -- Distributed on multiple XPUs.
        - **DEEPSPEED** -- Using DeepSpeed.
        - **TPU** -- Distributed on TPUs.
    """

    # Subclassing str as well as Enum allows the `DistributedType` to be JSON-serializable out of the box.
    NO = "NO"
    MULTI_CPU = "MULTI_CPU"
    MULTI_GPU = "MULTI_GPU"
    MULTI_XPU = "MULTI_XPU"
    DEEPSPEED = "DEEPSPEED"
    FSDP = "FSDP"
    TPU = "TPU"
    MPS = "MPS"  # here for backward compatibility. Remove in v0.18.0
    MEGATRON_LM = "MEGATRON_LM"
    XLA_FSDP = "XLA_FSDP"
