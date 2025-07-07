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
from abc import abstractmethod

import torch
from neuronx_distributed.trace.model_builder import BaseModelInstance
from torch_neuronx import BucketModelConfig


class NxDModelWrapper(torch.nn.Module):
    def __init__(self, tag: str, priority_model_idx: int):
        super().__init__()
        self.tag = tag
        self.priority_model_idx = priority_model_idx

    @abstractmethod
    def input_generator(self) -> list[torch.Tensor]:
        """Return the list of the model input tensors

        Used at compilation time only when tracing the model.
        """
        raise NotImplementedError

    @abstractmethod
    def get_model_instance(self) -> BaseModelInstance:
        """Return the underlying ModelInstance

        Used at compilation time only when tracing the model.
        """
        raise NotImplementedError

    @abstractmethod
    def get_bucket_config(self) -> BucketModelConfig:
        """Return the bucket configuration

        Used at compilation time only when tracing the model.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, *args) -> list[torch.Tensor]:
        raise NotImplementedError
