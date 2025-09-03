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
# Adapted from https://github.com/aws-neuron/neuronx-distributed-inference/blob/9993358ce052fd7a1bb4a7497a6318aac36ed95c/src/neuronx_distributed_inference/modules/custom_calls.py
import torch
from torch import nn, ones
from torch_neuronx.xla_impl.ops import RmsNorm


class NeuronRMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Use this RMSNorm to perform customized rmsnorm on Neuron
        Note: NeuronRMSNorm forward method calls target="AwsNeuronRmsNorm"
        """
        super().__init__()
        self.weight = nn.Parameter(ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        original_dtype = hidden_states.dtype

        hidden_states = hidden_states.to(torch.float32)
        result = RmsNorm.apply(hidden_states, self.weight, self.variance_epsilon, len(hidden_states.shape) - 1)

        return result.to(original_dtype)
