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
# Adapted from https://github.com/aws-neuron/neuronx-distributed-inference/blob/9993358ce052fd7a1bb4a7497a6318aac36ed95c/src/neuronx_distributed_inference/utils/random.py
import random

import neuronx_distributed as nxd
import numpy as np
import torch


def set_random_seed(seed):
    """set random seed for reproducability."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if nxd.parallel_layers.parallel_state.model_parallel_is_initialized():
        nxd.parallel_layers.random.model_parallel_xla_manual_seed(seed)
