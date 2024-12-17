#!/usr/bin/env python
# coding=utf-8
# Copyright 2020 The HuggingFace Inc. team and Amazon Web Services, Inc. All rights reserved.
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
"""
This script compiles a model to be usable as an Optimum Neuron NeuronModelForCausalLM.
"""

from optimum.neuron import NeuronModelForCausalLM

# num_cores is the number of neuron cores. Find this with the command neuron-ls
compiler_args = {"num_cores": 12, "auto_cast_type": 'bf16'}
input_shapes = {"batch_size": 1, "sequence_length": 4096}

# Compiles an Optimum Neuron model from the previously trained (uncompiled) model
model = NeuronModelForCausalLM.from_pretrained(
    "mistral_trained",
    export=True,
    **compiler_args,
    **input_shapes
)

# Saves the compiled model to the directory mistral_neuron
model.save_pretrained("mistral_neuron")
